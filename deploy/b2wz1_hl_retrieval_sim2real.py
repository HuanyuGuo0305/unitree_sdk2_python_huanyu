#!/usr/bin/env python3
"""
B2WZ1 hierarchical retrieval sim2real deployment.

Control hierarchy
-----------------
Low-level ONNX : 50 Hz
High-level ONNX: 10 Hz
5 low-level control steps per high-level step.

This deployment intentionally preserves the validated B2W/Z1 hardware path from
b2wz1_locomanipulation_plb.py, while replacing the joystick / sampled EE command
with the hierarchical high-level policy used by the validated MuJoCo sim2sim.

Important alignment rules
-------------------------
1) Low-level 80-D observation ordering is unchanged.
2) Low-level 5-frame history is FEATURE-MAJOR, total 400-D.
3) High-level 56-D observation ordering exactly matches hierarchical sim2sim.
4) High-level 3-frame history is FEATURE-MAJOR, total 168-D.
5) Low action order:
       leg(12) | arm(6) | wheel(4)
6) Policy B2W joint order is NOT hardware motor order. Explicit name-based
   policy<->hardware mappings are preserved from the validated LL sim2real.
7) High-level scheduling matches training/sim2sim:
       HL decode -> LL observation/inference -> next 20-ms hardware block.
8) No startup policy-action blend-in.
9) No arm joint-target rate limiter.
10) Z1 runtime actuators.
    Gripper: ALWAYS "dcmotor" -- training-space IdealPD + exact IsaacLab
      DCMotor torque-speed law, sent as external tau_f with no post-DCMotor
      deployment torque cap. Not selectable.
    Arm: selected by z1_arm_runtime_mode.
       position_pd : Z1 firmware position loop with arm_kps_runtime /
                     arm_kds_runtime, tau_f = 0. Default.
       dcmotor     : firmware arm gains forced to zero, training-exact
                     tau = Kp*(q_target - q) - Kd*qd clipped to the per-joint
                     effort limits and sent as tau_f. Training models the arm
                     as IdealPD + symmetric effort clip, so NO torque-speed
                     envelope is applied (unlike the gripper).
11) HL gripper observation and grasp proxy use q_training = q_sdk - offset.
12) Grasp proxy supports two deployment modes:
       heuristic       : original geometric/stall heuristic.
       command_assumed : first executed CLOSE/grasp command immediately latches
                         grasp_confidence_proxy = True for the rest of the run.

Perception
----------
Uses the already-validated CRL unified perception system:

    obs = perception.get_latest_snapshot(
        projected_gravity_b=current_projected_gravity_b
    )

VO-derived Base height
----------------------
The fixed deployment approximation base_height=0.6017 is no longer used as a
permanent runtime root height.  Instead:

    - after B2W reaches DEFAULT and the user accepts policy start, the current
      physical Base height is explicitly anchored to vo_base_height_anchor_m;
    - RearStereoVO metric translation is projected onto gravity-up using the B2
      IMU, through the CRL VO Base-height estimator;
    - valid VO height updates self.base_height;
    - stale/invalid VO height freezes the last valid self.base_height;
    - a VO session/epoch reset preserves height continuity and re-anchors the
      new VO frame using the current B2 IMU gravity.

No additional runtime low-pass filter is applied.

Object before grasp:
    obs["object"]["position_base"]

Retrieval:
    obs["retrieval"]["retrieval_target_base"]

After grasp_confidence_proxy becomes true, the high-level object position no
longer depends on visual/VO object perception. It is set directly to the
training-defined gripper-center position in Base.

Perception fail-safe
--------------------
Before policy start:
    require raw object.valid AND retrieval.valid.

During policy:
    - before grasp: require object.valid AND retrieval.valid
    - after grasp proxy: object is synthesized from gripper center, so only the
      retrieval channel remains externally required.
    - invalid perception is NEVER replaced with zeros/stale values in HL obs.
    - instead, HL inference/history update is suspended and safe commands are:
          base command = 0
          EE command   = current measured EE keypoints in PLB
          gripper      = close if grasp proxy is true, otherwise hold current
      LL continues at 50 Hz.
    - if invalidity lasts longer than perception_fault_timeout_s, enter the
      existing high-KD damping protection mode.
    - on perception recovery, HL history is re-seeded with the first current
      valid frame before resuming inference.

Run
---
From the unitree_sdk2_python_huanyu repository root:

    python3 deploy/b2wz1_hl_retrieval_sim2real.py \
        <network_interface> \
        deploy/configs/b2wz1_hl_retrieval_sim2real.yaml

Developer PC must already be running the integrated perception UDP server:

    python3 ~/b2w_vision/rear_vo_udp_server.py --crl-host 192.168.123.222
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import yaml


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT,
    )

CAMERA_DIR = os.path.join(
    PROJECT_ROOT,
    "example",
    "b2w",
    "camera",
)

if CAMERA_DIR not in sys.path:
    sys.path.insert(
        0,
        CAMERA_DIR,
    )


from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (  # noqa: E402
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # noqa: E402
    LowCmd_ as LowCmdGo,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # noqa: E402
    LowState_ as LowStateGo,
)
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

from utils.command_helper import (  # noqa: E402
    InitLowCmd,
    create_zero_cmd,
)
from utils.math import (  # noqa: E402
    euler_xyz_from_quat_wxyz,
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_rotmat_wxyz,
    quat_from_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
)
from utils.remote_controller import (  # noqa: E402
    KeyMap,
    RemoteController,
)
from utils.z1_helper import (  # noqa: E402
    Z1ArmAdapter,
    compute_ee_current_kp_plb,
)

from b2w_perception import (  # noqa: E402
    CRLB2WPerceptionSystem,
)


def quat_from_euler_xyz_wxyz(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """Standard XYZ roll/pitch/yaw -> quaternion [w,x,y,z]."""
    cr = math.cos(
        0.5 * roll
    )
    sr = math.sin(
        0.5 * roll
    )
    cp = math.cos(
        0.5 * pitch
    )
    sp = math.sin(
        0.5 * pitch
    )
    cy = math.cos(
        0.5 * yaw
    )
    sy = math.sin(
        0.5 * yaw
    )

    q = np.array(
        [
            cr * cp * cy
            + sr * sp * sy,

            sr * cp * cy
            - cr * sp * sy,

            cr * sp * cy
            + sr * cp * sy,

            cr * cp * sy
            - sr * sp * cy,
        ],
        dtype=np.float32,
    )

    return quat_unique_wxyz(
        quat_normalize_wxyz(
            q
        )
    )


def build_keypoints_from_kp0_yaw_pitch_plb(
    kp0: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    kp_dx: float,
    kp_dz: float,
) -> np.ndarray:
    """
    Build [kp0,kp1,kp2] in PLB exactly like hierarchical sim2sim.
    """
    kp0 = np.asarray(
        kp0,
        dtype=np.float32,
    ).reshape(
        3,
    )

    q_plb = (
        quat_from_euler_xyz_wxyz(
            float(
                roll
            ),
            float(
                pitch
            ),
            float(
                yaw
            ),
        )
    )

    off_x = np.array(
        [
            kp_dx,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    off_z = np.array(
        [
            0.0,
            0.0,
            kp_dz,
        ],
        dtype=np.float32,
    )

    kp1 = (
        kp0
        + quat_apply_wxyz(
            q_plb,
            off_x,
        )
    )

    kp2 = (
        kp0
        + quat_apply_wxyz(
            q_plb,
            off_z,
        )
    )

    return np.concatenate(
        [
            kp0,
            kp1,
            kp2,
        ],
        dtype=np.float32,
    )


def split_features(
    frame: np.ndarray,
    dims: Sequence[int],
) -> list[np.ndarray]:
    frame = np.asarray(
        frame,
        dtype=np.float32,
    ).reshape(
        -1,
    )

    out = []
    i = 0

    for dim in dims:
        dim = int(
            dim
        )

        out.append(
            frame[
                i:i + dim
            ].copy()
        )

        i += dim

    if i != frame.shape[0]:
        raise RuntimeError(
            "Feature split mismatch: "
            f"consumed={i}, "
            f"frame={frame.shape[0]}"
        )

    return out


def fill_histories(
    histories: Sequence[deque],
    frame: np.ndarray,
    dims: Sequence[int],
    length: int,
) -> None:
    features = split_features(
        frame,
        dims,
    )

    for hist, feature in zip(
        histories,
        features,
    ):
        hist.clear()

        for _ in range(
            int(
                length
            )
        ):
            hist.append(
                feature.copy()
            )


def append_histories(
    histories: Sequence[deque],
    frame: np.ndarray,
    dims: Sequence[int],
) -> None:
    for hist, feature in zip(
        histories,
        split_features(
            frame,
            dims,
        ),
    ):
        hist.append(
            feature.copy()
        )


def flatten_feature_major(
    histories: Sequence[deque],
) -> np.ndarray:
    """
    IMPORTANT:
    feature-major history, NOT frame-major.

    [feature_0(t-H+1...t),
     feature_1(t-H+1...t),
     ...]
    """
    return np.concatenate(
        [
            np.asarray(
                hist,
                dtype=np.float32,
            ).reshape(
                -1,
            )
            for hist in histories
        ],
        dtype=np.float32,
    )


def point3_or_none(
    value: Any,
) -> Optional[np.ndarray]:
    if value is None:
        return None

    try:
        p = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(
            3,
        )
    except Exception:
        return None

    if not np.isfinite(
        p
    ).all():
        return None

    return p


class B2WZ1HierarchicalRetrievalController:
    """
    Final hierarchical HL(10 Hz) -> frozen LL(50 Hz) sim2real controller.
    """

    def __init__(
        self,
        cfg_path: str,
        network_interface: str,
    ):
        with open(
            cfg_path,
            "r",
        ) as f:
            self.cfg = (
                yaml.safe_load(
                    f
                )
            )

        self.network_interface = str(
            network_interface
        )

        # ------------------------------------------------------------------
        # Timing / policy dimensions
        # ------------------------------------------------------------------
        self.control_dt = float(
            self.cfg[
                "control_dt"
            ]
        )

        self.ll_steps_per_hl_step = int(
            self.cfg[
                "ll_steps_per_hl_step"
            ]
        )

        self.hl_control_dt = (
            self.control_dt
            * self.ll_steps_per_hl_step
        )

        if abs(
            1.0
            / self.control_dt
            - 50.0
        ) > 1e-5:
            raise ValueError(
                "Expected 50-Hz low-level control, "
                f"got dt={self.control_dt}."
            )

        if abs(
            1.0
            / self.hl_control_dt
            - 10.0
        ) > 1e-5:
            raise ValueError(
                "Expected 10-Hz high-level control, "
                f"got dt={self.hl_control_dt}."
            )

        self.ll_history_length = int(
            self.cfg[
                "ll_history_length"
            ]
        )

        self.ll_obs_dim_per_step = int(
            self.cfg[
                "ll_obs_dim_per_step"
            ]
        )

        self.ll_obs_dim = int(
            self.cfg[
                "ll_obs_dim"
            ]
        )

        self.ll_action_dim = int(
            self.cfg[
                "ll_action_dim"
            ]
        )

        self.hl_history_length = int(
            self.cfg[
                "hl_history_length"
            ]
        )

        self.hl_obs_dim_per_step = int(
            self.cfg[
                "hl_obs_dim_per_step"
            ]
        )

        self.hl_obs_dim = int(
            self.cfg[
                "hl_obs_dim"
            ]
        )

        self.hl_action_dim = int(
            self.cfg[
                "hl_action_dim"
            ]
        )

        assert (
            self.ll_history_length
            == 5
        )

        assert (
            self.ll_obs_dim_per_step
            == 80
        )

        assert (
            self.ll_obs_dim
            == 400
        )

        assert (
            self.ll_action_dim
            == 22
        )

        assert (
            self.hl_history_length
            == 3
        )

        assert (
            self.hl_obs_dim_per_step
            == 56
        )

        assert (
            self.hl_obs_dim
            == 168
        )

        assert (
            self.hl_action_dim
            == 9
        )

        # ------------------------------------------------------------------
        # ONNX policies: exactly the same two policies used in sim2sim.
        # ------------------------------------------------------------------
        self.low_policy_path = (
            self._resolve_path(
                self.cfg[
                    "low_policy_path"
                ]
            )
        )

        self.high_policy_path = (
            self._resolve_path(
                self.cfg[
                    "high_policy_path"
                ]
            )
        )

        self.low_session = (
            ort.InferenceSession(
                self.low_policy_path,
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.high_session = (
            ort.InferenceSession(
                self.high_policy_path,
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.low_input_name = (
            self.low_session
            .get_inputs()[0]
            .name
        )

        self.low_output_name = (
            self.low_session
            .get_outputs()[0]
            .name
        )

        self.high_input_name = (
            self.high_session
            .get_inputs()[0]
            .name
        )

        self.high_output_name = (
            self.high_session
            .get_outputs()[0]
            .name
        )

        self._assert_onnx_shapes()

        # ------------------------------------------------------------------
        # B2W policy order / hardware order.
        # DO NOT change without retraining / hardware remapping.
        # ------------------------------------------------------------------
        self.policy_leg_joint_names = [
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",

            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",

            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        ]

        self.policy_wheel_joint_names = [
            "FL_wheel_joint",
            "FR_wheel_joint",
            "RL_wheel_joint",
            "RR_wheel_joint",
        ]

        self.hardware_joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",

            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",

            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",

            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",

            "FR_wheel_joint",
            "FL_wheel_joint",
            "RR_wheel_joint",
            "RL_wheel_joint",
        ]

        self.policy_joint_names = (
            self.policy_leg_joint_names
            + self.policy_wheel_joint_names
        )

        self.hardware_to_policy_joint_indices = [
            self.hardware_joint_names.index(
                name
            )
            for name
            in self.policy_joint_names
        ]

        self.policy_to_hardware_joint_indices = [
            self.policy_joint_names.index(
                name
            )
            for name
            in self.hardware_joint_names
        ]

        self.num_b2w_dof = 16

        self.leg_policy_indices = list(
            range(
                12
            )
        )

        self.wheel_policy_indices = list(
            range(
                12,
                16,
            )
        )

        self.leg_hardware_indices = [
            self.hardware_joint_names.index(
                name
            )
            for name
            in self.policy_leg_joint_names
        ]

        self.wheel_hardware_indices = [
            self.hardware_joint_names.index(
                name
            )
            for name
            in self.policy_wheel_joint_names
        ]

        self.hw_to_wheel_cmd_indices = {
            self.hardware_joint_names.index(
                name
            ):
                idx
            for idx, name
            in enumerate(
                self.policy_wheel_joint_names
            )
        }

        # ------------------------------------------------------------------
        # Robot defaults / gains / LL action scales.
        # ------------------------------------------------------------------
        self.default_b2w_pos_policy = np.asarray(
            self.cfg[
                "default_b2w_pos_policy"
            ],
            dtype=np.float32,
        ).reshape(
            16,
        )

        self.squat_b2w_pos_policy = np.asarray(
            self.cfg[
                "squat_b2w_pos_policy"
            ],
            dtype=np.float32,
        ).reshape(
            16,
        )

        self.default_arm_pos = np.asarray(
            self.cfg[
                "default_arm_pos"
            ],
            dtype=np.float32,
        ).reshape(
            6,
        )

        # Training observation reference for jointGripper is 0 rad.
        self.default_gripper_pos = float(
            self.cfg[
                "default_gripper_pos"
            ]
        )

        self.gripper_open_pos = float(
            self.cfg[
                "gripper_open_pos"
            ]
        )

        self.gripper_close_pos = float(
            self.cfg[
                "gripper_close_pos"
            ]
        )

        self.default_joint_pos_policy = np.asarray(
            self.cfg[
                "default_joint_pos_policy"
            ],
            dtype=np.float32,
        ).reshape(
            18,
        )

        self.kps_rl = np.asarray(
            self.cfg[
                "kps_rl"
            ],
            dtype=np.float32,
        ).reshape(
            16,
        )

        self.kds_rl = np.asarray(
            self.cfg[
                "kds_rl"
            ],
            dtype=np.float32,
        ).reshape(
            16,
        )

        self.kps_pd = np.asarray(
            self.cfg[
                "kps_pd"
            ],
            dtype=np.float32,
        ).reshape(
            16,
        )

        self.kds_pd = np.asarray(
            self.cfg[
                "kds_pd"
            ],
            dtype=np.float32,
        ).reshape(
            16,
        )

        self.kps_rl_hw = (
            self.kps_rl[
                self.policy_to_hardware_joint_indices
            ]
        )

        self.kds_rl_hw = (
            self.kds_rl[
                self.policy_to_hardware_joint_indices
            ]
        )

        self.kps_pd_hw = (
            self.kps_pd[
                self.policy_to_hardware_joint_indices
            ]
        )

        self.kds_pd_hw = (
            self.kds_pd[
                self.policy_to_hardware_joint_indices
            ]
        )

        self.wheel_kps_rl_hw = np.zeros(
            self.num_b2w_dof,
            dtype=np.float32,
        )

        self.wheel_kds_rl_hw = np.zeros(
            self.num_b2w_dof,
            dtype=np.float32,
        )

        self.wheel_kps_pd_hw = np.zeros(
            self.num_b2w_dof,
            dtype=np.float32,
        )

        self.wheel_kds_pd_hw = np.zeros(
            self.num_b2w_dof,
            dtype=np.float32,
        )

        for hw_idx in (
            self.wheel_hardware_indices
        ):
            self.wheel_kps_rl_hw[
                hw_idx
            ] = self.kps_rl_hw[
                hw_idx
            ]

            self.wheel_kds_rl_hw[
                hw_idx
            ] = self.kds_rl_hw[
                hw_idx
            ]

            self.wheel_kps_pd_hw[
                hw_idx
            ] = self.kps_pd_hw[
                hw_idx
            ]

            self.wheel_kds_pd_hw[
                hw_idx
            ] = self.kds_pd_hw[
                hw_idx
            ]

        self.default_leg_pos_policy = (
            self.default_b2w_pos_policy[
                :12
            ].copy()
        )

        self.leg_action_indices = np.arange(
            0,
            12,
        )

        self.arm_action_indices = np.arange(
            12,
            18,
        )

        self.wheel_action_indices = np.arange(
            18,
            22,
        )

        self.leg_action_scale = float(
            self.cfg[
                "leg_action_scale"
            ]
        )

        self.arm_action_scale = np.asarray(
            self.cfg[
                "arm_action_scale"
            ],
            dtype=np.float32,
        ).reshape(
            6,
        )

        self.wheel_action_scale = float(
            self.cfg[
                "wheel_action_scale"
            ]
        )

        # ------------------------------------------------------------------
        # High-level decode semantics.
        # ------------------------------------------------------------------
        self.base_cmd_scale = np.asarray(
            self.cfg[
                "base_cmd_scale"
            ],
            dtype=np.float32,
        ).reshape(
            3,
        )

        self.kp0_delta_scale = np.asarray(
            self.cfg[
                "kp0_delta_scale"
            ],
            dtype=np.float32,
        ).reshape(
            3,
        )

        self.kp0_x_range = np.asarray(
            self.cfg[
                "kp0_x_range"
            ],
            dtype=np.float32,
        ).reshape(
            2,
        )

        self.kp0_y_range = np.asarray(
            self.cfg[
                "kp0_y_range"
            ],
            dtype=np.float32,
        ).reshape(
            2,
        )

        self.kp0_z_range = np.asarray(
            self.cfg[
                "kp0_z_range"
            ],
            dtype=np.float32,
        ).reshape(
            2,
        )

        self.ee_yaw_delta_scale = float(
            self.cfg[
                "ee_yaw_delta_scale"
            ]
        )

        self.ee_pitch_delta_scale = float(
            self.cfg[
                "ee_pitch_delta_scale"
            ]
        )

        self.ee_yaw_range = np.asarray(
            self.cfg[
                "ee_yaw_range"
            ],
            dtype=np.float32,
        ).reshape(
            2,
        )

        self.ee_pitch_range = np.asarray(
            self.cfg[
                "ee_pitch_range"
            ],
            dtype=np.float32,
        ).reshape(
            2,
        )

        self.neutral_kp0 = np.asarray(
            self.cfg[
                "neutral_kp0"
            ],
            dtype=np.float32,
        ).reshape(
            3,
        )

        self.neutral_ee_yaw = float(
            self.cfg[
                "neutral_ee_yaw"
            ]
        )

        self.neutral_ee_pitch = float(
            self.cfg[
                "neutral_ee_pitch"
            ]
        )

        self.fixed_ee_roll = float(
            self.cfg[
                "fixed_ee_roll"
            ]
        )

        self.ee_kp_dx = float(
            self.cfg[
                "ee_kp_dx"
            ]
        )

        self.ee_kp_dz = float(
            self.cfg[
                "ee_kp_dz"
            ]
        )

        self.ground_z = float(
            self.cfg.get(
                "ground_z",
                0.0,
            )
        )

        # ------------------------------------------------------------------
        # VO-derived physical Base height.
        #
        # Training/sim2sim uses the CURRENT root z relative to ground.  The
        # real robot therefore must not keep a permanently fixed 0.6017-m
        # approximation.  0.6017 is only the known physical height at the
        # explicit policy-start anchor.
        # ------------------------------------------------------------------
        self.vo_base_height_enabled = bool(
            self.cfg.get(
                "vo_base_height_enabled",
                True,
            )
        )

        self.base_height_anchor_m = float(
            self.cfg.get(
                "vo_base_height_anchor_m",
                self.cfg.get(
                    "base_height",
                    0.6017,
                ),
            )
        )

        if (
            not np.isfinite(
                self.base_height_anchor_m
            )
            or self.base_height_anchor_m
            <= 0.0
        ):
            raise ValueError(
                "Invalid vo_base_height_anchor_m="
                f"{self.base_height_anchor_m!r}."
            )

        # Runtime value consumed by BOTH PLB EE geometry paths.
        # Before the explicit VO anchor it is initialized to the known
        # default-pose anchor height.  After anchoring it is updated from VO.
        self.base_height = float(
            self.base_height_anchor_m
        )

        self.last_base_height_state: Optional[
            Dict[str, Any]
        ] = None

        self.gripper_binary_threshold = float(
            self.cfg[
                "gripper_binary_threshold"
            ]
        )

        self.stage2_force_gripper_close_enabled = bool(
            self.cfg.get(
                "stage2_force_gripper_close_enabled",
                True,
            )
        )

        # Grasp proxy deployment mode:
        #
        #   heuristic
        #       Original Stage-2 heuristic:
        #       CLOSE + gripper stall + hysteresis.
        #       (the object/gripper distance gate was removed)
        #
        #   command_assumed
        #       The first executed CLOSE/grasp command is treated as successful
        #       grasp immediately. The proxy is latched True until reset/restart.
        self.grasp_proxy_mode = str(
            self.cfg.get(
                "grasp_proxy_mode",
                "heuristic",
            )
        ).strip().lower()

        valid_grasp_proxy_modes = {
            "heuristic",
            "command_assumed",
        }

        if self.grasp_proxy_mode not in valid_grasp_proxy_modes:
            raise ValueError(
                "Invalid grasp_proxy_mode="
                f"{self.grasp_proxy_mode!r}. "
                "Expected one of "
                f"{sorted(valid_grasp_proxy_modes)}."
            )

        # ------------------------------------------------------------------
        # Grasp confidence proxy.
        # ------------------------------------------------------------------
        self.gripper_center_offset_local = np.asarray(
            self.cfg[
                "gripper_center_offset_local"
            ],
            dtype=np.float32,
        ).reshape(
            3,
        )

        # NOTE: the object <-> gripper-center distance gate
        # (grasp_proxy_error_threshold) was deliberately removed from the
        # heuristic proxy.  The distance is still computed as a diagnostic,
        # but no threshold is applied to it any more.

        self.gripper_not_fully_closed_angle_threshold = float(
            self.cfg[
                "gripper_not_fully_closed_angle_threshold"
            ]
        )

        self.gripper_angle_hold_threshold = float(
            self.cfg[
                "gripper_angle_hold_threshold"
            ]
        )

        self.grasp_proxy_enter_steps = int(
            self.cfg[
                "grasp_proxy_enter_steps"
            ]
        )

        self.grasp_proxy_exit_steps = int(
            self.cfg[
                "grasp_proxy_exit_steps"
            ]
        )

        # ------------------------------------------------------------------
        # Z1.
        # ------------------------------------------------------------------
        self.z1 = Z1ArmAdapter(
            self.cfg,
            PROJECT_ROOT,
        )

        # Z1 runtime actuator modes.
        #
        # gripper: ALWAYS "dcmotor" (training-exact IdealPD + IsaacLab DCMotor
        #          torque-speed law through tau_f). Not selectable.
        #
        # arm    : z1_arm_runtime_mode selects
        #            "position_pd" -> firmware position loop (default)
        #            "dcmotor"     -> zero firmware gains + training-exact
        #                             IdealPD + effort clip through tau_f
        #
        # Selection is made by z1_helper; mirror it here for validation and
        # startup reporting.
        self.valid_z1_arm_runtime_modes = {
            "position_pd",
            "dcmotor",
        }

        self.z1_gripper_runtime_mode = str(
            self.z1.gripper_runtime_mode
        ).strip().lower()

        self.z1_arm_runtime_mode = str(
            self.z1.arm_runtime_mode
        ).strip().lower()

        # Validate all actuator constants for the selected mode before any
        # hardware motion is allowed.
        self._validate_z1_runtime_mode()

        # ------------------------------------------------------------------
        # Perception.
        # ChannelFactoryInitialize is done ONCE in main(), so perception must
        # not initialize it a second time.
        # ------------------------------------------------------------------
        self.perception = (
            CRLB2WPerceptionSystem(
                interface=
                    self.network_interface,

                repo_root=
                    Path(
                        PROJECT_ROOT
                    ),

                developer_host=
                    str(
                        self.cfg.get(
                            "developer_host",
                            "192.168.123.164",
                        )
                    ),

                data_port=
                    int(
                        self.cfg.get(
                            "perception_data_port",
                            50020,
                        )
                    ),

                sync_port=
                    int(
                        self.cfg.get(
                            "perception_sync_port",
                            50021,
                        )
                    ),

                tag_id=
                    int(
                        self.cfg.get(
                            "tag_id",
                            0,
                        )
                    ),

                tag_size_m=
                    float(
                        self.cfg.get(
                            "tag_size_m",
                            0.195,
                        )
                    ),

                retrieval_target_tag=
                    np.asarray(
                        self.cfg.get(
                            "retrieval_target_tag",
                            [
                                0.0,
                                0.0,
                                1.0,
                            ],
                        ),
                        dtype=np.float64,
                    ).reshape(
                        3,
                    ),

                visual_max_age_s=
                    float(
                        self.cfg.get(
                            "tag_visual_max_age_s",
                            0.200,
                        )
                    ),

                object_max_age_ms=
                    float(
                        self.cfg.get(
                            "object_max_age_ms",
                            300.0,
                        )
                    ),

                retrieval_max_age_ms=
                    float(
                        self.cfg.get(
                            "retrieval_max_age_ms",
                            250.0,
                        )
                    ),

                vo_max_age_ms=
                    float(
                        self.cfg.get(
                            "vo_max_age_ms",
                            250.0,
                        )
                    ),

                initialize_channel_factory=
                    False,
            )
        )

        if self.vo_base_height_enabled:
            required_height_api = (
                "anchor_vo_base_height",
                "get_vo_base_height_snapshot",
                "reset_vo_base_height_estimator",
            )

            missing_height_api = [
                name
                for name
                in required_height_api
                if not hasattr(
                    self.perception,
                    name,
                )
            ]

            if missing_height_api:
                raise RuntimeError(
                    "VO Base-height integration is enabled, but "
                    "b2w_perception.py is missing required API(s): "
                    + ", ".join(
                        missing_height_api
                    )
                    + ". Install the VO-height-enabled "
                    "b2w_perception.py / rear_vo_udp_receiver.py."
                )

        self.perception_fault_timeout_s = float(
            self.cfg.get(
                "perception_fault_timeout_s",
                0.5,
            )
        )

        self.perception_status_print_s = float(
            self.cfg.get(
                "perception_status_print_s",
                1.0,
            )
        )

        # ------------------------------------------------------------------
        # B2W DDS.
        # ------------------------------------------------------------------
        self.remote_controller = (
            RemoteController()
        )

        self.low_cmd = (
            unitree_go_msg_dds__LowCmd_()
        )

        self.low_state = (
            unitree_go_msg_dds__LowState_()
        )

        self.lowcmd_publisher_ = (
            ChannelPublisher(
                "rt/lowcmd",
                LowCmdGo,
            )
        )

        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = (
            ChannelSubscriber(
                "rt/lowstate",
                LowStateGo,
            )
        )

        self.lowstate_subscriber.Init(
            self.low_state_handler,
            10,
        )

        self.crc = CRC()

        # ------------------------------------------------------------------
        # Measured robot state.
        # ------------------------------------------------------------------
        self.base_quat_wxyz = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )

        self.base_ang_vel_b = np.zeros(
            3,
            dtype=np.float32,
        )

        self.gravity_w = np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float32,
        )

        self.projected_gravity_b = np.array(
            [
                0.0,
                0.0,
                -1.0,
            ],
            dtype=np.float32,
        )

        # Stored in POLICY order:
        # leg12 + wheel4.
        self.b2w_joint_pos = np.zeros(
            16,
            dtype=np.float32,
        )

        self.b2w_joint_vel = np.zeros(
            16,
            dtype=np.float32,
        )

        # ------------------------------------------------------------------
        # Current commands/actions.
        # ------------------------------------------------------------------
        self.base_command = np.zeros(
            3,
            dtype=np.float32,
        )

        self.ee_cmd_plb_current = np.zeros(
            9,
            dtype=np.float32,
        )

        self.last_ll_action = np.zeros(
            self.ll_action_dim,
            dtype=np.float32,
        )

        self.current_hl_action = np.zeros(
            self.hl_action_dim,
            dtype=np.float32,
        )

        self.raw_gripper_action = -1.0
        self.executed_gripper_cmd_norm = -1.0

        self.leg_target = (
            self.default_leg_pos_policy.copy()
        )

        self.arm_target = (
            self.default_arm_pos.copy()
        )

        self.wheel_cmd = np.zeros(
            4,
            dtype=np.float32,
        )

        # Runtime gripper targets are ALWAYS in TRAINING coordinates.
        self.gripper_target = float(
            self.gripper_open_pos
        )

        # ------------------------------------------------------------------
        # Grasp state.
        # ------------------------------------------------------------------
        self.grasp_confidence_proxy = False
        self.grasp_proxy_enter_count = 0
        self.grasp_proxy_exit_count = 0
        self.prev_gripper_joint_pos = float(
            self.gripper_open_pos
        )

        self.last_grasp_error = float(
            "inf"
        )

        # ------------------------------------------------------------------
        # Histories: exactly one deque per feature group.
        # ------------------------------------------------------------------
        self.ll_feature_dims = [
            3,
            3,
            3,
            9,
            12,
            6,
            12,
            6,
            4,
            22,
        ]

        self.hl_feature_dims = [
            3,
            3,
            12,
            6,
            1,
            6,
            3,
            6,
            3,
            3,
            9,
            1,
        ]

        assert (
            sum(
                self.ll_feature_dims
            )
            == self.ll_obs_dim_per_step
        )

        assert (
            sum(
                self.hl_feature_dims
            )
            == self.hl_obs_dim_per_step
        )

        self.ll_histories = [
            deque(
                maxlen=
                    self.ll_history_length
            )
            for _
            in self.ll_feature_dims
        ]

        self.hl_histories = [
            deque(
                maxlen=
                    self.hl_history_length
            )
            for _
            in self.hl_feature_dims
        ]

        # ------------------------------------------------------------------
        # Runtime scheduling / perception hold state.
        # ------------------------------------------------------------------
        self.ll_tick = 0
        self.hl_tick = 0

        self.perception_hold_active = False
        self.perception_invalid_since: Optional[
            float
        ] = None

        self.perception_hold_reason = ""
        self.last_perception_snapshot: Optional[
            Dict[str, Any]
        ] = None

        self.last_object_source = None
        self.last_retrieval_source = None

        self.debug_print_period_steps = int(
            self.cfg.get(
                "debug_print_period_steps",
                50,
            )
        )

        self.debug_hl_obs_enabled = bool(
            self.cfg.get(
                "debug_hl_obs_enabled",
                False,
            )
        )

        self.debug_ll_obs_enabled = bool(
            self.cfg.get(
                "debug_ll_obs_enabled",
                False,
            )
        )

        self.debug_obs_print_max = int(
            self.cfg.get(
                "debug_obs_print_max",
                5,
            )
        )

        self.debug_hl_obs_count = 0
        self.debug_ll_obs_count = 0

        # ------------------------------------------------------------------
        # Damping protection.
        # ------------------------------------------------------------------
        self.damping_kd_b2w = float(
            self.cfg.get(
                "damping_kd_b2w",
                150.0,
            )
        )

        self.damping_kd_wheel = float(
            self.cfg.get(
                "damping_kd_wheel",
                10.0,
            )
        )

        self.damping_kp_z1 = np.asarray(
            self.cfg.get(
                "damping_kp_z1",
                [
                    0.0,
                ] * 6,
            ),
            dtype=np.float32,
        ).reshape(
            6,
        )

        self.damping_kd_z1 = np.asarray(
            self.cfg.get(
                "damping_kd_z1",
                [
                    2000.0,
                ] * 6,
            ),
            dtype=np.float32,
        ).reshape(
            6,
        )

        self.damping_print_period = int(
            self.cfg.get(
                "damping_print_period",
                100,
            )
        )

        # ------------------------------------------------------------------
        # Manual LEG recovery from protection mode.
        #
        # This is deliberately NOT automatic.  A Z1 PASSIVE event may represent
        # a physical fault/E-stop, so moving any joint requires an explicit
        # operator A-button press after protection has been entered.
        #
        # Recovery semantics:
        #   legs   : B2W leg12 current measured q -> damping_recover_leg_target
        #            (default: the SQUAT pose, i.e. the leg block of
        #             squat_b2w_pos_policy, so the robot folds down safely)
        #   arm    : NOT moved.  Z1 keeps receiving exactly the same protection
        #            command as the steady loop, so PASSIVE is never forced back
        #            into LOWCMD and the arm simply stays where it is.
        #   wheels : zero velocity / damping only (no absolute-angle servo)
        #   gripper: untouched
        #   finish : return to the high-damping protection loop; never resume RL
        # ------------------------------------------------------------------
        self.damping_leg_recovery_enabled = bool(
            self.cfg.get(
                "damping_leg_recovery_enabled",
                True,
            )
        )

        self.damping_recover_leg_duration_s = float(
            self.cfg.get(
                "damping_recover_leg_duration_s",
                4.0,
            )
        )

        self.damping_recover_leg_tolerance_rad = float(
            self.cfg.get(
                "damping_recover_leg_tolerance_rad",
                0.15,
            )
        )

        # Leg-only target, in POLICY leg order (12).
        # Defaults to the SQUAT pose already used by the startup sequence.
        self.damping_recover_leg_target = np.asarray(
            self.cfg.get(
                "damping_recover_leg_target",
                self.squat_b2w_pos_policy[
                    :12
                ].tolist(),
            ),
            dtype=np.float32,
        ).reshape(
            12,
        )

        if self.damping_recover_leg_duration_s <= 0.0:
            raise ValueError(
                "damping_recover_leg_duration_s must be > 0."
            )

        if self.damping_recover_leg_tolerance_rad <= 0.0:
            raise ValueError(
                "damping_recover_leg_tolerance_rad must be > 0."
            )

        # ------------------------------------------------------------------
        # Optional MuJoCo debug visualizer.
        #
        # Renders, from real telemetry only (no physics stepping): the live
        # B2W/Z1 kinematic pose, the detected object position, the detected
        # retrieval-target position, and the decoded HL end-effector command.
        # ------------------------------------------------------------------
        self.visualizer_enabled = bool(
            self.cfg.get(
                "visualizer_enabled",
                False,
            )
        )

        self.visualizer = None

        if self.visualizer_enabled:
            from utils.mj_visualizer import (  # noqa: E402
                MujocoDebugVisualizer,
            )

            visualizer_xml_path = (
                self._resolve_path(
                    self.cfg[
                        "visualizer_xml_path"
                    ]
                )
            )

            self.visualizer = (
                MujocoDebugVisualizer(
                    xml_path=
                        visualizer_xml_path,

                    update_hz=
                        float(
                            self.cfg.get(
                                "visualizer_update_hz",
                                30.0,
                            )
                        ),

                    object_marker_radius=
                        float(
                            self.cfg.get(
                                "visualizer_object_marker_radius",
                                0.03,
                            )
                        ),

                    retrieval_marker_radius=
                        float(
                            self.cfg.get(
                                "visualizer_retrieval_marker_radius",
                                0.05,
                            )
                        ),

                    ee_target_sphere_radius=
                        float(
                            self.cfg.get(
                                "visualizer_ee_target_sphere_radius",
                                0.03,
                            )
                        ),

                    ee_target_axis_len=
                        float(
                            self.cfg.get(
                                "visualizer_ee_target_axis_len",
                                0.20,
                            )
                        ),

                    ee_target_axis_radius=
                        float(
                            self.cfg.get(
                                "visualizer_ee_target_axis_radius",
                                0.01,
                            )
                        ),

                    show_gripper_center=
                        bool(
                            self.cfg.get(
                                "visualizer_show_gripper_center",
                                True,
                            )
                        ),

                    gripper_center_marker_radius=
                        float(
                            self.cfg.get(
                                "visualizer_gripper_center_marker_radius",
                                0.018,
                            )
                        ),

                    show_floor=
                        bool(
                            self.cfg.get(
                                "visualizer_show_floor",
                                True,
                            )
                        ),

                    floor_half_extent=
                        float(
                            self.cfg.get(
                                "visualizer_floor_half_extent",
                                3.0,
                            )
                        ),

                    show_light=
                        bool(
                            self.cfg.get(
                                "visualizer_show_light",
                                True,
                            )
                        ),

                    # The floor geom is static, so it is placed once at the
                    # same ground reference the PLB EE markers use.
                    ground_z=
                        float(
                            self.ground_z
                        ),
                )
            )

    # ======================================================================
    # Initialization helpers
    # ======================================================================

    def _resolve_path(
        self,
        path_str: str,
    ) -> str:
        if os.path.isabs(
            path_str
        ):
            return path_str

        return os.path.abspath(
            os.path.join(
                PROJECT_ROOT,
                path_str,
            )
        )

    def _get_gripper_q_training(
        self,
    ) -> float:
        """
        Measured gripper position in TRAINING coordinates.
        """
        return float(
            self.z1.get_gripper_q_training()
        )

    def _gripper_training_to_sdk(
        self,
        q_training: float,
    ) -> float:
        """
        Convert a training-space gripper target to raw SDK coordinates.
        Used only by startup/hold position-servo paths.
        """
        return float(
            self.z1.gripper_training_to_sdk(
                float(q_training)
            )
        )

    def _validate_z1_runtime_mode(
        self,
    ) -> None:
        """
        Validate the selected Z1 runtime actuator modes before any hardware
        motion is allowed.

        GRIPPER: always "dcmotor".  Internal gripper gains are zeroed and the
        training-space IdealPD + exact IsaacLab DCMotor torque-speed law is
        sent as tau_f, with no post-DCMotor deployment torque cap.  There is
        no alternative: the legacy position servo does not reproduce the
        trained grasp dynamics, and is used only by startup/hold paths.

        ARM: selected by z1_arm_runtime_mode.

            "position_pd" (default):
                Z1 firmware closes the position loop with
                arm_kps_runtime / arm_kds_runtime; tau_f = 0.

            "dcmotor":
                Firmware arm gains forced to zero, training-exact
                    tau = Kp*(q_target - q) - Kd*qd
                clipped to the per-joint effort limits, sent as tau_f.
                Training models the arm as IdealPD + symmetric effort clip,
                so NO torque-speed envelope is applied (unlike the gripper).
        """
        if not hasattr(
            self.z1,
            "track_target_pd_runtime_once",
        ):
            raise RuntimeError(
                "utils/z1_helper.py does not expose "
                "track_target_pd_runtime_once(). "
                "Install the mode-dispatching z1_helper.py."
            )

        if (
            self.z1_gripper_runtime_mode
            != "dcmotor"
        ):
            raise ValueError(
                "The runtime gripper only supports 'dcmotor', got "
                f"{self.z1.gripper_runtime_mode!r}."
            )

        if (
            self.z1_arm_runtime_mode
            not in self.valid_z1_arm_runtime_modes
        ):
            raise ValueError(
                "Invalid z1_arm_runtime_mode="
                f"{self.z1.arm_runtime_mode!r}. "
                "Expected one of "
                f"{sorted(self.valid_z1_arm_runtime_modes)}."
            )

        if not np.isfinite(
            float(
                self.z1.gripper_q_offset
            )
        ):
            raise ValueError(
                "Invalid z1_gripper_q_offset."
            )

        self._validate_gripper_dcmotor_alignment()

        if (
            self.z1_arm_runtime_mode
            == "dcmotor"
        ):
            self._validate_arm_dcmotor_alignment()

    def _validate_arm_dcmotor_alignment(
        self,
    ) -> None:
        """
        Fail loudly unless the external arm torque path matches the training
        arm actuator exactly.

        Required law (IsaacLab / sim2sim apply_low_level_actuation):
            tau = Kp * (q_target - q) - Kd * qd
            tau = clip(tau, -effort_limit, +effort_limit)

        with the training constants:
            Kp     = [76.8, 89.6, 89.6, 76.8, 76.8, 76.8]
            Kd     = 4.0 on every joint
            effort = [30.0, 60.0, 30.0, 30.0, 30.0, 30.0] Nm
        """
        if not hasattr(
            self.z1,
            "_compute_arm_external_tau",
        ):
            raise RuntimeError(
                "utils/z1_helper.py does not expose "
                "_compute_arm_external_tau(), required by "
                "z1_arm_runtime_mode='dcmotor'."
            )

        expected = {
            "arm_dcmotor_kp": [
                76.8,
                89.6,
                89.6,
                76.8,
                76.8,
                76.8,
            ],

            "arm_dcmotor_kd": [
                4.0,
            ] * 6,

            "arm_dcmotor_effort_limit": [
                30.0,
                60.0,
                30.0,
                30.0,
                30.0,
                30.0,
            ],
        }

        for name, target in expected.items():
            actual = np.asarray(
                getattr(
                    self.z1,
                    name,
                ),
                dtype=np.float32,
            ).reshape(
                6,
            )

            target_arr = np.asarray(
                target,
                dtype=np.float32,
            ).reshape(
                6,
            )

            if not np.allclose(
                actual,
                target_arr,
                rtol=0.0,
                atol=1e-4,
            ):
                raise ValueError(
                    "Arm actuator mismatch: "
                    f"{name}={actual}, expected {target_arr}."
                )

        print(
            "[Z1-MODE][WARN] z1_arm_runtime_mode='dcmotor': the 6 arm joints "
            "are driven by EXTERNAL TORQUE with zero firmware position gains. "
            "The arm will go limp if the command stream stalls."
        )

    def _validate_gripper_dcmotor_alignment(
        self,
    ) -> None:
        """
        Fail loudly unless the real-robot Z1 helper is configured for the exact
        hierarchical sim2sim/training gripper actuator.

        Required path:
            q_training = q_sdk - offset
            tau_pd = 76.8 * (q_target - q_training) - 4.0 * qd
            exact IsaacLab DCMotor four-quadrant torque-speed envelope
            tau_f = tau_dcmotor

        No deployment-only torque clip is allowed after the DCMotor envelope.
        """
        if not hasattr(
            self.z1,
            "track_target_pd_gripper_dcmotor_once",
        ):
            raise RuntimeError(
                "utils/z1_helper.py does not expose "
                "track_target_pd_gripper_dcmotor_once()."
            )

        expected = {
            "gripper_dcmotor_kp": 76.8,
            "gripper_dcmotor_kd": 4.0,
            "gripper_dcmotor_effort_limit": 30.0,
            "gripper_dcmotor_saturation_effort": 30.0,
            "gripper_dcmotor_velocity_limit": 2.0,
        }

        for name, target in expected.items():
            actual = float(
                getattr(
                    self.z1,
                    name,
                )
            )

            if not np.isclose(
                actual,
                target,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "Gripper actuator mismatch: "
                    f"{name}={actual}, expected {target}."
                )

        # The NEW helper intentionally has no post-DCMotor deployment cap.
        # Reject an older helper so this cannot silently fall back to the
        # previous real_tau_cap implementation.
        if hasattr(
            self.z1,
            "gripper_real_tau_cap",
        ):
            raise RuntimeError(
                "Installed z1_helper.py still exposes the old "
                "gripper_real_tau_cap deployment clip. "
                "Use the new exact-DCMotor helper."
            )

    def _assert_onnx_shapes(
        self,
    ) -> None:
        low_in = (
            self.low_session
            .get_inputs()[0]
            .shape
        )

        low_out = (
            self.low_session
            .get_outputs()[0]
            .shape
        )

        high_in = (
            self.high_session
            .get_inputs()[0]
            .shape
        )

        high_out = (
            self.high_session
            .get_outputs()[0]
            .shape
        )

        if isinstance(
            low_in[-1],
            int,
        ):
            assert (
                low_in[-1]
                == self.ll_obs_dim
            ), low_in

        if isinstance(
            low_out[-1],
            int,
        ):
            assert (
                low_out[-1]
                == self.ll_action_dim
            ), low_out

        if isinstance(
            high_in[-1],
            int,
        ):
            assert (
                high_in[-1]
                == self.hl_obs_dim
            ), high_in

        if isinstance(
            high_out[-1],
            int,
        ):
            assert (
                high_out[-1]
                == self.hl_action_dim
            ), high_out

    # ======================================================================
    # B2W / Z1 state
    # ======================================================================

    def low_state_handler(
        self,
        msg: LowStateGo,
    ) -> None:
        self.low_state = msg

        self.remote_controller.set(
            msg.wireless_remote
        )

    def wait_for_low_state(
        self,
    ) -> None:
        print(
            "[B2WZ1-HIER] Waiting for first B2W lowstate..."
        )

        while getattr(
            self.low_state,
            "tick",
            0,
        ) == 0:
            time.sleep(
                self.control_dt
            )

        print(
            "[B2WZ1-HIER] First lowstate received: "
            f"tick={self.low_state.tick}"
        )

    def send_b2w_cmd(
        self,
    ) -> None:
        self.low_cmd.crc = (
            self.crc.Crc(
                self.low_cmd
            )
        )

        self.lowcmd_publisher_.Write(
            self.low_cmd
        )

    def _read_b2w_sensors_once(
        self,
    ) -> None:
        q = (
            self.low_state
            .imu_state
            .quaternion
        )

        self.base_quat_wxyz[:] = (
            quat_unique_wxyz(
                np.array(
                    [
                        q[0],
                        q[1],
                        q[2],
                        q[3],
                    ],
                    dtype=np.float32,
                )
            )
        )

        self.base_quat_wxyz[:] = (
            self.base_quat_wxyz
            / max(
                np.linalg.norm(
                    self.base_quat_wxyz
                ),
                1e-8,
            )
        )

        gyro = (
            self.low_state
            .imu_state
            .gyroscope
        )

        self.base_ang_vel_b[:] = np.array(
            [
                gyro[0],
                gyro[1],
                gyro[2],
            ],
            dtype=np.float32,
        )

        self.projected_gravity_b[:] = (
            quat_rotate_inverse_numpy(
                self.base_quat_wxyz,
                self.gravity_w,
            )
        )

        # Hardware motor_state -> POLICY order.
        for policy_idx in range(
            self.num_b2w_dof
        ):
            hw_idx = (
                self.hardware_to_policy_joint_indices[
                    policy_idx
                ]
            )

            self.b2w_joint_pos[
                policy_idx
            ] = (
                self.low_state
                .motor_state[
                    hw_idx
                ]
                .q
            )

            self.b2w_joint_vel[
                policy_idx
            ] = (
                self.low_state
                .motor_state[
                    hw_idx
                ]
                .dq
            )

        # IMPORTANT:
        # Do NOT overwrite base_command from joystick here.
        # base_command is owned by the high-level policy.

    def _read_all_sensors_once(
        self,
    ) -> None:
        self._read_b2w_sensors_once()
        self.z1.read_state()

    # ======================================================================
    # PLB / gripper geometry
    # ======================================================================

    def compute_ee_current_kp_plb(
        self,
    ) -> np.ndarray:
        return compute_ee_current_kp_plb(
            base_quat_wxyz=
                self.base_quat_wxyz,

            base_height=
                self.base_height,

            ground_z=
                self.ground_z,

            z1_adapter=
                self.z1,

            kp_dx=
                self.ee_kp_dx,

            kp_dz=
                self.ee_kp_dz,
        )

    def compute_actual_ee_pose_plb(
        self,
    ) -> Tuple[
        np.ndarray,
        float,
        float,
    ]:
        """
        Hardware equivalent of hierarchical sim2sim compute_actual_ee_pose_plb().
        """
        base_q = quat_unique_wxyz(
            quat_normalize_wxyz(
                self.base_quat_wxyz
            )
        )

        ee_pos_b, ee_rot_b = (
            self.z1
            .compute_policy_ee_pose_in_base()
        )

        ee_quat_b = (
            quat_from_rotmat_wxyz(
                ee_rot_b
            )
        )

        ee_quat_b = quat_unique_wxyz(
            quat_normalize_wxyz(
                ee_quat_b
            )
        )

        _, _, base_yaw = (
            euler_xyz_from_quat_wxyz(
                base_q
            )
        )

        plb_q_w = (
            quat_from_yaw_wxyz(
                base_yaw
            )
        )

        plb_q_w = quat_unique_wxyz(
            quat_normalize_wxyz(
                plb_q_w
            )
        )

        base_pos_w = np.array(
            [
                0.0,
                0.0,
                self.base_height,
            ],
            dtype=np.float32,
        )

        plb_pos_w = np.array(
            [
                0.0,
                0.0,
                self.ground_z,
            ],
            dtype=np.float32,
        )

        ee_pos_w = (
            base_pos_w
            + quat_apply_wxyz(
                base_q,
                ee_pos_b,
            )
        )

        ee_quat_w = (
            quat_mul_wxyz(
                base_q,
                ee_quat_b,
            )
        )

        ee_quat_w = quat_unique_wxyz(
            quat_normalize_wxyz(
                ee_quat_w
            )
        )

        ee_pos_plb = (
            quat_apply_inverse_wxyz(
                plb_q_w,
                ee_pos_w
                - plb_pos_w,
            )
        )

        ee_quat_plb = (
            quat_mul_wxyz(
                quat_conjugate_wxyz(
                    plb_q_w
                ),
                ee_quat_w,
            )
        )

        ee_quat_plb = quat_unique_wxyz(
            quat_normalize_wxyz(
                ee_quat_plb
            )
        )

        _, pitch, yaw = (
            euler_xyz_from_quat_wxyz(
                ee_quat_plb
            )
        )

        return (
            np.asarray(
                ee_pos_plb,
                dtype=np.float32,
            ).reshape(
                3,
            ),
            float(
                yaw
            ),
            float(
                pitch
            ),
        )

    def get_gripper_geometry(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Returns:
            gripper_orientation_base : 6-D [local +X in B, local +Y in B]
            gripper_center_pos_base  : 3-D

        The Z1 policy EE frame is treated as the sim2sim gripperStator frame.
        """
        stator_pos_b, stator_rot_b = (
            self.z1
            .compute_policy_ee_pose_in_base()
        )

        stator_pos_b = np.asarray(
            stator_pos_b,
            dtype=np.float32,
        ).reshape(
            3,
        )

        stator_rot_b = np.asarray(
            stator_rot_b,
            dtype=np.float32,
        ).reshape(
            3,
            3,
        )

        gripper_center_b = (
            stator_pos_b
            + stator_rot_b
            @ self.gripper_center_offset_local
        ).astype(
            np.float32
        )

        gripper_x_axis_b = (
            stator_rot_b[
                :,
                0,
            ].astype(
                np.float32
            )
        )

        gripper_y_axis_b = (
            stator_rot_b[
                :,
                1,
            ].astype(
                np.float32
            )
        )

        orientation_b = np.concatenate(
            [
                gripper_x_axis_b,
                gripper_y_axis_b,
            ],
            dtype=np.float32,
        )

        return (
            orientation_b,
            gripper_center_b,
        )

    def build_neutral_ee_command(
        self,
    ) -> np.ndarray:
        return (
            build_keypoints_from_kp0_yaw_pitch_plb(
                kp0=
                    self.neutral_kp0,

                yaw=
                    self.neutral_ee_yaw,

                pitch=
                    self.neutral_ee_pitch,

                roll=
                    self.fixed_ee_roll,

                kp_dx=
                    self.ee_kp_dx,

                kp_dz=
                    self.ee_kp_dz,
            )
        )

    # ======================================================================
    # Perception / grasp proxy
    # ======================================================================

    def _read_perception_nonblocking(
        self,
    ) -> Dict[str, Any]:
        if self.vo_base_height_enabled:
            snap = (
                self.perception
                .get_latest_snapshot(
                    projected_gravity_b=
                        self.projected_gravity_b.copy()
                )
            )

            height_state = snap.get(
                "base_height",
                {},
            )

            self.last_base_height_state = (
                height_state
            )

            # IMPORTANT:
            #   valid height -> use the fresh VO estimate
            #   invalid/stale height -> FREEZE the last valid value
            # Never jump back to the 0.6017-m anchor during runtime.
            if bool(
                height_state.get(
                    "valid",
                    False,
                )
            ):
                height_m = float(
                    height_state[
                        "height_m"
                    ]
                )

                if np.isfinite(
                    height_m
                ):
                    self.base_height = (
                        height_m
                    )
        else:
            snap = (
                self.perception
                .get_latest_snapshot()
            )

        self.last_perception_snapshot = (
            snap
        )

        return snap

    def _raw_object_point(
        self,
        perception_snap: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        obj = perception_snap.get(
            "object",
            {},
        )

        if not bool(
            obj.get(
                "valid",
                False,
            )
        ):
            return None

        return point3_or_none(
            obj.get(
                "position_base"
            )
        )

    def _raw_retrieval_point(
        self,
        perception_snap: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        ret = perception_snap.get(
            "retrieval",
            {},
        )

        if not bool(
            ret.get(
                "valid",
                False,
            )
        ):
            return None

        return point3_or_none(
            ret.get(
                "retrieval_target_base"
            )
        )

    def _latch_command_assumed_grasp_if_needed(
        self,
        executed_close: bool,
    ) -> None:
        """
        command_assumed mode:

        The first executed CLOSE/grasp command is itself treated as successful
        grasp.  The transition is latched:

            False -> True

        There is no automatic True -> False transition in this mode.  A new
        controller run/reset initializes the proxy back to False.

        This helper is called inside decode_hl_action(), before the corresponding
        50-Hz hardware command is sent.  Therefore the first physical CLOSE
        command and proxy=True occur in the same control iteration.
        """
        if (
            self.grasp_proxy_mode
            != "command_assumed"
        ):
            return

        if (
            not executed_close
            or self.grasp_confidence_proxy
        ):
            return

        old_proxy = bool(
            self.grasp_confidence_proxy
        )

        self.grasp_confidence_proxy = True
        self.grasp_proxy_enter_count = 0
        self.grasp_proxy_exit_count = 0
        self.last_grasp_error = 0.0

        print(
            "[GRASP-PROXY] "
            f"{int(old_proxy)} -> 1 | "
            "mode=COMMAND_ASSUMED | "
            "reason=FIRST_EXECUTED_CLOSE_COMMAND"
        )

    def update_grasp_confidence_proxy(
        self,
        perception_snap: Dict[str, Any],
    ) -> None:
        """
        Update grasp_confidence_proxy at the 10-Hz HL boundary.

        heuristic:
            Deployable Stage-2 heuristic:
                CLOSE commanded
                AND gripper is not fully closed
                AND gripper angle is holding
            with configured enter/exit hysteresis.

            The object <-> gripper-center distance gate was intentionally
            removed, so the heuristic no longer depends on object perception.

        command_assumed:
            No geometric/stall inference is done here. The proxy is latched
            immediately in decode_hl_action() when the first executed CLOSE
            command is generated, and remains True until controller reset.
        """
        # All gripper-angle semantics use TRAINING coordinates.
        gripper_q = (
            self._get_gripper_q_training()
        )

        if (
            self.grasp_proxy_mode
            == "command_assumed"
        ):
            self.prev_gripper_joint_pos = (
                gripper_q
            )

            if self.grasp_confidence_proxy:
                self.last_grasp_error = 0.0

            return

        # ------------------------------------------------------------------
        # Heuristic mode.
        #
        # grasp_error (object <-> gripper-center distance) is still computed,
        # but ONLY as a diagnostic for the runtime debug print.  It no longer
        # gates the proxy, so the proxy does not depend on object perception.
        # ------------------------------------------------------------------
        _, gripper_center_b = (
            self.get_gripper_geometry()
        )

        raw_object_b = (
            self._raw_object_point(
                perception_snap
            )
        )

        if self.grasp_confidence_proxy:
            grasp_error = 0.0
        elif raw_object_b is None:
            grasp_error = float(
                "inf"
            )
        else:
            grasp_error = float(
                np.linalg.norm(
                    raw_object_b
                    - gripper_center_b
                )
            )

        self.last_grasp_error = (
            grasp_error
        )

        close_commanded = (
            self.executed_gripper_cmd_norm
            > 0.0
        )

        gripper_angle_delta = abs(
            gripper_q
            - self.prev_gripper_joint_pos
        )

        gripper_not_fully_closed = (
            gripper_q
            < self.gripper_not_fully_closed_angle_threshold
        )

        gripper_angle_holding = (
            gripper_angle_delta
            < self.gripper_angle_hold_threshold
        )

        proxy_candidate = (
            close_commanded
            and gripper_not_fully_closed
            and gripper_angle_holding
        )

        old_proxy = bool(
            self.grasp_confidence_proxy
        )

        if not self.grasp_confidence_proxy:
            if proxy_candidate:
                self.grasp_proxy_enter_count += 1
            else:
                self.grasp_proxy_enter_count = 0

            if (
                self.grasp_proxy_enter_count
                >= self.grasp_proxy_enter_steps
            ):
                self.grasp_confidence_proxy = True
                self.grasp_proxy_enter_count = 0

        else:
            if not proxy_candidate:
                self.grasp_proxy_exit_count += 1
            else:
                self.grasp_proxy_exit_count = 0

            if (
                self.grasp_proxy_exit_count
                >= self.grasp_proxy_exit_steps
            ):
                self.grasp_confidence_proxy = False
                self.grasp_proxy_exit_count = 0

        self.prev_gripper_joint_pos = (
            gripper_q
        )

        if (
            old_proxy
            != self.grasp_confidence_proxy
        ):
            print(
                "[GRASP-PROXY] "
                f"{int(old_proxy)} -> "
                f"{int(self.grasp_confidence_proxy)} | "
                "mode=HEURISTIC | "
                f"error={grasp_error:.3f} m | "
                f"gripper_q_train={gripper_q:+.3f}"
            )

    def resolve_effective_task_state(
        self,
        perception_snap: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Resolve exactly what may be supplied to the HL observation.

        Never returns zero/stale stand-ins.
        """
        obj = perception_snap.get(
            "object",
            {},
        )

        ret = perception_snap.get(
            "retrieval",
            {},
        )

        gripper_orientation_b, gripper_center_b = (
            self.get_gripper_geometry()
        )

        if self.grasp_confidence_proxy:
            object_valid = True
            object_pos_b = (
                gripper_center_b.copy()
            )
            object_source = (
                "GRIPPER_CENTER"
            )

        else:
            object_pos_b = (
                self._raw_object_point(
                    perception_snap
                )
            )

            object_valid = (
                object_pos_b
                is not None
            )

            object_source = (
                obj.get(
                    "source"
                )
                if object_valid
                else None
            )

        retrieval_pos_b = (
            self._raw_retrieval_point(
                perception_snap
            )
        )

        retrieval_valid = (
            retrieval_pos_b
            is not None
        )

        retrieval_source = (
            ret.get(
                "source"
            )
            if retrieval_valid
            else None
        )

        valid = bool(
            object_valid
            and retrieval_valid
        )

        reasons = []

        if not object_valid:
            reasons.append(
                "OBJECT_INVALID:"
                + str(
                    obj.get(
                        "reason"
                    )
                )
            )

        if not retrieval_valid:
            reasons.append(
                "RETRIEVAL_INVALID:"
                + str(
                    ret.get(
                        "reason"
                    )
                )
            )

        return {
            "valid":
                valid,

            "reason":
                (
                    "VALID"
                    if valid
                    else "|".join(
                        reasons
                    )
                ),

            "object_valid":
                bool(
                    object_valid
                ),

            "object_position_base":
                object_pos_b,

            "object_source":
                object_source,

            "retrieval_valid":
                bool(
                    retrieval_valid
                ),

            "retrieval_target_base":
                retrieval_pos_b,

            "retrieval_source":
                retrieval_source,

            "gripper_orientation_base":
                gripper_orientation_b,

            "gripper_center_pos_base":
                gripper_center_b,

            "raw_object_valid":
                bool(
                    obj.get(
                        "valid",
                        False,
                    )
                ),

            "raw_retrieval_valid":
                bool(
                    ret.get(
                        "valid",
                        False,
                    )
                ),
        }

    # ======================================================================
    # High-level observation / policy
    # ======================================================================

    def build_previous_hl_action(
        self,
    ) -> np.ndarray:
        denom = np.maximum(
            np.abs(
                self.base_cmd_scale
            ),
            1e-6,
        )

        effective_base_action = np.clip(
            self.base_command
            / denom,
            -1.0,
            1.0,
        )

        return np.concatenate(
            [
                effective_base_action,

                self.current_hl_action[
                    3:8
                ],

                np.array(
                    [
                        self.executed_gripper_cmd_norm
                    ],
                    dtype=np.float32,
                ),
            ],
            dtype=np.float32,
        )

    def build_hl_obs_frame(
        self,
        task: Dict[str, Any],
    ) -> np.ndarray:
        if not task[
            "valid"
        ]:
            raise RuntimeError(
                "Refusing to build HL observation from invalid task state: "
                + str(
                    task[
                        "reason"
                    ]
                )
            )

        leg_pos = (
            self.b2w_joint_pos[
                :12
            ].copy()
        )

        arm_pos = (
            self.z1.q.copy()
        )

        arm_joint_vel = (
            self.z1.qd.copy()
        )

        leg_pos_rel = (
            leg_pos
            - self.default_joint_pos_policy[
                :12
            ]
        ).astype(
            np.float32
        )

        arm_pos_rel = (
            arm_pos
            - self.default_joint_pos_policy[
                12:18
            ]
        ).astype(
            np.float32
        )

        gripper_q_training = (
            self._get_gripper_q_training()
        )

        gripper_pos_rel = np.array(
            [
                gripper_q_training
                - self.default_gripper_pos
            ],
            dtype=np.float32,
        )

        previous_hl_action = (
            self.build_previous_hl_action()
        )

        obs = np.concatenate(
            [
                self.base_ang_vel_b,                 # 3
                self.projected_gravity_b,            # 3
                leg_pos_rel,                         # 12
                arm_pos_rel,                         # 6
                gripper_pos_rel,                     # 1
                arm_joint_vel,                       # 6
                task["object_position_base"],        # 3
                task["gripper_orientation_base"],    # 6
                task["gripper_center_pos_base"],     # 3
                task["retrieval_target_base"],       # 3
                previous_hl_action,                  # 9
                np.array(
                    [
                        float(
                            self.grasp_confidence_proxy
                        )
                    ],
                    dtype=np.float32,
                ),                                   # 1
            ],
            dtype=np.float32,
        )

        assert (
            obs.shape
            == (
                self.hl_obs_dim_per_step,
            )
        ), obs.shape

        if not np.isfinite(
            obs
        ).all():
            raise RuntimeError(
                "Non-finite value in HL observation."
            )

        return obs

    def decode_hl_action(
        self,
        action: np.ndarray,
    ) -> None:
        action = np.clip(
            np.asarray(
                action,
                dtype=np.float32,
            ).reshape(
                self.hl_action_dim,
            ),
            -1.0,
            1.0,
        )

        self.current_hl_action[:] = (
            action
        )

        self.base_command[:] = (
            action[
                0:3
            ]
            * self.base_cmd_scale
        ).astype(
            np.float32
        )

        (
            actual_ee_pos_plb,
            actual_ee_yaw_plb,
            actual_ee_pitch_plb,
        ) = (
            self.compute_actual_ee_pose_plb()
        )

        kp0_cmd = (
            actual_ee_pos_plb
            + action[
                3:6
            ]
            * self.kp0_delta_scale
        )

        kp0_cmd[
            0
        ] = np.clip(
            kp0_cmd[
                0
            ],
            self.kp0_x_range[
                0
            ],
            self.kp0_x_range[
                1
            ],
        )

        kp0_cmd[
            1
        ] = np.clip(
            kp0_cmd[
                1
            ],
            self.kp0_y_range[
                0
            ],
            self.kp0_y_range[
                1
            ],
        )

        kp0_cmd[
            2
        ] = np.clip(
            kp0_cmd[
                2
            ],
            self.kp0_z_range[
                0
            ],
            self.kp0_z_range[
                1
            ],
        )

        yaw_cmd = np.clip(
            actual_ee_yaw_plb
            + float(
                action[
                    6
                ]
            )
            * self.ee_yaw_delta_scale,
            self.ee_yaw_range[
                0
            ],
            self.ee_yaw_range[
                1
            ],
        )

        pitch_cmd = np.clip(
            actual_ee_pitch_plb
            + float(
                action[
                    7
                ]
            )
            * self.ee_pitch_delta_scale,
            self.ee_pitch_range[
                0
            ],
            self.ee_pitch_range[
                1
            ],
        )

        self.ee_cmd_plb_current[:] = (
            build_keypoints_from_kp0_yaw_pitch_plb(
                kp0=
                    kp0_cmd,

                yaw=
                    float(
                        yaw_cmd
                    ),

                pitch=
                    float(
                        pitch_cmd
                    ),

                roll=
                    self.fixed_ee_roll,

                kp_dx=
                    self.ee_kp_dx,

                kp_dz=
                    self.ee_kp_dz,
            )
        )

        self.raw_gripper_action = float(
            action[
                8
            ]
        )

        binary_close = (
            self.raw_gripper_action
            > self.gripper_binary_threshold
        )

        executed_close = (
            binary_close
            or (
                self.stage2_force_gripper_close_enabled
                and self.grasp_confidence_proxy
            )
        )

        self.executed_gripper_cmd_norm = (
            1.0
            if executed_close
            else -1.0
        )

        self.gripper_target = (
            self.gripper_close_pos
            if executed_close
            else self.gripper_open_pos
        )

        # command_assumed mode:
        # first executed CLOSE/grasp command -> proxy=True immediately.
        self._latch_command_assumed_grasp_if_needed(
            executed_close=
                executed_close,
        )

    def run_high_level_policy(
        self,
        task: Dict[str, Any],
        reseed_history: bool = False,
    ) -> None:
        hl_frame = (
            self.build_hl_obs_frame(
                task
            )
        )

        if reseed_history:
            fill_histories(
                self.hl_histories,
                hl_frame,
                self.hl_feature_dims,
                self.hl_history_length,
            )

        else:
            append_histories(
                self.hl_histories,
                hl_frame,
                self.hl_feature_dims,
            )

        hl_obs_stack = (
            flatten_feature_major(
                self.hl_histories
            )
        )

        assert (
            hl_obs_stack.shape
            == (
                self.hl_obs_dim,
            )
        ), hl_obs_stack.shape

        if (
            self.debug_hl_obs_enabled
            and self.debug_hl_obs_count
            < self.debug_obs_print_max
        ):
            print(
                "[HL-OBS] "
                f"tick={self.hl_tick} "
                f"shape={hl_obs_stack.shape} "
                f"min={hl_obs_stack.min():+.3f} "
                f"max={hl_obs_stack.max():+.3f}"
            )

            self.debug_hl_obs_count += 1

        action = (
            self.high_session.run(
                [
                    self.high_output_name
                ],
                {
                    self.high_input_name:
                        hl_obs_stack[
                            None,
                            :,
                        ]
                },
            )[0][0]
            .astype(
                np.float32
            )
        )

        self.decode_hl_action(
            action
        )

        self.hl_tick += 1

    def run_first_high_level_policy_from_initialized_history(
        self,
    ) -> None:
        """
        Match sim2sim reset semantics:
            initialized HL history -> first HL action/decode
        without appending an extra HL frame first.
        """
        hl_obs_stack = (
            flatten_feature_major(
                self.hl_histories
            )
        )

        assert (
            hl_obs_stack.shape
            == (
                self.hl_obs_dim,
            )
        ), hl_obs_stack.shape

        action = (
            self.high_session.run(
                [
                    self.high_output_name
                ],
                {
                    self.high_input_name:
                        hl_obs_stack[
                            None,
                            :,
                        ]
                },
            )[0][0]
            .astype(
                np.float32
            )
        )

        self.decode_hl_action(
            action
        )

        self.hl_tick = 1

    # ======================================================================
    # Low-level observation / policy
    # ======================================================================

    def build_ll_obs_frame(
        self,
    ) -> np.ndarray:
        leg_pos = (
            self.b2w_joint_pos[
                :12
            ].copy()
        )

        wheel_vel = (
            self.b2w_joint_vel[
                12:16
            ].copy()
        )

        arm_pos = (
            self.z1.q.copy()
        )

        arm_vel = (
            self.z1.qd.copy()
        )

        joint_pos_policy = np.concatenate(
            [
                leg_pos,
                arm_pos,
            ],
            dtype=np.float32,
        )

        joint_pos_rel = (
            joint_pos_policy
            - self.default_joint_pos_policy
        )

        joint_pos_leg_rel = (
            joint_pos_rel[
                :12
            ]
        )

        joint_pos_arm_rel = (
            joint_pos_rel[
                12:18
            ]
        )

        joint_vel_leg = (
            self.b2w_joint_vel[
                :12
            ].copy()
        )

        joint_vel_arm = (
            arm_vel
        )

        joint_vel_wheel = (
            wheel_vel
        )

        obs = np.concatenate(
            [
                self.base_ang_vel_b,        # 3
                self.projected_gravity_b,   # 3
                self.base_command,          # 3
                self.ee_cmd_plb_current,    # 9
                joint_pos_leg_rel,          # 12
                joint_pos_arm_rel,          # 6
                joint_vel_leg,              # 12
                joint_vel_arm,              # 6
                joint_vel_wheel,            # 4
                self.last_ll_action,         # 22
            ],
            dtype=np.float32,
        )

        assert (
            obs.shape
            == (
                self.ll_obs_dim_per_step,
            )
        ), obs.shape

        if not np.isfinite(
            obs
        ).all():
            raise RuntimeError(
                "Non-finite value in LL observation."
            )

        return obs

    def run_low_level_policy(
        self,
    ) -> None:
        ll_frame = (
            self.build_ll_obs_frame()
        )

        append_histories(
            self.ll_histories,
            ll_frame,
            self.ll_feature_dims,
        )

        ll_obs_stack = (
            flatten_feature_major(
                self.ll_histories
            )
        )

        assert (
            ll_obs_stack.shape
            == (
                self.ll_obs_dim,
            )
        ), ll_obs_stack.shape

        if (
            self.debug_ll_obs_enabled
            and self.debug_ll_obs_count
            < self.debug_obs_print_max
        ):
            print(
                "[LL-OBS] "
                f"tick={self.ll_tick} "
                f"shape={ll_obs_stack.shape} "
                f"min={ll_obs_stack.min():+.3f} "
                f"max={ll_obs_stack.max():+.3f}"
            )

            self.debug_ll_obs_count += 1

        action = (
            self.low_session.run(
                [
                    self.low_output_name
                ],
                {
                    self.low_input_name:
                        ll_obs_stack[
                            None,
                            :,
                        ]
                },
            )[0][0]
            .astype(
                np.float32
            )
        )

        self.last_ll_action[:] = (
            action
        )

        leg_act = (
            action[
                self.leg_action_indices
            ]
        )

        arm_act = (
            action[
                self.arm_action_indices
            ]
        )

        wheel_act = (
            action[
                self.wheel_action_indices
            ]
        )

        # EXACT hierarchical sim2sim semantics:
        # no policy blend-in, no arm target rate limiter.
        self.leg_target[:] = (
            self.default_leg_pos_policy
            + self.leg_action_scale
            * leg_act
        ).astype(
            np.float32
        )

        self.arm_target[:] = (
            self.default_arm_pos
            + self.arm_action_scale
            * arm_act
        ).astype(
            np.float32
        )

        self.wheel_cmd[:] = (
            self.wheel_action_scale
            * wheel_act
        ).astype(
            np.float32
        )

    # ======================================================================
    # B2W / Z1 actuation
    # ======================================================================

    def _write_b2w_pose_cmd_policy(
        self,
        target_b2w_pos_policy: np.ndarray,
        use_pd_gains: bool,
    ) -> None:
        target_b2w_pos_policy = np.asarray(
            target_b2w_pos_policy,
            dtype=np.float32,
        ).reshape(
            16,
        )

        target_b2w_pos_hw = (
            target_b2w_pos_policy[
                self.policy_to_hardware_joint_indices
            ]
        )

        if use_pd_gains:
            kps_hw = (
                self.kps_pd_hw
            )

            kds_hw = (
                self.kds_pd_hw
            )

            wheel_kps_hw = (
                self.wheel_kps_pd_hw
            )

            wheel_kds_hw = (
                self.wheel_kds_pd_hw
            )

        else:
            kps_hw = (
                self.kps_rl_hw
            )

            kds_hw = (
                self.kds_rl_hw
            )

            wheel_kps_hw = (
                self.wheel_kps_rl_hw
            )

            wheel_kds_hw = (
                self.wheel_kds_rl_hw
            )

        for hw_idx in range(
            self.num_b2w_dof
        ):
            if (
                hw_idx
                in self.leg_hardware_indices
            ):
                self.low_cmd.motor_cmd[
                    hw_idx
                ].q = float(
                    target_b2w_pos_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].dq = 0.0

                self.low_cmd.motor_cmd[
                    hw_idx
                ].kp = float(
                    kps_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].kd = float(
                    kds_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].tau = 0.0

            else:
                self.low_cmd.motor_cmd[
                    hw_idx
                ].q = 0.0

                self.low_cmd.motor_cmd[
                    hw_idx
                ].dq = 0.0

                self.low_cmd.motor_cmd[
                    hw_idx
                ].kp = float(
                    wheel_kps_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].kd = float(
                    wheel_kds_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].tau = 0.0

    def _write_b2w_rl_cmd(
        self,
    ) -> None:
        processed_policy = (
            self.default_b2w_pos_policy.copy()
        )

        processed_policy[
            self.leg_policy_indices
        ] = self.leg_target

        target_hw = (
            processed_policy[
                self.policy_to_hardware_joint_indices
            ]
        )

        for hw_idx in range(
            self.num_b2w_dof
        ):
            if (
                hw_idx
                in self.leg_hardware_indices
            ):
                self.low_cmd.motor_cmd[
                    hw_idx
                ].q = float(
                    target_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].dq = 0.0

                self.low_cmd.motor_cmd[
                    hw_idx
                ].kp = float(
                    self.kps_rl_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].kd = float(
                    self.kds_rl_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].tau = 0.0

            else:
                wheel_idx = (
                    self.hw_to_wheel_cmd_indices[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].q = 0.0

                self.low_cmd.motor_cmd[
                    hw_idx
                ].dq = float(
                    self.wheel_cmd[
                        wheel_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].kp = float(
                    self.wheel_kps_rl_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].kd = float(
                    self.wheel_kds_rl_hw[
                        hw_idx
                    ]
                )

                self.low_cmd.motor_cmd[
                    hw_idx
                ].tau = 0.0

    def send_policy_targets(
        self,
    ) -> None:
        self._write_b2w_rl_cmd()
        self.send_b2w_cmd()

        # Arm: validated runtime PD.
        # Gripper: external training-space IdealPD + exact IsaacLab DCMotor
        # torque-speed law through tau_f.  The gripper target is in TRAINING
        # coordinates (close=0, open=-pi/2).
        self.z1.track_target_pd_runtime_once(
            q_target=
                self.arm_target.copy(),

            gripper_q_target_training=
                float(
                    self.gripper_target
                ),

            use_startup_gains=
                False,
        )

    # ======================================================================
    # Perception hold / protection
    # ======================================================================

    def _set_safe_perception_hold_commands(
        self,
    ) -> None:
        self.base_command[:] = 0.0

        # Track the current measured EE command, never a stale HL EE target.
        self.ee_cmd_plb_current[:] = (
            self.compute_ee_current_kp_plb()
        )

        self.current_hl_action[:] = 0.0
        self.raw_gripper_action = 0.0

        if self.grasp_confidence_proxy:
            self.gripper_target = (
                self.gripper_close_pos
            )

            self.executed_gripper_cmd_norm = (
                1.0
            )

        else:
            # Hold around the current gripper position during perception
            # interruption.  The external DCMotor API expects a TRAINING-space
            # target, never the raw SDK coordinate.
            self.gripper_target = (
                self._get_gripper_q_training()
            )

            self.executed_gripper_cmd_norm = (
                -1.0
            )

    def _begin_perception_hold(
        self,
        reason: str,
    ) -> None:
        if not self.perception_hold_active:
            print(
                "[PERCEPTION-HOLD] ENTER | "
                + reason
            )

        self.perception_hold_active = True
        self.perception_hold_reason = str(
            reason
        )

        if self.perception_invalid_since is None:
            self.perception_invalid_since = (
                time.monotonic()
            )

        self._set_safe_perception_hold_commands()

    def _perception_fault_timed_out(
        self,
    ) -> bool:
        if (
            self.perception_invalid_since
            is None
        ):
            return False

        elapsed = (
            time.monotonic()
            - self.perception_invalid_since
        )

        return (
            elapsed
            >= self.perception_fault_timeout_s
        )

    def _recover_perception_hold_and_run_hl(
        self,
        task: Dict[str, Any],
    ) -> None:
        print(
            "[PERCEPTION-HOLD] RECOVER | "
            f"object={task['object_source']} | "
            f"retrieval={task['retrieval_source']}"
        )

        self.perception_hold_active = False
        self.perception_invalid_since = None
        self.perception_hold_reason = ""

        # Current safe-hold commands are represented as a zero normalized HL action.
        # Re-seed the entire 3-frame HL history with the first current valid frame
        # before resuming inference.
        self.current_hl_action[:] = 0.0
        self.base_command[:] = 0.0

        self.run_high_level_policy(
            task,
            reseed_history=
                True,
        )

    # ======================================================================
    # Startup / damping
    # ======================================================================

    def _write_b2w_high_damping_cmd(
        self,
    ) -> None:
        for hw_idx in range(
            self.num_b2w_dof
        ):
            self.low_cmd.motor_cmd[
                hw_idx
            ].q = 0.0

            self.low_cmd.motor_cmd[
                hw_idx
            ].dq = 0.0

            self.low_cmd.motor_cmd[
                hw_idx
            ].kp = 0.0

            self.low_cmd.motor_cmd[
                hw_idx
            ].kd = float(
                self.damping_kd_wheel
                if hw_idx
                in self.wheel_hardware_indices
                else self.damping_kd_b2w
            )

            self.low_cmd.motor_cmd[
                hw_idx
            ].tau = 0.0

    def _send_b2w_high_damping_once(
        self,
    ) -> None:
        """
        Send one B2W high-damping command.

        Used both by the steady protection loop and while the Z1 arm is moving
        during manual recovery, so B2W protection packets do not stop for the
        several-second Z1 trajectory.
        """
        self._write_b2w_high_damping_cmd()
        self.send_b2w_cmd()

    def _send_z1_high_damping_once(
        self,
    ) -> None:
        """
        Best-effort one-shot Z1 high-damping/current-position command.

        If Z1 is physically PASSIVE, the SDK may ignore LOWCMD actuation.  This
        helper intentionally does NOT force setFsmLowcmd(): nothing in
        protection mode -- including the manual leg recovery -- ever moves the
        arm or drags it out of PASSIVE.
        """
        with self.z1._comm_lock:
            self.z1._send_arm_command_once(
                q_cmd=
                    self.z1.q.copy(),

                gripper_q_cmd=
                    float(
                        self.z1.gripper_q
                    ),

                kp_cmd=
                    self.damping_kp_z1,

                kd_cmd=
                    self.damping_kd_z1,

                qd_cmd=
                    np.zeros(
                        6,
                        dtype=np.float32,
                    ),

                tau_cmd=
                    np.zeros(
                        6,
                        dtype=np.float32,
                    ),
            )

    def _move_b2w_legs_to_recover_target(
        self,
    ) -> bool:
        """
        Slowly move ONLY the 12 B2W leg joints from measured q ->
        damping_recover_leg_target (default: the SQUAT pose).

        The 4 wheel joints are intentionally NOT position-servoed to an
        absolute angle.  _write_b2w_pose_cmd_policy(..., use_pd_gains=True)
        applies the existing wheel PD configuration, whose Kp is zero and Kd
        is damping.

        The Z1 arm is NOT commanded to move: every tick re-sends exactly the
        same protection command as the steady damping loop, so the arm holds
        its current position and PASSIVE is never forced back into LOWCMD.
        """
        duration_s = float(
            self.damping_recover_leg_duration_s
        )

        target_leg = (
            self.damping_recover_leg_target.copy()
        )

        # Refresh measured B2W state immediately before constructing the path.
        self._read_b2w_sensors_once()

        init_leg = (
            self.b2w_joint_pos[
                self.leg_policy_indices
            ].copy()
        )

        num_steps = max(
            1,
            int(
                round(
                    duration_s
                    / self.control_dt
                )
            ),
        )

        print(
            "[RECOVERY] B2W leg12 current q -> target | "
            f"duration={duration_s:.2f}s | "
            "wheels=dq0+damping | "
            "Z1=untouched (protection hold)"
        )

        print(
            "[RECOVERY] leg_start ="
            + np.array2string(
                init_leg,
                precision=3,
                suppress_small=True,
                max_line_width=200,
            )
        )

        print(
            "[RECOVERY] leg_target="
            + np.array2string(
                target_leg,
                precision=3,
                suppress_small=True,
                max_line_width=200,
            )
        )

        next_deadline = (
            time.perf_counter()
        )

        for step in range(
            num_steps
        ):
            alpha = (
                float(
                    step + 1
                )
                / float(
                    num_steps
                )
            )

            target_step_policy = np.zeros(
                self.num_b2w_dof,
                dtype=np.float32,
            )

            # Only leg12 are position-interpolated.
            target_step_policy[
                self.leg_policy_indices
            ] = (
                init_leg
                * (
                    1.0
                    - alpha
                )
                + target_leg
                * alpha
            )

            # Wheel q values are irrelevant here because the wheel PD path
            # explicitly commands q=0, Kp=0 and damping Kd.
            target_step_policy[
                self.wheel_policy_indices
            ] = 0.0

            self._write_b2w_pose_cmd_policy(
                target_step_policy,
                use_pd_gains=
                    True,
            )

            self.send_b2w_cmd()

            # The arm is NOT part of this recovery.  Keep sending exactly the
            # protection-mode Z1 command so it stays damped where it is.
            try:
                self._send_z1_high_damping_once()
            except Exception as exc:
                print(
                    "[RECOVERY][WARN] Z1 protection command failed: "
                    + repr(
                        exc
                    )
                )

            next_deadline += (
                self.control_dt
            )

            sleep_s = (
                next_deadline
                - time.perf_counter()
            )

            if sleep_s > 0.0:
                time.sleep(
                    sleep_s
                )

        # Verify measured leg position before declaring success.
        self._read_b2w_sensors_once()

        leg_error_max = float(
            np.max(
                np.abs(
                    self.b2w_joint_pos[
                        self.leg_policy_indices
                    ]
                    - target_leg
                )
            )
        )

        print(
            "[RECOVERY] B2W leg verification | "
            f"max_abs_err={leg_error_max:.4f} rad | "
            f"tol={self.damping_recover_leg_tolerance_rad:.4f} rad"
        )

        if (
            not np.isfinite(
                leg_error_max
            )
            or leg_error_max
            > self.damping_recover_leg_tolerance_rad
        ):
            print(
                "[RECOVERY][ERROR] "
                "B2W legs did not reach the target within tolerance."
            )

            return False

        return True

    def recover_legs_to_target(
        self,
    ) -> bool:
        """
        Explicit operator-triggered LEG recovery, used ONLY inside protection
        mode.

        Only the 12 B2W leg joints are moved, from their measured position to
        damping_recover_leg_target (default: the SQUAT pose).

        The Z1 arm is deliberately left alone:
          - no setFsmLowcmd(), so a PASSIVE arm stays PASSIVE
          - the protection-mode Z1 command keeps being sent every tick

        The function never resumes the hierarchical policy.  Its caller returns
        to the normal high-damping protection loop after success or failure.
        """
        print(
            "\n"
            + "=" * 96
        )

        print(
            "[RECOVERY] MANUAL LEG RECOVERY STARTED"
        )

        print(
            "[RECOVERY] B2W leg12 -> target pose. "
            "Z1 arm and gripper are NOT moved."
        )

        print(
            "[RECOVERY] Wheels stay zero-velocity/damped."
        )

        print(
            "=" * 96
        )

        try:
            self._read_all_sensors_once()
        except Exception as exc:
            print(
                "[RECOVERY][ERROR] Initial sensor read failed: "
                + repr(
                    exc
                )
            )

            return False

        try:
            legs_ok = (
                self._move_b2w_legs_to_recover_target()
            )

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            print(
                "[RECOVERY][ERROR] B2W leg trajectory failed: "
                + repr(
                    exc
                )
            )

            return False

        if not legs_ok:
            return False

        print(
            "[RECOVERY] SUCCESS: leg12 reached the target within tolerance."
        )

        print(
            "[RECOVERY] Returning to high-damping protection; "
            "hierarchical policy will NOT resume."
        )

        print(
            "=" * 96
            + "\n"
        )

        return True

    def enter_damping_protection_mode(
        self,
        reason: str,
    ) -> None:
        print(
            "\n"
            + "!" * 96
        )

        print(
            "[B2WZ1-HIER][PROTECT] "
            + str(
                reason
            )
        )

        print(
            "[B2WZ1-HIER][PROTECT] "
            "B2W: kp=0, dq=0, tau=0, high kd."
        )

        print(
            "[B2WZ1-HIER][PROTECT] "
            "Z1: q_cmd=current q, qd=0, tau=0, high kd "
            "(PASSIVE is not automatically forced back to LOWCMD)."
        )

        print(
            "[B2WZ1-HIER][PROTECT] "
            "Gripper: hold current measured q."
        )

        if self.damping_leg_recovery_enabled:
            print(
                "[B2WZ1-HIER][PROTECT] "
                "Release A, then press A to run MANUAL LEG recovery: "
                "leg12 -> target pose (arm is NOT moved)."
            )

        print(
            "[B2WZ1-HIER][PROTECT] "
            "Press Ctrl-C to quit."
        )

        print(
            "!" * 96
            + "\n"
        )

        counter = 0

        # Require a release after entering protection if A happened to already
        # be held.  This prevents a previously-held A from immediately causing
        # Z1 to leave PASSIVE and start moving.
        a_is_pressed = bool(
            self.remote_controller.button[
                KeyMap.A
            ]
            == 1
        )

        recovery_a_armed = bool(
            not a_is_pressed
        )

        if (
            self.damping_leg_recovery_enabled
            and not recovery_a_armed
        ):
            print(
                "[PROTECT] A is currently held; "
                "release it before leg recovery can be armed."
            )

        try:
            while True:
                t0 = time.perf_counter()

                try:
                    self._read_all_sensors_once()
                except Exception as exc:
                    print(
                        "[PROTECT][WARN] sensor read failed: "
                        + repr(
                            exc
                        )
                    )

                try:
                    self._send_b2w_high_damping_once()
                except Exception as exc:
                    print(
                        "[PROTECT][WARN] B2W damping command failed: "
                        + repr(
                            exc
                        )
                    )

                try:
                    self._send_z1_high_damping_once()

                except Exception as exc:
                    print(
                        "[PROTECT][WARN] Z1 damping command failed: "
                        + repr(
                            exc
                        )
                    )

                # ----------------------------------------------------------
                # Manual recovery button state machine.
                #
                # Trigger ONLY on a fresh A press after A has been observed
                # released in protection.  After one trigger, a new release is
                # required before another recovery can be triggered.
                # ----------------------------------------------------------
                if self.damping_leg_recovery_enabled:
                    a_pressed_now = bool(
                        self.remote_controller.button[
                            KeyMap.A
                        ]
                        == 1
                    )

                    if not recovery_a_armed:
                        if not a_pressed_now:
                            recovery_a_armed = (
                                True
                            )

                            print(
                                "[PROTECT] A released; "
                                "manual leg recovery is armed."
                            )

                    elif a_pressed_now:
                        # Disarm before starting the blocking recovery.  The
                        # operator must release A afterwards to arm again.
                        recovery_a_armed = (
                            False
                        )

                        print(
                            "[PROTECT] Fresh A press detected -> "
                            "starting manual leg recovery."
                        )

                        recovery_ok = (
                            self.recover_legs_to_target()
                        )

                        if recovery_ok:
                            print(
                                "[PROTECT] Leg recovery completed. "
                                "Remaining in protection mode."
                            )
                        else:
                            print(
                                "[PROTECT][WARN] Leg recovery did not "
                                "complete successfully. Remaining in protection."
                            )

                        # Immediately restore one protection packet on both
                        # subsystems before continuing the steady loop.
                        try:
                            self._read_all_sensors_once()
                        except Exception:
                            pass

                        try:
                            self._send_b2w_high_damping_once()
                        except Exception:
                            pass

                        try:
                            self._send_z1_high_damping_once()
                        except Exception:
                            pass

                        # Restart loop timing after the multi-second recovery.
                        t0 = (
                            time.perf_counter()
                        )

                if (
                    counter
                    % max(
                        1,
                        self.damping_print_period,
                    )
                    == 0
                ):
                    print(
                        "[PROTECT] active | "
                        f"kd_b2w={self.damping_kd_b2w:.1f} | "
                        f"kd_wheel={self.damping_kd_wheel:.1f} | "
                        f"z1_fsm={self.z1.get_fsm_state()} | "
                        f"gripper_q={self.z1.gripper_q:+.3f} | "
                        f"recovery_armed={int(recovery_a_armed)}"
                    )

                counter += 1

                elapsed = (
                    time.perf_counter()
                    - t0
                )

                time.sleep(
                    max(
                        0.0,
                        self.control_dt
                        - elapsed,
                    )
                )

        except KeyboardInterrupt:
            print(
                "[B2WZ1-HIER][PROTECT] Ctrl-C received."
            )

    def zero_torque_state(
        self,
    ) -> None:
        print(
            "[B2WZ1-HIER] Zero torque state. "
            "Press START to continue."
        )

        while (
            self.remote_controller.button[
                KeyMap.start
            ]
            != 1
        ):
            create_zero_cmd(
                self.low_cmd
            )

            self.send_b2w_cmd()

            time.sleep(
                self.control_dt
            )

        print(
            "[B2WZ1-HIER] START pressed."
        )

    def move_b2w_to_pose_policy(
        self,
        target_b2w_pos_policy: np.ndarray,
        duration: float,
        hold_arm: bool = False,
    ) -> None:
        print(
            "[B2WZ1-HIER] Moving B2W to target pose..."
        )

        target_b2w_pos_policy = np.asarray(
            target_b2w_pos_policy,
            dtype=np.float32,
        ).reshape(
            16,
        )

        num_steps = max(
            1,
            int(
                round(
                    duration
                    / self.control_dt
                )
            ),
        )

        init_b2w_pos_hw = np.zeros(
            self.num_b2w_dof,
            dtype=np.float32,
        )

        for hw_idx in range(
            self.num_b2w_dof
        ):
            init_b2w_pos_hw[
                hw_idx
            ] = (
                self.low_state
                .motor_state[
                    hw_idx
                ]
                .q
            )

        init_b2w_pos_policy = (
            init_b2w_pos_hw[
                self.hardware_to_policy_joint_indices
            ]
        )

        for step in range(
            num_steps
        ):
            alpha = (
                float(
                    step + 1
                )
                / float(
                    num_steps
                )
            )

            target_step_policy = (
                init_b2w_pos_policy
                * (
                    1.0
                    - alpha
                )
                + target_b2w_pos_policy
                * alpha
            )

            self._write_b2w_pose_cmd_policy(
                target_step_policy,
                use_pd_gains=
                    True,
            )

            self.send_b2w_cmd()

            if hold_arm:
                self.z1.hold_pose_lowcmd(
                    self.default_arm_pos.copy(),
                    self._gripper_training_to_sdk(
                        self.gripper_open_pos
                    ),
                )

            time.sleep(
                self.control_dt
            )

        print(
            "[B2WZ1-HIER] Reached target B2W pose."
        )

    def hold_arm_default_until_A(
        self,
    ) -> None:
        print(
            "[B2WZ1-HIER] Holding arm default + OPEN gripper. "
            "Press A to continue."
        )

        while (
            self.remote_controller.button[
                KeyMap.A
            ]
            != 1
        ):
            create_zero_cmd(
                self.low_cmd
            )

            self.send_b2w_cmd()

            self.z1.hold_pose_lowcmd(
                self.default_arm_pos.copy(),
                self._gripper_training_to_sdk(
                    self.gripper_open_pos
                ),
            )

            time.sleep(
                self.control_dt
            )

        print(
            "[B2WZ1-HIER] A pressed."
        )

    def hold_default_until_perception_ready_and_A(
        self,
    ) -> None:
        print(
            "[B2WZ1-HIER] Holding full default pose + OPEN gripper."
        )

        print(
            "[B2WZ1-HIER] Policy start requires CURRENT raw "
            "object.valid && retrieval.valid."
        )

        print(
            "[B2WZ1-HIER] When both are valid, press A to start."
        )

        last_print = 0.0

        while True:
            self._write_b2w_pose_cmd_policy(
                self.default_b2w_pos_policy,
                use_pd_gains=
                    True,
            )

            self.send_b2w_cmd()

            self.z1.hold_pose_lowcmd(
                self.default_arm_pos.copy(),
                self._gripper_training_to_sdk(
                    self.gripper_open_pos
                ),
            )

            snap = (
                self._read_perception_nonblocking()
            )

            raw_obj = snap.get(
                "object",
                {},
            )

            raw_ret = snap.get(
                "retrieval",
                {},
            )

            ready = bool(
                raw_obj.get(
                    "valid",
                    False,
                )
                and raw_ret.get(
                    "valid",
                    False,
                )
                and point3_or_none(
                    raw_obj.get(
                        "position_base"
                    )
                )
                is not None
                and point3_or_none(
                    raw_ret.get(
                        "retrieval_target_base"
                    )
                )
                is not None
            )

            now = time.monotonic()

            if (
                now
                - last_print
                >= self.perception_status_print_s
            ):
                print(
                    "[PERCEPTION-WAIT] "
                    f"ready={int(ready)} | "
                    f"OBJ="
                    f"{int(bool(raw_obj.get('valid', False)))}:"
                    f"{raw_obj.get('source')} "
                    f"reason={raw_obj.get('reason')} | "
                    f"RET="
                    f"{int(bool(raw_ret.get('valid', False)))}:"
                    f"{raw_ret.get('source')} "
                    f"reason={raw_ret.get('reason')}"
                )

                last_print = now

            if (
                ready
                and self.remote_controller.button[
                    KeyMap.A
                ]
                == 1
            ):
                # Re-read immediately so an old "ready" state is not accepted.
                final_snap = (
                    self._read_perception_nonblocking()
                )

                final_obj = final_snap.get(
                    "object",
                    {},
                )

                final_ret = final_snap.get(
                    "retrieval",
                    {},
                )

                final_ready = bool(
                    final_obj.get(
                        "valid",
                        False,
                    )
                    and final_ret.get(
                        "valid",
                        False,
                    )
                    and point3_or_none(
                        final_obj.get(
                            "position_base"
                        )
                    )
                    is not None
                    and point3_or_none(
                        final_ret.get(
                            "retrieval_target_base"
                        )
                    )
                    is not None
                )

                if final_ready:
                    print(
                        "[B2WZ1-HIER] A pressed with valid perception. "
                        "Starting hierarchical policy."
                    )

                    return

                print(
                    "[B2WZ1-HIER] A pressed but perception changed invalid; "
                    "continue holding."
                )

            time.sleep(
                self.control_dt
            )

    def anchor_vo_base_height_before_policy_start(
        self,
    ) -> None:
        """
        Establish the one absolute physical-height reference used by VO.

        Timing is intentional:
            B2W is already holding DEFAULT
            + object/retrieval are current-valid
            + user A has accepted policy start
            -> refresh B2 IMU
            -> anchor current VO pose to vo_base_height_anchor_m
            -> initialize LL/HL policy histories

        If VO is temporarily unavailable, keep holding the full DEFAULT pose and
        retry.  No second A press is required.
        """
        if not self.vo_base_height_enabled:
            self.base_height = float(
                self.base_height_anchor_m
            )

            print(
                "[BASE-HEIGHT] VO estimator DISABLED. "
                "Using fixed anchor height "
                f"{self.base_height:.4f} m."
            )

            return

        print(
            "[BASE-HEIGHT] Waiting for fresh VO to establish "
            f"policy-start anchor h={self.base_height_anchor_m:.4f} m..."
        )

        self.perception.reset_vo_base_height_estimator(
            nominal_height_m=
                self.base_height_anchor_m
        )

        last_print = 0.0

        while True:
            # Continue the exact full-default startup hold while waiting.
            self._write_b2w_pose_cmd_policy(
                self.default_b2w_pos_policy,
                use_pd_gains=
                    True,
            )

            self.send_b2w_cmd()

            self.z1.hold_pose_lowcmd(
                self.default_arm_pos.copy(),
                self._gripper_training_to_sdk(
                    self.gripper_open_pos
                ),
            )

            # This refreshes base_quat_wxyz and projected_gravity_b from the
            # latest rt/lowstate before the anchor is computed.
            self._read_all_sensors_once()

            anchor = (
                self.perception
                .anchor_vo_base_height(
                    projected_gravity_b=
                        self.projected_gravity_b.copy(),

                    height_m=
                        self.base_height_anchor_m,
                )
            )

            self.last_base_height_state = (
                anchor
            )

            if bool(
                anchor.get(
                    "valid",
                    False,
                )
            ):
                height_m = float(
                    anchor.get(
                        "height_m",
                        self.base_height_anchor_m,
                    )
                )

                if np.isfinite(
                    height_m
                ):
                    self.base_height = (
                        height_m
                    )

                    print(
                        "[BASE-HEIGHT] ANCHORED | "
                        f"h={self.base_height:.4f} m | "
                        f"session={anchor.get('session_id')} | "
                        f"epoch={anchor.get('epoch')} | "
                        f"up_V={anchor.get('up_V')} | "
                        f"vo_age={anchor.get('vo_age_ms')} ms"
                    )

                    return

            now = time.monotonic()

            if (
                now
                - last_print
                >= self.perception_status_print_s
            ):
                print(
                    "[BASE-HEIGHT-WAIT] "
                    f"valid={int(bool(anchor.get('valid', False)))} | "
                    f"reason={anchor.get('reason')} | "
                    f"vo_reason={anchor.get('vo_reason')} | "
                    f"vo_age={anchor.get('vo_age_ms')} ms"
                )

                last_print = now

            time.sleep(
                self.control_dt
            )

    # ======================================================================
    # History initialization / first action
    # ======================================================================

    def initialize_policy_state_and_history(
        self,
    ) -> Dict[str, Any]:
        """
        Match hierarchical sim2sim reset policy state:
          base_command     = 0
          EE command       = neutral command
          LL last action   = 0
          HL previous act  = 0 / executed open
          proxy            = false
          physical gripper = already OPEN from startup
          histories        = repeat first valid frame
        """
        self._read_all_sensors_once()

        snap = (
            self._read_perception_nonblocking()
        )

        raw_obj = snap.get(
            "object",
            {},
        )

        raw_ret = snap.get(
            "retrieval",
            {},
        )

        if not (
            bool(
                raw_obj.get(
                    "valid",
                    False,
                )
            )
            and bool(
                raw_ret.get(
                    "valid",
                    False,
                )
            )
        ):
            raise RuntimeError(
                "Perception became invalid before history initialization."
            )

        self.base_command[:] = 0.0

        self.ee_cmd_plb_current[:] = (
            self.build_neutral_ee_command()
        )

        self.last_ll_action[:] = 0.0
        self.current_hl_action[:] = 0.0

        self.raw_gripper_action = -1.0
        self.executed_gripper_cmd_norm = -1.0

        self.leg_target[:] = (
            self.default_leg_pos_policy
        )

        self.arm_target[:] = (
            self.default_arm_pos
        )

        self.wheel_cmd[:] = 0.0

        self.gripper_target = (
            self.gripper_open_pos
        )

        self.grasp_confidence_proxy = False
        self.grasp_proxy_enter_count = 0
        self.grasp_proxy_exit_count = 0

        self.prev_gripper_joint_pos = (
            self._get_gripper_q_training()
        )

        task = (
            self.resolve_effective_task_state(
                snap
            )
        )

        if not task[
            "valid"
        ]:
            raise RuntimeError(
                "Invalid initial effective task: "
                + task[
                    "reason"
                ]
            )

        ll_frame0 = (
            self.build_ll_obs_frame()
        )

        hl_frame0 = (
            self.build_hl_obs_frame(
                task
            )
        )

        fill_histories(
            self.ll_histories,
            ll_frame0,
            self.ll_feature_dims,
            self.ll_history_length,
        )

        fill_histories(
            self.hl_histories,
            hl_frame0,
            self.hl_feature_dims,
            self.hl_history_length,
        )

        self.ll_tick = 0
        self.hl_tick = 0

        self.perception_hold_active = False
        self.perception_invalid_since = None
        self.perception_hold_reason = ""

        return task

    def prime_first_hl_and_ll(
        self,
    ) -> None:
        """
        EXACT scheduler alignment with sim2sim:
            initialized history
              -> first HL inference/decode
              -> first LL obs/inference
              -> first 20-ms hardware command block
        """
        self.run_first_high_level_policy_from_initialized_history()

        self.run_low_level_policy()

        self.send_policy_targets()

        # One LL 20-ms block has now been commanded.
        self.ll_tick = 1

    # ======================================================================
    # Runtime control step
    # ======================================================================

    def step_after_prime(
        self,
    ) -> Tuple[
        bool,
        Optional[str],
    ]:
        self._read_all_sensors_once()

        if (
            self.remote_controller.button[
                KeyMap.select
            ]
            == 1
        ):
            return (
                False,
                "SELECT pressed",
            )

        if (
            self.z1.get_fsm_state()
            == self.z1.unitree_arm_interface.ArmFSMState.PASSIVE
        ):
            return (
                False,
                "Z1 entered PASSIVE state",
            )

        perception_snap = (
            self._read_perception_nonblocking()
        )

        is_hl_boundary = (
            self.ll_tick
            % self.ll_steps_per_hl_step
            == 0
        )

        if is_hl_boundary:
            # Proxy summarizes the preceding 0.1-s HL interval.
            self.update_grasp_confidence_proxy(
                perception_snap
            )

        task = (
            self.resolve_effective_task_state(
                perception_snap
            )
        )

        self.last_object_source = (
            task[
                "object_source"
            ]
        )

        self.last_retrieval_source = (
            task[
                "retrieval_source"
            ]
        )

        if not task[
            "valid"
        ]:
            self._begin_perception_hold(
                task[
                    "reason"
                ]
            )

            if (
                self._perception_fault_timed_out()
            ):
                return (
                    False,
                    "Perception invalid for "
                    f">={self.perception_fault_timeout_s:.3f}s: "
                    + self.perception_hold_reason,
                )

        elif self.perception_hold_active:
            # Current task is valid again. Stop the invalid-duration timer
            # immediately, but keep safe hold until a 10-Hz HL boundary.
            self.perception_invalid_since = None

            if is_hl_boundary:
                self._recover_perception_hold_and_run_hl(
                    task
                )
            else:
                self._set_safe_perception_hold_commands()

        elif is_hl_boundary:
            self.run_high_level_policy(
                task,
                reseed_history=
                    False,
            )

        # LL always runs at 50 Hz.
        self.run_low_level_policy()
        self.send_policy_targets()

        self.ll_tick += 1

        self._push_visualizer_state(
            task
        )

        if (
            self.debug_print_period_steps
            > 0
            and self.ll_tick
            % self.debug_print_period_steps
            == 0
        ):
            self.print_runtime_debug(
                task
            )

        return (
            True,
            None,
        )

    def _push_visualizer_state(
        self,
        task: Dict[str, Any],
    ) -> None:
        if self.visualizer is None:
            return

        object_pos_base = (
            task.get(
                "object_position_base"
            )
        )

        retrieval_pos_base = (
            task.get(
                "retrieval_target_base"
            )
        )

        gripper_center_pos_base = (
            task.get(
                "gripper_center_pos_base"
            )
        )

        self.visualizer.push_state(
            base_quat_wxyz=
                self.base_quat_wxyz.copy(),

            base_height=
                float(
                    self.base_height
                ),

            ground_z=
                float(
                    self.ground_z
                ),

            b2w_joint_pos_policy=
                self.b2w_joint_pos.copy(),

            z1_q=
                self.z1.q.copy(),

            gripper_q_training=
                self._get_gripper_q_training(),

            object_pos_base=
                None
                if object_pos_base is None
                else np.asarray(
                    object_pos_base,
                    dtype=np.float32,
                ).copy(),

            retrieval_pos_base=
                None
                if retrieval_pos_base is None
                else np.asarray(
                    retrieval_pos_base,
                    dtype=np.float32,
                ).copy(),

            ee_cmd_plb=
                self.ee_cmd_plb_current.copy(),

            gripper_center_pos_base=
                None
                if gripper_center_pos_base is None
                else np.asarray(
                    gripper_center_pos_base,
                    dtype=np.float32,
                ).copy(),

            grasp_confidence_proxy=
                bool(
                    self.grasp_confidence_proxy
                ),
        )

    def print_runtime_debug(
        self,
        task: Dict[str, Any],
    ) -> None:
        print(
            "[HIER] "
            f"ll={self.ll_tick:06d} "
            f"hl={self.hl_tick:06d} | "
            f"hold={int(self.perception_hold_active)} | "
            f"OBJ={task.get('object_source')} "
            f"RET={task.get('retrieval_source')} | "
            f"proxy={int(self.grasp_confidence_proxy)} "
            f"proxy_mode={self.grasp_proxy_mode} "
            f"grasp_err={self.last_grasp_error:.3f} | "
            f"grip_q_sdk={self.z1.gripper_q:+.3f} "
            f"grip_q_train={self._get_gripper_q_training():+.3f} "
            f"grip_tgt_train={self.gripper_target:+.3f} | "
            f"base_cmd={np.round(self.base_command, 3)}"
        )

        height_state = (
            self.last_base_height_state
            if self.last_base_height_state
            is not None
            else {}
        )

        print(
            "       "
            f"base_h={self.base_height:.4f}m "
            f"base_h_valid="
            f"{int(bool(height_state.get('valid', False)))} "
            f"base_h_reason={height_state.get('reason')} "
            f"base_h_epoch={height_state.get('epoch')} | "
            f"HL={np.round(self.current_hl_action, 3)} | "
            f"leg_act=["
            f"{self.last_ll_action[self.leg_action_indices].min():+.2f},"
            f"{self.last_ll_action[self.leg_action_indices].max():+.2f}] | "
            f"arm_act=["
            f"{self.last_ll_action[self.arm_action_indices].min():+.2f},"
            f"{self.last_ll_action[self.arm_action_indices].max():+.2f}] | "
            f"wheel_act=["
            f"{self.last_ll_action[self.wheel_action_indices].min():+.2f},"
            f"{self.last_ll_action[self.wheel_action_indices].max():+.2f}]"
        )

    # ======================================================================
    # Setup / run / exit
    # ======================================================================

    def setup(
        self,
    ) -> None:
        print(
            "=" * 108
        )

        print(
            "B2WZ1 HIERARCHICAL RETRIEVAL SIM2REAL"
        )

        print(
            "=" * 108
        )

        print(
            f"Low policy       : {self.low_policy_path}"
        )

        print(
            f"High policy      : {self.high_policy_path}"
        )

        print(
            f"LL               : "
            f"{1.0 / self.control_dt:.1f} Hz | "
            f"obs={self.ll_obs_dim} | "
            f"act={self.ll_action_dim}"
        )

        print(
            f"HL               : "
            f"{1.0 / self.hl_control_dt:.1f} Hz | "
            f"obs={self.hl_obs_dim} | "
            f"act={self.hl_action_dim}"
        )

        print(
            f"LL/HL ratio      : {self.ll_steps_per_hl_step}"
        )

        print(
            "LL history       : 5-frame FEATURE-MAJOR"
        )

        print(
            "HL history       : 3-frame FEATURE-MAJOR"
        )

        print(
            "Policy filters   : NONE "
            "(no blend-in, no arm-target rate limiter)"
        )

        print(
            "Gripper runtime  : DCMOTOR "
            "(external IdealPD + exact IsaacLab DCMotor)"
        )

        print(
            "Gripper coord    : q_train = q_sdk - "
            f"{self.z1.gripper_q_offset:.5f}"
        )

        print(
            "Gripper actuator : "
            f"Kp={self.z1.gripper_dcmotor_kp:.1f}, "
            f"Kd={self.z1.gripper_dcmotor_kd:.1f}, "
            f"effort={self.z1.gripper_dcmotor_effort_limit:.1f} Nm, "
            f"stall={self.z1.gripper_dcmotor_saturation_effort:.1f} Nm, "
            f"vel={self.z1.gripper_dcmotor_velocity_limit:.1f} rad/s"
        )

        print(
            "Post-DCMotor cap : NONE"
        )

        if (
            self.z1_arm_runtime_mode
            == "dcmotor"
        ):
            print(
                "Arm runtime      : DCMOTOR "
                "(zero firmware gains, external tau_f)"
            )

            print(
                "Arm actuator     : "
                f"Kp={np.round(self.z1.arm_dcmotor_kp, 1)} | "
                f"Kd={np.round(self.z1.arm_dcmotor_kd, 1)} | "
                f"effort={np.round(self.z1.arm_dcmotor_effort_limit, 1)} Nm"
            )

            print(
                "Arm law          : IdealPD + symmetric effort clip "
                "(training-exact; no torque-speed envelope)"
            )

        else:
            print(
                "Arm runtime      : POSITION_PD "
                "(Z1 firmware position loop)"
            )

            print(
                "Arm actuator     : "
                f"Kp={np.round(self.z1.arm_kps_runtime, 2)} | "
                f"Kd={np.round(self.z1.arm_kds_runtime, 1)}"
            )

        if (
            self.grasp_proxy_mode
            == "command_assumed"
        ):
            grasp_mode_desc = (
                "COMMAND_ASSUMED "
                "(first CLOSE -> proxy=1, latched)"
            )
        else:
            grasp_mode_desc = (
                "HEURISTIC "
                "(gripper stall + hysteresis, no distance gate)"
            )

        print(
            "Grasp proxy mode : "
            + grasp_mode_desc
        )

        print(
            "Object after grasp: gripper center"
        )

        print(
            "Perception fail  : safe hold, then damping after "
            f"{self.perception_fault_timeout_s:.3f}s"
        )

        print(
            "VO Base height   : "
            + (
                "ENABLED | "
                f"policy-start anchor={self.base_height_anchor_m:.4f} m | "
                "invalid/stale -> freeze last valid"
                if self.vo_base_height_enabled
                else (
                    "DISABLED | fixed="
                    f"{self.base_height_anchor_m:.4f} m"
                )
            )
        )

        print(
            "Damping recovery : "
            + (
                "ENABLED | A in protection -> legs only, "
                f"{self.damping_recover_leg_duration_s:.1f}s "
                "(arm NOT moved)"
                if self.damping_leg_recovery_enabled
                else "DISABLED"
            )
        )

        print(
            "Developer server : "
            f"{self.cfg.get('developer_host', '192.168.123.164')}:"
            f"{self.cfg.get('perception_data_port', 50020)}"
        )

        print(
            "=" * 108
        )

        self.wait_for_low_state()
        InitLowCmd(
            self.low_cmd
        )

        self.z1.connect()
        self._read_all_sensors_once()

        print(
            "[B2WZ1-HIER] Initializing perception..."
        )

        self.perception.initialize()
        self.perception.start()

        print(
            "[B2WZ1-HIER] Perception background system started."
        )

    def _send_one_exit_damping_command(
        self,
    ) -> None:
        try:
            self._read_all_sensors_once()
        except Exception:
            pass

        self._write_b2w_high_damping_cmd()

        try:
            self.send_b2w_cmd()
        except Exception:
            pass

        try:
            with self.z1._comm_lock:
                self.z1._send_arm_command_once(
                    q_cmd=
                        self.z1.q.copy(),

                    gripper_q_cmd=
                        float(
                            self.z1.gripper_q
                        ),

                    kp_cmd=
                        self.damping_kp_z1,

                    kd_cmd=
                        self.damping_kd_z1,

                    qd_cmd=
                        np.zeros(
                            6,
                            dtype=np.float32,
                        ),

                    tau_cmd=
                        np.zeros(
                            6,
                            dtype=np.float32,
                        ),
                )

        except Exception as exc:
            print(
                "[EXIT][WARN] Z1 damping command failed: "
                + repr(
                    exc
                )
            )

    def run(
        self,
    ) -> None:
        protection_reason: Optional[
            str
        ] = None

        perception_started = False

        try:
            self.setup()
            perception_started = True

            self.zero_torque_state()

            print(
                "[B2WZ1-HIER] Moving arm to DEFAULT with gripper OPEN..."
            )

            self.z1.move_to_pose_official(
                target_q=
                    self.default_arm_pos.copy(),

                # move_to_pose_official() is a startup/legacy position-servo
                # API and therefore expects the RAW SDK coordinate.
                target_gripper=
                    self._gripper_training_to_sdk(
                        self.gripper_open_pos
                    ),

                duration_s=
                    float(
                        self.cfg[
                            "arm_default_transition_s"
                        ]
                    ),

                step_callback=
                    None,
            )

            self.hold_arm_default_until_A()

            self.move_b2w_to_pose_policy(
                target_b2w_pos_policy=
                    self.squat_b2w_pos_policy,

                duration=
                    float(
                        self.cfg[
                            "squat_transition_s"
                        ]
                    ),

                hold_arm=
                    True,
            )

            self.move_b2w_to_pose_policy(
                target_b2w_pos_policy=
                    self.default_b2w_pos_policy,

                duration=
                    float(
                        self.cfg[
                            "default_transition_s"
                        ]
                    ),

                hold_arm=
                    True,
            )

            self.hold_default_until_perception_ready_and_A()

            # The robot is now physically at the DEFAULT pose and the user has
            # accepted policy start.  This is the only place where the known
            # absolute height reference is injected.
            self.anchor_vo_base_height_before_policy_start()

            print(
                "[B2WZ1-HIER] Initializing aligned LL/HL history..."
            )

            self.initialize_policy_state_and_history()

            print(
                "[B2WZ1-HIER] Priming first HL -> LL action before first block..."
            )

            self.prime_first_hl_and_ll()

            if self.visualizer is not None:
                print(
                    "[B2WZ1-HIER] Starting MuJoCo debug visualizer..."
                )

                self.visualizer.start()

            print(
                "[B2WZ1-HIER] Main loop started. "
                "SELECT = damping protection."
            )

            next_deadline = (
                time.perf_counter()
                + self.control_dt
            )

            while True:
                now = (
                    time.perf_counter()
                )

                sleep_s = (
                    next_deadline
                    - now
                )

                if sleep_s > 0.0:
                    time.sleep(
                        sleep_s
                    )

                loop_start = (
                    time.perf_counter()
                )

                ok, reason = (
                    self.step_after_prime()
                )

                if not ok:
                    protection_reason = (
                        reason
                        or "Unknown protection trigger"
                    )

                    break

                next_deadline += (
                    self.control_dt
                )

                # If the process falls badly behind, do not execute a burst of
                # back-to-back 50-Hz steps. Re-anchor the deadline.
                if (
                    next_deadline
                    < loop_start
                    - self.control_dt
                ):
                    print(
                        "[TIMING][WARN] control loop overrun; "
                        "re-anchoring 50-Hz deadline."
                    )

                    next_deadline = (
                        time.perf_counter()
                        + self.control_dt
                    )

        except KeyboardInterrupt:
            print(
                "[B2WZ1-HIER] KeyboardInterrupt received."
            )

        finally:
            if (
                protection_reason
                is not None
            ):
                self.enter_damping_protection_mode(
                    protection_reason
                )

            else:
                self._send_one_exit_damping_command()

            if perception_started:
                try:
                    self.perception.stop()
                except Exception as exc:
                    print(
                        "[EXIT][WARN] perception stop failed: "
                        + repr(
                            exc
                        )
                    )

            if self.visualizer is not None:
                try:
                    self.visualizer.stop()
                except Exception as exc:
                    print(
                        "[EXIT][WARN] visualizer stop failed: "
                        + repr(
                            exc
                        )
                    )

            print(
                "[B2WZ1-HIER] Exit."
            )


def main(
) -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "net",
        type=str,
        help=(
            "Unitree network interface, "
            "e.g. enxa0cec819e15f"
        ),
    )

    parser.add_argument(
        "config",
        type=str,
        help=(
            "Path to hierarchical sim2real YAML."
        ),
    )

    args = (
        parser.parse_args()
    )

    # Exactly once. The perception wrapper is configured with
    # initialize_channel_factory=False.
    ChannelFactoryInitialize(
        0,
        args.net,
    )

    controller = (
        B2WZ1HierarchicalRetrievalController(
            cfg_path=
                args.config,

            network_interface=
                args.net,
        )
    )

    controller.run()


if __name__ == "__main__":
    main()