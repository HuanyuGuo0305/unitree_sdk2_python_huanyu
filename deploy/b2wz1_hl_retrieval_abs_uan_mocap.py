#!/usr/bin/env python3
"""
B2WZ1 ABS high-level retrieval -> frozen WBC -> real robot, OptiTrack sensing.

This is a single self-contained controller. It replaces the four-file
inheritance chain it grew out of (sim2real -> mocap -> mocap_UAN -> abs), all
of which are gone.

Control stack
-------------
    HL ABS retrieval policy   10 Hz   base command + absolute EE keypoints
      -> frozen WBC           50 Hz   leg / arm / wheel joint targets
      -> hardware            500 Hz   zero-order hold on B2W + Z1 lowcmd

The 500 Hz hold is not cosmetic. The WBC was trained against an unsupervised
actuator net (UAN) fitted to data collected with a 500 Hz command stream and a
50 Hz target update, and deploy/b2wz1_wbc_uan.py -- the standalone WBC test
that runs smoothly on this robot -- drives the hardware exactly that way. This
script reproduces that loop shape: one timer at 500 Hz, the WBC on a
decimation, and a hardware send on every tick.

UAN is SIMULATION-ONLY
----------------------
    Training:  HL 10 Hz -> frozen WBC 50 Hz
                        -> simulated Z1: nominal PD + frozen UAN residual
                        -> PhysX
    Real:      HL 10 Hz -> frozen WBC 50 Hz -> q_des
                        -> REAL Z1 firmware position-PD -> hardware

The UAN exists to make the SIMULATED actuator behave like this real firmware
path. Running it again on the robot would double-count the correction. The
arm therefore receives the raw Unitree firmware gains the UAN was fitted
against, and the protocol scales (25.6 / 0.0128) are never applied in Python.
The script audits that alignment before it will move anything.

ABS high-level action
---------------------
The 9-D actor output maps affinely and directly onto configured ranges:

    action[0:3] -> base velocity command * base_cmd_scale
    action[3:6] -> absolute kp0 x / y / z
    action[6]   -> absolute EE yaw
    action[7]   -> absolute EE pitch, POLICY convention
    action[8]   -> binary gripper, > 0 closes

There is no measured-EE re-anchoring, no delta scale and no delta clip: the
affine map cannot leave its range by construction. This is the arithmetic of
the sim2sim reference,

    unitree_mujoco/simulate_python/deploy_mujoco/b2wz1_hl_retrieval_abs_uan.py

with one deliberate addition, kp0_z_cmd_min: the trained kp0_z range bottoms
out at ground level, so a saturated-low action[5] would drive the wrist into
the floor. The floor is applied to the DECODED command, after the affine map,
so the action -> range mapping the policy trained with is untouched.

Z1 end-effector pitch convention
--------------------------------
The Z1 gripper's local +X axis points from the wrist toward the fingertips.
Feeding policy pitch straight into standard XYZ-Euler geometry makes negative
pitch point +X UP, which is the wrong way round. So everywhere in this file:

    pitch_geom = ee_pitch_to_euler_sign * pitch_policy      (sign == -1)

Ranges, the neutral pose and every debug print use POLICY pitch; the
conversion happens only inside build_ee_keypoints_plb().

High-level observation
----------------------
The policy was retrained without grasp_confidence_proxy, so the high-level
frame is 55-D (165-D over 3 frames) and ends at previous_hl_action -- matching
the sim2sim reference and its hl_obs_dim_per_step: 55. No grasp proxy is
computed here: the actor's object vector is always the measured mocap object,
as in sim2sim, so a mocap dropout of the object (e.g. occluded inside the
gripper) invalidates the task state and enters damping protection. The
executed gripper command is action[8] alone.

Sensing
-------
Three Motive rigid bodies -- "B2" (the robot, whose pose IS base_link),
"cube6" (the object) and "retrieval" (the target) -- supply the object
position, the retrieval target and the base pose and height. No camera, no
AprilTag, no visual odometry. The mocap root frame must be base_link; the
script refuses to run on raw, unvouched-for Motive asset coordinates.

Faults
------
Any fault -- SELECT, the Z1 dropping to PASSIVE, invalid mocap, a WBC arm
target outside the live SDK range -- ends the policy and enters high-damping
protection, from which an operator A press folds the legs down.

Run from the repository root:

    python3 deploy/b2wz1_hl_retrieval_abs_uan_mocap.py \
        enxa0cec819e15f \
        deploy/configs/b2wz1_hl_retrieval_abs_uan_mocap.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import yaml

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_DEPLOY_DIR, ".."))
for _p in (PROJECT_ROOT, _DEPLOY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (  # noqa: E402
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

from utils.command_helper import InitLowCmd, create_zero_cmd  # noqa: E402
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
from utils.mocap_perception import MocapBodySelector, MocapPerceptionSystem  # noqa: E402
from utils.remote_controller import KeyMap, RemoteController  # noqa: E402
from utils.z1_helper import Z1ArmAdapter, compute_ee_current_kp_plb  # noqa: E402


# Z1 firmware gains the UAN was fitted against, and the training-space PD they
# must scale to. Hard-coded because a deployment that does not match these is
# not the controller the high-level policy was trained on top of.
Z1_KP_FIRMWARE = np.array([2.5, 4.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float64)
Z1_KD_FIRMWARE = np.array(
    [234.375, 312.5, 234.375, 234.375, 234.375, 234.375], dtype=np.float64
)
Z1_KP_TRAINING = np.array([64.0, 115.2, 64.0, 64.0, 64.0, 64.0], dtype=np.float64)
Z1_KD_TRAINING = np.array([3.0, 4.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float64)

# Low-level (WBC) observation, 80 per frame:
#   base_ang_vel(3) projected_gravity(3) base_cmd(3) ee_cmd_plb(9)
#   leg_pos_rel(12) arm_pos_rel(6) leg_vel(12) arm_vel(6) wheel_vel(4)
#   last_action(22)
LL_FEATURE_DIMS = (3, 3, 3, 9, 12, 6, 12, 6, 4, 22)

# High-level observation, 55 per frame:
#   base_ang_vel(3) projected_gravity(3) leg_pos_rel(12) arm_pos_rel(6)
#   gripper_pos_rel(1) arm_vel(6) object_center_base(3)
#   gripper_orientation_base(6) gripper_center_base(3) retrieval_target_base(3)
#   previous_hl_action(9)
#
# grasp_confidence_proxy is deliberately NOT here: the actor was retrained
# without it. Every other term and its order are unchanged.
HL_FEATURE_DIMS = (3, 3, 12, 6, 1, 6, 3, 6, 3, 3, 9)


# ======================================================================
# Small helpers
# ======================================================================


def sleep_until(deadline: float) -> float:
    """Sleep to `deadline`. Returns how late we already were, 0.0 if on time."""
    remaining = deadline - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)
        return 0.0
    return -remaining


def map_to_range(x: float, limits: np.ndarray) -> float:
    """Map a normalized action in [-1, 1] affinely onto [low, high]."""
    low, high = float(limits[0]), float(limits[1])
    return low + 0.5 * (float(x) + 1.0) * (high - low)


def map_from_range(value: float, limits: np.ndarray) -> float:
    """Inverse of map_to_range: encode an absolute value as a [-1, 1] action."""
    low, high = float(limits[0]), float(limits[1])
    return 2.0 * (float(value) - low) / (high - low) - 1.0


def as_range(cfg: Dict[str, Any], key: str) -> np.ndarray:
    limits = np.asarray(cfg[key], dtype=np.float64).reshape(-1)
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError(f"{key} must be two finite values, got {cfg[key]!r}.")
    if not limits[1] > limits[0]:
        raise ValueError(f"{key} must satisfy high > low, got {limits}.")
    return limits


def point3_or_none(value: Any) -> Optional[np.ndarray]:
    """A finite 3-vector, or None. Used to reject junk perception points."""
    if value is None:
        return None
    try:
        p = np.asarray(value, dtype=np.float32).reshape(3)
    except Exception:  # noqa: BLE001 - any malformed value is simply "no point"
        return None
    return p if np.isfinite(p).all() else None


def quat_from_euler_xyz_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Standard XYZ roll/pitch/yaw -> quaternion [w, x, y, z]."""
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )
    return quat_unique_wxyz(quat_normalize_wxyz(q))


def build_ee_keypoints_plb(
    kp0: np.ndarray,
    yaw: float,
    pitch_policy: float,
    roll: float,
    kp_dx: float,
    kp_dz: float,
    pitch_to_euler_sign: float,
) -> np.ndarray:
    """Build the 9-D [kp0, kp1, kp2] EE command in the PLB frame.

    `pitch_policy` is in POLICY convention; the sign flip to geometric
    XYZ-Euler pitch happens here and nowhere else. See the module docstring.
    """
    kp0 = np.asarray(kp0, dtype=np.float32).reshape(3)
    q_plb = quat_from_euler_xyz_wxyz(
        float(roll), float(pitch_to_euler_sign) * float(pitch_policy), float(yaw)
    )
    kp1 = kp0 + quat_apply_wxyz(q_plb, np.array([kp_dx, 0.0, 0.0], dtype=np.float32))
    kp2 = kp0 + quat_apply_wxyz(q_plb, np.array([0.0, 0.0, kp_dz], dtype=np.float32))
    return np.concatenate([kp0, kp1, kp2], dtype=np.float32)


class FeatureHistory:
    """Feature-major observation history.

    Both policies stack their history feature-major, NOT frame-major:

        [feature_0(t-H+1 .. t), feature_1(t-H+1 .. t), ...]

    Getting this backwards produces a plausible-looking observation of the
    right size that the policy cannot read, so the layout lives in one place.
    """

    def __init__(self, feature_dims: Sequence[int], length: int) -> None:
        self.dims = [int(d) for d in feature_dims]
        self.length = int(length)
        self.frame_dim = sum(self.dims)
        self.total_dim = self.frame_dim * self.length
        self._features = [deque(maxlen=self.length) for _ in self.dims]

    def _split(self, frame: np.ndarray) -> List[np.ndarray]:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.shape[0] != self.frame_dim:
            raise ValueError(
                f"Observation frame is {frame.shape[0]}-D, expected {self.frame_dim}."
            )
        out, start = [], 0
        for dim in self.dims:
            out.append(frame[start:start + dim].copy())
            start += dim
        return out

    def reset(self, frame: np.ndarray) -> None:
        """Fill the whole history with one frame, as training does on reset."""
        for feature_history, feature in zip(self._features, self._split(frame)):
            feature_history.clear()
            for _ in range(self.length):
                feature_history.append(feature.copy())

    def append(self, frame: np.ndarray) -> None:
        for feature_history, feature in zip(self._features, self._split(frame)):
            feature_history.append(feature)

    def flat(self) -> np.ndarray:
        obs = np.concatenate(
            [np.asarray(f, dtype=np.float32).reshape(-1) for f in self._features],
            dtype=np.float32,
        )
        if obs.shape != (self.total_dim,):
            raise RuntimeError(f"History flattened to {obs.shape}, expected {self.total_dim}.")
        return obs


# ======================================================================
# Controller
# ======================================================================


class B2WZ1AbsRetrievalMocapController:
    """ABS high-level retrieval policy over a frozen WBC, on the real B2W+Z1."""

    def __init__(self, cfg_path: str) -> None:
        with open(cfg_path, "r") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        self._configure_timing()
        self._configure_robot_layout()
        self._configure_wbc()
        self._configure_high_level()
        self._configure_protection()
        self._load_policies()

        self.z1 = Z1ArmAdapter(self.cfg, PROJECT_ROOT)
        self._audit_z1_runtime_alignment()

        self.perception = self._build_mocap_perception()
        self._connect_dds()
        self.visualizer = self._build_visualizer()

        self._reset_runtime_state()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _resolve_path(self, path_str: str) -> str:
        path_str = str(path_str)
        if os.path.isabs(path_str):
            return path_str
        return os.path.abspath(os.path.join(PROJECT_ROOT, path_str))

    def _configure_timing(self) -> None:
        cfg = self.cfg

        self.control_dt = float(cfg["control_dt"])
        self.ll_steps_per_hl_step = int(cfg["ll_steps_per_hl_step"])
        self.hl_control_dt = self.control_dt * self.ll_steps_per_hl_step

        if abs(1.0 / self.control_dt - 50.0) > 1e-5:
            raise ValueError(f"WBC must run at 50 Hz; control_dt={self.control_dt}.")
        if abs(1.0 / self.hl_control_dt - 10.0) > 1e-5:
            raise ValueError(
                f"High level must run at 10 Hz; got {1.0 / self.hl_control_dt:.3f} Hz."
            )

        # Hardware stream. The WBC targets are re-sent at this rate, exactly as
        # deploy/b2wz1_wbc_uan.py does and as the UAN data was collected.
        self.hardware_hz = float(cfg["hardware_command_hz"])
        self.hardware_dt = 1.0 / self.hardware_hz
        ratio = self.hardware_hz * self.control_dt
        if ratio < 1.0 - 1e-9 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                f"hardware_command_hz={self.hardware_hz:g} must be an integer "
                f"multiple of the {1.0 / self.control_dt:g} Hz WBC rate."
            )
        self.hardware_ticks_per_wbc_step = int(round(ratio))
        self.runtime_settle_s = float(cfg.get("z1_runtime_settle_s", 0.5))

        self.ll_history_length = int(cfg["ll_history_length"])
        self.ll_obs_dim = int(cfg["ll_obs_dim"])
        self.ll_action_dim = int(cfg["ll_action_dim"])
        self.hl_history_length = int(cfg["hl_history_length"])
        self.hl_obs_dim = int(cfg["hl_obs_dim"])
        self.hl_action_dim = int(cfg["hl_action_dim"])

        self.ll_history = FeatureHistory(LL_FEATURE_DIMS, self.ll_history_length)
        self.hl_history = FeatureHistory(HL_FEATURE_DIMS, self.hl_history_length)

        for name, history, per_step, total, action_dim in (
            ("Low", self.ll_history, cfg["ll_obs_dim_per_step"], self.ll_obs_dim, 22),
            ("High", self.hl_history, cfg["hl_obs_dim_per_step"], self.hl_obs_dim, 9),
        ):
            if history.frame_dim != int(per_step) or history.total_dim != int(total):
                raise ValueError(
                    f"{name}-level observation layout mismatch: this build has "
                    f"{history.frame_dim}/step and {history.total_dim} total, "
                    f"the YAML declares {per_step}/step and {total} total."
                )
        if self.ll_action_dim != 22 or self.hl_action_dim != 9:
            raise ValueError(
                f"Expected 22-D WBC / 9-D HL actions, got "
                f"{self.ll_action_dim}/{self.hl_action_dim}."
            )

    def _configure_robot_layout(self) -> None:
        """Policy joint order and its mapping onto hardware motor order.

        These are NOT the same order, and neither may change without
        retraining or a hardware remap.
        """
        cfg = self.cfg

        self.policy_leg_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.policy_wheel_joint_names = [
            "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
        ]
        self.hardware_joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
        ]
        policy_joint_names = self.policy_leg_joint_names + self.policy_wheel_joint_names

        self.num_b2w_dof = 16
        self.hardware_to_policy = [
            self.hardware_joint_names.index(n) for n in policy_joint_names
        ]
        self.policy_to_hardware = [
            policy_joint_names.index(n) for n in self.hardware_joint_names
        ]
        self.leg_policy_indices = list(range(12))
        self.leg_hardware_indices = [
            self.hardware_joint_names.index(n) for n in self.policy_leg_joint_names
        ]
        self.wheel_hardware_indices = [
            self.hardware_joint_names.index(n) for n in self.policy_wheel_joint_names
        ]
        # hardware wheel motor index -> its slot in the 4-D policy wheel command
        self.hardware_to_wheel_cmd = {
            self.hardware_joint_names.index(n): i
            for i, n in enumerate(self.policy_wheel_joint_names)
        }

        # Poses, policy order.
        self.default_b2w_pos_policy = np.asarray(
            cfg["default_b2w_pos_policy"], dtype=np.float32
        ).reshape(16)
        self.squat_b2w_pos_policy = np.asarray(
            cfg["squat_b2w_pos_policy"], dtype=np.float32
        ).reshape(16)
        self.default_leg_pos_policy = self.default_b2w_pos_policy[:12].copy()
        self.default_arm_pos = np.asarray(
            cfg["default_arm_pos"], dtype=np.float32
        ).reshape(6)
        self.default_joint_pos_policy = np.asarray(
            cfg["default_joint_pos_policy"], dtype=np.float32
        ).reshape(18)

        # Gripper positions are TRAINING coordinates: closed = 0, open = -pi/2.
        self.default_gripper_pos = float(cfg["default_gripper_pos"])
        self.gripper_open_pos = float(cfg["gripper_open_pos"])
        self.gripper_close_pos = float(cfg["gripper_close_pos"])

        # B2W gains, policy order -> hardware order.
        def gains_hw(key: str) -> np.ndarray:
            return np.asarray(cfg[key], dtype=np.float32).reshape(16)[
                self.policy_to_hardware
            ]

        self.kps_rl_hw = gains_hw("kps_rl")
        self.kds_rl_hw = gains_hw("kds_rl")
        self.kps_pd_hw = gains_hw("kps_pd")
        self.kds_pd_hw = gains_hw("kds_pd")

    def _configure_wbc(self) -> None:
        """Frozen WBC action decoding, identical to deploy/b2wz1_wbc_uan.py."""
        cfg = self.cfg
        self.leg_action_scale = float(cfg["leg_action_scale"])
        self.arm_action_scale = np.asarray(
            cfg["arm_action_scale"], dtype=np.float32
        ).reshape(6)
        self.wheel_action_scale = float(cfg["wheel_action_scale"])

        # Live SDK hard range. A target outside it is a fault, not a clamp:
        # the arm would jam against its stop at clipped torque.
        self.arm_q_lower = np.asarray(cfg["arm_q_lower"], dtype=np.float32).reshape(6)
        self.arm_q_upper = np.asarray(cfg["arm_q_upper"], dtype=np.float32).reshape(6)

    def _configure_high_level(self) -> None:
        """ABS action ranges, EE geometry and the neutral reset command."""
        cfg = self.cfg

        self.base_cmd_scale = np.asarray(
            cfg["base_cmd_scale"], dtype=np.float32
        ).reshape(3)

        self.kp0_x_range = as_range(cfg, "kp0_x_range")
        self.kp0_y_range = as_range(cfg, "kp0_y_range")
        self.kp0_z_range = as_range(cfg, "kp0_z_range")
        self.ee_yaw_range = as_range(cfg, "ee_yaw_range")
        self.ee_pitch_range = as_range(cfg, "ee_pitch_range")

        self.kp0_z_cmd_min = float(cfg.get("kp0_z_cmd_min", self.kp0_z_range[0]))
        if not self.kp0_z_range[0] <= self.kp0_z_cmd_min <= self.kp0_z_range[1]:
            raise ValueError(
                f"kp0_z_cmd_min={self.kp0_z_cmd_min} must lie inside kp0_z_range "
                f"{self.kp0_z_range.tolist()}."
            )

        self.ee_pitch_to_euler_sign = float(cfg["ee_pitch_to_euler_sign"])
        if abs(self.ee_pitch_to_euler_sign + 1.0) > 1e-12:
            raise ValueError(
                "The Z1 requires ee_pitch_to_euler_sign: -1.0 -- negative policy "
                "pitch must tilt the gripper's local +X (toward the fingertips) "
                f"DOWN. Got {self.ee_pitch_to_euler_sign}."
            )

        self.fixed_ee_roll = float(cfg["fixed_ee_roll"])
        self.ee_kp_dx = float(cfg["ee_kp_dx"])
        self.ee_kp_dz = float(cfg["ee_kp_dz"])
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.gripper_binary_threshold = float(cfg["gripper_binary_threshold"])
        # Gripper-center offset in the Z1 policy EE (gripperStator) frame; feeds
        # the gripper_center_base actor observation.
        self.gripper_center_offset_local = np.asarray(
            cfg["gripper_center_offset_local"], dtype=np.float32
        ).reshape(3)

        # Neutral EE command, used at reset.
        self.neutral_kp0 = np.asarray(cfg["neutral_kp0"], dtype=np.float32).reshape(3)
        self.neutral_ee_yaw = float(cfg["neutral_ee_yaw"])
        self.neutral_ee_pitch = float(cfg["neutral_ee_pitch"])

        for name, value, limits in (
            ("neutral_kp0.x", float(self.neutral_kp0[0]), self.kp0_x_range),
            ("neutral_kp0.y", float(self.neutral_kp0[1]), self.kp0_y_range),
            ("neutral_kp0.z", float(self.neutral_kp0[2]), self.kp0_z_range),
            ("neutral_ee_yaw", self.neutral_ee_yaw, self.ee_yaw_range),
            ("neutral_ee_pitch", self.neutral_ee_pitch, self.ee_pitch_range),
        ):
            if not limits[0] - 1e-12 <= value <= limits[1] + 1e-12:
                raise ValueError(f"{name}={value} lies outside {limits.tolist()}.")

        # ABS reset semantics: previous_hl_action[3:8] encodes the neutral EE
        # command in normalized coordinates. Zeros would NOT be neutral here --
        # in ABS coordinates zero is a real command at the midpoint of every
        # range.
        self.neutral_arm_action = np.array(
            [
                map_from_range(self.neutral_kp0[0], self.kp0_x_range),
                map_from_range(self.neutral_kp0[1], self.kp0_y_range),
                map_from_range(self.neutral_kp0[2], self.kp0_z_range),
                map_from_range(self.neutral_ee_yaw, self.ee_yaw_range),
                map_from_range(self.neutral_ee_pitch, self.ee_pitch_range),
            ],
            dtype=np.float32,
        )

        self.abs_decode_debug = bool(cfg.get("abs_decode_debug", True))
        self.debug_print_period_steps = int(cfg.get("debug_print_period_steps", 50))
        self.debug_hl_obs_enabled = bool(cfg.get("debug_hl_obs_enabled", False))
        self.debug_ll_obs_enabled = bool(cfg.get("debug_ll_obs_enabled", False))
        self.debug_obs_print_max = int(cfg.get("debug_obs_print_max", 5))

    def _configure_protection(self) -> None:
        cfg = self.cfg

        self.damping_kd_b2w = float(cfg.get("damping_kd_b2w", 150.0))
        self.damping_kd_wheel = float(cfg.get("damping_kd_wheel", 10.0))
        self.damping_kp_z1 = np.asarray(
            cfg.get("damping_kp_z1", [0.0] * 6), dtype=np.float32
        ).reshape(6)
        self.damping_kd_z1 = np.asarray(
            cfg.get("damping_kd_z1", [2000.0] * 6), dtype=np.float32
        ).reshape(6)
        self.damping_print_period = int(cfg.get("damping_print_period", 100))

        # Manual leg recovery is deliberately NOT automatic: a Z1 PASSIVE event
        # may be a real fault, so moving anything needs an explicit A press.
        self.leg_recovery_enabled = bool(cfg.get("damping_leg_recovery_enabled", True))
        self.leg_recovery_duration_s = float(cfg.get("damping_recover_leg_duration_s", 4.0))
        self.leg_recovery_tolerance_rad = float(
            cfg.get("damping_recover_leg_tolerance_rad", 0.15)
        )
        self.leg_recovery_target = np.asarray(
            cfg.get("damping_recover_leg_target", self.squat_b2w_pos_policy[:12].tolist()),
            dtype=np.float32,
        ).reshape(12)
        if self.leg_recovery_duration_s <= 0.0 or self.leg_recovery_tolerance_rad <= 0.0:
            raise ValueError(
                "damping_recover_leg_duration_s and _tolerance_rad must be > 0."
            )

        self.perception_status_print_s = float(cfg.get("perception_status_print_s", 1.0))

    def _load_policies(self) -> None:
        self.low_policy_path = self._resolve_path(self.cfg["low_policy_path"])
        self.high_policy_path = self._resolve_path(self.cfg["high_policy_path"])

        for path in (self.low_policy_path, self.high_policy_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Policy not found: {path}")

        self.low_session = ort.InferenceSession(
            self.low_policy_path, providers=["CPUExecutionProvider"]
        )
        self.high_session = ort.InferenceSession(
            self.high_policy_path, providers=["CPUExecutionProvider"]
        )
        self.low_input_name = self.low_session.get_inputs()[0].name
        self.low_output_name = self.low_session.get_outputs()[0].name
        self.high_input_name = self.high_session.get_inputs()[0].name
        self.high_output_name = self.high_session.get_outputs()[0].name

        for label, session, obs_dim, action_dim in (
            ("low", self.low_session, self.ll_obs_dim, self.ll_action_dim),
            ("high", self.high_session, self.hl_obs_dim, self.hl_action_dim),
        ):
            in_dim = session.get_inputs()[0].shape[-1]
            out_dim = session.get_outputs()[0].shape[-1]
            if isinstance(in_dim, int) and in_dim != obs_dim:
                raise RuntimeError(f"{label} policy takes {in_dim}-D obs, expected {obs_dim}.")
            if isinstance(out_dim, int) and out_dim != action_dim:
                raise RuntimeError(
                    f"{label} policy emits {out_dim}-D actions, expected {action_dim}."
                )

    def _audit_z1_runtime_alignment(self) -> None:
        """Refuse to run unless the real arm reproduces the training actuator.

        UAN is simulation-only, so the alignment the high-level policy depends
        on is entirely in these gains: the Unitree protocol scales the raw
        firmware values by 25.6 / 0.0128, and the product must be the nominal
        PD the WBC was trained with.
        """
        cfg = self.cfg

        forbidden = [
            key
            for key in (
                "uan_model_path", "uan_history_length", "uan_action_scale",
                "uan_nominal_torque_limits", "uan_final_torque_limits",
                "enable_arm_target_rate_limit", "arm_target_rate_limit",
            )
            if key in cfg
        ]
        if forbidden:
            raise ValueError(
                "The real robot runs neither the UAN nor an arm target rate "
                f"limiter. Remove these simulation-only keys: {forbidden}"
            )

        if str(cfg.get("z1_arm_runtime_mode", "")).strip().lower() != "position_pd":
            raise ValueError(
                'z1_arm_runtime_mode must be "position_pd": the real arm closes '
                "its position loop in firmware, and no external residual torque "
                "may be sent."
            )
        gripper_mode = str(cfg.get("z1_gripper_runtime_mode", "")).strip().lower()
        if gripper_mode not in ("dcmotor", "position_pd"):
            raise ValueError(
                "z1_gripper_runtime_mode must be 'dcmotor' (the trained actuator) "
                f"or 'position_pd' (firmware gripper loop); got {gripper_mode!r}."
            )

        kp_fw = np.asarray(cfg["arm_kps_runtime"], dtype=np.float64).reshape(6)
        kd_fw = np.asarray(cfg["arm_kds_runtime"], dtype=np.float64).reshape(6)
        kp_train = np.asarray(cfg["arm_kp_training"], dtype=np.float64).reshape(6)
        kd_train = np.asarray(cfg["arm_kd_training"], dtype=np.float64).reshape(6)
        kp_scale = float(cfg.get("z1_kp_protocol_scale", 25.6))
        kd_scale = float(cfg.get("z1_kd_protocol_scale", 0.0128))

        for label, actual, expected, tol in (
            ("firmware Kp", kp_fw, Z1_KP_FIRMWARE, 1e-9),
            ("firmware Kd", kd_fw, Z1_KD_FIRMWARE, 1e-9),
            ("training Kp", kp_train, Z1_KP_TRAINING, 1e-6),
            ("training Kd", kd_train, Z1_KD_TRAINING, 1e-6),
            ("scaled Kp", kp_fw * kp_scale, kp_train, 1e-6),
            ("scaled Kd", kd_fw * kd_scale, kd_train, 1e-6),
        ):
            if not np.allclose(actual, expected, atol=tol, rtol=0.0):
                raise ValueError(
                    f"Z1 gain alignment failed on {label}:\n"
                    f"  got      {np.round(actual, 6)}\n"
                    f"  expected {np.round(expected, 6)}"
                )

        # The adapter is what actually sends them; audit it, not just the YAML.
        for label, adapter_value, yaml_value in (
            ("Kp", self.z1.arm_kps_runtime, kp_fw),
            ("Kd", self.z1.arm_kds_runtime, kd_fw),
        ):
            adapter_value = np.asarray(adapter_value, dtype=np.float64).reshape(6)
            if not np.allclose(adapter_value, yaml_value, atol=1e-9, rtol=0.0):
                raise RuntimeError(
                    f"Z1 adapter runtime {label} {adapter_value} differs from the "
                    f"YAML {yaml_value}."
                )

        self.z1_kp_effective = kp_fw * kp_scale
        self.z1_kd_effective = kd_fw * kd_scale

    def _build_mocap_perception(self) -> MocapPerceptionSystem:
        """Build the OptiTrack sensing stack and prove its root frame.

        The mocap root frame must BE base_link. Either the asset's pivot and
        axes were aligned with base_link inside Motive, or the offset was
        measured with deploy/b2w_mocap_root_calibration.py. Raw Motive asset
        coordinates standing in for base_link are refused: every derived
        quantity -- base height, the PLB frame, both target positions -- would
        be wrong by whatever that asset's pivot happens to be.
        """
        cfg = self.cfg

        def selector(key: str, default_name: str) -> MocapBodySelector:
            return MocapBodySelector(
                label=key,
                name=cfg.get(f"mocap_{key}_name", default_name),
                rb_id=cfg.get(f"mocap_{key}_id"),
                offset_local=cfg.get(f"mocap_{key}_offset_local", [0.0, 0.0, 0.0]),
                max_age_s=float(cfg.get(f"mocap_{key}_max_age_s", 0.25)),
            )

        offset_path = cfg.get("mocap_root_offset_path")
        inline_offset = cfg.get("mocap_root_offset")
        motive_aligned = bool(cfg.get("mocap_root_frame_is_base_link", False))

        if offset_path:
            root_offset = self._resolve_path(offset_path)
            if not os.path.isfile(root_offset):
                raise FileNotFoundError(
                    f"mocap_root_offset_path does not exist: {root_offset}. Run "
                    "deploy/b2w_mocap_root_calibration.py, or set "
                    "mocap_root_frame_is_base_link: true if the asset frame was "
                    "already aligned with base_link in Motive."
                )
            self.mocap_root_offset_source = root_offset
        elif inline_offset:
            root_offset = inline_offset
            self.mocap_root_offset_source = "<inline>"
        elif motive_aligned:
            root_offset = None
            self.mocap_root_offset_source = (
                "<identity: asset frame aligned with base_link in Motive>"
            )
        else:
            raise ValueError(
                "The mocap root frame must be base_link. Either set "
                "mocap_root_frame_is_base_link: true (the asset was aligned in "
                "Motive) or point mocap_root_offset_path at the YAML produced by "
                "deploy/b2w_mocap_root_calibration.py."
            )

        perception = MocapPerceptionSystem(
            root_frame_is_base_link=motive_aligned,
            body=selector("body", "B2"),
            object_body=selector("object", "cube6"),
            retrieval_body=selector("retrieval", "retrieval"),
            root_offset=root_offset,
            ground_z=self.ground_z,
            local_ip=str(cfg.get("mocap_local_ip", "")),
            server_ip=cfg.get("mocap_server_ip") or None,
            multicast_group=str(cfg.get("mocap_multicast_group", "239.255.42.99")),
            data_port=int(cfg.get("mocap_data_port", 1511)),
            command_port=int(cfg.get("mocap_command_port", 1510)),
            join_multicast=bool(cfg.get("mocap_join_multicast", True)),
            up_axis=str(cfg.get("mocap_up_axis", "auto")),
            startup_timeout_s=float(cfg.get("mocap_timeout_s", 10.0)),
        )

        if not bool(getattr(perception, "root_offset_calibrated", False)):
            raise ValueError(
                "MocapPerceptionSystem reports an untrusted root offset. "
                "Refusing to run the policy with the Motive asset frame "
                "pretending to be base_link."
            )
        return perception

    def _connect_dds(self) -> None:
        """B2W low-level DDS. ChannelFactoryInitialize() is done once in main()."""
        self.remote_controller = RemoteController()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.crc = CRC()

        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

    def _build_visualizer(self):
        """Optional MuJoCo debug view. Renders telemetry; steps no physics."""
        if not bool(self.cfg.get("visualizer_enabled", False)):
            return None

        from utils.mj_visualizer import MujocoDebugVisualizer  # noqa: PLC0415

        cfg = self.cfg
        visualizer = MujocoDebugVisualizer(
            xml_path=self._resolve_path(cfg["visualizer_xml_path"]),
            update_hz=float(cfg.get("visualizer_update_hz", 30.0)),
            object_marker_radius=float(cfg.get("visualizer_object_marker_radius", 0.03)),
            retrieval_marker_radius=float(cfg.get("visualizer_retrieval_marker_radius", 0.05)),
            ee_target_sphere_radius=float(cfg.get("visualizer_ee_target_sphere_radius", 0.03)),
            ee_target_axis_len=float(cfg.get("visualizer_ee_target_axis_len", 0.20)),
            ee_target_axis_radius=float(cfg.get("visualizer_ee_target_axis_radius", 0.01)),
            show_gripper_center=bool(cfg.get("visualizer_show_gripper_center", True)),
            gripper_center_marker_radius=float(
                cfg.get("visualizer_gripper_center_marker_radius", 0.018)
            ),
            show_floor=bool(cfg.get("visualizer_show_floor", True)),
            floor_half_extent=float(cfg.get("visualizer_floor_half_extent", 3.0)),
            show_light=bool(cfg.get("visualizer_show_light", True)),
            ground_z=self.ground_z,
        )

        # Mocap measures the object, the target and the robot absolutely, so
        # the scene can be drawn where it physically is rather than
        # robot-centric.
        visualizer.configure_world_view(
            show_world_origin=bool(cfg.get("visualizer_show_world_origin", True)),
            camera_follow=bool(cfg.get("visualizer_camera_follow", True)),
            world_origin_axis_len=float(cfg.get("visualizer_world_origin_axis_len", 0.4)),
            camera_follow_smoothing=float(
                cfg.get("visualizer_camera_follow_smoothing", 0.15)
            ),
            show_all_markers=bool(cfg.get("visualizer_show_all_markers", True)),
            marker_radius=float(cfg.get("visualizer_marker_radius", 0.012)),
            show_root_frame=bool(cfg.get("visualizer_show_root_frame", True)),
            root_frame_axis_len=float(cfg.get("visualizer_root_frame_axis_len", 0.25)),
            root_frame_axis_radius=float(
                cfg.get("visualizer_root_frame_axis_radius", 0.008)
            ),
            object_cube_size=float(cfg.get("visualizer_object_cube_size", 0.0)),
        )

        self.viz_show_pivots = bool(cfg.get("visualizer_show_mocap_pivots", True))
        self.viz_pivot_radius = float(cfg.get("visualizer_mocap_pivot_radius", 0.022))
        self.monitor_enabled = bool(cfg.get("visualizer_monitor_enabled", True))
        self.monitor_hz = float(cfg.get("visualizer_monitor_hz", 30.0))
        self.monitor_takeover_s = float(cfg.get("visualizer_monitor_takeover_s", 0.25))
        return visualizer

    def _reset_runtime_state(self) -> None:
        """All mutable control state, in one place."""
        # Measured B2W state; joint arrays are in POLICY order (12 leg + 4 wheel).
        self.base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.base_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.b2w_joint_pos = np.zeros(16, dtype=np.float32)
        self.b2w_joint_vel = np.zeros(16, dtype=np.float32)

        # Base height above ground. Mocap measures it absolutely, so there is
        # nothing to anchor -- but there is also no sane value before the first
        # tracked frame, hence None until then.
        self.base_height: Optional[float] = None

        # High-level outputs.
        self.base_command = np.zeros(3, dtype=np.float32)
        self.ee_cmd_plb_current = np.zeros(9, dtype=np.float32)
        self.current_hl_action = np.zeros(self.hl_action_dim, dtype=np.float32)
        self.raw_gripper_action = -1.0
        self.executed_gripper_cmd_norm = -1.0
        self.gripper_target = float(self.gripper_open_pos)

        # WBC outputs.
        self.last_ll_action = np.zeros(self.ll_action_dim, dtype=np.float32)
        self.leg_target = self.default_leg_pos_policy.copy()
        self.arm_target = self.default_arm_pos.copy()
        self.wheel_cmd = np.zeros(4, dtype=np.float32)

        # Scheduling / diagnostics.
        self.ll_tick = 0
        self.hl_tick = 0
        self.last_perception_snapshot: Optional[Dict[str, Any]] = None
        self.debug_hl_obs_count = 0
        self.debug_ll_obs_count = 0
        self._abs_decode_debug_printed = False

        # Visualizer monitor.
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._monitor_warned = False
        self._last_control_push = 0.0
        self._viz_no_marker_warned = False
        self._viz_marker_count_announced = False

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def _low_state_handler(self, msg: LowStateGo) -> None:
        self.low_state = msg
        self.remote_controller.set(msg.wireless_remote)

    def wait_for_low_state(self) -> None:
        print("[B2WZ1-ABS] Waiting for the first B2W lowstate...")
        while getattr(self.low_state, "tick", 0) == 0:
            time.sleep(self.control_dt)
        print(f"[B2WZ1-ABS] First lowstate received: tick={self.low_state.tick}")

    def read_b2w_state(self) -> None:
        """Refresh IMU and joint state from the latest rt/lowstate.

        base_command is NOT touched here: it belongs to the high-level policy,
        never to the joystick.
        """
        raw_quat = self.low_state.imu_state.quaternion
        quat = quat_unique_wxyz(np.asarray(raw_quat, dtype=np.float32).reshape(4))
        self.base_quat_wxyz[:] = quat / max(float(np.linalg.norm(quat)), 1e-8)

        self.base_ang_vel_b[:] = np.asarray(
            self.low_state.imu_state.gyroscope, dtype=np.float32
        ).reshape(3)
        self.projected_gravity_b[:] = quat_rotate_inverse_numpy(
            self.base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )

        motors = self.low_state.motor_state
        for policy_idx in range(self.num_b2w_dof):
            motor = motors[self.hardware_to_policy[policy_idx]]
            self.b2w_joint_pos[policy_idx] = motor.q
            self.b2w_joint_vel[policy_idx] = motor.dq

    def read_robot_state(self) -> None:
        self.read_b2w_state()
        self.z1.read_state()

    def read_perception(self) -> Dict[str, Any]:
        """Nonblocking mocap read; also refreshes the measured base height."""
        snap = self.perception.get_latest_snapshot()
        self.last_perception_snapshot = snap

        height_state = snap.get("base_height") or {}
        if height_state.get("valid"):
            height_m = float(height_state["height_m"])
            if np.isfinite(height_m):
                self.base_height = height_m
        return snap

    @staticmethod
    def _channel_point(snap: Dict[str, Any], channel: str, key: str) -> Optional[np.ndarray]:
        state = snap.get(channel) or {}
        if not state.get("valid"):
            return None
        return point3_or_none(state.get(key))

    def gripper_q_training(self) -> float:
        """Measured gripper position in TRAINING coordinates (q_sdk - offset)."""
        return float(self.z1.get_gripper_q_training())

    def gripper_training_to_sdk(self, q_training: float) -> float:
        """Raw SDK gripper coordinate, for the startup position-servo paths."""
        return float(self.z1.gripper_training_to_sdk(float(q_training)))

    # ------------------------------------------------------------------
    # End-effector geometry, in the projected-base (PLB) frame
    #
    #   PLB origin      = [base_x, base_y, ground_z]
    #   PLB orientation = yaw-only(base_quat)
    #
    # base_x / base_y cancel because the EE is computed relative to the base,
    # but base_height must be accurate -- which is why it comes from mocap.
    # ------------------------------------------------------------------

    def _require_base_height(self) -> float:
        if self.base_height is None:
            raise RuntimeError(
                "Base height is unknown: mocap has not yet delivered a tracked "
                "body frame."
            )
        return float(self.base_height)

    def compute_measured_ee_keypoints_plb(self) -> np.ndarray:
        return compute_ee_current_kp_plb(
            base_quat_wxyz=self.base_quat_wxyz,
            base_height=self._require_base_height(),
            ground_z=self.ground_z,
            z1_adapter=self.z1,
            kp_dx=self.ee_kp_dx,
            kp_dz=self.ee_kp_dz,
        )

    def compute_measured_ee_pose_plb(self) -> Tuple[np.ndarray, float, float]:
        """Measured EE pose in PLB: (position, yaw, POLICY pitch)."""
        base_q = quat_unique_wxyz(quat_normalize_wxyz(self.base_quat_wxyz))
        ee_pos_b, ee_rot_b = self.z1.compute_policy_ee_pose_in_base()
        ee_quat_b = quat_unique_wxyz(quat_normalize_wxyz(quat_from_rotmat_wxyz(ee_rot_b)))

        _, _, base_yaw = euler_xyz_from_quat_wxyz(base_q)
        plb_q_w = quat_unique_wxyz(quat_normalize_wxyz(quat_from_yaw_wxyz(base_yaw)))

        base_pos_w = np.array([0.0, 0.0, self._require_base_height()], dtype=np.float32)
        plb_pos_w = np.array([0.0, 0.0, self.ground_z], dtype=np.float32)

        ee_pos_w = base_pos_w + quat_apply_wxyz(base_q, ee_pos_b)
        ee_quat_w = quat_unique_wxyz(quat_normalize_wxyz(quat_mul_wxyz(base_q, ee_quat_b)))

        ee_pos_plb = quat_apply_inverse_wxyz(plb_q_w, ee_pos_w - plb_pos_w)
        ee_quat_plb = quat_unique_wxyz(
            quat_normalize_wxyz(
                quat_mul_wxyz(quat_conjugate_wxyz(plb_q_w), ee_quat_w)
            )
        )
        _, geometric_pitch, yaw = euler_xyz_from_quat_wxyz(ee_quat_plb)

        # Back to POLICY pitch. The sign is +/-1, so it is its own inverse.
        policy_pitch = self.ee_pitch_to_euler_sign * float(geometric_pitch)
        return np.asarray(ee_pos_plb, dtype=np.float32).reshape(3), float(yaw), policy_pitch

    def get_gripper_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """(gripper_orientation_base 6-D, gripper_center_base 3-D).

        Orientation is the gripper's local +X and +Y axes expressed in the base
        frame. The Z1 policy EE frame is the sim2sim gripperStator frame.
        """
        stator_pos_b, stator_rot_b = self.z1.compute_policy_ee_pose_in_base()
        stator_pos_b = np.asarray(stator_pos_b, dtype=np.float32).reshape(3)
        stator_rot_b = np.asarray(stator_rot_b, dtype=np.float32).reshape(3, 3)

        gripper_center_b = (
            stator_pos_b + stator_rot_b @ self.gripper_center_offset_local
        ).astype(np.float32)
        orientation_b = np.concatenate(
            [stator_rot_b[:, 0], stator_rot_b[:, 1]], dtype=np.float32
        )
        return orientation_b, gripper_center_b

    def build_neutral_ee_command(self) -> np.ndarray:
        return build_ee_keypoints_plb(
            kp0=self.neutral_kp0,
            yaw=self.neutral_ee_yaw,
            pitch_policy=self.neutral_ee_pitch,
            roll=self.fixed_ee_roll,
            kp_dx=self.ee_kp_dx,
            kp_dz=self.ee_kp_dz,
            pitch_to_euler_sign=self.ee_pitch_to_euler_sign,
        )

    def resolve_task_state(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        """What the actor is allowed to see this step. Never a stale stand-in.

        The object vector is always the measured mocap object, as in sim2sim.
        """
        gripper_orientation_b, gripper_center_b = self.get_gripper_geometry()

        object_pos_b = self._channel_point(snap, "object", "position_base")
        object_source = (snap.get("object") or {}).get("source") if object_pos_b is not None else None

        retrieval_pos_b = self._channel_point(snap, "retrieval", "retrieval_target_base")
        retrieval_source = (
            (snap.get("retrieval") or {}).get("source")
            if retrieval_pos_b is not None
            else None
        )

        reasons = []
        if object_pos_b is None:
            reasons.append("OBJECT_INVALID:" + str((snap.get("object") or {}).get("reason")))
        if retrieval_pos_b is None:
            reasons.append(
                "RETRIEVAL_INVALID:" + str((snap.get("retrieval") or {}).get("reason"))
            )
        if self.base_height is None:
            reasons.append("BASE_HEIGHT_INVALID:" + str((snap.get("base_height") or {}).get("reason")))

        return {
            "valid": not reasons,
            "reason": "VALID" if not reasons else "|".join(reasons),
            "object_position_base": object_pos_b,
            "object_source": object_source,
            "retrieval_target_base": retrieval_pos_b,
            "retrieval_source": retrieval_source,
            "gripper_orientation_base": gripper_orientation_b,
            "gripper_center_pos_base": gripper_center_b,
        }

    # ------------------------------------------------------------------
    # High-level policy, 10 Hz
    # ------------------------------------------------------------------

    def build_previous_hl_action(self) -> np.ndarray:
        """What the actor observes as its own last action.

        Base and gripper entries are EXECUTION-aware -- what was actually
        commanded, renormalized -- while the five arm entries are the raw ABS
        action, exactly as in training.
        """
        denom = np.maximum(np.abs(self.base_cmd_scale), 1e-6)
        effective_base_action = np.clip(self.base_command / denom, -1.0, 1.0)
        return np.concatenate(
            [
                effective_base_action,
                self.current_hl_action[3:8],
                np.array([self.executed_gripper_cmd_norm], dtype=np.float32),
            ],
            dtype=np.float32,
        )

    def build_hl_obs_frame(self, task: Dict[str, Any]) -> np.ndarray:
        if not task["valid"]:
            raise RuntimeError(
                "Refusing to build a high-level observation from invalid task "
                "state: " + str(task["reason"])
            )

        leg_pos_rel = self.b2w_joint_pos[:12] - self.default_joint_pos_policy[:12]
        arm_pos_rel = self.z1.q - self.default_joint_pos_policy[12:18]
        gripper_pos_rel = np.array(
            [self.gripper_q_training() - self.default_gripper_pos], dtype=np.float32
        )

        obs = np.concatenate(
            [
                self.base_ang_vel_b,                  # 3
                self.projected_gravity_b,             # 3
                leg_pos_rel.astype(np.float32),       # 12
                arm_pos_rel.astype(np.float32),       # 6
                gripper_pos_rel,                      # 1
                np.asarray(self.z1.qd, dtype=np.float32).reshape(6),   # 6
                task["object_position_base"],         # 3
                task["gripper_orientation_base"],     # 6
                task["gripper_center_pos_base"],      # 3
                task["retrieval_target_base"],        # 3
                self.build_previous_hl_action(),      # 9
            ],
            dtype=np.float32,
        )
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite value in the high-level observation.")
        return obs

    def run_high_level_policy(self, task: Dict[str, Any], *, append_frame: bool) -> None:
        """Infer and decode one high-level action.

        append_frame=False reproduces the sim2sim reset schedule: the history
        was just filled with the first valid frame, so the first inference runs
        on it directly instead of pushing a duplicate frame first.
        """
        if append_frame:
            self.hl_history.append(self.build_hl_obs_frame(task))

        obs = self.hl_history.flat()
        if self.debug_hl_obs_enabled and self.debug_hl_obs_count < self.debug_obs_print_max:
            self.debug_hl_obs_count += 1
            print(
                f"[HL-OBS] tick={self.hl_tick} shape={obs.shape} "
                f"min={obs.min():+.3f} max={obs.max():+.3f}"
            )

        action = self.high_session.run(
            [self.high_output_name], {self.high_input_name: obs[None, :]}
        )[0][0].astype(np.float32)

        self.decode_hl_action(action)
        self.hl_tick += 1

    def decode_hl_action(self, action: np.ndarray) -> None:
        """Turn the 9-D ABS action into the commands the WBC consumes.

        Same arithmetic as the sim2sim reference: every normalized component
        maps affinely onto its configured range, with no measured-EE
        re-anchoring and no delta semantics anywhere.
        """
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(self.hl_action_dim), -1.0, 1.0
        )
        self.current_hl_action[:] = action

        self.base_command[:] = (action[0:3] * self.base_cmd_scale).astype(np.float32)

        kp0_cmd = np.array(
            [
                map_to_range(action[3], self.kp0_x_range),
                map_to_range(action[4], self.kp0_y_range),
                # Floor applied AFTER the affine map: the action -> range
                # mapping the policy trained with stays untouched.
                max(map_to_range(action[5], self.kp0_z_range), self.kp0_z_cmd_min),
            ],
            dtype=np.float32,
        )
        yaw_cmd = map_to_range(action[6], self.ee_yaw_range)
        pitch_cmd = map_to_range(action[7], self.ee_pitch_range)

        self.ee_cmd_plb_current[:] = build_ee_keypoints_plb(
            kp0=kp0_cmd,
            yaw=yaw_cmd,
            pitch_policy=pitch_cmd,
            roll=self.fixed_ee_roll,
            kp_dx=self.ee_kp_dx,
            kp_dz=self.ee_kp_dz,
            pitch_to_euler_sign=self.ee_pitch_to_euler_sign,
        )

        # Binary gripper. Strictly positive closes; zero stays open.
        self.raw_gripper_action = float(action[8])
        close = self.raw_gripper_action > self.gripper_binary_threshold
        self.executed_gripper_cmd_norm = 1.0 if close else -1.0
        self.gripper_target = self.gripper_close_pos if close else self.gripper_open_pos

        if self.abs_decode_debug and not self._abs_decode_debug_printed:
            self._abs_decode_debug_printed = True
            print(
                "[ABS-DECODE] first command | "
                f"kp0={np.array2string(kp0_cmd, precision=4)} | "
                f"yaw={yaw_cmd:+.4f} | pitch_policy={pitch_cmd:+.4f} -> "
                f"pitch_geom={self.ee_pitch_to_euler_sign * pitch_cmd:+.4f}"
            )

    # ------------------------------------------------------------------
    # Frozen WBC, 50 Hz
    # ------------------------------------------------------------------

    def build_ll_obs_frame(self) -> np.ndarray:
        leg_pos_rel = self.b2w_joint_pos[:12] - self.default_joint_pos_policy[:12]
        arm_pos_rel = self.z1.q - self.default_joint_pos_policy[12:18]

        obs = np.concatenate(
            [
                self.base_ang_vel_b,                                    # 3
                self.projected_gravity_b,                               # 3
                self.base_command,                                      # 3
                self.ee_cmd_plb_current,                                # 9
                leg_pos_rel.astype(np.float32),                         # 12
                arm_pos_rel.astype(np.float32),                         # 6
                self.b2w_joint_vel[:12],                                # 12
                np.asarray(self.z1.qd, dtype=np.float32).reshape(6),    # 6
                self.b2w_joint_vel[12:16],                              # 4
                self.last_ll_action,                                    # 22
            ],
            dtype=np.float32,
        )
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite value in the WBC observation.")
        return obs

    def run_low_level_policy(self) -> None:
        """One WBC inference -> leg / arm / wheel targets.

        Plain JointPositionAction semantics, exactly as in training and in
        deploy/b2wz1_wbc_uan.py: no startup blend-in, no arm target rate
        limiter.
        """
        self.ll_history.append(self.build_ll_obs_frame())
        obs = self.ll_history.flat()

        if self.debug_ll_obs_enabled and self.debug_ll_obs_count < self.debug_obs_print_max:
            self.debug_ll_obs_count += 1
            print(
                f"[LL-OBS] tick={self.ll_tick} shape={obs.shape} "
                f"min={obs.min():+.3f} max={obs.max():+.3f}"
            )

        action = self.low_session.run(
            [self.low_output_name], {self.low_input_name: obs[None, :]}
        )[0][0].astype(np.float32)
        if action.shape != (self.ll_action_dim,) or not np.isfinite(action).all():
            raise RuntimeError(f"Invalid WBC action: {action}")

        self.last_ll_action[:] = action
        self.leg_target[:] = (
            self.default_leg_pos_policy + self.leg_action_scale * action[0:12]
        ).astype(np.float32)
        self.arm_target[:] = (
            self.default_arm_pos + self.arm_action_scale * action[12:18]
        ).astype(np.float32)
        self.wheel_cmd[:] = (self.wheel_action_scale * action[18:22]).astype(np.float32)

    def check_arm_target_range(self) -> Optional[str]:
        """Fault if the WBC commanded past the live SDK range.

        Same check deploy/b2wz1_wbc_uan.py makes. jointProtect() does NOT clamp
        position, and the MuJoCo model the policy trained against gives joint 2
        five degrees more travel than the real arm has -- so the policy can
        legitimately command range the hardware does not have and jam the stop
        at clipped torque, which is what faults the arm.
        """
        below = self.arm_target < self.arm_q_lower
        above = self.arm_target > self.arm_q_upper
        if not (below.any() or above.any()):
            return None
        joint = int(np.argmax(below | above))
        return (
            f"WBC arm target outside the live SDK range at joint{joint + 1}: "
            f"q_des={np.round(self.arm_target, 4).tolist()} "
            f"lower={self.arm_q_lower.tolist()} upper={self.arm_q_upper.tolist()}"
        )

    # ------------------------------------------------------------------
    # Actuation
    # ------------------------------------------------------------------

    def send_b2w_cmd(self) -> None:
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def _write_b2w_pose_cmd(
        self, target_b2w_pos_policy: np.ndarray, *, use_pd_gains: bool
    ) -> None:
        """Position-servo all 16 B2W joints toward a policy-order pose.

        Wheels are never position-servoed: their kp is 0 in both gain sets, so
        they receive q=0, dq=0 and damping only.
        """
        target_hw = np.asarray(target_b2w_pos_policy, dtype=np.float32).reshape(16)[
            self.policy_to_hardware
        ]
        kps_hw = self.kps_pd_hw if use_pd_gains else self.kps_rl_hw
        kds_hw = self.kds_pd_hw if use_pd_gains else self.kds_rl_hw

        for hw_idx in range(self.num_b2w_dof):
            motor = self.low_cmd.motor_cmd[hw_idx]
            is_leg = hw_idx in self.leg_hardware_indices
            motor.q = float(target_hw[hw_idx]) if is_leg else 0.0
            motor.dq = 0.0
            motor.kp = float(kps_hw[hw_idx])
            motor.kd = float(kds_hw[hw_idx])
            motor.tau = 0.0

    def _write_b2w_wbc_cmd(self) -> None:
        """The WBC command: leg position targets + wheel velocity targets."""
        target_policy = self.default_b2w_pos_policy.copy()
        target_policy[self.leg_policy_indices] = self.leg_target
        target_hw = target_policy[self.policy_to_hardware]

        for hw_idx in range(self.num_b2w_dof):
            motor = self.low_cmd.motor_cmd[hw_idx]
            motor.kp = float(self.kps_rl_hw[hw_idx])
            motor.kd = float(self.kds_rl_hw[hw_idx])
            motor.tau = 0.0
            if hw_idx in self.leg_hardware_indices:
                motor.q = float(target_hw[hw_idx])
                motor.dq = 0.0
            else:
                motor.q = 0.0
                motor.dq = float(self.wheel_cmd[self.hardware_to_wheel_cmd[hw_idx]])

    def send_hardware_targets(self) -> None:
        """One hardware tick: re-send the current WBC targets to B2W and Z1.

        Called at hardware_command_hz, so the 50 Hz WBC targets are held on
        both robots the way deploy/b2wz1_wbc_uan.py holds them -- and the way
        the UAN data behind the WBC was collected.
        """
        self._write_b2w_wbc_cmd()
        self.send_b2w_cmd()
        self.z1.track_target_pd_runtime_once(
            q_target=self.arm_target.copy(),
            gripper_q_target_training=float(self.gripper_target),
            use_startup_gains=False,
        )

    # ------------------------------------------------------------------
    # Damping protection
    # ------------------------------------------------------------------

    def _write_b2w_damping_cmd(self) -> None:
        for hw_idx in range(self.num_b2w_dof):
            motor = self.low_cmd.motor_cmd[hw_idx]
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = float(
                self.damping_kd_wheel
                if hw_idx in self.wheel_hardware_indices
                else self.damping_kd_b2w
            )
            motor.tau = 0.0

    def send_b2w_damping_once(self) -> None:
        self._write_b2w_damping_cmd()
        self.send_b2w_cmd()

    def send_z1_damping_once(self) -> None:
        """Best-effort Z1 hold-in-place command.

        Deliberately does NOT call setFsmLowcmd(): nothing in protection mode,
        including the manual leg recovery, may drag a PASSIVE arm back into
        LOWCMD and start moving it.
        """
        with self.z1._comm_lock:  # noqa: SLF001 - the adapter exposes no public form
            self.z1._send_arm_command_once(  # noqa: SLF001
                q_cmd=self.z1.q.copy(),
                gripper_q_cmd=float(self.z1.gripper_q),
                kp_cmd=self.damping_kp_z1,
                kd_cmd=self.damping_kd_z1,
                qd_cmd=np.zeros(6, dtype=np.float32),
                tau_cmd=np.zeros(6, dtype=np.float32),
            )

    def _move_legs_to_recovery_target(self) -> bool:
        """Interpolate the 12 leg joints from measured q to the recovery pose.

        The wheels get damping only, and the Z1 keeps receiving exactly the
        protection command, so the arm holds where it is.
        """
        self.read_b2w_state()
        start_leg = self.b2w_joint_pos[self.leg_policy_indices].copy()
        target_leg = self.leg_recovery_target.copy()
        num_steps = max(1, int(round(self.leg_recovery_duration_s / self.control_dt)))

        print(
            f"[RECOVERY] leg12 -> target over {self.leg_recovery_duration_s:.2f}s | "
            "wheels damped | Z1 untouched"
        )
        print("[RECOVERY] start =" + np.array2string(start_leg, precision=3, max_line_width=200))
        print("[RECOVERY] target=" + np.array2string(target_leg, precision=3, max_line_width=200))

        deadline = time.perf_counter()
        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            pose = np.zeros(self.num_b2w_dof, dtype=np.float32)
            pose[self.leg_policy_indices] = (1.0 - alpha) * start_leg + alpha * target_leg
            self._write_b2w_pose_cmd(pose, use_pd_gains=True)
            self.send_b2w_cmd()
            try:
                self.send_z1_damping_once()
            except Exception as exc:  # noqa: BLE001 - never abort recovery on a Z1 hiccup
                print(f"[RECOVERY][WARN] Z1 protection command failed: {exc!r}")
            deadline += self.control_dt
            sleep_until(deadline)

        self.read_b2w_state()
        error = float(
            np.max(np.abs(self.b2w_joint_pos[self.leg_policy_indices] - target_leg))
        )
        print(
            f"[RECOVERY] verification | max_abs_err={error:.4f} rad | "
            f"tol={self.leg_recovery_tolerance_rad:.4f} rad"
        )
        if not np.isfinite(error) or error > self.leg_recovery_tolerance_rad:
            print("[RECOVERY][ERROR] legs did not reach the target within tolerance.")
            return False
        return True

    def recover_legs(self) -> bool:
        """Operator-triggered leg recovery. Never resumes the policy."""
        print("\n" + "=" * 96)
        print("[RECOVERY] MANUAL LEG RECOVERY STARTED -- the arm is NOT moved.")
        print("=" * 96)
        try:
            self.read_robot_state()
            ok = self._move_legs_to_recovery_target()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[RECOVERY][ERROR] {exc!r}")
            return False

        print(
            "[RECOVERY] SUCCESS." if ok else "[RECOVERY] FAILED.",
            "Returning to high-damping protection; the policy will NOT resume.",
        )
        print("=" * 96 + "\n")
        return ok

    def enter_damping_protection_mode(self, reason: str) -> None:
        """Steady high-damping hold, with an operator-armed leg recovery."""
        print("\n" + "!" * 96)
        print(f"[B2WZ1-ABS][PROTECT] {reason}")
        print("[B2WZ1-ABS][PROTECT] B2W: kp=0, dq=0, tau=0, high kd.")
        print(
            "[B2WZ1-ABS][PROTECT] Z1: hold current q, high kd "
            "(PASSIVE is never forced back into LOWCMD)."
        )
        if self.leg_recovery_enabled:
            print(
                "[B2WZ1-ABS][PROTECT] Release A, then press A to fold the legs "
                "down. The arm is not moved."
            )
        print("[B2WZ1-ABS][PROTECT] Press Ctrl-C to quit.")
        print("!" * 96 + "\n")

        # Require a release first: a previously-held A must not trigger
        # recovery the instant protection is entered.
        recovery_armed = self.remote_controller.button[KeyMap.A] != 1
        if self.leg_recovery_enabled and not recovery_armed:
            print("[PROTECT] A is held; release it to arm leg recovery.")

        counter = 0
        try:
            while True:
                loop_start = time.perf_counter()

                for label, action in (
                    ("sensor read", self.read_robot_state),
                    ("B2W damping", self.send_b2w_damping_once),
                    ("Z1 damping", self.send_z1_damping_once),
                ):
                    try:
                        action()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[PROTECT][WARN] {label} failed: {exc!r}")

                if self.leg_recovery_enabled:
                    pressed = self.remote_controller.button[KeyMap.A] == 1
                    if not recovery_armed:
                        if not pressed:
                            recovery_armed = True
                            print("[PROTECT] A released; leg recovery armed.")
                    elif pressed:
                        recovery_armed = False
                        print("[PROTECT] Fresh A press -> starting leg recovery.")
                        self.recover_legs()
                        print("[PROTECT] Remaining in protection mode.")
                        loop_start = time.perf_counter()

                if counter % max(1, self.damping_print_period) == 0:
                    print(
                        f"[PROTECT] active | kd_b2w={self.damping_kd_b2w:.1f} | "
                        f"kd_wheel={self.damping_kd_wheel:.1f} | "
                        f"z1_fsm={self.z1.get_fsm_state()} | "
                        f"gripper_q={self.z1.gripper_q:+.3f} | "
                        f"recovery_armed={int(recovery_armed)}"
                    )
                counter += 1
                sleep_until(loop_start + self.control_dt)

        except KeyboardInterrupt:
            print("[B2WZ1-ABS][PROTECT] Ctrl-C received.")

    def send_exit_damping_once(self) -> None:
        """One damping packet on a clean exit, so nothing is left driven."""
        try:
            self.read_robot_state()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.send_b2w_damping_once()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.send_z1_damping_once()
        except Exception as exc:  # noqa: BLE001
            print(f"[EXIT][WARN] Z1 damping command failed: {exc!r}")

    # ------------------------------------------------------------------
    # Startup sequence
    # ------------------------------------------------------------------

    def zero_torque_state(self) -> None:
        print("[B2WZ1-ABS] Zero torque. Press START to continue.")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_b2w_cmd()
            time.sleep(self.control_dt)
        print("[B2WZ1-ABS] START pressed.")

    def move_arm_to_default(self) -> None:
        """Park the Z1 at the policy default before any B2W motion."""
        print("[B2WZ1-ABS] Moving the arm to DEFAULT with the gripper OPEN...")
        self.z1.move_to_pose_official(
            target_q=self.default_arm_pos.copy(),
            # A startup position-servo API: it expects the RAW SDK coordinate.
            target_gripper=self.gripper_training_to_sdk(self.gripper_open_pos),
            duration_s=float(self.cfg["arm_default_transition_s"]),
            step_callback=None,
        )
        self.z1.read_state()
        error = float(np.max(np.abs(self.z1.q - self.default_arm_pos)))
        print(f"[B2WZ1-ABS] Arm at DEFAULT, max err={error:.4f} rad.")

    def hold_arm_default_until_A(self) -> None:
        print("[B2WZ1-ABS] Holding arm DEFAULT + gripper OPEN. Press A to continue.")
        while self.remote_controller.button[KeyMap.A] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_b2w_cmd()
            self._hold_arm_default_lowcmd()
            time.sleep(self.control_dt)
        print("[B2WZ1-ABS] A pressed.")

    def _hold_arm_default_lowcmd(self) -> None:
        self.z1.hold_pose_lowcmd(
            self.default_arm_pos.copy(),
            self.gripper_training_to_sdk(self.gripper_open_pos),
        )

    def move_b2w_to_pose(self, target_b2w_pos_policy: np.ndarray, duration_s: float) -> None:
        """Interpolate all 16 B2W joints from measured q to a policy-order pose."""
        target = np.asarray(target_b2w_pos_policy, dtype=np.float32).reshape(16)
        num_steps = max(1, int(round(duration_s / self.control_dt)))

        start_hw = np.array(
            [self.low_state.motor_state[i].q for i in range(self.num_b2w_dof)],
            dtype=np.float32,
        )
        start_policy = start_hw[self.hardware_to_policy]

        print(f"[B2WZ1-ABS] Moving B2W to target pose over {duration_s:.2f}s...")
        deadline = time.perf_counter()
        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            self._write_b2w_pose_cmd(
                (1.0 - alpha) * start_policy + alpha * target, use_pd_gains=True
            )
            self.send_b2w_cmd()
            self._hold_arm_default_lowcmd()
            deadline += self.control_dt
            sleep_until(deadline)
        print("[B2WZ1-ABS] B2W pose reached.")

    def hold_default_until_ready_and_A(self) -> None:
        """Hold full DEFAULT until mocap is live and the operator accepts start.

        Everything the first observation needs must be current: the object, the
        retrieval target, and a tracked body (which is what supplies the base
        height). A is only accepted while all three hold, and is re-checked
        against a fresh snapshot so a stale "ready" cannot be accepted.
        """
        print("[B2WZ1-ABS] Holding full DEFAULT pose + gripper OPEN.")
        print("[B2WZ1-ABS] Policy start needs live object + retrieval + base height.")
        print("[B2WZ1-ABS] When all three are valid, press A to start.")

        last_print = 0.0
        while True:
            self._write_b2w_pose_cmd(self.default_b2w_pos_policy, use_pd_gains=True)
            self.send_b2w_cmd()
            self._hold_arm_default_lowcmd()
            self.read_robot_state()

            snap = self.read_perception()
            ready, status = self._perception_start_readiness(snap)

            now = time.monotonic()
            if now - last_print >= self.perception_status_print_s:
                last_print = now
                print(f"[PERCEPTION-WAIT] ready={int(ready)} | {status}")

            if ready and self.remote_controller.button[KeyMap.A] == 1:
                # Re-read so an old "ready" state cannot be accepted.
                final_ready, final_status = self._perception_start_readiness(
                    self.read_perception()
                )
                if final_ready:
                    print(
                        "[B2WZ1-ABS] A pressed with live perception "
                        f"(base_height={self.base_height:.4f} m). Starting policy."
                    )
                    return
                print(f"[B2WZ1-ABS] A pressed but perception went invalid: {final_status}")

            time.sleep(self.control_dt)

    def _perception_start_readiness(self, snap: Dict[str, Any]) -> Tuple[bool, str]:
        obj = snap.get("object") or {}
        ret = snap.get("retrieval") or {}
        height = snap.get("base_height") or {}

        object_ok = self._channel_point(snap, "object", "position_base") is not None
        retrieval_ok = (
            self._channel_point(snap, "retrieval", "retrieval_target_base") is not None
        )
        height_ok = bool(height.get("valid")) and self.base_height is not None

        status = (
            f"OBJ={int(object_ok)}:{obj.get('source')} reason={obj.get('reason')} | "
            f"RET={int(retrieval_ok)}:{ret.get('source')} reason={ret.get('reason')} | "
            f"HEIGHT={int(height_ok)}:"
            f"{'%.4f m' % self.base_height if self.base_height is not None else 'none'} "
            f"reason={height.get('reason')}"
        )
        return bool(object_ok and retrieval_ok and height_ok), status

    def settle_on_runtime_gains(self) -> None:
        """Hold DEFAULT on the RUNTIME gains before the first WBC step.

        Going straight from the stiff startup-gain hold into the policy would
        let the WBC take its first observations mid gain-switch. This is the
        last startup block of deploy/b2wz1_wbc_uan.py.
        """
        num_ticks = int(round(self.runtime_settle_s * self.hardware_hz))
        if num_ticks <= 0:
            return
        print(
            f"[B2WZ1-ABS] Settling {self.runtime_settle_s:.2f}s at DEFAULT on "
            f"runtime gains ({num_ticks} ticks at {self.hardware_hz:g} Hz)."
        )
        t0 = time.perf_counter()
        for tick in range(num_ticks):
            self._write_b2w_pose_cmd(self.default_b2w_pos_policy, use_pd_gains=False)
            self.send_b2w_cmd()
            self.z1.track_target_pd_runtime_once(
                q_target=self.default_arm_pos.copy(),
                gripper_q_target_training=float(self.gripper_open_pos),
                use_startup_gains=False,
            )
            sleep_until(t0 + (tick + 1) * self.hardware_dt)

    def initialize_policy_state_and_history(self) -> None:
        """Reset to the training reset state and seed both histories.

            base command   = 0
            EE command     = neutral
            WBC last action= 0
            previous HL    = neutral ABS arm action, executed gripper OPEN
            histories      = the first valid frame, repeated
        """
        self.read_robot_state()
        snap = self.read_perception()

        ready, status = self._perception_start_readiness(snap)
        if not ready:
            raise RuntimeError(f"Perception went invalid before initialization: {status}")

        self.base_command[:] = 0.0
        self.ee_cmd_plb_current[:] = self.build_neutral_ee_command()
        self.last_ll_action[:] = 0.0

        # NOT zeros: in ABS coordinates zero is a real command at the midpoint
        # of every range, so the reset action encodes the neutral EE pose.
        self.current_hl_action[:] = 0.0
        self.current_hl_action[3:8] = self.neutral_arm_action
        self.raw_gripper_action = -1.0
        self.executed_gripper_cmd_norm = -1.0

        self.leg_target[:] = self.default_leg_pos_policy
        self.arm_target[:] = self.default_arm_pos
        self.wheel_cmd[:] = 0.0
        self.gripper_target = float(self.gripper_open_pos)

        task = self.resolve_task_state(snap)
        if not task["valid"]:
            raise RuntimeError("Invalid initial task state: " + task["reason"])

        self.ll_history.reset(self.build_ll_obs_frame())
        self.hl_history.reset(self.build_hl_obs_frame(task))

        self.ll_tick = 0
        self.hl_tick = 0
        self._abs_decode_debug_printed = False
        self.debug_hl_obs_count = 0
        self.debug_ll_obs_count = 0

        print(
            "[B2WZ1-ABS] Policy state initialized | neutral ABS arm action="
            + np.array2string(self.neutral_arm_action, precision=4, floatmode="fixed")
        )

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def wbc_step(self) -> Optional[str]:
        """One 50 Hz step. Returns a protection reason, or None to continue.

        The high level runs on every ll_steps_per_hl_step-th call, before the
        WBC, so the WBC always consumes the freshest high-level command --
        the training and sim2sim schedule.
        """
        self.read_robot_state()

        if self.remote_controller.button[KeyMap.select] == 1:
            return "SELECT pressed"
        if self.z1.get_fsm_state() == self.z1.unitree_arm_interface.ArmFSMState.PASSIVE:
            return "Z1 entered PASSIVE state"

        snap = self.read_perception()
        is_hl_boundary = self.ll_tick % self.ll_steps_per_hl_step == 0

        task = self.resolve_task_state(snap)
        if not task["valid"]:
            return "Mocap perception invalid: " + task["reason"]

        if is_hl_boundary:
            # hl_tick == 0 only on the very first step, where the history was
            # just seeded and must be inferred from as-is.
            self.run_high_level_policy(task, append_frame=self.hl_tick > 0)

        self.run_low_level_policy()
        out_of_range = self.check_arm_target_range()
        if out_of_range is not None:
            return out_of_range

        self.ll_tick += 1
        self.push_visualizer_state(task)

        if (
            self.debug_print_period_steps > 0
            and self.ll_tick % self.debug_print_period_steps == 0
        ):
            self.print_runtime_debug(task)
        return None

    def policy_loop(self) -> str:
        """The 500 Hz hardware loop. Returns the reason it stopped.

        One timer, exactly as deploy/b2wz1_wbc_uan.py: the WBC runs on a
        decimation of it, and every tick re-sends the current targets so both
        robots are held at the hardware rate rather than at the policy rate.
        """
        print(
            f"[B2WZ1-ABS] Main loop started | HL {1.0 / self.hl_control_dt:.0f} Hz "
            f"-> WBC {1.0 / self.control_dt:.0f} Hz -> hardware "
            f"{self.hardware_hz:g} Hz ({self.hardware_ticks_per_wbc_step} sends per "
            "WBC step). SELECT = damping protection."
        )

        deadline = time.perf_counter()
        tick = 0
        while True:
            if tick % self.hardware_ticks_per_wbc_step == 0:
                reason = self.wbc_step()
                if reason is not None:
                    return reason

            self.send_hardware_targets()

            deadline += self.hardware_dt
            now = time.perf_counter()
            if deadline < now - self.control_dt:
                # A whole WBC step behind. Re-anchor the wall clock rather than
                # burst out the backlog as identical back-to-back packets. The
                # tick counter keeps running, so the WBC decimation stays put.
                print("[TIMING][WARN] hardware loop overrun; re-anchoring.")
                deadline = now + self.hardware_dt
            elif deadline > now:
                time.sleep(deadline - now)
            tick += 1

    def print_runtime_debug(self, task: Dict[str, Any]) -> None:
        object_b = task.get("object_position_base")
        grasp_err = (
            float(np.linalg.norm(object_b - task["gripper_center_pos_base"]))
            if object_b is not None
            else float("nan")
        )
        print(
            f"[ABS] ll={self.ll_tick:06d} hl={self.hl_tick:06d} | "
            f"OBJ={task.get('object_source')} RET={task.get('retrieval_source')} | "
            f"grasp_err={grasp_err:.3f} | "
            f"grip_q_train={self.gripper_q_training():+.3f} "
            f"grip_tgt={self.gripper_target:+.3f} | "
            f"base_h={self._require_base_height():.4f}m | "
            f"base_cmd={np.round(self.base_command, 3)}"
        )
        print(
            f"      HL={np.round(self.current_hl_action, 3)} | "
            f"leg_act=[{self.last_ll_action[0:12].min():+.2f},"
            f"{self.last_ll_action[0:12].max():+.2f}] | "
            f"arm_act=[{self.last_ll_action[12:18].min():+.2f},"
            f"{self.last_ll_action[12:18].max():+.2f}] | "
            f"wheel_act=[{self.last_ll_action[18:22].min():+.2f},"
            f"{self.last_ll_action[18:22].max():+.2f}]"
        )

    # ------------------------------------------------------------------
    # MuJoCo debug view
    #
    # Two producers feed one window. The control loop pushes the rich state --
    # it alone knows the decoded EE command and the resolved task -- and a
    # background monitor takes over whenever the control loop has been quiet
    # for monitor_takeover_s, so the view stays live through the startup
    # holds, the wait for A and damping protection.
    # ------------------------------------------------------------------

    def _mocap_marker_cloud(self, snap: Optional[Dict[str, Any]] = None):
        """Every raw marker in the latest mocap frame."""
        snap = snap if snap is not None else self.last_perception_snapshot
        if not snap:
            return None
        markers = snap.get("markers")
        if not markers or not markers.get("count"):
            # Say so once: drawing nothing looks identical to a broken
            # visualizer, and the usual cause is a Motive setting -- rigid
            # bodies stream on their own toggle, so poses can arrive perfectly
            # while the marker sections are empty.
            if not self._viz_no_marker_warned:
                self._viz_no_marker_warned = True
                print(
                    "[VIZ][WARN] visualizer_show_all_markers is on, but this "
                    "NatNet stream carries NO markers. Rigid-body poses are "
                    "unaffected. Enable Motive > View > Data Streaming Pane > "
                    "'Labeled Markers' and 'Unlabeled Markers'."
                )
            return None

        if not self._viz_marker_count_announced:
            self._viz_marker_count_announced = True
            labeled = markers.get("labeled") or {}
            print(
                f"[VIZ] drawing {markers['count']} mocap markers "
                f"({len(labeled)} labeled set(s): "
                + ", ".join(f"{k}={len(v)}" for k, v in sorted(labeled.items()))
                + f"; unlabeled={len(markers.get('unlabeled') or [])})"
            )
        return markers

    def _mocap_root_frame(self, snap: Optional[Dict[str, Any]] = None):
        """The CALIBRATED root pose (base_link), drawn as a triad.

        Drawing this next to the raw Motive pivot makes the applied offset
        visible as the gap between them.
        """
        snap = snap if snap is not None else self.last_perception_snapshot
        body = (snap or {}).get("body") or {}
        if not body.get("valid"):
            return None
        pos, quat = body.get("root_position_world"), body.get("root_quat_wxyz")
        if pos is None or quat is None:
            return None
        return {"pos": pos, "quat_wxyz": quat}

    def _mocap_pivot_markers(self, snap: Optional[Dict[str, Any]] = None):
        """Raw Motive pivots for the three assets, as world-frame spheres."""
        if not getattr(self, "viz_show_pivots", False):
            return None
        snap = snap if snap is not None else self.last_perception_snapshot
        if not snap:
            return None

        markers = []
        for channel, rgba in (
            ("body", [1.0, 1.0, 1.0, 0.85]),
            ("object", [0.2, 1.0, 0.4, 0.55]),
            ("retrieval", [1.0, 0.3, 1.0, 0.55]),
        ):
            state = snap.get(channel) or {}
            if not state.get("valid") or state.get("position_world") is None:
                continue
            markers.append(
                {
                    "pos": state["position_world"],
                    "rgba": rgba,
                    "radius": self.viz_pivot_radius,
                }
            )
        return markers or None

    def push_visualizer_state(self, task: Dict[str, Any]) -> None:
        """Draw the scene from the control loop, where it physically is.

        `base_quat_wxyz` is the MOCAP orientation, not the IMU's: the object
        and target are drawn by rotating their base-frame positions back out to
        world, and the IMU's yaw is arbitrary, so using it would spin both
        targets around the robot by an unknown angle. The policy itself is
        untouched and still observes the IMU.
        """
        if self.visualizer is None:
            return

        state: Dict[str, Any] = {
            "base_quat_wxyz": self.base_quat_wxyz.copy(),
            "base_height": self._require_base_height(),
            "ground_z": self.ground_z,
            "b2w_joint_pos_policy": self.b2w_joint_pos.copy(),
            "z1_q": np.asarray(self.z1.q, dtype=np.float32).copy(),
            "gripper_q_training": self.gripper_q_training(),
            "ee_cmd_plb": self.ee_cmd_plb_current.copy(),
            "object_pos_base": task.get("object_position_base"),
            "retrieval_pos_base": task.get("retrieval_target_base"),
            "gripper_center_pos_base": task.get("gripper_center_pos_base"),
        }

        body = (self.last_perception_snapshot or {}).get("body") or {}
        if body.get("valid") and body.get("root_position_world") is not None:
            state.update(
                base_pos_w=np.asarray(body["root_position_world"], dtype=np.float64),
                base_quat_wxyz=np.asarray(body["root_quat_wxyz"], dtype=np.float32),
                mocap_points=self._mocap_pivot_markers(),
                mocap_markers=self._mocap_marker_cloud(),
                root_frame=self._mocap_root_frame(),
            )

        self.visualizer.push_state(**state)
        # Tells the monitor thread the control loop is driving the view.
        self._last_control_push = time.monotonic()

    def start_monitor(self) -> None:
        """Keep the window fed in every phase, not just the policy loop.

        This thread samples telemetry that is live regardless of phase --
        `low_state` is refreshed by its DDS callback and the mocap snapshot by
        the NatNet thread -- and it only reads, so it cannot perturb control.
        """
        if self.visualizer is None or not self.monitor_enabled or self._monitor_thread:
            return

        self.visualizer.start()
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="mocap-scene-monitor", daemon=True
        )
        self._monitor_thread.start()
        print(
            f"[B2WZ1-ABS] Scene monitor running at {self.monitor_hz:g} Hz "
            "(live through startup, policy and damping)."
        )

    def stop_monitor(self) -> None:
        # getattr, not attribute access: this runs from run()'s finally, and
        # raising here on a partly built controller would MASK the exception
        # that actually ended the run.
        if getattr(self, "_monitor_thread", None) is None:
            return
        self._monitor_stop.set()
        self._monitor_thread.join(timeout=2.0)
        self._monitor_thread = None

    def _monitor_loop(self) -> None:
        period = 1.0 / max(self.monitor_hz, 1.0)
        while not self._monitor_stop.is_set():
            loop_start = time.perf_counter()
            try:
                if time.monotonic() - self._last_control_push >= self.monitor_takeover_s:
                    state = self._monitor_state()
                    if state is not None:
                        self.visualizer.push_state(**state)
            except Exception as exc:  # noqa: BLE001 - a debug view must never kill a run
                if not self._monitor_warned:
                    self._monitor_warned = True
                    print(f"[MONITOR][WARN] disabled after error: {exc!r}")
                return
            remaining = period - (time.perf_counter() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)

    def _monitor_state(self) -> Optional[Dict[str, Any]]:
        """A render snapshot built from phase-independent telemetry.

        Joint angles come straight from `low_state` rather than the
        controller's cached copies, because not every startup phase refreshes
        those whereas the DDS callback always does.
        """
        low = self.low_state
        if low is None or getattr(low, "tick", 0) == 0:
            return None

        joints = np.array(
            [low.motor_state[self.hardware_to_policy[i]].q for i in range(self.num_b2w_dof)],
            dtype=np.float32,
        )
        quat = np.asarray(low.imu_state.quaternion, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(quat))
        quat = quat / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        state: Dict[str, Any] = {
            "base_quat_wxyz": quat,
            "base_height": float(self.base_height or 0.0),
            "ground_z": self.ground_z,
            "b2w_joint_pos_policy": joints,
            "z1_q": np.asarray(self.z1.q, dtype=np.float32).copy(),
            "gripper_q_training": self.gripper_q_training(),
        }

        snap = self.perception.get_latest_snapshot()
        body = snap.get("body") or {}
        if body.get("valid"):
            state["base_pos_w"] = np.asarray(body["root_position_world"], dtype=np.float64)
            state["base_quat_wxyz"] = np.asarray(body["root_quat_wxyz"], dtype=np.float32)
            height = (snap.get("base_height") or {}).get("height_m")
            if height is not None:
                state["base_height"] = float(height)

        obj = snap.get("object") or {}
        if obj.get("valid"):
            state["object_pos_base"] = np.asarray(obj["position_base"], dtype=np.float32)
            if obj.get("quat_wxyz") is not None:
                state["object_quat_w"] = np.asarray(obj["quat_wxyz"], dtype=np.float32)

        ret = snap.get("retrieval") or {}
        if ret.get("valid"):
            state["retrieval_pos_base"] = np.asarray(
                ret["retrieval_target_base"], dtype=np.float32
            )

        state["mocap_points"] = self._mocap_pivot_markers(snap)
        state["mocap_markers"] = self._mocap_marker_cloud(snap)
        state["root_frame"] = self._mocap_root_frame(snap)
        return state

    # ------------------------------------------------------------------
    # Setup / run
    # ------------------------------------------------------------------

    def setup(self) -> None:
        print("=" * 100)
        print("B2WZ1 ABS HIGH-LEVEL RETRIEVAL -> FROZEN WBC -> REAL ROBOT (MOCAP)")
        print("=" * 100)
        print(f"High policy      : {self.high_policy_path}")
        print(f"Low policy (WBC) : {self.low_policy_path}")
        print(
            f"Rates            : HL {1.0 / self.hl_control_dt:.0f} Hz -> "
            f"WBC {1.0 / self.control_dt:.0f} Hz -> hardware {self.hardware_hz:g} Hz "
            f"({self.hardware_ticks_per_wbc_step} sends per WBC step, as "
            "b2wz1_wbc_uan.py)"
        )
        print("UAN model        : NOT LOADED / NOT EXECUTED (simulation-only)")
        print("Arm control      : WBC q_des -> Z1 firmware POSITION_PD -> hardware")
        print(
            "Z1 gains         : "
            f"Kp={np.array2string(self.z1_kp_effective, precision=3)} | "
            f"Kd={np.array2string(self.z1_kd_effective, precision=3)}"
        )
        print(f"Gripper runtime  : {self.z1.gripper_runtime_mode} | "
              f"q_train = q_sdk - {self.z1.gripper_q_offset:.5f}")
        gripper_gains = (
            f"firmware kp={self.z1.gripper_kp:g} kd={self.z1.gripper_kd:g}"
            if self.z1.gripper_runtime_mode == "position_pd"
            else "firmware gains zeroed, DCMotor tau_f"
        )
        print(f"Gripper targets  : close {self.gripper_close_pos:+.4f} / "
              f"open {self.gripper_open_pos:+.4f} rad (training) | {gripper_gains}")
        print("Arm filtering    : NONE (no startup blend, no target rate limiter)")
        print("HL arm semantics : ABS normalized action -> PLB xyz / yaw / policy-pitch")
        print(
            f"kp0_z floor      : {self.kp0_z_cmd_min:.3f} m "
            f"(decoded range [{self.kp0_z_range[0]:.3f}, {self.kp0_z_range[1]:.3f}] m)"
        )
        print(
            "Pitch convention : pitch_geom = "
            f"{self.ee_pitch_to_euler_sign:+.0f} * pitch_policy "
            "(Z1 local +X points at the fingertips)"
        )
        print(
            "Neutral ABS act  : "
            + np.array2string(self.neutral_arm_action, precision=6, floatmode="fixed")
        )
        print("Sensing          : OPTITRACK MOCAP (no camera, no AprilTag, no VO)")
        for selector in (
            self.perception.body,
            self.perception.object_body,
            self.perception.retrieval_body,
        ):
            print(f"  {selector.describe()}")
        offset = self.perception.root_offset
        roll, pitch, yaw = offset.rpy()
        print(
            "  root offset    : "
            f"pos={np.round(offset.pos, 4).tolist()} m | "
            f"rpy={np.round(np.degrees([roll, pitch, yaw]), 3).tolist()} deg | "
            f"{self.perception.root_offset_source()} | {self.mocap_root_offset_source}"
        )
        print(f"  ground_z       : {self.ground_z:.4f} m")
        print("Base height      : MOCAP (absolute, nothing anchored)")
        print(
            "Fault handling   : SELECT / Z1 PASSIVE / invalid mocap / arm target "
            "out of SDK range -> damping protection"
        )
        print(
            "Leg recovery     : "
            + (
                f"ENABLED | A in protection -> legs only over "
                f"{self.leg_recovery_duration_s:.1f}s (arm NOT moved)"
                if self.leg_recovery_enabled
                else "DISABLED"
            )
        )
        print("=" * 100)

        self.wait_for_low_state()
        InitLowCmd(self.low_cmd)

        self.z1.connect()
        self.read_robot_state()

        print("[B2WZ1-ABS] Connecting to Motive...")
        self.perception.initialize()
        self.perception.start()
        print("[B2WZ1-ABS] Mocap perception running.")

        self.start_monitor()

    def run(self) -> None:
        protection_reason: Optional[str] = None
        perception_started = False

        try:
            self.setup()
            perception_started = True

            self.zero_torque_state()
            self.move_arm_to_default()
            self.hold_arm_default_until_A()
            self.move_b2w_to_pose(
                self.squat_b2w_pos_policy, float(self.cfg["squat_transition_s"])
            )
            self.move_b2w_to_pose(
                self.default_b2w_pos_policy, float(self.cfg["default_transition_s"])
            )
            self.hold_default_until_ready_and_A()

            self.settle_on_runtime_gains()
            self.initialize_policy_state_and_history()

            protection_reason = self.policy_loop()

        except KeyboardInterrupt:
            print("[B2WZ1-ABS] KeyboardInterrupt received.")
        except Exception as exc:  # noqa: BLE001 - any fault must reach protection
            protection_reason = f"{type(exc).__name__}: {exc}"
            print(f"[B2WZ1-ABS][ERROR] {protection_reason}")
        finally:
            if protection_reason is not None:
                self.enter_damping_protection_mode(protection_reason)
            else:
                self.send_exit_damping_once()

            if perception_started:
                try:
                    self.perception.stop()
                except Exception as exc:  # noqa: BLE001
                    print(f"[EXIT][WARN] perception stop failed: {exc!r}")

            self.stop_monitor()
            if self.visualizer is not None:
                try:
                    self.visualizer.stop()
                except Exception as exc:  # noqa: BLE001
                    print(f"[EXIT][WARN] visualizer stop failed: {exc!r}")

            print("[B2WZ1-ABS] Exit.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "B2WZ1 ABS high-level retrieval over a frozen WBC, on the real "
            "robot with OptiTrack sensing."
        )
    )
    parser.add_argument("net", type=str, help="Unitree network interface, e.g. enxa0cec819e15f")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="deploy/configs/b2wz1_hl_retrieval_abs_uan_mocap.yaml",
        help="Path to the deployment YAML.",
    )
    args = parser.parse_args()

    # Exactly once, before the Z1 SDK and the B2W adapter come up.
    ChannelFactoryInitialize(0, args.net)

    B2WZ1AbsRetrievalMocapController(cfg_path=args.config).run()


if __name__ == "__main__":
    main()
