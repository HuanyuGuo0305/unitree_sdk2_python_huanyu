#!/usr/bin/env python3
"""
B2WZ1 hierarchical retrieval sim2real -- MOCAP sensing -- UAN-trained policies.

This is a deployment wrapper around the validated
b2wz1_hl_retrieval_mocap.py pipeline.

CRITICAL CONTROL SEMANTICS
--------------------------
The UAN is NOT deployed on the real robot.

Training:
    HL @ 10 Hz
      -> frozen WBC @ 50 Hz
      -> simulated Z1 arm: nominal PD + frozen UAN residual @ 250 Hz
      -> PhysX

Real deployment:
    HL @ 10 Hz
      -> frozen WBC @ 50 Hz
      -> q_des
      -> REAL Z1 firmware position-PD
      -> real actuator

The UAN was trained to make the simulated Z1 actuator behave like this real
firmware/hardware path. Running the UAN again on the robot would double-count
the actuator correction and would be the wrong controller.

The arm-side deployment alignment is therefore:
    SDK raw Kp * 25.6   == [64, 115.2, 64, 64, 64, 64]
    SDK raw Kd * 0.0128 == [3, 4, 3, 3, 3, 3]

which gives:
    arm_kps_runtime = [2.5, 4.5, 2.5, 2.5, 2.5, 2.5]
    arm_kds_runtime = [234.375, 312.5, 234.375, 234.375, 234.375, 234.375]

No arm target rate limiter is allowed. The 50-Hz WBC target is held and
re-sent by the existing arm command thread. The gripper remains the original
external DCMotor-style actuator used by retrieval training.

This wrapper also closes three observation-side sim2real gaps:

1. The mocap root frame must BE base_link, established either by aligning
   the asset frame inside Motive (mocap_root_frame_is_base_link: true) or by
   an externally measured offset. Raw, unvouched-for Motive rigid-body
   coordinates are rejected before policy execution.

2. The deployable grasp-confidence proxy again matches training/sim2sim:
       CLOSE commanded
       AND ||object_center - gripper_center|| < 0.10 m
       AND gripper is not fully closed
       AND gripper angle is holding
   with 3-step enter/exit hysteresis.

3. On the FIRST false->true grasp-proxy transition, the actor's
   object_center_pos_base observation is frozen to the ACTUAL measured value
   from that same HL step. The frozen 3-vector is then held unchanged for the
   rest of that policy run, even if object mocap later disappears or the proxy
   later exits. The 3-frame feature-major object/proxy histories are rewritten
   immediately before high-level ONNX inference so the actor sees the intended
   semantics without modifying the validated common mocap controller.

Run from the unitree_sdk2_python_huanyu repository root:

    python3 deploy/b2wz1_hl_retrieval_mocap_UAN.py \
        enxa0cec819e15f \
        deploy/configs/b2wz1_hl_retrieval_mocap_UAN.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

import numpy as np

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DEPLOY_DIR, ".."))
for _p in (_PROJECT_ROOT, _DEPLOY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from b2wz1_hl_retrieval_mocap import B2WZ1MocapRetrievalController


_EXPECTED_KP_FW = np.array(
    [2.5, 4.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float64
)
_EXPECTED_KD_FW = np.array(
    [234.375, 312.5, 234.375, 234.375, 234.375, 234.375],
    dtype=np.float64,
)
_EXPECTED_KP_TRAIN = np.array(
    [64.0, 115.2, 64.0, 64.0, 64.0, 64.0], dtype=np.float64
)
_EXPECTED_KD_TRAIN = np.array(
    [3.0, 4.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float64
)

# High-level actor frame after root_lin_vel was removed:
#   root_ang_vel_b(3), projected_gravity_b(3), leg_pos_rel(12),
#   arm_pos_rel(6), gripper_pos_rel(1), arm_joint_vel(6),
#   object_center_pos_base(3), gripper_orientation_base(6),
#   gripper_center_pos_base(3), retrieval_target_pos_base(3),
#   previous_hl_action(9), grasp_confidence_proxy(1) = 56.
_HL_FEATURE_DIMS = (3, 3, 12, 6, 1, 6, 3, 6, 3, 3, 9, 1)
_HL_OBJECT_FEATURE = 6
_HL_GRIPPER_POS_FEATURE = 4
_HL_GRIPPER_CENTER_FEATURE = 8
_HL_PREVIOUS_ACTION_FEATURE = 10
_HL_GRASP_PROXY_FEATURE = 11

_EXPECTED_GRASP_PROXY_ERROR_THRESHOLD = 0.10
_EXPECTED_GRIPPER_NOT_FULLY_CLOSED_THRESHOLD = -0.08726646259971647  # -5 deg
_EXPECTED_GRIPPER_HOLD_THRESHOLD = 0.05235987755982989  # 3 deg
_EXPECTED_GRASP_PROXY_ENTER_STEPS = 3
_EXPECTED_GRASP_PROXY_EXIT_STEPS = 3


def _feature_major_block_slice(
    feature_index: int,
    *,
    history_length: int,
) -> slice:
    start = sum(
        int(dim) * int(history_length)
        for dim in _HL_FEATURE_DIMS[:feature_index]
    )
    stop = start + int(_HL_FEATURE_DIMS[feature_index]) * int(history_length)
    return slice(start, stop)


class _HighPolicyObservationAlignmentSession:
    """ONNX session proxy that repairs only the two affected HL feature groups.

    The common real-robot controller still owns all sensing, safety, history
    construction and action decoding. This adapter intercepts the already-built
    168-D high-level actor observation immediately before ONNX inference and
    replaces:

      - object_center_pos_base history
      - grasp_confidence_proxy history

    with the UAN-deployment semantics requested here.

    Everything else is passed through bit-for-bit.
    """

    def __init__(
        self,
        owner: "B2WZ1MocapRetrievalUANTrainedController",
        wrapped_session: Any,
    ) -> None:
        self._owner = owner
        self._wrapped = wrapped_session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def run(self, output_names, input_feed, *args, **kwargs):
        new_feed = dict(input_feed)

        obs_key: Optional[str] = None
        obs_value: Optional[np.ndarray] = None
        for key, value in new_feed.items():
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[-1] == self._owner.hl_obs_dim:
                obs_key = key
                obs_value = arr
                break

        if obs_key is None or obs_value is None:
            raise RuntimeError(
                "Could not find the 168-D high-level actor observation in the "
                "ONNX input feed; grasp/object alignment was not applied."
            )

        fixed = self._owner._rewrite_high_level_actor_observation(obs_value)
        new_feed[obs_key] = fixed
        return self._wrapped.run(output_names, new_feed, *args, **kwargs)


class B2WZ1MocapRetrievalUANTrainedController(
    B2WZ1MocapRetrievalController
):
    """Validated mocap retrieval pipeline with UAN-trained frozen HL/LL policies."""

    def __init__(self, cfg_path: str, network_interface: str) -> None:
        super().__init__(
            cfg_path=cfg_path,
            network_interface=network_interface,
        )

        self._audit_uan_trained_real_deployment()

        self._hl_alignment_session: Optional[
            _HighPolicyObservationAlignmentSession
        ] = None
        self._hl_alignment_session_attr: Optional[str] = None

        self._grasp_proxy_error_threshold = float(
            self.cfg["grasp_proxy_error_threshold"]
        )
        self._gripper_not_fully_closed_threshold = float(
            self.cfg["gripper_not_fully_closed_angle_threshold"]
        )
        self._gripper_angle_hold_threshold = float(
            self.cfg["gripper_angle_hold_threshold"]
        )
        self._grasp_proxy_enter_steps = int(self.cfg["grasp_proxy_enter_steps"])
        self._grasp_proxy_exit_steps = int(self.cfg["grasp_proxy_exit_steps"])

        self._reset_hl_alignment_state()

        # Most versions create ONNX sessions in the common controller __init__.
        # Install immediately when possible; setup()/prime_first_hl_and_ll()
        # retry for versions that create them later.
        self._ensure_high_policy_observation_adapter(required=False)

    # ------------------------------------------------------------------
    # Startup/audit
    # ------------------------------------------------------------------

    def _audit_uan_trained_real_deployment(self) -> None:
        cfg = self.cfg

        # -------------------------------------------------------------
        # 1. UAN must NOT exist in the real-robot execution graph.
        # -------------------------------------------------------------
        forbidden_uan_keys = (
            "uan_model_path",
            "uan_history_length",
            "uan_action_scale",
            "uan_nominal_torque_limits",
            "uan_final_torque_limits",
        )
        present = [key for key in forbidden_uan_keys if key in cfg]
        if present:
            raise ValueError(
                "Real deployment must not load or execute the UAN. "
                f"Remove simulation-only UAN keys from the YAML: {present}"
            )

        # -------------------------------------------------------------
        # 2. Current WBC training has plain JointPositionAction semantics.
        # -------------------------------------------------------------
        rate_limit_keys = (
            "enable_arm_target_rate_limit",
            "arm_target_rate_limit",
        )
        present = [key for key in rate_limit_keys if key in cfg]
        if present:
            raise ValueError(
                "Arm target rate limiting is not part of the UAN-trained WBC. "
                f"Remove these keys: {present}"
            )

        # -------------------------------------------------------------
        # 3. Real Z1 arm MUST use firmware position-PD.
        # -------------------------------------------------------------
        mode = str(cfg.get("z1_arm_runtime_mode", "")).strip().lower()
        if mode != "position_pd":
            raise ValueError(
                "UAN-trained real deployment must use "
                'z1_arm_runtime_mode: "position_pd". '
                f"Got {mode!r}. Do not deploy UAN or external arm residual torque."
            )

        # -------------------------------------------------------------
        # 4. Audit Unitree protocol gain scaling.
        # -------------------------------------------------------------
        kp_fw = np.asarray(cfg["arm_kps_runtime"], dtype=np.float64).reshape(6)
        kd_fw = np.asarray(cfg["arm_kds_runtime"], dtype=np.float64).reshape(6)

        kp_scale = float(cfg.get("z1_kp_protocol_scale", 25.6))
        kd_scale = float(cfg.get("z1_kd_protocol_scale", 0.0128))

        kp_train = np.asarray(
            cfg.get("arm_kp_training", _EXPECTED_KP_TRAIN),
            dtype=np.float64,
        ).reshape(6)
        kd_train = np.asarray(
            cfg.get("arm_kd_training", _EXPECTED_KD_TRAIN),
            dtype=np.float64,
        ).reshape(6)

        kp_effective = kp_fw * kp_scale
        kd_effective = kd_fw * kd_scale

        if not np.allclose(kp_fw, _EXPECTED_KP_FW, atol=1e-9, rtol=0.0):
            raise ValueError(
                f"Unexpected Z1 firmware Kp: {kp_fw}. "
                f"Expected {_EXPECTED_KP_FW}."
            )
        if not np.allclose(kd_fw, _EXPECTED_KD_FW, atol=1e-9, rtol=0.0):
            raise ValueError(
                f"Unexpected Z1 firmware Kd: {kd_fw}. "
                f"Expected {_EXPECTED_KD_FW}."
            )
        if not np.allclose(kp_effective, kp_train, atol=1e-6, rtol=0.0):
            raise ValueError(
                "Z1 Kp protocol scaling mismatch:\n"
                f"  kp_fw * {kp_scale} = {kp_effective}\n"
                f"  training Kp       = {kp_train}"
            )
        if not np.allclose(kd_effective, kd_train, atol=1e-6, rtol=0.0):
            raise ValueError(
                "Z1 Kd protocol scaling mismatch:\n"
                f"  kd_fw * {kd_scale} = {kd_effective}\n"
                f"  training Kd       = {kd_train}"
            )
        if not np.allclose(kp_train, _EXPECTED_KP_TRAIN, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"Training Kp must be {_EXPECTED_KP_TRAIN}, got {kp_train}."
            )
        if not np.allclose(kd_train, _EXPECTED_KD_TRAIN, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"Training Kd must be {_EXPECTED_KD_TRAIN}, got {kd_train}."
            )

        self._uan_deploy_kp_effective = kp_effective
        self._uan_deploy_kd_effective = kd_effective

        # -------------------------------------------------------------
        # 5. Hierarchical policy timing/shape/history contract.
        # -------------------------------------------------------------
        if abs(float(self.control_dt) - 0.020) > 1e-9:
            raise ValueError(
                f"Frozen WBC must run at 50 Hz (control_dt=0.02), "
                f"got {self.control_dt}."
            )
        if int(self.ll_steps_per_hl_step) != 5:
            raise ValueError(
                "High-level policy must run every 5 low-level steps "
                f"(10 Hz), got {self.ll_steps_per_hl_step}."
            )
        if int(self.ll_obs_dim) != 400 or int(self.ll_action_dim) != 22:
            raise ValueError(
                f"Unexpected WBC interface: obs={self.ll_obs_dim}, "
                f"action={self.ll_action_dim}; expected 400/22."
            )
        if int(self.hl_obs_dim) != 168 or int(self.hl_action_dim) != 9:
            raise ValueError(
                f"Unexpected HL interface: obs={self.hl_obs_dim}, "
                f"action={self.hl_action_dim}; expected 168/9."
            )
        if int(self.hl_history_length) != 3:
            raise ValueError(
                f"High-level history must be 3 frames, got {self.hl_history_length}."
            )
        if sum(_HL_FEATURE_DIMS) * int(self.hl_history_length) != int(self.hl_obs_dim):
            raise ValueError("Internal HL feature-major layout audit failed.")

        if str(cfg.get("z1_gripper_runtime_mode", "")).lower() != "dcmotor":
            raise ValueError(
                "jointGripper must remain in the validated DCMotor-style "
                "runtime path used by retrieval training."
            )

        # -------------------------------------------------------------
        # 6. Bug fix #1: the mocap root frame must BE base_link.
        # -------------------------------------------------------------
        # Two ways to satisfy this, and the wrapper accepts either:
        #
        #   a. mocap_root_frame_is_base_link: true -- the asset's pivot and
        #      axes were aligned with base_link inside Motive, so the correct
        #      offset is identity and no offset file exists or is needed.
        #   b. mocap_root_offset_path / mocap_root_offset -- the offset was
        #      measured externally by deploy/b2w_mocap_root_calibration.py.
        #
        # What stays forbidden is neither: raw Motive asset coordinates
        # standing in for base_link by default.
        offset_path = cfg.get("mocap_root_offset_path")
        inline_offset = cfg.get("mocap_root_offset")
        motive_aligned = bool(cfg.get("mocap_root_frame_is_base_link", False))

        if offset_path:
            resolved = self._resolve_path(str(offset_path))
            if not os.path.isfile(resolved):
                raise FileNotFoundError(
                    "mocap_root_offset_path does not exist: "
                    f"configured={offset_path!r}, resolved={resolved!r}. "
                    "Run deploy/b2w_mocap_root_calibration.py first, or set "
                    "mocap_root_frame_is_base_link: true if the asset frame "
                    "was already aligned with base_link in Motive."
                )
            self._mocap_root_offset_resolved_path = resolved
        elif inline_offset:
            self._mocap_root_offset_resolved_path = "<inline>"
        elif motive_aligned:
            self._mocap_root_offset_resolved_path = (
                "<identity: asset frame aligned with base_link in Motive>"
            )
        else:
            raise ValueError(
                "UAN retrieval deployment requires the mocap root frame to be "
                "base_link. Either set mocap_root_frame_is_base_link: true (the "
                "asset was aligned in Motive) or point mocap_root_offset_path at "
                "the YAML produced by deploy/b2w_mocap_root_calibration.py."
            )

        if not bool(getattr(self.perception, "root_offset_calibrated", False)):
            raise ValueError(
                "MocapPerceptionSystem reports an untrusted root offset. "
                "Refusing to run the UAN-trained policy with the Motive asset "
                "frame pretending to be base_link."
            )

        # -------------------------------------------------------------
        # 7. Bug fix #2: real grasp proxy must match training/sim2sim.
        # -------------------------------------------------------------
        if str(cfg.get("grasp_proxy_mode", "")).strip().lower() != "heuristic":
            raise ValueError(
                'UAN deployment requires grasp_proxy_mode: "heuristic" so the '
                "proxy can match training/sim2sim exactly."
            )

        proxy_threshold = float(cfg.get("grasp_proxy_error_threshold", np.nan))
        if not np.isfinite(proxy_threshold) or abs(
            proxy_threshold - _EXPECTED_GRASP_PROXY_ERROR_THRESHOLD
        ) > 1e-12:
            raise ValueError(
                "grasp_proxy_error_threshold must be 0.10 m to match "
                f"training/sim2sim; got {proxy_threshold}."
            )

        not_closed_threshold = float(
            cfg.get("gripper_not_fully_closed_angle_threshold", np.nan)
        )
        hold_threshold = float(cfg.get("gripper_angle_hold_threshold", np.nan))
        if abs(
            not_closed_threshold - _EXPECTED_GRIPPER_NOT_FULLY_CLOSED_THRESHOLD
        ) > 1e-12:
            raise ValueError(
                "gripper_not_fully_closed_angle_threshold must remain -5 deg "
                "to match training/sim2sim."
            )
        if abs(hold_threshold - _EXPECTED_GRIPPER_HOLD_THRESHOLD) > 1e-12:
            raise ValueError(
                "gripper_angle_hold_threshold must remain 3 deg to match "
                "training/sim2sim."
            )
        if int(cfg.get("grasp_proxy_enter_steps", -1)) != _EXPECTED_GRASP_PROXY_ENTER_STEPS:
            raise ValueError("grasp_proxy_enter_steps must be 3.")
        if int(cfg.get("grasp_proxy_exit_steps", -1)) != _EXPECTED_GRASP_PROXY_EXIT_STEPS:
            raise ValueError("grasp_proxy_exit_steps must be 3.")

        # -------------------------------------------------------------
        # 8. Bug fix #3: freeze object_center_pos_base at first grasp=True.
        # -------------------------------------------------------------
        if not bool(cfg.get("freeze_object_center_pos_base_on_grasp", False)):
            raise ValueError(
                "freeze_object_center_pos_base_on_grasp must be true for this "
                "deployment: the actor must hold the object base-frame vector "
                "measured on the first grasp=True HL step."
            )

    # ------------------------------------------------------------------
    # High-level observation alignment
    # ------------------------------------------------------------------

    def _reset_hl_alignment_state(self) -> None:
        self._aligned_grasp_proxy = False
        self._aligned_grasp_proxy_enter_count = 0
        self._aligned_grasp_proxy_exit_count = 0
        self._aligned_prev_gripper_q: Optional[float] = None
        self._aligned_object_history: Optional[np.ndarray] = None
        self._aligned_proxy_history: Optional[np.ndarray] = None
        self._frozen_object_center_pos_base: Optional[np.ndarray] = None
        self._grasp_freeze_announced = False

    def _session_input_last_dim(self, session: Any) -> Optional[int]:
        try:
            inputs = session.get_inputs()
            if not inputs:
                return None
            shape = inputs[0].shape
            if not shape:
                return None
            last = shape[-1]
            return int(last) if isinstance(last, (int, np.integer)) else None
        except Exception:
            return None

    def _find_high_policy_session_attr(self) -> Optional[str]:
        preferred_names = (
            "high_sess",
            "high_session",
            "high_policy_sess",
            "high_policy_session",
            "hl_sess",
            "hl_session",
        )
        for name in preferred_names:
            session = getattr(self, name, None)
            if session is None or isinstance(session, _HighPolicyObservationAlignmentSession):
                continue
            if not callable(getattr(session, "run", None)):
                continue
            last_dim = self._session_input_last_dim(session)
            if last_dim in (None, int(self.hl_obs_dim)):
                return name

        # Fallback for a renamed controller member: find a session-like object
        # whose ONNX input is the 168-D HL observation. Do not touch the 400-D LL session.
        for name, session in vars(self).items():
            if isinstance(session, _HighPolicyObservationAlignmentSession):
                continue
            if not callable(getattr(session, "run", None)):
                continue
            if self._session_input_last_dim(session) == int(self.hl_obs_dim):
                return name
        return None

    def _ensure_high_policy_observation_adapter(self, *, required: bool) -> None:
        if self._hl_alignment_session is not None:
            return

        attr = self._find_high_policy_session_attr()
        if attr is None:
            if required:
                session_like = [
                    name
                    for name, value in vars(self).items()
                    if callable(getattr(value, "run", None))
                ]
                raise RuntimeError(
                    "Could not locate the high-level ONNX session to install the "
                    "object/proxy observation alignment adapter. Session-like "
                    f"attributes found: {session_like}"
                )
            return

        wrapped = getattr(self, attr)
        adapter = _HighPolicyObservationAlignmentSession(self, wrapped)
        setattr(self, attr, adapter)
        self._hl_alignment_session = adapter
        self._hl_alignment_session_attr = attr

    def _latest_actual_object_pos_base(
        self,
        fallback: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Return current measured object_center_pos_base and measurement validity.

        The exact mocap snapshot already used by the common controller is preferred
        so the distance gate and the actor observation refer to the same HL step.
        If the object is unavailable, keep the last aligned object value for the
        observation but report measurement_valid=False so the grasp proxy cannot
        newly enter without a real object measurement.
        """
        snap = self.last_perception_snapshot or {}
        obj = snap.get("object") or {}
        if bool(obj.get("valid", False)):
            pos = obj.get("position_base")
            if pos is not None:
                p = np.asarray(pos, dtype=np.float32).reshape(3)
                if np.all(np.isfinite(p)):
                    return p.copy(), True

        if self._aligned_object_history is not None:
            return self._aligned_object_history[-1].copy(), False

        fb = np.asarray(fallback, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(fb)):
            fb = np.zeros(3, dtype=np.float32)
        return fb.copy(), False

    def _update_aligned_grasp_proxy(
        self,
        *,
        object_pos_base: np.ndarray,
        object_measurement_valid: bool,
        gripper_center_pos_base: np.ndarray,
        gripper_joint_pos_train: float,
        close_commanded: bool,
    ) -> tuple[bool, bool, float]:
        if self._aligned_prev_gripper_q is None:
            gripper_angle_delta = 0.0
        else:
            gripper_angle_delta = abs(
                float(gripper_joint_pos_train) - float(self._aligned_prev_gripper_q)
            )

        grasp_error = float(
            np.linalg.norm(
                np.asarray(object_pos_base, dtype=np.float64)
                - np.asarray(gripper_center_pos_base, dtype=np.float64)
            )
        )

        gripper_not_fully_closed = (
            float(gripper_joint_pos_train)
            < self._gripper_not_fully_closed_threshold
        )
        gripper_angle_holding = (
            gripper_angle_delta < self._gripper_angle_hold_threshold
        )

        proxy_candidate = bool(
            close_commanded
            and object_measurement_valid
            and grasp_error < self._grasp_proxy_error_threshold
            and gripper_not_fully_closed
            and gripper_angle_holding
        )

        was_proxy = bool(self._aligned_grasp_proxy)
        if not self._aligned_grasp_proxy:
            if proxy_candidate:
                self._aligned_grasp_proxy_enter_count += 1
            else:
                self._aligned_grasp_proxy_enter_count = 0

            if self._aligned_grasp_proxy_enter_count >= self._grasp_proxy_enter_steps:
                self._aligned_grasp_proxy = True
                self._aligned_grasp_proxy_enter_count = 0
                self._aligned_grasp_proxy_exit_count = 0
        else:
            if not proxy_candidate:
                self._aligned_grasp_proxy_exit_count += 1
            else:
                self._aligned_grasp_proxy_exit_count = 0

            if self._aligned_grasp_proxy_exit_count >= self._grasp_proxy_exit_steps:
                self._aligned_grasp_proxy = False
                self._aligned_grasp_proxy_exit_count = 0
                self._aligned_grasp_proxy_enter_count = 0

        self._aligned_prev_gripper_q = float(gripper_joint_pos_train)
        rising_edge = (not was_proxy) and bool(self._aligned_grasp_proxy)
        return bool(self._aligned_grasp_proxy), rising_edge, grasp_error

    @staticmethod
    def _append_history(history: np.ndarray, current: np.ndarray) -> np.ndarray:
        out = np.empty_like(history)
        out[:-1] = history[1:]
        out[-1] = current
        return out

    def _rewrite_high_level_actor_observation(
        self,
        obs_batch: np.ndarray,
    ) -> np.ndarray:
        """Repair object/proxy history in the 168-D feature-major HL actor input."""
        obs = np.asarray(obs_batch)
        if obs.shape[-1] != int(self.hl_obs_dim):
            raise RuntimeError(
                f"HL observation dim mismatch in adapter: got {obs.shape}, "
                f"expected last dim {self.hl_obs_dim}."
            )

        if obs.ndim == 1:
            work = obs.reshape(1, -1).copy()
            squeeze = True
        elif obs.ndim == 2 and obs.shape[0] == 1:
            work = obs.copy()
            squeeze = False
        else:
            raise RuntimeError(
                "Real-robot HL observation adapter expects batch size 1; "
                f"got shape {obs.shape}."
            )

        h = int(self.hl_history_length)
        object_slice = _feature_major_block_slice(
            _HL_OBJECT_FEATURE,
            history_length=h,
        )
        gripper_pos_slice = _feature_major_block_slice(
            _HL_GRIPPER_POS_FEATURE,
            history_length=h,
        )
        gripper_center_slice = _feature_major_block_slice(
            _HL_GRIPPER_CENTER_FEATURE,
            history_length=h,
        )
        previous_action_slice = _feature_major_block_slice(
            _HL_PREVIOUS_ACTION_FEATURE,
            history_length=h,
        )
        proxy_slice = _feature_major_block_slice(
            _HL_GRASP_PROXY_FEATURE,
            history_length=h,
        )

        row = work[0]
        incoming_object_history = row[object_slice].reshape(h, 3).astype(
            np.float32, copy=True
        )
        incoming_gripper_pos_history = row[gripper_pos_slice].reshape(h, 1)
        incoming_gripper_center_history = row[gripper_center_slice].reshape(h, 3)
        incoming_previous_action_history = row[previous_action_slice].reshape(h, 9)
        incoming_proxy_history = row[proxy_slice].reshape(h, 1).astype(
            np.float32, copy=True
        )

        incoming_object_current = incoming_object_history[-1]
        object_current, object_measurement_valid = self._latest_actual_object_pos_base(
            incoming_object_current
        )

        gripper_center_current = np.asarray(
            incoming_gripper_center_history[-1], dtype=np.float32
        ).reshape(3)
        if not np.all(np.isfinite(gripper_center_current)):
            raise RuntimeError(
                "Non-finite gripper_center_pos_base reached the high-level actor."
            )

        default_gripper_pos = float(
            getattr(
                self,
                "default_gripper_pos",
                self.cfg.get("default_gripper_pos", 0.0),
            )
        )
        gripper_joint_pos_train = (
            float(incoming_gripper_pos_history[-1, 0]) + default_gripper_pos
        )
        close_commanded = bool(incoming_previous_action_history[-1, 8] > 0.0)

        proxy, rising_edge, grasp_error = self._update_aligned_grasp_proxy(
            object_pos_base=object_current,
            object_measurement_valid=object_measurement_valid,
            gripper_center_pos_base=gripper_center_current,
            gripper_joint_pos_train=gripper_joint_pos_train,
            close_commanded=close_commanded,
        )

        if rising_edge and self._frozen_object_center_pos_base is None:
            # The distance-gated proxy cannot rise without a valid object
            # measurement, so object_current here is the actual measured object
            # base-frame vector from this same grasp=True HL step.
            self._frozen_object_center_pos_base = object_current.copy()
            print(
                "[OBS-ALIGN] grasp proxy TRUE | "
                f"grasp_err={grasp_error:.4f} m | "
                "freezing object_center_pos_base="
                + np.array2string(
                    self._frozen_object_center_pos_base,
                    precision=4,
                    floatmode="fixed",
                )
            )
            self._grasp_freeze_announced = True

        if self._frozen_object_center_pos_base is not None:
            object_for_actor = self._frozen_object_center_pos_base.copy()
        else:
            object_for_actor = object_current.copy()

        if self._aligned_object_history is None:
            self._aligned_object_history = incoming_object_history.copy()
            self._aligned_object_history[-1] = object_for_actor
        else:
            self._aligned_object_history = self._append_history(
                self._aligned_object_history,
                object_for_actor,
            )

        proxy_value = np.array([1.0 if proxy else 0.0], dtype=np.float32)
        if self._aligned_proxy_history is None:
            self._aligned_proxy_history = incoming_proxy_history.copy()
            self._aligned_proxy_history[-1] = proxy_value
        else:
            self._aligned_proxy_history = self._append_history(
                self._aligned_proxy_history,
                proxy_value,
            )

        row[object_slice] = self._aligned_object_history.reshape(-1).astype(
            row.dtype, copy=False
        )
        row[proxy_slice] = self._aligned_proxy_history.reshape(-1).astype(
            row.dtype, copy=False
        )

        # Keep the common controller's execution-side proxy synchronized with
        # the corrected actor semantics. In particular, the existing
        # stage2_force_gripper_close_enabled logic that runs immediately after
        # high-level inference will see this corrected value.
        self.grasp_confidence_proxy = bool(proxy)

        # Best-effort synchronization of the common controller's heuristic
        # counters prevents its legacy no-distance candidate from accumulating
        # hidden state between HL calls. These names exist in the validated
        # controller; guards keep this wrapper compatible with minor refactors.
        if hasattr(self, "grasp_proxy_enter_count"):
            self.grasp_proxy_enter_count = int(self._aligned_grasp_proxy_enter_count)
        if hasattr(self, "grasp_proxy_exit_count"):
            self.grasp_proxy_exit_count = int(self._aligned_grasp_proxy_exit_count)
        if hasattr(self, "prev_gripper_joint_pos"):
            self.prev_gripper_joint_pos = float(gripper_joint_pos_train)

        return work[0] if squeeze else work

    def prime_first_hl_and_ll(self):
        # A new policy run gets fresh proxy/history/freeze state. The common
        # mocap controller still owns the actual reset/startup sequence.
        self._reset_hl_alignment_state()
        self._ensure_high_policy_observation_adapter(required=True)
        return super().prime_first_hl_and_ll()

    def setup(self) -> None:
        print("=" * 108)
        print("UAN-TRAINED HIERARCHICAL RETRIEVAL -> REAL ROBOT DEPLOYMENT")
        print("UAN model         : NOT LOADED / NOT EXECUTED")
        print("Arm control       : WBC q_des -> Z1 firmware POSITION_PD -> real actuator")
        print(
            "Z1 gain alignment : "
            f"Kp={np.array2string(self._uan_deploy_kp_effective, precision=3)} | "
            f"Kd={np.array2string(self._uan_deploy_kd_effective, precision=3)}"
        )
        print(
            f"Policy hierarchy  : HL {1.0 / self.hl_control_dt:.0f} Hz -> "
            f"WBC {1.0 / self.control_dt:.0f} Hz"
        )
        print(
            "Arm target filter : NONE "
            "(no startup policy blend, no arm target rate limiter)"
        )
        print(
            f"Arm target resend : {self._arm_command_hz:g} Hz "
            "(hardware communication detail; not a deployed UAN frequency)"
        )
        print(
            f"Mocap root offset : {self.perception.root_offset_source()} | "
            f"{self._mocap_root_offset_resolved_path}"
        )
        print(
            "Grasp proxy       : SIM2SIM-ALIGNED | close + dist<0.10m + "
            "not-fully-closed + holding | 3/3 hysteresis"
        )
        print(
            "Object HL obs     : freeze object_center_pos_base on FIRST "
            "grasp=False->True transition; hold until next policy run"
        )
        print("=" * 108)

        super().setup()
        self._ensure_high_policy_observation_adapter(required=True)
        print(
            "[OBS-ALIGN] high-level ONNX adapter installed on "
            f"{self._hl_alignment_session_attr!r}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "B2WZ1 UAN-trained hierarchical retrieval sim2real "
            "with OptiTrack mocap sensing."
        )
    )
    parser.add_argument(
        "net",
        type=str,
        help="Unitree network interface, e.g. enxa0cec819e15f",
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="deploy/configs/b2wz1_hl_retrieval_mocap_UAN.yaml",
        help="Path to the UAN-trained mocap hierarchical sim2real YAML.",
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)

    controller = B2WZ1MocapRetrievalUANTrainedController(
        cfg_path=args.config,
        network_interface=args.net,
    )
    controller.run()


if __name__ == "__main__":
    main()