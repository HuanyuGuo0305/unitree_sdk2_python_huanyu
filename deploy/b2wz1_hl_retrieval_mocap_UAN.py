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

No arm target rate limiter is allowed. The 50-Hz WBC targets are held and
re-sent to the B2W and the Z1 every 2 ms (hardware_command_hz), exactly as
deploy/b2wz1_wbc_uan.py -- the low-level WBC test -- does. The gripper remains the original
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
import time
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

# object_freeze_mode -> what the actor's object_center_pos_base shows after a
# grasp. The keys are the valid modes.
_OBJECT_FREEZE_MODE_SUMMARY = {
    "off": "live measured object_center_pos_base",
    "while_proxy": "object value captured at the proxy rising edge, held while proxy=1",
    "Stage2_object_freeze": (
        "object value captured at the FIRST proxy rising edge, held until the "
        "next policy run"
    ),
    "overwrite_proxy": (
        "from the FIRST proxy rising edge: proxy held at 1 and "
        "object_center_pos_base := live gripper_center_pos_base, until the next "
        "policy run"
    ),
}


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
        self._configure_hardware_stream()

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

        # Lower bound on gripper closure, TRAINING coordinates
        # (open = gripper_open_pos ~ -pi/2, closed = 0).
        #
        # gripper_not_fully_closed is only an UPPER bound: it rejects "slammed
        # fully shut", but a WIDE-OPEN gripper passes it trivially. Without a
        # lower bound the proxy can therefore declare a grasp before the
        # fingers have moved at all -- observed latching at q=-1.5522 with the
        # gripper still on its open stop, which permanently froze the object
        # observation and latched Stage2_gripper_close onto nothing.
        #
        # Requiring q_train > this makes the term "PARTIALLY closed": the
        # fingers have travelled off the open stop, but have not shut on air.
        # Defaults to gripper_open_pos, i.e. no-op, so existing configs keep
        # their exact behaviour.
        self._gripper_open_pos_training = float(
            self.cfg.get("gripper_open_pos", -0.5 * np.pi)
        )
        self._gripper_min_closed_threshold = float(
            self.cfg.get(
                "gripper_min_closed_angle_threshold",
                self._gripper_open_pos_training,
            )
        )
        if not (
            self._gripper_open_pos_training - 1e-9
            <= self._gripper_min_closed_threshold
            < self._gripper_not_fully_closed_threshold
        ):
            raise ValueError(
                "gripper_min_closed_angle_threshold must satisfy "
                f"gripper_open_pos ({self._gripper_open_pos_training:.4f}) <= "
                f"value ({self._gripper_min_closed_threshold:.4f}) < "
                "gripper_not_fully_closed_angle_threshold "
                f"({self._gripper_not_fully_closed_threshold:.4f})."
            )
        self._grasp_proxy_enter_steps = int(self.cfg["grasp_proxy_enter_steps"])
        self._grasp_proxy_exit_steps = int(self.cfg["grasp_proxy_exit_steps"])

        # Terminal logging of the grasp proxy and every term feeding it.
        #   grasp_proxy_log          : master on/off
        #   grasp_proxy_log_period_s : heartbeat interval when nothing changes.
        #                              0 logs EVERY high-level step (10 Hz).
        # A line is always printed the moment any term, the candidate, the
        # proxy or a counter changes, so a transition is never missed between
        # heartbeats.
        self._grasp_log = bool(self.cfg.get("grasp_proxy_log", True))
        self._grasp_log_period_s = float(
            self.cfg.get("grasp_proxy_log_period_s", 1.0)
        )
        self._grasp_log_last_t = 0.0
        self._grasp_log_last_key = None

        # The wrapper's [GRASP] line is the authoritative one: it logs the
        # proxy that actually drives behaviour. Silence the base controller's
        # [GRASP-PROXY] transitions unless the YAML explicitly asks for them.
        self.log_base_grasp_proxy = bool(
            self.cfg.get("log_base_grasp_proxy", False)
        )

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

        # jointGripper runtime path.
        #
        # "dcmotor" reproduces the IsaacLab actuator used in retrieval training.
        # "position_pd" hands the gripper loop to the z1_ctrl firmware instead:
        # not the trained actuator model, but the loop closes at the controller
        # rate rather than at the Python send rate. That matters because the
        # DCMotor envelope's braking term is velocity-dependent and the gripper
        # reaches ~3.8 rad/s, so a slow send rate cannot sample it.
        #
        # This is permitted because the high-level gripper action is BINARY
        # open/close and grasp feedback reaches the policy through the grasp
        # proxy, not through gripper dynamics. The arm joints, which the policy
        # does control continuously, are unaffected by this setting.
        _grip_mode = str(cfg.get("z1_gripper_runtime_mode", "")).lower()
        if _grip_mode not in ("dcmotor", "position_pd"):
            raise ValueError(
                "z1_gripper_runtime_mode must be 'dcmotor' (the validated "
                "retrieval-training path) or 'position_pd' (firmware gripper "
                f"loop); got {_grip_mode!r}."
            )
        if _grip_mode != "dcmotor":
            print(
                "[UAN-AUDIT] jointGripper is NOT on the DCMotor path used in "
                f"retrieval training (z1_gripper_runtime_mode={_grip_mode!r}). "
                "Grasp dynamics will differ from training; the arm joints and "
                "the policy interface are unchanged."
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
        if not np.isfinite(proxy_threshold) or proxy_threshold <= 0.0:
            raise ValueError(
                "grasp_proxy_error_threshold must be a positive distance in m; "
                f"got {proxy_threshold}."
            )
        if abs(proxy_threshold - _EXPECTED_GRASP_PROXY_ERROR_THRESHOLD) > 1e-12:
            print(
                "[UAN-AUDIT] grasp_proxy_error_threshold="
                f"{proxy_threshold:.4f} m deviates from the training/sim2sim "
                f"value of {_EXPECTED_GRASP_PROXY_ERROR_THRESHOLD:.4f} m. The "
                "grasp proxy is an actor OBSERVATION, so the policy sees a "
                "signal calibrated differently from the one it trained "
                "against; a looser gate also latches on near-misses."
            )

        not_closed_threshold = float(
            cfg.get("gripper_not_fully_closed_angle_threshold", np.nan)
        )
        if abs(
            not_closed_threshold - _EXPECTED_GRIPPER_NOT_FULLY_CLOSED_THRESHOLD
        ) > 1e-12:
            raise ValueError(
                "gripper_not_fully_closed_angle_threshold must remain -5 deg "
                "to match training/sim2sim."
            )
        # gripper_angle_hold_threshold is intentionally NOT audited: the
        # gripper_angle_holding term it configures is not evaluated by this
        # deployment's proxy, so pinning its value would assert an alignment
        # that no longer exists.
        # The debounce lengths are editable. Training/sim2sim used 3 HL steps
        # for both, so any other value is announced rather than refused. Zero
        # is refused: the counters test `count >= steps`, so 0 would latch the
        # proxy on a step with no candidate at all.
        for key, expected_steps in (
            ("grasp_proxy_enter_steps", _EXPECTED_GRASP_PROXY_ENTER_STEPS),
            ("grasp_proxy_exit_steps", _EXPECTED_GRASP_PROXY_EXIT_STEPS),
        ):
            value = cfg.get(key)
            try:
                steps = int(value)
                valid = (
                    not isinstance(value, bool)
                    and steps == value
                    and steps >= 1
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(
                    f"{key} must be an integer >= 1 (HL steps); got {value!r}."
                )
            if steps != expected_steps:
                print(
                    f"[UAN-AUDIT] {key}={steps} deviates from the "
                    f"training/sim2sim value of {expected_steps} HL steps "
                    f"({expected_steps * self.hl_control_dt:.2f} s). The grasp "
                    "proxy is an actor OBSERVATION, so the policy sees it latch "
                    "or release on a different delay than it trained against."
                )

        # -------------------------------------------------------------
        # 8. Bug fix #3: freeze object_center_pos_base at first grasp=True.
        # -------------------------------------------------------------
        # Object-position freeze mode. Mirrors gripper_force_close_mode.
        #
        #   "off"
        #       The actor always sees the live measured object_center_pos_base.
        #
        #   "while_proxy"
        #       Hold the value captured at the proxy's rising edge WHILE the
        #       proxy stays true, and revert to the live measurement if the
        #       proxy falls back to false. The capture is refreshed on each new
        #       rising edge.
        #
        #   "Stage2_object_freeze"
        #       Freeze on the FIRST rising edge and hold that value for the
        #       rest of the run, regardless of what the proxy does afterwards.
        #       Cleared only by a full policy reset.
        #
        #   "overwrite_proxy"
        #       On the FIRST rising edge, latch grasp_confidence_proxy at 1 and
        #       replace object_center_pos_base with the live
        #       gripper_center_pos_base, both for the rest of the run. Cleared
        #       only by a full policy reset.
        #
        # Default is derived from the legacy boolean so existing YAMLs keep
        # their exact behaviour without naming the new key.
        self.valid_object_freeze_modes = tuple(_OBJECT_FREEZE_MODE_SUMMARY)
        _legacy_freeze = bool(
            cfg.get("freeze_object_center_pos_base_on_grasp", False)
        )
        self.object_freeze_mode = str(
            cfg.get(
                "object_freeze_mode",
                "Stage2_object_freeze" if _legacy_freeze else "off",
            )
        ).strip()
        if self.object_freeze_mode not in self.valid_object_freeze_modes:
            raise ValueError(
                "Invalid object_freeze_mode="
                f"{self.object_freeze_mode!r}. Expected one of "
                f"{list(self.valid_object_freeze_modes)}."
            )
        print(f"[OBS-ALIGN] object_freeze_mode = {self.object_freeze_mode}")

        if self.object_freeze_mode != "Stage2_object_freeze":
            # Previously this deployment hard-required the latched freeze. It
            # is now an explicit mode, so a different choice is allowed but
            # never silent: training froze the object vector at the first
            # grasp=True step, and the actor observes that vector directly.
            print(
                "[UAN-AUDIT] object_freeze_mode="
                f"{self.object_freeze_mode!r} differs from the "
                "'Stage2_object_freeze' behaviour used in training, where the "
                "actor holds the object base-frame vector captured on the "
                "first grasp=True high-level step."
            )
        if self.object_freeze_mode == "overwrite_proxy":
            print(
                "[UAN-AUDIT] overwrite_proxy holds grasp_confidence_proxy at 1 "
                "after the first grasp, so grasp_proxy_exit_steps is unused and "
                f"gripper_force_close_mode={self.gripper_force_close_mode!r} "
                "sees a proxy that never falls for the rest of the run."
            )
            print(
                "[UAN-AUDIT] overwrite_proxy never ends the run on object "
                "tracking loss: after the first grasp the object is not needed; "
                "before it, the safe hold waits for the object with no damping "
                "timeout. Retrieval loss still times out after "
                f"{self.perception_fault_timeout_s:.2f}s."
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
        self._object_only_invalid = False
        self._perception_hold_exempt = False

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
        grasp_error = float(
            np.linalg.norm(
                np.asarray(object_pos_base, dtype=np.float64)
                - np.asarray(gripper_center_pos_base, dtype=np.float64)
            )
        )

        # DEPLOYMENT CONDITION (three terms + validity, over N consecutive
        # high-level steps):
        #
        #   1. close_commanded          previous HL action[8] > 0
        #   2. grasp_error              < grasp_proxy_error_threshold
        #   3. gripper_not_fully_closed q_train < gripper_not_fully_closed_angle_threshold
        #                               i.e. the fingers did NOT close all the
        #                               way, so something is between them
        #   4. held for grasp_proxy_enter_steps consecutive HL steps
        #                               (the hysteresis below)
        #
        # object_measurement_valid is a data-validity guard rather than a
        # condition: grasp_error is meaningless without a live object fix.
        #
        # NOTE vs training: sim2sim additionally requires
        #   gripper_angle_holding : |dq_train| < gripper_angle_hold_threshold
        # ("the fingers have stopped moving"). That term is deliberately NOT
        # evaluated here, so this proxy is easier to satisfy than the one the
        # policy trained against. gripper_angle_hold_threshold is consequently
        # unused by this deployment.
        gripper_not_fully_closed = (
            float(gripper_joint_pos_train)
            < self._gripper_not_fully_closed_threshold
        )
        gripper_has_closed = (
            float(gripper_joint_pos_train)
            > self._gripper_min_closed_threshold
        )
        gripper_partially_closed = bool(
            gripper_not_fully_closed and gripper_has_closed
        )

        proxy_candidate = bool(
            close_commanded
            and object_measurement_valid
            and grasp_error < self._grasp_proxy_error_threshold
            and gripper_partially_closed
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
        elif self._proxy_latched_for_run:
            # overwrite_proxy: held at 1 for the rest of the run, so the exit
            # hysteresis never runs.
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

        rising_edge = (not was_proxy) and bool(self._aligned_grasp_proxy)

        self._log_grasp_proxy(
            proxy=bool(self._aligned_grasp_proxy),
            candidate=bool(proxy_candidate),
            close_commanded=bool(close_commanded),
            object_valid=bool(object_measurement_valid),
            grasp_error=grasp_error,
            gripper_q=float(gripper_joint_pos_train),
            not_fully_closed=bool(gripper_not_fully_closed),
            has_closed=bool(gripper_has_closed),
            rising_edge=rising_edge,
        )

        return bool(self._aligned_grasp_proxy), rising_edge, grasp_error

    def _log_grasp_proxy(
        self,
        *,
        proxy: bool,
        candidate: bool,
        close_commanded: bool,
        object_valid: bool,
        grasp_error: float,
        gripper_q: float,
        not_fully_closed: bool,
        has_closed: bool,
        rising_edge: bool,
    ) -> None:
        """One terminal line with the proxy and every term that produced it.

        Printed whenever anything changes, and otherwise at a heartbeat, so a
        transition is never lost between heartbeats and a static state still
        shows its current values.
        """
        if not self._grasp_log:
            return

        err_ok = grasp_error < self._grasp_proxy_error_threshold

        # Only the booleans and the counters gate "did something change";
        # the floats move every step and would defeat the throttle.
        key = (
            proxy, candidate, close_commanded, object_valid,
            err_ok, not_fully_closed, has_closed,
            self._aligned_grasp_proxy_enter_count,
            self._aligned_grasp_proxy_exit_count,
        )
        now = time.monotonic()
        changed = key != self._grasp_log_last_key
        due = (now - self._grasp_log_last_t) >= self._grasp_log_period_s
        if not (changed or due or rising_edge):
            return
        self._grasp_log_last_key = key
        self._grasp_log_last_t = now

        def mark(ok: bool) -> str:
            return "PASS" if ok else "fail"

        exit_text = (
            "exit=LATCHED"
            if self._proxy_latched_for_run
            else (
                f"exit={self._aligned_grasp_proxy_exit_count}/"
                f"{self._grasp_proxy_exit_steps}"
            )
        )

        print(
            "[GRASP] "
            f"proxy={int(proxy)} cand={int(candidate)} | "
            f"1.close={int(close_commanded)} {mark(close_commanded)} | "
            f"2.err={grasp_error:6.3f}<{self._grasp_proxy_error_threshold:.3f} "
            f"{mark(err_ok)} | "
            f"3.grip_q={gripper_q:+7.4f} in "
            f"({self._gripper_min_closed_threshold:+.3f},"
            f"{self._gripper_not_fully_closed_threshold:+.3f}) "
            f"closed={mark(has_closed)} notshut={mark(not_fully_closed)} | "
            f"4.enter={self._aligned_grasp_proxy_enter_count}/"
            f"{self._grasp_proxy_enter_steps} "
            f"{exit_text} | "
            f"obj_valid={int(object_valid)}"
            + ("   *** PROXY RISING EDGE ***" if rising_edge else "")
        )

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

        # Capture on a rising edge. "Stage2_object_freeze" keeps the FIRST
        # capture for the whole run; "while_proxy" re-captures on every rising
        # edge and releases when the proxy drops. "overwrite_proxy" captures
        # nothing; it switches to the gripper center below.
        if (
            self.object_freeze_mode in ("while_proxy", "Stage2_object_freeze")
            and rising_edge
        ):
            first_capture = self._frozen_object_center_pos_base is None
            if first_capture or self.object_freeze_mode == "while_proxy":
                # The distance-gated proxy cannot rise without a valid object
                # measurement, so object_current here is the actual measured
                # object base-frame vector from this same grasp=True HL step.
                self._frozen_object_center_pos_base = object_current.copy()
                print(
                    "[OBS-ALIGN] grasp proxy TRUE | "
                    f"mode={self.object_freeze_mode} | "
                    f"grasp_err={grasp_error:.4f} m | "
                    "freezing object_center_pos_base="
                    + np.array2string(
                        self._frozen_object_center_pos_base,
                        precision=4,
                        floatmode="fixed",
                    )
                )
                self._grasp_freeze_announced = True

        if self.object_freeze_mode == "overwrite_proxy" and rising_edge:
            # The latch makes this rising edge happen once per run.
            print(
                "[OBS-ALIGN] grasp proxy TRUE | mode=overwrite_proxy | "
                f"grasp_err={grasp_error:.4f} m | proxy LATCHED to 1 and "
                "object_center_pos_base := gripper_center_pos_base for the "
                "rest of the run (now "
                + np.array2string(
                    gripper_center_current,
                    precision=4,
                    floatmode="fixed",
                )
                + ")"
            )

        if self.object_freeze_mode == "off":
            object_for_actor = object_current.copy()
        elif self.object_freeze_mode == "overwrite_proxy":
            # Latched: the object term IS the gripper center, taken from this
            # frame's gripper_center_pos_base feature so both actor inputs
            # agree exactly. Before the first grasp it is the live measurement,
            # as in "off".
            if proxy:
                object_for_actor = gripper_center_current.copy()
            else:
                object_for_actor = object_current.copy()
        elif self.object_freeze_mode == "while_proxy":
            # Released the moment the proxy drops: the actor goes back to the
            # live measurement rather than steering at a stale point.
            if proxy and self._frozen_object_center_pos_base is not None:
                object_for_actor = self._frozen_object_center_pos_base.copy()
            else:
                if not proxy and self._frozen_object_center_pos_base is not None:
                    print(
                        "[OBS-ALIGN] proxy FALSE | mode=while_proxy | "
                        "releasing object_center_pos_base back to live mocap."
                    )
                    self._frozen_object_center_pos_base = None
                    self._grasp_freeze_announced = False
                object_for_actor = object_current.copy()
        else:  # "Stage2_object_freeze"
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

    @property
    def _proxy_latched_for_run(self) -> bool:
        """True once overwrite_proxy has latched the proxy for this run."""
        return (
            self.object_freeze_mode == "overwrite_proxy"
            and bool(self._aligned_grasp_proxy)
        )

    def update_grasp_confidence_proxy(self, perception_snap: dict) -> None:
        # The base heuristic runs at every HL boundary BEFORE the observation
        # adapter and knows nothing about the overwrite_proxy latch, so it can
        # clear the proxy for that step. resolve_effective_task_state() would
        # then demand a live object fix again -- which an object held in the
        # gripper tends to lose -- and a perception hold would also stop the
        # adapter from ever restoring the latch.
        super().update_grasp_confidence_proxy(perception_snap)
        if self._proxy_latched_for_run:
            self.grasp_confidence_proxy = True
            self.grasp_proxy_exit_count = 0

    def resolve_effective_task_state(self, perception_snap: dict) -> dict:
        task = super().resolve_effective_task_state(perception_snap)
        # Read by _begin_perception_hold(), which the step loop calls with
        # this same task.
        self._object_only_invalid = bool(
            not task["object_valid"] and task["retrieval_valid"]
        )
        return task

    def _begin_perception_hold(self, reason: str) -> None:
        was_active = bool(self.perception_hold_active)
        super()._begin_perception_hold(reason)

        # overwrite_proxy: losing the object must never end the run. Once the
        # proxy has latched, the base already treats the object as valid (it
        # is the gripper center), so this only matters BEFORE the first grasp.
        # There is no honest object value for the actor then, so the safe hold
        # still applies -- but it waits for the object instead of running the
        # damping timer. Any other invalid channel still times out.
        exempt = (
            self.object_freeze_mode == "overwrite_proxy"
            and self._object_only_invalid
        )
        if exempt:
            self.perception_invalid_since = None
            if not was_active or not self._perception_hold_exempt:
                print(
                    "[PERCEPTION-HOLD] overwrite_proxy | object untracked before "
                    "the first grasp: holding until it is tracked again; the "
                    f"{self.perception_fault_timeout_s:.2f}s damping timeout is "
                    "suspended."
                )
        elif was_active and self._perception_hold_exempt:
            print(
                "[PERCEPTION-HOLD] overwrite_proxy | not only the object is "
                f"invalid now ({reason}): the "
                f"{self.perception_fault_timeout_s:.2f}s damping timeout is running."
            )
        self._perception_hold_exempt = exempt

    # ------------------------------------------------------------------
    # Hardware command stream -- identical to b2wz1_wbc_uan.py
    # ------------------------------------------------------------------
    #
    # b2wz1_wbc_uan.py, the low-level WBC test, runs ONE loop at control_hz
    # (500 Hz). Every 10th tick it reads the robot and runs the WBC; EVERY tick
    # it sends
    #
    #     b2w.send(leg_target, wheel_cmd)
    #     z1.track_target_pd_once(arm_target, ..., use_startup_gains=False)
    #
    # so both robots get the 50 Hz WBC targets re-sent every 2 ms. The UAN the
    # WBC was trained against was collected the same way (loop_hz 500,
    # target_update_hz 50). The inherited hierarchical loop sent each target
    # once per 20 ms instead. Here tick 0 is the inherited policy step (read ->
    # HL -> WBC -> send) and ticks 1..N-1 re-send what it produced, on the same
    # 2 ms grid.

    def _configure_hardware_stream(self) -> None:
        policy_hz = 1.0 / float(self.control_dt)
        self._hw_rate_hz = float(self.cfg.get("hardware_command_hz", 500.0))
        ratio = self._hw_rate_hz / policy_hz
        if ratio < 1.0 - 1e-9 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                f"hardware_command_hz={self._hw_rate_hz:g} must be an integer "
                f"multiple of the {policy_hz:g} Hz WBC rate "
                "(b2wz1_wbc_uan.py uses control_hz 500 / policy_hz 50)."
            )
        self._hw_ticks_per_step = int(round(ratio))
        self._hw_dt = 1.0 / self._hw_rate_hz
        self._hw_settle_s = float(self.cfg.get("z1_runtime_settle_s", 0.5))

        if self._hw_ticks_per_step > 1 and self._arm_thread_enabled:
            raise ValueError(
                "arm_command_hz enables the separate Z1 arm thread, which cannot "
                "run alongside hardware_command_hz: both would own Z1 comms. Set "
                "arm_command_hz to the policy rate (50); hardware_command_hz "
                f"already re-sends the arm target at {self._hw_rate_hz:g} Hz in "
                "lockstep with the WBC, as b2wz1_wbc_uan.py does."
            )

        self._hw_last_arm_send = 0.0
        self._hw_send_count = 0
        self._hw_window_start = 0.0

    def send_policy_targets(self) -> None:
        super().send_policy_targets()
        self._hw_note_arm_send(time.perf_counter())

    def step_after_prime(self):
        block_start = time.perf_counter()
        ok, reason = super().step_after_prime()
        if ok:
            self._hw_hold_targets_for_block(block_start)
        return ok, reason

    def initialize_policy_state_and_history(self):
        self._hw_runtime_gain_settle()
        return super().initialize_policy_state_and_history()

    def _hw_send_held_targets(self) -> None:
        # The same two sends as tick 0 (sim2real send_policy_targets). low_cmd
        # still holds the RL leg/wheel command it wrote, and arm_target /
        # gripper_target still hold the WBC / HL outputs of this step.
        self.send_b2w_cmd()
        self.z1.track_target_pd_runtime_once(
            q_target=self.arm_target.copy(),
            gripper_q_target_training=float(self.gripper_target),
            use_startup_gains=False,
        )
        self._hw_note_arm_send(time.perf_counter())

    def _hw_hold_targets_for_block(self, block_start: float) -> None:
        for tick in range(1, self._hw_ticks_per_step):
            deadline = block_start + tick * self._hw_dt
            now = time.perf_counter()
            if now < deadline:
                time.sleep(deadline - now)
            elif now >= deadline + self._hw_dt:
                # A whole tick late after a slow policy step: skip it rather
                # than burst identical packets. run() still ends the block.
                continue
            self._hw_send_held_targets()

    def _hw_note_arm_send(self, t: float) -> None:
        # Fills the inherited [HEALTH]/[TRACE] arm-rate and send-gap fields,
        # which otherwise only the (disabled) arm thread writes.
        if self._hw_last_arm_send > 0.0:
            gap = t - self._hw_last_arm_send
            self._arm_max_stall_s = max(self._arm_max_stall_s, gap)
            self._window_arm_stall = max(self._window_arm_stall, gap)
        self._hw_last_arm_send = t

        if self._hw_window_start == 0.0:
            self._hw_window_start = t
            self._hw_send_count = 0
        self._hw_send_count += 1
        elapsed = t - self._hw_window_start
        if elapsed >= 1.0:
            self._arm_achieved_hz = self._hw_send_count / elapsed
            self._hw_window_start = t
            self._hw_send_count = 0

    def _hw_runtime_gain_settle(self) -> None:
        """The last startup block of b2wz1_wbc_uan.py, before its policy starts.

        Hold DEFAULT for z1_runtime_settle_s on the RUNTIME gains -- B2W RL
        gains, Z1 runtime firmware gains -- at the hardware rate. The inherited
        startup went straight from the stiff startup-gain hold to the first WBC
        step, so the WBC's first observations saw the arm mid gain-switch.
        """
        n = int(round(self._hw_settle_s * self._hw_rate_hz))
        if n <= 0:
            return
        print(
            f"[HW-RATE] settling {self._hw_settle_s:.2f}s at DEFAULT on runtime "
            f"gains ({n} ticks at {self._hw_rate_hz:g} Hz) before the first WBC step."
        )
        t0 = time.perf_counter()
        for i in range(n):
            self._write_b2w_pose_cmd_policy(
                self.default_b2w_pos_policy,
                use_pd_gains=False,
            )
            self.send_b2w_cmd()
            self.z1.track_target_pd_runtime_once(
                q_target=self.default_arm_pos.copy(),
                gripper_q_target_training=float(self.gripper_open_pos),
                use_startup_gains=False,
            )
            deadline = t0 + (i + 1) * self._hw_dt
            now = time.perf_counter()
            if now < deadline:
                time.sleep(deadline - now)

    def prime_first_hl_and_ll(self):
        # A new policy run gets fresh proxy/history/freeze state. The common
        # mocap controller still owns the actual reset/startup sequence.
        self._reset_hl_alignment_state()
        self._ensure_high_policy_observation_adapter(required=True)
        block_start = time.perf_counter()
        result = super().prime_first_hl_and_ll()
        # The primed step is the first 20 ms block, so it gets its re-sends too.
        self._hw_hold_targets_for_block(block_start)
        # run() waits one control_dt before its first step. That one-off gap is
        # startup, not a send stall, so start the rate/gap statistics after it.
        self._hw_last_arm_send = 0.0
        self._hw_window_start = 0.0
        print(
            f"[HW-RATE] main loop re-sends B2W + Z1 lowcmd every "
            f"{self._hw_dt * 1e3:.0f} ms ({self._hw_ticks_per_step} sends per "
            f"{self.control_dt * 1e3:.0f} ms WBC step), as b2wz1_wbc_uan.py does."
        )
        return result

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
            f"Hardware stream   : B2W + Z1 lowcmd at {self._hw_rate_hz:g} Hz, "
            f"{self._hw_ticks_per_step} sends per "
            f"{self.control_dt * 1e3:.0f} ms WBC step (as b2wz1_wbc_uan.py)"
        )
        print(
            f"Policy hand-off   : {self._hw_settle_s:.2f}s DEFAULT settle on RL "
            "leg gains + Z1 runtime gains before the first WBC step"
        )
        print(
            f"Mocap root offset : {self.perception.root_offset_source()} | "
            f"{self._mocap_root_offset_resolved_path}"
        )
        exit_steps = (
            "latched"
            if self.object_freeze_mode == "overwrite_proxy"
            else str(self._grasp_proxy_exit_steps)
        )
        print(
            "Grasp proxy       : close + "
            f"dist<{self._grasp_proxy_error_threshold:.2f}m + partially closed | "
            f"enter {self._grasp_proxy_enter_steps} / exit {exit_steps} HL steps"
        )
        print(
            f"Object HL obs     : {self.object_freeze_mode} -> "
            f"{_OBJECT_FREEZE_MODE_SUMMARY[self.object_freeze_mode]}"
        )
        if self.object_freeze_mode == "overwrite_proxy":
            print(
                "Object tracking   : loss never ends the run (before first "
                "grasp: safe hold until tracked again; after: not needed)"
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