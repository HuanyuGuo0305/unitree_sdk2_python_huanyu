#!/usr/bin/env python3
"""
B2WZ1 hierarchical retrieval sim2real -- MOCAP sensing -- ABS HL arm action.

This deployment is intentionally built on top of the previously validated
delta-action mocap/UAN real-robot wrapper:

    b2wz1_hl_retrieval_mocap_UAN.py
        -> b2wz1_hl_retrieval_mocap.py
        -> b2wz1_hl_retrieval_sim2real.py

The inherited pipeline keeps the validated mocap sensing, safety/protection,
startup transitions, 50-Hz frozen WBC, 200-Hz Z1 target resend, external
gripper DCMotor path and observation construction.

CRITICAL REAL-ROBOT CONTROL SEMANTICS
-------------------------------------
UAN is SIMULATION-SIDE ONLY and is NOT loaded here.

Training:
    HL ABS policy @ 10 Hz
      -> frozen WBC @ 50 Hz
      -> simulated Z1 nominal PD + frozen UAN residual @ 250 Hz
      -> PhysX

Real:
    HL ABS policy @ 10 Hz
      -> frozen WBC @ 50 Hz
      -> q_des
      -> REAL Z1 firmware position-PD
      -> hardware

The real arm receives the raw Unitree firmware gains used to collect the UAN
data. The Python deployment never multiplies those raw gains by the protocol
scales before sending them.

ABS HIGH-LEVEL ACTION
---------------------
The ONNX policy output is 9-D:

    action[0:3] : normalized base command
    action[3]   : ABS kp0_x in configured range
    action[4]   : ABS kp0_y in configured range
    action[5]   : ABS kp0_z in configured range
    action[6]   : ABS policy yaw in configured range
    action[7]   : ABS policy pitch in configured range
    action[8]   : binary gripper (>0 close, <=0 open)

There is no policy-level "current measured EE pose + delta" semantics.

Compatibility note
------------------
The validated common real-robot controller underneath this file still contains
the old measured-EE-relative delta decoder. To avoid duplicating thousands of
lines of safety-critical deployment code, this wrapper inserts an algebraic
compatibility adapter immediately after high-level ONNX inference:

    desired_abs = map(raw_ABS_action)
    legacy_delta = (desired_abs - measured_EE) / compatibility_scale

The inherited decoder then reconstructs

    measured_EE + compatibility_scale * legacy_delta == desired_abs

within the same HL step. The actor observation is separately repaired so its
previous_hl_action contains the RAW ABS policy action, never this internal
compatibility action. The bridge fails loudly if any compatibility action
would exceed [-1, 1].

Z1 END-EFFECTOR PITCH CONVENTION -- IMPORTANT
----------------------------------------------
The Z1 gripper/end-effector local +X axis points from the wrist/stator toward
the fingertips. The MuJoCo sim2sim bug was caused by feeding policy pitch
directly into standard XYZ Euler geometry: negative policy pitch made local +X
point UP instead of DOWN.

This real deployment therefore enforces:

    pitch_geom = ee_pitch_to_euler_sign * pitch_policy
               = -pitch_policy

with ee_pitch_to_euler_sign == -1.0.

The inherited delta decoder receives GEOMETRIC pitch, so its ee_pitch_range and
neutral_ee_pitch in YAML are geometric (+) values. The actual policy range is
stored separately as hl_abs_ee_pitch_range = [-70 deg, 0].

Run from the unitree_sdk2_python_huanyu repository root:

    python3 deploy/b2wz1_hl_retrieval_abs_uan.py \
        enxa0cec819e15f \
        deploy/configs/b2wz1_hl_retrieval_abs_uan.yaml
"""

from __future__ import annotations

import argparse
import inspect
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

# Reuse the validated UAN-trained mocap real-robot wrapper. It already enforces:
#   - NO real-robot UAN execution
#   - exact raw firmware gains / protocol scaling audit
#   - 50-Hz WBC / 10-Hz HL interface
#   - trusted mocap root frame
#   - deployable grasp proxy
#   - first-grasp object observation freeze
#   - external gripper DCMotor semantics
from b2wz1_hl_retrieval_mocap_UAN import (
    B2WZ1MocapRetrievalUANTrainedController,
)


# High-level actor frame:
#   root_ang_vel_b(3), projected_gravity_b(3), leg_pos_rel(12),
#   arm_pos_rel(6), gripper_pos_rel(1), arm_joint_vel(6),
#   object_center_pos_base(3), gripper_orientation_base(6),
#   gripper_center_pos_base(3), retrieval_target_pos_base(3),
#   previous_hl_action(9), grasp_confidence_proxy(1) = 56.
_HL_FEATURE_DIMS = (3, 3, 12, 6, 1, 6, 3, 6, 3, 3, 9, 1)
_HL_PREVIOUS_ACTION_FEATURE = 10


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


def _map_normalized_to_range(x: float, limits: np.ndarray) -> float:
    low, high = float(limits[0]), float(limits[1])
    if not high > low:
        raise ValueError(f"Invalid range [{low}, {high}].")
    return low + 0.5 * (float(x) + 1.0) * (high - low)


def _map_range_to_normalized(value: float, limits: np.ndarray) -> float:
    low, high = float(limits[0]), float(limits[1])
    if not high > low:
        raise ValueError(f"Invalid range [{low}, {high}].")
    return 2.0 * (float(value) - low) / (high - low) - 1.0


class _ABSHighPolicyActionAdapterSession:
    """Wrap the already-aligned 168-D high-level ONNX session.

    Input side:
      The inherited UAN/mocap observation adapter runs first. The owner class
      additionally repairs previous_hl_action so the actor sees ABS semantics.

    Output side:
      The raw ABS 9-D actor output is saved for the next observation, then
      converted into an internal compatibility action for the inherited legacy
      delta decoder. Base and gripper components are passed through unchanged.
    """

    def __init__(
        self,
        owner: "B2WZ1MocapRetrievalABSUANController",
        wrapped_session: Any,
    ) -> None:
        self._owner = owner
        self._wrapped = wrapped_session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    @staticmethod
    def _find_action_output(outputs: Any) -> tuple[int, np.ndarray]:
        if not isinstance(outputs, (list, tuple)):
            raise RuntimeError(
                "High-level ONNX session must return a list/tuple of outputs; "
                f"got {type(outputs).__name__}."
            )
        for i, value in enumerate(outputs):
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[-1] == 9:
                return i, arr
        raise RuntimeError(
            "Could not find the 9-D high-level action in ONNX outputs."
        )

    def run(self, output_names, input_feed, *args, **kwargs):
        outputs = self._wrapped.run(output_names, input_feed, *args, **kwargs)
        out_index, out_array = self._find_action_output(outputs)

        if out_array.ndim == 1:
            raw_abs = out_array.reshape(9).astype(np.float32, copy=True)
            batch_shape = "vector"
        elif out_array.ndim == 2 and out_array.shape[0] == 1:
            raw_abs = out_array[0].reshape(9).astype(np.float32, copy=True)
            batch_shape = "batch1"
        else:
            raise RuntimeError(
                "Real-robot high-level policy expects batch size 1; "
                f"got output shape {out_array.shape}."
            )

        if not np.all(np.isfinite(raw_abs)):
            raise RuntimeError(f"Non-finite ABS high-level action: {raw_abs}")

        raw_abs = np.clip(raw_abs, -1.0, 1.0).astype(np.float32)
        self._owner._record_raw_abs_hl_action(raw_abs)

        bridge_action = self._owner._abs_action_to_legacy_delta_bridge(raw_abs)

        replacement = out_array.copy()
        if batch_shape == "vector":
            replacement[...] = bridge_action.astype(replacement.dtype, copy=False)
        else:
            replacement[0, ...] = bridge_action.astype(
                replacement.dtype, copy=False
            )

        # ONNX Runtime returns a list, but preserve tuple/list shape defensively.
        mutable = list(outputs)
        mutable[out_index] = replacement
        return tuple(mutable) if isinstance(outputs, tuple) else mutable


class B2WZ1MocapRetrievalABSUANController(
    B2WZ1MocapRetrievalUANTrainedController
):
    """Mocap sim2real wrapper for the ABS high-level retrieval policy."""

    def __init__(self, cfg_path: str, network_interface: str) -> None:
        super().__init__(
            cfg_path=cfg_path,
            network_interface=network_interface,
        )

        self._audit_abs_policy_semantics()
        self._configure_abs_policy_semantics()
        self._reset_abs_policy_alignment_state()

        self._abs_action_session: Optional[_ABSHighPolicyActionAdapterSession] = None
        self._abs_action_session_attr: Optional[str] = None

        self._actual_ee_pose_getter_name = self._find_actual_ee_pose_getter_name()
        if self._actual_ee_pose_getter_name is None:
            raise RuntimeError(
                "Could not locate the inherited zero-argument PLB EE-pose getter. "
                "The ABS compatibility bridge cannot safely cancel the legacy "
                "measured-EE delta decoder."
            )
        self._actual_ee_pose_getter_original = getattr(
            self, self._actual_ee_pose_getter_name
        )
        self._abs_cached_pose_armed = False

        # Sessions normally already exist after the inherited constructor.
        self._ensure_abs_high_policy_action_adapter(required=False)

    # ------------------------------------------------------------------
    # ABS semantics / audits
    # ------------------------------------------------------------------

    @staticmethod
    def _as_range(cfg: dict, key: str) -> np.ndarray:
        value = np.asarray(cfg[key], dtype=np.float64).reshape(-1)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{key} must contain two finite values.")
        if not value[1] > value[0]:
            raise ValueError(f"{key} must satisfy high > low; got {value}.")
        return value

    def _audit_abs_policy_semantics(self) -> None:
        cfg = self.cfg

        mode = str(cfg.get("high_level_arm_action_mode", "")).strip().lower()
        if mode != "absolute":
            raise ValueError(
                'This controller requires high_level_arm_action_mode: "absolute".'
            )

        sign = float(cfg.get("ee_pitch_to_euler_sign", np.nan))
        if not np.isfinite(sign) or abs(sign + 1.0) > 1e-12:
            raise ValueError(
                "Z1 EE pitch mapping must be ee_pitch_to_euler_sign: -1.0. "
                "Negative policy pitch must tilt local +X (toward fingertips) DOWN."
            )

        x_range = self._as_range(cfg, "kp0_x_range")
        y_range = self._as_range(cfg, "kp0_y_range")
        z_range = self._as_range(cfg, "kp0_z_range")
        yaw_range = self._as_range(cfg, "ee_yaw_range")
        policy_pitch_range = self._as_range(cfg, "hl_abs_ee_pitch_range")
        geom_pitch_range = self._as_range(cfg, "ee_pitch_range")

        expected_geom_pitch_range = np.sort(sign * policy_pitch_range)
        if not np.allclose(
            geom_pitch_range, expected_geom_pitch_range, atol=1e-12, rtol=0.0
        ):
            raise ValueError(
                "Inherited decoder ee_pitch_range must be the GEOMETRIC range "
                "obtained after pitch_geom = -pitch_policy.\n"
                f"  policy range    = {policy_pitch_range}\n"
                f"  expected geom   = {expected_geom_pitch_range}\n"
                f"  configured geom = {geom_pitch_range}"
            )

        neutral_kp0 = np.asarray(cfg["neutral_kp0"], dtype=np.float64).reshape(3)
        for name, value, limits in (
            ("neutral_kp0.x", neutral_kp0[0], x_range),
            ("neutral_kp0.y", neutral_kp0[1], y_range),
            ("neutral_kp0.z", neutral_kp0[2], z_range),
            ("neutral_ee_yaw", float(cfg["neutral_ee_yaw"]), yaw_range),
            (
                "hl_abs_neutral_ee_pitch",
                float(cfg["hl_abs_neutral_ee_pitch"]),
                policy_pitch_range,
            ),
        ):
            if value < limits[0] - 1e-12 or value > limits[1] + 1e-12:
                raise ValueError(f"{name}={value} lies outside {limits}.")

        expected_geom_neutral_pitch = sign * float(
            cfg["hl_abs_neutral_ee_pitch"]
        )
        if abs(float(cfg["neutral_ee_pitch"]) - expected_geom_neutral_pitch) > 1e-12:
            raise ValueError(
                "neutral_ee_pitch is consumed by the inherited GEOMETRIC decoder "
                "and must equal -hl_abs_neutral_ee_pitch.\n"
                f"  expected={expected_geom_neutral_pitch}, "
                f"configured={cfg['neutral_ee_pitch']}"
            )

        # The inherited controller may apply additional per-step delta clips.
        # They MUST be disabled because the bridge already guarantees [-1,1].
        kp_clip = np.asarray(
            cfg.get("kp0_delta_clip", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        if np.any(np.abs(kp_clip) > 1e-12):
            raise ValueError(
                "kp0_delta_clip must be [0,0,0] for the ABS compatibility bridge."
            )
        if abs(float(cfg.get("ee_yaw_delta_clip", 0.0))) > 1e-12:
            raise ValueError(
                "ee_yaw_delta_clip must be 0 for the ABS compatibility bridge."
            )
        if abs(float(cfg.get("ee_pitch_delta_clip", 0.0))) > 1e-12:
            raise ValueError(
                "ee_pitch_delta_clip must be 0 for the ABS compatibility bridge."
            )

        bridge_pos = np.asarray(cfg["kp0_delta_scale"], dtype=np.float64).reshape(3)
        bridge_yaw = float(cfg["ee_yaw_delta_scale"])
        bridge_pitch = float(cfg["ee_pitch_delta_scale"])
        if np.any(bridge_pos <= 0.0) or bridge_yaw <= 0.0 or bridge_pitch <= 0.0:
            raise ValueError("ABS compatibility bridge scales must all be > 0.")

        # Sufficient scale to express any transition within the configured command
        # ranges without the inherited action clipping to [-1,1].
        if np.any(bridge_pos + 1e-12 < np.array(
            [np.ptp(x_range), np.ptp(y_range), np.ptp(z_range)]
        )):
            raise ValueError(
                "kp0_delta_scale is too small for full-range ABS compatibility."
            )
        if bridge_yaw + 1e-12 < np.ptp(yaw_range):
            raise ValueError(
                "ee_yaw_delta_scale is too small for full-range ABS compatibility."
            )
        if bridge_pitch + 1e-12 < np.ptp(geom_pitch_range):
            raise ValueError(
                "ee_pitch_delta_scale is too small for full-range ABS compatibility."
            )

    def _configure_abs_policy_semantics(self) -> None:
        cfg = self.cfg

        self._abs_kp0_x_range = np.asarray(
            cfg["kp0_x_range"], dtype=np.float64
        ).reshape(2)
        self._abs_kp0_y_range = np.asarray(
            cfg["kp0_y_range"], dtype=np.float64
        ).reshape(2)
        self._abs_kp0_z_range = np.asarray(
            cfg["kp0_z_range"], dtype=np.float64
        ).reshape(2)
        self._abs_yaw_range = np.asarray(
            cfg["ee_yaw_range"], dtype=np.float64
        ).reshape(2)
        self._abs_policy_pitch_range = np.asarray(
            cfg["hl_abs_ee_pitch_range"], dtype=np.float64
        ).reshape(2)

        self._ee_pitch_to_euler_sign = float(cfg["ee_pitch_to_euler_sign"])

        # These old names are intentionally retained in YAML only because the
        # inherited validated controller still decodes a delta-shaped action.
        # They are NOT high-level policy action scales in this ABS deployment.
        self._bridge_kp0_scale = np.asarray(
            cfg["kp0_delta_scale"], dtype=np.float64
        ).reshape(3)
        self._bridge_yaw_scale = float(cfg["ee_yaw_delta_scale"])
        self._bridge_pitch_scale = float(cfg["ee_pitch_delta_scale"])

        neutral_kp0 = np.asarray(cfg["neutral_kp0"], dtype=np.float64).reshape(3)
        neutral_policy_pitch = float(cfg["hl_abs_neutral_ee_pitch"])

        neutral_abs_arm = np.array(
            [
                _map_range_to_normalized(
                    neutral_kp0[0], self._abs_kp0_x_range
                ),
                _map_range_to_normalized(
                    neutral_kp0[1], self._abs_kp0_y_range
                ),
                _map_range_to_normalized(
                    neutral_kp0[2], self._abs_kp0_z_range
                ),
                _map_range_to_normalized(
                    float(cfg["neutral_ee_yaw"]), self._abs_yaw_range
                ),
                _map_range_to_normalized(
                    neutral_policy_pitch, self._abs_policy_pitch_range
                ),
            ],
            dtype=np.float32,
        )
        if np.max(np.abs(neutral_abs_arm)) > 1.0 + 1e-6:
            raise ValueError(
                f"Neutral ABS arm action lies outside [-1,1]: {neutral_abs_arm}"
            )

        self._neutral_abs_previous_action = np.concatenate(
            [
                np.zeros(3, dtype=np.float32),
                neutral_abs_arm,
                np.array([-1.0], dtype=np.float32),
            ]
        ).astype(np.float32)

        self._abs_bridge_debug = bool(cfg.get("abs_bridge_debug", True))
        self._abs_bridge_debug_printed = False

    def _reset_abs_policy_alignment_state(self) -> None:
        # If a previous run stopped between high-level inference and the inherited
        # decoder, remove any one-shot cached pose hook before restarting.
        if getattr(self, "_abs_cached_pose_armed", False):
            setattr(
                self,
                self._actual_ee_pose_getter_name,
                self._actual_ee_pose_getter_original,
            )
            self._abs_cached_pose_armed = False

        self._last_raw_abs_hl_action: Optional[np.ndarray] = None
        self._abs_previous_action_history: Optional[np.ndarray] = None
        self._last_abs_kp0_cmd: Optional[np.ndarray] = None
        self._last_abs_yaw_cmd: Optional[float] = None
        self._last_abs_policy_pitch_cmd: Optional[float] = None
        self._last_abs_geom_pitch_cmd: Optional[float] = None

    # ------------------------------------------------------------------
    # High-level observation: preserve RAW ABS previous-action semantics
    # ------------------------------------------------------------------

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
        # First let the validated UAN/mocap wrapper repair object history and
        # grasp-confidence proxy exactly as before.
        fixed = super()._rewrite_high_level_actor_observation(obs_batch)

        arr = np.asarray(fixed)
        if arr.ndim == 1:
            work = arr.reshape(1, -1).copy()
            squeeze = True
        elif arr.ndim == 2 and arr.shape[0] == 1:
            work = arr.copy()
            squeeze = False
        else:
            raise RuntimeError(
                f"Expected high-level actor batch size 1, got {arr.shape}."
            )

        h = int(self.hl_history_length)
        prev_slice = _feature_major_block_slice(
            _HL_PREVIOUS_ACTION_FEATURE,
            history_length=h,
        )
        incoming = work[0, prev_slice].reshape(h, 9).astype(
            np.float32, copy=True
        )

        if self._abs_previous_action_history is None:
            # Training ABS reset semantics: zero base, neutral ABS EE command,
            # and open gripper repeated through the entire 3-frame history.
            self._abs_previous_action_history = np.repeat(
                self._neutral_abs_previous_action.reshape(1, 9),
                h,
                axis=0,
            )
        else:
            current_prev = incoming[-1].copy()

            # Base and gripper are already execution-aware in the common controller.
            # Replace only the five arm entries that otherwise contain the INTERNAL
            # compatibility delta action rather than the RAW ABS actor action.
            if self._last_raw_abs_hl_action is None:
                current_prev[3:8] = self._neutral_abs_previous_action[3:8]
            else:
                current_prev[3:8] = self._last_raw_abs_hl_action[3:8]

            self._abs_previous_action_history = self._append_history(
                self._abs_previous_action_history,
                current_prev,
            )

        work[0, prev_slice] = self._abs_previous_action_history.reshape(-1).astype(
            work.dtype, copy=False
        )
        return work[0] if squeeze else work

    def _record_raw_abs_hl_action(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float32).reshape(9)
        if not np.all(np.isfinite(a)):
            raise RuntimeError(f"Non-finite raw ABS action: {a}")
        self._last_raw_abs_hl_action = np.clip(a, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Output adapter: ABS policy -> inherited delta decoder
    # ------------------------------------------------------------------

    def _find_actual_ee_pose_getter_name(self) -> Optional[str]:
        preferred = (
            "_get_actual_ee_pose_plb",
            "get_actual_ee_pose_plb",
            "_compute_actual_ee_pose_plb",
            "compute_actual_ee_pose_plb",
        )
        for name in preferred:
            fn = getattr(self, name, None)
            if callable(fn):
                return name

        # Minor-refactor fallback: look only for zero-argument EE/pose/PLB methods.
        for name in dir(self):
            low = name.lower()
            if "ee" not in low or "pose" not in low or "plb" not in low:
                continue
            fn = getattr(self, name, None)
            if not callable(fn):
                continue
            try:
                sig = inspect.signature(fn)
            except Exception:
                continue
            required = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Signature.empty
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if not required:
                return name
        return None

    def _read_actual_ee_pose_plb(
        self,
    ) -> tuple[np.ndarray, float, float]:
        if self._actual_ee_pose_getter_name is None:
            raise RuntimeError("PLB EE-pose getter was not resolved.")

        # Always use the original bound getter here. The public attribute may be
        # temporarily replaced by the one-shot cache used by the inherited decoder.
        result = self._actual_ee_pose_getter_original()
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise RuntimeError(
                f"{self._actual_ee_pose_getter_name} must return "
                "(position3, yaw, pitch)."
            )

        pos = np.asarray(result[0], dtype=np.float64).reshape(3)
        yaw = float(np.asarray(result[1]).reshape(()))
        pitch = float(np.asarray(result[2]).reshape(()))
        if not np.all(np.isfinite(pos)) or not np.isfinite(yaw) or not np.isfinite(pitch):
            raise RuntimeError(
                "Non-finite actual EE pose reached ABS compatibility bridge: "
                f"pos={pos}, yaw={yaw}, pitch={pitch}"
            )
        return pos, yaw, pitch

    def _arm_same_pose_for_inherited_decoder(
        self,
        pos: np.ndarray,
        yaw: float,
        pitch: float,
    ) -> None:
        """Make the inherited delta decoder reuse the exact same EE pose sample.

        Without this one-shot hook the bridge and legacy decoder could read the arm
        state twice a few hundred microseconds apart:

            delta = (abs_cmd - measured_1) / scale
            cmd   = measured_2 + scale * delta

        leaving a tiny measured_2-measured_1 residual. Reusing one sample gives
        exact algebraic cancellation and preserves true ABS command semantics.
        """
        if self._abs_cached_pose_armed:
            # A prior bridge action was not consumed by the inherited decoder.
            # Continuing would silently change the control graph.
            raise RuntimeError(
                "Previous ABS bridge pose cache was not consumed by the inherited "
                "high-level decoder. Refusing to continue."
            )

        cached_pos = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        cached_yaw = float(yaw)
        cached_pitch = float(pitch)
        original = self._actual_ee_pose_getter_original
        getter_name = self._actual_ee_pose_getter_name

        def _one_shot_cached_pose(*args, **kwargs):
            # Restore before returning so any later call in the same control step
            # observes the live state again.
            setattr(self, getter_name, original)
            self._abs_cached_pose_armed = False
            return cached_pos.copy(), cached_yaw, cached_pitch

        setattr(self, getter_name, _one_shot_cached_pose)
        self._abs_cached_pose_armed = True

    def _abs_action_to_legacy_delta_bridge(
        self,
        raw_abs_action: np.ndarray,
    ) -> np.ndarray:
        a = np.clip(
            np.asarray(raw_abs_action, dtype=np.float64).reshape(9),
            -1.0,
            1.0,
        )

        kp0_cmd = np.array(
            [
                _map_normalized_to_range(a[3], self._abs_kp0_x_range),
                _map_normalized_to_range(a[4], self._abs_kp0_y_range),
                _map_normalized_to_range(a[5], self._abs_kp0_z_range),
            ],
            dtype=np.float64,
        )
        yaw_cmd = _map_normalized_to_range(a[6], self._abs_yaw_range)
        policy_pitch_cmd = _map_normalized_to_range(
            a[7], self._abs_policy_pitch_range
        )

        # FIX for the sim2sim bug:
        # Z1 gripper local +X points from wrist/stator toward fingertips.
        # Negative POLICY pitch must point +X downward, so standard geometric
        # XYZ-Euler pitch uses the opposite sign.
        geom_pitch_cmd = self._ee_pitch_to_euler_sign * policy_pitch_cmd

        actual_pos, actual_yaw_geom, actual_pitch_geom = (
            self._read_actual_ee_pose_plb()
        )

        # The inherited legacy decoder is called immediately after this ONNX
        # session returns. Force it to reuse this exact pose sample so the measured
        # term cancels identically rather than approximately.
        self._arm_same_pose_for_inherited_decoder(
            actual_pos,
            actual_yaw_geom,
            actual_pitch_geom,
        )

        bridge = a.copy()
        bridge[3:6] = (kp0_cmd - actual_pos) / self._bridge_kp0_scale
        bridge[6] = (yaw_cmd - actual_yaw_geom) / self._bridge_yaw_scale
        bridge[7] = (
            geom_pitch_cmd - actual_pitch_geom
        ) / self._bridge_pitch_scale
        # bridge[0:3] and bridge[8] remain exactly the raw ABS base/gripper action.

        if not np.all(np.isfinite(bridge)):
            raise RuntimeError(f"Non-finite ABS compatibility action: {bridge}")

        max_arm = float(np.max(np.abs(bridge[3:8])))
        if max_arm > 1.0 + 1e-6:
            raise RuntimeError(
                "ABS compatibility action exceeded [-1,1]. Refusing to let the "
                "inherited controller clip and silently change the absolute EE "
                "command.\n"
                f"  raw_abs       = {a}\n"
                f"  kp0_cmd       = {kp0_cmd}\n"
                f"  yaw_cmd       = {yaw_cmd:.6f}\n"
                f"  pitch_policy  = {policy_pitch_cmd:.6f}\n"
                f"  pitch_geom    = {geom_pitch_cmd:.6f}\n"
                f"  actual_pos    = {actual_pos}\n"
                f"  actual_yaw    = {actual_yaw_geom:.6f}\n"
                f"  actual_pitch  = {actual_pitch_geom:.6f}\n"
                f"  bridge_action = {bridge}"
            )

        self._last_abs_kp0_cmd = kp0_cmd.astype(np.float32)
        self._last_abs_yaw_cmd = float(yaw_cmd)
        self._last_abs_policy_pitch_cmd = float(policy_pitch_cmd)
        self._last_abs_geom_pitch_cmd = float(geom_pitch_cmd)

        if self._abs_bridge_debug and not self._abs_bridge_debug_printed:
            print(
                "[ABS-BRIDGE] first command | "
                f"kp0_abs={np.array2string(kp0_cmd, precision=4)} | "
                f"yaw={yaw_cmd:+.4f} | "
                f"pitch_policy={policy_pitch_cmd:+.4f} -> "
                f"pitch_geom={geom_pitch_cmd:+.4f} | "
                f"legacy_action={np.array2string(bridge[3:8], precision=4)}"
            )
            self._abs_bridge_debug_printed = True

        return bridge.astype(np.float32)

    # ------------------------------------------------------------------
    # Session installation
    # ------------------------------------------------------------------

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

    def _find_high_policy_session_attr_for_abs(self) -> Optional[str]:
        # Prefer the exact session member already located by the inherited
        # observation-alignment wrapper.
        inherited_attr = getattr(self, "_hl_alignment_session_attr", None)
        if inherited_attr:
            session = getattr(self, inherited_attr, None)
            if session is not None and callable(getattr(session, "run", None)):
                return str(inherited_attr)

        for name, session in vars(self).items():
            if isinstance(session, _ABSHighPolicyActionAdapterSession):
                return name
            if not callable(getattr(session, "run", None)):
                continue
            if self._session_input_last_dim(session) == int(self.hl_obs_dim):
                return name
        return None

    def _ensure_abs_high_policy_action_adapter(self, *, required: bool) -> None:
        attr = self._find_high_policy_session_attr_for_abs()
        if attr is None:
            if required:
                raise RuntimeError(
                    "Could not locate the 168-D high-level ONNX session for "
                    "ABS action adaptation."
                )
            return

        current = getattr(self, attr)
        if isinstance(current, _ABSHighPolicyActionAdapterSession):
            self._abs_action_session = current
            self._abs_action_session_attr = attr
            return

        adapter = _ABSHighPolicyActionAdapterSession(self, current)
        setattr(self, attr, adapter)
        self._abs_action_session = adapter
        self._abs_action_session_attr = attr

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def prime_first_hl_and_ll(self):
        # New policy run: reset ABS previous-action history BEFORE the first
        # actor observation is inferred.
        self._reset_abs_policy_alignment_state()
        self._ensure_abs_high_policy_action_adapter(required=True)
        return super().prime_first_hl_and_ll()

    def setup(self) -> None:
        print("=" * 108)
        print("ABS UAN-TRAINED MOCAP RETRIEVAL -> REAL ROBOT")
        print("HL arm semantics   : ABS normalized action -> PLB XYZ / yaw / policy-pitch")
        print("Policy feedback    : NO measured-EE re-anchoring semantics")
        print(
            "Pitch convention   : pitch_geom = "
            f"{self._ee_pitch_to_euler_sign:+.0f} * pitch_policy "
            "(Z1 local +X points toward fingertips)"
        )
        print(
            "Neutral ABS action : "
            + np.array2string(
                self._neutral_abs_previous_action[3:8],
                precision=6,
                floatmode="fixed",
            )
        )
        print(
            "Legacy delta bridge: internal algebraic compatibility ONLY; "
            "RAW ABS action is preserved in previous_hl_action obs"
        )
        print(
            f"EE pose getter      : {self._actual_ee_pose_getter_name}"
        )
        print("UAN model           : NOT LOADED / NOT EXECUTED")
        print("=" * 108)

        super().setup()

        # setup() may create ONNX sessions in some controller versions.
        self._ensure_abs_high_policy_action_adapter(required=True)
        print(
            "[ABS-ACTION] adapter installed on "
            f"{self._abs_action_session_attr!r}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "B2WZ1 ABS high-level retrieval sim2real with OptiTrack mocap "
            "and UAN-trained frozen WBC."
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
        default="deploy/configs/b2wz1_hl_retrieval_abs_uan.yaml",
        help="Path to the ABS UAN-trained mocap sim2real YAML.",
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)

    controller = B2WZ1MocapRetrievalABSUANController(
        cfg_path=args.config,
        network_interface=args.net,
    )
    controller.run()


if __name__ == "__main__":
    main()