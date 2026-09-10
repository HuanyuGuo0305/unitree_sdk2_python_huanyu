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

HOW THE HL ACTION REACHES THE LOW-LEVEL CONTROLLER
--------------------------------------------------
This wrapper overrides ``decode_hl_action`` and writes the EE keypoint command
with the SAME arithmetic as the hierarchical sim2sim reference

    unitree_mujoco/simulate_python/deploy_mujoco/b2wz1_hl_retrieval_abs_uan.py

i.e. each normalized action component maps affinely and directly onto its
configured absolute PLB range:

    kp0_cmd      = map(action[3:6] -> kp0_{x,y,z}_range)
    kp0_cmd.z    = max(kp0_cmd.z, kp0_z_cmd_min)      # safety floor
    yaw_cmd      = map(action[6]   -> ee_yaw_range)
    pitch_policy = map(action[7]   -> hl_abs_ee_pitch_range)
    pitch_geom   = ee_pitch_to_euler_sign * pitch_policy
    ee_cmd_plb_current = build_keypoints(kp0_cmd, yaw_cmd, pitch_geom, ...)

There is NO measured-EE re-anchoring, NO delta scale, NO delta clip and NO
range re-clipping: the affine map already lands inside the configured range by
construction, exactly as in sim2sim.

The single exception is ``kp0_z_cmd_min``, matching the sim2sim reference: the
trained kp0_z range bottoms out at ground level, so a saturated-low action[5]
would drive the wrist into the floor. The floor is applied to the DECODED
command, after the affine map, so the action -> range mapping the policy was
trained with is unchanged. The raw action still goes into previous_hl_action.

An earlier revision instead converted the ABS action into a fake delta action
for the inherited legacy decoder. That indirection is removed. It had also
been mis-configured (``kp0_delta_clip: [0,0,0]``, whose real "disabled"
sentinel in the inherited decoder is ``+inf``), which silently zeroed every arm
delta and reduced the EE command to "hold the currently measured EE pose" --
the arm target stayed up in the air instead of descending to the object.

The base command, gripper binarization and grasp latch are reproduced exactly
as the inherited decoder does them.

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

The shared ``build_keypoints_from_kp0_yaw_pitch_plb`` helper takes GEOMETRIC
pitch, so the sign conversion happens here before the call. The policy range
lives in YAML as ``hl_abs_ee_pitch_range`` = [-70 deg, 0]; ``neutral_ee_pitch``
stays GEOMETRIC because the inherited neutral keypoint builder consumes it.

Run from the unitree_sdk2_python_huanyu repository root:

    python3 deploy/b2wz1_hl_retrieval_abs_uan_mocap.py \
        enxa0cec819e15f \
        deploy/configs/b2wz1_hl_retrieval_abs_uan_mocap.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import numpy as np

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DEPLOY_DIR, ".."))
for _p in (_PROJECT_ROOT, _DEPLOY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

# Shared EE keypoint builder. It consumes GEOMETRIC pitch and applies no sign
# convention of its own -- identical geometry to the sim2sim helper.
from b2wz1_hl_retrieval_sim2real import build_keypoints_from_kp0_yaw_pitch_plb

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


def _map_normalized_to_range(x: float, limits: np.ndarray) -> float:
    """Map normalized action x in [-1,1] affinely to [low, high]."""
    low, high = float(limits[0]), float(limits[1])
    if not high > low:
        raise ValueError(f"Invalid range [{low}, {high}].")
    return low + 0.5 * (float(x) + 1.0) * (high - low)


def _map_range_to_normalized(value: float, limits: np.ndarray) -> float:
    """Inverse affine map, used to encode neutral / held EE commands."""
    low, high = float(limits[0]), float(limits[1])
    if not high > low:
        raise ValueError(f"Invalid range [{low}, {high}].")
    return 2.0 * (float(value) - low) / (high - low) - 1.0


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

        # False whenever current_hl_action[3:8] holds a reset/safe-hold sentinel
        # rather than a real ABS actor output. See build_previous_hl_action().
        self._abs_arm_action_valid = False
        self._abs_arm_action_for_hold_or_reset = (
            self._neutral_abs_arm_action.copy()
        )

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

        # Safety floor on the DECODED kp0_z command. The trained ABS range starts
        # at ground level, so a saturated-low action[5] drives the arm into the
        # floor. Clamping AFTER the affine map keeps the action->range mapping
        # identical to training instead of silently rescaling it.
        kp0_z_cmd_min = float(cfg.get("kp0_z_cmd_min", z_range[0]))
        if not (z_range[0] <= kp0_z_cmd_min <= z_range[1]):
            raise ValueError(
                f"kp0_z_cmd_min={kp0_z_cmd_min} must lie inside kp0_z_range "
                f"[{float(z_range[0])}, {float(z_range[1])}]."
            )
        yaw_range = self._as_range(cfg, "ee_yaw_range")
        policy_pitch_range = self._as_range(cfg, "hl_abs_ee_pitch_range")

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

        # The inherited neutral EE keypoint builder (used on reset) consumes
        # GEOMETRIC pitch, so YAML must carry the sign-converted value.
        expected_geom_neutral_pitch = sign * float(
            cfg["hl_abs_neutral_ee_pitch"]
        )
        if abs(float(cfg["neutral_ee_pitch"]) - expected_geom_neutral_pitch) > 1e-12:
            raise ValueError(
                "neutral_ee_pitch is consumed by the inherited GEOMETRIC neutral "
                "EE keypoint builder and must equal -hl_abs_neutral_ee_pitch.\n"
                f"  expected={expected_geom_neutral_pitch}, "
                f"configured={cfg['neutral_ee_pitch']}"
            )

        # decode_hl_action() is fully overridden below, so the inherited legacy
        # delta decoder never runs. Guard that the override is really the one in
        # effect before trusting any of the ABS semantics above.
        if (
            type(self).decode_hl_action
            is not B2WZ1MocapRetrievalABSUANController.decode_hl_action
        ):
            raise RuntimeError(
                "decode_hl_action has been overridden again by "
                f"{type(self).__name__}. The ABS action would no longer be "
                "decoded with sim2sim-identical semantics."
            )

        # kp0_delta_scale / ee_yaw_delta_scale / ee_pitch_delta_scale stay in
        # YAML only because the inherited constructor requires them; they are
        # inert here. The optional per-step clips belong to the same dead path
        # and must not reappear: a stale kp0_delta_clip: [0,0,0] previously
        # zeroed the whole arm command (the inherited "disabled" sentinel is
        # +inf, not 0), leaving the EE target parked at the measured arm pose.
        dead_clip_keys = [
            key
            for key in (
                "kp0_delta_clip",
                "ee_yaw_delta_clip",
                "ee_pitch_delta_clip",
            )
            if key in cfg
        ]
        if dead_clip_keys:
            raise ValueError(
                "This ABS deployment decodes the high-level action directly, "
                "exactly like sim2sim, so the inherited delta decoder and its "
                "per-step clips are bypassed. Remove these keys from the YAML "
                "rather than implying they still have an effect:\n  "
                + ", ".join(dead_clip_keys)
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

        self._abs_kp0_z_cmd_min = float(
            cfg.get("kp0_z_cmd_min", self._abs_kp0_z_range[0])
        )

        self._ee_pitch_to_euler_sign = float(cfg["ee_pitch_to_euler_sign"])

        neutral_kp0 = np.asarray(cfg["neutral_kp0"], dtype=np.float64).reshape(3)
        neutral_policy_pitch = float(cfg["hl_abs_neutral_ee_pitch"])

        # Training reset semantics for the ABS policy: previous_hl_action[3:8]
        # encodes the neutral EE command in normalized ABS coordinates, not zeros.
        self._neutral_abs_arm_action = np.array(
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
        if np.max(np.abs(self._neutral_abs_arm_action)) > 1.0 + 1e-6:
            raise ValueError(
                "Neutral ABS arm action lies outside [-1,1]: "
                f"{self._neutral_abs_arm_action}"
            )

        self._abs_decode_debug = bool(cfg.get("abs_decode_debug", True))
        self._abs_decode_debug_printed = False

    # ------------------------------------------------------------------
    # High-level action -> low-level EE command (sim2sim-identical)
    # ------------------------------------------------------------------

    def decode_hl_action(self, action: np.ndarray) -> None:
        """Write the ABS high-level action to the low-level controller.

        Same arithmetic as the sim2sim reference
        ``deploy_mujoco/b2wz1_hl_retrieval_abs_uan.py::decode_hl_action``.
        """
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(self.hl_action_dim),
            -1.0,
            1.0,
        )

        self.current_hl_action[:] = action
        self._abs_arm_action_valid = True

        self.base_command[:] = (
            action[0:3] * self.base_cmd_scale
        ).astype(np.float32)

        # ABS high-level arm/EE semantics: each normalized action maps directly
        # to the configured PLB command range. There is NO measured-EE
        # re-anchoring, and the affine map cannot leave the range.
        kp0_cmd = np.array(
            [
                _map_normalized_to_range(action[3], self._abs_kp0_x_range),
                _map_normalized_to_range(action[4], self._abs_kp0_y_range),
                # Safety floor applied AFTER the affine map, exactly as sim2sim
                # does; the action -> range mapping itself is untouched.
                max(
                    _map_normalized_to_range(action[5], self._abs_kp0_z_range),
                    self._abs_kp0_z_cmd_min,
                ),
            ],
            dtype=np.float32,
        )
        yaw_cmd = _map_normalized_to_range(action[6], self._abs_yaw_range)
        policy_pitch_cmd = _map_normalized_to_range(
            action[7], self._abs_policy_pitch_range
        )

        # Z1 gripper local +X points from wrist/stator toward the fingertips.
        # Negative POLICY pitch must point +X downward, so the standard geometric
        # XYZ-Euler pitch consumed by the keypoint builder takes the opposite sign.
        geom_pitch_cmd = self._ee_pitch_to_euler_sign * policy_pitch_cmd

        self.ee_cmd_plb_current[:] = build_keypoints_from_kp0_yaw_pitch_plb(
            kp0=kp0_cmd,
            yaw=float(yaw_cmd),
            pitch=float(geom_pitch_cmd),
            roll=self.fixed_ee_roll,
            kp_dx=self.ee_kp_dx,
            kp_dz=self.ee_kp_dz,
        )

        self.raw_gripper_action = float(action[8])

        # P1 alignment with training: strictly positive closes; zero remains open.
        binary_close = self.raw_gripper_action > self.gripper_binary_threshold
        # Force-close policy lives in the base controller so the ABS and delta
        # decoders cannot drift apart. See gripper_force_close_mode.
        executed_close = self.resolve_executed_gripper_close(binary_close)

        self.executed_gripper_cmd_norm = 1.0 if executed_close else -1.0
        self.gripper_target = (
            self.gripper_close_pos if executed_close else self.gripper_open_pos
        )

        # command_assumed mode:
        # first executed CLOSE/grasp command -> proxy=True immediately.
        self._latch_command_assumed_grasp_if_needed(
            executed_close=executed_close,
        )

        if self._abs_decode_debug and not self._abs_decode_debug_printed:
            print(
                "[ABS-DECODE] first command | "
                f"kp0_abs={np.array2string(kp0_cmd, precision=4)} | "
                f"yaw={yaw_cmd:+.4f} | "
                f"pitch_policy={policy_pitch_cmd:+.4f} -> "
                f"pitch_geom={geom_pitch_cmd:+.4f}"
            )
            self._abs_decode_debug_printed = True

    # ------------------------------------------------------------------
    # previous_hl_action observation: ABS reset / safe-hold semantics
    # ------------------------------------------------------------------

    def _abs_arm_action_from_measured_ee(self) -> np.ndarray:
        """Normalized ABS encoding of the currently measured EE pose.

        The inherited safe-hold path commands the measured EE keypoints and
        represents that as a ZERO normalized action. Zero is meaningful in ABS
        coordinates (the midpoint of every range), so encode the pose that is
        actually being held instead. Falls back to the neutral ABS action if the
        measured pose is unavailable during a perception fault.
        """
        try:
            pos, yaw_geom, pitch_geom = self.compute_actual_ee_pose_plb()
            pos = np.asarray(pos, dtype=np.float64).reshape(3)
            policy_pitch = self._ee_pitch_to_euler_sign * float(pitch_geom)
            encoded = np.array(
                [
                    _map_range_to_normalized(pos[0], self._abs_kp0_x_range),
                    _map_range_to_normalized(pos[1], self._abs_kp0_y_range),
                    _map_range_to_normalized(pos[2], self._abs_kp0_z_range),
                    _map_range_to_normalized(float(yaw_geom), self._abs_yaw_range),
                    _map_range_to_normalized(
                        policy_pitch, self._abs_policy_pitch_range
                    ),
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(encoded)):
                raise ValueError(f"Non-finite measured ABS encoding: {encoded}")
            return np.clip(encoded, -1.0, 1.0).astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - observation must never crash control
            print(
                "[ABS-HOLD] could not encode measured EE pose "
                f"({type(exc).__name__}: {exc}); using neutral ABS arm action."
            )
            return self._neutral_abs_arm_action.copy()

    def build_previous_hl_action(self) -> np.ndarray:
        prev = np.asarray(
            super().build_previous_hl_action(), dtype=np.float32
        ).reshape(9).copy()

        # Base and gripper entries are already execution-aware in the inherited
        # controller and carry the same semantics as sim2sim. Only the five arm
        # entries need ABS-correct reset/safe-hold values: the inherited resets
        # write zeros, which in ABS coordinates is a real (wrong) command rather
        # than a neutral sentinel.
        if not self._abs_arm_action_valid:
            prev[3:8] = self._abs_arm_action_for_hold_or_reset

        return prev

    # ------------------------------------------------------------------
    # Reset / safe-hold hooks
    #
    # The inherited methods below set current_hl_action[:] = 0 and then build the
    # actor observation from it, so the ABS sentinel must be selected BEFORE
    # delegating to super().
    # ------------------------------------------------------------------

    def initialize_policy_state_and_history(self) -> Dict[str, Any]:
        # Reset commands the neutral EE pose, exactly like the sim2sim reset.
        self._abs_arm_action_valid = False
        self._abs_arm_action_for_hold_or_reset = (
            self._neutral_abs_arm_action.copy()
        )
        return super().initialize_policy_state_and_history()

    def _set_safe_perception_hold_commands(self) -> None:
        super()._set_safe_perception_hold_commands()
        # The hold commands the measured EE pose, so encode that.
        self._abs_arm_action_valid = False
        self._abs_arm_action_for_hold_or_reset = (
            self._abs_arm_action_from_measured_ee()
        )

    def _recover_perception_hold_and_run_hl(self, task: Dict[str, Any]) -> None:
        # super() re-seeds the whole HL history from the current safe-hold
        # commands, which are still the measured EE pose from the hold.
        self._abs_arm_action_valid = False
        self._abs_arm_action_for_hold_or_reset = (
            self._abs_arm_action_from_measured_ee()
        )
        super()._recover_perception_hold_and_run_hl(task)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def prime_first_hl_and_ll(self):
        # New policy run: the first actor observation must see the neutral ABS
        # command, never a stale arm action from a previous run.
        self._abs_arm_action_valid = False
        self._abs_arm_action_for_hold_or_reset = (
            self._neutral_abs_arm_action.copy()
        )
        self._abs_decode_debug_printed = False
        return super().prime_first_hl_and_ll()

    def setup(self) -> None:
        print("=" * 108)
        print("ABS UAN-TRAINED MOCAP RETRIEVAL -> REAL ROBOT")
        print("HL arm semantics   : ABS normalized action -> PLB XYZ / yaw / policy-pitch")
        print("Decode path        : direct, sim2sim-identical (no delta decoder)")
        print(
            "kp0_z command floor: "
            f"{self._abs_kp0_z_cmd_min:.3f} m "
            f"(decoded kp0_z range [{float(self._abs_kp0_z_range[0]):.3f}, "
            f"{float(self._abs_kp0_z_range[1]):.3f}] m)"
        )
        print("Policy feedback    : NO measured-EE re-anchoring semantics")
        print(
            "Pitch convention   : pitch_geom = "
            f"{self._ee_pitch_to_euler_sign:+.0f} * pitch_policy "
            "(Z1 local +X points toward fingertips)"
        )
        print(
            "Neutral ABS action : "
            + np.array2string(
                self._neutral_abs_arm_action,
                precision=6,
                floatmode="fixed",
            )
        )
        print("UAN model           : NOT LOADED / NOT EXECUTED")
        print("=" * 108)

        super().setup()


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
        default="deploy/configs/b2wz1_hl_retrieval_abs_uan_mocap.yaml",
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
