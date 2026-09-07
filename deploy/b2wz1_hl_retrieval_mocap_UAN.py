#!/usr/bin/env python3
"""
B2WZ1 hierarchical retrieval sim2real -- MOCAP sensing -- UAN-trained policies.

This is a thin deployment wrapper around the already-validated
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

The only required arm-side deployment alignment is therefore:
    SDK raw Kp * 25.6  == [64, 115.2, 64, 64, 64, 64]
    SDK raw Kd * 0.0128 == [3, 4, 3, 3, 3, 3]

which gives:
    arm_kps_runtime = [2.5, 4.5, 2.5, 2.5, 2.5, 2.5]
    arm_kds_runtime = [234.375, 312.5, 234.375, 234.375, 234.375, 234.375]

No arm target rate limiter is allowed.  The 50-Hz WBC target is held and
re-sent by the existing arm command thread.  The gripper remains the original
external DCMotor-style actuator used by the retrieval training.

Run from the unitree_sdk2_python_huanyu repository root:

    python3 deploy/b2wz1_hl_retrieval_mocap_UAN.py \
        enxa0cec819e15f \
        deploy/configs/b2wz1_hl_retrieval_mocap_UAN.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

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
        # 5. Hierarchical policy timing/shape contract.
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

        if str(cfg.get("z1_gripper_runtime_mode", "")).lower() != "dcmotor":
            raise ValueError(
                "jointGripper must remain in the validated DCMotor-style "
                "runtime path used by retrieval training."
            )

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
        print("=" * 108)
        super().setup()


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