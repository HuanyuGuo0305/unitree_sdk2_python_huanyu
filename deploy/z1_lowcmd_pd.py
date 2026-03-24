"""
Simple Z1 low-level external PD test.

Purpose:
- Verify that the modified Python binding exposes lowcmd.setControlGain()
- Verify that external PD style command pipeline works
- Move Z1 to default_arm_pos and hold it there

Run from repository root:

    python3 deploy/z1_lowlevel_pd.py deploy/configs/b2wz1_locomanipulation.yaml

    python3 deploy/z1_lowlevel_pd.py deploy/configs/b2wz1_locomanipulation.yaml --hold
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.z1_helper import Z1ArmAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to yaml config")
    parser.add_argument("--move-duration", type=float, default=3.0, help="Move-to-default duration in seconds")
    parser.add_argument("--hold", action="store_true", help="Hold default pose until Ctrl+C")
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.abspath(os.path.join(PROJECT_ROOT, cfg_path))

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    z1 = Z1ArmAdapter(cfg, PROJECT_ROOT)

    print("=" * 80)
    print("Z1 low-level external PD test")
    print("=" * 80)
    print(f"Config              : {cfg_path}")
    print(f"Control dt          : {cfg['control_dt']}")
    print(f"Default arm pos     : {cfg['default_arm_pos']}")
    print(f"Startup kp          : {cfg.get('arm_kps_startup')}")
    print(f"Startup kd          : {cfg.get('arm_kds_startup')}")
    print(f"Runtime kp          : {cfg.get('arm_kps_runtime')}")
    print(f"Runtime kd          : {cfg.get('arm_kds_runtime')}")
    print(f"arm_qd_mode         : {cfg.get('arm_qd_mode', 'zero')}")
    print(f"arm_tau_mode        : {cfg.get('arm_tau_mode', 'zero')}")
    print("=" * 80)

    z1.connect()

    print("[Z1Test] Reading initial state...")
    z1.read_state()
    print(f"[Z1Test] q_init = {np.array2string(z1.q, precision=4, suppress_small=True)}")
    print(f"[Z1Test] qd_init = {np.array2string(z1.qd, precision=4, suppress_small=True)}")

    print("[Z1Test] Moving to default arm pose using startup gains...")
    z1.move_to_pose(
        target_q=np.array(cfg["default_arm_pos"], dtype=np.float32),
        duration=float(args.move_duration),
        use_startup_gains=True,
    )

    print("[Z1Test] Reached default pose. Reading back state...")
    z1.read_state()
    err = np.array(cfg["default_arm_pos"], dtype=np.float32) - z1.q
    print(f"[Z1Test] q_now  = {np.array2string(z1.q, precision=4, suppress_small=True)}")
    print(f"[Z1Test] q_err  = {np.array2string(err, precision=4, suppress_small=True)}")
    print(f"[Z1Test] max|err| = {np.max(np.abs(err)):.6f}")

    if not args.hold:
        print("[Z1Test] Done.")
        return

    print("[Z1Test] Holding default pose with startup gains. Press Ctrl+C to stop.")
    try:
        counter = 0
        while True:
            z1.send_arm_command(
                q_cmd=np.array(cfg["default_arm_pos"], dtype=np.float32),
                gripper_q_cmd=float(cfg["default_gripper_pos"]),
                use_startup_gains=True,
            )

            counter += 1
            if counter % 50 == 0:
                z1.read_state()
                err = np.array(cfg["default_arm_pos"], dtype=np.float32) - z1.q
                print(
                    f"[Z1Test {counter:5d}] "
                    f"max|err|={np.max(np.abs(err)):.6f} | "
                    f"q={np.array2string(z1.q, precision=4, suppress_small=True)}"
                )

            time.sleep(float(cfg["control_dt"]))

    except KeyboardInterrupt:
        print("[Z1Test] KeyboardInterrupt received.")

    print("[Z1Test] Exit.")


if __name__ == "__main__":
    main()