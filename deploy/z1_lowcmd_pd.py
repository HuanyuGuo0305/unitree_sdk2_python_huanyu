#!/usr/bin/env python3
import os
import sys
import time
import numpy as np

# Add project root to sys.path so we can import utils.z1_helper
PROJECT_ROOT = os.path.abspath("/home/huanyuguo/Workspace_huanyu/unitree_sdk2_python_huanyu")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.z1_helper import Z1ArmAdapter, compute_ee_current_kp_lb


# ---------------------------------------------------------------------
# User config
# ---------------------------------------------------------------------

CFG = {
    "z1_sdk_lib": "/home/huanyuguo/Workspace_huanyu/z1_sdk_huanyu/lib",

    # Control timing
    "control_dt": 0.02,
    "z1_control_dt": 0.02,

    # Arm setup
    "z1_has_gripper": True,
    "z1_fk_ee_index": 6,

    # Default target
    "default_arm_pos": [0.0, 1.48, -1.0, -0.54, 0.0, 0.0],
    "default_gripper_pos": 0.0,

    # Startup gains used by hold_pose_lowcmd()
    "arm_kps_startup": [3.33, 5.0, 5.0, 3.33, 2.5, 1.66],
    "arm_kds_startup": [333.0, 333.0, 333.0, 333.0, 333.0, 333.0],

    # Runtime gains (not important for this test, but adapter expects them)
    "arm_kps_runtime": [1.56, 1.56, 1.56, 1.56, 1.56, 1.56],
    "arm_kds_runtime": [235, 235, 235, 235, 235, 235],

    # Step limiter for runtime PD path
    "z1_runtime_q_step_clip": [0.04, 0.04, 0.04, 0.04, 0.04, 0.04],

    # Gripper gains
    "z1_gripper_kp": 20.0,
    "z1_gripper_kd": 2000.0,

    # Kinematic transforms
    "arm_base_offset_pos": [0.10, 0.0, 0.12],
    "arm_base_offset_rpy": [0.0, 0.0, 0.0],
    "z1_fk_to_policy_ee_pos": [0.051, 0.0, 0.0],
    "z1_fk_to_policy_ee_rpy": [0.0, 0.0, 0.0],

    # Debug
    "z1_debug_print": False,
}

ZERO_Q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_Q = np.array([0.0, 1.48, -1.0, -0.54, 0.0, 0.0], dtype=np.float32)
GRIPPER_Q = 0.0

MOVE_TIME = 4.0
HOLD_TIME = 2.0
PRINT_EVERY = 20

# Base orientation for standalone Z1 test.
# If the arm base is level and world-aligned, identity is correct.
BASE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Keypoint definition, must match your sim/deploy pipeline
KP_DX = 0.30
KP_DZ = 0.30


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def get_q(z1: Z1ArmAdapter) -> np.ndarray:
    z1.read_state()
    return z1.q.copy()


def get_qd(z1: Z1ArmAdapter) -> np.ndarray:
    z1.read_state()
    return z1.qd.copy()


def print_ee_current_lb(z1: Z1ArmAdapter, title: str, repeat: int = 5, sleep_dt: float = 0.02):
    """
    Print current EE keypoints in LB frame for several consecutive timesteps.
    """
    print(f"\n=== {title} ===")
    for i in range(repeat):
        z1.read_state()
        ee_kp_lb = compute_ee_current_kp_lb(
            base_quat_wxyz=BASE_QUAT_WXYZ,
            z1_adapter=z1,
            kp_dx=KP_DX,
            kp_dz=KP_DZ,
        )
        print(f"[EE_CURRENT_LB {i+1}/{repeat}] {np.round(ee_kp_lb, 6)}")
        time.sleep(sleep_dt)


def move_to_target_pd(z1: Z1ArmAdapter, q_target: np.ndarray, hold_time: float, name: str):
    """
    Move to a target using lowcmd PD hold path.
    """
    dt = z1.get_arm_dt()
    steps = max(1, int(round(MOVE_TIME / dt)))

    q_start = get_q(z1)
    prev_q_cmd = q_start.copy()

    print(f"\n=== Move to {name} ===")
    print("q_start  =", np.round(q_start, 3))
    print("q_target =", np.round(q_target, 3))

    max_err = np.zeros(6, dtype=np.float32)
    step_clip = z1.runtime_q_step_clip.copy()

    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        q_ref = (1.0 - alpha) * q_start + alpha * q_target

        dq = np.clip(q_ref - prev_q_cmd, -step_clip, step_clip)
        q_cmd = prev_q_cmd + dq

        z1.hold_pose_lowcmd(
            q_cmd=q_cmd,
            gripper_q_cmd=GRIPPER_Q,
        )

        q_meas = get_q(z1)
        qd_meas = get_qd(z1)
        err = q_cmd - q_meas
        max_err = np.maximum(max_err, np.abs(err))
        prev_q_cmd = q_cmd.copy()

        if i % PRINT_EVERY == 0 or i == steps - 1:
            print(
                f"[{name} {i+1:04d}/{steps}] "
                f"err={np.round(err, 3)} "
                f"qd={np.round(qd_meas, 3)}"
            )

        time.sleep(dt)

    print(f"--- Hold {name} for {hold_time:.1f}s ---")
    hold_steps = max(1, int(round(hold_time / dt)))

    for i in range(hold_steps):
        z1.hold_pose_lowcmd(
            q_cmd=q_target,
            gripper_q_cmd=GRIPPER_Q,
        )

        q_meas = get_q(z1)
        qd_meas = get_qd(z1)
        err = q_target - q_meas
        max_err = np.maximum(max_err, np.abs(err))

        if i % PRINT_EVERY == 0 or i == hold_steps - 1:
            print(
                f"[hold {name} {i+1:04d}/{hold_steps}] "
                f"err={np.round(err, 3)} "
                f"qd={np.round(qd_meas, 3)}"
            )

        time.sleep(dt)

    print(f"max_abs_err({name}) = {np.round(max_err, 4)}")

    if name == "DEFAULT":
        print_ee_current_lb(
            z1=z1,
            title="EE keypoints in LB frame at DEFAULT pose",
            repeat=5,
            sleep_dt=dt,
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    np.set_printoptions(precision=3, suppress=True)

    print("Press Ctrl+C to stop.")
    print("Startup KP =", np.array(CFG["arm_kps_startup"], dtype=np.float32))
    print("Startup KD =", np.array(CFG["arm_kds_startup"], dtype=np.float32))

    z1 = Z1ArmAdapter(cfg=CFG, project_root=PROJECT_ROOT)
    z1.connect()

    print("Current q =", np.round(get_q(z1), 3))

    try:
        move_to_target_pd(z1, ZERO_Q, HOLD_TIME, "ZERO")
        move_to_target_pd(z1, DEFAULT_Q, HOLD_TIME, "DEFAULT")
        move_to_target_pd(z1, ZERO_Q, HOLD_TIME, "ZERO_BACK")
        print("\nDone.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()