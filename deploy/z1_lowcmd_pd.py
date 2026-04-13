#!/usr/bin/env python3
import sys
import time
import numpy as np

# 改成你的 SDK lib 路径
sys.path.append("/home/huanyuguo/Workspace_huanyu/z1_sdk_huanyu/lib")
import unitree_arm_interface


HAS_GRIPPER = True

KP = np.array([18.0, 28.0, 28.0, 18.0, 12.0, 8.0], dtype=np.float32)
KD = np.array([1800.0, 1800.0, 1800.0, 1800.0, 1800.0, 1800.0], dtype=np.float32)

ZERO_Q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_Q = np.array([0.0, 1.48, -1.0, -0.54, 0.0, 0.0], dtype=np.float32)

GRIPPER_Q = 0.0

MOVE_TIME = 4.0
HOLD_TIME = 2.0
STEP_CLIP = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04], dtype=np.float32)

PRINT_EVERY = 20


def set_gain(arm, kp, kd):
    kp_full = [float(x) for x in kp] + [20.0]
    kd_full = [float(x) for x in kd] + [2000.0]
    arm._ctrlComp.lowcmd.setControlGain(kp_full, kd_full)


def get_q(arm):
    return np.asarray(arm.lowstate.getQ(), dtype=np.float32).reshape(6,)


def get_qd(arm):
    return np.asarray(arm.lowstate.getQd(), dtype=np.float32).reshape(6,)


def send_pd(arm, q_cmd, gripper_q):
    qd_cmd = np.zeros(6, dtype=np.float32)
    tau_cmd = np.zeros(6, dtype=np.float32)

    arm.q = q_cmd
    arm.qd = qd_cmd
    arm.tau = tau_cmd
    arm.gripperQ = float(gripper_q)

    arm.setArmCmd(arm.q, arm.qd, arm.tau)
    arm.setGripperCmd(arm.gripperQ, arm.gripperQd, arm.gripperTau)
    arm.sendRecv()


def move_to_target(arm, q_target, hold_time, name):
    dt = float(arm._ctrlComp.dt)
    steps = max(1, int(round(MOVE_TIME / dt)))

    q_start = get_q(arm).copy()
    prev_q_cmd = q_start.copy()

    print(f"\n=== move to {name} ===")
    print("q_start =", np.round(q_start, 3))
    print("q_target=", np.round(q_target, 3))

    max_err = np.zeros(6, dtype=np.float32)

    for i in range(steps):
        alpha = float(i + 1) / float(steps)
        q_ref = (1.0 - alpha) * q_start + alpha * q_target

        dq = np.clip(q_ref - prev_q_cmd, -STEP_CLIP, STEP_CLIP)
        q_cmd = prev_q_cmd + dq

        send_pd(arm, q_cmd, GRIPPER_Q)

        q_meas = get_q(arm)
        qd_meas = get_qd(arm)
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

    print(f"--- hold {name} for {hold_time:.1f}s ---")
    hold_steps = max(1, int(round(hold_time / dt)))
    for i in range(hold_steps):
        send_pd(arm, q_target, GRIPPER_Q)
        q_meas = get_q(arm)
        qd_meas = get_qd(arm)
        err = q_target - q_meas
        max_err = np.maximum(max_err, np.abs(err))

        if i % PRINT_EVERY == 0 or i == hold_steps - 1:
            print(
                f"[hold {name} {i+1:04d}/{hold_steps}] "
                f"err={np.round(err, 3)} "
                f"qd={np.round(qd_meas, 3)}"
            )

        time.sleep(dt)

    print(f"max_abs_err({name}) =", np.round(max_err, 4))


def main():
    np.set_printoptions(precision=3, suppress=True)
    print("Press Ctrl+C to stop.")
    print("KP =", KP)
    print("KD =", KD)

    arm = unitree_arm_interface.ArmInterface(hasGripper=HAS_GRIPPER)
    arm.setFsmLowcmd()
    time.sleep(0.02)

    for _ in range(20):
        arm.sendRecv()
        time.sleep(0.002)

    set_gain(arm, KP, KD)

    print("current q =", np.round(get_q(arm), 3))

    move_to_target(arm, ZERO_Q, HOLD_TIME, "ZERO")
    move_to_target(arm, DEFAULT_Q, HOLD_TIME, "DEFAULT")
    move_to_target(arm, ZERO_Q, HOLD_TIME, "ZERO_BACK")

    print("\nDone.")


if __name__ == "__main__":
    main()