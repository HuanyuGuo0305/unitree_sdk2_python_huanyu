#!/usr/bin/env python3
"""
Minimal Z1 SDK2 move test via DDS.

What this script does:
1. Subscribe to rt/z1/lowstate
2. Publish MotorCmds_ to rt/z1/lowcmd
3. Move joints 1..6 from current position to:
       [0.0, 1.48, -1.0, -0.54, 0.0, 0.0]
   while keeping the gripper at its current position
4. Print detailed diagnostics during motion

Run from repository root:
    cd ~/Workspace_huanyu/unitree_sdk2_python_huanyu
    python3 deploy/test_z1.py
"""

import os
import sys
import time
import threading
from typing import Optional, List

# Make sure local package import works when run as: python3 deploy/test_z1.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmd_, MotorCmds_, MotorStates_


# =========================
# User-configurable params
# =========================

STATE_TOPIC = "rt/z1/lowstate"
CMD_TOPIC = "rt/z1/lowcmd"
NETWORK_INTERFACE = "eno1"

CONTROL_HZ = 500.0
DT = 1.0 / CONTROL_HZ

NUM_MOTORS = 7  # 6 arm joints + 1 gripper

# IMPORTANT:
# We still do not know with full certainty which mode value is the correct one
# for your z1_udp_service build. From your previous tests, mode=1 produced no motion.
# Try 1 first if you want to reproduce, but most likely you should scan a few values.
#
# Suggested candidates to try manually:
#   0, 1, 2, 10
#
# Change this value and rerun if the arm does not move.
CMD_MODE = 10

# Target arm posture (6 joints only)
TARGET_ARM_Q = [0.0, 1.48, -1.0, -0.54, 0.0, 0.0]

# Move duration and hold duration
MOVE_DURATION_SEC = 4.0
HOLD_DURATION_SEC = 3.0

# PD gains
KP_ARM = [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]
KD_ARM = [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]

KP_GRIPPER = 2.0
KD_GRIPPER = 0.1

# Print interval
PRINT_EVERY_STEPS = 100


# =========================
# Shared state container
# =========================

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_msg: Optional[MotorStates_] = None
        self.recv_count: int = 0

    def update(self, msg: MotorStates_):
        with self.lock:
            self.latest_msg = msg
            self.recv_count += 1

    def get(self):
        with self.lock:
            return self.latest_msg, self.recv_count


SHARED = SharedState()


# =========================
# DDS callback
# =========================

def state_callback(msg: MotorStates_):
    SHARED.update(msg)


# =========================
# Helpers
# =========================

def wait_for_first_state(timeout_sec: float = 10.0) -> MotorStates_:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        msg, _ = SHARED.get()
        if msg is not None:
            return msg
        time.sleep(0.01)
    raise TimeoutError(f"Timeout waiting for first state on {STATE_TOPIC}")


def extract_q(msg: MotorStates_) -> List[float]:
    return [float(s.q) for s in msg.states]


def extract_dq(msg: MotorStates_) -> List[float]:
    return [float(s.dq) for s in msg.states]


def extract_mode(msg: MotorStates_) -> List[int]:
    return [int(s.mode) for s in msg.states]


def extract_tau_est(msg: MotorStates_) -> List[float]:
    return [float(s.tau_est) for s in msg.states]


def extract_lost(msg: MotorStates_) -> List[int]:
    return [int(s.lost) for s in msg.states]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def build_motor_cmd(
    mode: int,
    q: float,
    dq: float,
    tau: float,
    kp: float,
    kd: float,
) -> MotorCmd_:
    # NOTE:
    # reserve is NOT an int.
    # From your generated IDL:
    #   reserve: types.array[types.uint32, 3]
    # So it must be a length-3 array/list.
    return MotorCmd_(
        mode=mode,
        q=float(q),
        dq=float(dq),
        tau=float(tau),
        kp=float(kp),
        kd=float(kd),
        reserve=[0, 0, 0],
    )


def build_cmd_msg(
    q_cmd: List[float],
    dq_cmd: List[float],
    tau_cmd: List[float],
    kp_cmd: List[float],
    kd_cmd: List[float],
    mode_cmd: List[int],
) -> MotorCmds_:
    if not (
        len(q_cmd) == len(dq_cmd) == len(tau_cmd) == len(kp_cmd) == len(kd_cmd) == len(mode_cmd) == NUM_MOTORS
    ):
        raise ValueError("All command arrays must have length NUM_MOTORS")

    cmds = []
    for i in range(NUM_MOTORS):
        cmds.append(
            build_motor_cmd(
                mode=mode_cmd[i],
                q=q_cmd[i],
                dq=dq_cmd[i],
                tau=tau_cmd[i],
                kp=kp_cmd[i],
                kd=kd_cmd[i],
            )
        )

    return MotorCmds_(cmds=cmds)


def format_vec(v: List[float], n: int = 6) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in v[:n]) + "]"


# =========================
# Main
# =========================

def main():
    print(f"[DEBUG] python executable: {sys.executable}")
    print(f"[DEBUG] python version   : {sys.version}")
    print(f"[DEBUG] cwd              : {os.getcwd()}")
    print(f"[DEBUG] CYCLONEDDS_URI   : {os.environ.get('CYCLONEDDS_URI')}")
    print(f"[DEBUG] PYTHONPATH       : {os.environ.get('PYTHONPATH')}")

    print("=" * 70)
    print("Z1 SDK2 minimal move test")
    print(f"State topic : {STATE_TOPIC}")
    print(f"Cmd topic   : {CMD_TOPIC}")
    print(f"Interface   : {NETWORK_INTERFACE}")
    print(f"Control Hz  : {CONTROL_HZ}")
    print(f"CMD_MODE    : {CMD_MODE}")
    print("=" * 70)

    # IMPORTANT:
    # In your environment, ChannelFactoryInitialize returns None on success.
    # So do NOT check for truth value. Just call it.
    print("[INFO] Initializing ChannelFactory ...")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)
    print("[INFO] ChannelFactoryInitialize done")

    sub = ChannelSubscriber(STATE_TOPIC, MotorStates_)
    sub.Init(state_callback, 10)

    pub = ChannelPublisher(CMD_TOPIC, MotorCmds_)
    pub.Init()

    print("[INFO] Waiting for first state message ...")
    first_msg = wait_for_first_state(timeout_sec=10.0)

    q_init = extract_q(first_msg)
    dq_init = extract_dq(first_msg)
    mode_init = extract_mode(first_msg)
    lost_init = extract_lost(first_msg)

    print(f"[INFO] Initial q     = {format_vec(q_init, 7)}")
    print(f"[INFO] Initial dq    = {format_vec(dq_init, 7)}")
    print(f"[INFO] Initial mode  = {mode_init}")
    print(f"[INFO] Initial lost  = {lost_init}")

    # Keep gripper at current position
    target_q = TARGET_ARM_Q[:] + [q_init[6]]

    print(f"[INFO] Target q      = {format_vec(target_q, 7)}")

    mode_cmd = [CMD_MODE] * NUM_MOTORS
    kp_cmd = KP_ARM[:] + [KP_GRIPPER]
    kd_cmd = KD_ARM[:] + [KD_GRIPPER]
    tau_cmd = [0.0] * NUM_MOTORS

    print(f"[INFO] Command mode  = {mode_cmd}")
    print(f"[INFO] KP           = {kp_cmd}")
    print(f"[INFO] KD           = {kd_cmd}")

    move_steps = int(MOVE_DURATION_SEC * CONTROL_HZ)
    hold_steps = int(HOLD_DURATION_SEC * CONTROL_HZ)

    print("[INFO] Start moving ... Press Ctrl+C to stop.")

    try:
        # ==========
        # Move phase
        # ==========
        for step in range(move_steps):
            latest_msg, recv_count = SHARED.get()
            if latest_msg is None:
                time.sleep(DT)
                continue

            q_now = extract_q(latest_msg)
            dq_now = extract_dq(latest_msg)

            # Linear interpolation from initial q to target q
            alpha = float(step + 1) / float(move_steps)
            alpha = clamp(alpha, 0.0, 1.0)

            q_cmd = []
            dq_cmd = []
            for i in range(NUM_MOTORS):
                q_des = (1.0 - alpha) * q_init[i] + alpha * target_q[i]
                q_cmd.append(q_des)

            # Feedforward dq from finite slope of interpolation
            for i in range(NUM_MOTORS):
                dq_des = (target_q[i] - q_init[i]) / MOVE_DURATION_SEC
                dq_cmd.append(dq_des)

            msg = build_cmd_msg(
                q_cmd=q_cmd,
                dq_cmd=dq_cmd,
                tau_cmd=tau_cmd,
                kp_cmd=kp_cmd,
                kd_cmd=kd_cmd,
                mode_cmd=mode_cmd,
            )

            ok = pub.Write(msg, 1.0)
            print("[DEBUG] pub.Write ->", ok)

            if step % PRINT_EVERY_STEPS == 0:
                mode_now = extract_mode(latest_msg)
                tau_now = extract_tau_est(latest_msg)
                lost_now = extract_lost(latest_msg)

                err = [target_q[i] - q_now[i] for i in range(NUM_MOTORS)]
                max_abs_err = max(abs(e) for e in err[:6])

                print(
                    f"[MOVE {step:04d}] "
                    f"write_ok={ok} | "
                    f"recv_count={recv_count} | "
                    f"max_abs_err={max_abs_err:.6f} | "
                    f"state_mode={mode_now} | "
                    f"lost={lost_now} | "
                    f"tau_est={format_vec(tau_now, 6)} | "
                    f"q_now={format_vec(q_now, 6)}"
                )

            time.sleep(DT)

        print("[INFO] Reached trajectory end, holding target ...")

        # ==========
        # Hold phase
        # ==========
        for step in range(hold_steps):
            latest_msg, recv_count = SHARED.get()
            if latest_msg is None:
                time.sleep(DT)
                continue

            q_now = extract_q(latest_msg)

            q_cmd = target_q[:]
            dq_cmd = [0.0] * NUM_MOTORS

            msg = build_cmd_msg(
                q_cmd=q_cmd,
                dq_cmd=dq_cmd,
                tau_cmd=tau_cmd,
                kp_cmd=kp_cmd,
                kd_cmd=kd_cmd,
                mode_cmd=mode_cmd,
            )

            ok = pub.Write(msg, 1.0)
            print("[DEBUG] pub.Write ->", ok)

            if step % PRINT_EVERY_STEPS == 0:
                mode_now = extract_mode(latest_msg)
                tau_now = extract_tau_est(latest_msg)
                lost_now = extract_lost(latest_msg)

                err = [target_q[i] - q_now[i] for i in range(NUM_MOTORS)]
                max_abs_err = max(abs(e) for e in err[:6])

                print(
                    f"[HOLD {step:04d}] "
                    f"write_ok={ok} | "
                    f"recv_count={recv_count} | "
                    f"max_abs_err={max_abs_err:.6f} | "
                    f"state_mode={mode_now} | "
                    f"lost={lost_now} | "
                    f"tau_est={format_vec(tau_now, 6)} | "
                    f"q_now={format_vec(q_now, 6)}"
                )

            time.sleep(DT)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received, exiting...")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()