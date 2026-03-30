#!/usr/bin/env python3
"""
Z1 SDK2 lowcmd mode scanner.

What this script does:
1. Subscribe to rt/z1/lowstate
2. Publish MotorCmds_ to rt/z1/lowcmd
3. For each candidate mode, hold the current joint positions for a few seconds
4. Print state diagnostics to help determine whether the mode is accepted

Run:
    python3 deploy/test_z1_mode_scan.py
"""

import os
import sys
import time
import threading
from typing import Optional, List

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

STATE_TOPIC = "rt/z1/lowstate"
CMD_TOPIC = "rt/z1/lowcmd"
NETWORK_INTERFACE = "eno1"

CONTROL_HZ = 500.0
DT = 1.0 / CONTROL_HZ
NUM_MOTORS = 7

MODE_CANDIDATES = [0, 1, 2, 5, 10]
TEST_DURATION_SEC = 3.0

KP_ARM = [40.0] * 6
KD_ARM = [3.0] * 6
KP_GRIPPER = 2.0
KD_GRIPPER = 0.1

PRINT_EVERY_STEPS = 100


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_msg: Optional[MotorStates_] = None
        self.recv_count = 0

    def update(self, msg: MotorStates_):
        with self.lock:
            self.latest_msg = msg
            self.recv_count += 1

    def get(self):
        with self.lock:
            return self.latest_msg, self.recv_count


shared = SharedState()


def state_callback(msg: MotorStates_):
    shared.update(msg)


def wait_for_first_state(timeout_sec: float = 10.0) -> MotorStates_:
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        msg, _ = shared.get()
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


def build_motor_cmd(mode: int, q: float, dq: float, tau: float, kp: float, kd: float) -> MotorCmd_:
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


def format_vec(v: List[float], n: int = 7) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in v[:n]) + "]"


def run_mode_test(pub: ChannelPublisher, mode_value: int):
    latest_msg, _ = shared.get()
    if latest_msg is None:
        raise RuntimeError("No state available before mode test")

    q_ref = extract_q(latest_msg)
    q_cmd = q_ref[:]
    dq_cmd = [0.0] * NUM_MOTORS
    tau_cmd = [0.0] * NUM_MOTORS
    kp_cmd = KP_ARM[:] + [KP_GRIPPER]
    kd_cmd = KD_ARM[:] + [KD_GRIPPER]
    mode_cmd = [mode_value] * NUM_MOTORS

    total_steps = int(TEST_DURATION_SEC * CONTROL_HZ)

    print("=" * 80)
    print(f"[TEST] mode={mode_value}")
    print(f"[TEST] q_ref={format_vec(q_ref)}")

    for step in range(total_steps):
        msg = build_cmd_msg(
            q_cmd=q_cmd,
            dq_cmd=dq_cmd,
            tau_cmd=tau_cmd,
            kp_cmd=kp_cmd,
            kd_cmd=kd_cmd,
            mode_cmd=mode_cmd,
        )

        ok = pub.Write(msg, 1.0)

        latest_msg, recv_count = shared.get()
        if latest_msg is None:
            time.sleep(DT)
            continue

        if step % PRINT_EVERY_STEPS == 0:
            q_now = extract_q(latest_msg)
            dq_now = extract_dq(latest_msg)
            mode_now = extract_mode(latest_msg)
            tau_now = extract_tau_est(latest_msg)
            lost_now = extract_lost(latest_msg)

            pos_err = [q_cmd[i] - q_now[i] for i in range(NUM_MOTORS)]
            max_abs_err = max(abs(x) for x in pos_err[:6])

            print(
                f"[mode={mode_value} step={step:04d}] "
                f"write_ok={ok} | recv_count={recv_count} | "
                f"state_mode={mode_now} | lost={lost_now} | "
                f"max_abs_err={max_abs_err:.6f} | "
                f"q_now={format_vec(q_now)} | "
                f"dq_now={format_vec(dq_now)} | "
                f"tau_est={format_vec(tau_now)}"
            )

        time.sleep(DT)


def main():
    print("[INFO] Initializing DDS channel factory...")
    ChannelFactoryInitialize(0, NETWORK_INTERFACE)

    sub = ChannelSubscriber(STATE_TOPIC, MotorStates_)
    sub.Init(state_callback, 10)

    pub = ChannelPublisher(CMD_TOPIC, MotorCmds_)
    pub.Init()

    print("[INFO] Waiting for first lowstate...")
    first_msg = wait_for_first_state()

    print(f"[INFO] Initial q    : {format_vec(extract_q(first_msg))}")
    print(f"[INFO] Initial dq   : {format_vec(extract_dq(first_msg))}")
    print(f"[INFO] Initial mode : {extract_mode(first_msg)}")
    print(f"[INFO] Initial lost : {extract_lost(first_msg)}")

    try:
        for mode_value in MODE_CANDIDATES:
            run_mode_test(pub, mode_value)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    print("[INFO] Done")


if __name__ == "__main__":
    main()