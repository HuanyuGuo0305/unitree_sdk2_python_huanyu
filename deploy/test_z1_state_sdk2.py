#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal Z1 SDK2 low-level test.

What this script does:
1. Subscribe to rt/z1/lowstate
2. Publish to rt/z1/lowcmd
3. Wait for the first motor state message
4. Record the current joint positions as hold targets
5. Continuously send low-level PD commands to hold the current pose

Safety notes:
- Clear the workspace before running
- Start with conservative gains
- Test holding only, before adding any policy output
"""

import time
import signal
import threading
from typing import Any, List, Optional
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_, MotorCmd_

TOPIC_STATE = "rt/z1/lowstate"
TOPIC_CMD = "rt/z1/lowcmd"

NUM_MOTORS = 7          # 6 arm joints + 1 gripper
CTRL_HZ = 250.0
DT = 1.0 / CTRL_HZ

# Conservative gains for first test
DEFAULT_KP_ARM = 5.0
DEFAULT_KD_ARM = 0.2
DEFAULT_KP_GRIPPER = 2.0
DEFAULT_KD_GRIPPER = 0.1

RUNNING = True


def on_sigint(sig, frame):
    global RUNNING
    RUNNING = False
    print("\n[INFO] Ctrl+C received, exiting...")


signal.signal(signal.SIGINT, on_sigint)


def first_attr(obj: Any, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def has_attr(obj: Any, names: List[str]) -> Optional[str]:
    for n in names:
        if hasattr(obj, n):
            return n
    return None


def set_first_attr(obj: Any, names: List[str], value) -> bool:
    for n in names:
        if hasattr(obj, n):
            try:
                setattr(obj, n, value)
                return True
            except Exception:
                pass
    return False


def try_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return -1


def pretty_dir(obj: Any) -> List[str]:
    return sorted([x for x in dir(obj) if not x.startswith("__")])


class StateBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_msg = None
        self.recv_count = 0
        self.first_print_done = False

    def callback(self, msg: MotorStates_):
        with self.lock:
            self.latest_msg = msg
            self.recv_count += 1

            if not self.first_print_done:
                self.first_print_done = True
                print("[INFO] First state message received")
                print("[INFO] Message type:", type(msg))
                print("[INFO] Message fields:", pretty_dir(msg))

                states = first_attr(msg, ["states", "motor_states", "state"], None)
                if states is not None:
                    print("[INFO] len(states) =", try_len(states))
                    if try_len(states) > 0:
                        s0 = states[0]
                        print("[INFO] state[0] type:", type(s0))
                        print("[INFO] state[0] fields:", pretty_dir(s0))
                        for cand in ["q", "pos", "position", "dq", "qd", "vel", "velocity", "tau", "torque"]:
                            if hasattr(s0, cand):
                                print(f"[INFO] state[0].{cand} =", getattr(s0, cand))
                else:
                    print("[WARN] Could not find state array field in MotorStates_")

    def get_latest(self):
        with self.lock:
            return self.latest_msg, self.recv_count


def get_state_array(msg: Any):
    return first_attr(msg, ["states", "motor_states", "state"], None)


def get_cmd_array(msg: Any):
    return first_attr(msg, ["cmds", "motor_cmds", "motor_cmd", "cmd"], None)


def get_state_q_list(msg: Any) -> List[float]:
    states = get_state_array(msg)
    if states is None:
        raise RuntimeError("Cannot find state array in MotorStates_ message")

    q_list = []
    for i in range(min(NUM_MOTORS, len(states))):
        s = states[i]
        q = first_attr(s, ["q", "pos", "position"], None)
        if q is None:
            raise RuntimeError(
                f"Cannot find q/pos/position in states[{i}], fields = {pretty_dir(s)}"
            )
        q_list.append(float(q))
    return q_list


def get_state_dq_list(msg: Any) -> List[float]:
    states = get_state_array(msg)
    if states is None:
        return [0.0] * NUM_MOTORS

    dq_list = []
    for i in range(min(NUM_MOTORS, len(states))):
        s = states[i]
        dq = first_attr(s, ["dq", "qd", "vel", "velocity"], 0.0)
        dq_list.append(float(dq))
    return dq_list


def build_cmd_msg() -> MotorCmds_:
    msg = MotorCmds_()

    # MotorCmds_.cmds is empty by default, so we must create 7 MotorCmd_ objects manually.
    if hasattr(msg, "cmds"):
        msg.cmds = [
            MotorCmd_(
                mode=1,
                q=0.0,
                dq=0.0,
                tau=0.0,
                kp=0.0,
                kd=0.0,
                reserve=[0, 0, 0],
            )
            for _ in range(NUM_MOTORS)
        ]
        cmds = msg.cmds
    elif hasattr(msg, "motor_cmds"):
        msg.motor_cmds = [MotorCmd_() for _ in range(NUM_MOTORS)]
        cmds = msg.motor_cmds
    elif hasattr(msg, "cmd"):
        msg.cmd = [MotorCmd_() for _ in range(NUM_MOTORS)]
        cmds = msg.cmd
    else:
        raise RuntimeError(
            f"Cannot find command array field in MotorCmds_, fields = {pretty_dir(msg)}"
        )

    # Print the command message structure once for debugging
    if not hasattr(build_cmd_msg, "_printed"):
        build_cmd_msg._printed = True
        print("[INFO] MotorCmds_ fields:", pretty_dir(msg))
        if len(cmds) > 0:
            print("[INFO] MotorCmd_ fields:", pretty_dir(cmds[0]))

    return msg


def fill_cmd_msg(
    cmd_msg: Any,
    q_des: List[float],
    dq_des: List[float],
    kp_list: List[float],
    kd_list: List[float],
    tau_ff_list: List[float],
):
    cmds = get_cmd_array(cmd_msg)
    if cmds is None:
        raise RuntimeError(
            f"Cannot find command array field in MotorCmds_, fields = {pretty_dir(cmd_msg)}"
        )

    if len(cmds) < NUM_MOTORS:
        raise RuntimeError(f"Command array length {len(cmds)} < {NUM_MOTORS}")

    for i in range(NUM_MOTORS):
        c = cmds[i]

        # Try to enable the motor command if such a field exists
        mode_name = has_attr(c, ["mode", "enable"])
        if mode_name is not None:
            try:
                setattr(c, mode_name, 1)
            except Exception:
                pass

        ok_q = set_first_attr(c, ["q", "pos", "position"], float(q_des[i]))
        ok_dq = set_first_attr(c, ["dq", "qd", "vel", "velocity"], float(dq_des[i]))
        ok_kp = set_first_attr(c, ["kp", "k_p", "Kp", "K_P"], float(kp_list[i]))
        ok_kd = set_first_attr(c, ["kd", "k_d", "Kd", "K_D"], float(kd_list[i]))
        ok_tau = set_first_attr(c, ["tau", "torque", "tau_ff", "t"], float(tau_ff_list[i]))

        if not ok_q:
            raise RuntimeError(f"Cannot set q for cmds[{i}], fields = {pretty_dir(c)}")
        if not ok_dq:
            raise RuntimeError(f"Cannot set dq for cmds[{i}], fields = {pretty_dir(c)}")
        if not ok_kp:
            raise RuntimeError(f"Cannot set kp for cmds[{i}], fields = {pretty_dir(c)}")
        if not ok_kd:
            raise RuntimeError(f"Cannot set kd for cmds[{i}], fields = {pretty_dir(c)}")
        if not ok_tau:
            raise RuntimeError(f"Cannot set tau for cmds[{i}], fields = {pretty_dir(c)}")

    return cmd_msg


def main():
    net_if = "eno1"

    print("=" * 70)
    print("Z1 SDK2 hold-current test")
    print(f"State topic : {TOPIC_STATE}")
    print(f"Cmd topic   : {TOPIC_CMD}")
    print(f"Interface   : {net_if}")
    print(f"Control Hz  : {CTRL_HZ}")
    print("=" * 70)

    ok = ChannelFactoryInitialize(0, net_if)
    if ok is False:
        raise RuntimeError("ChannelFactoryInitialize failed")

    state_buf = StateBuffer()

    sub = ChannelSubscriber(TOPIC_STATE, MotorStates_)
    sub.Init(state_buf.callback, 10)

    pub = ChannelPublisher(TOPIC_CMD, MotorCmds_)
    pub.Init()

    print(f"[INFO] Waiting for first message on {TOPIC_STATE} ...")
    t0 = time.time()
    while RUNNING:
        msg, cnt = state_buf.get_latest()
        if msg is not None:
            break
        if time.time() - t0 > 10.0:
            raise TimeoutError(f"Timeout waiting for {TOPIC_STATE}")
        time.sleep(0.01)

    if not RUNNING:
        return

    first_msg, _ = state_buf.get_latest()
    hold_q = get_state_q_list(first_msg)
    hold_dq = [0.0] * NUM_MOTORS
    tau_ff = [0.0] * NUM_MOTORS

    kp = [DEFAULT_KP_ARM] * 6 + [DEFAULT_KP_GRIPPER]
    kd = [DEFAULT_KD_ARM] * 6 + [DEFAULT_KD_GRIPPER]

    print("[INFO] Hold target q =", ["%.6f" % x for x in hold_q])
    print("[INFO] kp =", kp)
    print("[INFO] kd =", kd)
    print("[INFO] Entering hold loop. Press Ctrl+C to stop.")

    step = 0
    next_t = time.time()

    while RUNNING:
        latest_msg, recv_count = state_buf.get_latest()
        if latest_msg is None:
            print("[WARN] No state received in this cycle")
            time.sleep(DT)
            continue

        q_now = get_state_q_list(latest_msg)
        dq_now = get_state_dq_list(latest_msg)

        cmd_msg = build_cmd_msg()
        fill_cmd_msg(
            cmd_msg=cmd_msg,
            q_des=hold_q,
            dq_des=hold_dq,
            kp_list=kp,
            kd_list=kd,
            tau_ff_list=tau_ff,
        )

        ok = pub.Write(cmd_msg)
        if ok is False:
            print("[WARN] pub.Write returned False")

        if step % 100 == 0:
            err = [hold_q[i] - q_now[i] for i in range(NUM_MOTORS)]
            max_abs_err = max(abs(x) for x in err)
            print(
                f"[{step:06d}] recv_count={recv_count} | "
                f"max_abs_err={max_abs_err:.6f} | "
                f"q_now={[round(x, 4) for x in q_now]} | "
                f"dq_now={[round(x, 4) for x in dq_now]}"
            )

        step += 1
        next_t += DT
        sleep_t = next_t - time.time()
        if sleep_t > 0:
            time.sleep(sleep_t)
        else:
            next_t = time.time()

    print("[INFO] Stopped.")


if __name__ == "__main__":
    main()