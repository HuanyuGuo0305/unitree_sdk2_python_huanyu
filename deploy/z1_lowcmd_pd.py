import os
import sys
import time
import yaml
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def resolve_path(path_str: str, project_root: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(project_root, path_str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--kp", type=float, default=40.0)
    parser.add_argument("--kd", type=float, default=3.0)
    parser.add_argument("--duration", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    z1_sdk_lib = resolve_path(cfg["z1_sdk_lib"], PROJECT_ROOT)
    if z1_sdk_lib not in sys.path:
        sys.path.insert(0, z1_sdk_lib)

    import unitree_arm_interface

    np.set_printoptions(precision=4, suppress=True)

    has_gripper = bool(cfg.get("z1_has_gripper", True))
    target_pos = np.array(cfg["default_arm_pos"], dtype=np.float32).reshape(6,)
    target_gripper = float(cfg.get("default_gripper_pos", 0.0))

    print("=" * 80)
    print("[TEST] official example + external gain")
    print("[TEST] z1_sdk_lib     =", z1_sdk_lib)
    print("[TEST] target_pos     =", target_pos)
    print("[TEST] target_gripper =", target_gripper)
    print("[TEST] kp             =", args.kp)
    print("[TEST] kd             =", args.kd)
    print("=" * 80)

    arm = unitree_arm_interface.ArmInterface(hasGripper=has_gripper)
    armModel = arm._ctrlComp.armModel
    arm.setFsmLowcmd()

    duration = args.duration
    dt = float(arm._ctrlComp.dt)

    last_pos = np.array(arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
    kp = [float(args.kp)] * 6 + [20.0]
    kd = [float(args.kd)] * 6 + [2000.0]

    print("[TEST] dt      =", dt)
    print("[TEST] FSM     =", arm.getCurrentState())
    print("[TEST] q_init  =", last_pos)
    print("[TEST] q_delta =", target_pos - last_pos)

    input("[TEST] Press Enter to start...")

    arm._ctrlComp.lowcmd.setControlGain(kp, kd)

    for i in range(duration):
        alpha = float(i) / float(duration)

        q_cmd = last_pos * (1.0 - alpha) + target_pos * alpha
        qd_cmd = np.zeros(6, dtype=np.float32)
        tau_cmd = np.zeros(6, dtype=np.float32)

        arm.q = q_cmd
        arm.qd = qd_cmd
        arm.tau = tau_cmd
        arm.gripperQ = target_gripper

        arm.setArmCmd(arm.q, arm.qd, arm.tau)
        arm.setGripperCmd(arm.gripperQ, arm.gripperQd, arm.gripperTau)
        arm.sendRecv()

        fsm = arm.getCurrentState()

        if (i % 20 == 0) or (i == duration - 1) or (fsm != unitree_arm_interface.ArmFSMState.LOWCMD):
            q_meas = np.array(arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
            qd_meas = np.array(arm.lowstate.getQd(), dtype=np.float32).reshape(6,)
            tau_meas = np.array(arm.lowstate.getTau(), dtype=np.float32).reshape(6,)
            err = q_cmd - q_meas

            print(
                f"[{i+1:04d}/{duration}] "
                f"FSM={fsm} | "
                f"q_cmd={np.round(q_cmd, 3)} | "
                f"q_meas={np.round(q_meas, 3)} | "
                f"err={np.round(err, 3)} | "
                f"qd_meas={np.round(qd_meas, 3)} | "
                f"tau_meas={np.round(tau_meas, 3)}"
            )

        if fsm != unitree_arm_interface.ArmFSMState.LOWCMD:
            print(f"[ERROR] FSM dropped to {fsm} at step {i+1}")
            break

        time.sleep(dt)

    arm.loopOn()
    arm.backToStart()
    arm.loopOff()


if __name__ == "__main__":
    main()