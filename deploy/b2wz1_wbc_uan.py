#!/usr/bin/env python3
"""
Real-robot deployment for the B2W + Z1 low-level WBC policy trained with UAN.

Important:
  * UAN is NOT deployed on the real robot.
  * The WBC policy runs at 50 Hz.
  * B2W lowcmd and Z1 lowcmd are refreshed at 500 Hz.
  * Z1 receives position targets through its firmware position-PD loop.
  * There is NO arm target rate limiter.
  * EE commands are direct random sample-and-hold only (no sequential trajectory class).

Control path:
    WBC ONNX -> arm q_des @ 50 Hz
             -> zero-order hold @ 500 Hz
             -> Z1 firmware position PD
             -> real Z1

The firmware gains are audited against the training-space gains using Unitree's
documented communication scaling:
    Kp_effective = kp_fw * 25.6
    Kd_effective = kd_fw * 0.0128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import onnxruntime as ort
import yaml


_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DEPLOY_DIR, ".."))
for _p in (_PROJECT_ROOT, _DEPLOY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_PROJECT_ROOT, path))


def _sleep_until(deadline: float) -> float:
    now = time.perf_counter()
    dt = deadline - now
    if dt > 0.0:
        time.sleep(dt)
    return max(0.0, time.perf_counter() - deadline)


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v from world to body by inverse unit quaternion q=[w,x,y,z]."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise RuntimeError("Invalid IMU quaternion norm.")
    q = q / n
    w = q[0]
    xyz = q[1:4]
    # inverse rotation R(q)^T v
    return (
        v * (2.0 * w * w - 1.0)
        - 2.0 * w * np.cross(xyz, v)
        + 2.0 * xyz * np.dot(xyz, v)
    ).astype(np.float32)


class PresampledKeypointsDirectCommandPLBReal:
    """Direct random sample-and-hold EE keypoint command in PLB frame."""

    def __init__(
        self,
        file_path: str,
        policy_dt: float,
        cycle_duration_s: float = 8.0,
        seed: int = 0,
    ):
        table = np.load(file_path).astype(np.float32)
        if table.ndim != 2 or table.shape[1] != 9:
            raise ValueError(
                f"Expected EE command table shape (N,9), got {table.shape} from {file_path}"
            )
        if len(table) == 0:
            raise ValueError("EE command table is empty.")

        self.table = table
        self.rng = np.random.default_rng(seed)
        self.cycle_steps = max(1, int(round(float(cycle_duration_s) / float(policy_dt))))
        self.step_in_cycle = 0
        self.current = self.table[0].copy()

    def reset(self, sample_first: bool = True):
        self.step_in_cycle = 0
        if sample_first:
            self.current = self.table[int(self.rng.integers(0, len(self.table)))].copy()

    def update(self) -> np.ndarray:
        self.step_in_cycle += 1
        if self.step_in_cycle >= self.cycle_steps:
            self.current = self.table[int(self.rng.integers(0, len(self.table)))].copy()
            self.step_in_cycle = 0
        return self.current.copy()


@dataclass
class B2WState:
    leg_q_policy: np.ndarray
    leg_qd_policy: np.ndarray
    wheel_qd_policy: np.ndarray
    imu_quat_wxyz: np.ndarray
    imu_gyro: np.ndarray


class B2WLowLevel:
    """
    Thin B2W low-level DDS adapter.

    Policy-space joint order is intentionally explicit and configured by motor
    indices in YAML. This avoids silently assuming that SDK motor order equals
    the policy/IsaacLab order.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        b = cfg["b2w"]

        self.interface = str(b["network_interface"])
        self.lowcmd_topic = str(b.get("lowcmd_topic", "rt/lowcmd"))
        self.lowstate_topic = str(b.get("lowstate_topic", "rt/lowstate"))
        self.release_motion_service = bool(b.get("release_motion_service", True))

        self.leg_idx = np.asarray(b["policy_leg_motor_indices"], dtype=np.int32)
        self.wheel_idx = np.asarray(b["policy_wheel_motor_indices"], dtype=np.int32)

        self.leg_kp = np.asarray(b["leg_kp"], dtype=np.float32)
        self.leg_kd = np.asarray(b["leg_kd"], dtype=np.float32)
        self.wheel_kd = np.asarray(b["wheel_kd"], dtype=np.float32)
        self.wheel_vel_limit = np.asarray(b["wheel_velocity_limits"], dtype=np.float32)

        if self.leg_idx.shape != (12,):
            raise ValueError("b2w.policy_leg_motor_indices must contain 12 entries.")
        if self.wheel_idx.shape != (4,):
            raise ValueError("b2w.policy_wheel_motor_indices must contain 4 entries.")
        if len(set(self.leg_idx.tolist() + self.wheel_idx.tolist())) != 16:
            raise ValueError("B2W leg/wheel motor indices must be unique.")
        if np.any(self.leg_idx < 0) or np.any(self.leg_idx >= 20):
            raise ValueError("B2W leg motor index out of [0,19].")
        if np.any(self.wheel_idx < 0) or np.any(self.wheel_idx >= 20):
            raise ValueError("B2W wheel motor index out of [0,19].")

        for name, arr, shape in [
            ("leg_kp", self.leg_kp, (12,)),
            ("leg_kd", self.leg_kd, (12,)),
            ("wheel_kd", self.wheel_kd, (4,)),
            ("wheel_velocity_limits", self.wheel_vel_limit, (4,)),
        ]:
            if arr.shape != shape:
                raise ValueError(f"b2w.{name} must have shape {shape}, got {arr.shape}")

        self.low_state = None
        self.low_cmd = None
        self.publisher = None
        self.subscriber = None
        self.crc = None
        self._sdk = None

    def connect(self):
        # Lazy imports allow --dry-run on a machine without the robot SDK.
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._sdk = {
            "ChannelPublisher": ChannelPublisher,
            "ChannelSubscriber": ChannelSubscriber,
            "LowCmd_": LowCmd_,
            "LowState_": LowState_,
        }

        ChannelFactoryInitialize(0, self.interface)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0

        for i in range(20):
            mc = self.low_cmd.motor_cmd[i]
            mc.mode = 0x01
            mc.q = POS_STOP_F
            mc.kp = 0.0
            mc.dq = VEL_STOP_F
            mc.kd = 0.0
            mc.tau = 0.0

        self.crc = CRC()

        self.publisher = ChannelPublisher(self.lowcmd_topic, LowCmd_)
        self.publisher.Init()

        self.subscriber = ChannelSubscriber(self.lowstate_topic, LowState_)
        self.subscriber.Init(self._lowstate_handler, 10)

        deadline = time.time() + float(self.cfg["b2w"].get("state_timeout_s", 5.0))
        while self.low_state is None and time.time() < deadline:
            time.sleep(0.01)
        if self.low_state is None:
            raise RuntimeError(f"No B2W lowstate received on {self.lowstate_topic}")

        if self.release_motion_service:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )
            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()
            status, result = msc.CheckMode()
            while result.get("name", ""):
                print(f"[B2W] releasing active motion service: {result}")
                ret = msc.ReleaseMode()
                if ret != 0:
                    raise RuntimeError(f"MotionSwitcher ReleaseMode failed: {ret}")
                time.sleep(1.0)
                status, result = msc.CheckMode()

        print("[B2W] low-level DDS connected.")
        print(f"[B2W] policy leg motor indices  : {self.leg_idx.tolist()}")
        print(f"[B2W] policy wheel motor indices: {self.wheel_idx.tolist()}")

    def _lowstate_handler(self, msg):
        self.low_state = msg

    def get_state(self) -> B2WState:
        if self.low_state is None:
            raise RuntimeError("B2W lowstate not available.")

        ms = self.low_state.motor_state
        leg_q = np.array([ms[int(i)].q for i in self.leg_idx], dtype=np.float32)
        leg_qd = np.array([ms[int(i)].dq for i in self.leg_idx], dtype=np.float32)
        wheel_qd = np.array([ms[int(i)].dq for i in self.wheel_idx], dtype=np.float32)

        imu = self.low_state.imu_state
        quat = np.asarray(imu.quaternion, dtype=np.float32).reshape(4)
        gyro = np.asarray(imu.gyroscope, dtype=np.float32).reshape(3)

        return B2WState(
            leg_q_policy=leg_q,
            leg_qd_policy=leg_qd,
            wheel_qd_policy=wheel_qd,
            imu_quat_wxyz=quat,
            imu_gyro=gyro,
        )

    def send(self, leg_q_target_policy: np.ndarray, wheel_dq_target_policy: np.ndarray):
        if self.low_cmd is None or self.publisher is None:
            raise RuntimeError("B2W low-level interface is not connected.")

        leg_q_target_policy = np.asarray(leg_q_target_policy, dtype=np.float32).reshape(12)
        wheel_dq_target_policy = np.asarray(wheel_dq_target_policy, dtype=np.float32).reshape(4)
        wheel_dq_target_policy = np.clip(
            wheel_dq_target_policy, -self.wheel_vel_limit, self.wheel_vel_limit
        )

        # Position-PD legs.
        for j, motor_idx in enumerate(self.leg_idx):
            mc = self.low_cmd.motor_cmd[int(motor_idx)]
            mc.mode = 0x01
            mc.q = float(leg_q_target_policy[j])
            mc.kp = float(self.leg_kp[j])
            mc.dq = 0.0
            mc.kd = float(self.leg_kd[j])
            mc.tau = 0.0

        # Velocity-PD wheels.
        for j, motor_idx in enumerate(self.wheel_idx):
            mc = self.low_cmd.motor_cmd[int(motor_idx)]
            mc.mode = 0x01
            mc.q = 0.0
            mc.kp = 0.0
            mc.dq = float(wheel_dq_target_policy[j])
            mc.kd = float(self.wheel_kd[j])
            mc.tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.publisher.Write(self.low_cmd)


class WBCUANSim2Real:
    def __init__(self, cfg_path: str, mode: str, dry_run: bool = False):
        self.cfg_path = os.path.abspath(cfg_path)
        with open(self.cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.mode = mode
        self.dry_run = dry_run

        # Explicitly reject stale deployment semantics.
        if "uan_model_path" in self.cfg:
            raise ValueError(
                "uan_model_path must not exist in real deployment config. "
                "UAN is simulation-only."
            )
        if "enable_arm_target_rate_limit" in self.cfg or "arm_target_rate_limit" in self.cfg:
            raise ValueError(
                "Arm target rate limiter keys are not allowed: current WBC training "
                "uses plain JointPositionActionCfg."
            )

        self.policy_path = _resolve_path(self.cfg["policy_path"])
        self.ee_command_path = _resolve_path(self.cfg["ee_command_path"])

        self.control_hz = float(self.cfg.get("control_hz", 500.0))
        self.policy_hz = float(self.cfg.get("policy_hz", 50.0))
        self.control_dt = 1.0 / self.control_hz
        self.policy_dt = 1.0 / self.policy_hz
        ratio = self.control_hz / self.policy_hz
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("control_hz / policy_hz must be an integer.")
        self.policy_decimation = int(round(ratio))

        self.history_length = int(self.cfg["history_length"])
        self.obs_dim_per_step = int(self.cfg["obs_dim_per_step"])
        self.obs_dim = int(self.cfg["obs_dim"])
        self.action_dim = int(self.cfg["action_dim"])

        if self.history_length != 5 or self.obs_dim_per_step != 80 or self.obs_dim != 400:
            raise ValueError("Expected WBC observation layout: history=5, 80/step, 400 total.")
        if self.action_dim != 22:
            raise ValueError("Expected WBC action_dim=22.")

        self.base_command = np.asarray(self.cfg["base_command"], dtype=np.float32).reshape(3)

        # Policy-space order matches the working MuJoCo sim2sim:
        # legs = [FL,FR,RL,RR] hips, then thighs, then calves.
        self.default_leg = np.asarray(self.cfg["default_leg_pos_policy"], dtype=np.float32).reshape(12)
        self.default_arm = np.asarray(self.cfg["default_arm_pos"], dtype=np.float32).reshape(6)
        self.default_gripper = float(self.cfg.get("default_gripper_pos", 0.0))

        self.leg_action_scale = float(self.cfg["leg_action_scale"])
        self.arm_action_scale = np.asarray(self.cfg["arm_action_scale"], dtype=np.float32).reshape(6)
        self.wheel_action_scale = float(self.cfg["wheel_action_scale"])

        self.startup_hold_s = float(self.cfg.get("startup_hold_s", 1.0))
        self.startup_blend_s = float(self.cfg.get("startup_blend_s", 2.0))
        self.move_to_default_s = float(self.cfg.get("move_to_default_s", 3.0))
        self.z1_runtime_settle_s = float(self.cfg.get("z1_runtime_settle_s", 0.5))

        self.arm_q_lower = np.asarray(self.cfg["arm_q_lower"], dtype=np.float32).reshape(6)
        self.arm_q_upper = np.asarray(self.cfg["arm_q_upper"], dtype=np.float32).reshape(6)

        self.gripper_hold_q = float(self.cfg.get("gripper_hold_q", self.default_gripper))

        # Verify Z1 firmware gains map EXACTLY to training-space nominal PD.
        self.arm_kp_fw = np.asarray(self.cfg["arm_kps_runtime"], dtype=np.float64).reshape(6)
        self.arm_kd_fw = np.asarray(self.cfg["arm_kds_runtime"], dtype=np.float64).reshape(6)
        self.arm_kp_train = np.asarray(self.cfg["arm_kp_training"], dtype=np.float64).reshape(6)
        self.arm_kd_train = np.asarray(self.cfg["arm_kd_training"], dtype=np.float64).reshape(6)

        kp_eff = self.arm_kp_fw * float(self.cfg.get("z1_kp_protocol_scale", 25.6))
        kd_eff = self.arm_kd_fw * float(self.cfg.get("z1_kd_protocol_scale", 0.0128))

        if not np.allclose(kp_eff, self.arm_kp_train, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"Z1 Kp mismatch after protocol scaling:\n"
                f"  fw*25.6 = {kp_eff}\n"
                f"  training = {self.arm_kp_train}"
            )
        if not np.allclose(kd_eff, self.arm_kd_train, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"Z1 Kd mismatch after protocol scaling:\n"
                f"  fw*0.0128 = {kd_eff}\n"
                f"  training = {self.arm_kd_train}"
            )

        print("=" * 82)
        print("B2WZ1 WBC-UAN-trained policy | REAL ROBOT deployment")
        print("=" * 82)
        print(f"Mode:              {self.mode}")
        print(f"Policy:            {self.policy_path}")
        print("UAN model:         NOT DEPLOYED (simulation-only)")
        print(f"Hardware loop:     {self.control_hz:.1f} Hz")
        print(f"WBC inference:     {self.policy_hz:.1f} Hz (every {self.policy_decimation} ticks)")
        print("Arm target RL:     NONE")
        print(f"Z1 kp_fw:          {self.arm_kp_fw}")
        print(f"Z1 kd_fw:          {self.arm_kd_fw}")
        print(f"Scaled Kp:         {kp_eff}")
        print(f"Scaled Kd:         {kd_eff}")
        print(f"Training Kp:       {self.arm_kp_train}")
        print(f"Training Kd:       {self.arm_kd_train}")
        print("Gain alignment:    PASS")
        print("=" * 82)

        if not os.path.isfile(self.policy_path):
            raise FileNotFoundError(f"Policy not found: {self.policy_path}")
        if not os.path.isfile(self.ee_command_path):
            raise FileNotFoundError(f"EE command table not found: {self.ee_command_path}")

        self.sess = ort.InferenceSession(
            self.policy_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        if self.sess.get_inputs()[0].shape[-1] != 400:
            raise RuntimeError(f"ONNX input shape mismatch: {self.sess.get_inputs()[0].shape}")
        if self.sess.get_outputs()[0].shape[-1] != 22:
            raise RuntimeError(f"ONNX output shape mismatch: {self.sess.get_outputs()[0].shape}")

        self.ee_sampler = PresampledKeypointsDirectCommandPLBReal(
            file_path=self.ee_command_path,
            policy_dt=self.policy_dt,
            cycle_duration_s=float(self.cfg.get("ee_cycle_duration_s", 8.0)),
            seed=int(self.cfg.get("ee_command_seed", 0)),
        )

        self.b2w: Optional[B2WLowLevel] = None
        self.z1 = None

        self.last_action = np.zeros(22, dtype=np.float32)
        self.leg_target = self.default_leg.copy()
        self.arm_target = self.default_arm.copy()
        self.wheel_cmd = np.zeros(4, dtype=np.float32)

        self._init_history_buffers()

    def _init_history_buffers(self):
        self.base_ang_vel_hist = deque(maxlen=self.history_length)
        self.projected_gravity_hist = deque(maxlen=self.history_length)
        self.base_cmd_hist = deque(maxlen=self.history_length)
        self.ee_cmd_hist = deque(maxlen=self.history_length)
        self.joint_pos_leg_hist = deque(maxlen=self.history_length)
        self.joint_pos_arm_hist = deque(maxlen=self.history_length)
        self.joint_vel_leg_hist = deque(maxlen=self.history_length)
        self.joint_vel_arm_hist = deque(maxlen=self.history_length)
        self.joint_vel_wheel_hist = deque(maxlen=self.history_length)
        self.last_action_hist = deque(maxlen=self.history_length)

    def _make_obs_parts(self, ee_cmd_plb: np.ndarray):
        assert self.b2w is not None and self.z1 is not None

        bs = self.b2w.get_state()
        self.z1.read_state()

        base_ang_vel = bs.imu_gyro.astype(np.float32)
        projected_gravity = quat_rotate_inverse_wxyz(
            bs.imu_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )

        leg_pos_rel = bs.leg_q_policy - self.default_leg
        arm_q = np.asarray(self.z1.q, dtype=np.float32).reshape(6)
        arm_qd = np.asarray(self.z1.qd, dtype=np.float32).reshape(6)
        arm_pos_rel = arm_q - self.default_arm

        return (
            base_ang_vel,
            projected_gravity,
            self.base_command.copy(),
            np.asarray(ee_cmd_plb, dtype=np.float32).reshape(9),
            leg_pos_rel.astype(np.float32),
            arm_pos_rel.astype(np.float32),
            bs.leg_qd_policy.astype(np.float32),
            arm_qd,
            bs.wheel_qd_policy.astype(np.float32),
            self.last_action.copy(),
        )

    def _append_parts(self, parts):
        (
            base_ang_vel,
            projected_gravity,
            base_cmd,
            ee_cmd,
            q_leg,
            q_arm,
            qd_leg,
            qd_arm,
            qd_wheel,
            last_action,
        ) = parts

        self.base_ang_vel_hist.append(base_ang_vel.copy())
        self.projected_gravity_hist.append(projected_gravity.copy())
        self.base_cmd_hist.append(base_cmd.copy())
        self.ee_cmd_hist.append(ee_cmd.copy())
        self.joint_pos_leg_hist.append(q_leg.copy())
        self.joint_pos_arm_hist.append(q_arm.copy())
        self.joint_vel_leg_hist.append(qd_leg.copy())
        self.joint_vel_arm_hist.append(qd_arm.copy())
        self.joint_vel_wheel_hist.append(qd_wheel.copy())
        self.last_action_hist.append(last_action.copy())

    def _initialize_history(self, ee_cmd_plb: np.ndarray):
        parts = self._make_obs_parts(ee_cmd_plb)
        for _ in range(self.history_length):
            self._append_parts(parts)

    def _obs_stack(self) -> np.ndarray:
        obs = np.concatenate(
            [
                np.asarray(self.base_ang_vel_hist).reshape(-1),
                np.asarray(self.projected_gravity_hist).reshape(-1),
                np.asarray(self.base_cmd_hist).reshape(-1),
                np.asarray(self.ee_cmd_hist).reshape(-1),
                np.asarray(self.joint_pos_leg_hist).reshape(-1),
                np.asarray(self.joint_pos_arm_hist).reshape(-1),
                np.asarray(self.joint_vel_leg_hist).reshape(-1),
                np.asarray(self.joint_vel_arm_hist).reshape(-1),
                np.asarray(self.joint_vel_wheel_hist).reshape(-1),
                np.asarray(self.last_action_hist).reshape(-1),
            ],
            dtype=np.float32,
        )
        if obs.shape != (self.obs_dim,):
            raise RuntimeError(f"Observation shape {obs.shape}, expected {(self.obs_dim,)}")
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("Non-finite value in WBC observation.")
        return obs

    def _compute_blend(self, t: float) -> float:
        if t < self.startup_hold_s:
            return 0.0
        if t < self.startup_hold_s + self.startup_blend_s:
            return float((t - self.startup_hold_s) / max(self.startup_blend_s, 1e-6))
        return 1.0

    def _update_policy(self, ee_cmd_plb: np.ndarray, blend: float):
        parts = self._make_obs_parts(ee_cmd_plb)
        self._append_parts(parts)
        obs = self._obs_stack()

        if self.mode == "pd-stand":
            action = np.zeros(22, dtype=np.float32)
        else:
            action = self.sess.run(
                [self.output_name], {self.input_name: obs[None, :]}
            )[0][0].astype(np.float32)

        if action.shape != (22,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"Invalid policy action: shape={action.shape}, action={action}")

        self.last_action[:] = action
        leg_act = action[0:12]
        arm_act = action[12:18]
        wheel_act = action[18:22]

        if self.mode == "pd-stand":
            self.leg_target = self.default_leg.copy()
            self.arm_target = self.default_arm.copy()
            self.wheel_cmd[:] = 0.0
        elif self.mode == "lock-arm-policy":
            self.leg_target = self.default_leg + blend * self.leg_action_scale * leg_act
            self.arm_target = self.default_arm.copy()
            self.wheel_cmd = blend * self.wheel_action_scale * wheel_act
        elif self.mode == "full-policy":
            self.leg_target = self.default_leg + blend * self.leg_action_scale * leg_act
            # IMPORTANT: exact training semantics. No target rate limiter.
            self.arm_target = self.default_arm + blend * self.arm_action_scale * arm_act
            self.wheel_cmd = blend * self.wheel_action_scale * wheel_act
        else:
            raise ValueError(self.mode)

        if np.any(self.arm_target < self.arm_q_lower) or np.any(
            self.arm_target > self.arm_q_upper
        ):
            raise RuntimeError(
                "WBC requested Z1 target outside live-SDK hard range:\n"
                f"  q_des={np.round(self.arm_target, 3)}\n"
                f"  lower={self.arm_q_lower}\n"
                f"  upper={self.arm_q_upper}"
            )

    def _connect_hardware(self):
        # Z1 adapter used in the UAN hardware collection / prior deployment stack.
        from utils.z1_helper import Z1ArmAdapter

        self.z1 = Z1ArmAdapter(cfg=self.cfg, project_root=_PROJECT_ROOT)
        self.z1.connect()
        self.z1.arm_runtime_mode = "position_pd"
        self.z1.read_state()

        # Audit the adapter's actual runtime gains, not just YAML.
        kp_actual = np.asarray(self.z1.arm_kps_runtime, dtype=np.float64).reshape(6)
        kd_actual = np.asarray(self.z1.arm_kds_runtime, dtype=np.float64).reshape(6)
        if not np.allclose(kp_actual, self.arm_kp_fw, atol=1e-9, rtol=0.0):
            raise RuntimeError(f"Z1 adapter Kp differs from YAML: {kp_actual} vs {self.arm_kp_fw}")
        if not np.allclose(kd_actual, self.arm_kd_fw, atol=1e-9, rtol=0.0):
            raise RuntimeError(f"Z1 adapter Kd differs from YAML: {kd_actual} vs {self.arm_kd_fw}")

        self.b2w = B2WLowLevel(self.cfg)
        self.b2w.connect()

    def _move_to_default(self):
        """Slowly bring B2W legs and Z1 arm to policy default pose."""
        assert self.b2w is not None and self.z1 is not None

        bs0 = self.b2w.get_state()
        self.z1.read_state()
        leg0 = bs0.leg_q_policy.astype(np.float32).copy()
        arm0 = np.asarray(self.z1.q, dtype=np.float32).reshape(6).copy()

        n = max(1, int(round(self.move_to_default_s * self.control_hz)))
        t0 = time.perf_counter()
        print(
            f"[startup] moving to policy default over {self.move_to_default_s:.1f}s\n"
            f"  leg start={np.round(leg0, 3)}\n"
            f"  arm start={np.round(arm0, 3)} -> {np.round(self.default_arm, 3)}"
        )

        for i in range(n):
            u = (i + 1) / n
            s = 0.5 * (1.0 - np.cos(np.pi * u))
            leg_cmd = (1.0 - s) * leg0 + s * self.default_leg
            arm_cmd = (1.0 - s) * arm0 + s * self.default_arm
            self.b2w.send(leg_cmd, np.zeros(4, dtype=np.float32))
            self.z1.hold_pose_lowcmd(
                q_cmd=arm_cmd.astype(np.float32),
                gripper_q_cmd=self.gripper_hold_q,
            )
            _sleep_until(t0 + (i + 1) * self.control_dt)

        # Hand off from startup gains to deployment runtime firmware gains.
        self.z1.arm_runtime_mode = "position_pd"
        settle_n = max(1, int(round(self.z1_runtime_settle_s * self.control_hz)))
        t0 = time.perf_counter()
        for i in range(settle_n):
            self.b2w.send(self.default_leg, np.zeros(4, dtype=np.float32))
            self.z1.track_target_pd_once(
                q_target=self.default_arm.astype(np.float32),
                gripper_q_target=self.gripper_hold_q,
                use_startup_gains=False,
            )
            _sleep_until(t0 + (i + 1) * self.control_dt)

        print("[startup] default pose reached; runtime gains engaged.")

    def _safe_hold_current(self, duration_s: float = 0.5):
        if self.b2w is None or self.z1 is None:
            return
        try:
            bs = self.b2w.get_state()
            self.z1.read_state()
            leg_hold = bs.leg_q_policy.copy()
            arm_hold = np.asarray(self.z1.q, dtype=np.float32).copy()
            n = max(1, int(round(duration_s * self.control_hz)))
            for _ in range(n):
                self.b2w.send(leg_hold, np.zeros(4, dtype=np.float32))
                self.z1.track_target_pd_once(
                    q_target=arm_hold,
                    gripper_q_target=self.gripper_hold_q,
                    use_startup_gains=False,
                )
                time.sleep(self.control_dt)
        except Exception as e:
            print(f"[WARN] safe-hold failed: {e}")

    def run(self):
        if self.dry_run:
            print("[dry-run] configuration, ONNX shape, paths, and gain alignment are valid.")
            print("[dry-run] No robot communication was opened.")
            return

        print("\nREAL ROBOT LOW-LEVEL CONTROL WILL BE ENABLED.")
        print("Confirm B2W is supported, workspace is clear, E-stop is ready.")
        if input("Type 'go' to connect and enable low-level control: ").strip().lower() != "go":
            print("Aborted before hardware connection.")
            return

        self._connect_hardware()

        try:
            self._move_to_default()

            sample_first = self.mode != "pd-stand"
            self.ee_sampler.reset(sample_first=sample_first)
            ee_cmd = self.ee_sampler.current.copy()
            self._initialize_history(ee_cmd)

            self.leg_target = self.default_leg.copy()
            self.arm_target = self.default_arm.copy()
            self.wheel_cmd[:] = 0.0
            self.last_action[:] = 0.0

            print("[run] WBC active.")
            t0 = time.perf_counter()
            tick = 0
            last_report = -1.0
            max_duration = float(self.cfg.get("max_duration_s", 180.0))

            while True:
                loop_deadline = t0 + (tick + 1) * self.control_dt
                t = tick * self.control_dt
                if t >= max_duration:
                    print("[run] max_duration_s reached.")
                    break

                if tick % self.policy_decimation == 0:
                    if self.mode != "pd-stand":
                        ee_cmd = self.ee_sampler.update()
                    blend = self._compute_blend(t)
                    self._update_policy(ee_cmd, blend)

                    if t - last_report >= float(self.cfg.get("report_period_s", 1.0)):
                        assert self.z1 is not None
                        self.z1.read_state()
                        arm_err = self.arm_target - np.asarray(self.z1.q, dtype=np.float32)
                        print(
                            f"[{t:7.2f}s] blend={blend:.2f} "
                            f"| action arm=[{self.last_action[12:18].min():+.2f},"
                            f"{self.last_action[12:18].max():+.2f}] "
                            f"| max arm err={np.max(np.abs(arm_err)):.3f} rad "
                            f"| wheel cmd max={np.max(np.abs(self.wheel_cmd)):.2f}"
                        )
                        last_report = t

                # 500-Hz zero-order hold of the latest 50-Hz WBC command.
                assert self.b2w is not None and self.z1 is not None
                self.b2w.send(self.leg_target, self.wheel_cmd)
                self.z1.track_target_pd_once(
                    q_target=self.arm_target.astype(np.float32),
                    gripper_q_target=self.gripper_hold_q,
                    use_startup_gains=False,
                )

                late = _sleep_until(loop_deadline)
                if late > 0.5 * self.control_dt and tick % 500 == 0:
                    print(f"[WARN] control loop late by {late * 1e3:.2f} ms")

                tick += 1

        except KeyboardInterrupt:
            print("\n[run] interrupted by user.")
        except Exception as e:
            print(f"\n[run][ERROR] {type(e).__name__}: {e}")
            raise
        finally:
            print("[run] holding current pose and stopping wheels.")
            self._safe_hold_current(duration_s=0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_path", type=str)
    parser.add_argument(
        "--mode",
        choices=["pd-stand", "lock-arm-policy", "full-policy"],
        default="full-policy",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate YAML, ONNX, and Z1 gain scaling without touching hardware.",
    )
    args = parser.parse_args()

    runner = WBCUANSim2Real(args.yaml_path, mode=args.mode, dry_run=args.dry_run)
    runner.run()


if __name__ == "__main__":
    main()
