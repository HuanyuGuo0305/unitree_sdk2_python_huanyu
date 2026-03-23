"""
B2WZ1 loco-manipulation sim2real deployment.

Run from repository root:

    python3 deploy/b2wz1_locomanipulation.py <network_interface> deploy/configs/b2wz1_locomanipulation.yaml --mode pd-stand
    python3 deploy/b2wz1_locomanipulation.py <network_interface> deploy/configs/b2wz1_locomanipulation.yaml --mode lock-arm-policy
    python3 deploy/b2wz1_locomanipulation.py <network_interface> deploy/configs/b2wz1_locomanipulation.yaml --mode full-policy

Design goals:
- Keep observation / action layout aligned with MuJoCo sim2sim validation.
- Reuse the proven B2W sim2real locomotion control style.
- Use Z1 LOWCMD in the same spirit as the official example_lowcmd.py.
"""

import os
import sys
import time
import yaml
import argparse
from collections import deque

import numpy as np
import onnxruntime as ort

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo

from utils.command_helper import create_damping_cmd, create_zero_cmd, InitLowCmd
from utils.remote_controller import RemoteController, KeyMap
from utils.math import (
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
)
from utils.z1_helper import (
    PresampledKeypointsInterpolateCommandLBSim,
    Z1ArmAdapter,
    compute_ee_current_kp_lb,
)


class B2WZ1LocoManipController:
    def __init__(self, cfg_path: str, mode: str):
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.mode = mode
        self.use_policy = mode in ["lock-arm-policy", "full-policy"]

        # Timing
        self.control_dt = float(self.cfg["control_dt"])  # 50 Hz
        self.startup_hold_s = float(self.cfg.get("startup_hold_s", 1.0))
        self.startup_blend_s = float(self.cfg.get("startup_blend_s", 2.0))

        # Policy
        self.policy_path = self._resolve_path(self.cfg["policy_path"])
        self.history_length = int(self.cfg["history_length"])
        self.obs_dim_per_step = int(self.cfg["obs_dim_per_step"])
        self.obs_dim = int(self.cfg["obs_dim"])
        self.action_dim = int(self.cfg["action_dim"])

        assert self.obs_dim_per_step == 89, f"Expected obs_dim_per_step=89, got {self.obs_dim_per_step}"
        assert self.obs_dim == 267, f"Expected obs_dim=267, got {self.obs_dim}"
        assert self.action_dim == 22, f"Expected action_dim=22, got {self.action_dim}"

        self.session = None
        self.input_name = None
        self.output_name = None

        if self.use_policy:
            self.session = ort.InferenceSession(self.policy_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

        # Commands
        # Base command is updated from remote each tick, then clipped/scaled to match the training command range.
        self.base_command = np.zeros(3, dtype=np.float32)
        self.command_scale = np.array(self.cfg["command_scale"], dtype=np.float32).reshape(3,)
        self.command_deadband_lin = float(self.cfg.get("command_deadband_lin", 0.2))
        self.command_deadband_ang = float(self.cfg.get("command_deadband_ang", 0.2))

        # Default poses and gains
        self.default_leg_pos_policy = np.array(self.cfg["default_leg_pos_policy"], dtype=np.float32).reshape(12,)
        self.default_leg_pos_hw = np.array(self.cfg["default_leg_pos_hw"], dtype=np.float32).reshape(12,)
        self.squat_leg_pos_hw = np.array(self.cfg["squat_leg_pos_hw"], dtype=np.float32).reshape(12,)

        self.default_arm_pos = np.array(self.cfg["default_arm_pos"], dtype=np.float32).reshape(6,)
        self.default_gripper_pos = float(self.cfg["default_gripper_pos"])

        # Policy observation default: leg(12) + arm(6)
        self.default_joint_pos_policy = np.array(self.cfg["default_joint_pos_policy"], dtype=np.float32).reshape(18,)

        # RL/runtime gains for B2W
        self.kps_rl = np.array(self.cfg["kps_rl"], dtype=np.float32).reshape(16,)
        self.kds_rl = np.array(self.cfg["kds_rl"], dtype=np.float32).reshape(16,)

        # Stronger gains for default-pose PD hold / transition
        self.kps_pd = np.array(self.cfg["kps_pd"], dtype=np.float32).reshape(16,)
        self.kds_pd = np.array(self.cfg["kds_pd"], dtype=np.float32).reshape(16,)

        self.leg_action_scale = float(self.cfg["leg_action_scale"])
        self.arm_action_scale = np.array(self.cfg["arm_action_scale"], dtype=np.float32).reshape(6,)
        self.wheel_action_scale = float(self.cfg["wheel_action_scale"])

        # Joint naming / ordering
        # Policy order
        self.leg_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.wheel_joint_names = ["FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"]

        # B2W hardware order
        self.hardware_joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
        ]

        self.hardware_leg_names = self.hardware_joint_names[:12]
        self.hardware_wheel_names = self.hardware_joint_names[12:]

        # Full B2W policy order = leg(12) + wheel(4)
        self.b2w_policy_joint_names = self.leg_joint_names + self.wheel_joint_names

        # Mapping:
        # policy index -> hardware index
        self.policy_to_hardware_joint_indices = [
            self.hardware_joint_names.index(name) for name in self.b2w_policy_joint_names
        ]
        # hardware index -> policy index
        self.hardware_to_policy_joint_indices = [
            self.b2w_policy_joint_names.index(name) for name in self.hardware_joint_names
        ]

        # Leg-only mappings
        # Reorder a leg array from hardware order -> policy order: policy_leg = hardware_leg[self.leg_policy_to_hw_indices]
        self.leg_policy_to_hw_indices = [
            self.hardware_leg_names.index(name) for name in self.leg_joint_names
        ]

        # Reorder a leg array from policy order -> hardware order: hardware_leg = policy_leg[self.leg_hw_to_policy_indices]
        self.leg_hw_to_policy_indices = [
            self.leg_joint_names.index(name) for name in self.hardware_leg_names
        ]

        self.leg_hardware_indices = list(range(12))
        self.wheel_hardware_indices = list(range(12, 16))

        # Hardware index -> wheel command index in policy wheel order [FL, FR, RL, RR]
        self.hw_to_wheel_cmd_indices = {
            self.hardware_joint_names.index(name): idx
            for idx, name in enumerate(self.wheel_joint_names)
        }

        # Convert squat pose from hardware-leg order -> policy-leg order
        self.squat_leg_pos_policy = self.squat_leg_pos_hw[self.leg_policy_to_hw_indices]

        # Optional consistency check between default_leg_pos_policy and default_leg_pos_hw
        default_leg_from_hw = self.default_leg_pos_hw[self.leg_policy_to_hw_indices]
        if not np.allclose(default_leg_from_hw, self.default_leg_pos_policy, atol=1e-5):
            raise ValueError(
                "default_leg_pos_policy and default_leg_pos_hw are inconsistent in yaml."
            )

        # Action split
        self.leg_action_indices = list(range(0, 12))
        self.arm_action_indices = list(range(12, 18))
        self.wheel_action_indices = list(range(18, 22))

        # Z1 helper
        self.z1 = Z1ArmAdapter(self.cfg, PROJECT_ROOT)

        # EE command sampler
        ee_command_path = self._resolve_path(self.cfg["ee_command_path"])
        self.ee_kp_dx = float(self.cfg["ee_kp_dx"])
        self.ee_kp_dz = float(self.cfg["ee_kp_dz"])

        self.ee_cmd_sampler = PresampledKeypointsInterpolateCommandLBSim(
            file_path=ee_command_path,
            kp_dx=self.ee_kp_dx,
            kp_dz=self.ee_kp_dz,
            kp0_threshold=float(self.cfg["ee_kp0_threshold"]),
            rot_threshold=float(self.cfg["ee_rot_threshold"]),
            seed=int(self.cfg.get("ee_command_seed", 0)),
        )
        self.ee_resample_interval = int(self.cfg["ee_resample_interval"])
        self.ee_cmd_lb_current = np.zeros(9, dtype=np.float32)

        # Runtime states
        self.remote_controller = RemoteController()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.crc = CRC()

        # IMU/base
        self.base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.base_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # B2W state in policy order [leg(12), wheel(4)]
        self.b2w_joint_pos = np.zeros(16, dtype=np.float32)
        self.b2w_joint_vel = np.zeros(16, dtype=np.float32)

        # Policy state
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)
        self.leg_target = self.default_leg_pos_policy.copy()
        self.arm_target = self.default_arm_pos.copy()
        self.wheel_cmd = np.zeros(4, dtype=np.float32)

        # History buffers
        self.base_ang_vel_hist = deque(maxlen=self.history_length)
        self.projected_gravity_hist = deque(maxlen=self.history_length)
        self.base_cmd_hist = deque(maxlen=self.history_length)
        self.ee_cmd_hist = deque(maxlen=self.history_length)
        self.ee_cur_hist = deque(maxlen=self.history_length)
        self.joint_pos_leg_hist = deque(maxlen=self.history_length)
        self.joint_pos_arm_hist = deque(maxlen=self.history_length)
        self.joint_vel_leg_hist = deque(maxlen=self.history_length)
        self.joint_vel_arm_hist = deque(maxlen=self.history_length)
        self.joint_vel_wheel_hist = deque(maxlen=self.history_length)
        self.last_action_hist = deque(maxlen=self.history_length)

        self.counter = 0
        self.policy_tick = 0
        self.start_time = None

    def _resolve_path(self, path_str: str) -> str:
        if os.path.isabs(path_str):
            return path_str
        return os.path.abspath(os.path.join(PROJECT_ROOT, path_str))

    # DDS callbacks and command publishing
    def low_state_handler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(msg.wireless_remote)

    def wait_for_low_state(self):
        print("[B2WZ1] Waiting for first B2W lowstate...")
        while getattr(self.low_state, "tick", 0) == 0:
            time.sleep(0.01)
        print(f"[B2WZ1] First lowstate received: tick={self.low_state.tick}")

    def send_b2w_cmd(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    # Sensor reading
    def _update_base_command_from_remote(self):
        cmd = np.zeros(3, dtype=np.float32)

        # Same sign convention as previous B2W deployment
        cmd[0] = np.clip(self.remote_controller.ly, -1.0, 1.0) * self.command_scale[0]   # vx
        cmd[1] = np.clip(-self.remote_controller.lx, -1.0, 1.0) * self.command_scale[1]  # vy
        cmd[2] = np.clip(-self.remote_controller.rx, -1.0, 1.0) * self.command_scale[2]  # wz

        lin_norm = np.linalg.norm(cmd[:2], ord=2)
        if lin_norm < self.command_deadband_lin * max(self.command_scale[0], self.command_scale[1]):
            cmd[0] = 0.0
            cmd[1] = 0.0
        if abs(cmd[2]) < self.command_deadband_ang * self.command_scale[2]:
            cmd[2] = 0.0

        self.base_command[:] = cmd

    def _read_b2w_sensors_once(self):
        # Base IMU quaternion in wxyz
        q = self.low_state.imu_state.quaternion
        self.base_quat_wxyz[:] = quat_unique_wxyz(
            np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)
        )
        self.base_quat_wxyz[:] = self.base_quat_wxyz / max(np.linalg.norm(self.base_quat_wxyz), 1e-8)
        gyro = self.low_state.imu_state.gyroscope
        self.base_ang_vel_b[:] = np.array([gyro[0], gyro[1], gyro[2]], dtype=np.float32)

        self.projected_gravity_b[:] = quat_rotate_inverse_numpy(self.base_quat_wxyz, self.gravity_w)

        # Read B2W joints from hardware-order lowstate and store them in policy order:
        # [FL_hip, FR_hip, RL_hip, RR_hip,
        #  FL_thigh, FR_thigh, RL_thigh, RR_thigh,
        #  FL_calf, FR_calf, RL_calf, RR_calf,
        #  FL_wheel, FR_wheel, RL_wheel, RR_wheel]
        for p_idx in range(16):
            hw_idx = self.policy_to_hardware_joint_indices[p_idx]
            self.b2w_joint_pos[p_idx] = self.low_state.motor_state[hw_idx].q
            self.b2w_joint_vel[p_idx] = self.low_state.motor_state[hw_idx].dq

        self._update_base_command_from_remote()

    def _read_all_sensors_once(self):
        self._read_b2w_sensors_once()
        self.z1.read_state()

    # Observation
    def compute_ee_current_kp_lb(self) -> np.ndarray:
        return compute_ee_current_kp_lb(
            base_quat_wxyz=self.base_quat_wxyz,
            z1_adapter=self.z1,
            kp_dx=self.ee_kp_dx,
            kp_dz=self.ee_kp_dz,
        )

    def build_obs_step(self, ee_cmd_lb: np.ndarray) -> np.ndarray:
        ee_cur_lb = self.compute_ee_current_kp_lb()

        leg_pos = self.b2w_joint_pos[:12].copy()
        wheel_vel = self.b2w_joint_vel[12:16].copy()
        arm_pos = self.z1.q.copy()
        arm_vel = self.z1.qd.copy()

        joint_pos_policy = np.concatenate([leg_pos, arm_pos], dtype=np.float32)
        joint_pos_rel = joint_pos_policy - self.default_joint_pos_policy

        joint_pos_leg_rel = joint_pos_rel[:12]
        joint_pos_arm_rel = joint_pos_rel[12:18]

        joint_vel_leg = self.b2w_joint_vel[:12].copy()
        joint_vel_arm = arm_vel
        joint_vel_wheel = wheel_vel

        obs = np.concatenate(
            [
                self.base_ang_vel_b,
                self.projected_gravity_b,
                self.base_command,
                ee_cmd_lb,
                ee_cur_lb,
                joint_pos_leg_rel,
                joint_pos_arm_rel,
                joint_vel_leg,
                joint_vel_arm,
                joint_vel_wheel,
                self.last_action,
            ],
            dtype=np.float32,
        )

        assert obs.shape[0] == self.obs_dim_per_step, f"Obs dim mismatch: {obs.shape[0]}"
        return obs

    def _init_history(self):
        obs0 = self.build_obs_step(self.ee_cmd_lb_current)

        i = 0
        obs0_base_ang_vel = obs0[i:i + 3]; i += 3
        obs0_projected_gravity = obs0[i:i + 3]; i += 3
        obs0_base_cmd = obs0[i:i + 3]; i += 3
        obs0_ee_cmd = obs0[i:i + 9]; i += 9
        obs0_ee_cur = obs0[i:i + 9]; i += 9
        obs0_joint_pos_leg = obs0[i:i + 12]; i += 12
        obs0_joint_pos_arm = obs0[i:i + 6]; i += 6
        obs0_joint_vel_leg = obs0[i:i + 12]; i += 12
        obs0_joint_vel_arm = obs0[i:i + 6]; i += 6
        obs0_joint_vel_wheel = obs0[i:i + 4]; i += 4
        obs0_last_action = obs0[i:i + 22]; i += 22

        for _ in range(self.history_length):
            self.base_ang_vel_hist.append(obs0_base_ang_vel.copy())
            self.projected_gravity_hist.append(obs0_projected_gravity.copy())
            self.base_cmd_hist.append(obs0_base_cmd.copy())
            self.ee_cmd_hist.append(obs0_ee_cmd.copy())
            self.ee_cur_hist.append(obs0_ee_cur.copy())
            self.joint_pos_leg_hist.append(obs0_joint_pos_leg.copy())
            self.joint_pos_arm_hist.append(obs0_joint_pos_arm.copy())
            self.joint_vel_leg_hist.append(obs0_joint_vel_leg.copy())
            self.joint_vel_arm_hist.append(obs0_joint_vel_arm.copy())
            self.joint_vel_wheel_hist.append(obs0_joint_vel_wheel.copy())
            self.last_action_hist.append(obs0_last_action.copy())

    # B2W command construction
    def _write_b2w_hold_or_rl_cmd(self, leg_target: np.ndarray, wheel_cmd: np.ndarray, use_pd_gains: bool):
        leg_target = np.asarray(leg_target, dtype=np.float32).reshape(12,)
        wheel_cmd = np.asarray(wheel_cmd, dtype=np.float32).reshape(4,)

        # Build target in B2W policy order [leg(12), wheel(4)]
        target_policy = np.concatenate([leg_target, np.zeros(4, dtype=np.float32)], dtype=np.float32)
        target_hw = target_policy[self.policy_to_hardware_joint_indices]

        kps = self.kps_pd if use_pd_gains else self.kps_rl
        kds = self.kds_pd if use_pd_gains else self.kds_rl

        for i in range(16):
            if i in self.leg_hardware_indices:
                self.low_cmd.motor_cmd[i].q = float(target_hw[i])
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = float(kps[i])
                self.low_cmd.motor_cmd[i].kd = float(kds[i])
                self.low_cmd.motor_cmd[i].tau = 0.0
            else:
                wheel_idx = self.hw_to_wheel_cmd_indices[i]  # policy wheel index
                vel_cmd = float(wheel_cmd[wheel_idx])

                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = vel_cmd
                self.low_cmd.motor_cmd[i].kp = float(kps[i])
                self.low_cmd.motor_cmd[i].kd = float(kds[i])
                self.low_cmd.motor_cmd[i].tau = 0.0

    # Startup states
    def zero_torque_state(self):
        print("[B2WZ1] Zero torque state. Press START to continue.")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_b2w_cmd()
            time.sleep(self.control_dt)
        print("[B2WZ1] START pressed. Exit zero torque state.")

    def move_b2w_to_pose(self, target_leg_pos_policy: np.ndarray, duration: float):
        print("[B2WZ1] Moving B2W legs to target pose...")
        target_leg_pos_policy = np.asarray(target_leg_pos_policy, dtype=np.float32).reshape(12,)
        num_steps = max(1, int(round(duration / self.control_dt)))

        self._read_b2w_sensors_once()
        init_leg_pos_policy = self.b2w_joint_pos[:12].copy()

        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            leg_cmd = (1.0 - alpha) * init_leg_pos_policy + alpha * target_leg_pos_policy

            self._write_b2w_hold_or_rl_cmd(
                leg_target=leg_cmd,
                wheel_cmd=np.zeros(4, dtype=np.float32),
                use_pd_gains=True,
            )
            self.send_b2w_cmd()
            time.sleep(self.control_dt)

        print("[B2WZ1] B2W pose transition finished.")

    def hold_default_state(self):
        print("[B2WZ1] Holding default pose. Press A to start policy.")
        while self.remote_controller.button[KeyMap.A] != 1:
            self._write_b2w_hold_or_rl_cmd(
                leg_target=self.default_leg_pos_policy,
                wheel_cmd=np.zeros(4, dtype=np.float32),
                use_pd_gains=True,
            )
            self.send_b2w_cmd()

            self.z1.hold_default_step()
            time.sleep(self.control_dt)

        print("[B2WZ1] A pressed. Start policy.")

    # One control step
    def _compute_blend(self, elapsed_s: float) -> float:
        if elapsed_s < self.startup_hold_s:
            return 0.0
        if elapsed_s < self.startup_hold_s + self.startup_blend_s:
            return float((elapsed_s - self.startup_hold_s) / self.startup_blend_s)
        return 1.0

    def step(self):
        # 1) Read sensors once per tick
        self._read_all_sensors_once()

        # 2) Update EE command
        if self.policy_tick > 0 and (self.policy_tick % self.ee_resample_interval == 0):
            self.ee_cmd_sampler.resample()
        self.ee_cmd_lb_current = self.ee_cmd_sampler.command.copy()

        # 3) Build one-step observation and update history
        obs_step = self.build_obs_step(self.ee_cmd_lb_current)

        i = 0
        curr_base_ang_vel = obs_step[i:i + 3]; i += 3
        curr_projected_gravity = obs_step[i:i + 3]; i += 3
        curr_base_cmd = obs_step[i:i + 3]; i += 3
        curr_ee_cmd = obs_step[i:i + 9]; i += 9
        curr_ee_cur = obs_step[i:i + 9]; i += 9
        curr_joint_pos_leg = obs_step[i:i + 12]; i += 12
        curr_joint_pos_arm = obs_step[i:i + 6]; i += 6
        curr_joint_vel_leg = obs_step[i:i + 12]; i += 12
        curr_joint_vel_arm = obs_step[i:i + 6]; i += 6
        curr_joint_vel_wheel = obs_step[i:i + 4]; i += 4
        curr_last_action = obs_step[i:i + 22]; i += 22

        self.base_ang_vel_hist.append(curr_base_ang_vel.copy())
        self.projected_gravity_hist.append(curr_projected_gravity.copy())
        self.base_cmd_hist.append(curr_base_cmd.copy())
        self.ee_cmd_hist.append(curr_ee_cmd.copy())
        self.ee_cur_hist.append(curr_ee_cur.copy())
        self.joint_pos_leg_hist.append(curr_joint_pos_leg.copy())
        self.joint_pos_arm_hist.append(curr_joint_pos_arm.copy())
        self.joint_vel_leg_hist.append(curr_joint_vel_leg.copy())
        self.joint_vel_arm_hist.append(curr_joint_vel_arm.copy())
        self.joint_vel_wheel_hist.append(curr_joint_vel_wheel.copy())
        self.last_action_hist.append(curr_last_action.copy())

        # 4) Stack history
        obs_stack = np.concatenate(
            [
                np.array(self.base_ang_vel_hist).reshape(-1),
                np.array(self.projected_gravity_hist).reshape(-1),
                np.array(self.base_cmd_hist).reshape(-1),
                np.array(self.ee_cmd_hist).reshape(-1),
                np.array(self.ee_cur_hist).reshape(-1),
                np.array(self.joint_pos_leg_hist).reshape(-1),
                np.array(self.joint_pos_arm_hist).reshape(-1),
                np.array(self.joint_vel_leg_hist).reshape(-1),
                np.array(self.joint_vel_arm_hist).reshape(-1),
                np.array(self.joint_vel_wheel_hist).reshape(-1),
                np.array(self.last_action_hist).reshape(-1),
            ],
            dtype=np.float32,
        )
        assert obs_stack.shape[0] == self.obs_dim, f"obs_stack dim mismatch: {obs_stack.shape[0]}"

        # 5) Run policy
        if self.mode == "pd-stand":
            action = np.zeros(self.action_dim, dtype=np.float32)
        else:
            action = self.session.run(
                [self.output_name],
                {self.input_name: obs_stack[None, :]},
            )[0][0].astype(np.float32)

        self.last_action[:] = action

        leg_act = action[self.leg_action_indices]
        arm_act = action[self.arm_action_indices]
        wheel_act = action[self.wheel_action_indices]

        elapsed_s = time.time() - self.start_time
        blend = np.clip(self._compute_blend(elapsed_s), 0.0, 1.0)

        # 6) Apply mode logic
        if self.mode == "pd-stand":
            self.leg_target = self.default_leg_pos_policy.copy()
            self.arm_target = self.default_arm_pos.copy()
            self.wheel_cmd[:] = 0.0

        elif self.mode == "lock-arm-policy":
            self.leg_target = self.default_leg_pos_policy + blend * (self.leg_action_scale * leg_act)
            self.arm_target = self.default_arm_pos.copy()
            self.wheel_cmd[:] = blend * (self.wheel_action_scale * wheel_act)

        elif self.mode == "full-policy":
            self.leg_target = self.default_leg_pos_policy + blend * (self.leg_action_scale * leg_act)
            self.arm_target = self.default_arm_pos + blend * (self.arm_action_scale * arm_act)
            self.wheel_cmd[:] = blend * (self.wheel_action_scale * wheel_act)

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # 7) Send B2W command
        use_pd_gains = (self.mode == "pd-stand")
        self._write_b2w_hold_or_rl_cmd(
            leg_target=self.leg_target,
            wheel_cmd=self.wheel_cmd,
            use_pd_gains=use_pd_gains,
        )
        self.send_b2w_cmd()

        # 8) Send Z1 lowcmd step
        self.z1.send_arm_command(self.arm_target, self.default_gripper_pos)

        # 9) Bookkeeping / logging
        self.counter += 1
        self.policy_tick += 1

        if self.counter % 100 == 0:
            print(
                f"[{self.counter:5d}] "
                f"mode={self.mode} | "
                f"cmd={self.base_command} | "
                f"blend={blend:.2f} | "
                f"leg_act=[{leg_act.min():+.2f},{leg_act.max():+.2f}] | "
                f"arm_act=[{arm_act.min():+.2f},{arm_act.max():+.2f}] | "
                f"wheel_act=[{wheel_act.min():+.2f},{wheel_act.max():+.2f}]"
            )

    # Main run
    def setup(self):
        print("=" * 80)
        print("B2WZ1 loco-manipulation sim2real")
        print("=" * 80)
        print(f"Mode              : {self.mode}")
        print(f"Policy            : {self.policy_path}")
        print(f"Control dt        : {self.control_dt:.4f}s ({1.0 / self.control_dt:.1f} Hz)")
        print(f"Obs dim per step  : {self.obs_dim_per_step}")
        print(f"Obs stacked dim   : {self.obs_dim}")
        print(f"Action dim        : {self.action_dim}")
        print("=" * 80)

        self.wait_for_low_state()
        InitLowCmd(self.low_cmd)

        self.z1.connect()

        # Read once after both sides are ready
        self._read_all_sensors_once()

        # Do NOT initialize EE sampler / history here.
        # They should be initialized after the whole startup motion finishes,
        # right before RL actually starts.


    def run(self):
        self.setup()

        # ------------------------------------------------------------------
        # Startup state machine
        # ------------------------------------------------------------------
        self.zero_torque_state()

        # B2W: squat -> default
        self.move_b2w_to_pose(
            target_leg_pos_policy=self.squat_leg_pos_policy,
            duration=float(self.cfg["squat_transition_s"]),
        )

        print("[B2WZ1] Hold squat pose. Press A to continue to default pose...")
        while self.remote_controller.button[KeyMap.A] != 1:
            self._write_b2w_hold_or_rl_cmd(
                leg_target=self.squat_leg_pos_policy,
                wheel_cmd=np.zeros(4, dtype=np.float32),
                use_pd_gains=True,
            )
            self.send_b2w_cmd()

            # Keep arm near default during startup
            self.z1.hold_default_step()
            time.sleep(self.control_dt)

        self.move_b2w_to_pose(
            target_leg_pos_policy=self.default_leg_pos_policy,
            duration=float(self.cfg["default_transition_s"]),
        )

        # Z1: move to training default pose
        self.z1.move_to_pose(
            self.default_arm_pos,
            duration=float(self.cfg["arm_default_transition_s"]),
        )

        # Hold default until A is pressed again
        self.hold_default_state()

        # ------------------------------------------------------------------
        # Re-initialize runtime state at the ACTUAL RL start pose
        # ------------------------------------------------------------------
        print("[B2WZ1] Re-initializing observation history at RL start pose...")

        # Read fresh state after all startup motions are done
        self._read_all_sensors_once()

        # Reset policy-side runtime state
        self.last_action[:] = 0.0
        self.leg_target = self.default_leg_pos_policy.copy()
        self.arm_target = self.default_arm_pos.copy()
        self.wheel_cmd[:] = 0.0

        # Clear history buffers explicitly
        self.base_ang_vel_hist.clear()
        self.projected_gravity_hist.clear()
        self.base_cmd_hist.clear()
        self.ee_cmd_hist.clear()
        self.ee_cur_hist.clear()
        self.joint_pos_leg_hist.clear()
        self.joint_pos_arm_hist.clear()
        self.joint_vel_leg_hist.clear()
        self.joint_vel_arm_hist.clear()
        self.joint_vel_wheel_hist.clear()
        self.last_action_hist.clear()

        # Reset EE sampler from the current arm pose at actual RL start
        ee_cur_init_lb = self.compute_ee_current_kp_lb()
        self.ee_cmd_sampler.reset(ee_cur_init_lb, sample_first=True)
        self.ee_cmd_lb_current = self.ee_cmd_sampler.command.copy()

        # Initialize history from the actual RL start state
        self._init_history()

        # Reset counters for clean RL rollout
        self.counter = 0
        self.policy_tick = 0
        self.start_time = time.time()

        print("[B2WZ1] RL started. Press SELECT to stop.")

        try:
            while True:
                tick_start = time.time()

                self.step()

                if self.remote_controller.button[KeyMap.select] == 1:
                    print("[B2WZ1] SELECT pressed. Exit control loop.")
                    break

                sleep_t = self.control_dt - (time.time() - tick_start)
                if sleep_t > 0.0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("[B2WZ1] KeyboardInterrupt received.")

        # Safe exit
        create_damping_cmd(self.low_cmd)
        self.send_b2w_cmd()

        try:
            self.z1.safe_back_to_start()
        except Exception:
            pass

        print("[B2WZ1] Exit.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="Network interface, e.g. enp3s0")
    parser.add_argument("config", type=str, help="Path to yaml config")
    parser.add_argument(
        "--mode",
        type=str,
        default="full-policy",
        choices=["pd-stand", "lock-arm-policy", "full-policy"],
        help="Run mode",
    )
    args = parser.parse_args()

    ChannelFactoryInitialize(0, args.net)

    controller = B2WZ1LocoManipController(cfg_path=args.config, mode=args.mode)
    controller.run()


if __name__ == "__main__":
    main()