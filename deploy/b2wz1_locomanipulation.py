"""
B2WZ1 loco-manipulation sim2real deployment.

Run from repository root:

    python3 deploy/b2wz1_locomanipulation.py <network_interface> deploy/configs/b2wz1_locomanipulation.yaml --mode pd-stand
    python3 deploy/b2wz1_locomanipulation.py <network_interface> deploy/configs/b2wz1_locomanipulation.yaml --mode lock-arm-policy
    python3 deploy/b2wz1_locomanipulation.py <network_interface> deploy/configs/b2wz1_locomanipulation.yaml --mode full-policy

Design goals:
- Startup: use official-style Z1 lowcmd motion + lowcmd hold.
- Runtime: pure lowcmd PD for Z1 at policy rate.
- Observation format should match sim2sim as closely as possible.
"""

import os
import sys
import time
import yaml
import argparse
from collections import deque

import numpy as np
import onnxruntime as ort

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
        self.control_dt = float(self.cfg["control_dt"])

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

        # Base command
        self.base_command = np.zeros(3, dtype=np.float32)
        self.command_scale = np.array(self.cfg["command_scale"], dtype=np.float32).reshape(3,)
        self.command_deadband_lin = float(self.cfg.get("command_deadband_lin", 0.2))
        self.command_deadband_ang = float(self.cfg.get("command_deadband_ang", 0.2))

        # B2W default poses and gains
        self.default_b2w_pos_policy = np.array(self.cfg["default_b2w_pos_policy"], dtype=np.float32).reshape(16,)
        self.squat_b2w_pos_policy = np.array(self.cfg["squat_b2w_pos_policy"], dtype=np.float32).reshape(16,)

        self.kps_rl = np.array(self.cfg["kps_rl"], dtype=np.float32).reshape(16,)
        self.kds_rl = np.array(self.cfg["kds_rl"], dtype=np.float32).reshape(16,)
        self.kps_pd = np.array(self.cfg["kps_pd"], dtype=np.float32).reshape(16,)
        self.kds_pd = np.array(self.cfg["kds_pd"], dtype=np.float32).reshape(16,)

        self.leg_action_scale = float(self.cfg["leg_action_scale"])
        self.arm_action_scale = np.array(self.cfg["arm_action_scale"], dtype=np.float32).reshape(6,)
        self.wheel_action_scale = float(self.cfg["wheel_action_scale"])

        # Action clipping
        self.arm_action_clip = float(self.cfg.get("arm_action_clip", 2.0))

        # Arm defaults
        self.default_arm_pos = np.array(self.cfg["default_arm_pos"], dtype=np.float32).reshape(6,)
        self.default_gripper_pos = float(self.cfg["default_gripper_pos"])

        # Observation defaults
        self.default_joint_pos_policy = np.array(self.cfg["default_joint_pos_policy"], dtype=np.float32).reshape(18,)

        # Policy joint order
        self.policy_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
            "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
        ]

        self.leg_joint_names = self.policy_joint_names[:12]
        self.wheel_joint_names = self.policy_joint_names[12:]
        self.arm_joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

        # B2W hardware order
        self.hardware_joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
        ]

        self.hardware_to_policy_joint_indices = [
            self.hardware_joint_names.index(name) for name in self.policy_joint_names
        ]
        self.policy_to_hardware_joint_indices = [
            self.policy_joint_names.index(name) for name in self.hardware_joint_names
        ]

        self.num_b2w_dof = 16

        # Reorder gains from policy order to hardware order
        self.kps_rl_hw = self.kps_rl[self.policy_to_hardware_joint_indices]
        self.kds_rl_hw = self.kds_rl[self.policy_to_hardware_joint_indices]
        self.kps_pd_hw = self.kps_pd[self.policy_to_hardware_joint_indices]
        self.kds_pd_hw = self.kds_pd[self.policy_to_hardware_joint_indices]

        self.leg_policy_indices = list(range(12))
        self.wheel_policy_indices = list(range(12, 16))

        self.leg_hardware_indices = [
            self.hardware_joint_names.index(name) for name in self.leg_joint_names
        ]
        self.wheel_hardware_indices = [
            self.hardware_joint_names.index(name) for name in self.wheel_joint_names
        ]

        self.wheel_kps_rl_hw = np.zeros(self.num_b2w_dof, dtype=np.float32)
        self.wheel_kds_rl_hw = np.zeros(self.num_b2w_dof, dtype=np.float32)
        self.wheel_kps_pd_hw = np.zeros(self.num_b2w_dof, dtype=np.float32)
        self.wheel_kds_pd_hw = np.zeros(self.num_b2w_dof, dtype=np.float32)
        for hw_idx in self.wheel_hardware_indices:
            self.wheel_kps_rl_hw[hw_idx] = self.kps_rl_hw[hw_idx]
            self.wheel_kds_rl_hw[hw_idx] = self.kds_rl_hw[hw_idx]
            self.wheel_kps_pd_hw[hw_idx] = self.kps_pd_hw[hw_idx]
            self.wheel_kds_pd_hw[hw_idx] = self.kds_pd_hw[hw_idx]

        self.hw_to_wheel_cmd_indices = {
            self.hardware_joint_names.index(name): idx
            for idx, name in enumerate(self.wheel_joint_names)
        }

        self.default_leg_pos_policy = self.default_b2w_pos_policy[self.leg_policy_indices].copy()
        self.squat_leg_pos_policy = self.squat_b2w_pos_policy[self.leg_policy_indices].copy()

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

        # Runtime state
        self.remote_controller = RemoteController()

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        self.crc = CRC()

        # Base state
        self.base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.base_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)

        # B2W state in policy order
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

        # Full-policy arm enable / warmup
        self.arm_enable_delay_s = float(self.cfg.get("arm_enable_delay_s", 2.0))
        self.arm_enable_delay_steps = max(1, int(round(self.arm_enable_delay_s / self.control_dt)))

        self.arm_policy_warmup_s = float(self.cfg.get("arm_policy_warmup_s", 4.0))
        self.arm_policy_warmup_steps = max(1, int(round(self.arm_policy_warmup_s / self.control_dt)))

        # Observation debug in pd-stand mode
        self.debug_obs_enabled = (self.mode == "pd-stand")
        self.debug_obs_print_max = 5
        self.debug_obs_print_count = 0
        self.debug_obs_started = False

    def _resolve_path(self, path_str: str) -> str:
        if os.path.isabs(path_str):
            return path_str
        return os.path.abspath(os.path.join(PROJECT_ROOT, path_str))

    # DDS callbacks
    def low_state_handler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(msg.wireless_remote)

    def wait_for_low_state(self):
        print("[B2WZ1] Waiting for first B2W lowstate...")
        while getattr(self.low_state, "tick", 0) == 0:
            time.sleep(self.control_dt)
        print(f"[B2WZ1] First lowstate received: tick={self.low_state.tick}")

    def send_b2w_cmd(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    # Sensor reading
    def _update_base_command_from_remote(self):
        cmd = np.zeros(3, dtype=np.float32)

        cmd[0] = np.clip(self.remote_controller.ly, -1.0, 1.0) * self.command_scale[0]
        cmd[1] = np.clip(-self.remote_controller.lx, -1.0, 1.0) * self.command_scale[1]
        cmd[2] = np.clip(-self.remote_controller.rx, -1.0, 1.0) * self.command_scale[2]

        lin_norm = np.linalg.norm(cmd[:2], ord=2)
        if lin_norm < self.command_deadband_lin * max(self.command_scale[0], self.command_scale[1]):
            cmd[0] = 0.0
            cmd[1] = 0.0
        if abs(cmd[2]) < self.command_deadband_ang * self.command_scale[2]:
            cmd[2] = 0.0

        self.base_command[:] = cmd

    def _read_b2w_sensors_once(self):
        q = self.low_state.imu_state.quaternion
        self.base_quat_wxyz[:] = quat_unique_wxyz(
            np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)
        )
        self.base_quat_wxyz[:] = self.base_quat_wxyz / max(np.linalg.norm(self.base_quat_wxyz), 1e-8)

        gyro = self.low_state.imu_state.gyroscope
        self.base_ang_vel_b[:] = np.array([gyro[0], gyro[1], gyro[2]], dtype=np.float32)

        self.projected_gravity_b[:] = quat_rotate_inverse_numpy(self.base_quat_wxyz, self.gravity_w)

        for p_idx in range(self.num_b2w_dof):
            hw_idx = self.hardware_to_policy_joint_indices[p_idx]
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

    def print_obs_step_debug(self, obs_step: np.ndarray, tag: str):
        i = 0
        base_ang_vel = obs_step[i:i + 3]; i += 3
        projected_gravity = obs_step[i:i + 3]; i += 3
        base_cmd_dbg = obs_step[i:i + 3]; i += 3
        ee_cmd_dbg = obs_step[i:i + 9]; i += 9
        ee_cur_dbg = obs_step[i:i + 9]; i += 9
        joint_pos_leg_dbg = obs_step[i:i + 12]; i += 12
        joint_pos_arm_dbg = obs_step[i:i + 6]; i += 6
        joint_vel_leg_dbg = obs_step[i:i + 12]; i += 12
        joint_vel_arm_dbg = obs_step[i:i + 6]; i += 6
        joint_vel_wheel_dbg = obs_step[i:i + 4]; i += 4
        last_action_dbg = obs_step[i:i + 22]; i += 22

        print("=" * 96)
        print(f"[OBS-DEBUG] {tag}")
        print(f"base_ang_vel      : {np.round(base_ang_vel, 6)}")
        print(f"projected_gravity : {np.round(projected_gravity, 6)}")
        print(f"base_command      : {np.round(base_cmd_dbg, 6)}")
        print(f"ee_cmd_lb         : {np.round(ee_cmd_dbg, 6)}")
        print(f"ee_cur_lb         : {np.round(ee_cur_dbg, 6)}")
        print(f"joint_pos_leg_rel : {np.round(joint_pos_leg_dbg, 6)}")
        print(f"joint_pos_arm_rel : {np.round(joint_pos_arm_dbg, 6)}")
        print(f"joint_vel_leg     : {np.round(joint_vel_leg_dbg, 6)}")
        print(f"joint_vel_arm     : {np.round(joint_vel_arm_dbg, 6)}")
        print(f"joint_vel_wheel   : {np.round(joint_vel_wheel_dbg, 6)}")
        print(f"last_action       : {np.round(last_action_dbg, 6)}")
        print(f"obs_step_full     : {np.round(obs_step, 6)}")
        print("=" * 96)

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

    # B2W command helpers
    def _write_b2w_pose_cmd_policy(self, target_b2w_pos_policy: np.ndarray, use_pd_gains: bool):
        target_b2w_pos_policy = np.asarray(target_b2w_pos_policy, dtype=np.float32).reshape(16,)
        target_b2w_pos_hw = target_b2w_pos_policy[self.policy_to_hardware_joint_indices]

        if use_pd_gains:
            kps_hw = self.kps_pd_hw
            kds_hw = self.kds_pd_hw
            wheel_kps_hw = self.wheel_kps_pd_hw
            wheel_kds_hw = self.wheel_kds_pd_hw
        else:
            kps_hw = self.kps_rl_hw
            kds_hw = self.kds_rl_hw
            wheel_kps_hw = self.wheel_kps_rl_hw
            wheel_kds_hw = self.wheel_kds_rl_hw

        for i in range(self.num_b2w_dof):
            if i in self.leg_hardware_indices:
                self.low_cmd.motor_cmd[i].q = float(target_b2w_pos_hw[i])
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = float(kps_hw[i])
                self.low_cmd.motor_cmd[i].kd = float(kds_hw[i])
                self.low_cmd.motor_cmd[i].tau = 0.0
            else:
                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = float(wheel_kps_hw[i])
                self.low_cmd.motor_cmd[i].kd = float(wheel_kds_hw[i])
                self.low_cmd.motor_cmd[i].tau = 0.0

    def _write_b2w_rl_cmd(self, leg_target_policy: np.ndarray, wheel_cmd_policy: np.ndarray):
        leg_target_policy = np.asarray(leg_target_policy, dtype=np.float32).reshape(12,)
        wheel_cmd_policy = np.asarray(wheel_cmd_policy, dtype=np.float32).reshape(4,)

        processed_actions_policy = self.default_b2w_pos_policy.copy()
        processed_actions_policy[self.leg_policy_indices] = leg_target_policy
        target_b2w_pos_hw = processed_actions_policy[self.policy_to_hardware_joint_indices]

        for i in range(self.num_b2w_dof):
            if i in self.leg_hardware_indices:
                self.low_cmd.motor_cmd[i].q = float(target_b2w_pos_hw[i])
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = float(self.kps_rl_hw[i])
                self.low_cmd.motor_cmd[i].kd = float(self.kds_rl_hw[i])
                self.low_cmd.motor_cmd[i].tau = 0.0
            else:
                wheel_idx = self.hw_to_wheel_cmd_indices[i]
                vel_cmd = float(wheel_cmd_policy[wheel_idx])

                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = vel_cmd
                self.low_cmd.motor_cmd[i].kp = float(self.wheel_kps_rl_hw[i])
                self.low_cmd.motor_cmd[i].kd = float(self.wheel_kds_rl_hw[i])
                self.low_cmd.motor_cmd[i].tau = 0.0

    def _write_b2w_pd_stand_cmd(self):
        self._write_b2w_pose_cmd_policy(self.default_b2w_pos_policy, use_pd_gains=True)

    # Startup states
    def zero_torque_state(self):
        print("[B2WZ1] Zero torque state. Press START to continue.")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_b2w_cmd()
            time.sleep(self.control_dt)
        print("[B2WZ1] START pressed. Exit zero torque state.")

    def move_b2w_to_pose_policy(
        self,
        target_b2w_pos_policy: np.ndarray,
        duration: float,
        hold_arm: bool = False,
    ):
        print("[B2WZ1] Moving B2W to target pose...")
        target_b2w_pos_policy = np.asarray(target_b2w_pos_policy, dtype=np.float32).reshape(16,)
        num_steps = max(1, int(round(duration / self.control_dt)))

        init_b2w_pos_hw = np.zeros(self.num_b2w_dof, dtype=np.float32)
        for i in range(self.num_b2w_dof):
            init_b2w_pos_hw[i] = self.low_state.motor_state[i].q

        init_b2w_pos_policy = init_b2w_pos_hw[self.hardware_to_policy_joint_indices]

        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            target_step_policy = init_b2w_pos_policy * (1.0 - alpha) + target_b2w_pos_policy * alpha

            self._write_b2w_pose_cmd_policy(
                target_b2w_pos_policy=target_step_policy,
                use_pd_gains=True,
            )
            self.send_b2w_cmd()

            if hold_arm:
                self.z1.hold_pose_lowcmd(
                    self.default_arm_pos.copy(),
                    self.default_gripper_pos,
                )

            time.sleep(self.control_dt)

        print("[B2WZ1] Reached target B2W pose.")

    def hold_arm_default_until_A(self):
        print("[B2WZ1] Holding arm default pose while B2W stays still. Press A to continue to squat...")
        while self.remote_controller.button[KeyMap.A] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_b2w_cmd()

            self.z1.hold_pose_lowcmd(
                self.default_arm_pos.copy(),
                self.default_gripper_pos,
            )

            time.sleep(self.control_dt)

        print("[B2WZ1] A pressed.")

    def hold_all_default_until_A(self):
        print("[B2WZ1] Holding full default pose. Press A to start main loop...")

        debug_counter = 0
        debug_max = 5
        debug_done = False

        if self.mode == "pd-stand":
            print("\n" + "=" * 80)
            print("[DEBUG] PD-STAND: start printing obs at DEFAULT pose")
            print("=" * 80)

            self._read_all_sensors_once()
            self.last_action[:] = 0.0

            ee_cur_lb = self.compute_ee_current_kp_lb()
            self.ee_cmd_lb_current = ee_cur_lb.copy()

        while self.remote_controller.button[KeyMap.A] != 1:
            self._write_b2w_pose_cmd_policy(
                target_b2w_pos_policy=self.default_b2w_pos_policy,
                use_pd_gains=True,
            )
            self.send_b2w_cmd()

            self.z1.hold_pose_lowcmd(
                self.default_arm_pos.copy(),
                self.default_gripper_pos,
            )

            if self.mode == "pd-stand" and not debug_done:
                self._read_all_sensors_once()

                obs_step = self.build_obs_step(self.ee_cmd_lb_current)

                self.print_obs_step_debug(
                    obs_step,
                    tag=f"DEFAULT_POSE step {debug_counter+1}/5"
                )

                debug_counter += 1
                if debug_counter >= debug_max:
                    debug_done = True
                    print("[DEBUG] Finished printing 5 obs.\n")

            time.sleep(self.control_dt)

        print("[B2WZ1] A pressed. Start main loop.")

    def step(self):
        # Read sensors
        self._read_all_sensors_once()

        # Update EE command
        if self.mode == "pd-stand":
            self.ee_cmd_lb_current = self.ee_cmd_sampler.command.copy()
        else:
            if self.policy_tick > 0 and (self.policy_tick % self.ee_resample_interval == 0):
                self.ee_cmd_sampler.resample()
            self.ee_cmd_lb_current = self.ee_cmd_sampler.command.copy()

        # Build current observation step
        obs_step = self.build_obs_step(self.ee_cmd_lb_current)

        if self.debug_obs_enabled and self.debug_obs_started and self.debug_obs_print_count < self.debug_obs_print_max:
            self.print_obs_step_debug(
                obs_step,
                tag=f"control_step={self.policy_tick}, mode={self.mode}"
            )
            self.debug_obs_print_count += 1

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

        # Stack history
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

        # Policy inference
        if self.mode == "pd-stand":
            action = np.zeros(self.action_dim, dtype=np.float32)
        else:
            action = self.session.run(
                [self.output_name],
                {self.input_name: obs_stack[None, :]},
            )[0][0].astype(np.float32)

        self.last_action[:] = action

        # Raw actions
        raw_leg_act = action[self.leg_action_indices].copy()
        raw_arm_act = action[self.arm_action_indices].copy()
        raw_wheel_act = action[self.wheel_action_indices].copy()

        # Use raw leg/wheel actions directly, only clip arm action
        leg_act = raw_leg_act
        wheel_act = raw_wheel_act
        arm_act = np.clip(raw_arm_act, -self.arm_action_clip, self.arm_action_clip)

        # Apply mode logic
        if self.mode == "pd-stand":
            self.leg_target = self.default_leg_pos_policy.copy()
            self.arm_target = self.default_arm_pos.copy()
            self.wheel_cmd[:] = 0.0

            self._write_b2w_pd_stand_cmd()

        elif self.mode == "lock-arm-policy":
            self.leg_target = self.default_leg_pos_policy + self.leg_action_scale * leg_act
            self.arm_target = self.default_arm_pos.copy()
            self.wheel_cmd[:] = self.wheel_action_scale * wheel_act

            self._write_b2w_rl_cmd(
                leg_target_policy=self.leg_target,
                wheel_cmd_policy=self.wheel_cmd,
            )

        elif self.mode == "full-policy":
            self.leg_target = self.default_leg_pos_policy + self.leg_action_scale * leg_act
            self.wheel_cmd[:] = self.wheel_action_scale * wheel_act

            # Arm: first lock for arm_enable_delay_s, then warmup for arm_policy_warmup_s
            if self.policy_tick < self.arm_enable_delay_steps:
                self.arm_target = self.default_arm_pos.copy()
                warmup_alpha = 0.0
            else:
                arm_target_policy = self.default_arm_pos + self.arm_action_scale * arm_act
                arm_target_policy = np.asarray(arm_target_policy, dtype=np.float32).reshape(6,)

                warmup_progress_steps = self.policy_tick - self.arm_enable_delay_steps
                warmup_alpha = min(1.0, float(warmup_progress_steps) / float(self.arm_policy_warmup_steps))

                self.arm_target = (
                    (1.0 - warmup_alpha) * self.default_arm_pos
                    + warmup_alpha * arm_target_policy
                ).astype(np.float32)

            self._write_b2w_rl_cmd(
                leg_target_policy=self.leg_target,
                wheel_cmd_policy=self.wheel_cmd,
            )

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Send B2W command
        self.send_b2w_cmd()

        # Send one Z1 PD command at policy rate
        use_startup_gains = (self.mode == "pd-stand")
        self.z1.track_target_pd_once(
            q_target=self.arm_target.copy(),
            gripper_q_target=self.default_gripper_pos,
            use_startup_gains=use_startup_gains,
        )

        # Bookkeeping
        self.counter += 1
        self.policy_tick += 1

        if self.counter % 100 == 0:
            if self.mode == "full-policy":
                if self.policy_tick < self.arm_enable_delay_steps:
                    arm_phase = "lock"
                    warmup_alpha_dbg = 0.0
                else:
                    arm_phase = "warmup_or_policy"
                    warmup_progress_steps = self.policy_tick - self.arm_enable_delay_steps
                    warmup_alpha_dbg = min(1.0, float(warmup_progress_steps) / float(self.arm_policy_warmup_steps))
            else:
                arm_phase = "n/a"
                warmup_alpha_dbg = 1.0

            print(
                f"[{self.counter:5d}] "
                f"mode={self.mode} | "
                f"cmd={self.base_command} | "
                f"arm_phase={arm_phase} | "
                f"warmup_alpha={warmup_alpha_dbg:.2f} | "
                f"leg_act_raw=[{raw_leg_act.min():+.2f},{raw_leg_act.max():+.2f}] | "
                f"arm_act_raw=[{raw_arm_act.min():+.2f},{raw_arm_act.max():+.2f}] | "
                f"arm_act_clip=[{arm_act.min():+.2f},{arm_act.max():+.2f}] | "
                f"wheel_act_raw=[{raw_wheel_act.min():+.2f},{raw_wheel_act.max():+.2f}] | "
                f"arm_tgt=[{self.arm_target.min():+.2f},{self.arm_target.max():+.2f}]"
            )

            ee_cur_lb = self.compute_ee_current_kp_lb()
            ee_err_lb = self.ee_cmd_lb_current - ee_cur_lb
            print(
                "[EE-OBS] "
                f"ee_cmd={np.round(self.ee_cmd_lb_current, 3)} | "
                f"ee_cur={np.round(ee_cur_lb, 3)} | "
                f"ee_err={np.round(ee_err_lb, 3)}"
            )

            # Debug: arm raw/clipped action vector
            print(
                "[ARM-ACT] "
                f"raw={np.round(raw_arm_act, 3)} | "
                f"clipped={np.round(arm_act, 3)}"
            )

            # Debug: arm tracking error
            arm_q_meas = self.z1.q.copy()
            arm_q_err = self.arm_target - arm_q_meas
            print(
                "[ARM-TRACK] "
                f"q_tgt={np.round(self.arm_target, 3)} | "
                f"q_meas={np.round(arm_q_meas, 3)} | "
                f"q_err={np.round(arm_q_err, 3)}"
            )

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
        print(f"Arm action clip   : +/-{self.arm_action_clip}")
        print(f"Arm enable delay  : {self.arm_enable_delay_s:.2f}s ({self.arm_enable_delay_steps} steps)")
        print(f"Arm warmup        : {self.arm_policy_warmup_s:.2f}s ({self.arm_policy_warmup_steps} steps)")
        print("=" * 80)

        self.wait_for_low_state()
        InitLowCmd(self.low_cmd)
        self.z1.connect()

        self._read_all_sensors_once()

    def run(self):
        self.setup()

        self.zero_torque_state()

        print("[B2WZ1] Moving arm to default before any B2W startup motion...")
        self.z1.move_to_pose_official(
            target_q=self.default_arm_pos.copy(),
            target_gripper=self.default_gripper_pos,
            duration_s=float(self.cfg["arm_default_transition_s"]),
            step_callback=None,
        )

        self.hold_arm_default_until_A()

        self.move_b2w_to_pose_policy(
            target_b2w_pos_policy=self.squat_b2w_pos_policy,
            duration=float(self.cfg["squat_transition_s"]),
            hold_arm=True,
        )

        self.move_b2w_to_pose_policy(
            target_b2w_pos_policy=self.default_b2w_pos_policy,
            duration=float(self.cfg["default_transition_s"]),
            hold_arm=True,
        )

        self.hold_all_default_until_A()

        print("[B2WZ1] Re-initializing observation history at loop start pose...")

        self._read_all_sensors_once()

        self.last_action[:] = 0.0
        self.leg_target = self.default_leg_pos_policy.copy()
        self.arm_target = self.default_arm_pos.copy()
        self.wheel_cmd[:] = 0.0

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

        ee_cur_init_lb = self.compute_ee_current_kp_lb()

        sample_first = False if self.mode == "pd-stand" else True
        self.ee_cmd_sampler.reset(ee_cur_init_lb, sample_first=sample_first)
        self.ee_cmd_lb_current = self.ee_cmd_sampler.command.copy()

        self._init_history()

        self.counter = 0
        self.policy_tick = 0

        if self.debug_obs_enabled:
            self.debug_obs_started = True
            self.debug_obs_print_count = 0

        print("[B2WZ1] Main loop started. Press SELECT to stop.")

        try:
            while True:
                loop_t0 = time.perf_counter()

                self.step()

                if self.remote_controller.button[KeyMap.select] == 1:
                    print("[B2WZ1] SELECT pressed. Exit control loop.")
                    break

                elapsed = time.perf_counter() - loop_t0
                time.sleep(max(0.0, self.control_dt - elapsed))

        except KeyboardInterrupt:
            print("[B2WZ1] KeyboardInterrupt received.")

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