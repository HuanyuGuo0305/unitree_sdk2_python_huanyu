"""
B2W sim2real deployment script for locomotion tasks.
"""


import os
import sys
"""
Ensure the repository root (parent of 'deploy/') is on sys.path 
so sibling packages like 'utils' are importable when running this
script directly from the 'deploy/' directory.
"""
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import time
import yaml
import numpy as np
import onnxruntime as ort
from collections import deque


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

from utils.command_helper import create_damping_cmd, create_zero_cmd, init_rl_cmd_go
from utils.remote_controller import RemoteController, KeyMap
from utils.math import quat_rotate_inverse_numpy as quat_rotate_inverse_np


class B2WController:
    def __init__(self, cfg_path: str) -> None:
        with open(cfg_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
        # 1) Load config parameters
        self.policy_path = config["policy_path"]

        # In policy order
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.squat_angles = np.array(config["squat_angles"], dtype=np.float32)
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.kps_pos = np.array(config["kps_pos"], dtype=np.float32)
        self.kds_pos = np.array(config["kds_pos"], dtype=np.float32)
        self.num_actions = config["num_actions"]
        self.leg_action_scale = float(config["leg_action_scale"])
        self.wheel_action_scale = float(config["wheel_action_scale"])
        self.num_history = config.get("num_history", 5)
        self.num_commands = config["num_commands"]
        if "control_dt" in config:
            self.control_dt = config["control_dt"]
        else:
            sim_dt = config["simulation_dt"]
            control_decimation = config["control_decimation"]
            self.control_dt = sim_dt * control_decimation
        
        # 2) Joint mapping
        self.policy_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
            "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"
        ]

        hardware_joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint"
        ]

        self.hardware_joint_names = hardware_joint_names

        self.hardware_to_policy_joint_indices = [
            hardware_joint_names.index(name) for name in self.policy_joint_names
        ]  # Get hardware index given policy index
        self.policy_to_hardware_joint_indices = [
            self.policy_joint_names.index(name) for name in hardware_joint_names
        ]  # Get policy index given hardware index

        self.num_dof = len(self.policy_joint_names)  # 16

        self.kps_hw = self.kps[self.policy_to_hardware_joint_indices]  # Get corresponding hardware gains given policy order
        self.kds_hw = self.kds[self.policy_to_hardware_joint_indices]
        self.kps_pos_hw = self.kps_pos[self.policy_to_hardware_joint_indices]
        self.kds_pos_hw = self.kds_pos[self.policy_to_hardware_joint_indices]

        # Split leg and wheel joints (IN POLICY ORDER)
        self.leg_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]
        self.wheel_joint_names = [
            "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint"
        ]
        self.num_leg_joints = len(self.leg_joint_names)  # 12
        self.num_wheel_joints = len(self.wheel_joint_names)  # 4

        self.leg_policy_indices = [
            self.policy_joint_names.index(name) for name in self.leg_joint_names
        ]  # [0, ..., 11]
        self.wheel_policy_indices = [
            self.policy_joint_names.index(name) for name in self.wheel_joint_names
        ]  # [12, 13, 14, 15]
        self.leg_hardware_indices = [
            self.hardware_joint_names.index(name) for name in self.leg_joint_names
        ]  # < 12
        self.wheel_hardware_indices = [
            self.hardware_joint_names.index(name) for name in self.wheel_joint_names
        ]
        
        # Wheel gains in hardware order
        self.wheel_kps_hw = np.zeros(self.num_dof, dtype=np.float32)  # [0, ..., 0, wheel kp at index 12, ...13, ...14, ...15]
        self.wheel_kds_hw = np.zeros(self.num_dof, dtype=np.float32)
        for hw_idx in self.wheel_hardware_indices:
            self.wheel_kps_hw[hw_idx] = self.kps_hw[hw_idx]
            self.wheel_kds_hw[hw_idx] = self.kds_hw[hw_idx]
        
        # Wheel: hardware index -> wheel cmd index [0..3] in policy order
        # {13:0, 12:1, 15:2, 14:3} indexes in hardware order
        self.hw_to_wheel_cmd_indices = {
            self.hardware_joint_names.index(name): idx
            for idx, name in enumerate(self.wheel_joint_names)
        }

        self.default_joint_pos = self.default_angles.copy()
        self.squat_joint_pos = self.squat_angles.copy()
        self.default_leg_pos = self.default_joint_pos[self.leg_policy_indices]       # (12,)

        # 3) ONNX session
        self.session = ort.InferenceSession(self.policy_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print("ONNX input :", self.input_name, self.session.get_inputs()[0].shape)
        print("ONNX output:", self.output_name, self.session.get_outputs()[0].shape)

        # 4) Controller / States / Buffer
        self.remote_controller = RemoteController()

        self.actions = np.zeros(self.num_actions, dtype=np.float32)
        self.commands = np.zeros(self.num_commands, dtype=np.float32)

        self.ang_vel_b = np.zeros(3, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.gravity_b = np.zeros(3, dtype=np.float32)

        # Joint states in policy order
        self.joint_pos = np.zeros(self.num_actions, dtype=np.float32)  # (16,)
        self.joint_vel = np.zeros(self.num_actions, dtype=np.float32)  # (16,)

        # Wheel velocity commands (policy order: FL, FR, RL, RR)
        self.wheel_vel_cmds = np.zeros(self.num_wheel_joints, dtype=np.float32)

        # History buffer
        self.history_length = self.num_history
        self.ang_vel_hist = deque(maxlen=self.history_length)    # (H, 3)
        self.gravity_hist = deque(maxlen=self.history_length)    # (H, 3)
        self.commands_hist = deque(maxlen=self.history_length)   # (H, 3)
        self.joint_pos_hist = deque(maxlen=self.history_length)  # (H, 12) legs only
        self.joint_vel_hist = deque(maxlen=self.history_length)  # (H, 16) all joints
        self.actions_hist = deque(maxlen=self.history_length)    # (H, 16)

        self.counter = 0

        # 5) DDS Channels
        self.low_cmd = unitree_go_msg_dds__LowCmd_()  # publisher -> q, kp, kd, tau
        self.low_state = unitree_go_msg_dds__LowState_()  # subscriber <- imu, motor_state, wireless_remote, flags

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self.low_state_handler, 10)

        # Wait for first lowstate message
        self.wait_for_low_state()

        # Initialize cmd
        init_rl_cmd_go(self.low_cmd)

        # Initialize history buffer
        self._init_history_with_current_state()


    def low_state_handler(self, msg: LowStateGo):
        """Called every time robot publishes its lowstate message through rt/lowstate channel."""
        self.low_state = msg
        self.level_flag_ = msg.level_flag  # 0xFF means all good    
        self.remote_controller.set(msg.wireless_remote)

    def wait_for_low_state(self):
        """Block until the first lowstate message is received."""
        print("[B2WController] Waiting for first lowstate message...")
        self.level_flag_ = 0
        while getattr(self.low_state, "tick", 0) == 0:
            time.sleep(self.control_dt)
        print(f"[B2WController] First lowstate message received, tick = {self.low_state.tick}.")
    
    def _read_sensors_once(self):
        """
        Read imu / joint states from lowstate message (POLICY ORDER).
        Get quat, ang_vel_b, gravity_b, joint_pos, joint_vel, commands.
        """

        # 1) IMU: quat (w, x, y, z)
        q = self.low_state.imu_state.quaternion
        self.quat[0] = q[0]
        self.quat[1] = q[1]
        self.quat[2] = q[2]
        self.quat[3] = q[3]

        # 2) IMU: gyroscope (x, y, z) in body frame
        gyro = self.low_state.imu_state.gyroscope
        self.ang_vel_b[0] = gyro[0]
        self.ang_vel_b[1] = gyro[1]
        self.ang_vel_b[2] = gyro[2]

        # 3) gravity in body frame
        self.gravity_b = quat_rotate_inverse_np(self.quat, self.gravity_w)

        # 4) joint pos / vel in policy order
        for p_idx in range(self.num_dof):
            hw_idx = self.hardware_to_policy_joint_indices[p_idx]  # get hardware index
            self.joint_pos[p_idx] = self.low_state.motor_state[hw_idx].q
            self.joint_vel[p_idx] = self.low_state.motor_state[hw_idx].dq
        
        # 5) commands from remote controller (vx, vy, vyaw)
        self._update_commands_from_remote()

    def _update_commands_from_remote(self):
        """Map remote controller inputs to velocity commands."""

        # left joystick: forward/backward -> vx
        self.commands[0] = self.remote_controller.ly
        self.commands[0] = np.clip(self.commands[0], -1.0, 1.0)

        # left joystick: left/right -> vy
        self.commands[1] = -self.remote_controller.lx
        self.commands[1] = np.clip(self.commands[1], -1.0, 1.0)

        # right joystick: left/right -> vyaw
        self.commands[2] = -self.remote_controller.rx
        self.commands[2] = np.clip(self.commands[2], -1.0, 1.0)

        lin_norm = np.linalg.norm(self.commands[:2], ord=2)
        if lin_norm < 0.2:
            self.commands[0] = 0.0
            self.commands[1] = 0.0
        if abs(self.commands[2]) < 0.2:
            self.commands[2] = 0.0
    
    def _init_history_with_current_state(self):
        """Initialize history deque with the current state (B2W obs layout)."""

        # Read current sensors once
        self._read_sensors_once()

        base_ang = self.ang_vel_b.copy()
        base_grav = self.gravity_b.copy()
        cmd_vec = self.commands.copy()

        # legs in policy order
        leg_pos = self.joint_pos[self.leg_policy_indices]       # (12,)
        leg_pos_rel  = (leg_pos - self.default_leg_pos).copy()  # (12,)
        jvel_all = self.joint_vel.copy()                        # (16,)
        last_act = np.zeros_like(self.actions)                  # (16,)

        # Fill history with the same frame
        for _ in range(self.history_length):
            self.ang_vel_hist.append(base_ang.copy())
            self.gravity_hist.append(base_grav.copy())
            self.commands_hist.append(cmd_vec.copy())
            self.joint_pos_hist.append(leg_pos_rel.copy())
            self.joint_vel_hist.append(jvel_all.copy())
            self.actions_hist.append(last_act.copy())
        
        print(f"[B2WController] History initialized with length = {self.history_length}")
    
    def send_cmd(self):
        if not hasattr(self, "crc"):
            self.crc = CRC()
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)
    
    def zero_torque_state(self):
        print("[B2WController] Enter zero torque state. Press START to continue...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd()
            time.sleep(self.control_dt)
        print("[B2WController] START pressed, exiting zero torque state.")
    
    def move_to_squat_pose(self, duration: float = 2.0):
        print("[B2WController] Moving to squat pose...")
        num_steps = int(duration / self.control_dt)
        
        init_dof_pos_hw = np.zeros(self.num_dof, dtype=np.float32)
        for i in range(self.num_dof):
            init_dof_pos_hw[i] = self.low_state.motor_state[i].q
        
        init_dof_pos_policy = init_dof_pos_hw[self.hardware_to_policy_joint_indices]

        for step in range(num_steps):
            alpha = (step + 1) / num_steps
            target_dof_pos_policy = (
                init_dof_pos_policy * (1.0 - alpha) + self.squat_joint_pos * alpha
            )
            target_dof_pos_hw = target_dof_pos_policy[self.policy_to_hardware_joint_indices]
            
            for i in range(self.num_dof):
                # Legs: position control
                if i in self.leg_hardware_indices:  # first 12 joints
                    self.low_cmd.motor_cmd[i].q = float(target_dof_pos_hw[i])
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = float(self.kps_pos_hw[i])
                    self.low_cmd.motor_cmd[i].kd = float(self.kds_pos_hw[i])
                    self.low_cmd.motor_cmd[i].tau = 0.0
                # Wheels: keep stopped
                else:
                    self.low_cmd.motor_cmd[i].q = 0.0
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = float(self.wheel_kps_hw[i])
                    self.low_cmd.motor_cmd[i].kd = float(self.wheel_kds_hw[i])
                    self.low_cmd.motor_cmd[i].tau = 0.0

            self.send_cmd()
            time.sleep(self.control_dt)

        print("[B2WController] Reached squat pose.")

    def move_to_default_pose(self, duration: float = 2.0):
        print("[B2WController] Moving to default pose...")
        num_steps = int(duration / self.control_dt)
        
        init_dof_pos_hw = np.zeros(self.num_dof, dtype=np.float32)
        for i in range(self.num_dof):
            init_dof_pos_hw[i] = self.low_state.motor_state[i].q
        
        init_dof_pos_policy = init_dof_pos_hw[self.hardware_to_policy_joint_indices]

        for step in range(num_steps):
            alpha = (step + 1) / num_steps
            target_dof_pos_policy = (
                init_dof_pos_policy * (1.0 - alpha) + self.default_joint_pos * alpha
            )
            target_dof_pos_hw = target_dof_pos_policy[self.policy_to_hardware_joint_indices]
            
            for i in range(self.num_dof):
                # Legs: position control
                if i in self.leg_hardware_indices:  # first 12 joints
                    self.low_cmd.motor_cmd[i].q = float(target_dof_pos_hw[i])
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = float(self.kps_pos_hw[i])
                    self.low_cmd.motor_cmd[i].kd = float(self.kds_pos_hw[i])
                    self.low_cmd.motor_cmd[i].tau = 0.0
                # Wheels: keep stopped
                else:
                    self.low_cmd.motor_cmd[i].q = 0.0
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = float(self.wheel_kps_hw[i])
                    self.low_cmd.motor_cmd[i].kd = float(self.wheel_kds_hw[i])
                    self.low_cmd.motor_cmd[i].tau = 0.0

            self.send_cmd()
            time.sleep(self.control_dt)

        print("[B2WController] Reached default pose.")

    def squat_pos_state(self):
        """Stay at squat pose until A button is pressed."""
        print("[B2WController] Holding squat pose. Press A to go to default pose...")
        target_dof_pos_policy = self.squat_joint_pos.copy()
        target_dof_pos_hw = target_dof_pos_policy[self.policy_to_hardware_joint_indices]

        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(self.num_dof):
                if i in self.leg_hardware_indices:
                    self.low_cmd.motor_cmd[i].q = float(target_dof_pos_hw[i])
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = float(self.kps_pos_hw[i])
                    self.low_cmd.motor_cmd[i].kd = float(self.kds_pos_hw[i])
                    self.low_cmd.motor_cmd[i].tau = 0.0
                else:
                    self.low_cmd.motor_cmd[i].q = 0.0
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = float(self.wheel_kps_hw[i])
                    self.low_cmd.motor_cmd[i].kd = float(self.wheel_kds_hw[i])
                    self.low_cmd.motor_cmd[i].tau = 0.0
            self.send_cmd()
            time.sleep(self.control_dt)

        print("[B2WController] A pressed, go to default.")
    
    def default_pos_state(self):
        """Stay at default pose until A button is pressed."""
        print("[B2WController] Holding default pose. Press A to start RL locomotion...")
        target_dof_pos_policy = self.default_joint_pos.copy()
        target_dof_pos_hw = target_dof_pos_policy[self.policy_to_hardware_joint_indices]

        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(self.num_dof):
                if i in self.leg_hardware_indices:
                    self.low_cmd.motor_cmd[i].q = float(target_dof_pos_hw[i])
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = float(self.kps_pos_hw[i])
                    self.low_cmd.motor_cmd[i].kd = float(self.kds_pos_hw[i])
                    self.low_cmd.motor_cmd[i].tau = 0.0
                else:
                    self.low_cmd.motor_cmd[i].q = 0.0
                    self.low_cmd.motor_cmd[i].qd = 0.0
                    self.low_cmd.motor_cmd[i].kp = float(self.wheel_kps_hw[i])
                    self.low_cmd.motor_cmd[i].kd = float(self.wheel_kds_hw[i])
                    self.low_cmd.motor_cmd[i].tau = 0.0
            self.send_cmd()
            time.sleep(self.control_dt)

        print("[B2WController] A pressed, starting RL locomotion.")

    
    def step(self):
        """
        - read sensors
        - update history
        - construct obs
        - run ONNX policy
        - slip actions to leg / wheel
        - map to joint targets (legs: pos ctrl, wheels: vel ctrl)
        - send lowcmd
        """

        # 1) read sensors
        self._read_sensors_once()

        # 2) current obs
        curr_ang_vel = self.ang_vel_b.copy()
        curr_gravity = self.gravity_b.copy()
        curr_commands = self.commands.copy()
        leg_pos = self.joint_pos[self.leg_policy_indices]       # (12,)
        leg_pos_rel = (leg_pos - self.default_leg_pos).copy()   # (12,)
        curr_joint_vel = self.joint_vel.copy()    
        curr_actions = self.actions.copy()

        # 3) update history
        self.ang_vel_hist.append(curr_ang_vel)
        self.gravity_hist.append(curr_gravity)
        self.commands_hist.append(curr_commands)
        self.joint_pos_hist.append(leg_pos_rel)
        self.joint_vel_hist.append(curr_joint_vel)
        self.actions_hist.append(curr_actions)

        # 4) construct obs
        ang_arr = np.array(self.ang_vel_hist)      # (H, 3)
        grav_arr = np.array(self.gravity_hist)     # (H, 3)
        cmd_arr = np.array(self.commands_hist)     # (H, 3)
        jpos_arr = np.array(self.joint_pos_hist)   # (H, 12)
        jvel_arr = np.array(self.joint_vel_hist)   # (H, 16)
        act_arr = np.array(self.actions_hist)      # (H, 16)

        obs = np.concatenate(
            [
                ang_arr.reshape(-1),
                grav_arr.reshape(-1),
                cmd_arr.reshape(-1),
                jpos_arr.reshape(-1),
                jvel_arr.reshape(-1),
                act_arr.reshape(-1),
            ],
            dtype=np.float32,
        )

        # 5) run ONNX policy
        self.actions = self.session.run(
            [self.output_name], {self.input_name: obs[None, :]}
        )[0][0].astype(np.float32)

        # 6) split actions to leg / wheel (policy order)
        leg_actions = self.actions[self.leg_policy_indices]      # (12,)
        wheel_actions = self.actions[self.wheel_policy_indices]  # (4,)

        # 6.1 legs: target joint positons in policy order
        desired_leg_pos = self.default_leg_pos + self.leg_action_scale * leg_actions  # (12,)
        processed_actions_policy = self.default_joint_pos.copy()
        processed_actions_policy[self.leg_policy_indices] = desired_leg_pos  # (16,)

        # 6.2 wheels: desired velocities (policy order: FL, FR, RL, RR)
        self.wheel_vel_cmds = self.wheel_action_scale * wheel_actions  # (4,)

        # 7) convert to hardware order
        target_dof_pos_hw = processed_actions_policy[self.policy_to_hardware_joint_indices]

        # 8) construct lowcmd
        for i in range(self.num_dof):
            # Legs: postition control
            if i in self.leg_hardware_indices:
                self.low_cmd.motor_cmd[i].q = float(target_dof_pos_hw[i])
                self.low_cmd.motor_cmd[i].qd = 0.0
                self.low_cmd.motor_cmd[i].kp = float(self.kps_hw[i])
                self.low_cmd.motor_cmd[i].kd = float(self.kds_hw[i])
                self.low_cmd.motor_cmd[i].tau = 0.0
            
            # Wheels: velocity control
            else:
                wheel_idx = self.hw_to_wheel_cmd_indices[i]  # [1, 0, 3, 2]
                vel_cmd = float(self.wheel_vel_cmds[wheel_idx])  # desired wheel velocity

                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].qd = vel_cmd
                self.low_cmd.motor_cmd[i].kp = float(self.wheel_kps_hw[i])
                self.low_cmd.motor_cmd[i].kd = float(self.wheel_kds_hw[i])
                self.low_cmd.motor_cmd[i].tau = 0.0
        
        # 9) send lowcmd
        self.send_cmd()
        self.counter += 1

        # 10) control frequency
        time.sleep(self.control_dt)

        if self.counter % 100 == 0:
            print(
                f"[{self.counter:5d}] "
                f"cmd = {self.commands} | "
                f"leg_action = [{self.actions[self.leg_policy_indices].min():.2f}, "
                f"{self.actions[self.leg_policy_indices].max():.2f}] | "
                f"wheel_action = [{self.actions[self.wheel_policy_indices].min():.2f}, "
                f"{self.actions[self.wheel_policy_indices].max():.2f}]"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface, e.g. enp3s0")
    parser.add_argument("config", type=str, help="config file path (yaml)")
    args = parser.parse_args()

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = B2WController(args.config)

    # Enter zero torque state, wait for START
    controller.zero_torque_state()

    # Smoothly move to squat pose
    controller.move_to_squat_pose()

    # Wait in squat pose, press A to continue
    controller.squat_pos_state()

    # Smoothly move to default pose
    controller.move_to_default_pose()

    # Wait in default pose, press A to start RL locomotion
    controller.default_pos_state()

    print("[B2WController] RL locomotion started. Press SELECT to stop.")
    
    try:
        while True:
            controller.step()
            # Press SELECT to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("[B2WController] SELECT pressed, exiting control loop.")
                break
    except KeyboardInterrupt:
        print("[B2WController] KeyboardInterrupt, exiting...")
    
    # Exit forward damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd()
    print("[B2WController] Exit.")