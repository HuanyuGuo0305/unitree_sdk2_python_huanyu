"""
B2 sim2real deloyment script for locomotion tasks.
"""


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


class B2Controller:
    def __init__(self, cfg_path: str) -> None:
        with open(cfg_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 1) Load config parameters
        self.policy_path = config["policy_path"]

        # In policy order
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.num_actions = config["num_actions"]
        self.action_scale = np.array(config["action_scale"], dtype=np.float32)
        self.num_obs = config["num_obs"]
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
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"
        ]

        hardware_joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
        ]

        self.hardware_to_policy_joint_indices = [
            hardware_joint_names.index(name) for name in self.policy_joint_names
        ]
        self.policy_to_hardware_joint_indices = [
            self.policy_joint_names.index(name) for name in hardware_joint_names
        ]

        self.num_dof = len(self.policy_joint_names)

        self.kps_hw = self.kps[self.policy_to_hardware_joint_indices]
        self.kds_hw = self.kds[self.policy_to_hardware_joint_indices]
        
        # 3) ONNX session
        self.session = ort.InferenceSession(self.policy_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print("ONNX input :", self.input_name, self.session.get_inputs()[0].shape)
        print("ONNX output:", self.output_name, self.session.get_outputs()[0].shape)

        # 4) States / Buffer
        self.remote_controller = RemoteController()

        self.actions = np.zeros(self.num_actions, dtype=np.float32)
        self.commands = np.zeros(self.num_commands, dtype=np.float32)

        self.ang_vel_b = np.zeros(3, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.gravity_b = np.zeros(3, dtype=np.float32)

        self.joint_pos = np.zeros(self.num_actions, dtype=np.float32)
        self.joint_vel = np.zeros(self.num_actions, dtype=np.float32)

        self.default_joint_pos = self.default_angles.copy()  # In policy order

        # History buffer
        self.history_length = self.num_history
        self.ang_vel_hist = deque(maxlen=self.history_length)
        self.gravity_hist = deque(maxlen=self.history_length)
        self.commands_hist = deque(maxlen=self.history_length)
        self.joint_pos_hist = deque(maxlen=self.history_length)
        self.joint_vel_hist = deque(maxlen=self.history_length)
        self.actions_hist = deque(maxlen=self.history_length)

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
        """Called every time robot published its lowstate message through rt/lowstate channel."""
        self.low_state = msg
        self.level_flag_ = msg.level_flag
        self.remote_controller.set(self.low_state.wireless_remote)
    
    def wait_for_low_state(self):
        """Block until the first lowstate message is received."""
        print("[B2Controller] Waiting for first lowstate message...")
        self.level_flag_ = 0
        while getattr(self.low_state, "tick", 0) == 0:
            time.sleep(self.control_dt)
        print(f"[B2Controller] First lowstate message received, tick = {self.low_state.tick}.")

    def _read_sensors_once(self):
        """Read imu / joint states from lowstate message."""

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
            hw_idx = self.hardware_to_policy_joint_indices[p_idx]
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
        """Initialize history deque with the current state."""
        
        # Read current sensors once
        self._read_sensors_once()

        base_ang = self.ang_vel_b.copy()
        base_grav = self.gravity_b.copy()
        cmd_vec = self.commands.copy()
        jpos_rel = (self.joint_pos - self.default_joint_pos).copy()
        jvel_rel = self.joint_vel.copy()
        last_act = np.zeros_like(self.actions)

        # Fill history with the same frame
        for _ in range(self.history_length):
            self.ang_vel_hist.append(base_ang.copy())
            self.gravity_hist.append(base_grav.copy())
            self.commands_hist.append(cmd_vec.copy())
            self.joint_pos_hist.append(jpos_rel.copy())
            self.joint_vel_hist.append(jvel_rel.copy())
            self.actions_hist.append(last_act.copy())

        print(f"[B2Controller] History initialized with length = {self.history_length}")
    
    def send_cmd(self):
        if not hasattr(self, "crc"):
            self.crc = CRC()
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)

    def zero_torque_state(self):
        print("[B2Controller] Enter zero torque state. Press START to continue...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd()
            time.sleep(self.control_dt)
        print("[B2Controller] START pressed, exiting zero torque state.")
    
    def move_to_default_pose(self, duration: float = 2.0):
        print("[B2Controller] Moving to default pose...")
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
                self.low_cmd.motor_cmd[i].q = float(target_dof_pos_hw[i])
                self.low_cmd.motor_cmd[i].qd = 0.0
                self.low_cmd.motor_cmd[i].kp = float(self.kps_hw[i])
                self.low_cmd.motor_cmd[i].kd = float(self.kds_hw[i])
                self.low_cmd.motor_cmd[i].tau = 0.0

            self.send_cmd()
            time.sleep(self.control_dt)

        print("[B2Controller] Reached default pose.")
    
    def default_pos_state(self):
        """Stay at default pose until A button is pressed."""
        print("[B2Controller] Holding default pose. Press A to start RL locomotion...")
        target_dof_pos_policy = self.default_angles.copy()
        target_dof_pos_hw = target_dof_pos_policy[self.policy_to_hardware_joint_indices]

        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(self.num_dof):
                self.low_cmd.motor_cmd[i].q = float(target_dof_pos_hw[i])
                self.low_cmd.motor_cmd[i].qd = 0.0
                self.low_cmd.motor_cmd[i].kp = float(self.kps_hw[i])
                self.low_cmd.motor_cmd[i].kd = float(self.kds_hw[i])
                self.low_cmd.motor_cmd[i].tau = 0.0
            self.send_cmd()
            time.sleep(self.control_dt)

        print("[B2Controller] A pressed, starting RL locomotion.")

    
    def step(self):
        """
        - read sensors
        - update history
        - construct obs
        - run ONNX policy
        - map action to joint targets
        - send lowcmd
        """

        if getattr(self, "level_flag_", 0) != 0xFF:
            create_zero_cmd(self.low_cmd)
            self.send_cmd()
            time.sleep(self.control_dt)
            print("[B2Controller] Level flag not OK, sending zero torque command.")
            return

        # 1) Read sensors
        self._read_sensors_once()

        # 2) current obs
        curr_ang_vel = self.ang_vel_b.copy()
        curr_gravity = self.gravity_b.copy()
        curr_commands = self.commands.copy()
        curr_joint_pos = (self.joint_pos - self.default_joint_pos).copy()
        curr_joint_vel = self.joint_vel.copy()
        curr_actions = self.actions.copy()

        # 3) update history
        self.ang_vel_hist.append(curr_ang_vel)
        self.gravity_hist.append(curr_gravity)
        self.commands_hist.append(curr_commands)
        self.joint_pos_hist.append(curr_joint_pos)
        self.joint_vel_hist.append(curr_joint_vel)
        self.actions_hist.append(curr_actions)

        # 4) construct obs
        ang_arr = np.array(self.ang_vel_hist)      # (H, 3)
        grav_arr = np.array(self.gravity_hist)     # (H, 3)
        cmd_arr = np.array(self.commands_hist)     # (H, 3)
        jpos_arr = np.array(self.joint_pos_hist)   # (H, 12)
        jvel_arr = np.array(self.joint_vel_hist)   # (H, 12)
        act_arr = np.array(self.actions_hist)      # (H, 12)

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

        # 6) map action to joint targets (policy order)
        processed_actions_policy = self.actions * self.action_scale + self.default_joint_pos

        # 7) convert to hardware joint order
        target_dof_pos_hw = processed_actions_policy[self.policy_to_hardware_joint_indices]

        # 8) construct LowCmd (only set 12 leg joints, rest keep initialized state)
        for i in range(self.num_dof):
            self.low_cmd.motor_cmd[i].q = float(target_dof_pos_hw[i])
            self.low_cmd.motor_cmd[i].qd = 0.0
            self.low_cmd.motor_cmd[i].kp = float(self.kps_hw[i])
            self.low_cmd.motor_cmd[i].kd = float(self.kds_hw[i])
            self.low_cmd.motor_cmd[i].tau = 0.0

        # 9) send command
        self.send_cmd()
        self.counter += 1

        # 10) control frequency
        time.sleep(self.control_dt)

        if self.counter % 200 == 0:
            print(
                f"[{self.counter:5d}] "
                f"cmd = {self.commands} | "
                f"action = [{self.actions.min():.2f}, {self.actions.max():.2f}]"
            )   


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface, e.g. enp3s0")
    parser.add_argument("config", type=str, help="config file path (yaml)")
    args = parser.parse_args()

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = B2Controller(args.config)

    # Enter zero torque state, wait for START
    controller.zero_torque_state()

    # Smoothly move to default pose
    controller.move_to_default_pose()

    # Wait at default pose, press A to start RL locomotion
    controller.default_pos_state()

    print("[B2Controller] RL locomotion started. Press SELECT to stop.")

    try:
        while True:
            controller.step()
            # Press SELECT to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                print("[B2Controller] SELECT pressed, exiting control loop.")
                break
    except KeyboardInterrupt:
        print("[B2Controller] KeyboardInterrupt, exiting...")
    
    # Exit forward damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd()
    print("[B2Controller] Exit.")