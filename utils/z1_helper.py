import os
import sys
import time
import importlib
import threading
from typing import Tuple

import numpy as np

from utils.math import (
    quat_apply_inverse_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
    quat_from_keypoints_lb,
    quat_from_rotmat_wxyz,
    quat_from_yaw_wxyz,
    quat_mul_wxyz,
    quat_normalize_wxyz,
    quat_slerp_wxyz,
    quat_unique_wxyz,
    quat_angle_wxyz,
    euler_xyz_from_quat_wxyz,
)


def resolve_path(path_str: str, project_root: str) -> str:
    """Resolve a possibly-relative path against project root."""
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(project_root, path_str))


def rotmat_from_rpy_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build a rotation matrix from roll-pitch-yaw.

    Convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)

    rx = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, cr, -sr],
         [0.0, sr, cr]],
        dtype=np.float32,
    )
    ry = np.array(
        [[cp, 0.0, sp],
         [0.0, 1.0, 0.0],
         [-sp, 0.0, cp]],
        dtype=np.float32,
    )
    rz = np.array(
        [[cy, -sy, 0.0],
         [sy,  cy, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    return (rz @ ry @ rx).astype(np.float32)


class PresampledKeypointsInterpolateCommandLBSim:
    """
    Single-environment NumPy version of PresampledKeypointsInterpolateCommandLB.
    """

    def __init__(
        self,
        file_path: str,
        kp_dx: float = 0.30,
        kp_dz: float = 0.30,
        kp0_threshold: float = 0.20,
        rot_threshold: float = 0.40,
        seed: int = 0,
    ):
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(f"Expected npy shape (N, 9), got {arr.shape} from '{file_path}'.")

        self._table = arr
        self._num_rows = int(arr.shape[0])

        self._dx = float(kp_dx)
        self._dz = float(kp_dz)
        self._kp0_threshold = float(kp0_threshold)
        self._rot_threshold = float(rot_threshold)

        self._rng = np.random.default_rng(seed)
        self.keypoints_command_lb = np.zeros(9, dtype=np.float32)
        self._has_cmd = False

    @property
    def command(self) -> np.ndarray:
        return self.keypoints_command_lb.copy()

    def _pick_index(self) -> int:
        return int(self._rng.integers(0, self._num_rows))

    @staticmethod
    def _split_kps(kps_9: np.ndarray):
        return kps_9[0:3], kps_9[3:6], kps_9[6:9]

    @staticmethod
    def _pack_kps(kp0: np.ndarray, kp1: np.ndarray, kp2: np.ndarray) -> np.ndarray:
        return np.concatenate([kp0, kp1, kp2]).astype(np.float32)

    def _kps_from_pose(self, kp0: np.ndarray, quat: np.ndarray):
        off_x = np.array([self._dx, 0.0, 0.0], dtype=np.float32)
        off_z = np.array([0.0, 0.0, self._dz], dtype=np.float32)
        kp1 = kp0 + quat_apply_wxyz(quat, off_x)
        kp2 = kp0 + quat_apply_wxyz(quat, off_z)
        return kp1.astype(np.float32), kp2.astype(np.float32)

    def _compute_next_from_reference(self, ref_kps_lb: np.ndarray, sampled_kps_lb: np.ndarray) -> np.ndarray:
        kp0_s, kp1_s, kp2_s = self._split_kps(sampled_kps_lb)
        kp0_r, kp1_r, kp2_r = self._split_kps(ref_kps_lb)

        quat_r = quat_from_keypoints_lb(kp0_r, kp1_r, kp2_r, self._dx, self._dz)
        quat_s = quat_from_keypoints_lb(kp0_s, kp1_s, kp2_s, self._dx, self._dz)

        delta = kp0_s - kp0_r
        dist = max(float(np.linalg.norm(delta)), 1e-8)
        alpha_pos = min(self._kp0_threshold / dist, 1.0)

        ang = max(float(quat_angle_wxyz(quat_r, quat_s)), 1e-8)
        alpha_rot = min(self._rot_threshold / ang, 1.0)

        alpha = min(alpha_pos, alpha_rot)
        within = (dist <= self._kp0_threshold) and (ang <= self._rot_threshold)
        alpha_eff = 1.0 if within else alpha

        kp0_new = kp0_r + alpha_eff * delta
        quat_new = quat_slerp_wxyz(quat_r, quat_s, float(alpha_eff))
        kp1_new, kp2_new = self._kps_from_pose(kp0_new, quat_new)

        return self._pack_kps(kp0_new, kp1_new, kp2_new)

    def reset(self, initial_kps_lb: np.ndarray, sample_first: bool = True):
        initial_kps_lb = np.asarray(initial_kps_lb, dtype=np.float32).reshape(9,)
        self.keypoints_command_lb = initial_kps_lb.copy()
        self._has_cmd = True

        if sample_first:
            sampled = self._table[self._pick_index()].copy()
            self.keypoints_command_lb = self._compute_next_from_reference(
                ref_kps_lb=initial_kps_lb,
                sampled_kps_lb=sampled,
            )

    def resample(self):
        if not self._has_cmd:
            raise RuntimeError("Command sampler not initialized. Call reset() first.")

        sampled = self._table[self._pick_index()].copy()
        self.keypoints_command_lb = self._compute_next_from_reference(
            ref_kps_lb=self.keypoints_command_lb,
            sampled_kps_lb=sampled,
        )


class Z1ArmAdapter:
    """
    Z1 arm helper with:

    1. blocking startup motion helpers
    2. dedicated 500 Hz runtime lowcmd loop
    3. runtime semantics:
         q_cmd   = policy target
         qd_cmd  = 0
         tau_cmd = PD(q_target - q_meas, 0 - qd_meas)
    """

    def __init__(self, cfg: dict, project_root: str):
        self.project_root = project_root

        z1_sdk_lib = resolve_path(cfg["z1_sdk_lib"], project_root)
        if z1_sdk_lib not in sys.path:
            sys.path.insert(0, z1_sdk_lib)

        self.unitree_arm_interface = importlib.import_module("unitree_arm_interface")

        self.has_gripper = bool(cfg.get("z1_has_gripper", True))
        self.ee_index = int(cfg.get("z1_fk_ee_index", 6))
        self.control_dt = float(cfg["control_dt"])
        self.arm_control_dt = float(cfg.get("z1_control_dt", 0.002))
        self.gripper_kp = float(cfg.get("z1_gripper_kp", 20.0))
        self.gripper_kd = float(cfg.get("z1_gripper_kd", 2000.0))

        # Fixed transforms used by policy EE computation
        self.arm_base_pos_in_base = np.array(cfg["arm_base_offset_pos"], dtype=np.float32).reshape(3,)
        arm_base_rpy = np.array(cfg["arm_base_offset_rpy"], dtype=np.float32).reshape(3,)
        self.arm_base_rot_in_base = rotmat_from_rpy_xyz(*arm_base_rpy)

        self.sdk_ee_to_policy_pos = np.array(cfg["z1_fk_to_policy_ee_pos"], dtype=np.float32).reshape(3,)
        sdk_ee_to_policy_rpy = np.array(cfg["z1_fk_to_policy_ee_rpy"], dtype=np.float32).reshape(3,)
        self.sdk_ee_to_policy_rot = rotmat_from_rpy_xyz(*sdk_ee_to_policy_rpy)

        # Default target
        self.default_arm_pos = np.array(cfg["default_arm_pos"], dtype=np.float32).reshape(6,)
        self.default_gripper_pos = float(cfg["default_gripper_pos"])

        # Startup gains
        self.arm_kps_startup = np.array(
            cfg.get("arm_kps_startup", [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_kds_startup = np.array(
            cfg.get("arm_kds_startup", [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
            dtype=np.float32,
        ).reshape(6,)

        # Runtime gains
        self.arm_kps_runtime = np.array(
            cfg.get("arm_kps_runtime", [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_kds_runtime = np.array(
            cfg.get("arm_kds_runtime", [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
            dtype=np.float32,
        ).reshape(6,)

        self.debug_print = bool(cfg.get("z1_debug_print", False))
        self.runtime_tau_clip = np.array(
            cfg.get("z1_runtime_tau_clip", [60.0, 60.0, 60.0, 30.0, 20.0, 20.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.runtime_target_step_clip = np.array(
            cfg.get("z1_runtime_target_step_clip", [0.03, 0.03, 0.03, 0.04, 0.04, 0.04]),
            dtype=np.float32,
        ).reshape(6,)
        
        self._rt_debug_print_every = int(cfg.get("z1_rt_debug_print_every", 200))
        self._rt_loop_counter = 0

        # communication objects
        self.arm = None
        self.arm_model = None
        self.lowcmd = None

        # measured state cache
        self.q = np.zeros(6, dtype=np.float32)
        self.qd = np.zeros(6, dtype=np.float32)
        self.tau = np.zeros(6, dtype=np.float32)
        self.gripper_q = 0.0

        # last command cache
        self.prev_q_cmd = self.default_arm_pos.copy()
        self.prev_gripper_q_cmd = float(self.default_gripper_pos)

        self._last_applied_kp = None
        self._last_applied_kd = None
        self._debug_counter = 0
        self._fsm_state = None
        self._last_fsm_error_print_time = 0.0

        # locks
        self._comm_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._target_lock = threading.Lock()

        # realtime loop
        self._rt_thread = None
        self._rt_stop_event = threading.Event()
        self._rt_running = False

        # high-level shared target
        self._rt_q_target = self.default_arm_pos.copy()
        self._rt_gripper_target = float(self.default_gripper_pos)
        self._rt_use_startup_gains = False

        print(f"[Z1ArmAdapter] z1_sdk_lib = {z1_sdk_lib}")

    # Connection / state

    def connect(self):
        self.arm = self.unitree_arm_interface.ArmInterface(self.has_gripper)
        self.arm_model = self.arm._ctrlComp.armModel
        self.lowcmd = self.arm._ctrlComp.lowcmd

        if self.arm_model is None:
            raise RuntimeError("Z1 armModel is not accessible from Python binding.")
        if self.lowcmd is None:
            raise RuntimeError("Z1 lowcmd is not accessible from Python binding.")
        if not hasattr(self.lowcmd, "setControlGain"):
            raise RuntimeError("Z1 lowcmd binding does not expose setControlGain().")

        self.arm.setFsmLowcmd()

        for _ in range(20):
            self.arm.sendRecv()
            time.sleep(0.002)

        self._read_state_from_sdk_once()

        self.prev_q_cmd = self.q.copy()
        self.prev_gripper_q_cmd = float(self.gripper_q)

        # self.set_control_gain(self.arm_kps_startup, self.arm_kds_startup)

        print("[Z1ArmAdapter] Connected.")
        print(f"[Z1ArmAdapter] FSM = {self.arm.getCurrentState()}")

    def get_arm_dt(self) -> float:
        if self.arm is None:
            return self.arm_control_dt
        try:
            return float(self.arm._ctrlComp.dt)
        except Exception:
            return self.arm_control_dt

    def _read_state_from_sdk_once(self):
        self.q = np.asarray(self.arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
        self.qd = np.asarray(self.arm.lowstate.getQd(), dtype=np.float32).reshape(6,)
        self.tau = np.asarray(self.arm.lowstate.getTau(), dtype=np.float32).reshape(6,)

        gripper_q_raw = np.asarray(self.arm.lowstate.getGripperQ(), dtype=np.float32).reshape(-1)
        if gripper_q_raw.size > 0:
            self.gripper_q = float(gripper_q_raw[0])
        else:
            self.gripper_q = 0.0

        self._fsm_state = self.arm.getCurrentState()

    def read_state(self):
        if not self.is_realtime_loop_running():
            with self._comm_lock:
                self.arm.sendRecv()
                self._read_state_from_sdk_once()

        with self._state_lock:
            return {
                "q": self.q.copy(),
                "qd": self.qd.copy(),
                "tau": self.tau.copy(),
                "gripper_q": float(self.gripper_q),
                "fsm": self._fsm_state,
            }

    def get_fsm_state(self):
        return self._fsm_state

    # Gain management

    def set_control_gain(self, kp: np.ndarray, kd: np.ndarray):
        kp = np.asarray(kp, dtype=np.float32).reshape(6,)
        kd = np.asarray(kd, dtype=np.float32).reshape(6,)

        if (
            self._last_applied_kp is not None
            and self._last_applied_kd is not None
            and np.allclose(kp, self._last_applied_kp)
            and np.allclose(kd, self._last_applied_kd)
        ):
            return

        kp_full = [float(x) for x in kp] + [self.gripper_kp]
        kd_full = [float(x) for x in kd] + [self.gripper_kd]
        self.lowcmd.setControlGain(kp_full, kd_full)

        self._last_applied_kp = kp.copy()
        self._last_applied_kd = kd.copy()

    def _get_lowcmd_gain_pair(self, use_startup_gains: bool) -> Tuple[np.ndarray, np.ndarray]:
        if use_startup_gains:
            return self.arm_kps_startup, self.arm_kds_startup
        return np.zeros(6, dtype=np.float32), np.zeros(6, dtype=np.float32)

    def _get_external_pd_gain_pair(self, use_startup_gains: bool) -> Tuple[np.ndarray, np.ndarray]:
        if use_startup_gains:
            return self.arm_kps_startup, self.arm_kds_startup
        return self.arm_kps_runtime, self.arm_kds_runtime

    # Helper functions

    def _protect_joint_cmd(self, q_cmd: np.ndarray, qd_cmd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            q_safe, qd_safe = self.arm_model.jointProtect(q_cmd.copy(), qd_cmd.copy())
            return (
                np.asarray(q_safe, dtype=np.float32).reshape(6,),
                np.asarray(qd_safe, dtype=np.float32).reshape(6,),
            )
        except Exception:
            return q_cmd, qd_cmd

    def _compute_runtime_tau(
        self,
        q_target: np.ndarray,
        qd_target: np.ndarray,
        q_meas: np.ndarray,
        qd_meas: np.ndarray,
        use_startup_gains: bool,
    ) -> np.ndarray:
        kp, kd = self._get_external_pd_gain_pair(use_startup_gains)
        tau_cmd = kp * (q_target - q_meas) + kd * (qd_target - qd_meas)
        tau_cmd = np.clip(tau_cmd, -self.runtime_tau_clip, self.runtime_tau_clip)
        return tau_cmd.astype(np.float32)

    # Immediate one-step send

    def _send_arm_command_once(
        self,
        q_cmd: np.ndarray,
        gripper_q_cmd: float,
        use_startup_gains: bool = False,
        qd_cmd: np.ndarray = None,
        tau_cmd: np.ndarray = None,
    ):
        q_cmd = np.asarray(q_cmd, dtype=np.float32).reshape(6,)

        if qd_cmd is None:
            qd_cmd = np.zeros(6, dtype=np.float32)
        else:
            qd_cmd = np.asarray(qd_cmd, dtype=np.float32).reshape(6,)

        if tau_cmd is None:
            tau_cmd = np.zeros(6, dtype=np.float32)
        else:
            tau_cmd = np.asarray(tau_cmd, dtype=np.float32).reshape(6,)

        q_cmd, qd_cmd = self._protect_joint_cmd(q_cmd, qd_cmd)

        kp_cmd, kd_cmd = self._get_lowcmd_gain_pair(use_startup_gains=use_startup_gains)
        self.set_control_gain(kp_cmd, kd_cmd)

        self.arm.q = q_cmd
        self.arm.qd = qd_cmd
        self.arm.tau = tau_cmd
        self.arm.gripperQ = float(gripper_q_cmd)

        self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
        self.arm.setGripperCmd(
            float(gripper_q_cmd),
            self.arm.gripperQd,
            self.arm.gripperTau,
        )
        self.arm.sendRecv()

        with self._state_lock:
            self._read_state_from_sdk_once()
            self.prev_q_cmd = q_cmd.copy()
            self.prev_gripper_q_cmd = float(gripper_q_cmd)

        self._debug_counter += 1
        if self.debug_print and (self._debug_counter % 100 == 0):
            print(
                "[Z1ArmAdapter] send_once:",
                "q_cmd =", np.round(q_cmd, 4),
                "qd_cmd =", np.round(qd_cmd, 4),
                "tau_cmd =", np.round(tau_cmd, 4),
                "fsm =", self._fsm_state,
            )

    def send_arm_command(self, q_cmd: np.ndarray, gripper_q_cmd: float, use_startup_gains: bool = False):
        if self.is_realtime_loop_running():
            raise RuntimeError("send_arm_command() must not be called while realtime loop is running.")

        q_cmd = np.asarray(q_cmd, dtype=np.float32).reshape(6,)
        qd_cmd = np.zeros(6, dtype=np.float32)

        with self._comm_lock:
            q_meas = np.asarray(self.arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
            qd_meas = np.asarray(self.arm.lowstate.getQd(), dtype=np.float32).reshape(6,)

            tau_cmd = self._compute_runtime_tau(
                q_target=q_cmd,
                qd_target=qd_cmd,
                q_meas=q_meas,
                qd_meas=qd_meas,
                use_startup_gains=use_startup_gains,
            )

            self._send_arm_command_once(
                q_cmd=q_cmd,
                gripper_q_cmd=gripper_q_cmd,
                use_startup_gains=use_startup_gains,
                qd_cmd=qd_cmd,
                tau_cmd=tau_cmd,
            )
    
    def send_arm_command_lowcmd_only(
        self,
        q_cmd: np.ndarray,
        gripper_q_cmd: float,
        use_startup_gains: bool = True,
    ):
        """
        Send arm command using lowcmd position control only:
        - nonzero lowcmd gains
        - zero external torque
        """
        if self.is_realtime_loop_running():
            raise RuntimeError("send_arm_command_lowcmd_only() must not be called while realtime loop is running.")

        q_cmd = np.asarray(q_cmd, dtype=np.float32).reshape(6,)
        qd_cmd = np.zeros(6, dtype=np.float32)
        tau_cmd = np.zeros(6, dtype=np.float32)

        with self._comm_lock:
            self._send_arm_command_once(
                q_cmd=q_cmd,
                gripper_q_cmd=gripper_q_cmd,
                use_startup_gains=use_startup_gains,
                qd_cmd=qd_cmd,
                tau_cmd=tau_cmd,
            )

    # Runtime realtime loop

    def is_realtime_loop_running(self) -> bool:
        return self._rt_running and (self._rt_thread is not None) and self._rt_thread.is_alive()

    def start_realtime_loop(
        self,
        initial_q_target: np.ndarray = None,
        gripper_q_target: float = None,
        use_startup_gains: bool = False,
    ):
        if self.is_realtime_loop_running():
            return

        if initial_q_target is None:
            initial_q_target = self.prev_q_cmd.copy()
        if gripper_q_target is None:
            gripper_q_target = float(self.prev_gripper_q_cmd)
        
        self.read_state()
        self.prev_q_cmd = self.q.copy()
        self.prev_gripper_q_cmd = float(self.gripper_q)

        with self._target_lock:
            self._rt_q_target = np.asarray(initial_q_target, dtype=np.float32).reshape(6,)
            self._rt_gripper_target = float(gripper_q_target)
            self._rt_use_startup_gains = bool(use_startup_gains)
        

        self._rt_stop_event.clear()
        self._rt_running = True
        self._rt_thread = threading.Thread(target=self._realtime_loop, daemon=True)
        self._rt_thread.start()

        print("[Z1ArmAdapter] Realtime loop started.")

    def stop_realtime_loop(self):
        if not self.is_realtime_loop_running():
            self._rt_running = False
            return

        self._rt_stop_event.set()
        self._rt_thread.join(timeout=2.0)
        self._rt_running = False
        self._rt_thread = None

        print("[Z1ArmAdapter] Realtime loop stopped.")

    def set_target(
        self,
        q_target: np.ndarray,
        gripper_q_target: float,
        duration_s: float,
        use_startup_gains: bool = False,
    ):
        q_target = np.asarray(q_target, dtype=np.float32).reshape(6,)

        with self._target_lock:
            self._rt_q_target = q_target.copy()
            self._rt_gripper_target = float(gripper_q_target)
            self._rt_use_startup_gains = bool(use_startup_gains)

    def _realtime_loop(self):
        dt = self.get_arm_dt()
        next_tick = time.perf_counter()

        # Runtime should use zero lowcmd gains
        self.set_control_gain(
            np.zeros(6, dtype=np.float32),
            np.zeros(6, dtype=np.float32),
        )

        self._rt_loop_counter = 0

        while not self._rt_stop_event.is_set():
            with self._target_lock:
                q_target = self._rt_q_target.copy()
                gripper_target = float(self._rt_gripper_target)

            with self._comm_lock:
                q_meas = np.asarray(self.arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
                qd_meas = np.asarray(self.arm.lowstate.getQd(), dtype=np.float32).reshape(6,)

                # Keep only a small position smoothing / slew limit
                dq = np.clip(
                    q_target - self.prev_q_cmd,
                    -self.runtime_target_step_clip,
                    self.runtime_target_step_clip,
                )
                q_target_limited = self.prev_q_cmd + dq

                # Match training semantics: desired joint velocity is zero
                qd_cmd = np.zeros(6, dtype=np.float32)

                tau_cmd = (
                    self.arm_kps_runtime * (q_target_limited - q_meas)
                    + self.arm_kds_runtime * (qd_cmd - qd_meas)
                )
                tau_cmd = np.clip(
                    tau_cmd,
                    -self.runtime_tau_clip,
                    self.runtime_tau_clip,
                ).astype(np.float32)

                self.arm.q = q_target_limited.astype(np.float32)
                self.arm.qd = qd_cmd
                self.arm.tau = tau_cmd
                self.arm.gripperQ = float(gripper_target)

                self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
                self.arm.setGripperCmd(
                    self.arm.gripperQ,
                    self.arm.gripperQd,
                    self.arm.gripperTau,
                )
                self.arm.sendRecv()

                with self._state_lock:
                    self._read_state_from_sdk_once()
                    self.prev_q_cmd = q_target_limited.copy()
                    self.prev_gripper_q_cmd = float(gripper_target)

            self._rt_loop_counter += 1

            if self.debug_print and (self._rt_loop_counter % self._rt_debug_print_every == 0):
                print(
                    "[Z1-RT] "
                    f"q_target={np.round(q_target, 3)} | "
                    f"q_tgt_lim={np.round(q_target_limited, 3)} | "
                    f"q_meas={np.round(q_meas, 3)} | "
                    f"qd_meas={np.round(qd_meas, 3)} | "
                    f"tau_cmd={np.round(tau_cmd, 3)} | "
                    f"fsm={self._fsm_state}"
                )

            fsm = self._fsm_state
            if fsm != self.unitree_arm_interface.ArmFSMState.LOWCMD:
                t_now = time.time()
                if t_now - self._last_fsm_error_print_time > 0.2:
                    print(
                        "[Z1ArmAdapter][ERROR] "
                        f"FSM dropped to {fsm} in realtime loop | "
                        f"q_target={np.round(q_target, 3)} | "
                        f"q_tgt_lim={np.round(q_target_limited, 3)} | "
                        f"q_meas={np.round(q_meas, 3)} | "
                        f"qd_meas={np.round(qd_meas, 3)} | "
                        f"tau_cmd={np.round(tau_cmd, 3)}"
                    )
                    self._last_fsm_error_print_time = t_now

            next_tick += dt
            sleep_time = next_tick - time.perf_counter()
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            else:
                next_tick = time.perf_counter()

    # Compatibility wrappers

    def hold_target_once(
        self,
        q_target: np.ndarray,
        gripper_q_target: float,
        use_startup_gains: bool = False,
    ):
        self.send_arm_command(
            q_cmd=q_target,
            gripper_q_cmd=gripper_q_target,
            use_startup_gains=use_startup_gains,
        )

    def hold_target_for_duration(
        self,
        q_target: np.ndarray,
        gripper_q_target: float,
        duration_s: float,
        use_startup_gains: bool = False,
        step_callback=None,
    ):
        if self.is_realtime_loop_running():
            self.set_target(
                q_target=q_target,
                gripper_q_target=gripper_q_target,
                duration_s=duration_s,
                use_startup_gains=use_startup_gains,
            )
            return

        q_target = np.asarray(q_target, dtype=np.float32).reshape(6,)
        dt = self.get_arm_dt()
        num_steps = max(1, int(round(duration_s / dt)))

        for step in range(num_steps):
            self.hold_target_once(
                q_target=q_target,
                gripper_q_target=gripper_q_target,
                use_startup_gains=use_startup_gains,
            )

            fsm = self.arm.getCurrentState()
            if fsm != self.unitree_arm_interface.ArmFSMState.LOWCMD:
                print(f"[Z1ArmAdapter][ERROR] FSM dropped to {fsm} during hold_target_for_duration at step {step+1}")
                break

            if step_callback is not None:
                step_callback()

            time.sleep(dt)

    def track_target_for_duration(
        self,
        q_start: np.ndarray,
        q_target: np.ndarray,
        gripper_q_target: float,
        duration_s: float,
        use_startup_gains: bool = False,
        step_callback=None,
    ):
        if self.is_realtime_loop_running():
            self.set_target(
                q_target=q_target,
                gripper_q_target=gripper_q_target,
                duration_s=duration_s,
                use_startup_gains=use_startup_gains,
            )
            return

        q_start = np.asarray(q_start, dtype=np.float32).reshape(6,)
        q_target = np.asarray(q_target, dtype=np.float32).reshape(6,)

        dt = self.get_arm_dt()
        num_steps = max(1, int(round(duration_s / dt)))

        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            q_cmd = (1.0 - alpha) * q_start + alpha * q_target

            self.send_arm_command(
                q_cmd=q_cmd,
                gripper_q_cmd=gripper_q_target,
                use_startup_gains=use_startup_gains,
            )

            fsm = self.arm.getCurrentState()
            if fsm != self.unitree_arm_interface.ArmFSMState.LOWCMD:
                print(f"[Z1ArmAdapter][ERROR] FSM dropped to {fsm} during track_target_for_duration at step {step+1}")
                break

            if step_callback is not None:
                step_callback()

            time.sleep(dt)

    # Startup helpers

    def move_to_pose(self, target_q: np.ndarray, duration: float, use_startup_gains: bool = True):
        if self.is_realtime_loop_running():
            raise RuntimeError("move_to_pose() must not be used after realtime loop starts.")

        self.read_state()
        q0 = self.q.copy()
        target_q = np.asarray(target_q, dtype=np.float32).reshape(6,)

        dt = self.get_arm_dt()
        num_steps = max(1, int(round(duration / dt)))
        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            q_cmd = (1.0 - alpha) * q0 + alpha * target_q
            self.send_arm_command(
                q_cmd=q_cmd,
                gripper_q_cmd=self.default_gripper_pos,
                use_startup_gains=use_startup_gains,
            )
            time.sleep(dt)

        self.read_state()
        self.prev_q_cmd = self.q.copy()

    def move_to_default_like_official(
        self,
        duration_s: float,
        step_callback=None,
    ):
        """
        Mimic the official Unitree lowcmd startup example:
        - arm lowcmd gains are zero for arm joints
        - q follows a linear trajectory
        - qd is constant trajectory velocity
        - tau is inverse dynamics feedforward only
        """
        if self.is_realtime_loop_running():
            raise RuntimeError("move_to_default_like_official() must not be used after realtime loop starts.")

        self.read_state()
        q0 = self.q.copy()
        target_q = self.default_arm_pos.copy()
        target_gripper = float(self.default_gripper_pos)

        dt = float(self.arm._ctrlComp.dt)
        num_steps = max(1, int(round(duration_s / dt)))

        print("[Z1ArmAdapter] move_to_default_like_official")
        print("[Z1ArmAdapter] q0       =", np.round(q0, 4))
        print("[Z1ArmAdapter] target_q =", np.round(target_q, 4))
        print("[Z1ArmAdapter] dt       =", dt)
        print("[Z1ArmAdapter] steps    =", num_steps)

        self.arm.setFsmLowcmd()
        time.sleep(0.02)

        for _ in range(10):
            self.arm.sendRecv()
            time.sleep(dt)

        # Keep the SDK/default lowcmd gains untouched here
        # to better match the official example behavior.

        qd_traj = ((target_q - q0) / max(duration_s, 1e-6)).astype(np.float32)

        for step in range(num_steps):
            alpha = float(step) / float(num_steps)
            q_cmd = ((1.0 - alpha) * q0 + alpha * target_q).astype(np.float32)
            qd_cmd = qd_traj.copy()

            q_cmd_safe = q_cmd.copy()
            qd_cmd_safe = qd_cmd.copy()

            try:
                tau_ff = np.asarray(
                    self.arm_model.inverseDynamics(
                        q_cmd_safe.astype(np.float32),
                        qd_cmd_safe.astype(np.float32),
                        np.zeros(6, dtype=np.float32),
                        np.zeros(6, dtype=np.float32),
                    ),
                    dtype=np.float32,
                ).reshape(6,)
            except Exception:
                tau_ff = np.zeros(6, dtype=np.float32)

            # Keep a safety clip, but do not add external PD here
            tau_cmd = np.clip(tau_ff, -self.runtime_tau_clip, self.runtime_tau_clip)

            self.arm.q = q_cmd_safe
            self.arm.qd = qd_cmd_safe
            self.arm.tau = tau_cmd
            self.arm.gripperQ = target_gripper

            self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
            self.arm.setGripperCmd(self.arm.gripperQ, self.arm.gripperQd, self.arm.gripperTau)
            self.arm.sendRecv()

            self._read_state_from_sdk_once()

            if step_callback is not None:
                step_callback()

            fsm = self.arm.getCurrentState()
            if (step % 20 == 0) or (step == num_steps - 1) or (fsm != self.unitree_arm_interface.ArmFSMState.LOWCMD):
                q_meas_dbg = np.array(self.arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
                qd_meas_dbg = np.array(self.arm.lowstate.getQd(), dtype=np.float32).reshape(6,)
                tau_meas_dbg = np.array(self.arm.lowstate.getTau(), dtype=np.float32).reshape(6,)
                err = q_cmd_safe - q_meas_dbg

                print(
                    f"[Z1-OFFICIAL {step+1:04d}/{num_steps}] "
                    f"FSM={fsm} | "
                    f"q_cmd={np.round(q_cmd_safe, 3)} | "
                    f"q_meas={np.round(q_meas_dbg, 3)} | "
                    f"err={np.round(err, 3)} | "
                    f"qd_cmd={np.round(qd_cmd_safe, 3)} | "
                    f"qd_meas={np.round(qd_meas_dbg, 3)} | "
                    f"tau_cmd={np.round(tau_cmd, 3)} | "
                    f"tau_meas={np.round(tau_meas_dbg, 3)}"
                )

            if fsm != self.unitree_arm_interface.ArmFSMState.LOWCMD:
                print(f"[Z1ArmAdapter][ERROR] FSM dropped to {fsm} at step {step+1}")
                break

            time.sleep(dt)

        self.read_state()
        self.prev_q_cmd = self.q.copy()
        self.prev_gripper_q_cmd = float(self.gripper_q)

        print("[Z1ArmAdapter] final q =", np.round(self.q, 4))

    def hold_default_step(self):
        self.send_arm_command(
            q_cmd=self.default_arm_pos,
            gripper_q_cmd=self.default_gripper_pos,
            use_startup_gains=True,
        )

    def safe_back_to_start(self):
        pass

    # FK / policy EE conversion

    def compute_policy_ee_pose_in_base(self) -> Tuple[np.ndarray, np.ndarray]:
        q_fk = self.q.copy()

        T_sdk = np.asarray(
            self.arm_model.forwardKinematics(q_fk, self.ee_index),
            dtype=np.float32,
        ).reshape(4, 4)

        R_sdk = T_sdk[:3, :3]
        p_sdk = T_sdk[:3, 3]

        R_policy_in_arm = (R_sdk @ self.sdk_ee_to_policy_rot).astype(np.float32)
        p_policy_in_arm = (p_sdk + R_sdk @ self.sdk_ee_to_policy_pos).astype(np.float32)

        R_policy_in_base = (self.arm_base_rot_in_base @ R_policy_in_arm).astype(np.float32)
        p_policy_in_base = (
            self.arm_base_pos_in_base + self.arm_base_rot_in_base @ p_policy_in_arm
        ).astype(np.float32)

        return p_policy_in_base, R_policy_in_base


def compute_ee_current_kp_lb(
    base_quat_wxyz: np.ndarray,
    z1_adapter: Z1ArmAdapter,
    kp_dx: float,
    kp_dz: float,
) -> np.ndarray:
    """
    Compute current EE keypoints in level-base (LB) frame.
    """
    base_quat_wxyz = quat_unique_wxyz(quat_normalize_wxyz(base_quat_wxyz))

    ee_pos_b, ee_rot_b = z1_adapter.compute_policy_ee_pose_in_base()
    ee_quat_b = quat_from_rotmat_wxyz(ee_rot_b)

    _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_wxyz)
    lb_quat_w = quat_from_yaw_wxyz(yaw)
    lb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(lb_quat_w))

    ee_quat_lb = quat_mul_wxyz(
        quat_conjugate_wxyz(lb_quat_w),
        quat_mul_wxyz(base_quat_wxyz, ee_quat_b),
    )
    ee_quat_lb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_lb))

    ee_pos_w = quat_apply_wxyz(base_quat_wxyz, ee_pos_b)
    ee_pos_lb = quat_apply_inverse_wxyz(lb_quat_w, ee_pos_w)

    off_x = np.array([kp_dx, 0.0, 0.0], dtype=np.float32)
    off_z = np.array([0.0, 0.0, kp_dz], dtype=np.float32)

    kp0 = ee_pos_lb
    kp1 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_x)
    kp2 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_z)

    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)