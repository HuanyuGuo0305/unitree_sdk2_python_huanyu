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
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(project_root, path_str))


def rotmat_from_rpy_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
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
    Z1 arm helper.

    Startup:
        - official lowcmd move_to_pose_official
        - lowcmd hold_pose_lowcmd

    Runtime:
        - pure lowcmd PD at 50 Hz
        - no realtime thread
        - no FF torque
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
        self.arm_control_dt = float(cfg.get("z1_control_dt", self.control_dt))
        self.gripper_kp = float(cfg.get("z1_gripper_kp", 20.0))
        self.gripper_kd = float(cfg.get("z1_gripper_kd", 2000.0))

        self.arm_base_pos_in_base = np.array(cfg["arm_base_offset_pos"], dtype=np.float32).reshape(3,)
        arm_base_rpy = np.array(cfg["arm_base_offset_rpy"], dtype=np.float32).reshape(3,)
        self.arm_base_rot_in_base = rotmat_from_rpy_xyz(*arm_base_rpy)

        self.sdk_ee_to_policy_pos = np.array(cfg["z1_fk_to_policy_ee_pos"], dtype=np.float32).reshape(3,)
        sdk_ee_to_policy_rpy = np.array(cfg["z1_fk_to_policy_ee_rpy"], dtype=np.float32).reshape(3,)
        self.sdk_ee_to_policy_rot = rotmat_from_rpy_xyz(*sdk_ee_to_policy_rpy)

        self.default_arm_pos = np.array(cfg["default_arm_pos"], dtype=np.float32).reshape(6,)
        self.default_gripper_pos = float(cfg["default_gripper_pos"])

        self.arm_kps_startup = np.array(
            cfg.get("arm_kps_startup", [20.0, 30.0, 30.0, 20.0, 15.0, 10.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_kds_startup = np.array(
            cfg.get("arm_kds_startup", [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]),
            dtype=np.float32,
        ).reshape(6,)

        # 运行期纯 PD lowcmd 增益
        self.arm_kps_runtime = np.array(
            cfg.get("arm_kps_runtime", [20.0, 30.0, 30.0, 20.0, 15.0, 10.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_kds_runtime = np.array(
            cfg.get("arm_kds_runtime", [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]),
            dtype=np.float32,
        ).reshape(6,)

        self.debug_print = bool(cfg.get("z1_debug_print", False))
        self.runtime_q_step_clip = np.array(
            cfg.get("z1_runtime_q_step_clip", [0.08, 0.08, 0.08, 0.08, 0.08, 0.08]),
            dtype=np.float32,
        ).reshape(6,)

        self.arm = None
        self.arm_model = None
        self.lowcmd = None

        self.q = np.zeros(6, dtype=np.float32)
        self.qd = np.zeros(6, dtype=np.float32)
        self.tau = np.zeros(6, dtype=np.float32)
        self.gripper_q = 0.0

        self.prev_q_cmd = self.default_arm_pos.copy()
        self.prev_gripper_q_cmd = float(self.default_gripper_pos)

        self._last_applied_kp = None
        self._last_applied_kd = None
        self._debug_counter = 0
        self._fsm_state = None

        self._comm_lock = threading.Lock()
        self._state_lock = threading.Lock()

        print(f"[Z1ArmAdapter] z1_sdk_lib = {z1_sdk_lib}")

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

        print("[Z1ArmAdapter] Connected.")
        print(f"[Z1ArmAdapter] FSM = {self.arm.getCurrentState()}")
        print("[Z1ArmAdapter] Runtime PD gains")
        print("  kp =", np.round(self.arm_kps_runtime, 3))
        print("  kd =", np.round(self.arm_kds_runtime, 3))

    def get_arm_dt(self) -> float:
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

    def _protect_joint_cmd(self, q_cmd: np.ndarray, qd_cmd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            q_safe, qd_safe = self.arm_model.jointProtect(q_cmd.copy(), qd_cmd.copy())
            return (
                np.asarray(q_safe, dtype=np.float32).reshape(6,),
                np.asarray(qd_safe, dtype=np.float32).reshape(6,),
            )
        except Exception:
            return q_cmd, qd_cmd

    def _send_arm_command_once(
        self,
        q_cmd: np.ndarray,
        gripper_q_cmd: float,
        kp_cmd: np.ndarray,
        kd_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        tau_cmd: np.ndarray,
    ):
        q_cmd = np.asarray(q_cmd, dtype=np.float32).reshape(6,)
        kp_cmd = np.asarray(kp_cmd, dtype=np.float32).reshape(6,)
        kd_cmd = np.asarray(kd_cmd, dtype=np.float32).reshape(6,)
        qd_cmd = np.asarray(qd_cmd, dtype=np.float32).reshape(6,)
        tau_cmd = np.asarray(tau_cmd, dtype=np.float32).reshape(6,)

        q_cmd, qd_cmd = self._protect_joint_cmd(q_cmd, qd_cmd)

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

    def hold_pose_lowcmd(
        self,
        q_cmd: np.ndarray,
        gripper_q_cmd: float,
    ):
        q_cmd = np.asarray(q_cmd, dtype=np.float32).reshape(6,)
        qd_cmd = np.zeros(6, dtype=np.float32)
        tau_cmd = np.zeros(6, dtype=np.float32)

        with self._comm_lock:
            self._send_arm_command_once(
                q_cmd=q_cmd,
                gripper_q_cmd=gripper_q_cmd,
                kp_cmd=self.arm_kps_startup,
                kd_cmd=self.arm_kds_startup,
                qd_cmd=qd_cmd,
                tau_cmd=tau_cmd,
            )

    def move_to_pose_official(
        self,
        target_q: np.ndarray,
        target_gripper: float,
        duration_s: float,
        step_callback=None,
    ):
        self.read_state()
        q0 = self.q.copy()
        target_q = np.asarray(target_q, dtype=np.float32).reshape(6,)
        target_gripper = float(target_gripper)

        dt = float(self.arm._ctrlComp.dt)
        num_steps = max(1, int(round(duration_s / dt)))

        print("[Z1ArmAdapter] move_to_pose_official")
        print("[Z1ArmAdapter] q0       =", np.round(q0, 4))
        print("[Z1ArmAdapter] target_q =", np.round(target_q, 4))
        print("[Z1ArmAdapter] dt       =", dt)
        print("[Z1ArmAdapter] steps    =", num_steps)

        self.arm.setFsmLowcmd()
        time.sleep(0.02)

        for _ in range(10):
            self.arm.sendRecv()
            time.sleep(dt)

        qd_traj = ((target_q - q0) / max(duration_s, 1e-6)).astype(np.float32)

        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            q_cmd = ((1.0 - alpha) * q0 + alpha * target_q).astype(np.float32)
            qd_cmd = qd_traj.copy()

            q_cmd_safe, qd_cmd_safe = self._protect_joint_cmd(q_cmd, qd_cmd)

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

            self.arm.q = q_cmd_safe
            self.arm.qd = qd_cmd_safe
            self.arm.tau = tau_ff
            self.arm.gripperQ = target_gripper

            self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
            self.arm.setGripperCmd(
                self.arm.gripperQ,
                self.arm.gripperQd,
                self.arm.gripperTau,
            )
            self.arm.sendRecv()

            self._read_state_from_sdk_once()

            if step_callback is not None:
                step_callback()

            fsm = self.arm.getCurrentState()
            if (step % 20 == 0) or (step == num_steps - 1) or (fsm != self.unitree_arm_interface.ArmFSMState.LOWCMD):
                q_meas_dbg = np.asarray(self.arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
                qd_meas_dbg = np.asarray(self.arm.lowstate.getQd(), dtype=np.float32).reshape(6,)
                tau_meas_dbg = np.asarray(self.arm.lowstate.getTau(), dtype=np.float32).reshape(6,)
                err = q_cmd_safe - q_meas_dbg

                print(
                    f"[Z1-OFFICIAL {step+1:04d}/{num_steps}] "
                    f"FSM={fsm} | "
                    f"q_cmd={np.round(q_cmd_safe, 3)} | "
                    f"q_meas={np.round(q_meas_dbg, 3)} | "
                    f"err={np.round(err, 3)} | "
                    f"qd_cmd={np.round(qd_cmd_safe, 3)} | "
                    f"qd_meas={np.round(qd_meas_dbg, 3)} | "
                    f"tau_cmd={np.round(tau_ff, 3)} | "
                    f"tau_meas={np.round(tau_meas_dbg, 3)}"
                )

            if fsm != self.unitree_arm_interface.ArmFSMState.LOWCMD:
                print(f"[Z1ArmAdapter][ERROR] FSM dropped to {fsm} at step {step+1}")
                break

            time.sleep(dt)

        self.hold_pose_lowcmd(
            q_cmd=target_q,
            gripper_q_cmd=target_gripper,
        )

        self.read_state()
        self.prev_q_cmd = self.q.copy()
        self.prev_gripper_q_cmd = float(self.gripper_q)

        print("[Z1ArmAdapter] final q =", np.round(self.q, 4))

    def track_target_pd_once(
        self,
        q_target: np.ndarray,
        gripper_q_target: float,
        use_startup_gains: bool = False,
    ):
        q_target = np.asarray(q_target, dtype=np.float32).reshape(6,)

        dq = np.clip(
            q_target - self.prev_q_cmd,
            -self.runtime_q_step_clip,
            self.runtime_q_step_clip,
        )
        q_target_limited = self.prev_q_cmd + dq

        kp = self.arm_kps_startup if use_startup_gains else self.arm_kps_runtime
        kd = self.arm_kds_startup if use_startup_gains else self.arm_kds_runtime

        with self._comm_lock:
            self._send_arm_command_once(
                q_cmd=q_target_limited,
                gripper_q_cmd=gripper_q_target,
                kp_cmd=kp,
                kd_cmd=kd,
                qd_cmd=np.zeros(6, dtype=np.float32),
                tau_cmd=np.zeros(6, dtype=np.float32),
            )

        if self.debug_print and (self._debug_counter % 100 == 0):
            print(
                "[Z1-PD-50HZ] "
                f"q_target={np.round(q_target, 3)} | "
                f"q_tgt_lim={np.round(q_target_limited, 3)} | "
                f"q_meas={np.round(self.q, 3)} | "
                f"qd_meas={np.round(self.qd, 3)} | "
                f"kp={np.round(kp, 3)} | "
                f"kd={np.round(kd, 3)} | "
                f"fsm={self._fsm_state}"
            )

    def safe_back_to_start(self):
        pass

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