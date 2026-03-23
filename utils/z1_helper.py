import os
import sys
import time
import importlib
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
    Build rotation matrix from roll-pitch-yaw.

    Convention used here:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    This is the common fixed-axis roll-pitch-yaw composition and is safer for
    configuration-driven extrinsics.
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

    This is intentionally kept numerically aligned with the MuJoCo sim2sim helper.
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
    Thin helper around Unitree Z1 Python binding.

    Design goals:
    - Keep deployment loop close to official `example_lowcmd.py`
    - Use SDK FK (`forwardKinematics(q, 6)`) as the raw EE frame
    - Apply a fixed transform from SDK EE frame -> policy EE frame (gripperStator)
    - Keep manual `sendRecv()` in the 50 Hz main control loop
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

        # Fixed transform: B2W base_link -> arm_base
        self.arm_base_pos_in_base = np.array(cfg["arm_base_offset_pos"], dtype=np.float32).reshape(3,)
        arm_base_rpy = np.array(cfg["arm_base_offset_rpy"], dtype=np.float32).reshape(3,)
        self.arm_base_rot_in_base = rotmat_from_rpy_xyz(*arm_base_rpy)

        # Fixed transform: SDK EE frame -> policy EE frame (gripperStator)
        self.sdk_ee_to_policy_pos = np.array(cfg["z1_fk_to_policy_ee_pos"], dtype=np.float32).reshape(3,)
        sdk_ee_to_policy_rpy = np.array(cfg["z1_fk_to_policy_ee_rpy"], dtype=np.float32).reshape(3,)
        self.sdk_ee_to_policy_rot = rotmat_from_rpy_xyz(*sdk_ee_to_policy_rpy)

        self.default_arm_pos = np.array(cfg["default_arm_pos"], dtype=np.float32).reshape(6,)
        self.default_gripper_pos = float(cfg["default_gripper_pos"])

        self.arm = None
        self.arm_model = None

        self.q = np.zeros(6, dtype=np.float32)
        self.qd = np.zeros(6, dtype=np.float32)
        self.tau = np.zeros(6, dtype=np.float32)
        self.gripper_q = 0.0

        self.prev_q_cmd = self.default_arm_pos.copy()

        print(f"[Z1ArmAdapter] z1_sdk_lib = {z1_sdk_lib}")

    def connect(self):
        """Create the Python arm object and switch to LOWCMD FSM."""
        self.arm = self.unitree_arm_interface.ArmInterface(self.has_gripper)
        self.arm_model = self.arm._ctrlComp.armModel

        # Align with official lowcmd example:
        # switch FSM to LOWCMD, then do manual sendRecv() in user loop.
        self.arm.setFsmLowcmd()

        # Pull several packets so lowstate is fresh.
        for _ in range(20):
            self.arm.sendRecv()
            time.sleep(0.002)

        self.read_state()
        self.prev_q_cmd = self.q.copy()

        print("[Z1ArmAdapter] Connected.")
        print(f"[Z1ArmAdapter] FSM = {self.arm.getCurrentState()}")

    def read_state(self):
        """Update cached arm lowstate."""
        self.arm.sendRecv()

        self.q = np.asarray(self.arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
        self.qd = np.asarray(self.arm.lowstate.getQd(), dtype=np.float32).reshape(6,)
        self.tau = np.asarray(self.arm.lowstate.getTau(), dtype=np.float32).reshape(6,)

        gripper_q_raw = np.asarray(self.arm.lowstate.getGripperQ(), dtype=np.float32).reshape(-1)
        if gripper_q_raw.size > 0:
            self.gripper_q = float(gripper_q_raw[0])
        else:
            self.gripper_q = 0.0

    def _protect_joint_cmd(self, q_cmd: np.ndarray, qd_cmd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use SDK joint protection if available.

        The binding returns a pair `(q_safe, qd_safe)`.
        """
        try:
            q_safe, qd_safe = self.arm_model.jointProtect(q_cmd.copy(), qd_cmd.copy())
            return np.asarray(q_safe, dtype=np.float32).reshape(6,), np.asarray(qd_safe, dtype=np.float32).reshape(6,)
        except Exception:
            return q_cmd, qd_cmd

    def send_arm_command(self, q_cmd: np.ndarray, gripper_q_cmd: float):
        """
        Send one lowcmd step to Z1.

        This follows the official example style:
        - set q
        - set qd
        - set tau via inverseDynamics(q, qd, 0, 0)
        - set gripper
        - sendRecv()
        """
        q_cmd = np.asarray(q_cmd, dtype=np.float32).reshape(6,)
        qd_cmd = ((q_cmd - self.prev_q_cmd) / self.control_dt).astype(np.float32)

        q_cmd, qd_cmd = self._protect_joint_cmd(q_cmd, qd_cmd)

        qdd_cmd = np.zeros(6, dtype=np.float32)
        ftip = np.zeros(6, dtype=np.float32)
        tau_cmd = np.asarray(
            self.arm_model.inverseDynamics(q_cmd, qd_cmd, qdd_cmd, ftip),
            dtype=np.float32,
        ).reshape(6,)

        self.arm.q = q_cmd
        self.arm.qd = qd_cmd
        self.arm.tau = tau_cmd
        self.arm.gripperQ = float(gripper_q_cmd)

        self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
        self.arm.setGripperCmd(float(gripper_q_cmd), 0.0, 0.0)
        self.arm.sendRecv()

        self.prev_q_cmd = q_cmd.copy()

    def move_to_pose(self, target_q: np.ndarray, duration: float):
        self.read_state()
        q0 = self.q.copy()
        target_q = np.asarray(target_q, dtype=np.float32).reshape(6,)

        num_steps = max(1, int(round(duration / self.control_dt)))
        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            q_cmd = (1.0 - alpha) * q0 + alpha * target_q
            self.send_arm_command(q_cmd, self.default_gripper_pos)
            time.sleep(self.control_dt)

        self.read_state()
        self.prev_q_cmd = self.q.copy()

    def hold_default_step(self):
        """Send one hold step at default pose."""
        self.send_arm_command(self.default_arm_pos, self.default_gripper_pos)

    def safe_back_to_start(self):
        """
        Optional SDK-provided recovery motion.

        For the first deployment version, do nothing by default to avoid triggering
        an extra uncontrolled motion at shutdown.
        """
        pass

    def compute_policy_ee_pose_in_base(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return policy EE pose in B2W base_link frame.

        Steps:
        1) SDK FK: forwardKinematics(q, ee_index) -> SDK EE pose in arm_base frame
        2) Apply fixed transform SDK_EE -> policy_EE
        3) Apply fixed transform arm_base -> base_link
        """
        # Important: do not call read_state() here.
        # State reading is handled by the main controller once per tick.

        T_sdk = np.asarray(self.arm_model.forwardKinematics(self.q, self.ee_index), dtype=np.float32).reshape(4, 4)

        R_sdk = T_sdk[:3, :3]
        p_sdk = T_sdk[:3, 3]

        # SDK EE -> policy EE
        R_policy_in_arm = (R_sdk @ self.sdk_ee_to_policy_rot).astype(np.float32)
        p_policy_in_arm = (p_sdk + R_sdk @ self.sdk_ee_to_policy_pos).astype(np.float32)

        # arm_base -> base_link
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

    We do not need global base position. Because LB shares the same origin as base_link,
    only the orientation change body->LB matters.
    """
    base_quat_wxyz = quat_unique_wxyz(quat_normalize_wxyz(base_quat_wxyz))

    ee_pos_b, ee_rot_b = z1_adapter.compute_policy_ee_pose_in_base()
    ee_quat_b = quat_from_rotmat_wxyz(ee_rot_b)

    _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_wxyz)
    lb_quat_w = quat_from_yaw_wxyz(yaw)
    lb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(lb_quat_w))

    # Convert orientation from base frame to level-base frame:
    # q_lb_ee = q_lb_w^{-1} * q_w_b * q_b_ee
    ee_quat_lb = quat_mul_wxyz(
        quat_conjugate_wxyz(lb_quat_w),
        quat_mul_wxyz(base_quat_wxyz, ee_quat_b),
    )
    ee_quat_lb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_lb))

    # Convert position from base frame to level-base frame.
    ee_pos_w = quat_apply_wxyz(base_quat_wxyz, ee_pos_b)
    ee_pos_lb = quat_apply_inverse_wxyz(lb_quat_w, ee_pos_w)

    off_x = np.array([kp_dx, 0.0, 0.0], dtype=np.float32)
    off_z = np.array([0.0, 0.0, kp_dz], dtype=np.float32)

    kp0 = ee_pos_lb
    kp1 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_x)
    kp2 = ee_pos_lb + quat_apply_wxyz(ee_quat_lb, off_z)

    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)