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


class SequentialKeypointsTrajectoryCommandLBSim:
    """
    Sequential keypoint trajectory sampler.

    Behavior:
      1) follow rows in the npy file sequentially
      2) interpolate between consecutive rows with cubic time scaling
      3) optionally hold at each waypoint
      4) loop forever

    Interface:
      - reset(initial_kps_lb, sample_first=True)
      - update()
      - command property
    """

    def __init__(
        self,
        file_path: str,
        control_dt: float,
        traj_duration_s: float = 2.5,
        hold_duration_s: float = 1.5,
    ):
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(f"Expected npy shape (N,9), got {arr.shape} from '{file_path}'.")

        self._table = arr
        self._num_rows = int(arr.shape[0])

        self._control_dt = float(control_dt)
        self._traj_duration_s = float(traj_duration_s)
        self._hold_duration_s = float(hold_duration_s)
        self._cycle_duration_s = self._traj_duration_s + self._hold_duration_s

        self._steps_per_traj = max(1, int(round(self._traj_duration_s / self._control_dt)))
        self._steps_per_hold = max(0, int(round(self._hold_duration_s / self._control_dt)))

        self._has_cmd = False
        self._row_idx = 0
        self._step_in_phase = 0
        self._phase = "move"

        self.keypoints_command_lb = np.zeros(9, dtype=np.float32)
        self._traj_start_kps_lb = np.zeros(9, dtype=np.float32)
        self._traj_end_kps_lb = np.zeros(9, dtype=np.float32)

    @property
    def command(self) -> np.ndarray:
        return self.keypoints_command_lb.copy()

    @staticmethod
    def _cubic_time_scaling(tau: float) -> float:
        tau = float(np.clip(tau, 0.0, 1.0))
        return 3.0 * tau * tau - 2.0 * tau * tau * tau

    def reset(self, initial_kps_lb: np.ndarray, sample_first: bool = True):
        initial_kps_lb = np.asarray(initial_kps_lb, dtype=np.float32).reshape(9,)

        self._has_cmd = True
        self._row_idx = 0
        self._step_in_phase = 0
        self._phase = "move"

        self.keypoints_command_lb = initial_kps_lb.copy()
        self._traj_start_kps_lb = initial_kps_lb.copy()
        self._traj_end_kps_lb = self._table[0].copy() if sample_first else initial_kps_lb.copy()

    def update(self) -> np.ndarray:
        if not self._has_cmd:
            raise RuntimeError("Command sampler not initialized. Call reset() first.")

        if self._phase == "move":
            tau = self._step_in_phase / max(1, self._steps_per_traj)
            s = self._cubic_time_scaling(tau)

            self.keypoints_command_lb = (
                (1.0 - s) * self._traj_start_kps_lb + s * self._traj_end_kps_lb
            ).astype(np.float32)

            self._step_in_phase += 1

            if self._step_in_phase >= self._steps_per_traj:
                self.keypoints_command_lb = self._traj_end_kps_lb.copy()
                self._phase = "hold"
                self._step_in_phase = 0

        elif self._phase == "hold":
            self.keypoints_command_lb = self._traj_end_kps_lb.copy()
            self._step_in_phase += 1

            if self._step_in_phase >= self._steps_per_hold:
                self._phase = "move"
                self._step_in_phase = 0
                self._row_idx = (self._row_idx + 1) % self._num_rows
                self._traj_start_kps_lb = self._traj_end_kps_lb.copy()
                self._traj_end_kps_lb = self._table[self._row_idx].copy()

        return self.command


class PresampledKeypointsCubicTrajectoryCommandLBSim:
    """
    Single-env NumPy version aligned with the current training command.

    Behavior:
      1) sample raw target from presampled LB table
      2) sample kp0 threshold each cycle
      3) apply adjacent target limit
      4) compute accepted distance
      5) map accepted distance to traj duration using threshold range bounds
      6) hold duration = cycle_duration - traj_duration
      7) cubic interpolation from current reference pose to accepted target
      8) hold at the target until cycle ends
    """

    def __init__(
        self,
        file_path: str,
        control_dt: float,
        kp_dx: float = 0.30,
        kp_dz: float = 0.30,
        kp0_threshold_range=(0.20, 0.30),
        cycle_duration_s: float = 6.0,
        traj_duration_min_s: float = 4.0,
        traj_duration_max_s: float = 5.0,
        seed: int = 0,
    ):
        arr = np.load(file_path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 9:
            raise ValueError(f"Expected npy shape (N,9), got {arr.shape} from '{file_path}'.")

        self._table = arr
        self._num_rows = int(arr.shape[0])

        self._dx = float(kp_dx)
        self._dz = float(kp_dz)

        kp0_threshold_range = np.asarray(kp0_threshold_range, dtype=np.float32).reshape(2,)
        self._kp0_threshold_min = float(min(kp0_threshold_range[0], kp0_threshold_range[1]))
        self._kp0_threshold_max = float(max(kp0_threshold_range[0], kp0_threshold_range[1]))

        self._cycle_duration_s = float(cycle_duration_s)
        self._traj_duration_min_s = float(traj_duration_min_s)
        self._traj_duration_max_s = float(traj_duration_max_s)

        if self._cycle_duration_s <= 0.0:
            raise ValueError(f"Invalid cycle_duration_s={self._cycle_duration_s}")
        if self._traj_duration_min_s <= 0.0 or self._traj_duration_max_s < self._traj_duration_min_s:
            raise ValueError(
                f"Invalid traj duration range: ({self._traj_duration_min_s}, {self._traj_duration_max_s})"
            )
        if self._traj_duration_max_s > self._cycle_duration_s:
            raise ValueError(
                f"traj_duration_max_s {self._traj_duration_max_s} exceeds cycle_duration_s {self._cycle_duration_s}"
            )

        self._control_dt = float(control_dt)
        self._cycle_steps = max(1, int(round(self._cycle_duration_s / self._control_dt)))

        self._rng = np.random.default_rng(seed)

        self.keypoints_command_lb = np.zeros(9, dtype=np.float32)

        self._has_cmd = False
        self._step_in_cycle = 0

        self._traj_start_pos_lb = np.zeros(3, dtype=np.float32)
        self._traj_start_quat_lb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._traj_end_pos_lb = np.zeros(3, dtype=np.float32)
        self._traj_end_quat_lb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self._current_kp0_threshold = self._kp0_threshold_min
        self._current_traj_duration_s = self._traj_duration_min_s
        self._current_hold_duration_s = self._cycle_duration_s - self._current_traj_duration_s

    @property
    def command(self) -> np.ndarray:
        return self.keypoints_command_lb.copy()

    @property
    def current_kp0_threshold(self) -> float:
        return float(self._current_kp0_threshold)

    @property
    def current_traj_duration_s(self) -> float:
        return float(self._current_traj_duration_s)

    @property
    def current_hold_duration_s(self) -> float:
        return float(self._current_hold_duration_s)

    def _pick_index(self) -> int:
        return int(self._rng.integers(0, self._num_rows))

    def _sample_kp0_threshold(self) -> float:
        return float(self._rng.uniform(self._kp0_threshold_min, self._kp0_threshold_max))

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

    @staticmethod
    def _cubic_time_scaling(tau: float) -> float:
        tau = float(np.clip(tau, 0.0, 1.0))
        return 3.0 * tau * tau - 2.0 * tau * tau * tau

    def _apply_adjacent_target_limit(
        self,
        kp0_ref: np.ndarray,
        quat_ref: np.ndarray,
        kp0_raw: np.ndarray,
        quat_raw: np.ndarray,
        kp0_threshold: float,
    ):
        delta = kp0_raw - kp0_ref
        dist = max(float(np.linalg.norm(delta)), 1e-8)

        alpha_pos = min(float(kp0_threshold) / dist, 1.0)
        alpha_eff = 1.0 if dist <= float(kp0_threshold) else alpha_pos

        kp0_new = kp0_ref + alpha_eff * delta
        quat_new = quat_slerp_wxyz(quat_ref, quat_raw, alpha_eff)

        return kp0_new.astype(np.float32), quat_new.astype(np.float32)

    def _compute_traj_duration_from_distance(self, start_pos: np.ndarray, end_pos: np.ndarray) -> float:
        dist_eff = float(np.linalg.norm(end_pos - start_pos))

        dist_min = self._kp0_threshold_min
        dist_max = self._kp0_threshold_max

        if dist_max <= dist_min + 1e-8:
            alpha = 0.0
        else:
            alpha = (dist_eff - dist_min) / (dist_max - dist_min)
            alpha = float(np.clip(alpha, 0.0, 1.0))

        traj = self._traj_duration_min_s + alpha * (self._traj_duration_max_s - self._traj_duration_min_s)
        hold = self._cycle_duration_s - traj

        self._current_traj_duration_s = float(traj)
        self._current_hold_duration_s = float(hold)
        return self._current_traj_duration_s

    def _start_new_cycle_from_reference(self, ref_kps_lb: np.ndarray):
        ref_kps_lb = np.asarray(ref_kps_lb, dtype=np.float32).reshape(9,)
        kp0_ref, kp1_ref, kp2_ref = self._split_kps(ref_kps_lb)
        quat_ref = quat_from_keypoints_lb(kp0_ref, kp1_ref, kp2_ref, self._dx, self._dz).astype(np.float32)

        sampled = self._table[self._pick_index()].copy()
        kp0_raw, kp1_raw, kp2_raw = self._split_kps(sampled)
        quat_raw = quat_from_keypoints_lb(kp0_raw, kp1_raw, kp2_raw, self._dx, self._dz).astype(np.float32)

        self._current_kp0_threshold = self._sample_kp0_threshold()

        kp0_end, quat_end = self._apply_adjacent_target_limit(
            kp0_ref=kp0_ref,
            quat_ref=quat_ref,
            kp0_raw=kp0_raw,
            quat_raw=quat_raw,
            kp0_threshold=self._current_kp0_threshold,
        )

        self._compute_traj_duration_from_distance(kp0_ref, kp0_end)

        self._traj_start_pos_lb = kp0_ref.astype(np.float32)
        self._traj_start_quat_lb = quat_ref.astype(np.float32)
        self._traj_end_pos_lb = kp0_end.astype(np.float32)
        self._traj_end_quat_lb = quat_end.astype(np.float32)

        kp1_start, kp2_start = self._kps_from_pose(self._traj_start_pos_lb, self._traj_start_quat_lb)
        self.keypoints_command_lb = self._pack_kps(self._traj_start_pos_lb, kp1_start, kp2_start)

        self._step_in_cycle = 0
        self._has_cmd = True

    def reset(self, initial_kps_lb: np.ndarray, sample_first: bool = True):
        initial_kps_lb = np.asarray(initial_kps_lb, dtype=np.float32).reshape(9,)
        self.keypoints_command_lb = initial_kps_lb.copy()
        self._has_cmd = False
        self._step_in_cycle = 0
        self._current_kp0_threshold = self._kp0_threshold_min
        self._current_traj_duration_s = self._traj_duration_min_s
        self._current_hold_duration_s = self._cycle_duration_s - self._current_traj_duration_s

        kp0_init, kp1_init, kp2_init = self._split_kps(initial_kps_lb)
        quat_init = quat_from_keypoints_lb(kp0_init, kp1_init, kp2_init, self._dx, self._dz).astype(np.float32)

        self._traj_start_pos_lb = kp0_init.astype(np.float32)
        self._traj_start_quat_lb = quat_init.astype(np.float32)
        self._traj_end_pos_lb = kp0_init.astype(np.float32)
        self._traj_end_quat_lb = quat_init.astype(np.float32)

        if sample_first:
            self._start_new_cycle_from_reference(initial_kps_lb)
        else:
            self._has_cmd = True

    def _eval_current_command(self):
        if self._step_in_cycle <= 0:
            tau = 0.0
        else:
            t = min(self._step_in_cycle * self._control_dt, self._cycle_duration_s)
            tau = min(t / max(self._current_traj_duration_s, 1e-6), 1.0)

        s = self._cubic_time_scaling(tau)

        pos = self._traj_start_pos_lb + s * (self._traj_end_pos_lb - self._traj_start_pos_lb)
        quat = quat_slerp_wxyz(self._traj_start_quat_lb, self._traj_end_quat_lb, s)

        kp1, kp2 = self._kps_from_pose(pos, quat)
        self.keypoints_command_lb = self._pack_kps(pos, kp1, kp2)

    def update(self) -> np.ndarray:
        if not self._has_cmd:
            raise RuntimeError("Command sampler not initialized. Call reset() first.")

        self._eval_current_command()

        self._step_in_cycle += 1
        if self._step_in_cycle >= self._cycle_steps:
            self._start_new_cycle_from_reference(self.keypoints_command_lb.copy())

        return self.command


class Z1ArmAdapter:
    """
    Z1 arm helper.

    Startup:
        - official lowcmd move_to_pose_official
        - lowcmd hold_pose_lowcmd

    Runtime:
        - arm: pure lowcmd PD at policy rate
        - optional gripper: external IsaacLab DCMotor through tau_f
        - no realtime thread
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

        # External gripper actuator used by HL sim2real.
        # These values match the IsaacLab DCMotor used in training.
        self.gripper_q_offset = float(cfg.get("z1_gripper_q_offset", 0.02367))
        self.gripper_dcmotor_kp = float(cfg.get("z1_gripper_dcmotor_kp", 76.8))
        self.gripper_dcmotor_kd = float(cfg.get("z1_gripper_dcmotor_kd", 4.0))
        self.gripper_dcmotor_effort_limit = float(
            cfg.get("z1_gripper_dcmotor_effort_limit", 30.0)
        )
        self.gripper_dcmotor_saturation_effort = float(
            cfg.get("z1_gripper_dcmotor_saturation_effort", 30.0)
        )
        self.gripper_dcmotor_velocity_limit = float(
            cfg.get("z1_gripper_dcmotor_velocity_limit", 2.0)
        )

        # Runtime gripper actuator mode.
        #
        # The gripper is ALWAYS driven by the exact training/sim2sim IdealPD +
        # IsaacLab DCMotor torque-speed law, with no post-DCMotor
        # deployment-only torque cap.  There is no alternative gripper mode:
        # the legacy position servo does not reproduce the trained grasp
        # dynamics and is used only by the startup/hold helpers.
        self.gripper_runtime_mode = str(
            cfg.get("z1_gripper_runtime_mode", "dcmotor")
        ).strip().lower()

        if self.gripper_runtime_mode != "dcmotor":
            raise ValueError(
                "Invalid z1_gripper_runtime_mode="
                f"{self.gripper_runtime_mode!r}; the runtime gripper only "
                "supports 'dcmotor'."
            )

        # Runtime ARM actuator mode.
        #
        # "position_pd":
        #     Z1 firmware closes the arm position loop with
        #     arm_kps_runtime / arm_kds_runtime.  q/qd are commanded, tau = 0.
        #
        # "dcmotor":
        #     Training-exact external torque.  The firmware arm gains are
        #     forced to zero and
        #         tau = Kp_train * (q_target - q) - Kd_train * qd
        #     clipped to the per-joint training effort limits, is sent as tau_f.
        #
        #     NOTE: training models the 6 arm joints as IdealPD + symmetric
        #     effort clip; unlike jointGripper there is NO torque-speed
        #     envelope on the arm, so none is applied here.
        self.arm_runtime_mode = str(
            cfg.get("z1_arm_runtime_mode", "position_pd")
        ).strip().lower()

        valid_arm_runtime_modes = {
            "position_pd",
            "dcmotor",
        }
        if self.arm_runtime_mode not in valid_arm_runtime_modes:
            raise ValueError(
                "Invalid z1_arm_runtime_mode="
                f"{self.arm_runtime_mode!r}; expected one of "
                f"{sorted(valid_arm_runtime_modes)}."
            )

        # Training arm actuator constants (IsaacLab / sim2sim).
        self.arm_dcmotor_kp = np.array(
            cfg.get("z1_arm_dcmotor_kp", [76.8, 89.6, 89.6, 76.8, 76.8, 76.8]),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_dcmotor_kd = np.array(
            cfg.get("z1_arm_dcmotor_kd", [4.0] * 6),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_dcmotor_effort_limit = np.array(
            cfg.get(
                "z1_arm_dcmotor_effort_limit",
                [30.0, 60.0, 30.0, 30.0, 30.0, 30.0],
            ),
            dtype=np.float32,
        ).reshape(6,)

        if not np.all(self.arm_dcmotor_effort_limit > 0.0):
            raise ValueError(
                "z1_arm_dcmotor_effort_limit must be positive, got "
                f"{self.arm_dcmotor_effort_limit}."
            )

        self._arm_external_torque_active = False

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

        self.arm_kps_runtime = np.array(
            cfg.get("arm_kps_runtime", [20.0, 30.0, 30.0, 20.0, 15.0, 10.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_kds_runtime = np.array(
            cfg.get("arm_kds_runtime", [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]),
            dtype=np.float32,
        ).reshape(6,)

        self.debug_print = bool(cfg.get("z1_debug_print", False))

        self.arm = None
        self.arm_model = None
        self.lowcmd = None

        self.q = np.zeros(6, dtype=np.float32)
        self.qd = np.zeros(6, dtype=np.float32)
        self.tau = np.zeros(6, dtype=np.float32)
        self.gripper_q = 0.0
        self.gripper_qd = 0.0

        self.prev_q_cmd = self.default_arm_pos.copy()
        self.prev_gripper_q_cmd = float(self.default_gripper_pos)

        self._last_applied_kp = None
        self._last_applied_kd = None
        self._gripper_external_dcmotor_active = False
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
            raise RuntimeError(
                "Z1 lowcmd binding does not expose setControlGain()."
            )

        # ============================================================
        # 1. Start communication in PASSIVE.
        #
        # IMPORTANT:
        # Do NOT enter LOWCMD before obtaining a valid measured state.
        # ============================================================
        print("[Z1ArmAdapter] Starting communication in PASSIVE...")

        self.arm.loopOn()
        self.arm.setFsm(
            self.unitree_arm_interface.ArmFSMState.PASSIVE
        )

        time.sleep(0.5)

        # Allow lowstate to become valid/stable.
        for _ in range(20):
            time.sleep(0.002)

        # ============================================================
        # 2. Read actual current arm/gripper state.
        # ============================================================
        self._read_state_from_sdk_once()

        q_hold = self.q.copy()
        gripper_hold = float(self.gripper_q)

        if not np.all(np.isfinite(q_hold)):
            raise RuntimeError(
                f"Invalid initial Z1 joint state: {q_hold}"
            )

        if not np.isfinite(gripper_hold):
            raise RuntimeError(
                f"Invalid initial Z1 gripper state: {gripper_hold}"
            )

        print("[Z1ArmAdapter] Initial measured state:")
        print("  q          =", np.round(q_hold, 4))
        print("  gripper_q  =", round(gripper_hold, 4))
        print("  fsm        =", self.arm.getCurrentState())

        # ============================================================
        # 3. Initialize ALL LOWCMD command buffers from measured state.
        #
        # Therefore the first controlled state should have:
        #
        #     q_cmd - q_meas ~= 0
        #
        # even if the arm was manually placed somewhere other than
        # the nominal/default pose.
        # ============================================================
        qd_zero = np.zeros(6, dtype=np.float32)
        tau_zero = np.zeros(6, dtype=np.float32)

        self.arm.q = q_hold.copy()
        self.arm.qd = qd_zero.copy()
        self.arm.tau = tau_zero.copy()

        self.arm.gripperQ = gripper_hold
        self.arm.gripperQd = 0.0
        self.arm.gripperTau = 0.0

        # Use the existing startup gains.
        kp_full = (
            [float(x) for x in self.arm_kps_startup]
            + [float(self.gripper_kp)]
        )
        kd_full = (
            [float(x) for x in self.arm_kds_startup]
            + [float(self.gripper_kd)]
        )

        self.lowcmd.setControlGain(kp_full, kd_full)

        # Put current-pose hold commands into the SDK command object
        # BEFORE switching to LOWCMD.
        self.arm.setArmCmd(
            self.arm.q,
            self.arm.qd,
            self.arm.tau,
        )

        self.arm.setGripperCmd(
            float(self.arm.gripperQ),
            float(self.arm.gripperQd),
            float(self.arm.gripperTau),
        )

        # ============================================================
        # 4. Stop the SDK background loop.
        #
        # Runtime in this adapter uses explicit sendRecv(), so avoid
        # having loopOn() and our explicit command loop operate at the
        # same time.
        # ============================================================
        self.arm.loopOff()

        time.sleep(0.02)

        # ============================================================
        # 5. Enter LOWCMD only after commands have been initialized.
        # ============================================================
        print("[Z1ArmAdapter] Entering LOWCMD with current-pose hold...")

        self.arm.setFsmLowcmd()
        time.sleep(0.02)

        # ============================================================
        # 6. Send current-pose hold packets during LOWCMD transition.
        #
        # Do NOT just call sendRecv() with potentially stale command
        # contents.
        # ============================================================
        transition_steps = 20
        transition_dt = 0.002

        for i in range(transition_steps):
            # Keep re-writing the known-safe hold command.
            self.arm.q = q_hold.copy()
            self.arm.qd = qd_zero.copy()
            self.arm.tau = tau_zero.copy()

            self.arm.gripperQ = gripper_hold
            self.arm.gripperQd = 0.0
            self.arm.gripperTau = 0.0

            self.arm.setArmCmd(
                self.arm.q,
                self.arm.qd,
                self.arm.tau,
            )

            self.arm.setGripperCmd(
                float(self.arm.gripperQ),
                float(self.arm.gripperQd),
                float(self.arm.gripperTau),
            )

            self.arm.sendRecv()

            # Read state after every command so we can detect a jump
            # immediately instead of waiting until the end.
            self._read_state_from_sdk_once()

            q_err = q_hold - self.q
            max_err = float(np.max(np.abs(q_err)))

            if max_err > 0.10:
                print("[Z1ArmAdapter][ERROR] Unsafe LOWCMD startup motion.")
                print("  q_hold =", np.round(q_hold, 4))
                print("  q_meas =", np.round(self.q, 4))
                print("  q_err  =", np.round(q_err, 4))
                print("  step   =", i)

                # Best-effort exit from LOWCMD.
                try:
                    # Send zero-feedforward/current-position command once.
                    self.arm.setArmCmd(
                        self.q.copy(),
                        qd_zero.copy(),
                        tau_zero.copy(),
                    )

                    self.arm.setGripperCmd(
                        float(self.gripper_q),
                        0.0,
                        0.0,
                    )

                    self.arm.sendRecv()
                except Exception:
                    pass

                try:
                    self.arm.loopOn()
                    self.arm.setFsm(
                        self.unitree_arm_interface.ArmFSMState.PASSIVE
                    )
                    time.sleep(0.2)
                    self.arm.loopOff()
                except Exception:
                    pass

                raise RuntimeError(
                    "Unsafe Z1 LOWCMD initialization jump: "
                    f"max |q_hold - q_meas| = {max_err:.4f} rad"
                )

            time.sleep(transition_dt)

        # ============================================================
        # 7. Final verification.
        # ============================================================
        self._read_state_from_sdk_once()

        q_err = q_hold - self.q
        max_err = float(np.max(np.abs(q_err)))

        fsm = self.arm.getCurrentState()

        print("[Z1ArmAdapter] LOWCMD initialization result:")
        print("  q_hold =", np.round(q_hold, 4))
        print("  q_meas =", np.round(self.q, 4))
        print("  q_err  =", np.round(q_err, 4))
        print("  maxerr =", round(max_err, 5))
        print("  fsm    =", fsm)

        if (
            fsm
            != self.unitree_arm_interface.ArmFSMState.LOWCMD
        ):
            raise RuntimeError(
                f"Z1 failed to remain in LOWCMD. Current FSM={fsm}"
            )

        if max_err > 0.10:
            raise RuntimeError(
                "Unsafe Z1 LOWCMD initialization jump after transition: "
                f"max error = {max_err:.4f} rad"
            )

        # ============================================================
        # 8. Synchronize adapter-side previous-command state.
        # ============================================================
        self.prev_q_cmd = self.q.copy()
        self.prev_gripper_q_cmd = float(self.gripper_q)

        # Reset gain cache because we applied startup gains directly
        # through lowcmd.setControlGain().
        #
        # This ensures the next normal adapter command explicitly
        # applies whichever gains it requests.
        self._last_applied_kp = None
        self._last_applied_kd = None

        self._fsm_state = fsm

        print("[Z1ArmAdapter] Connected safely.")
        print("[Z1ArmAdapter] Runtime PD gains:")
        print("  kp =", np.round(self.arm_kps_runtime, 3))
        print("  kd =", np.round(self.arm_kds_runtime, 3))
        print(
            "[Z1ArmAdapter] Gripper runtime mode:",
            self.gripper_runtime_mode,
        )
        if self.gripper_runtime_mode == "dcmotor":
            print(
                "[Z1ArmAdapter] Gripper DCMotor: "
                f"Kp={self.gripper_dcmotor_kp:.1f}, "
                f"Kd={self.gripper_dcmotor_kd:.1f}, "
                f"effort={self.gripper_dcmotor_effort_limit:.1f} Nm, "
                f"stall={self.gripper_dcmotor_saturation_effort:.1f} Nm, "
                f"vel={self.gripper_dcmotor_velocity_limit:.1f} rad/s, "
                "post_cap=NONE"
            )

    def get_arm_dt(self) -> float:
        return self.arm_control_dt

    def _read_state_from_sdk_once(self):
        self.q = np.asarray(self.arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
        self.qd = np.asarray(self.arm.lowstate.getQd(), dtype=np.float32).reshape(6,)
        self.tau = np.asarray(self.arm.lowstate.getTau(), dtype=np.float32).reshape(6,)

        gripper_q_raw = np.asarray(self.arm.lowstate.getGripperQ(), dtype=np.float32).reshape(-1)
        gripper_qd_raw = np.asarray(self.arm.lowstate.getGripperQd(), dtype=np.float32).reshape(-1)

        self.gripper_q = float(gripper_q_raw[0]) if gripper_q_raw.size > 0 else 0.0
        self.gripper_qd = float(gripper_qd_raw[0]) if gripper_qd_raw.size > 0 else 0.0

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
                "gripper_qd": float(self.gripper_qd),
                "fsm": self._fsm_state,
            }

    def get_fsm_state(self):
        return self._fsm_state

    def get_gripper_q_training(self) -> float:
        """
        Return the measured gripper position in the TRAINING coordinate.

            q_training = q_sdk - gripper_q_offset

        Training convention:
            fully closed ~= 0
            fully open   ~= -pi/2
        """
        return float(
            float(self.gripper_q)
            - float(self.gripper_q_offset)
        )

    def gripper_training_to_sdk(self, q_training: float) -> float:
        """
        Convert a training-space gripper position to the raw SDK coordinate.

            q_sdk = q_training + gripper_q_offset

        Used by startup / legacy position-servo paths.
        """
        return float(
            float(q_training)
            + float(self.gripper_q_offset)
        )

    def _compute_gripper_dcmotor_tau(self, q_target_sim: float):
        """
        Reproduce the IsaacLab gripper actuator used in HL training.

        Training-space gripper coordinate:
            close = 0
            open  = -pi/2

        Pipeline:
            real q/qd
              -> real-to-training q offset
              -> IdealPD
              -> exact IsaacLab DCMotor four-quadrant clipping
              -> tau_f sent directly to the real gripper

        IMPORTANT:
            No extra post-DCMotor deployment torque clipping is applied here.
            With the hierarchical retrieval YAML, this is the same actuator
            law used in sim2sim/training.
        """
        q_target_sim = float(q_target_sim)

        # Real -> training coordinate. A constant position offset does not
        # change velocity, so qd_sim == qd_real.
        q_real = float(self.gripper_q)
        qd_real = float(self.gripper_qd)
        q_sim = self.get_gripper_q_training()
        qd_sim = qd_real

        # IdealPD front-end from the training actuator. qd_target = 0.
        tau_computed = (
            self.gripper_dcmotor_kp * (q_target_sim - q_sim)
            - self.gripper_dcmotor_kd * qd_sim
        )

        # Exact IsaacLab DCMotor torque-speed clipping.
        effort_limit = self.gripper_dcmotor_effort_limit
        saturation_effort = self.gripper_dcmotor_saturation_effort
        velocity_limit = self.gripper_dcmotor_velocity_limit

        vel_at_effort_lim = velocity_limit * (
            1.0 + effort_limit / saturation_effort
        )
        joint_vel = float(
            np.clip(qd_sim, -vel_at_effort_lim, +vel_at_effort_lim)
        )

        top = saturation_effort * (1.0 - joint_vel / velocity_limit)
        bottom = saturation_effort * (-1.0 - joint_vel / velocity_limit)

        max_effort = min(float(top), effort_limit)
        min_effort = max(float(bottom), -effort_limit)

        tau_dcmotor = float(
            np.clip(tau_computed, min_effort, max_effort)
        )

        # Exact sim2sim/training actuator semantics:
        # the DCMotor output is the torque sent to the gripper. There is no
        # deployment-only post-DCMotor torque cap.
        tau_send = tau_dcmotor

        return tau_send, tau_computed, tau_dcmotor, q_sim, qd_sim

    def _compute_arm_external_tau(self, q_target: np.ndarray):
        """
        Reproduce the IsaacLab arm actuator used in HL training.

        Training/sim2sim law for the 6 arm joints:

            tau = Kp * (q_target - q) - Kd * qd
            tau = clip(tau, -effort_limit, +effort_limit)

        This is an IdealPD front-end plus a symmetric effort clip.  Unlike
        jointGripper, the arm has NO DCMotor torque-speed envelope in training,
        so none is applied here.

        Returns:
            tau_send   : torque actually sent through tau_f, 6-D
            tau_pd     : unclipped IdealPD torque, 6-D
        """
        q_target = np.asarray(q_target, dtype=np.float32).reshape(6,)

        tau_pd = (
            self.arm_dcmotor_kp * (q_target - self.q)
            - self.arm_dcmotor_kd * self.qd
        ).astype(np.float32)

        tau_send = np.clip(
            tau_pd,
            -self.arm_dcmotor_effort_limit,
            +self.arm_dcmotor_effort_limit,
        ).astype(np.float32)

        return tau_send, tau_pd

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

        # If the previous runtime path used external gripper torque, force one
        # gain refresh so the legacy gripper position-PD path is restored.
        if self._gripper_external_dcmotor_active:
            self._last_applied_kp = None
            self._last_applied_kd = None
            self._gripper_external_dcmotor_active = False

        self.set_control_gain(kp_cmd, kd_cmd)

        self.arm.q = q_cmd
        self.arm.qd = qd_cmd
        self.arm.tau = tau_cmd
        self.arm.gripperQ = float(gripper_q_cmd)
        self.arm.gripperQd = 0.0
        self.arm.gripperTau = 0.0

        self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
        self.arm.setGripperCmd(
            float(self.arm.gripperQ),
            float(self.arm.gripperQd),
            float(self.arm.gripperTau),
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
        q_target_limited = q_target.copy()

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
                "[Z1-PD] "
                f"q_target={np.round(q_target, 3)} | "
                f"q_tgt_lim={np.round(q_target_limited, 3)} | "
                f"q_meas={np.round(self.q, 3)} | "
                f"qd_meas={np.round(self.qd, 3)} | "
                f"kp={np.round(kp, 3)} | "
                f"kd={np.round(kd, 3)} | "
                f"fsm={self._fsm_state}"
            )

    def track_target_pd_gripper_dcmotor_once(
        self,
        q_target: np.ndarray,
        gripper_q_target_sim: float,
        use_startup_gains: bool = False,
    ):
        """
        Runtime arm PD + external gripper DCMotor.

        Arm behavior is unchanged from track_target_pd_once().

        gripper_q_target_sim is in the TRAINING coordinate:
            close = 0.0
            open  = -pi/2

        The Z1 internal gripper kp/kd are forced to zero and the externally
        reproduced IsaacLab IdealPD + DCMotor torque is sent through tau_f.
        """
        q_target = np.asarray(q_target, dtype=np.float32).reshape(6,)
        q_target_limited = q_target.copy()

        # Startup/hold paths always use the firmware position loop, regardless
        # of the runtime arm mode.
        arm_external_torque = (
            self.arm_runtime_mode == "dcmotor"
            and not use_startup_gains
        )

        if arm_external_torque:
            # Zero firmware arm gains: the joint is driven purely by tau_f.
            kp = np.zeros(6, dtype=np.float32)
            kd = np.zeros(6, dtype=np.float32)
        else:
            kp = self.arm_kps_startup if use_startup_gains else self.arm_kps_runtime
            kd = self.arm_kds_startup if use_startup_gains else self.arm_kds_runtime

        qd_cmd = np.zeros(6, dtype=np.float32)
        tau_cmd = np.zeros(6, dtype=np.float32)

        with self._comm_lock:
            q_cmd, qd_cmd = self._protect_joint_cmd(q_target_limited, qd_cmd)

            if arm_external_torque:
                # Training-exact IdealPD + effort clip, sent through tau_f.
                tau_cmd, arm_tau_pd = self._compute_arm_external_tau(q_target)

                # With zero arm gains the q/qd fields are dynamically
                # irrelevant, so command the latest measured q as the safest
                # benign value (same rationale as the gripper below).
                q_cmd = self.q.copy()
                qd_cmd = np.zeros(6, dtype=np.float32)
            else:
                arm_tau_pd = tau_cmd

            self._arm_external_torque_active = bool(arm_external_torque)

            # set_control_gain() writes arm gains and the legacy gripper gains.
            # Therefore zero ONLY the gripper gains immediately afterwards.
            self.set_control_gain(kp, kd)

            if not hasattr(self.lowcmd, "setGripperZeroGain"):
                raise RuntimeError(
                    "Z1 lowcmd binding does not expose setGripperZeroGain()."
                )
            self.lowcmd.setGripperZeroGain()
            self._gripper_external_dcmotor_active = True

            (
                gripper_tau_send,
                gripper_tau_pd,
                gripper_tau_dcmotor,
                gripper_q_sim,
                gripper_qd_sim,
            ) = self._compute_gripper_dcmotor_tau(gripper_q_target_sim)

            # Arm: firmware position PD, or zero-gain external tau_f depending
            # on arm_runtime_mode (resolved above).
            self.arm.q = q_cmd
            self.arm.qd = qd_cmd
            self.arm.tau = tau_cmd
            self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)

            # Gripper: zero internal PD, direct external torque through tau_f.
            # q field is dynamically irrelevant with zero gripper gains, so use
            # the latest measured raw q as the safest benign command value.
            self.arm.gripperQ = float(self.gripper_q)
            self.arm.gripperQd = 0.0
            self.arm.gripperTau = float(gripper_tau_send)
            self.arm.setGripperCmd(
                float(self.arm.gripperQ),
                float(self.arm.gripperQd),
                float(self.arm.gripperTau),
            )

            self.arm.sendRecv()

            with self._state_lock:
                self._read_state_from_sdk_once()
                self.prev_q_cmd = q_cmd.copy()
                self.prev_gripper_q_cmd = float(gripper_q_target_sim)

        self._debug_counter += 1
        if self.debug_print and (self._debug_counter % 100 == 0):
            arm_mode_tag = (
                "ARM-DCMOTOR"
                if arm_external_torque
                else "ARM-POSPD"
            )
            print(
                f"[Z1-{arm_mode_tag}+GRIPPER-DCMOTOR] "
                f"q_target={np.round(q_target, 3)} | "
                f"q_meas={np.round(self.q, 3)} | "
                f"arm_tau_pd={np.round(arm_tau_pd, 2)} | "
                f"arm_tau_send={np.round(tau_cmd, 2)} | "
                f"grip_target_sim={gripper_q_target_sim:+.3f} | "
                f"grip_q_real={self.gripper_q:+.3f} | "
                f"grip_q_sim={gripper_q_sim:+.3f} | "
                f"grip_qd={gripper_qd_sim:+.3f} | "
                f"tau_pd={gripper_tau_pd:+.3f} | "
                f"tau_motor={gripper_tau_dcmotor:+.3f} | "
                f"tau_send={gripper_tau_send:+.3f} | "
                f"fsm={self._fsm_state}"
            )

    def track_target_pd_runtime_once(
        self,
        q_target: np.ndarray,
        gripper_q_target_training: float,
        use_startup_gains: bool = False,
    ):
        """
        Unified 50-Hz runtime arm + gripper command.

        q_target:
            Z1 arm target, 6-D.

        gripper_q_target_training:
            Gripper target in TRAINING coordinates:
                closed = 0
                open   = -pi/2

        The GRIPPER is always driven by the exact external IdealPD + IsaacLab
        DCMotor tau_f law (internal gripper gains zeroed).

        The ARM follows z1_arm_runtime_mode:

            "position_pd":
                firmware position loop with arm_kps_runtime / arm_kds_runtime.

            "dcmotor":
                zero firmware arm gains, training-exact
                    tau = Kp*(q_target - q) - Kd*qd
                clipped to the per-joint effort limits and sent as tau_f.

        Both are handled inside track_target_pd_gripper_dcmotor_once().
        """
        return self.track_target_pd_gripper_dcmotor_once(
            q_target=q_target,
            gripper_q_target_sim=float(
                gripper_q_target_training
            ),
            use_startup_gains=use_startup_gains,
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


def compute_ee_current_kp_plb(
    base_quat_wxyz: np.ndarray,
    base_height: float,
    ground_z: float,
    z1_adapter: Z1ArmAdapter,
    kp_dx: float,
    kp_dz: float,
) -> np.ndarray:
    """
    Match sim2sim PLB exactly:

        PLB origin      = [base_x, base_y, ground_z]
        PLB orientation = yaw-only(base_quat)

    In sim2real, base_x/base_y cancel because EE is computed relative to base.
    But base_height must be accurate.
    """
    base_quat_wxyz = quat_unique_wxyz(quat_normalize_wxyz(base_quat_wxyz))

    ee_pos_b, ee_rot_b = z1_adapter.compute_policy_ee_pose_in_base()
    ee_quat_b = quat_from_rotmat_wxyz(ee_rot_b)
    ee_quat_b = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_b))

    _, _, yaw = euler_xyz_from_quat_wxyz(base_quat_wxyz)
    plb_quat_w = quat_from_yaw_wxyz(yaw)
    plb_quat_w = quat_unique_wxyz(quat_normalize_wxyz(plb_quat_w))

    base_pos_w = np.array([0.0, 0.0, float(base_height)], dtype=np.float32)
    plb_pos_w = np.array([0.0, 0.0, float(ground_z)], dtype=np.float32)

    ee_pos_w = base_pos_w + quat_apply_wxyz(base_quat_wxyz, ee_pos_b)
    ee_quat_w = quat_mul_wxyz(base_quat_wxyz, ee_quat_b)
    ee_quat_w = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_w))

    ee_pos_plb = quat_apply_inverse_wxyz(plb_quat_w, ee_pos_w - plb_pos_w)
    ee_quat_plb = quat_mul_wxyz(quat_conjugate_wxyz(plb_quat_w), ee_quat_w)
    ee_quat_plb = quat_unique_wxyz(quat_normalize_wxyz(ee_quat_plb))

    off_x = np.array([kp_dx, 0.0, 0.0], dtype=np.float32)
    off_z = np.array([0.0, 0.0, kp_dz], dtype=np.float32)

    kp0 = ee_pos_plb
    kp1 = kp0 + quat_apply_wxyz(ee_quat_plb, off_x)
    kp2 = kp0 + quat_apply_wxyz(ee_quat_plb, off_z)

    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)