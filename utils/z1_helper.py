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
    Build a rotation matrix from roll-pitch-yaw.

    Convention:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    This is the standard fixed-axis XYZ-roll-pitch-yaw composition used
    for configuration-driven extrinsics.
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

    This helper is intentionally kept numerically aligned with the MuJoCo
    sim2sim helper so that EE command generation remains consistent between
    simulation and deployment.
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
    Thin helper around the modified Unitree Z1 Python binding.

    Main design goals:
    1. Keep arm FK / state reading in the same helper as before.
    2. Change the command semantics so they are much closer to the training
       control law used in sim2sim:
            - external target q
            - external kp / kd
            - qd_cmd usually zero
            - tau_ff usually zero
    3. Avoid the old deployment behavior:
            - qd_cmd from finite difference of target q
            - tau_cmd from inverseDynamics(...)
       because that behaves more like trajectory tracking than the training-time
       PD controller.
    4. Support different gain sets for:
            - startup / hold / move-to-default
            - runtime policy control
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
        # base_link -> arm_base
        self.arm_base_pos_in_base = np.array(cfg["arm_base_offset_pos"], dtype=np.float32).reshape(3,)
        arm_base_rpy = np.array(cfg["arm_base_offset_rpy"], dtype=np.float32).reshape(3,)
        self.arm_base_rot_in_base = rotmat_from_rpy_xyz(*arm_base_rpy)

        # SDK_EE -> policy_EE (for example gripperStator)
        self.sdk_ee_to_policy_pos = np.array(cfg["z1_fk_to_policy_ee_pos"], dtype=np.float32).reshape(3,)
        sdk_ee_to_policy_rpy = np.array(cfg["z1_fk_to_policy_ee_rpy"], dtype=np.float32).reshape(3,)
        self.sdk_ee_to_policy_rot = rotmat_from_rpy_xyz(*sdk_ee_to_policy_rpy)

        # Default commanded pose
        self.default_arm_pos = np.array(cfg["default_arm_pos"], dtype=np.float32).reshape(6,)
        self.default_gripper_pos = float(cfg["default_gripper_pos"])

        # Explicit external gains for startup / runtime
        self.arm_kps_startup = np.array(
            cfg.get("arm_kps_startup", [60.0, 80.0, 60.0, 40.0, 30.0, 20.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_kds_startup = np.array(
            cfg.get("arm_kds_startup", [4.0, 5.0, 4.0, 3.0, 2.0, 2.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_kps_runtime = np.array(
            cfg.get("arm_kps_runtime", [40.0, 40.0, 40.0, 40.0, 40.0, 40.0]),
            dtype=np.float32,
        ).reshape(6,)
        self.arm_kds_runtime = np.array(
            cfg.get("arm_kds_runtime", [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
            dtype=np.float32,
        ).reshape(6,)

        self.debug_print = bool(cfg.get("z1_debug_print", False))
        self._debug_counter = 0

        # ------------------------------------------------------------------
        # Command mode options
        #
        # arm_qd_mode:
        #   "zero"        -> qd_cmd = 0
        #   "finite_diff" -> qd_cmd = (q_cmd - prev_q_cmd) / dt
        #
        # arm_tau_mode:
        #   "zero"        -> tau_ff = 0
        #   "gravity"     -> inverseDynamics(q, 0, 0, 0)
        #   "full_id"     -> inverseDynamics(q, qd_cmd, 0, 0)
        #
        # For training alignment, the recommended default is:
        #   arm_qd_mode  = "zero"
        #   arm_tau_mode = "zero"
        # ------------------------------------------------------------------
        self.arm_qd_mode = str(cfg.get("arm_qd_mode", "zero"))
        self.arm_tau_mode = str(cfg.get("arm_tau_mode", "zero"))

        # Runtime state cache
        self.arm = None
        self.arm_model = None
        self.lowcmd = None

        self.q = np.zeros(6, dtype=np.float32)
        self.qd = np.zeros(6, dtype=np.float32)
        self.tau = np.zeros(6, dtype=np.float32)
        self.gripper_q = 0.0

        self.prev_q_cmd = self.default_arm_pos.copy()

        # Track the last applied gains so we only push them when changed.
        self._last_applied_kp = None
        self._last_applied_kd = None

        print(f"[Z1ArmAdapter] z1_sdk_lib = {z1_sdk_lib}")

    # Connection / state
    def connect(self):
        """
        Create the Python arm object and switch to LOWCMD FSM.

        This helper assumes the lowcmd-capable Python binding has already been
        rebuilt so that:
            arm._ctrlComp.lowcmd
        is accessible from Python.
        """
        self.arm = self.unitree_arm_interface.ArmInterface(self.has_gripper)
        self.arm_model = self.arm._ctrlComp.armModel
        self.lowcmd = self.arm._ctrlComp.lowcmd

        # Binding / interface checks should happen BEFORE any lowcmd-dependent call.
        if self.arm_model is None:
            raise RuntimeError("Z1 armModel is not accessible from Python binding.")
        if self.lowcmd is None:
            raise RuntimeError("Z1 lowcmd is not accessible from Python binding.")
        if not hasattr(self.lowcmd, "setControlGain"):
            raise RuntimeError("Z1 lowcmd binding does not expose setControlGain().")

        # Enter LOWCMD mode following the official low-level workflow.
        self.arm.setFsmLowcmd()

        # Pull a few packets so the lowstate becomes fresh and the communication
        # path is warm before the first real control command is sent.
        for _ in range(20):
            self.arm.sendRecv()
            time.sleep(0.002)

        self.read_state()
        self.prev_q_cmd = self.q.copy()

        # Apply startup gains immediately so any subsequent hold/move command
        # uses explicit external gains rather than whatever stale values may
        # already be inside lowcmd.
        self.set_control_gain(self.arm_kps_startup, self.arm_kds_startup)

        print("[Z1ArmAdapter] Connected.")
        print(f"[Z1ArmAdapter] FSM = {self.arm.getCurrentState()}")
    
    def get_arm_dt(self) -> float:
        """
        Return the actual low-level control dt used by the Z1 SDK.
        """
        if self.arm is None:
            return self.arm_control_dt
        try:
            return float(self.arm._ctrlComp.dt)
        except Exception:
            return self.arm_control_dt

    def read_state(self):
        """
        Update cached arm lowstate.

        Important:
        - This performs one sendRecv() call.
        - The main controller should decide how often this is called.
        """
        self.arm.sendRecv()

        self.q = np.asarray(self.arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
        self.qd = np.asarray(self.arm.lowstate.getQd(), dtype=np.float32).reshape(6,)
        self.tau = np.asarray(self.arm.lowstate.getTau(), dtype=np.float32).reshape(6,)

        gripper_q_raw = np.asarray(self.arm.lowstate.getGripperQ(), dtype=np.float32).reshape(-1)
        if gripper_q_raw.size > 0:
            self.gripper_q = float(gripper_q_raw[0])
        else:
            self.gripper_q = 0.0

    # Low-level gain management
    def set_control_gain(self, kp: np.ndarray, kd: np.ndarray):
        """
        Push explicit external gains into lowcmd.

        Important:
        - lowcmd.setControlGain() must NOT be spammed every tick at the binding layer.
        - We therefore cache the last applied gains and only write when they change.
        - The lowcmd gain vector is 7-dim in practice:
            6 arm joints + 1 gripper channel.
        """
        kp = np.asarray(kp, dtype=np.float32).reshape(6,)
        kd = np.asarray(kd, dtype=np.float32).reshape(6,)

        # Avoid unnecessary repeated gain writes.
        if (
            self._last_applied_kp is not None
            and self._last_applied_kd is not None
            and np.allclose(kp, self._last_applied_kp)
            and np.allclose(kd, self._last_applied_kd)
        ):
            return

        # 7-dim lowcmd gains: 6 joints + 1 gripper channel.
        # These gripper defaults match the behavior you observed in testing.
        kp_full = [float(x) for x in kp] + [self.gripper_kp]
        kd_full = [float(x) for x in kd] + [self.gripper_kd]

        self.lowcmd.setControlGain(kp_full, kd_full)

        self._last_applied_kp = kp.copy()
        self._last_applied_kd = kd.copy()

    def _get_gain_pair(self, use_startup_gains: bool) -> Tuple[np.ndarray, np.ndarray]:
        if use_startup_gains:
            return self.arm_kps_startup, self.arm_kds_startup
        return self.arm_kps_runtime, self.arm_kds_runtime

    # Command shaping
    def _protect_joint_cmd(self, q_cmd: np.ndarray, qd_cmd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SDK joint protection if available.

        The binding returns a pair:
            (q_safe, qd_safe)
        """
        try:
            q_safe, qd_safe = self.arm_model.jointProtect(q_cmd.copy(), qd_cmd.copy())
            return (
                np.asarray(q_safe, dtype=np.float32).reshape(6,),
                np.asarray(qd_safe, dtype=np.float32).reshape(6,),
            )
        except Exception:
            return q_cmd, qd_cmd

    def _compute_qd_cmd(self, q_cmd: np.ndarray) -> np.ndarray:
        """
        Compute desired joint velocity according to the configured mode.

        Recommended training-aligned mode:
            qd_cmd = 0
        """
        if self.arm_qd_mode == "zero":
            return np.zeros(6, dtype=np.float32)

        if self.arm_qd_mode == "finite_diff":
            return ((q_cmd - self.prev_q_cmd) / self.get_arm_dt()).astype(np.float32)

        raise ValueError(f"Unsupported arm_qd_mode: {self.arm_qd_mode}")

    def _compute_tau_ff(self, q_cmd: np.ndarray, qd_cmd: np.ndarray) -> np.ndarray:
        """
        Compute the feed-forward torque according to the configured mode.

        Recommended training-aligned mode:
            tau_ff = 0
        """
        if self.arm_tau_mode == "zero":
            return np.zeros(6, dtype=np.float32)

        qdd_cmd = np.zeros(6, dtype=np.float32)
        ftip = np.zeros(6, dtype=np.float32)

        if self.arm_tau_mode == "gravity":
            return np.asarray(
                self.arm_model.inverseDynamics(
                    q_cmd,
                    np.zeros(6, dtype=np.float32),
                    qdd_cmd,
                    ftip,
                ),
                dtype=np.float32,
            ).reshape(6,)

        if self.arm_tau_mode == "full_id":
            return np.asarray(
                self.arm_model.inverseDynamics(q_cmd, qd_cmd, qdd_cmd, ftip),
                dtype=np.float32,
            ).reshape(6,)

        raise ValueError(f"Unsupported arm_tau_mode: {self.arm_tau_mode}")

    # Main lowcmd send
    def send_arm_command(self, q_cmd: np.ndarray, gripper_q_cmd: float, use_startup_gains: bool = False):
        """
        Send one lowcmd step to Z1 using explicit external gains.

        Control semantics:
        - q_cmd      : desired joint positions
        - qd_cmd     : usually zero for training alignment
        - tau_ff     : usually zero for training alignment
        - kp / kd    : explicitly pushed through lowcmd.setControlGain(),
                    but only when changed
        """
        q_cmd = np.asarray(q_cmd, dtype=np.float32).reshape(6,)
        qd_cmd = self._compute_qd_cmd(q_cmd)

        # Apply SDK protection before sending.
        q_cmd, qd_cmd = self._protect_joint_cmd(q_cmd, qd_cmd)

        # Compute feed-forward torque.
        tau_ff = self._compute_tau_ff(q_cmd, qd_cmd)

        # Gains are configuration-like: only pushed when changed.
        kp_cmd, kd_cmd = self._get_gain_pair(use_startup_gains=use_startup_gains)
        self.set_control_gain(kp_cmd, kd_cmd)

        # Fill the ArmInterface command fields.
        self.arm.q = q_cmd
        self.arm.qd = qd_cmd
        self.arm.tau = tau_ff
        self.arm.gripperQ = float(gripper_q_cmd)

        # Send one low-level step.
        self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
        self.arm.setGripperCmd(
            float(gripper_q_cmd),
            self.arm.gripperQd,
            self.arm.gripperTau,
        )
        self.arm.sendRecv()

        self.prev_q_cmd = q_cmd.copy()

        self._debug_counter += 1
        if self.debug_print and (self._debug_counter % 100 == 0):
            print(
                "[Z1ArmAdapter] Sending arm command:",
                "q_cmd =", q_cmd,
                "qd_cmd =", qd_cmd,
                "tau_ff =", tau_ff,
                "gripper_q_cmd =", gripper_q_cmd,
                "kp =", kp_cmd,
                "kd =", kd_cmd,
            )
    def hold_target_once(
        self,
        q_target: np.ndarray,
        gripper_q_target: float,
        use_startup_gains: bool = False,
    ):
        """
        Send one low-level command that holds the given arm target.
        """
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
        """
        Hold a fixed arm target for a given duration using the Z1 low-level rate.

        The target itself is not updated inside this function. This is intended
        for the case where a higher-level controller updates targets at a slower
        rate (e.g. 50 Hz), while the arm lowcmd is sent at a faster rate
        (e.g. 500 Hz).
        """
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
        """
        Smoothly track from q_start to q_target at the arm low-level rate.

        This replaces zero-order hold of a fixed target inside one policy interval
        with a linearly interpolated reference trajectory.
        """
        q_start = np.asarray(q_start, dtype=np.float32).reshape(6,)
        q_target = np.asarray(q_target, dtype=np.float32).reshape(6,)

        dt = self.get_arm_dt()
        num_steps = max(1, int(round(duration_s / dt)))

        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)

            # Linear interpolation from previous command to current target
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
        """
        Move smoothly to a target joint pose using linear interpolation in
        joint space, with explicit external startup gains.

        This is only for startup / transitions. Runtime policy control should
        call send_arm_command() directly once per tick.
        """
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
    
    def move_to_default_like_min_test(
        self,
        duration_s: float,
        kp: np.ndarray,
        kd: np.ndarray,
        step_callback=None,
    ):
        """
        Move the arm to default pose using the exact low-level pattern that was
        validated by the standalone minimum test script.

        Optional:
            step_callback:
                A function called once per arm control step, typically used to keep
                the leg robot holding a default pose while the arm is moving.
        """
        self.read_state()
        q0 = self.q.copy()
        target_q = self.default_arm_pos.copy()
        target_gripper = float(self.default_gripper_pos)

        dt = float(self.arm._ctrlComp.dt)
        num_steps = max(1, int(round(duration_s / dt)))

        kp = np.asarray(kp, dtype=np.float32).reshape(6,)
        kd = np.asarray(kd, dtype=np.float32).reshape(6,)

        kp_full = [float(x) for x in kp] + [self.gripper_kp]
        kd_full = [float(x) for x in kd] + [self.gripper_kd]

        print("[Z1ArmAdapter] move_to_default_like_min_test")
        print("[Z1ArmAdapter] q0       =", np.round(q0, 4))
        print("[Z1ArmAdapter] target_q =", np.round(target_q, 4))
        print("[Z1ArmAdapter] dt       =", dt)
        print("[Z1ArmAdapter] steps    =", num_steps)
        print("[Z1ArmAdapter] kp_full  =", kp_full)
        print("[Z1ArmAdapter] kd_full  =", kd_full)

        # Re-enter LOWCMD before motion.
        self.arm.setFsmLowcmd()
        time.sleep(0.02)

        # Warm up communication.
        for _ in range(10):
            self.arm.sendRecv()
            time.sleep(dt)

        # Apply gains once before the motion loop.
        self.arm._ctrlComp.lowcmd.setControlGain(kp_full, kd_full)

        for step in range(num_steps):
            alpha = float(step) / float(num_steps)

            q_cmd = q0 * (1.0 - alpha) + target_q * alpha
            qd_cmd = np.zeros(6, dtype=np.float32)
            tau_cmd = np.zeros(6, dtype=np.float32)

            self.arm.q = q_cmd
            self.arm.qd = qd_cmd
            self.arm.tau = tau_cmd
            self.arm.gripperQ = target_gripper

            self.arm.setArmCmd(self.arm.q, self.arm.qd, self.arm.tau)
            self.arm.setGripperCmd(self.arm.gripperQ, self.arm.gripperQd, self.arm.gripperTau)
            self.arm.sendRecv()

            # Keep the leg robot holding default pose if requested.
            if step_callback is not None:
                step_callback()

            fsm = self.arm.getCurrentState()

            if (step % 20 == 0) or (step == num_steps - 1) or (fsm != self.unitree_arm_interface.ArmFSMState.LOWCMD):
                q_meas = np.array(self.arm.lowstate.getQ(), dtype=np.float32).reshape(6,)
                qd_meas = np.array(self.arm.lowstate.getQd(), dtype=np.float32).reshape(6,)
                tau_meas = np.array(self.arm.lowstate.getTau(), dtype=np.float32).reshape(6,)
                err = q_cmd - q_meas

                print(
                    f"[Z1-MINLIKE {step+1:04d}/{num_steps}] "
                    f"FSM={fsm} | "
                    f"q_cmd={np.round(q_cmd, 3)} | "
                    f"q_meas={np.round(q_meas, 3)} | "
                    f"err={np.round(err, 3)} | "
                    f"qd_meas={np.round(qd_meas, 3)} | "
                    f"tau_meas={np.round(tau_meas, 3)}"
                )

            if fsm != self.unitree_arm_interface.ArmFSMState.LOWCMD:
                print(f"[Z1ArmAdapter][ERROR] FSM dropped to {fsm} at step {step+1}")
                break

            time.sleep(dt)

        self.read_state()
        self.prev_q_cmd = self.q.copy()

        print("[Z1ArmAdapter] final q =", np.round(self.q, 4))

    def hold_default_step(self):
        """
        Send one hold step at the default arm pose using startup gains.
        """
        self.send_arm_command(
            q_cmd=self.default_arm_pos,
            gripper_q_cmd=self.default_gripper_pos,
            use_startup_gains=True,
        )

    def safe_back_to_start(self):
        """
        Optional SDK-provided recovery motion.

        Intentionally left as a no-op in this deployment helper, so shutdown
        does not trigger an extra uncontrolled arm motion.
        """
        pass

    # FK / policy EE conversion
    def compute_policy_ee_pose_in_base(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the policy EE pose in B2W base_link frame.

        Steps:
        1) SDK FK:
              forwardKinematics(q, ee_index)
           gives the SDK EE pose in arm_base frame.
        2) Apply the fixed transform:
              SDK_EE -> policy_EE
        3) Apply the fixed transform:
              arm_base -> base_link
        """
        # Important:
        # do not call read_state() here.
        # State reading is owned by the main deployment controller so the timing
        # remains consistent and easy to reason about.

        T_sdk = np.asarray(
            self.arm_model.forwardKinematics(self.q, self.ee_index),
            dtype=np.float32,
        ).reshape(4, 4)

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

    Because the LB frame shares the same origin as base_link, we do not need a
    global base position here. We only need the orientation change from body
    frame to level-base frame.
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