#!/usr/bin/env python3
"""
B2WZ1 dynamic nonprehensile TRANSPORT -> frozen WBC -> real robot, OptiTrack.

An unsecured puck rides on a disk rigidly mounted on the Z1 gripperStator. The
high-level policy outputs a bounded 6-D residual on an EE pose reference so
the disk keeps the puck while the base follows an operator velocity command.

Control stack (loop shape of deploy/b2wz1_wbc_uan.py, as retrieval)
---------------------------------------------------------------------
    HL transport policy   10 Hz   6-D residual [dx,dy,dz,dyaw,dpitch,droll]
      -> frozen WBC       50 Hz   leg / arm / wheel joint targets
      -> hardware        500 Hz   zero-order hold on B2W + Z1 lowcmd

UAN is simulation-only (see b2wz1_hl_retrieval_abs_uan_mocap.py): the real arm
runs the Z1 firmware PD the UAN was fitted against; the gains are audited.

Source of truth
---------------
Every high-level quantity reproduces
    unitree_mujoco/simulate_python/deploy_mujoco/b2wz1_hl_transport_uan.py
(and its bit-exact headless twin transport_experiment/sim.py):

  * actor frame (64-D), 3 frames FEATURE-MAJOR = 192-D:
        ang_vel_b(3) proj_gravity_b(3) leg_q_rel(12) arm_q_rel(6)
        leg_qd(12) arm_qd(6) wheel_qd(4) base_vel_ref(3) ee_kp0_ref(3)
        ee_rot_ref[yaw,pitch,roll](3) executed_residual(6) puck_center_D(3)
  * residual decode:
        kp0_cmd = clip(kp0_ref + a[0:3] * 0.10 m,  kp0 ranges)
        rot_cmd = clip(rot_ref + a[3:6] * 45 deg,  yaw/pitch/roll ranges)
        executed = clip([(kp0_cmd-kp0_ref)/0.10, (rot_cmd-rot_ref)/45deg], -1, 1)
        base_command = base reference (no learned base residual)
    During the 0.5-s settle the residual is masked and kp0 is held at the
    measured EE; after it the EE kp0 is measured again and becomes HIGH.
  * EE pose -> 9-D WBC keypoints with the training adapter
    build_keypoints_from_kp0_yaw_pitch_roll_plb (yaw/pitch define local +X,
    roll about +X). NOT the retrieval ABS keypoint builder.
  * reference process: obs_t(ref_t) -> a_t -> execute -> contact check ->
    advance refs (min-jerk 1.5 s / LOW 2.0 s) -> obs_{t+1}.

Real-world substitutions (everything else is identical)
--------------------------------------------------------
  * Task references come from the OPERATOR instead of the training sampler:
      base reference = sticks (b2wz1_wbc_uan.py mapping), clipped to the
                       training range, read once per HL step;
      EE target      = D-pad up/down/left/right = HIGH/LOW/LEFT/RIGHT, with
                       the training minimum-jerk transitions and at least the
                       training minimum hold (4 s) between switches.
  * puck_center_D, the only non-proprioceptive actor input, is measured:
        puck_b   = OptiTrack puck centre in the mocap base_link frame
        disk     = Z1 FK gripperStator pose (base frame) + training mount
        puck_D   = R_disk^T (puck_b - disk_centre_b)
    No disk rigid body: the disk frame comes from the same FK the WBC uses.
  * Puck/disk contact (training: net contact force > 0.05 N) has no sensor.
    It is replaced by a geometric proxy derived from the training geometry
    and the actor's training noise: in contact iff the puck centre is over
    the disk (rho_D <= disk_radius) and at resting height
    (|z_D - 0.042| <= 0.01 m). Training's 5-step debounce is kept; a drop
    ends the run in damping protection.

The actor observes no puck velocity, so no velocity estimate is computed.

Faults -> damping protection (retrieval semantics)
---------------------------------------------------
SELECT, Z1 PASSIVE, invalid/stale/untracked mocap (base or puck), puck drop,
non-finite observation/action, WBC arm target outside the SDK range, any
exception. Protection: B2W kp=0/high kd, Z1 holds q with high kd; A then folds
the legs. The policy never resumes.

Run from the repository root:

    python3 deploy/b2wz1_hl_transport_uan_mocap.py \
        enxa0cec819e15f \
        deploy/configs/b2wz1_hl_transport_uan_mocap.yaml
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import yaml

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_DEPLOY_DIR, ".."))
for _p in (PROJECT_ROOT, _DEPLOY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.math import quat_rotate_inverse_numpy, quat_unique_wxyz  # noqa: E402
from utils.mocap_perception import MocapBodySelector, MocapPerceptionSystem  # noqa: E402
from utils.remote_controller import KeyMap, RemoteController  # noqa: E402


# Z1 firmware gains the UAN was fitted against, and the training-space PD
# they must scale to (identical to the retrieval sim2real audit).
Z1_KP_FIRMWARE = np.array([2.5, 4.5, 2.5, 2.5, 2.5, 2.5], dtype=np.float64)
Z1_KD_FIRMWARE = np.array(
    [234.375, 312.5, 234.375, 234.375, 234.375, 234.375], dtype=np.float64
)
Z1_KP_TRAINING = np.array([64.0, 115.2, 64.0, 64.0, 64.0, 64.0], dtype=np.float64)
Z1_KD_TRAINING = np.array([3.0, 4.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float64)

# Low-level (WBC) frame, 80-D (sim2sim ll_feature_dims).
LL_FEATURE_DIMS = (3, 3, 3, 9, 12, 6, 12, 6, 4, 22)
# High-level transport frame, 64-D (sim2sim hl_feature_dims).
HL_FEATURE_DIMS = (3, 3, 12, 6, 12, 6, 4, 3, 3, 3, 6, 3)

# Canonical EE targets, in the sim2sim index order.
EE_TARGET_NAMES = ("HIGH", "LOW", "LEFT", "RIGHT")
EE_TARGET_HIGH, EE_TARGET_LOW = 0, 1
# D-pad -> EE target index.
EE_TARGET_BUTTONS = (
    (KeyMap.up, 0),
    (KeyMap.down, 1),
    (KeyMap.left, 2),
    (KeyMap.right, 3),
)

# Keys the real deployment must never carry (simulation-only actuator model).
SIM_ONLY_KEYS = (
    "uan_model_path", "uan_history_length", "uan_action_scale",
    "uan_nominal_torque_limits", "uan_final_torque_limits",
    "enable_arm_target_rate_limit", "arm_target_rate_limit",
)


# ======================================================================
# Pure helpers (no hardware; imported by the offline tests)
# ======================================================================


def sleep_until(deadline: float) -> float:
    """Sleep to `deadline`. Returns how late we already were, 0.0 if on time."""
    remaining = deadline - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)
        return 0.0
    return -remaining


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_keypoints_from_kp0_yaw_pitch_roll_plb(
    kp0: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    kp_dx: float = 0.30,
    kp_dz: float = 0.30,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """VERBATIM copy of the sim2sim training-adapter keypoint builder.

      kp1 = kp0 + kp_dx * local +X     (yaw/pitch define +X directly)
      kp2 = kp0 + kp_dz * local +Z     (most-upward direction orthogonal to
                                        +X, then rotated by roll about +X)
    """
    kp0 = np.asarray(kp0, dtype=np.float32).reshape(3,)
    yaw = float(yaw)
    pitch = float(pitch)
    roll = float(roll)
    cp = math.cos(pitch)
    vx = np.array(
        [cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)],
        dtype=np.float32,
    )
    vx /= max(float(np.linalg.norm(vx)), eps)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    vz0 = up - float(np.dot(up, vx)) * vx
    vz0_norm = float(np.linalg.norm(vz0))
    if vz0_norm > eps:
        vz0 /= vz0_norm
    else:
        vz0 = np.array([-math.sin(yaw), math.cos(yaw), 0.0], dtype=np.float32)
        vz0 /= max(float(np.linalg.norm(vz0)), eps)
    vy0 = np.cross(vz0, vx).astype(np.float32)
    vy0 /= max(float(np.linalg.norm(vy0)), eps)
    vz = math.cos(roll) * vz0 - math.sin(roll) * vy0
    vz = vz.astype(np.float32)
    vz /= max(float(np.linalg.norm(vz)), eps)
    kp1 = kp0 + float(kp_dx) * vx
    kp2 = kp0 + float(kp_dz) * vz
    return np.concatenate([kp0, kp1, kp2]).astype(np.float32)


def minimum_jerk_blend(tau: float) -> float:
    """Fifth-order smoothstep with zero velocity/acceleration at both ends."""
    tau = float(np.clip(tau, 0.0, 1.0))
    return tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)


class FeatureHistory:
    """Feature-major observation history (identical to the retrieval class).

        [feature_0(t-H+1 .. t), feature_1(t-H+1 .. t), ...]
    """

    def __init__(self, feature_dims: Sequence[int], length: int) -> None:
        self.dims = [int(d) for d in feature_dims]
        self.length = int(length)
        self.frame_dim = sum(self.dims)
        self.total_dim = self.frame_dim * self.length
        self._features = [deque(maxlen=self.length) for _ in self.dims]

    def _split(self, frame: np.ndarray) -> List[np.ndarray]:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if frame.shape[0] != self.frame_dim:
            raise ValueError(
                f"Observation frame is {frame.shape[0]}-D, expected {self.frame_dim}."
            )
        out, start = [], 0
        for dim in self.dims:
            out.append(frame[start:start + dim].copy())
            start += dim
        return out

    def reset(self, frame: np.ndarray) -> None:
        """Fill the whole history with one frame, as training does on reset."""
        for feature_history, feature in zip(self._features, self._split(frame)):
            feature_history.clear()
            for _ in range(self.length):
                feature_history.append(feature.copy())

    def append(self, frame: np.ndarray) -> None:
        for feature_history, feature in zip(self._features, self._split(frame)):
            feature_history.append(feature)

    def flat(self) -> np.ndarray:
        obs = np.concatenate(
            [np.asarray(f, dtype=np.float32).reshape(-1) for f in self._features],
            dtype=np.float32,
        )
        if obs.shape != (self.total_dim,):
            raise RuntimeError(f"History flattened to {obs.shape}, expected {self.total_dim}.")
        return obs


def build_transport_hl_obs_frame(
    *,
    base_ang_vel_b: np.ndarray,
    projected_gravity_b: np.ndarray,
    leg_pos_rel: np.ndarray,
    arm_pos_rel: np.ndarray,
    leg_vel: np.ndarray,
    arm_vel: np.ndarray,
    wheel_vel: np.ndarray,
    base_velocity_ref: np.ndarray,
    ee_kp0_ref: np.ndarray,
    ee_rot_ref: np.ndarray,
    executed_action: np.ndarray,
    puck_pos_disk: np.ndarray,
) -> np.ndarray:
    """The 64-D actor frame, in sim2sim build_hl_obs_frame order."""
    parts = (
        (base_ang_vel_b, 3), (projected_gravity_b, 3), (leg_pos_rel, 12),
        (arm_pos_rel, 6), (leg_vel, 12), (arm_vel, 6), (wheel_vel, 4),
        (base_velocity_ref, 3), (ee_kp0_ref, 3), (ee_rot_ref, 3),
        (executed_action, 6), (puck_pos_disk, 3),
    )
    obs = np.concatenate(
        [np.asarray(v, dtype=np.float32).reshape(n) for v, n in parts], dtype=np.float32
    )
    if not np.isfinite(obs).all():
        raise FloatingPointError("Non-finite value in the high-level observation.")
    return obs


def build_wbc_obs_frame(
    *,
    base_ang_vel_b: np.ndarray,
    projected_gravity_b: np.ndarray,
    base_command: np.ndarray,
    ee_cmd_plb: np.ndarray,
    leg_pos_rel: np.ndarray,
    arm_pos_rel: np.ndarray,
    leg_vel: np.ndarray,
    arm_vel: np.ndarray,
    wheel_vel: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    """The 80-D WBC frame (sim2sim build_ll_obs_frame order)."""
    parts = (
        (base_ang_vel_b, 3), (projected_gravity_b, 3), (base_command, 3),
        (ee_cmd_plb, 9), (leg_pos_rel, 12), (arm_pos_rel, 6), (leg_vel, 12),
        (arm_vel, 6), (wheel_vel, 4), (last_action, 22),
    )
    obs = np.concatenate(
        [np.asarray(v, dtype=np.float32).reshape(n) for v, n in parts], dtype=np.float32
    )
    if not np.isfinite(obs).all():
        raise FloatingPointError("Non-finite value in the WBC observation.")
    return obs


class DiskGeometry:
    """Disk frame rigidly attached to the gripperStator (training mount)."""

    def __init__(
        self,
        center_offset: Sequence[float],
        x_axis: Sequence[float],
        y_axis: Sequence[float],
        z_axis: Sequence[float],
        thickness: float,
        radius: float,
    ) -> None:
        self.center_offset = np.asarray(center_offset, dtype=np.float32).reshape(3)
        self.axes = np.stack(
            [np.asarray(a, dtype=np.float32).reshape(3) for a in (x_axis, y_axis, z_axis)],
            axis=1,
        )
        if not np.allclose(self.axes.T @ self.axes, np.eye(3), atol=1e-6) or not (
            float(np.linalg.det(self.axes)) > 0.0
        ):
            raise ValueError("disk_{x,y,z}_axis_gripper must form a right-handed orthonormal frame.")
        self.thickness = float(thickness)
        self.radius = float(radius)
        if not (self.thickness > 0.0 and self.radius > 0.0):
            raise ValueError("disk_thickness and disk_radius must be positive.")

    def frame_in(self, stator_pos: np.ndarray, stator_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(disk centre, R_frame_disk) in whatever frame the stator pose is in."""
        stator_pos = np.asarray(stator_pos, dtype=np.float32).reshape(3)
        stator_rot = np.asarray(stator_rot, dtype=np.float32).reshape(3, 3)
        center = stator_pos + stator_rot @ self.center_offset
        rot = stator_rot @ self.axes
        return center.astype(np.float32), rot.astype(np.float32)

    def puck_pos_disk(
        self, puck_pos: np.ndarray, stator_pos: np.ndarray, stator_rot: np.ndarray
    ) -> np.ndarray:
        """sim2sim get_puck_pos_disk: [rel.x_D, rel.y_D, rel.z_D].

        puck_pos and the stator pose must be expressed in the SAME frame (the
        base frame on the robot, the world frame in MuJoCo); the projection is
        frame-invariant.
        """
        center, rot = self.frame_in(stator_pos, stator_rot)
        rel = np.asarray(puck_pos, dtype=np.float32).reshape(3) - center
        return np.array(
            [np.dot(rel, rot[:, 0]), np.dot(rel, rot[:, 1]), np.dot(rel, rot[:, 2])],
            dtype=np.float32,
        )


class PuckContactMonitor:
    """Geometric stand-in for the training puck/disk contact-force detector.

    Training: contact = net disk/puck force > 0.05 N; puck_drop after
    `drop_steps` consecutive ACTIVE HL steps without contact; the settle
    interval clears the counter. A puck lying on the disk has its centre over
    the disk at the resting height, so the proxy is

        contact  <=>  rho_D <= disk_radius  and  |z_D - rest| <= z_tolerance

    with z_tolerance = the actor's training observation noise amplitude.
    """

    def __init__(
        self, disk_radius: float, rest_height: float, z_tolerance: float, drop_steps: int
    ) -> None:
        self.disk_radius = float(disk_radius)
        self.rest_height = float(rest_height)
        self.z_tolerance = float(z_tolerance)
        self.drop_steps = int(drop_steps)
        if self.drop_steps <= 0 or self.z_tolerance <= 0.0:
            raise ValueError("drop_steps and z_tolerance must be positive.")
        self.no_contact_steps = 0

    def reset(self) -> None:
        self.no_contact_steps = 0

    def in_contact(self, puck_d: Optional[np.ndarray]) -> bool:
        if puck_d is None:
            return False
        p = np.asarray(puck_d, dtype=np.float64).reshape(3)
        if not np.isfinite(p).all():
            return False
        rho = float(np.hypot(p[0], p[1]))
        return rho <= self.disk_radius and abs(float(p[2]) - self.rest_height) <= self.z_tolerance

    def update(self, puck_d: Optional[np.ndarray], settling: bool) -> Tuple[bool, int, bool]:
        """One completed HL transition -> (contact, no_contact_steps, drop)."""
        contact = self.in_contact(puck_d)
        if settling or contact:
            self.no_contact_steps = 0
        else:
            self.no_contact_steps += 1
        return contact, self.no_contact_steps, self.no_contact_steps >= self.drop_steps


def puck_start_gate(
    puck_d: Optional[np.ndarray],
    spawn_xy: float,
    spawn_clearance: float,
    obs_noise: float,
    rest_height: float,
) -> Tuple[bool, str]:
    """Is the puck where training spawns it (within the actor's noise)?"""
    if puck_d is None:
        return False, "puck_D unavailable"
    p = np.asarray(puck_d, dtype=np.float64).reshape(3)
    if not np.isfinite(p).all():
        return False, "puck_D non-finite"
    xy_lim = float(spawn_xy) + float(obs_noise)
    z_nom = float(rest_height) + float(spawn_clearance)
    problems = []
    if abs(p[0]) > xy_lim or abs(p[1]) > xy_lim:
        problems.append(f"xy=({p[0]:+.4f},{p[1]:+.4f}) outside +/-{xy_lim:.3f}")
    if abs(p[2] - z_nom) > float(obs_noise):
        problems.append(f"z={p[2]:+.4f} not within {obs_noise:.3f} of {z_nom:.4f}")
    return (not problems), ("OK" if not problems else "; ".join(problems))


class TransportTask:
    """Reference process + residual decoder of the transport sim2sim.

    Pure state machine, no hardware. Arithmetic and ordering are those of
    b2wz1_hl_transport_uan.py; only the SOURCE of the references differs
    (operator instead of the training sampler).
    """

    def __init__(self, cfg: Dict[str, Any], hl_control_dt: float) -> None:
        f32 = lambda k: np.asarray(cfg[k], dtype=np.float32)  # noqa: E731
        self.hl_control_dt = float(hl_control_dt)
        self.kp_dx = float(cfg["ee_kp_dx"])
        self.kp_dz = float(cfg["ee_kp_dz"])
        self.pos_scale = f32("ee_pos_action_residual_scale").reshape(3)
        self.rot_scale = f32("ee_rot_action_residual_scale").reshape(3)
        self.pos_min = np.array(
            [cfg["kp0_x_range"][0], cfg["kp0_y_range"][0], cfg["kp0_z_range"][0]], dtype=np.float32
        )
        self.pos_max = np.array(
            [cfg["kp0_x_range"][1], cfg["kp0_y_range"][1], cfg["kp0_z_range"][1]], dtype=np.float32
        )
        self.rot_min = np.array(
            [cfg["ee_yaw_range"][0], cfg["ee_pitch_range"][0], cfg["ee_roll_range"][0]],
            dtype=np.float32,
        )
        self.rot_max = np.array(
            [cfg["ee_yaw_range"][1], cfg["ee_pitch_range"][1], cfg["ee_roll_range"][1]],
            dtype=np.float32,
        )
        self.canonical_positions = f32("canonical_ee_positions")
        if self.canonical_positions.shape != (4, 3):
            raise ValueError("canonical_ee_positions must have shape [4,3].")
        self.canonical_yaw = float(cfg["canonical_ee_yaw"])
        self.canonical_pitch = float(cfg["canonical_ee_pitch"])
        self.canonical_roll = float(cfg["canonical_ee_roll"])
        self.base_ref_ranges = f32("base_ref_ranges")
        self.base_cmd_scale = f32("base_cmd_scale").reshape(3)
        if self.base_ref_ranges.shape != (3, 2):
            raise ValueError("base_ref_ranges must have shape [3,2].")
        if np.any(self.base_ref_ranges[:, 0] < -self.base_cmd_scale) or np.any(
            self.base_ref_ranges[:, 1] > self.base_cmd_scale
        ):
            raise ValueError("base_ref_ranges must stay inside frozen-WBC base bounds.")

        dt = self.hl_control_dt
        self.settle_steps = int(round(float(cfg["reset_settle_time_s"]) / dt))
        self.fast_steps = int(round(float(cfg["ee_fast_transition_time_s"]) / dt))
        self.low_steps = int(round(float(cfg["ee_low_transition_time_s"]) / dt))
        self.min_hold_steps = int(round(float(cfg["ee_ref_resample_time_range_s"][0]) / dt))
        if min(self.settle_steps, self.fast_steps, self.low_steps, self.min_hold_steps) <= 0:
            raise ValueError("Settle / transition / hold times must be positive.")
        if max(self.fast_steps, self.low_steps) > self.min_hold_steps:
            raise ValueError("EE transitions must not exceed the minimum EE hold period.")
        self.reset(self.canonical_positions[0])

    # ------------------------------------------------------------------
    @property
    def settling(self) -> bool:
        return self.settle_steps_remaining > 0

    @property
    def rot_ref(self) -> np.ndarray:
        return np.array([self.ee_yaw_ref, self.ee_pitch_ref, self.ee_roll_ref], dtype=np.float32)

    def _keypoints(self, kp0: np.ndarray, rot: np.ndarray) -> np.ndarray:
        return build_keypoints_from_kp0_yaw_pitch_roll_plb(
            kp0, float(rot[0]), float(rot[1]), float(rot[2]), self.kp_dx, self.kp_dz
        )

    def _sync_commands_from_references(self) -> None:
        """Zero-residual path (sim2sim sync_ll_commands_from_references)."""
        self.base_command = self.base_velocity_ref.copy()
        self.kp0_cmd = self.ee_kp0_ref.copy()
        self.rot_cmd = self.rot_ref.copy()
        self.ee_cmd_plb = self._keypoints(self.kp0_cmd, self.rot_cmd)

    def _set_high_from_measured(self, measured_kp0: np.ndarray) -> None:
        measured = np.asarray(measured_kp0, dtype=np.float32).reshape(3).copy()
        if not np.isfinite(measured).all():
            raise FloatingPointError(f"Non-finite measured EE kp0: {measured}")
        self.episode_high_kp0 = measured.copy()
        self.ee_kp0_ref = measured.copy()
        self.current_target = EE_TARGET_HIGH
        self.transition_start = measured.copy()
        self.transition_target = measured.copy()
        self.transition_step = 0
        self.transition_total = 1
        self._sync_commands_from_references()

    def reset(self, measured_kp0: np.ndarray) -> None:
        """sim2sim reset_command_references + set_episode_high_from_measured_ee."""
        self.base_velocity_ref = np.zeros(3, dtype=np.float32)
        self.ee_yaw_ref = self.canonical_yaw
        self.ee_pitch_ref = self.canonical_pitch
        self.ee_roll_ref = self.canonical_roll
        self.settle_steps_remaining = self.settle_steps
        self.steps_since_switch = 0
        self.pending_target: Optional[int] = None
        self.last_event = "reset"
        self.raw_action = np.zeros(6, dtype=np.float32)
        self.effective_action = np.zeros(6, dtype=np.float32)
        self.executed_action = np.zeros(6, dtype=np.float32)
        self._set_high_from_measured(measured_kp0)

    def request_target(self, index: int) -> None:
        index = int(index)
        if not 0 <= index < len(EE_TARGET_NAMES):
            raise ValueError(f"EE target index {index} out of range.")
        self.pending_target = index

    def clip_base_reference(self, base_ref: np.ndarray) -> np.ndarray:
        v = np.asarray(base_ref, dtype=np.float32).reshape(3)
        if not np.isfinite(v).all():
            raise FloatingPointError(f"Non-finite operator base reference: {v}")
        return np.clip(v, self.base_ref_ranges[:, 0], self.base_ref_ranges[:, 1]).astype(np.float32)

    def _start_transition(self, index: int) -> None:
        """sim2sim sample_ee_pose_reference for an operator-chosen target."""
        self.current_target = int(index)
        target = (
            self.episode_high_kp0.copy()
            if index == EE_TARGET_HIGH
            else self.canonical_positions[index].copy()
        )
        self.transition_start = self.ee_kp0_ref.copy()
        self.transition_target = target.astype(np.float32)
        self.transition_step = 0
        self.transition_total = self.low_steps if index == EE_TARGET_LOW else self.fast_steps
        self.ee_kp0_ref = self.transition_start.copy()
        self.ee_yaw_ref = self.canonical_yaw
        self.ee_pitch_ref = self.canonical_pitch
        self.ee_roll_ref = self.canonical_roll
        self.steps_since_switch = 0

    def _advance_interpolation(self) -> None:
        total = max(int(self.transition_total), 1)
        self.transition_step = min(int(self.transition_step) + 1, total)
        blend = minimum_jerk_blend(float(self.transition_step) / float(total))
        self.ee_kp0_ref = (
            self.transition_start + blend * (self.transition_target - self.transition_start)
        ).astype(np.float32)

    def advance(
        self, measure_kp0: Callable[[], np.ndarray], operator_base_ref: np.ndarray
    ) -> str:
        """sim2sim advance_command_references_after_transition.

        Called after a completed 10-Hz transition and before the next actor
        observation. Returns a short event string for logging.
        """
        if self.settling:
            self.settle_steps_remaining -= 1
            if self.settle_steps_remaining > 0:
                self.last_event = "settle"
                return self.last_event
            # Settle complete: HIGH = live settled EE kp0, then the first
            # base reference and the first EE target (pending or HIGH).
            self._set_high_from_measured(measure_kp0())
            self.base_velocity_ref = self.clip_base_reference(operator_base_ref)
            target = EE_TARGET_HIGH if self.pending_target is None else self.pending_target
            self.pending_target = None
            self._start_transition(target)
            self.last_event = f"settled -> {EE_TARGET_NAMES[target]}"
            return self.last_event

        self.steps_since_switch += 1
        self._advance_interpolation()
        self.base_velocity_ref = self.clip_base_reference(operator_base_ref)
        self.last_event = ""
        if self.pending_target is not None:
            target = self.pending_target
            self.pending_target = None
            if self.steps_since_switch >= self.min_hold_steps:
                self._start_transition(target)
                self.last_event = f"target -> {EE_TARGET_NAMES[target]}"
            else:
                wait_s = (self.min_hold_steps - self.steps_since_switch) * self.hl_control_dt
                self.last_event = (
                    f"target {EE_TARGET_NAMES[target]} REJECTED (training min hold; "
                    f"retry in {wait_s:.1f}s)"
                )
        return self.last_event

    def decode(self, action: np.ndarray) -> None:
        """sim2sim decode_hl_action, line for line."""
        action = np.asarray(action, dtype=np.float32).reshape(6,)
        if not np.isfinite(action).all():
            raise FloatingPointError(f"Non-finite high-level action: {action}")
        self.raw_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.effective_action = self.raw_action.copy()
        if self.settling:
            self.effective_action[:] = 0.0

        self.base_command = self.base_velocity_ref.copy()

        pos_raw = self.ee_kp0_ref + self.effective_action[0:3] * self.pos_scale
        kp0_cmd = np.clip(pos_raw, self.pos_min, self.pos_max).astype(np.float32)
        if self.settling:
            kp0_cmd = self.ee_kp0_ref.copy()

        rot_ref = self.rot_ref
        rot_raw = rot_ref + self.effective_action[3:6] * self.rot_scale
        rot_cmd = np.clip(rot_raw, self.rot_min, self.rot_max).astype(np.float32)

        self.executed_action = np.clip(
            np.concatenate([
                (kp0_cmd - self.ee_kp0_ref) / self.pos_scale,
                (rot_cmd - rot_ref) / self.rot_scale,
            ]),
            -1.0,
            1.0,
        ).astype(np.float32)
        self.kp0_cmd = kp0_cmd
        self.rot_cmd = rot_cmd
        self.ee_cmd_plb = self._keypoints(kp0_cmd, rot_cmd)


def validate_transport_setup(cfg: Dict[str, Any]) -> List[str]:
    """Every operator-supplied hardware fact that is still missing or invalid.

    An empty list is required before the controller will touch the robot.
    """
    problems: List[str] = []

    for key in SIM_ONLY_KEYS:
        if key in cfg:
            problems.append(f"{key}: simulation-only key must not be in the real config")

    if cfg.get("mocap_puck_name") in (None, "") and cfg.get("mocap_puck_id") is None:
        problems.append("mocap_puck_name / mocap_puck_id: puck rigid body not configured")
    if (
        cfg.get("mocap_puck_name") not in (None, "")
        and cfg.get("mocap_puck_name") == cfg.get("mocap_body_name")
    ) or (
        cfg.get("mocap_puck_id") is not None
        and cfg.get("mocap_puck_id") == cfg.get("mocap_body_id")
    ):
        problems.append("mocap_puck_*: puck must be a different rigid body from the robot body")

    offset = cfg.get("mocap_puck_offset_local")
    if offset is None:
        problems.append("mocap_puck_offset_local: puck pivot->centre offset not configured")
    else:
        try:
            off = np.asarray(offset, dtype=np.float64).reshape(3)
            if not np.isfinite(off).all():
                raise ValueError
        except Exception:  # noqa: BLE001
            problems.append("mocap_puck_offset_local: must be 3 finite numbers [m]")

    mass = cfg.get("puck_mass_kg")
    lo, hi = (float(x) for x in cfg.get("puck_mass_range_kg", [0.5, 1.5]))
    if mass is None:
        problems.append("puck_mass_kg: real puck mass not configured")
    else:
        try:
            m = float(mass)
            if not (np.isfinite(m) and lo <= m <= hi):
                problems.append(f"puck_mass_kg={m}: outside the training range [{lo}, {hi}] kg")
        except (TypeError, ValueError):
            problems.append("puck_mass_kg: must be a number [kg]")

    if cfg.get("puck_geometry_verified") is not True:
        problems.append("puck_geometry_verified: real puck size not confirmed against training")
    if cfg.get("disk_mount_verified") is not True:
        problems.append("disk_mount_verified: physical disk mount not confirmed against training")

    has_offset_file = bool(cfg.get("mocap_root_offset_path")) or bool(cfg.get("mocap_root_offset"))
    if not has_offset_file and cfg.get("mocap_root_frame_is_base_link") is not True:
        problems.append("mocap root frame: neither aligned in Motive nor calibrated")

    if not (0.0 < float(cfg.get("mocap_puck_max_age_s", 0.0)) <= 0.1 + 1e-9):
        problems.append("mocap_puck_max_age_s: must be in (0, 0.1] s (one HL period)")

    scale = np.asarray(cfg.get("command_scale", [np.inf] * 3), dtype=np.float64).reshape(3)
    ranges = np.asarray(cfg["base_ref_ranges"], dtype=np.float64).reshape(3, 2)
    if np.any(scale < 0.0) or np.any(scale > np.minimum(-ranges[:, 0], ranges[:, 1]) + 1e-9):
        problems.append("command_scale: must be within the training base_ref_ranges")
    return problems


# ======================================================================
# OptiTrack: robot body + puck only
# ======================================================================


class TransportMocapPerception(MocapPerceptionSystem):
    """MocapPerceptionSystem with exactly two assets: the B2 and the puck.

    Reuses the validated NatNet client, root-offset handling, staleness and
    tracking classification and base-frame transform of the parent; only the
    asset set differs (no retrieval target).

    Snapshot:
        body         validated base_link pose (mocap root)
        base_height  root z - ground_z
        puck         puck centre: position_base, position_world, age_ms
    """

    def __init__(self, body: MocapBodySelector, puck_body: MocapBodySelector, **kwargs: Any) -> None:
        super().__init__(body=body, object_body=puck_body, retrieval_body=None, **kwargs)
        self.puck_body = puck_body

    def _selectors(self) -> Tuple[MocapBodySelector, MocapBodySelector]:
        return (self.body, self.puck_body)

    def initialize(self) -> None:
        if self._initialized:
            return
        self.client.start()
        if not self.client.wait_for_first_frame(timeout_s=self.startup_timeout_s):
            raise RuntimeError(
                "No NatNet frames received. Check Motive streaming (Rigid Bodies "
                "on), the mocap NIC address, and that the server is reachable."
            )
        streamed = self.client.rigid_body_ids()
        names = self.client.rigid_body_names()
        self._log("streaming rigid bodies: " + (
            ", ".join(f"{i}:{names.get(i, '?')!r}" for i in streamed) or "(none)"
        ))
        missing = []
        for selector in self._selectors():
            rb_id = self._resolve(selector, names, streamed)
            self._resolved[selector.label] = rb_id
            if rb_id is None or rb_id not in streamed:
                missing.append(selector.label)
            self._log("  " + selector.describe() + f" -> id {rb_id}")
        if missing:
            raise RuntimeError(
                "These rigid bodies are not being streamed by Motive: "
                + ", ".join(missing) + f". Streamed ids are {streamed}."
            )
        if self._resolved[self.body.label] == self._resolved[self.puck_body.label]:
            raise RuntimeError("The robot body and the puck resolve to the SAME rigid body.")
        untracked = [s.label for s in self._selectors() if not self._is_tracked(self._resolved[s.label])]
        if untracked:
            self._log("[WARN] currently UNTRACKED: " + ", ".join(untracked))
        self._initialized = True

    def get_latest_snapshot(self, projected_gravity_b: Any = None) -> Dict[str, Any]:
        now = time.monotonic()
        frame = self.client.latest_frame()
        body_rb, body_reason, body_age_ms = self._sample(self.body, now)
        mocap_meta = {
            "frame_number": int(frame.frame_number) if frame is not None else None,
            "frames_received": self.client.frame_count,
            "parse_errors": self.client.parse_error_count,
        }
        if body_rb is None:
            reason = f"BODY_{body_reason}"
            puck_state = self._invalid_channel(reason, body_age_ms)
            puck_state.update(position_base=None, position_world=None)
            return {
                "valid": False,
                "body": {"valid": False, "reason": body_reason, "age_ms": body_age_ms,
                         "root_position_world": None, "root_quat_wxyz": None},
                "base_height": self._invalid_height(reason, body_age_ms),
                "puck": puck_state,
                "mocap": mocap_meta,
            }

        p_root_w, q_root_w = self.root_offset.apply(body_rb.pos, body_rb.quat_wxyz)
        base_height = float(p_root_w[2] - self.ground_z)
        height_valid = bool(np.isfinite(base_height))
        puck_state = self._target_channel(self.puck_body, "position_base", p_root_w, q_root_w, now)
        return {
            "valid": bool(height_valid and puck_state["valid"]),
            "body": {
                "valid": True, "reason": "OK", "age_ms": body_age_ms,
                "root_position_world": p_root_w.tolist(),
                "root_quat_wxyz": q_root_w.tolist(),
            },
            "base_height": {
                "valid": height_valid,
                "height_m": base_height if height_valid else None,
                "reason": "OK" if height_valid else "NON_FINITE_HEIGHT",
                "age_ms": body_age_ms,
            },
            "puck": puck_state,
            "mocap": mocap_meta,
        }


# ======================================================================
# Controller
# ======================================================================


class B2WZ1TransportMocapController:
    """Transport HL policy over the frozen transport WBC, on the real B2W+Z1.

    Hardware objects are injectable for offline tests: pass `z1` and
    `perception` fakes and `connect_dds=False`; nothing is then sent anywhere.
    """

    def __init__(
        self,
        cfg_path: str,
        *,
        z1: Any = None,
        perception: Any = None,
        connect_dds: bool = True,
    ) -> None:
        with open(cfg_path, "r") as f:
            self.cfg: Dict[str, Any] = yaml.safe_load(f)

        problems = validate_transport_setup(self.cfg)
        if problems:
            raise ValueError(
                "Transport hardware setup is incomplete -- refusing to start:\n  - "
                + "\n  - ".join(problems)
                + "\nSee deploy/TRANSPORT_SIM2REAL_SETUP.md."
            )

        self._configure_timing()
        self._configure_robot_layout()
        self._configure_task()
        self._configure_protection()
        self._load_policies()

        if z1 is None:
            from utils.z1_helper import Z1ArmAdapter  # noqa: PLC0415 - hardware only

            z1 = Z1ArmAdapter(self.cfg, PROJECT_ROOT)
        self.z1 = z1
        self._audit_z1_runtime_alignment()

        self.perception = perception if perception is not None else self._build_mocap_perception()
        if connect_dds:
            self._connect_dds()
        self._reset_runtime_state()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _resolve_path(self, path_str: str) -> str:
        path_str = str(path_str)
        if os.path.isabs(path_str):
            return path_str
        return os.path.abspath(os.path.join(PROJECT_ROOT, path_str))

    def _configure_timing(self) -> None:
        cfg = self.cfg
        self.control_dt = float(cfg["control_dt"])
        self.ll_steps_per_hl_step = int(cfg["ll_steps_per_hl_step"])
        self.hl_control_dt = self.control_dt * self.ll_steps_per_hl_step
        if abs(1.0 / self.control_dt - 50.0) > 1e-5:
            raise ValueError(f"WBC must run at 50 Hz; control_dt={self.control_dt}.")
        if abs(1.0 / self.hl_control_dt - 10.0) > 1e-5:
            raise ValueError(f"High level must run at 10 Hz; got {1.0 / self.hl_control_dt:.3f} Hz.")

        self.hardware_hz = float(cfg["hardware_command_hz"])
        self.hardware_dt = 1.0 / self.hardware_hz
        ratio = self.hardware_hz * self.control_dt
        if ratio < 1.0 - 1e-9 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(
                f"hardware_command_hz={self.hardware_hz:g} must be an integer multiple of 50 Hz."
            )
        self.hardware_ticks_per_wbc_step = int(round(ratio))
        self.runtime_settle_s = float(cfg.get("z1_runtime_settle_s", 0.5))

        self.ll_obs_dim = int(cfg["ll_obs_dim"])
        self.ll_action_dim = int(cfg["ll_action_dim"])
        self.hl_obs_dim = int(cfg["hl_obs_dim"])
        self.hl_action_dim = int(cfg["hl_action_dim"])
        self.ll_history = FeatureHistory(LL_FEATURE_DIMS, int(cfg["ll_history_length"]))
        self.hl_history = FeatureHistory(HL_FEATURE_DIMS, int(cfg["hl_history_length"]))
        for name, history, per_step, total in (
            ("Low", self.ll_history, cfg["ll_obs_dim_per_step"], self.ll_obs_dim),
            ("High", self.hl_history, cfg["hl_obs_dim_per_step"], self.hl_obs_dim),
        ):
            if history.frame_dim != int(per_step) or history.total_dim != int(total):
                raise ValueError(
                    f"{name}-level observation layout mismatch: build has "
                    f"{history.frame_dim}/{history.total_dim}, YAML {per_step}/{total}."
                )
        if self.ll_action_dim != 22 or self.hl_action_dim != 6:
            raise ValueError(
                f"Expected 22-D WBC / 6-D transport actions, got "
                f"{self.ll_action_dim}/{self.hl_action_dim}."
            )

    def _configure_robot_layout(self) -> None:
        """Policy joint order and its mapping onto hardware motor order."""
        cfg = self.cfg
        self.policy_leg_joint_names = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        self.policy_wheel_joint_names = [
            "FL_wheel_joint", "FR_wheel_joint", "RL_wheel_joint", "RR_wheel_joint",
        ]
        self.hardware_joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "FR_wheel_joint", "FL_wheel_joint", "RR_wheel_joint", "RL_wheel_joint",
        ]
        policy_joint_names = self.policy_leg_joint_names + self.policy_wheel_joint_names
        self.num_b2w_dof = 16
        self.hardware_to_policy = [self.hardware_joint_names.index(n) for n in policy_joint_names]
        self.policy_to_hardware = [policy_joint_names.index(n) for n in self.hardware_joint_names]
        self.leg_policy_indices = list(range(12))
        self.leg_hardware_indices = [
            self.hardware_joint_names.index(n) for n in self.policy_leg_joint_names
        ]
        self.wheel_hardware_indices = [
            self.hardware_joint_names.index(n) for n in self.policy_wheel_joint_names
        ]
        self.hardware_to_wheel_cmd = {
            self.hardware_joint_names.index(n): i
            for i, n in enumerate(self.policy_wheel_joint_names)
        }

        self.default_b2w_pos_policy = np.asarray(cfg["default_b2w_pos_policy"], dtype=np.float32).reshape(16)
        self.squat_b2w_pos_policy = np.asarray(cfg["squat_b2w_pos_policy"], dtype=np.float32).reshape(16)
        self.default_leg_pos_policy = self.default_b2w_pos_policy[:12].copy()
        self.default_arm_pos = np.asarray(cfg["default_arm_pos"], dtype=np.float32).reshape(6)
        self.default_joint_pos_policy = np.asarray(
            cfg["default_joint_pos_policy"], dtype=np.float32
        ).reshape(18)
        self.gripper_hold_pos = float(cfg["gripper_hold_pos"])

        def gains_hw(key: str) -> np.ndarray:
            return np.asarray(cfg[key], dtype=np.float32).reshape(16)[self.policy_to_hardware]

        self.kps_rl_hw = gains_hw("kps_rl")
        self.kds_rl_hw = gains_hw("kds_rl")
        self.kps_pd_hw = gains_hw("kps_pd")
        self.kds_pd_hw = gains_hw("kds_pd")

        self.leg_action_scale = float(cfg["leg_action_scale"])
        self.arm_action_scale = np.asarray(cfg["arm_action_scale"], dtype=np.float32).reshape(6)
        self.wheel_action_scale = float(cfg["wheel_action_scale"])
        self.arm_q_lower = np.asarray(cfg["arm_q_lower"], dtype=np.float32).reshape(6)
        self.arm_q_upper = np.asarray(cfg["arm_q_upper"], dtype=np.float32).reshape(6)

    def _configure_task(self) -> None:
        cfg = self.cfg
        self.task = TransportTask(cfg, self.hl_control_dt)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.ee_kp_dx = float(cfg["ee_kp_dx"])
        self.ee_kp_dz = float(cfg["ee_kp_dz"])

        self.disk = DiskGeometry(
            cfg["disk_center_offset_gripper"], cfg["disk_x_axis_gripper"],
            cfg["disk_y_axis_gripper"], cfg["disk_z_axis_gripper"],
            cfg["disk_thickness"], cfg["disk_radius"],
        )
        self.puck_radius = float(cfg["puck_radius"])
        self.puck_height = float(cfg["puck_height"])
        self.puck_rest_height = 0.5 * self.disk.thickness + 0.5 * self.puck_height
        self.puck_spawn_xy = float(cfg["puck_spawn_xy_randomization"])
        self.puck_spawn_clearance = float(cfg["puck_spawn_clearance"])
        self.puck_obs_noise = float(cfg["puck_obs_noise_m"])
        self.contact_monitor = PuckContactMonitor(
            disk_radius=self.disk.radius,
            rest_height=self.puck_rest_height,
            z_tolerance=self.puck_obs_noise,
            drop_steps=int(cfg["puck_drop_no_contact_steps"]),
        )

        self.command_scale = np.asarray(cfg["command_scale"], dtype=np.float32).reshape(3)
        self.command_deadband_lin = float(cfg["command_deadband_lin"])
        self.command_deadband_ang = float(cfg["command_deadband_ang"])

        self.debug_print_period_steps = int(cfg.get("debug_print_period_steps", 50))
        self.hl_log_enabled = bool(cfg.get("hl_log_enabled", False))
        self.hl_log_dir = self._resolve_path(cfg.get("hl_log_dir", "Log/transport"))

    def _configure_protection(self) -> None:
        cfg = self.cfg
        self.damping_kd_b2w = float(cfg.get("damping_kd_b2w", 150.0))
        self.damping_kd_wheel = float(cfg.get("damping_kd_wheel", 10.0))
        self.damping_kp_z1 = np.asarray(cfg.get("damping_kp_z1", [0.0] * 6), dtype=np.float32).reshape(6)
        self.damping_kd_z1 = np.asarray(cfg.get("damping_kd_z1", [2000.0] * 6), dtype=np.float32).reshape(6)
        self.damping_print_period = int(cfg.get("damping_print_period", 100))
        self.leg_recovery_enabled = bool(cfg.get("damping_leg_recovery_enabled", True))
        self.leg_recovery_duration_s = float(cfg.get("damping_recover_leg_duration_s", 4.0))
        self.leg_recovery_tolerance_rad = float(cfg.get("damping_recover_leg_tolerance_rad", 0.15))
        self.leg_recovery_target = np.asarray(
            cfg.get("damping_recover_leg_target", self.squat_b2w_pos_policy[:12].tolist()),
            dtype=np.float32,
        ).reshape(12)
        if self.leg_recovery_duration_s <= 0.0 or self.leg_recovery_tolerance_rad <= 0.0:
            raise ValueError("damping_recover_leg_duration_s and _tolerance_rad must be > 0.")
        self.perception_status_print_s = float(cfg.get("perception_status_print_s", 1.0))

    def _load_policies(self) -> None:
        cfg = self.cfg
        self.high_policy_path = self._resolve_path(cfg["high_policy_path"])
        self.low_policy_path = self._resolve_path(cfg["low_policy_path"])
        for label, path, key in (
            ("high", self.high_policy_path, "high_policy_sha256"),
            ("low", self.low_policy_path, "low_policy_sha256"),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{label} policy not found: {path}")
            expected = str(cfg.get(key) or "").strip().lower()
            actual = file_sha256(path)
            if not expected:
                raise ValueError(f"{key} must pin the {label} policy.")
            if actual != expected:
                raise ValueError(
                    f"{label} policy hash mismatch: {path}\n  sha256 {actual}\n  expected {expected}\n"
                    "This is not the checkpoint the transport policy was trained with."
                )

        self.high_session = ort.InferenceSession(self.high_policy_path, providers=["CPUExecutionProvider"])
        self.low_session = ort.InferenceSession(self.low_policy_path, providers=["CPUExecutionProvider"])
        self.high_input_name = self.high_session.get_inputs()[0].name
        self.high_output_name = self.high_session.get_outputs()[0].name
        self.low_input_name = self.low_session.get_inputs()[0].name
        self.low_output_name = self.low_session.get_outputs()[0].name
        for label, session, obs_dim, action_dim in (
            ("low", self.low_session, self.ll_obs_dim, self.ll_action_dim),
            ("high", self.high_session, self.hl_obs_dim, self.hl_action_dim),
        ):
            in_dim = session.get_inputs()[0].shape[-1]
            out_dim = session.get_outputs()[0].shape[-1]
            if isinstance(in_dim, int) and in_dim != obs_dim:
                raise RuntimeError(f"{label} policy takes {in_dim}-D obs, expected {obs_dim}.")
            if isinstance(out_dim, int) and out_dim != action_dim:
                raise RuntimeError(f"{label} policy emits {out_dim}-D actions, expected {action_dim}.")

    def _audit_z1_runtime_alignment(self) -> None:
        """Refuse to run unless the real arm reproduces the training actuator."""
        cfg = self.cfg
        if str(cfg.get("z1_arm_runtime_mode", "")).strip().lower() != "position_pd":
            raise ValueError('z1_arm_runtime_mode must be "position_pd" (UAN is simulation-only).')
        if str(cfg.get("z1_gripper_runtime_mode", "")).strip().lower() not in ("dcmotor", "position_pd"):
            raise ValueError("z1_gripper_runtime_mode must be 'dcmotor' or 'position_pd'.")

        kp_fw = np.asarray(cfg["arm_kps_runtime"], dtype=np.float64).reshape(6)
        kd_fw = np.asarray(cfg["arm_kds_runtime"], dtype=np.float64).reshape(6)
        kp_train = np.asarray(cfg["arm_kp_training"], dtype=np.float64).reshape(6)
        kd_train = np.asarray(cfg["arm_kd_training"], dtype=np.float64).reshape(6)
        kp_scale = float(cfg.get("z1_kp_protocol_scale", 25.6))
        kd_scale = float(cfg.get("z1_kd_protocol_scale", 0.0128))
        for label, actual, expected, tol in (
            ("firmware Kp", kp_fw, Z1_KP_FIRMWARE, 1e-9),
            ("firmware Kd", kd_fw, Z1_KD_FIRMWARE, 1e-9),
            ("training Kp", kp_train, Z1_KP_TRAINING, 1e-6),
            ("training Kd", kd_train, Z1_KD_TRAINING, 1e-6),
            ("scaled Kp", kp_fw * kp_scale, kp_train, 1e-6),
            ("scaled Kd", kd_fw * kd_scale, kd_train, 1e-6),
        ):
            if not np.allclose(actual, expected, atol=tol, rtol=0.0):
                raise ValueError(
                    f"Z1 gain alignment failed on {label}:\n  got      {np.round(actual, 6)}\n"
                    f"  expected {np.round(expected, 6)}"
                )
        for label, adapter_value, yaml_value in (
            ("Kp", self.z1.arm_kps_runtime, kp_fw),
            ("Kd", self.z1.arm_kds_runtime, kd_fw),
        ):
            if not np.allclose(np.asarray(adapter_value, dtype=np.float64).reshape(6), yaml_value,
                               atol=1e-9, rtol=0.0):
                raise RuntimeError(f"Z1 adapter runtime {label} {adapter_value} differs from the YAML.")
        self.z1_kp_effective = kp_fw * kp_scale
        self.z1_kd_effective = kd_fw * kd_scale

    def _build_mocap_perception(self) -> TransportMocapPerception:
        """OptiTrack stack with the base_link root-frame guarantee of retrieval."""
        cfg = self.cfg

        def selector(key: str, max_age_default: float) -> MocapBodySelector:
            return MocapBodySelector(
                label=key,
                name=cfg.get(f"mocap_{key}_name"),
                rb_id=cfg.get(f"mocap_{key}_id"),
                offset_local=cfg.get(f"mocap_{key}_offset_local", [0.0, 0.0, 0.0]),
                max_age_s=float(cfg.get(f"mocap_{key}_max_age_s", max_age_default)),
            )

        offset_path = cfg.get("mocap_root_offset_path")
        inline_offset = cfg.get("mocap_root_offset")
        motive_aligned = bool(cfg.get("mocap_root_frame_is_base_link", False))
        if offset_path:
            root_offset = self._resolve_path(offset_path)
            if not os.path.isfile(root_offset):
                raise FileNotFoundError(
                    f"mocap_root_offset_path does not exist: {root_offset}. Run "
                    "deploy/b2w_mocap_root_calibration.py."
                )
        elif inline_offset:
            root_offset = inline_offset
        elif motive_aligned:
            root_offset = None
        else:
            raise ValueError("The mocap root frame must be base_link (see the retrieval YAML).")

        perception = TransportMocapPerception(
            body=selector("body", 0.25),
            puck_body=selector("puck", 0.10),
            root_offset=root_offset,
            root_frame_is_base_link=motive_aligned,
            ground_z=self.ground_z,
            local_ip=str(cfg.get("mocap_local_ip", "")),
            server_ip=cfg.get("mocap_server_ip") or None,
            multicast_group=str(cfg.get("mocap_multicast_group", "239.255.42.99")),
            data_port=int(cfg.get("mocap_data_port", 1511)),
            command_port=int(cfg.get("mocap_command_port", 1510)),
            join_multicast=bool(cfg.get("mocap_join_multicast", True)),
            up_axis=str(cfg.get("mocap_up_axis", "auto")),
            startup_timeout_s=float(cfg.get("mocap_timeout_s", 10.0)),
        )
        if not bool(getattr(perception, "root_offset_calibrated", False)):
            raise ValueError("MocapPerceptionSystem reports an untrusted root offset.")
        return perception

    def _connect_dds(self) -> None:
        """B2W low-level DDS. ChannelFactoryInitialize() is done once in main()."""
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber  # noqa: PLC0415
        from unitree_sdk2py.idl.default import (  # noqa: PLC0415
            unitree_go_msg_dds__LowCmd_,
            unitree_go_msg_dds__LowState_,
        )
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo  # noqa: PLC0415
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo  # noqa: PLC0415
        from unitree_sdk2py.utils.crc import CRC  # noqa: PLC0415

        self.remote_controller = RemoteController()
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.crc = CRC()
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmdGo)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowstate_subscriber.Init(self._low_state_handler, 10)

    def _reset_runtime_state(self) -> None:
        self.base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.base_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.projected_gravity_b = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.b2w_joint_pos = np.zeros(16, dtype=np.float32)
        self.b2w_joint_vel = np.zeros(16, dtype=np.float32)
        self.base_height: Optional[float] = None

        self.last_ll_action = np.zeros(self.ll_action_dim, dtype=np.float32)
        self.leg_target = self.default_leg_pos_policy.copy()
        self.arm_target = self.default_arm_pos.copy()
        self.wheel_cmd = np.zeros(4, dtype=np.float32)

        self.ll_tick = 0
        self.hl_tick = 0
        self.last_snapshot: Optional[Dict[str, Any]] = None
        self.last_puck_d: Optional[np.ndarray] = None
        self.last_contact: Tuple[bool, int, bool] = (True, 0, False)
        self._ee_buttons_prev = {button: 1 for button, _ in EE_TARGET_BUTTONS}
        self._t_start = time.monotonic()
        self._log_file = None
        self._log_writer = None

    # ------------------------------------------------------------------
    # Sensing
    # ------------------------------------------------------------------

    def _low_state_handler(self, msg: Any) -> None:
        self.low_state = msg
        self.remote_controller.set(msg.wireless_remote)

    def wait_for_low_state(self) -> None:
        print("[TRANSPORT] Waiting for the first B2W lowstate...")
        while getattr(self.low_state, "tick", 0) == 0:
            time.sleep(self.control_dt)
        print(f"[TRANSPORT] First lowstate received: tick={self.low_state.tick}")

    def read_b2w_state(self) -> None:
        raw_quat = self.low_state.imu_state.quaternion
        quat = quat_unique_wxyz(np.asarray(raw_quat, dtype=np.float32).reshape(4))
        self.base_quat_wxyz[:] = quat / max(float(np.linalg.norm(quat)), 1e-8)
        self.base_ang_vel_b[:] = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32).reshape(3)
        self.projected_gravity_b[:] = quat_rotate_inverse_numpy(
            self.base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        motors = self.low_state.motor_state
        for policy_idx in range(self.num_b2w_dof):
            motor = motors[self.hardware_to_policy[policy_idx]]
            self.b2w_joint_pos[policy_idx] = motor.q
            self.b2w_joint_vel[policy_idx] = motor.dq

    def read_robot_state(self) -> None:
        self.read_b2w_state()
        self.z1.read_state()

    def read_perception(self) -> Dict[str, Any]:
        """Nonblocking mocap read; refreshes base height and puck_D.

        The puck is resolved against the FK disk frame of THIS step, so the
        two never mix data from different control steps.
        """
        snap = self.perception.get_latest_snapshot()
        self.last_snapshot = snap
        height = snap.get("base_height") or {}
        h = height.get("height_m") if height.get("valid") else None
        self.base_height = float(h) if h is not None and np.isfinite(h) else None

        puck = snap.get("puck") or {}
        self.last_puck_d = None
        if puck.get("valid") and puck.get("position_base") is not None:
            puck_b = np.asarray(puck["position_base"], dtype=np.float32).reshape(3)
            if np.isfinite(puck_b).all():
                stator_pos_b, stator_rot_b = self.z1.compute_policy_ee_pose_in_base()
                puck_d = self.disk.puck_pos_disk(puck_b, stator_pos_b, stator_rot_b)
                if np.isfinite(puck_d).all():
                    self.last_puck_d = puck_d
        return snap

    def perception_fault(self, snap: Dict[str, Any]) -> Optional[str]:
        """Why the actor may not use this snapshot, or None. Never a stand-in."""
        reasons = []
        if self.base_height is None:
            reasons.append("BASE_HEIGHT:" + str((snap.get("base_height") or {}).get("reason")))
        if self.last_puck_d is None:
            reasons.append("PUCK:" + str((snap.get("puck") or {}).get("reason")))
        return None if not reasons else "|".join(reasons)

    def measure_ee_kp0_plb(self) -> np.ndarray:
        """Live gripperStator kp0 in PLB (IMU roll/pitch/yaw, mocap height)."""
        from utils.z1_helper import compute_ee_current_kp_plb  # noqa: PLC0415

        if self.base_height is None:
            raise RuntimeError("Base height is unknown: no valid mocap body frame.")
        kps = compute_ee_current_kp_plb(
            base_quat_wxyz=self.base_quat_wxyz,
            base_height=float(self.base_height),
            ground_z=self.ground_z,
            z1_adapter=self.z1,
            kp_dx=self.ee_kp_dx,
            kp_dz=self.ee_kp_dz,
        )
        return np.asarray(kps[0:3], dtype=np.float32)

    # ------------------------------------------------------------------
    # Operator commands
    # ------------------------------------------------------------------

    def read_operator_base_reference(self) -> np.ndarray:
        """Sticks -> base reference, mapping/deadband of b2wz1_wbc_uan.py."""
        rc = self.remote_controller
        cmd = np.zeros(3, dtype=np.float32)
        cmd[0] = np.clip(rc.ly, -1.0, 1.0) * self.command_scale[0]
        cmd[1] = np.clip(-rc.lx, -1.0, 1.0) * self.command_scale[1]
        cmd[2] = np.clip(-rc.rx, -1.0, 1.0) * self.command_scale[2]
        lin_norm = float(np.linalg.norm(cmd[:2], ord=2))
        if lin_norm < self.command_deadband_lin * max(self.command_scale[0], self.command_scale[1]):
            cmd[0] = 0.0
            cmd[1] = 0.0
        if abs(cmd[2]) < self.command_deadband_ang * self.command_scale[2]:
            cmd[2] = 0.0
        return cmd

    def poll_operator_ee_request(self) -> None:
        """Latch a D-pad press (rising edge) as the next EE target request."""
        rc = self.remote_controller
        for button, index in EE_TARGET_BUTTONS:
            pressed = rc.button[button] == 1
            if pressed and not self._ee_buttons_prev[button]:
                self.task.request_target(index)
            self._ee_buttons_prev[button] = int(pressed)

    # ------------------------------------------------------------------
    # High level, 10 Hz
    # ------------------------------------------------------------------

    def build_hl_obs_frame(self) -> np.ndarray:
        if self.last_puck_d is None:
            raise RuntimeError("Refusing to build the actor observation without a valid puck.")
        return build_transport_hl_obs_frame(
            base_ang_vel_b=self.base_ang_vel_b,
            projected_gravity_b=self.projected_gravity_b,
            leg_pos_rel=self.b2w_joint_pos[:12] - self.default_joint_pos_policy[:12],
            arm_pos_rel=np.asarray(self.z1.q, dtype=np.float32) - self.default_joint_pos_policy[12:18],
            leg_vel=self.b2w_joint_vel[:12],
            arm_vel=np.asarray(self.z1.qd, dtype=np.float32),
            wheel_vel=self.b2w_joint_vel[12:16],
            base_velocity_ref=self.task.base_velocity_ref,
            ee_kp0_ref=self.task.ee_kp0_ref,
            ee_rot_ref=self.task.rot_ref,
            executed_action=self.task.executed_action,
            puck_pos_disk=self.last_puck_d,
        )

    def run_high_level_policy(self, *, append_frame: bool) -> None:
        if append_frame:
            self.hl_history.append(self.build_hl_obs_frame())
        obs = self.hl_history.flat()
        action = self.high_session.run(
            [self.high_output_name], {self.high_input_name: obs[None, :]}
        )[0][0].astype(np.float32)
        self.task.decode(action)
        self.hl_tick += 1

    # ------------------------------------------------------------------
    # Frozen WBC, 50 Hz
    # ------------------------------------------------------------------

    def build_ll_obs_frame(self) -> np.ndarray:
        return build_wbc_obs_frame(
            base_ang_vel_b=self.base_ang_vel_b,
            projected_gravity_b=self.projected_gravity_b,
            base_command=self.task.base_command,
            ee_cmd_plb=self.task.ee_cmd_plb,
            leg_pos_rel=self.b2w_joint_pos[:12] - self.default_joint_pos_policy[:12],
            arm_pos_rel=np.asarray(self.z1.q, dtype=np.float32) - self.default_joint_pos_policy[12:18],
            leg_vel=self.b2w_joint_vel[:12],
            arm_vel=np.asarray(self.z1.qd, dtype=np.float32),
            wheel_vel=self.b2w_joint_vel[12:16],
            last_action=self.last_ll_action,
        )

    def run_low_level_policy(self) -> None:
        self.ll_history.append(self.build_ll_obs_frame())
        obs = self.ll_history.flat()
        action = self.low_session.run(
            [self.low_output_name], {self.low_input_name: obs[None, :]}
        )[0][0].astype(np.float32)
        if action.shape != (self.ll_action_dim,) or not np.isfinite(action).all():
            raise FloatingPointError(f"Invalid WBC action: {action}")
        self.last_ll_action[:] = action
        self.leg_target[:] = (self.default_leg_pos_policy + self.leg_action_scale * action[0:12]).astype(np.float32)
        self.arm_target[:] = (self.default_arm_pos + self.arm_action_scale * action[12:18]).astype(np.float32)
        self.wheel_cmd[:] = (self.wheel_action_scale * action[18:22]).astype(np.float32)

    def check_arm_target_range(self) -> Optional[str]:
        below = self.arm_target < self.arm_q_lower
        above = self.arm_target > self.arm_q_upper
        if not (below.any() or above.any()):
            return None
        joint = int(np.argmax(below | above))
        return (
            f"WBC arm target outside the live SDK range at joint{joint + 1}: "
            f"q_des={np.round(self.arm_target, 4).tolist()}"
        )

    # ------------------------------------------------------------------
    # Policy state initialization and the 50-Hz step
    # ------------------------------------------------------------------

    def startup_readiness(self) -> Tuple[bool, str]:
        """Everything the first actor observation needs, plus the spawn gate."""
        snap = self.read_perception()
        fault = self.perception_fault(snap)
        if fault is not None:
            return False, fault
        ok, gate = puck_start_gate(
            self.last_puck_d, self.puck_spawn_xy, self.puck_spawn_clearance,
            self.puck_obs_noise, self.puck_rest_height,
        )
        d = self.last_puck_d
        status = (
            f"base_h={self.base_height:.4f}m | puck_D=({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f}) "
            f"| z_err={d[2] - self.puck_rest_height:+.4f}m | "
            f"puck_age={((snap.get('puck') or {}).get('age_ms') or 0.0):.1f}ms | gate={gate}"
        )
        return ok, status

    def initialize_policy_state_and_history(self) -> None:
        """sim2sim reset_episode: measured HIGH hold, zero action, filled histories."""
        self.read_robot_state()
        ready, status = self.startup_readiness()
        if not ready:
            raise RuntimeError(f"Not ready at policy initialization: {status}")

        self.task.reset(self.measure_ee_kp0_plb())
        self.contact_monitor.reset()
        self.last_ll_action[:] = 0.0
        self.leg_target[:] = self.default_leg_pos_policy
        self.arm_target[:] = self.default_arm_pos
        self.wheel_cmd[:] = 0.0

        self.ll_history.reset(self.build_ll_obs_frame())
        self.hl_history.reset(self.build_hl_obs_frame())
        self.ll_tick = 0
        self.hl_tick = 0
        self._ee_buttons_prev = {b: int(self.remote_controller.button[b] == 1) for b, _ in EE_TARGET_BUTTONS}
        print(
            "[TRANSPORT] Policy state initialized | HIGH(reset)="
            + np.array2string(self.task.episode_high_kp0, precision=4)
            + f" | {status}"
        )

    def wbc_step(self) -> Optional[str]:
        """One 50 Hz step. Returns a protection reason, or None to continue.

        HL boundary order = sim2sim main loop:
            contact check on the completed transition -> advance references
            -> actor obs + inference + decode -> WBC.
        """
        self.read_robot_state()
        if self.remote_controller.button[KeyMap.select] == 1:
            return "SELECT pressed"
        if self.z1.get_fsm_state() == self.z1.unitree_arm_interface.ArmFSMState.PASSIVE:
            return "Z1 entered PASSIVE state"

        snap = self.read_perception()
        fault = self.perception_fault(snap)
        if fault is not None:
            return "Mocap perception invalid: " + fault

        self.poll_operator_ee_request()

        if self.ll_tick % self.ll_steps_per_hl_step == 0:
            if self.hl_tick == 0:
                # History was just seeded with the reset frame: infer on it
                # directly (sim2sim infer_first_hl_and_ll_after_reset).
                self.run_high_level_policy(append_frame=False)
            else:
                contact, count, drop = self.contact_monitor.update(self.last_puck_d, self.task.settling)
                self.last_contact = (contact, count, drop)
                if drop:
                    return (
                        f"PUCK DROP: no disk contact for {count} HL steps "
                        f"(puck_D={np.round(self.last_puck_d, 4).tolist()})"
                    )
                event = self.task.advance(self.measure_ee_kp0_plb, self.read_operator_base_reference())
                if event:
                    print(f"[TRANSPORT] hl={self.hl_tick} {event}")
                self.run_high_level_policy(append_frame=True)
            self.write_hl_log_row()

        self.run_low_level_policy()
        out_of_range = self.check_arm_target_range()
        if out_of_range is not None:
            return out_of_range

        self.ll_tick += 1
        if self.debug_print_period_steps > 0 and self.ll_tick % self.debug_print_period_steps == 0:
            self.print_runtime_debug()
        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    HL_LOG_HEADER = (
        ["t_s", "hl_tick", "settling", "ee_target", "event"]
        + [f"base_ref_{a}" for a in ("vx", "vy", "wz")]
        + [f"kp0_ref_{a}" for a in "xyz"] + ["yaw_ref", "pitch_ref", "roll_ref"]
        + [f"raw_{i}" for i in range(6)] + [f"exec_{i}" for i in range(6)]
        + [f"kp0_cmd_{a}" for a in "xyz"] + ["yaw_cmd", "pitch_cmd", "roll_cmd"]
        + [f"puck_D_{a}" for a in "xyz"] + ["puck_rho", "contact", "no_contact_steps"]
        + ["puck_age_ms", "body_age_ms", "base_height"]
        + [f"kp0_meas_{a}" for a in "xyz"]
    )

    def open_hl_log(self) -> None:
        if not self.hl_log_enabled:
            return
        os.makedirs(self.hl_log_dir, exist_ok=True)
        path = os.path.join(self.hl_log_dir, time.strftime("transport_%Y%m%d_%H%M%S.csv"))
        self._log_file = open(path, "w", newline="")
        self._log_writer = csv.writer(self._log_file)
        self._log_writer.writerow(self.HL_LOG_HEADER)
        print(f"[TRANSPORT] HL log: {path}")

    def write_hl_log_row(self) -> None:
        if self._log_writer is None:
            return
        t = self.task
        snap = self.last_snapshot or {}
        d = self.last_puck_d if self.last_puck_d is not None else np.full(3, np.nan)
        try:
            kp0_meas = self.measure_ee_kp0_plb()
        except Exception:  # noqa: BLE001 - logging must never fault the run
            kp0_meas = np.full(3, np.nan)
        self._log_writer.writerow(
            [f"{time.monotonic() - self._t_start:.4f}", self.hl_tick, int(t.settling),
             EE_TARGET_NAMES[t.current_target], t.last_event]
            + t.base_velocity_ref.tolist() + t.ee_kp0_ref.tolist() + t.rot_ref.tolist()
            + t.raw_action.tolist() + t.executed_action.tolist()
            + t.kp0_cmd.tolist() + t.rot_cmd.tolist()
            + d.tolist() + [float(np.hypot(d[0], d[1])), int(self.last_contact[0]), self.last_contact[1]]
            + [(snap.get("puck") or {}).get("age_ms"), (snap.get("body") or {}).get("age_ms"),
               self.base_height]
            + kp0_meas.tolist()
        )
        if self.hl_tick % 10 == 0:
            self._log_file.flush()

    def close_hl_log(self) -> None:
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:  # noqa: BLE001
                pass
            self._log_file = None
            self._log_writer = None

    def print_runtime_debug(self) -> None:
        t = self.task
        d = self.last_puck_d if self.last_puck_d is not None else np.full(3, np.nan)
        print(
            f"[TRANSPORT] ll={self.ll_tick:06d} hl={self.hl_tick:05d} "
            f"settle={int(t.settling)} tgt={EE_TARGET_NAMES[t.current_target]} | "
            f"puck_D=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) rho={np.hypot(d[0], d[1]):.3f} "
            f"contact={int(self.last_contact[0])} nc={self.last_contact[1]} | "
            f"base_ref={np.round(t.base_velocity_ref, 2)} base_h={self.base_height:.3f}"
        )
        print(
            f"            raw={np.round(t.raw_action, 2)} exec={np.round(t.executed_action, 2)} "
            f"kp0_ref={np.round(t.ee_kp0_ref, 3)} kp0_cmd={np.round(t.kp0_cmd, 3)} "
            f"rot_cmd={np.round(t.rot_cmd, 3)}"
        )

    # ------------------------------------------------------------------
    # Actuation (retrieval sim2real, unchanged)
    # ------------------------------------------------------------------

    def send_b2w_cmd(self) -> None:
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)

    def _write_b2w_pose_cmd(self, target_b2w_pos_policy: np.ndarray, *, use_pd_gains: bool) -> None:
        target_hw = np.asarray(target_b2w_pos_policy, dtype=np.float32).reshape(16)[self.policy_to_hardware]
        kps_hw = self.kps_pd_hw if use_pd_gains else self.kps_rl_hw
        kds_hw = self.kds_pd_hw if use_pd_gains else self.kds_rl_hw
        for hw_idx in range(self.num_b2w_dof):
            motor = self.low_cmd.motor_cmd[hw_idx]
            is_leg = hw_idx in self.leg_hardware_indices
            motor.q = float(target_hw[hw_idx]) if is_leg else 0.0
            motor.dq = 0.0
            motor.kp = float(kps_hw[hw_idx])
            motor.kd = float(kds_hw[hw_idx])
            motor.tau = 0.0

    def _write_b2w_wbc_cmd(self) -> None:
        target_policy = self.default_b2w_pos_policy.copy()
        target_policy[self.leg_policy_indices] = self.leg_target
        target_hw = target_policy[self.policy_to_hardware]
        for hw_idx in range(self.num_b2w_dof):
            motor = self.low_cmd.motor_cmd[hw_idx]
            motor.kp = float(self.kps_rl_hw[hw_idx])
            motor.kd = float(self.kds_rl_hw[hw_idx])
            motor.tau = 0.0
            if hw_idx in self.leg_hardware_indices:
                motor.q = float(target_hw[hw_idx])
                motor.dq = 0.0
            else:
                motor.q = 0.0
                motor.dq = float(self.wheel_cmd[self.hardware_to_wheel_cmd[hw_idx]])

    def send_hardware_targets(self) -> None:
        """One 500-Hz tick: re-send the WBC targets to B2W and Z1."""
        self._write_b2w_wbc_cmd()
        self.send_b2w_cmd()
        self.z1.track_target_pd_runtime_once(
            q_target=self.arm_target.copy(),
            gripper_q_target_training=float(self.gripper_hold_pos),
            use_startup_gains=False,
        )

    # ------------------------------------------------------------------
    # Damping protection (retrieval sim2real, unchanged)
    # ------------------------------------------------------------------

    def _write_b2w_damping_cmd(self) -> None:
        for hw_idx in range(self.num_b2w_dof):
            motor = self.low_cmd.motor_cmd[hw_idx]
            motor.q = 0.0
            motor.dq = 0.0
            motor.kp = 0.0
            motor.kd = float(
                self.damping_kd_wheel if hw_idx in self.wheel_hardware_indices else self.damping_kd_b2w
            )
            motor.tau = 0.0

    def send_b2w_damping_once(self) -> None:
        self._write_b2w_damping_cmd()
        self.send_b2w_cmd()

    def send_z1_damping_once(self) -> None:
        """Hold-in-place; never forces a PASSIVE arm back into LOWCMD."""
        with self.z1._comm_lock:  # noqa: SLF001
            self.z1._send_arm_command_once(  # noqa: SLF001
                q_cmd=self.z1.q.copy(),
                gripper_q_cmd=float(self.z1.gripper_q),
                kp_cmd=self.damping_kp_z1,
                kd_cmd=self.damping_kd_z1,
                qd_cmd=np.zeros(6, dtype=np.float32),
                tau_cmd=np.zeros(6, dtype=np.float32),
            )

    def _move_legs_to_recovery_target(self) -> bool:
        self.read_b2w_state()
        start_leg = self.b2w_joint_pos[self.leg_policy_indices].copy()
        target_leg = self.leg_recovery_target.copy()
        num_steps = max(1, int(round(self.leg_recovery_duration_s / self.control_dt)))
        print(f"[RECOVERY] leg12 -> target over {self.leg_recovery_duration_s:.2f}s | arm untouched")
        deadline = time.perf_counter()
        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            pose = np.zeros(self.num_b2w_dof, dtype=np.float32)
            pose[self.leg_policy_indices] = (1.0 - alpha) * start_leg + alpha * target_leg
            self._write_b2w_pose_cmd(pose, use_pd_gains=True)
            self.send_b2w_cmd()
            try:
                self.send_z1_damping_once()
            except Exception as exc:  # noqa: BLE001
                print(f"[RECOVERY][WARN] Z1 protection command failed: {exc!r}")
            deadline += self.control_dt
            sleep_until(deadline)
        self.read_b2w_state()
        error = float(np.max(np.abs(self.b2w_joint_pos[self.leg_policy_indices] - target_leg)))
        print(f"[RECOVERY] max_abs_err={error:.4f} rad (tol {self.leg_recovery_tolerance_rad:.4f})")
        return bool(np.isfinite(error) and error <= self.leg_recovery_tolerance_rad)

    def recover_legs(self) -> bool:
        print("\n" + "=" * 96 + "\n[RECOVERY] MANUAL LEG RECOVERY -- the arm is NOT moved.\n" + "=" * 96)
        try:
            self.read_robot_state()
            ok = self._move_legs_to_recovery_target()
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[RECOVERY][ERROR] {exc!r}")
            return False
        print("[RECOVERY] SUCCESS." if ok else "[RECOVERY] FAILED.", "The policy will NOT resume.")
        return ok

    def enter_damping_protection_mode(self, reason: str) -> None:
        print("\n" + "!" * 96)
        print(f"[TRANSPORT][PROTECT] {reason}")
        print("[TRANSPORT][PROTECT] B2W: kp=0, high kd. Z1: hold q, high kd. Puck is NOT held.")
        if self.leg_recovery_enabled:
            print("[TRANSPORT][PROTECT] Release A, then press A to fold the legs down.")
        print("[TRANSPORT][PROTECT] Press Ctrl-C to quit.")
        print("!" * 96 + "\n")
        recovery_armed = self.remote_controller.button[KeyMap.A] != 1
        counter = 0
        try:
            while True:
                loop_start = time.perf_counter()
                for label, action in (
                    ("sensor read", self.read_robot_state),
                    ("B2W damping", self.send_b2w_damping_once),
                    ("Z1 damping", self.send_z1_damping_once),
                ):
                    try:
                        action()
                    except Exception as exc:  # noqa: BLE001
                        print(f"[PROTECT][WARN] {label} failed: {exc!r}")
                if self.leg_recovery_enabled:
                    pressed = self.remote_controller.button[KeyMap.A] == 1
                    if not recovery_armed:
                        if not pressed:
                            recovery_armed = True
                            print("[PROTECT] A released; leg recovery armed.")
                    elif pressed:
                        recovery_armed = False
                        self.recover_legs()
                        loop_start = time.perf_counter()
                if counter % max(1, self.damping_print_period) == 0:
                    print(f"[PROTECT] active | z1_fsm={self.z1.get_fsm_state()} | "
                          f"recovery_armed={int(recovery_armed)}")
                counter += 1
                sleep_until(loop_start + self.control_dt)
        except KeyboardInterrupt:
            print("[TRANSPORT][PROTECT] Ctrl-C received.")

    def send_exit_damping_once(self) -> None:
        for action in (self.read_robot_state, self.send_b2w_damping_once, self.send_z1_damping_once):
            try:
                action()
            except Exception as exc:  # noqa: BLE001
                print(f"[EXIT][WARN] {exc!r}")

    # ------------------------------------------------------------------
    # Startup (retrieval sequence; gripper held, puck gate added)
    # ------------------------------------------------------------------

    def _hold_arm_default_lowcmd(self) -> None:
        self.z1.hold_pose_lowcmd(
            self.default_arm_pos.copy(), self.z1.gripper_training_to_sdk(self.gripper_hold_pos)
        )

    def zero_torque_state(self) -> None:
        from utils.command_helper import create_zero_cmd  # noqa: PLC0415

        print("[TRANSPORT] Zero torque. Press START to continue.")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_b2w_cmd()
            time.sleep(self.control_dt)

    def move_arm_to_default(self) -> None:
        print("[TRANSPORT] Moving the arm to DEFAULT (gripper held at the training position)...")
        self.z1.move_to_pose_official(
            target_q=self.default_arm_pos.copy(),
            target_gripper=self.z1.gripper_training_to_sdk(self.gripper_hold_pos),
            duration_s=float(self.cfg["arm_default_transition_s"]),
            step_callback=None,
        )
        self.z1.read_state()
        print(f"[TRANSPORT] Arm at DEFAULT, max err={np.max(np.abs(self.z1.q - self.default_arm_pos)):.4f} rad.")

    def hold_arm_default_until_A(self) -> None:
        from utils.command_helper import create_zero_cmd  # noqa: PLC0415

        print("[TRANSPORT] Holding arm DEFAULT. Press A to stand up.")
        while self.remote_controller.button[KeyMap.A] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_b2w_cmd()
            self._hold_arm_default_lowcmd()
            time.sleep(self.control_dt)
        while self.remote_controller.button[KeyMap.A] == 1:  # require release
            create_zero_cmd(self.low_cmd)
            self.send_b2w_cmd()
            self._hold_arm_default_lowcmd()
            time.sleep(self.control_dt)

    def move_b2w_to_pose(self, target_b2w_pos_policy: np.ndarray, duration_s: float) -> None:
        target = np.asarray(target_b2w_pos_policy, dtype=np.float32).reshape(16)
        num_steps = max(1, int(round(duration_s / self.control_dt)))
        start_hw = np.array([self.low_state.motor_state[i].q for i in range(self.num_b2w_dof)], dtype=np.float32)
        start_policy = start_hw[self.hardware_to_policy]
        deadline = time.perf_counter()
        for step in range(num_steps):
            alpha = float(step + 1) / float(num_steps)
            self._write_b2w_pose_cmd((1.0 - alpha) * start_policy + alpha * target, use_pd_gains=True)
            self.send_b2w_cmd()
            self._hold_arm_default_lowcmd()
            deadline += self.control_dt
            sleep_until(deadline)

    def hold_default_until_ready_and_A(self) -> None:
        """Hold DEFAULT; the operator places the puck; A starts only when ready.

        Ready = valid base + puck mocap AND puck_D inside the training spawn
        region (centre +/- 1 cm, resting height) widened by the actor's
        training noise. Re-checked on a fresh snapshot at the A press.
        """
        print("[TRANSPORT] Holding DEFAULT. Place the puck at the disk centre.")
        print("[TRANSPORT] Press A when the gate reads OK to ENABLE the transport policy.")
        last_print = 0.0
        while True:
            self._write_b2w_pose_cmd(self.default_b2w_pos_policy, use_pd_gains=True)
            self.send_b2w_cmd()
            self._hold_arm_default_lowcmd()
            self.read_robot_state()
            ready, status = self.startup_readiness()
            now = time.monotonic()
            if now - last_print >= self.perception_status_print_s:
                last_print = now
                print(f"[READY-WAIT] ready={int(ready)} | {status}")
            if ready and self.remote_controller.button[KeyMap.A] == 1:
                final_ready, final_status = self.startup_readiness()
                if final_ready:
                    print(f"[TRANSPORT] ENABLED by operator | {final_status}")
                    return
                print(f"[TRANSPORT] A pressed but not ready: {final_status}")
            time.sleep(self.control_dt)

    def settle_on_runtime_gains(self) -> None:
        num_ticks = int(round(self.runtime_settle_s * self.hardware_hz))
        t0 = time.perf_counter()
        for tick in range(num_ticks):
            self._write_b2w_pose_cmd(self.default_b2w_pos_policy, use_pd_gains=False)
            self.send_b2w_cmd()
            self.z1.track_target_pd_runtime_once(
                q_target=self.default_arm_pos.copy(),
                gripper_q_target_training=float(self.gripper_hold_pos),
                use_startup_gains=False,
            )
            sleep_until(t0 + (tick + 1) * self.hardware_dt)

    # ------------------------------------------------------------------
    # Control loop / run
    # ------------------------------------------------------------------

    def policy_loop(self) -> str:
        """500-Hz hardware loop with the WBC on a decimation (retrieval)."""
        print(
            f"[TRANSPORT] Main loop | HL 10 Hz -> WBC 50 Hz -> hardware {self.hardware_hz:g} Hz. "
            "Sticks = base ref, D-pad = EE target, SELECT = damping protection."
        )
        deadline = time.perf_counter()
        tick = 0
        while True:
            if tick % self.hardware_ticks_per_wbc_step == 0:
                reason = self.wbc_step()
                if reason is not None:
                    return reason
            self.send_hardware_targets()
            deadline += self.hardware_dt
            now = time.perf_counter()
            if deadline < now - self.control_dt:
                print("[TIMING][WARN] hardware loop overrun; re-anchoring.")
                deadline = now + self.hardware_dt
            elif deadline > now:
                time.sleep(deadline - now)
            tick += 1

    def print_banner(self) -> None:
        print("=" * 100)
        print("B2WZ1 TRANSPORT (6-D EE residual) -> FROZEN TRANSPORT WBC -> REAL ROBOT (MOCAP)")
        print("=" * 100)
        print(f"High policy : {self.high_policy_path} (sha256 verified)")
        print(f"Low policy  : {self.low_policy_path} (sha256 verified)")
        print("UAN         : NOT LOADED (simulation-only); Z1 firmware POSITION_PD")
        print(f"Z1 gains    : Kp={np.round(self.z1_kp_effective, 3)} Kd={np.round(self.z1_kd_effective, 3)}")
        print(f"Puck        : mass={self.cfg['puck_mass_kg']} kg, r={self.puck_radius} h={self.puck_height}")
        print(f"Disk        : r={self.disk.radius} t={self.disk.thickness}, frame from Z1 FK")
        print(f"Contact     : rho<={self.disk.radius} and |z_D-{self.puck_rest_height:.3f}|<="
              f"{self.puck_obs_noise}; drop after {self.contact_monitor.drop_steps} HL steps")
        print(f"Operator    : base ref sticks x{self.command_scale.tolist()}; D-pad "
              f"up/down/left/right = HIGH/LOW/LEFT/RIGHT (min hold "
              f"{self.task.min_hold_steps * self.hl_control_dt:.1f}s)")
        print("=" * 100)

    def run(self) -> None:
        protection_reason: Optional[str] = None
        perception_started = False
        try:
            self.print_banner()
            from utils.command_helper import InitLowCmd  # noqa: PLC0415

            self.wait_for_low_state()
            InitLowCmd(self.low_cmd)
            self.z1.connect()
            self.read_robot_state()
            print("[TRANSPORT] Connecting to Motive...")
            self.perception.initialize()
            self.perception.start()
            perception_started = True

            self.zero_torque_state()
            self.move_arm_to_default()
            self.hold_arm_default_until_A()
            self.move_b2w_to_pose(self.squat_b2w_pos_policy, float(self.cfg["squat_transition_s"]))
            self.move_b2w_to_pose(self.default_b2w_pos_policy, float(self.cfg["default_transition_s"]))
            self.hold_default_until_ready_and_A()
            self.settle_on_runtime_gains()
            self.initialize_policy_state_and_history()
            self.open_hl_log()
            protection_reason = self.policy_loop()
        except KeyboardInterrupt:
            print("[TRANSPORT] KeyboardInterrupt received.")
        except Exception as exc:  # noqa: BLE001 - any fault must reach protection
            protection_reason = f"{type(exc).__name__}: {exc}"
            print(f"[TRANSPORT][ERROR] {protection_reason}")
        finally:
            self.close_hl_log()
            if protection_reason is not None:
                self.enter_damping_protection_mode(protection_reason)
            else:
                self.send_exit_damping_once()
            if perception_started:
                try:
                    self.perception.stop()
                except Exception as exc:  # noqa: BLE001
                    print(f"[EXIT][WARN] perception stop failed: {exc!r}")
            print("[TRANSPORT] Exit.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B2WZ1 transport policy over the frozen WBC, real robot + OptiTrack."
    )
    parser.add_argument("net", type=str, help="Unitree network interface, e.g. enxa0cec819e15f")
    parser.add_argument(
        "config", type=str, nargs="?",
        default="deploy/configs/b2wz1_hl_transport_uan_mocap.yaml",
        help="Path to the deployment YAML.",
    )
    args = parser.parse_args()

    # Validate the setup BEFORE any DDS / arm / mocap connection is opened.
    with open(args.config, "r") as f:
        problems = validate_transport_setup(yaml.safe_load(f))
    if problems:
        print("[TRANSPORT] Setup incomplete -- nothing was started:\n  - " + "\n  - ".join(problems))
        sys.exit(2)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: PLC0415

    ChannelFactoryInitialize(0, args.net)
    B2WZ1TransportMocapController(cfg_path=args.config).run()


if __name__ == "__main__":
    main()
