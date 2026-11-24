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

Source of truth (synchronized 2026-11-20, NO-SETTLE policy)
------------------------------------------------------------
    IsaacLab_huanyu/.../direct/b2wz1_hl_transport_UAN/b2wz1_hl_env.py
    unitree_mujoco/simulate_python/deploy_mujoco/b2wz1_hl_transport_uan.py
    (TransportSim2Sim; validated sim2sim of the same checkpoint)

  * actor frame (64-D), 3 frames FEATURE-MAJOR = 192-D:
        ang_vel_b(3) proj_gravity_b(3) leg_q_rel(12) arm_q_rel(6)
        leg_qd(12) arm_qd(6) wheel_qd(4) base_vel_ref(3) ee_kp0_ref(3)
        ee_ypr_ref[yaw,pitch,roll](3) prev_normalized_action(6) puck_center_base(3)
    prev_normalized_action = clamp(a_(t-1), -1, 1): the policy action of the
        immediately completed HL step, BEFORE residual scaling / WBC clipping
        (zero after reset). NOT the executed residual (reward-only now).
    puck_center_base = R_WB^T (p_puck,W - p_base,W) with the FULL base
        orientation (roll/pitch/yaw), base = base_link (root link) origin.
  * residual decode (_decode_hl_action_to_ll_commands):
        a       = clamp(raw, -1, 1)
        kp0_cmd = clip(kp0_ref + a[0:3] * 0.10 m,  kp0 ranges)
        ypr_cmd = clip(ypr_ref + a[3:6] * 45 deg,  yaw/pitch/roll ranges)
        base_command = base reference (no learned base residual)
    There is NO settle: the residual is active from the first HL step.
  * EE pose -> 9-D WBC keypoints with the training adapter
    build_keypoints_from_kp0_yaw_pitch_roll_plb (yaw/pitch define local +X,
    roll about +X). NOT the retrieval ABS keypoint builder.
  * reset: HIGH (kp0 AND yaw/pitch/roll, PLB) = gripperStator pose of the
    reset state (default joints, nominal base pose). That pose is the same in
    every training episode, so it is a CONSTANT (training_high_kp0 /
    training_high_ypr, bit-exact from the MuJoCo reset). Its orientation is
    the EE orientation reference of the whole run. HIGH is held for at least
    one EE interval; the base reference is taken at reset.
  * reference process: obs_t(ref_t, a_(t-1)) -> a_t -> execute at ref_t ->
    contact check -> advance refs (min-jerk 1.5 s / LOW 2.0 s) ->
    obs_(t+1)(ref_(t+1), a_t).

Real-world substitutions (everything else is identical)
--------------------------------------------------------
  * Task references come from the OPERATOR instead of the training sampler:
      base reference = sticks (b2wz1_wbc_uan.py mapping), clipped to the
                       training range, read once per HL step;
      EE target      = D-pad up/down/left/right = HIGH/LOW/LEFT/RIGHT, with
                       the training minimum-jerk transitions and at least the
                       training minimum hold (4 s) between switches.
  * puck_center_base, the only non-proprioceptive actor input, is measured by
    OptiTrack: puck rigid body (Motive pivot = puck geometric centre) and the
    B2 rigid body (Motive pivot/axes = base_link), BOTH taken from the SAME
    NatNet frame, so there is no time skew between them:
        puck_base = R_root^T (puck_W - p_root)
    The B2 asset calibration therefore enters the actor observation directly
    (READY gates: mocap-vs-IMU tilt, live-vs-training HIGH EE pose).
  * Puck/disk contact (training: net contact force > 0.05 N) has no sensor.
    It is replaced by a geometric proxy in the disk frame D (Z1 FK + training
    mount; there is no Motive disk asset): in contact iff the puck centre is
    over the disk (rho_D <= disk_radius) and at resting height
    (|z_D - 0.042| <= 0.01 m).
    It is LOGGED ONLY (CSV contact / no_contact_steps, visualizer colour):
    a lost contact does NOT end the run. puck_D is NOT an actor input.

The actor observes no puck velocity, so no velocity estimate is computed.

Faults -> damping protection (retrieval semantics)
---------------------------------------------------
SELECT, Z1 PASSIVE, invalid/stale/untracked mocap (base or puck), base/puck
from different NatNet frames,
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
import math
import os
import sys
import threading
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

from utils.math import (  # noqa: E402
    euler_xyz_from_quat_wxyz,
    quat_apply_wxyz,
    quat_from_rotmat_wxyz,
    quat_from_yaw_wxyz,
    quat_rotate_inverse_numpy,
    quat_unique_wxyz,
)
from utils.mocap_frames import q_to_mat  # noqa: E402
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

# Keys of the superseded settle / canonical-orientation policy. A YAML that
# still carries them was written for different observation semantics.
LEGACY_SEMANTICS_KEYS = (
    "reset_settle_time_s", "canonical_ee_yaw", "canonical_ee_pitch", "canonical_ee_roll",
)

# Where the episode HIGH pose comes from:
#   "training" the reset EE pose of training (constant, bit-exact sim2sim);
#   "measured" the live EE pose at enable (FK + IMU + mocap base height).
HIGH_POSE_SOURCES = ("training", "measured")
# Free joint of the puck in the sim2sim scene MJCF (b2wz1_disk_puck.xml).
VIZ_PUCK_FREE_JOINT = "transport_puck_freejoint"
VIZ_PUCK_GEOM = "transport_puck_geom"


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


def minimum_jerk_blend(tau: float) -> np.float32:
    """Fifth-order smoothstep, float32 exactly as Isaac / sim2sim."""
    tau = np.float32(min(max(float(tau), 0.0), 1.0))
    return np.float32(tau ** 3 * (np.float32(10.0) - np.float32(15.0) * tau + np.float32(6.0) * tau ** 2))


def ee_ypr_from_axes_plb(v_x: np.ndarray, v_z: np.ndarray) -> np.ndarray:
    """Isaac _measure_ee_ypr_plb: exact inverse of the keypoint adapter.

    v_x / v_z = EE local +x / +z expressed in PLB. Returns [yaw, pitch, roll].
    """
    v_x = np.asarray(v_x, dtype=np.float64).reshape(3)
    v_z = np.asarray(v_z, dtype=np.float64).reshape(3)
    yaw = math.atan2(v_x[1], v_x[0])
    pitch = math.asin(float(np.clip(v_x[2], -1.0, 1.0)))
    ez = np.array([0.0, 0.0, 1.0])
    z0 = ez - v_x[2] * v_x
    z0 = z0 / max(float(np.linalg.norm(z0)), 1.0e-6)
    y0 = np.cross(z0, v_x)
    roll = math.atan2(-float(np.dot(v_z, y0)), float(np.dot(v_z, z0)))
    return np.array([yaw, pitch, roll], dtype=np.float64)


def keypoint_axes_rotmat(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """PLB rotation whose x / z columns are the adapter's local +X / +Z."""
    kp = build_keypoints_from_kp0_yaw_pitch_roll_plb(np.zeros(3), yaw, pitch, roll, 1.0, 1.0)
    x = kp[3:6].astype(np.float64)
    z = kp[6:9].astype(np.float64)
    x /= max(np.linalg.norm(x), 1e-12)
    z -= np.dot(z, x) * x
    z /= max(np.linalg.norm(z), 1e-12)
    return np.stack([x, np.cross(z, x), z], axis=1)


def tilt_angle_deg(gravity_a: np.ndarray, gravity_b: np.ndarray) -> float:
    """Angle between two projected-gravity vectors (IMU vs mocap base tilt)."""
    a = np.asarray(gravity_a, dtype=np.float64).reshape(3)
    b = np.asarray(gravity_b, dtype=np.float64).reshape(3)
    c = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


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
    ee_ypr_ref: np.ndarray,
    prev_normalized_action: np.ndarray,
    puck_pos_base: np.ndarray,
) -> np.ndarray:
    """The 64-D actor frame, in Isaac _build_hl_obs_frames / sim2sim order.

    prev_normalized_action: clamp(a_(t-1), -1, 1) of the completed HL step.
    puck_pos_base:          R_WB^T (p_puck - p_base), FULL base orientation.
    """
    parts = (
        (base_ang_vel_b, 3), (projected_gravity_b, 3), (leg_pos_rel, 12),
        (arm_pos_rel, 6), (leg_vel, 12), (arm_vel, 6), (wheel_vel, 4),
        (base_velocity_ref, 3), (ee_kp0_ref, 3), (ee_ypr_ref, 3),
        (prev_normalized_action, 6), (puck_pos_base, 3),
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


def puck_pos_in_frame(
    puck_pos: np.ndarray, frame_pos: np.ndarray, frame_rot: np.ndarray
) -> np.ndarray:
    """R^T (puck - origin): the puck centre in the (FK) disk frame D.

    Both inputs must be in the same frame (base_link on the robot).
    """
    rot = np.asarray(frame_rot, dtype=np.float64).reshape(3, 3)
    rel = np.asarray(puck_pos, dtype=np.float64).reshape(3) - np.asarray(
        frame_pos, dtype=np.float64
    ).reshape(3)
    return (rot.T @ rel).astype(np.float32)


class PuckContactMonitor:
    """Geometric stand-in for the training puck/disk contact-force detector.

    Training: contact = net disk/puck force > 0.05 N; puck_drop after
    `drop_steps` consecutive HL steps without contact (no settle phase: every
    completed HL step counts). A puck lying on the disk has its centre over
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

    def update(self, puck_d: Optional[np.ndarray]) -> Tuple[bool, int, bool]:
        """One completed HL transition -> (contact, no_contact_steps, drop)."""
        contact = self.in_contact(puck_d)
        if contact:
            self.no_contact_steps = 0
        else:
            self.no_contact_steps += 1
        return contact, self.no_contact_steps, self.no_contact_steps >= self.drop_steps


def puck_start_gate(
    puck_d: Optional[np.ndarray],
    xy_limit: float,
    spawn_clearance: float,
    obs_noise: float,
    rest_height: float,
) -> Tuple[bool, str]:
    """Is the puck near the disk centre (|x_D|,|y_D| <= xy_limit) at resting height?"""
    if puck_d is None:
        return False, "puck_D unavailable"
    p = np.asarray(puck_d, dtype=np.float64).reshape(3)
    if not np.isfinite(p).all():
        return False, "puck_D non-finite"
    xy_lim = float(xy_limit)
    z_nom = float(rest_height) + float(spawn_clearance)
    problems = []
    if abs(p[0]) > xy_lim or abs(p[1]) > xy_lim:
        problems.append(f"xy=({p[0]:+.4f},{p[1]:+.4f}) outside +/-{xy_lim:.3f}")
    if abs(p[2] - z_nom) > float(obs_noise):
        problems.append(f"z={p[2]:+.4f} not within {obs_noise:.3f} of {z_nom:.4f}")
    return (not problems), ("OK" if not problems else "; ".join(problems))


class TransportTask:
    """Reference process + residual decoder of the NO-SETTLE transport env.

    Pure state machine, no hardware. Arithmetic (float32) and ordering are
    those of Isaac B2WZ1HLTransportUANEnv and the sim2sim TransportSim2Sim
    (TransportReferenceProcess + decode_hl_action); only the SOURCE of the
    references differs (operator instead of the training sampler).

    Action bookkeeping per HL step (Isaac _pre_physics_step):
        raw_action        policy output as received
        normalized_action clamp(raw, -1, 1) = a_t; decoder input AND the
                          previous-action term of the NEXT observation
        executed_action   ((cmd - ref) / scale).clamp(-1, 1) after WBC
                          absolute clipping; reward/diagnostics only
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
        if np.any(self.canonical_positions < self.pos_min) or np.any(self.canonical_positions > self.pos_max):
            raise ValueError("canonical_ee_positions must lie inside the WBC kp0 bounds.")
        self.base_ref_ranges = f32("base_ref_ranges")
        self.base_cmd_scale = f32("base_cmd_scale").reshape(3)
        if self.base_ref_ranges.shape != (3, 2):
            raise ValueError("base_ref_ranges must have shape [3,2].")
        if np.any(self.base_ref_ranges[:, 0] < -self.base_cmd_scale) or np.any(
            self.base_ref_ranges[:, 1] > self.base_cmd_scale
        ):
            raise ValueError("base_ref_ranges must stay inside frozen-WBC base bounds.")

        dt = self.hl_control_dt
        self.fast_steps = int(round(float(cfg["ee_fast_transition_time_s"]) / dt))
        self.low_steps = int(round(float(cfg["ee_low_transition_time_s"]) / dt))
        self.min_hold_steps = int(round(float(cfg["ee_ref_resample_time_range_s"][0]) / dt))
        if min(self.fast_steps, self.low_steps, self.min_hold_steps) <= 0:
            raise ValueError("Transition / hold times must be positive.")
        if max(self.fast_steps, self.low_steps) > self.min_hold_steps:
            raise ValueError("EE transitions must not exceed the minimum EE hold period.")
        self.reset(self.canonical_positions[0], np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))

    # ------------------------------------------------------------------
    @property
    def rot_ref(self) -> np.ndarray:
        """EE orientation reference [yaw, pitch, roll] (constant per run)."""
        return self.ee_ypr_ref.copy()

    def _keypoints(self, kp0: np.ndarray, rot: np.ndarray) -> np.ndarray:
        return build_keypoints_from_kp0_yaw_pitch_roll_plb(
            kp0, float(rot[0]), float(rot[1]), float(rot[2]), self.kp_dx, self.kp_dz
        )

    def _sync_commands_from_references(self) -> None:
        """Zero-residual path (Isaac _sync_ll_commands_from_references; reset only)."""
        self.base_command = self.base_velocity_ref.copy()
        self.kp0_cmd = self.ee_kp0_ref.copy()
        self.rot_cmd = self.ee_ypr_ref.copy()
        self.ee_cmd_plb = self._keypoints(self.kp0_cmd, self.rot_cmd)

    def reset(self, high_kp0: np.ndarray, high_ypr: np.ndarray, operator_base_ref: np.ndarray) -> None:
        """Isaac _reset_command_references(high_kp0, high_ypr) for one env.

        HIGH (position AND orientation) is clipped to the WBC command bounds,
        becomes the first EE target (no transition) and its orientation the
        reference of the whole run. The base reference is taken NOW (Isaac
        samples it at reset). The previous action is zero.
        """
        kp0 = np.asarray(high_kp0, dtype=np.float32).reshape(3)
        ypr = np.asarray(high_ypr, dtype=np.float32).reshape(3)
        if not (np.isfinite(kp0).all() and np.isfinite(ypr).all()):
            raise FloatingPointError(f"Non-finite HIGH pose: kp0={kp0} ypr={ypr}")
        kp0 = np.maximum(np.minimum(kp0, self.pos_max), self.pos_min).astype(np.float32)
        ypr = np.maximum(np.minimum(ypr, self.rot_max), self.rot_min).astype(np.float32)
        self.episode_high_kp0 = kp0.copy()
        self.ee_kp0_ref = kp0.copy()
        self.ee_ypr_ref = ypr.copy()
        self.current_target = EE_TARGET_HIGH
        self.transition_start = kp0.copy()
        self.transition_target = kp0.copy()
        self.transition_step = 0
        self.transition_total = 1
        self.steps_since_switch = 0
        self.pending_target: Optional[int] = None
        self.last_event = "reset"
        self.raw_action = np.zeros(6, dtype=np.float32)
        self.normalized_action = np.zeros(6, dtype=np.float32)
        self.executed_action = np.zeros(6, dtype=np.float32)
        self.raw_out_of_range_count = 0
        self.base_velocity_ref = self.clip_base_reference(operator_base_ref)
        self._sync_commands_from_references()

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
        """Isaac _sample_ee_pose_reference for an operator-chosen target.

        Starts at the reference last exposed to the actor (no jump); only the
        POSITION moves, the orientation reference is the episode HIGH one.
        """
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
        self.steps_since_switch = 0

    def _advance_interpolation(self) -> None:
        total = max(int(self.transition_total), 1)
        self.transition_step = min(int(self.transition_step) + 1, total)
        blend = minimum_jerk_blend(np.float32(self.transition_step) / np.float32(total))
        self.ee_kp0_ref = (
            self.transition_start + blend * (self.transition_target - self.transition_start)
        ).astype(np.float32)

    def advance(self, operator_base_ref: np.ndarray) -> str:
        """Isaac _advance_command_references_for_next_step.

        Called after a completed 10-Hz transition and before the next actor
        observation (never right after reset). Order as Isaac: interpolation
        step, base reference, then a new EE target (starting at the current
        reference). Returns a short event string for logging.
        """
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
        """Isaac _pre_physics_step + _decode_hl_action_to_ll_commands.

        Decoded against the CURRENT references; no reference update here.
        """
        raw = np.asarray(action, dtype=np.float32).reshape(6,)
        if not np.isfinite(raw).all():
            raise FloatingPointError(f"Non-finite high-level action: {raw}")
        if np.any(np.abs(raw) > 1.0 + 1e-4):
            self.raw_out_of_range_count += 1
        a = np.clip(raw, np.float32(-1.0), np.float32(1.0)).astype(np.float32)
        self.raw_action = raw.copy()
        self.normalized_action = a

        self.base_command = self.base_velocity_ref.copy()

        pos_ref = self.ee_kp0_ref.astype(np.float32)
        rot_ref = self.ee_ypr_ref.astype(np.float32)
        pos_raw = (pos_ref + (a[0:3] * self.pos_scale).astype(np.float32)).astype(np.float32)
        rot_raw = (rot_ref + (a[3:6] * self.rot_scale).astype(np.float32)).astype(np.float32)
        kp0_cmd = np.maximum(np.minimum(pos_raw, self.pos_max), self.pos_min).astype(np.float32)
        rot_cmd = np.maximum(np.minimum(rot_raw, self.rot_max), self.rot_min).astype(np.float32)

        self.executed_action = np.concatenate([
            np.clip((kp0_cmd - pos_ref) / self.pos_scale, -1.0, 1.0),
            np.clip((rot_cmd - rot_ref) / self.rot_scale, -1.0, 1.0),
        ]).astype(np.float32)
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
    for key in LEGACY_SEMANTICS_KEYS:
        if key in cfg:
            problems.append(
                f"{key}: belongs to the superseded settle/canonical-orientation policy; "
                "the no-settle policy takes HIGH from training_high_kp0/_ypr"
            )

    for key in ("training_high_kp0", "training_high_ypr"):
        try:
            v = np.asarray(cfg.get(key), dtype=np.float64).reshape(3)
            if not np.isfinite(v).all():
                raise ValueError
        except Exception:  # noqa: BLE001
            problems.append(f"{key}: must be 3 finite numbers (MuJoCo/Isaac reset EE pose)")
    if str(cfg.get("high_pose_source", "")).strip().lower() not in HIGH_POSE_SOURCES:
        problems.append(f"high_pose_source: must be one of {HIGH_POSE_SOURCES}")
    for key in ("high_pose_max_deviation_m", "high_pose_max_deviation_deg"):
        try:
            if not float(cfg.get(key)) > 0.0:
                raise ValueError
        except Exception:  # noqa: BLE001
            problems.append(f"{key}: must be a positive number")
    tilt = cfg.get("mocap_imu_max_tilt_deg")
    if tilt is not None:
        try:
            if not float(tilt) > 0.0:
                raise ValueError
        except Exception:  # noqa: BLE001
            problems.append("mocap_imu_max_tilt_deg: must be positive (or null to disable)")

    # The disk frame D is always Z1 FK + the training mount: a YAML still
    # asking for the removed Motive disk asset path is refused, not ignored.
    if str(cfg.get("disk_frame_source", "fk")).strip().lower() != "fk":
        problems.append("disk_frame_source: the mocap disk path was removed; delete the key (D = Z1 FK)")
    # Rigid bodies the run needs besides the B2: the puck.
    assets = ["puck"]

    for label in assets:
        if cfg.get(f"mocap_{label}_name") in (None, "") and cfg.get(f"mocap_{label}_id") is None:
            problems.append(
                f"mocap_{label}_name / mocap_{label}_id: {label} rigid body not configured"
            )
        # The pivot is placed in Motive (as retrieval); the YAML offset is an
        # optional extra that defaults to zero, but must be sane when given.
        offset = cfg.get(f"mocap_{label}_offset_local")
        if offset is not None:
            try:
                off = np.asarray(offset, dtype=np.float64).reshape(3)
                if not np.isfinite(off).all():
                    raise ValueError
            except Exception:  # noqa: BLE001
                problems.append(f"mocap_{label}_offset_local: must be 3 finite numbers [m]")
        if not (0.0 < float(cfg.get(f"mocap_{label}_max_age_s", 0.0)) <= 0.1 + 1e-9):
            problems.append(f"mocap_{label}_max_age_s: must be in (0, 0.1] s (one HL period)")

    # Every rigid body must be a different Motive asset.
    for i, a in enumerate(["body"] + assets):
        for b in (["body"] + assets)[i + 1:]:
            name_a, name_b = cfg.get(f"mocap_{a}_name"), cfg.get(f"mocap_{b}_name")
            id_a, id_b = cfg.get(f"mocap_{a}_id"), cfg.get(f"mocap_{b}_id")
            if (name_a not in (None, "") and name_a == name_b) or (
                id_a is not None and id_a == id_b
            ):
                problems.append(f"mocap_{b}_*: {b} must be a different rigid body from the {a}")

    if cfg.get("puck_geometry_verified") is not True:
        problems.append("puck_geometry_verified: real puck size not confirmed against training")
    if cfg.get("disk_mount_verified") is not True:
        problems.append("disk_mount_verified: physical disk mount not confirmed against training")

    has_offset_file = bool(cfg.get("mocap_root_offset_path")) or bool(cfg.get("mocap_root_offset"))
    if not has_offset_file and cfg.get("mocap_root_frame_is_base_link") is not True:
        problems.append("mocap root frame: neither aligned in Motive nor calibrated")

    scale = np.asarray(cfg.get("command_scale", [np.inf] * 3), dtype=np.float64).reshape(3)
    ranges = np.asarray(cfg["base_ref_ranges"], dtype=np.float64).reshape(3, 2)
    if np.any(scale < 0.0) or np.any(scale > np.minimum(-ranges[:, 0], ranges[:, 1]) + 1e-9):
        problems.append("command_scale: must be within the training base_ref_ranges")
    return problems


# ======================================================================
# OptiTrack: robot body + puck
# ======================================================================


class TransportMocapPerception(MocapPerceptionSystem):
    """MocapPerceptionSystem with the transport assets: B2 and puck.

    Reuses the validated NatNet client, root-offset handling, staleness and
    tracking classification and base-frame transform of the parent; only the
    asset set differs (no retrieval target).

    Every body of one snapshot is sampled from the SAME NatNet frame (see
    _sample_in), so puck position_base pairs poses of one capture instant.

    Snapshot:
        body         validated base_link pose (mocap root), frame_number
        base_height  root z - ground_z
        puck         puck centre: position_world, quat_wxyz, position_base
                     (= R_root^T (puck_W - p_root): the ACTOR input), age_ms
        mocap        frame number, received / parse-error / SKIPPED frame counts
    The raw marker cloud is NOT part of the 50-Hz snapshot (it is costly to
    convert); the visualizer pulls it through marker_cloud() at its own rate.

    The puck is sampled even while the B2 body is invalid (its WORLD pose
    does not depend on it); only position_base is then None. The
    controller still faults on the missing base height.
    """

    def __init__(
        self,
        body: MocapBodySelector,
        puck_body: MocapBodySelector,
        **kwargs: Any,
    ) -> None:
        super().__init__(body=body, object_body=puck_body, retrieval_body=None, **kwargs)
        self.puck_body = puck_body

    def _selectors(self) -> Tuple[MocapBodySelector, ...]:
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
        labels = [s.label for s in self._selectors()]
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                if self._resolved[a] == self._resolved[b]:
                    raise RuntimeError(
                        f"The {a} and the {b} resolve to the SAME rigid body "
                        f"(id {self._resolved[a]})."
                    )
        untracked = [s.label for s in self._selectors() if not self._is_tracked(self._resolved[s.label])]
        if untracked:
            self._log("[WARN] currently UNTRACKED: " + ", ".join(untracked))
        self._initialized = True

    def _sample_in(
        self, frame: Any, selector: MocapBodySelector, now: float
    ) -> Tuple[Optional[Any], str, Optional[float]]:
        """Parent _sample() classification, but on ONE pinned NatNet frame.

        The parent re-reads the client's latest frame per body, so the B2 and
        the puck can come from two consecutive frames. puck_center_base
        combines both poses, hence every body of a snapshot is taken from the
        same frame (one capture instant, one receive time).
        """
        rb_id = self._resolved.get(selector.label)
        if rb_id is None:
            return None, "NOT_CONFIGURED", None
        rb = None if frame is None else frame.rigid_bodies.get(int(rb_id))
        if rb is None:
            return None, "NOT_IN_STREAM", None
        age_ms = 1000.0 * rb.age_s(now)
        if not rb.tracking_valid:
            return None, "UNTRACKED", age_ms
        if rb.age_s(now) > selector.max_age_s:
            return None, "STALE", age_ms
        return rb, "OK", age_ms

    def _world_channel(
        self,
        frame: Any,
        selector: MocapBodySelector,
        p_root_w: Optional[np.ndarray],
        q_root_w: Optional[np.ndarray],
        now: float,
    ) -> Dict[str, Any]:
        """World pose of one asset (+ base-frame position when the root is known)."""
        rb, reason, age_ms = self._sample_in(frame, selector, now)
        if rb is None:
            state = self._invalid_channel(reason, age_ms)
            state.update(position_world=None, quat_wxyz=None, position_base=None, frame_number=None)
            return state
        p_world = self._world_point(rb, selector)
        p_base = None
        if p_root_w is not None and q_root_w is not None:
            p_base = (q_to_mat(q_root_w).T @ (p_world - p_root_w)).tolist()
        return {
            "valid": True,
            "source": f"MOCAP:{selector.name or selector.rb_id}",
            "reason": "OK",
            "position_world": p_world.tolist(),
            "quat_wxyz": rb.quat_wxyz.tolist(),
            "position_base": p_base,
            "age_ms": age_ms,
            "source_age_ms": age_ms,
            "frame_number": int(rb.frame_number),
            "mean_marker_error_m": rb.mean_marker_error,
        }

    def get_latest_snapshot(self, projected_gravity_b: Any = None) -> Dict[str, Any]:
        now = time.monotonic()
        frame = self.client.latest_frame()  # pinned: every body below comes from it
        body_rb, body_reason, body_age_ms = self._sample_in(frame, self.body, now)
        mocap_meta = {
            "frame_number": int(frame.frame_number) if frame is not None else None,
            "frames_received": self.client.frame_count,
            "parse_errors": self.client.parse_error_count,
            "frames_skipped": getattr(self.client, "skipped_frame_count", 0),
        }

        if body_rb is None:
            reason = f"BODY_{body_reason}"
            p_root_w = q_root_w = None
            body_state = {"valid": False, "reason": body_reason, "age_ms": body_age_ms,
                          "position_world": None, "root_position_world": None,
                          "root_quat_wxyz": None}
            height_state = self._invalid_height(reason, body_age_ms)
        else:
            p_root_w, q_root_w = self.root_offset.apply(body_rb.pos, body_rb.quat_wxyz)
            base_height = float(p_root_w[2] - self.ground_z)
            height_valid = bool(np.isfinite(base_height))
            body_state = {
                "valid": True, "reason": "OK", "age_ms": body_age_ms,
                "position_world": body_rb.pos.tolist(),
                "root_position_world": p_root_w.tolist(),
                "root_quat_wxyz": q_root_w.tolist(),
                "frame_number": int(body_rb.frame_number),
            }
            height_state = {
                "valid": height_valid,
                "height_m": base_height if height_valid else None,
                "reason": "OK" if height_valid else "NON_FINITE_HEIGHT",
                "age_ms": body_age_ms,
            }

        puck_state = self._world_channel(frame, self.puck_body, p_root_w, q_root_w, now)
        snap: Dict[str, Any] = {
            "body": body_state,
            "base_height": height_state,
            "puck": puck_state,
            "mocap": mocap_meta,
        }
        snap["valid"] = bool(height_state["valid"] and puck_state["valid"])
        return snap

    def marker_cloud(self) -> Dict[str, Any]:
        """Raw markers of the latest frame (visualisation only, call sparingly)."""
        return self._marker_cloud(self.client.latest_frame())


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
        self.visualizer = self._build_visualizer()

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
        self.puck_start_xy_limit = float(cfg["puck_start_xy_limit_m"])
        if not self.puck_start_xy_limit > 0.0:
            raise ValueError("puck_start_xy_limit_m must be > 0.")
        self.puck_spawn_clearance = float(cfg["puck_spawn_clearance"])
        self.puck_obs_noise = float(cfg["puck_obs_noise_m"])
        self.contact_monitor = PuckContactMonitor(
            disk_radius=self.disk.radius,
            rest_height=self.puck_rest_height,
            z_tolerance=self.puck_obs_noise,
            drop_steps=int(cfg["puck_drop_no_contact_steps"]),
        )

        # Episode HIGH pose (see HIGH_POSE_SOURCES) and the READY gates that
        # check the live robot against it.
        self.training_high_kp0 = np.asarray(cfg["training_high_kp0"], dtype=np.float32).reshape(3)
        self.training_high_ypr = np.asarray(cfg["training_high_ypr"], dtype=np.float32).reshape(3)
        self.high_pose_source = str(cfg.get("high_pose_source", "training")).strip().lower()
        if self.high_pose_source not in HIGH_POSE_SOURCES:
            raise ValueError(f"high_pose_source must be one of {HIGH_POSE_SOURCES}.")
        self.high_pose_max_dev_m = float(cfg["high_pose_max_deviation_m"])
        self.high_pose_max_dev_rad = math.radians(float(cfg["high_pose_max_deviation_deg"]))
        tilt = cfg.get("mocap_imu_max_tilt_deg")
        self.mocap_imu_max_tilt_deg = None if tilt is None else float(tilt)

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
        for label, path in (("high", self.high_policy_path), ("low", self.low_policy_path)):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{label} policy not found: {path}")

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
        # Actor input: puck centre in the full base_link frame (mocap B2 + puck,
        # same NatNet frame). None = unavailable -> perception fault.
        self.last_puck_base: Optional[np.ndarray] = None
        # Base tilt disagreement IMU vs mocap B2 asset [deg] (None = n/a).
        self.last_tilt_deg: Optional[float] = None
        # Diagnostics: inference latency [ms] and the measured HIGH deviation.
        self.hl_infer_ms = 0.0
        self.ll_infer_ms = 0.0
        self.hl_infer_ms_max = 0.0
        self.ll_infer_ms_max = 0.0
        self.last_high_deviation: Optional[Tuple[float, float]] = None
        # puck_D: puck_base in the FK disk frame D (contact proxy / start gate).
        self.last_puck_d: Optional[np.ndarray] = None
        # FK disk frame D in WORLD (via the mocap root), for the view only.
        self.last_fk_disk_w: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.last_contact: Tuple[bool, int, bool] = (True, 0, False)
        self.policy_active = False
        self._ee_buttons_prev = {button: 1 for button, _ in EE_TARGET_BUTTONS}
        self._t_start = time.monotonic()
        self._log_file = None
        self._log_writer = None

        # Visualizer monitor (retrieval pattern).
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._monitor_warned = False
        self._last_control_push = 0.0

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
        """Nonblocking mocap read; refreshes base height, puck_base and puck_D.

        puck_D = the puck (base frame, via the B2 asset) resolved against the
        Z1-FK disk frame of THIS step.
        """
        snap = self.perception.get_latest_snapshot()
        self.last_snapshot = snap
        height = snap.get("base_height") or {}
        h = height.get("height_m") if height.get("valid") else None
        self.base_height = float(h) if h is not None and np.isfinite(h) else None

        puck = snap.get("puck") or {}
        body = snap.get("body") or {}
        self.last_puck_base = None
        self.last_tilt_deg = None
        self.last_puck_d = None
        self.last_fk_disk_w = None

        # ACTOR INPUT: puck centre in the full base_link frame,
        #   R_root^T (puck_W - p_root), computed by the perception from the
        #   B2 and puck poses of one NatNet frame (Isaac _puck_center_pos_base).
        if puck.get("valid") and body.get("valid") and puck.get("position_base") is not None:
            pb = np.asarray(puck["position_base"], dtype=np.float32).reshape(3)
            if np.isfinite(pb).all():
                self.last_puck_base = pb
        # Cross-check of the mocap base orientation (used for puck_base) with
        # the IMU (used for projected gravity): only gravity direction, yaw
        # is arbitrary between the two.
        if body.get("valid") and body.get("root_quat_wxyz") is not None:
            R_root = q_to_mat(np.asarray(body["root_quat_wxyz"], dtype=np.float64))
            self.last_tilt_deg = tilt_angle_deg(
                R_root.T @ np.array([0.0, 0.0, -1.0]), self.projected_gravity_b
            )

        # FK disk frame D (base): Z1 FK of the gripperStator + training mount.
        stator_pos_b, stator_rot_b = self.z1.compute_policy_ee_pose_in_base()
        fk_pos_b, fk_rot_b = self.disk.frame_in(stator_pos_b, stator_rot_b)
        if self.last_puck_base is not None:
            d = puck_pos_in_frame(self.last_puck_base, fk_pos_b, fk_rot_b)
            if np.isfinite(d).all():
                self.last_puck_d = d
        if body.get("valid"):
            p_root = np.asarray(body["root_position_world"], dtype=np.float64)
            R_root = q_to_mat(np.asarray(body["root_quat_wxyz"], dtype=np.float64))
            self.last_fk_disk_w = (p_root + R_root @ np.asarray(fk_pos_b, dtype=np.float64),
                                   R_root @ np.asarray(fk_rot_b, dtype=np.float64))
        return snap

    def perception_fault(self, snap: Dict[str, Any]) -> Optional[str]:
        """Why the actor may not use this snapshot, or None. Never a stand-in."""
        reasons = []
        if self.base_height is None:
            reasons.append("BASE_HEIGHT:" + str((snap.get("base_height") or {}).get("reason")))
        puck = snap.get("puck") or {}
        body = snap.get("body") or {}
        if self.last_puck_base is None and self.base_height is not None and puck.get("valid"):
            reasons.append("PUCK_BASE:NON_FINITE")
        fn_body, fn_puck = body.get("frame_number"), puck.get("frame_number")
        if fn_body is not None and fn_puck is not None and int(fn_body) != int(fn_puck):
            reasons.append(f"FRAME_MISMATCH:body={fn_body},puck={fn_puck}")
        if self.last_puck_d is None:
            if not puck.get("valid"):
                reasons.append("PUCK:" + str(puck.get("reason")))
            elif self.last_puck_base is not None:
                reasons.append("PUCK_D:NON_FINITE")
        return None if not reasons else "|".join(reasons)

    def measure_ee_pose_plb(self) -> Tuple[np.ndarray, np.ndarray]:
        """Live gripperStator pose in PLB: (kp0, [yaw, pitch, roll]).

        Isaac _measure_ee_kp0_plb + _measure_ee_ypr_plb on the robot: Z1 FK,
        IMU roll/pitch/yaw, mocap base height. The EE axes come back out of
        the measured keypoints (kp1 - kp0 = dx * local +X, kp2 - kp0 = dz *
        local +Z), then the exact inverse of the keypoint adapter.
        """
        from utils.z1_helper import compute_ee_current_kp_plb  # noqa: PLC0415

        if self.base_height is None:
            raise RuntimeError("Base height is unknown: no valid mocap body frame.")
        kps = np.asarray(compute_ee_current_kp_plb(
            base_quat_wxyz=self.base_quat_wxyz,
            base_height=float(self.base_height),
            ground_z=self.ground_z,
            z1_adapter=self.z1,
            kp_dx=self.ee_kp_dx,
            kp_dz=self.ee_kp_dz,
        ), dtype=np.float64).reshape(3, 3)
        v_x = (kps[1] - kps[0]) / self.ee_kp_dx
        v_z = (kps[2] - kps[0]) / self.ee_kp_dz
        return kps[0].astype(np.float32), ee_ypr_from_axes_plb(v_x, v_z).astype(np.float32)

    def measure_ee_kp0_plb(self) -> np.ndarray:
        return self.measure_ee_pose_plb()[0]

    def high_pose_deviation(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Live EE pose vs the training HIGH: (|dpos| m, rotation angle rad, kp0, ypr)."""
        kp0, ypr = self.measure_ee_pose_plb()
        dpos = float(np.linalg.norm(kp0.astype(np.float64) - self.training_high_kp0))
        R_live = keypoint_axes_rotmat(*ypr)
        R_train = keypoint_axes_rotmat(*self.training_high_ypr)
        c = (np.trace(R_live.T @ R_train) - 1.0) * 0.5
        return dpos, math.acos(float(np.clip(c, -1.0, 1.0))), kp0, ypr

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
        """Actor frame at the CURRENT state, ref_t and a_(t-1) (Isaac order)."""
        if self.last_puck_base is None:
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
            ee_ypr_ref=self.task.ee_ypr_ref,
            prev_normalized_action=self.task.normalized_action,
            puck_pos_base=self.last_puck_base,
        )

    def run_high_level_policy(self, *, append_frame: bool) -> None:
        """obs_t -> a_t -> decode against ref_t (Isaac step / _pre_physics_step).

        The history gets exactly one frame per HL step, built BEFORE decode,
        so its previous-action term is a_(t-1) (never shifted twice).
        """
        if append_frame:
            self.hl_history.append(self.build_hl_obs_frame())
        obs = self.hl_history.flat()
        if obs.shape != (self.hl_obs_dim,) or not np.isfinite(obs).all():
            raise FloatingPointError(f"Invalid actor observation (shape {obs.shape}).")
        t0 = time.perf_counter()
        action = self.high_session.run(
            [self.high_output_name], {self.high_input_name: obs[None, :]}
        )[0][0].astype(np.float32)
        self.hl_infer_ms = 1e3 * (time.perf_counter() - t0)
        self.hl_infer_ms_max = max(self.hl_infer_ms_max, self.hl_infer_ms)
        if action.shape != (self.hl_action_dim,):
            raise RuntimeError(f"HL policy returned shape {action.shape}.")
        warned = self.task.raw_out_of_range_count > 0
        self.task.decode(action)  # raises on non-finite; clamps exactly as Isaac
        if self.task.raw_out_of_range_count > 0 and not warned:
            print(f"[TRANSPORT][WARN] HL output outside [-1,1] ({np.round(action, 4)}); clamped as in "
                  "training, but the Beta export should already be inside -- check the checkpoint.")
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
        t0 = time.perf_counter()
        action = self.low_session.run(
            [self.low_output_name], {self.low_input_name: obs[None, :]}
        )[0][0].astype(np.float32)
        self.ll_infer_ms = 1e3 * (time.perf_counter() - t0)
        self.ll_infer_ms_max = max(self.ll_infer_ms_max, self.ll_infer_ms)
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
            self.last_puck_d, self.puck_start_xy_limit, self.puck_spawn_clearance,
            self.puck_obs_noise, self.puck_rest_height,
        )
        gates = [] if ok else [gate]
        # The live EE pose must be close to the training HIGH (= reset pose):
        # a large error means a wrong arm pose, base height or FK, which would
        # also corrupt every EE observation of the run.
        dpos, dang, _, _ = self.high_pose_deviation()
        self.last_high_deviation = (dpos, dang)
        if not (dpos <= self.high_pose_max_dev_m and dang <= self.high_pose_max_dev_rad):
            gates.append(f"EE vs training HIGH: {dpos * 1e3:.1f}mm/{math.degrees(dang):.1f}deg > "
                         f"{self.high_pose_max_dev_m * 1e3:.0f}mm/{math.degrees(self.high_pose_max_dev_rad):.0f}deg")
        # The mocap B2 orientation rotates puck_base: it must agree with the IMU.
        if self.mocap_imu_max_tilt_deg is not None and (
            self.last_tilt_deg is None or not self.last_tilt_deg <= self.mocap_imu_max_tilt_deg
        ):
            gates.append(f"mocap-vs-IMU tilt {self.last_tilt_deg} deg > {self.mocap_imu_max_tilt_deg} deg")
        ok = not gates
        gate = "OK" if ok else "; ".join(gates)
        d = self.last_puck_d
        pb = self.last_puck_base
        status = (
            f"base_h={self.base_height:.4f}m | puck_base=({pb[0]:+.4f},{pb[1]:+.4f},{pb[2]:+.4f}) "
            f"| puck_D="
            f"({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f}) "
            f"| z_err={d[2] - self.puck_rest_height:+.4f}m | "
            f"EE-HIGH={dpos * 1e3:.1f}mm/{math.degrees(dang):.2f}deg "
            f"tilt(mocap-IMU)={self.last_tilt_deg if self.last_tilt_deg is None else round(self.last_tilt_deg, 2)}deg | "
            f"puck_age={((snap.get('puck') or {}).get('age_ms') or 0.0):.1f}ms"
            + f" skipped={(snap.get('mocap') or {}).get('frames_skipped', 0)}"
            + f" | gate={gate}"
        )
        return ok, status

    def initialize_policy_state_and_history(self) -> None:
        """Isaac _reset_idx / sim2sim reset(): HIGH first, base reference now,
        zero previous action, both histories filled with the reset frame."""
        self.read_robot_state()
        ready, status = self.startup_readiness()
        if not ready:
            raise RuntimeError(f"Not ready at policy initialization: {status}")

        if self.high_pose_source == "training":
            high_kp0, high_ypr = self.training_high_kp0.copy(), self.training_high_ypr.copy()
        else:
            high_kp0, high_ypr = self.measure_ee_pose_plb()
        self.task.reset(high_kp0, high_ypr, self.read_operator_base_reference())
        self.contact_monitor.reset()
        self.last_ll_action[:] = 0.0
        self.leg_target[:] = self.default_leg_pos_policy
        self.arm_target[:] = self.default_arm_pos
        self.wheel_cmd[:] = 0.0

        self.ll_history.reset(self.build_ll_obs_frame())
        self.hl_history.reset(self.build_hl_obs_frame())
        self.ll_tick = 0
        self.hl_tick = 0
        self.policy_active = True
        self._ee_buttons_prev = {b: int(self.remote_controller.button[b] == 1) for b, _ in EE_TARGET_BUTTONS}
        print(
            f"[TRANSPORT] Policy state initialized | HIGH[{self.high_pose_source}] kp0="
            + np.array2string(self.task.episode_high_kp0, precision=4)
            + " ypr=" + np.array2string(self.task.ee_ypr_ref, precision=4)
            + " base_ref=" + np.array2string(self.task.base_velocity_ref, precision=3)
            + f" | {status}"
        )

    def wbc_step(self) -> Optional[str]:
        """One 50 Hz step. Returns a protection reason, or None to continue.

        HL boundary order = Isaac step() / sim2sim hl_step():
            contact check on the completed transition -> advance references
            (ref_(t+1)) -> actor obs (ref_(t+1), a_t) + inference + decode
            -> WBC. Right after reset the filled history is used as is.
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
                # Contact proxy: logged only, never a protection trigger.
                self.last_contact = self.contact_monitor.update(self.last_puck_d)
                event = self.task.advance(self.read_operator_base_reference())
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
        self.push_visualizer_state()
        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    HL_LOG_HEADER = (
        ["t_s", "hl_tick", "ee_target", "event"]
        + [f"base_ref_{a}" for a in ("vx", "vy", "wz")]
        + [f"kp0_ref_{a}" for a in "xyz"] + ["yaw_ref", "pitch_ref", "roll_ref"]
        # raw policy output, clamped a_t (= next obs' previous action), executed residual
        + [f"raw_{i}" for i in range(6)] + [f"norm_{i}" for i in range(6)] + [f"exec_{i}" for i in range(6)]
        + [f"kp0_cmd_{a}" for a in "xyz"] + ["yaw_cmd", "pitch_cmd", "roll_cmd"]
        + [f"base_cmd_{a}" for a in ("vx", "vy", "wz")]
        # actor puck input (base frame) and the contact-proxy puck_D
        + [f"puck_base_{a}" for a in "xyz"]
        + [f"puck_D_{a}" for a in "xyz"] + ["puck_rho", "contact", "no_contact_steps"]
        + ["puck_age_ms", "body_age_ms", "body_frame", "puck_frame", "base_height", "tilt_mocap_imu_deg"]
        + [f"kp0_meas_{a}" for a in "xyz"] + ["yaw_meas", "pitch_meas", "roll_meas"]
        + ["hl_infer_ms", "ll_infer_ms"]
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
        """One row per HL step, written right after decode (cheap: no extra I/O)."""
        if self._log_writer is None:
            return

        def nan3(v: Optional[np.ndarray]) -> List[float]:
            return [float("nan")] * 3 if v is None else np.asarray(v, dtype=np.float64).tolist()

        t = self.task
        snap = self.last_snapshot or {}
        d = self.last_puck_d if self.last_puck_d is not None else np.full(3, np.nan)
        try:
            kp0_meas, ypr_meas = self.measure_ee_pose_plb()
        except Exception:  # noqa: BLE001 - logging must never fault the run
            kp0_meas, ypr_meas = np.full(3, np.nan), np.full(3, np.nan)
        body, puck = snap.get("body") or {}, snap.get("puck") or {}
        self._log_writer.writerow(
            [f"{time.monotonic() - self._t_start:.4f}", self.hl_tick,
             EE_TARGET_NAMES[t.current_target], t.last_event]
            + t.base_velocity_ref.tolist() + t.ee_kp0_ref.tolist() + t.ee_ypr_ref.tolist()
            + t.raw_action.tolist() + t.normalized_action.tolist() + t.executed_action.tolist()
            + t.kp0_cmd.tolist() + t.rot_cmd.tolist() + t.base_command.tolist()
            + nan3(self.last_puck_base)
            + d.tolist() + [float(np.hypot(d[0], d[1])), int(self.last_contact[0]), self.last_contact[1]]
            + [puck.get("age_ms"), body.get("age_ms"), body.get("frame_number"), puck.get("frame_number"),
               self.base_height, self.last_tilt_deg]
            + np.asarray(kp0_meas).tolist() + np.asarray(ypr_meas).tolist()
            + [round(self.hl_infer_ms, 3), round(self.ll_infer_ms, 3)]
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
        pb = self.last_puck_base if self.last_puck_base is not None else np.full(3, np.nan)
        print(
            f"[TRANSPORT] ll={self.ll_tick:06d} hl={self.hl_tick:05d} "
            f"tgt={EE_TARGET_NAMES[t.current_target]} | "
            f"puck_base=({pb[0]:+.3f},{pb[1]:+.3f},{pb[2]:+.3f}) "
            f"puck_D=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f}) rho={np.hypot(d[0], d[1]):.3f} "
            f"contact={int(self.last_contact[0])} nc={self.last_contact[1]} | "
            f"base_ref={np.round(t.base_velocity_ref, 2)} base_h={self.base_height:.3f}"
        )
        print(
            f"            raw={np.round(t.raw_action, 2)} a={np.round(t.normalized_action, 2)} "
            f"exec={np.round(t.executed_action, 2)} "
            f"kp0_ref={np.round(t.ee_kp0_ref, 3)} kp0_cmd={np.round(t.kp0_cmd, 3)} "
            f"ypr_cmd={np.round(t.rot_cmd, 3)}"
        )
        snap = self.last_snapshot or {}
        ages = " ".join(
            f"{k}={(snap.get(k) or {}).get('age_ms'):.1f}ms"
            for k in ("body", "puck") if (snap.get(k) or {}).get("age_ms") is not None
        )
        tilt = "n/a" if self.last_tilt_deg is None else f"{self.last_tilt_deg:.2f}"
        print(f"            mocap age {ages} skipped={(snap.get('mocap') or {}).get('frames_skipped', 0)} "
              f"tilt(mocap-IMU)={tilt}deg | infer HL {self.hl_infer_ms:.2f}ms (max {self.hl_infer_ms_max:.2f}) "
              f"WBC {self.ll_infer_ms:.2f}ms (max {self.ll_infer_ms_max:.2f})")

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
    # MuJoCo debug view (retrieval pattern)
    #
    # Kinematic replay of telemetry in the sim2sim transport scene
    # (b2wz1_disk_puck.xml): robot from mocap root + joint encoders, the
    # rendered disk mesh therefore sits where Z1 FK puts it, and the puck free
    # body is placed at its MOCAP pose. On top:
    #   translucent disc + triad  FK disk frame D (green = contact proxy on,
    #                             red = off, cyan = policy not running)
    #   magenta sphere            EE kp0 REFERENCE (operator target, min-jerk)
    #   red sphere + red/blue     decoded EE COMMAND kp0 / local +X / +Z
    #   white dot                 raw Motive pivot of the B2
    #   see-through puck          yellow centre sphere + RGB = XYZ axes (mocap pose)
    # Two producers feed one window: the control loop at 50 Hz while the
    # policy runs, and a monitor thread whenever the control loop is quiet
    # (startup holds, READY-WAIT, damping protection). Steps no physics.
    # ------------------------------------------------------------------

    def _build_visualizer(self):
        cfg = self.cfg
        if not bool(cfg.get("visualizer_enabled", False)):
            return None
        from utils.mj_visualizer import MujocoDebugVisualizer  # noqa: PLC0415

        visualizer = MujocoDebugVisualizer(
            xml_path=self._resolve_path(cfg["visualizer_xml_path"]),
            update_hz=float(cfg.get("visualizer_update_hz", 30.0)),
            ee_target_sphere_radius=float(cfg.get("visualizer_ee_target_sphere_radius", 0.025)),
            ee_target_axis_len=float(cfg.get("visualizer_ee_target_axis_len", 0.20)),
            ee_target_axis_radius=float(cfg.get("visualizer_ee_target_axis_radius", 0.008)),
            show_gripper_center=False,
            # The transport scene MJCF already has its own floor and light.
            show_floor=bool(cfg.get("visualizer_show_floor", False)),
            floor_half_extent=float(cfg.get("visualizer_floor_half_extent", 3.0)),
            show_light=bool(cfg.get("visualizer_show_light", False)),
            ground_z=self.ground_z,
            free_joint_names=[VIZ_PUCK_FREE_JOINT],
            geom_alpha={VIZ_PUCK_GEOM: float(cfg.get("visualizer_puck_alpha", 0.45))},
        )
        visualizer.configure_world_view(
            show_world_origin=bool(cfg.get("visualizer_show_world_origin", True)),
            camera_follow=bool(cfg.get("visualizer_camera_follow", True)),
            world_origin_axis_len=float(cfg.get("visualizer_world_origin_axis_len", 0.4)),
            camera_follow_smoothing=float(cfg.get("visualizer_camera_follow_smoothing", 0.15)),
            show_all_markers=bool(cfg.get("visualizer_show_all_markers", True)),
            marker_radius=float(cfg.get("visualizer_marker_radius", 0.008)),
            show_root_frame=bool(cfg.get("visualizer_show_root_frame", True)),
            root_frame_axis_len=float(cfg.get("visualizer_root_frame_axis_len", 0.25)),
            root_frame_axis_radius=float(cfg.get("visualizer_root_frame_axis_radius", 0.008)),
        )
        self.viz_show_pivots = bool(cfg.get("visualizer_show_mocap_pivots", True))
        self.viz_show_puck_frame = bool(cfg.get("visualizer_show_puck_frame", True))
        self.viz_puck_frame_axis_len = float(cfg.get("visualizer_puck_frame_axis_len", 0.09))
        self.viz_puck_frame_axis_radius = float(cfg.get("visualizer_puck_frame_axis_radius", 0.003))
        self.viz_puck_center_radius = float(cfg.get("visualizer_puck_center_radius", 0.008))
        self.viz_show_markers = bool(cfg.get("visualizer_show_all_markers", False))
        self.viz_push_period_s = 1.0 / max(float(cfg.get("visualizer_update_hz", 30.0)), 1.0)
        self.viz_pivot_radius = float(cfg.get("visualizer_mocap_pivot_radius", 0.012))
        self.viz_ee_ref_radius = float(cfg.get("visualizer_ee_ref_radius", 0.03))
        self.monitor_enabled = bool(cfg.get("visualizer_monitor_enabled", True))
        self.monitor_hz = float(cfg.get("visualizer_monitor_hz", 30.0))
        self.monitor_takeover_s = float(cfg.get("visualizer_monitor_takeover_s", 0.25))
        return visualizer

    def _gripper_q_training_for_viz(self) -> float:
        getter = getattr(self.z1, "get_gripper_q_training", None)
        try:
            return float(getter()) if getter is not None else float(self.gripper_hold_pos)
        except Exception:  # noqa: BLE001
            return float(self.gripper_hold_pos)

    def _scene_state(
        self, snap: Dict[str, Any], b2w_joint_pos: np.ndarray, imu_quat: np.ndarray, with_task: bool
    ) -> Dict[str, Any]:
        """Render snapshot. World-frame drawing uses the MOCAP root pose."""
        state: Dict[str, Any] = {
            "base_quat_wxyz": np.asarray(imu_quat, dtype=np.float32).copy(),
            "base_height": float(self.base_height or 0.0),
            "ground_z": self.ground_z,
            "b2w_joint_pos_policy": np.asarray(b2w_joint_pos, dtype=np.float32).copy(),
            "z1_q": np.asarray(self.z1.q, dtype=np.float32).copy(),
            "gripper_q_training": self._gripper_q_training_for_viz(),
        }
        body = snap.get("body") or {}
        root_ok = bool(body.get("valid")) and body.get("root_position_world") is not None
        if root_ok:
            p_root = np.asarray(body["root_position_world"], dtype=np.float64)
            q_root = np.asarray(body["root_quat_wxyz"], dtype=np.float32)
            state.update(base_pos_w=p_root, base_quat_wxyz=q_root,
                         root_frame={"pos": p_root.tolist(), "quat_wxyz": q_root.tolist()})
            h = (snap.get("base_height") or {}).get("height_m")
            if h is not None:
                state["base_height"] = float(h)

        puck = snap.get("puck") or {}
        if puck.get("valid") and puck.get("position_world") is not None:
            state["free_body_poses"] = {
                VIZ_PUCK_FREE_JOINT: {"pos": puck["position_world"], "quat_wxyz": puck.get("quat_wxyz")}
            }

        if self.last_fk_disk_w is not None:
            fk_pos_w, fk_rot_w = self.last_fk_disk_w
            if not self.policy_active:
                rgba = [0.2, 0.9, 1.0, 0.35]
            elif self.last_contact[0]:
                rgba = [0.2, 1.0, 0.3, 0.40]
            else:
                rgba = [1.0, 0.2, 0.2, 0.55]
            state["discs"] = [{
                "pos": fk_pos_w.tolist(), "mat": np.asarray(fk_rot_w, dtype=np.float64),
                "radius": self.disk.radius, "half_height": 0.5 * self.disk.thickness + 0.001,
                "rgba": rgba,
            }]
            state["extra_frames"] = [{"pos": fk_pos_w.tolist(),
                                      "quat_wxyz": quat_from_rotmat_wxyz(fk_rot_w).tolist(),
                                      "length": 0.12, "radius": 0.004}]

        points = []
        # Puck centre + axes (mocap pose; RGB = XYZ of the Motive puck asset).
        if (getattr(self, "viz_show_puck_frame", False) and puck.get("valid")
                and puck.get("position_world") is not None and puck.get("quat_wxyz") is not None):
            state.setdefault("extra_frames", []).append({
                "pos": puck["position_world"], "quat_wxyz": puck["quat_wxyz"],
                "length": self.viz_puck_frame_axis_len,
                "radius": self.viz_puck_frame_axis_radius,
            })
            points.append({"pos": puck["position_world"], "rgba": [1.0, 1.0, 0.0, 1.0],
                           "radius": self.viz_puck_center_radius})
        if getattr(self, "viz_show_pivots", False):
            for channel, rgba in (("body", [1.0, 1.0, 1.0, 0.85]),
                                  ("puck", [1.0, 0.55, 0.1, 0.9])):
                if channel == "puck" and getattr(self, "viz_show_puck_frame", False):
                    continue  # same point as the yellow puck-centre sphere
                ch = snap.get(channel) or {}
                pos = ch.get("position_world")
                if ch.get("valid") and pos is not None:
                    points.append({"pos": pos, "rgba": rgba,
                                   "radius": getattr(self, "viz_pivot_radius", 0.012)})

        if with_task and root_ok:
            state["ee_cmd_plb"] = np.asarray(self.task.ee_cmd_plb, dtype=np.float32).copy()
            # EE kp0 reference (PLB) -> world, with the same PLB the
            # visualizer uses for the command: mocap root xy, ground z, yaw.
            _, _, yaw = euler_xyz_from_quat_wxyz(state["base_quat_wxyz"])
            plb_origin = np.array([state["base_pos_w"][0], state["base_pos_w"][1], self.ground_z],
                                  dtype=np.float32)
            ref_w = plb_origin + quat_apply_wxyz(
                quat_from_yaw_wxyz(float(yaw)), np.asarray(self.task.ee_kp0_ref, dtype=np.float32)
            )
            points.append({"pos": ref_w.tolist(), "rgba": [1.0, 0.0, 1.0, 0.8],
                           "radius": getattr(self, "viz_ee_ref_radius", 0.03)})
        state["mocap_points"] = points or None
        # Raw marker cloud only when it is drawn: converting it is the most
        # expensive part of a snapshot and nothing in the control path needs it.
        marker_cloud = getattr(self.perception, "marker_cloud", None)
        if getattr(self, "viz_show_markers", False) and marker_cloud is not None:
            state["mocap_markers"] = marker_cloud()
        return state

    def push_visualizer_state(self) -> None:
        """From the control loop: the rich state (task + EE command).

        Throttled to the viewer's own update rate: building the scene state
        at the 50-Hz WBC rate only to have the viewer drop 2 of every 5
        would cost control-thread time (and GIL) for nothing.
        """
        if self.visualizer is None or self.last_snapshot is None:
            return
        now = time.monotonic()
        if now - self._last_control_push < self.viz_push_period_s:
            return
        try:
            self.visualizer.push_state(**self._scene_state(
                self.last_snapshot, self.b2w_joint_pos, self.base_quat_wxyz, with_task=True
            ))
            self._last_control_push = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - a debug view must never fault the run
            if not self._monitor_warned:
                self._monitor_warned = True
                print(f"[VIZ][WARN] push failed, view may freeze: {exc!r}")

    def start_monitor(self) -> None:
        if self.visualizer is None:
            return
        self.visualizer.start()
        if not self.monitor_enabled or self._monitor_thread is not None:
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, name="transport-scene-monitor", daemon=True
        )
        self._monitor_thread.start()
        print(f"[TRANSPORT] MuJoCo view + scene monitor at {self.monitor_hz:g} Hz.")

    def stop_monitor(self) -> None:
        if getattr(self, "_monitor_thread", None) is not None:
            self._monitor_stop.set()
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        if getattr(self, "visualizer", None) is not None:
            try:
                self.visualizer.stop()
            except Exception as exc:  # noqa: BLE001
                print(f"[EXIT][WARN] visualizer stop failed: {exc!r}")

    def _monitor_loop(self) -> None:
        period = 1.0 / max(self.monitor_hz, 1.0)
        while not self._monitor_stop.is_set():
            loop_start = time.perf_counter()
            try:
                if time.monotonic() - self._last_control_push >= self.monitor_takeover_s:
                    state = self._monitor_state()
                    if state is not None:
                        self.visualizer.push_state(**state)
            except Exception as exc:  # noqa: BLE001 - a debug view must never kill a run
                if not self._monitor_warned:
                    self._monitor_warned = True
                    print(f"[MONITOR][WARN] disabled after error: {exc!r}")
                return
            remaining = period - (time.perf_counter() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)

    def _monitor_state(self) -> Optional[Dict[str, Any]]:
        """Phase-independent telemetry: low_state (DDS callback) + a fresh mocap read."""
        low = self.low_state
        if low is None or getattr(low, "tick", 0) == 0:
            return None
        joints = np.array(
            [low.motor_state[self.hardware_to_policy[i]].q for i in range(self.num_b2w_dof)],
            dtype=np.float32,
        )
        quat = np.asarray(low.imu_state.quaternion, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(quat))
        quat = quat / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self._scene_state(self.perception.get_latest_snapshot(), joints, quat, with_task=False)

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
        last_print = 0.0
        while self.remote_controller.button[KeyMap.A] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_b2w_cmd()
            self._hold_arm_default_lowcmd()
            now = time.monotonic()
            if now - last_print >= self.perception_status_print_s:
                last_print = now
                _, status = self.startup_readiness()
                print(f"[ARM-HOLD] {status}")
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

        Ready = valid base + puck mocap (one NatNet frame) AND puck_D within
        puck_start_xy_limit_m of the disk centre at resting height (+/- obs
        noise) AND live EE pose within high_pose_max_deviation
        of the training HIGH AND mocap-vs-IMU tilt within mocap_imu_max_tilt_deg.
        Re-checked on a fresh snapshot at the A press.
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
        print(f"High policy : {self.high_policy_path}")
        print(f"Low policy  : {self.low_policy_path}")
        print("UAN         : NOT LOADED (simulation-only); Z1 firmware POSITION_PD")
        print(f"Z1 gains    : Kp={np.round(self.z1_kp_effective, 3)} Kd={np.round(self.z1_kd_effective, 3)}")
        print("Actor obs   : NO-SETTLE semantics | prev action = clamped a_(t-1) | "
              "puck = R_root^T(puck - root), B2+puck from one NatNet frame")
        print(f"HIGH pose   : {self.high_pose_source} | training kp0={np.round(self.training_high_kp0, 4).tolist()} "
              f"ypr={np.round(self.training_high_ypr, 4).tolist()} | READY gate "
              f"{self.high_pose_max_dev_m * 1e3:.0f}mm/{math.degrees(self.high_pose_max_dev_rad):.0f}deg, "
              f"mocap-IMU tilt <= {self.mocap_imu_max_tilt_deg} deg")
        print(f"Puck        : r={self.puck_radius} h={self.puck_height} (rest z_D={self.puck_rest_height:.3f})")
        print(f"Disk        : r={self.disk.radius} t={self.disk.thickness}, frame D = Z1 FK + training mount")
        for sel in ("body", "puck_body"):
            selector = getattr(self.perception, sel, None)
            if isinstance(selector, MocapBodySelector):
                print(f"  mocap     : {selector.describe()}")
        print(f"Visualizer  : {'ON' if self.visualizer is not None else 'OFF'}")
        print(f"Contact     : rho<={self.disk.radius} and |z_D-{self.puck_rest_height:.3f}|<="
              f"{self.puck_obs_noise}; LOG ONLY (no puck-drop protection)")
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
            self.start_monitor()

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
            self.stop_monitor()
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
        cfg = yaml.safe_load(f)
    problems = validate_transport_setup(cfg)
    if problems:
        print("[TRANSPORT] Setup incomplete -- nothing was started:\n  - " + "\n  - ".join(problems))
        sys.exit(2)

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: PLC0415

    # As retrieval: the 500-Hz loop shares the GIL with the NatNet decoder,
    # the DDS callback, the MuJoCo view and the scene monitor. 0.5 ms bounds
    # a GIL hold to a quarter of one hardware tick.
    sys.setswitchinterval(5e-4)
    ChannelFactoryInitialize(0, args.net)
    B2WZ1TransportMocapController(cfg_path=args.config).run()


if __name__ == "__main__":
    main()
