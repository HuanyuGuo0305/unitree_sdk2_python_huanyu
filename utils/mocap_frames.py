"""
Shared float64 rigid-transform maths for the mocap tooling.

Two consumers depend on exactly the same convention, so the maths lives in
one place rather than being written twice:

    deploy/b2w_mocap_root_calibration.py   solves the mocap -> root offset
    utils/mocap_perception.py              applies it at 50 Hz

Everything here is float64. utils.math is float32 throughout, which is fine
for control but throws away roughly a milliradian of resolution -- enough to
matter when the whole point is resolving a sub-degree mounting rotation.

Quaternions are wxyz. Euler angles follow utils.math's convention,
R = Rz(yaw) . Ry(pitch) . Rx(roll).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import yaml


def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def q_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(a, dtype=np.float64).reshape(4)
    w2, x2, y2, z2 = np.asarray(b, dtype=np.float64).reshape(4)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def q_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def q_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return q_to_mat(q) @ np.asarray(v, dtype=np.float64).reshape(3)


def q_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return q_to_mat(q).T @ np.asarray(v, dtype=np.float64).reshape(3)


def q_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(axis))
    if n < 1.0e-12 or abs(angle) < 1.0e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = axis / n
    half = 0.5 * float(angle)
    return np.concatenate([[np.cos(half)], np.sin(half) * axis])


def q_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) . Ry(pitch) . Rx(roll), matching utils.math's RPY convention."""
    qz = q_from_axis_angle([0.0, 0.0, 1.0], yaw)
    qy = q_from_axis_angle([0.0, 1.0, 0.0], pitch)
    qx = q_from_axis_angle([1.0, 0.0, 0.0], roll)
    return q_mul(q_mul(qz, qy), qx)


def rpy_from_q(q: np.ndarray) -> Tuple[float, float, float]:
    w, x, y, z = q_normalize(q)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def q_log(q: np.ndarray) -> np.ndarray:
    """Quaternion -> rotation vector (axis * angle)."""
    q = q_normalize(q)
    if q[0] < 0.0:
        q = -q
    vec_norm = float(np.linalg.norm(q[1:4]))
    if vec_norm < 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vec_norm, float(q[0]))
    return (angle / vec_norm) * q[1:4]


def q_exp(rotvec: np.ndarray) -> np.ndarray:
    """Rotation vector -> quaternion."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q_from_axis_angle(rotvec / angle, angle)


def q_between_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking `v_from` onto `v_to`."""
    a = np.asarray(v_from, dtype=np.float64).reshape(3)
    b = np.asarray(v_to, dtype=np.float64).reshape(3)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1.0e-12 or nb < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    a, b = a / na, b / nb

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1.0e-12:
        # Antiparallel: any perpendicular axis is a valid 180 deg rotation.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if float(np.linalg.norm(axis)) < 1.0e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return q_from_axis_angle(axis, np.pi)

    return q_from_axis_angle(np.cross(a, b), np.arccos(dot))


def wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


class RootOffset:
    """Constant mocap-rigid-body -> base_link transform."""

    def __init__(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        self.pos = np.asarray(pos, dtype=np.float64).reshape(3).copy()
        self.quat = q_normalize(quat_wxyz)

    @classmethod
    def identity(cls) -> "RootOffset":
        """No offset: the mocap rigid body's own frame is used as the root."""
        return cls(np.zeros(3), [1.0, 0.0, 0.0, 0.0])

    def is_identity(self, pos_tol: float = 1e-9, rot_tol: float = 1e-9) -> bool:
        return bool(
            np.all(np.abs(self.pos) <= pos_tol)
            and abs(abs(float(self.quat[0])) - 1.0) <= rot_tol
        )

    @classmethod
    def from_rpy(cls, pos, rpy) -> "RootOffset":
        r, p, y = (float(v) for v in rpy)
        return cls(pos, q_from_rpy(r, p, y))

    def copy(self) -> "RootOffset":
        return RootOffset(self.pos, self.quat)

    def rpy(self) -> Tuple[float, float, float]:
        return rpy_from_q(self.quat)

    def apply(self, p_mocap: np.ndarray, q_mocap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct the root pose in the mocap world frame."""
        p_mocap = np.asarray(p_mocap, dtype=np.float64).reshape(3)
        p_root = p_mocap + q_apply(q_mocap, self.pos)
        q_root = q_normalize(q_mul(q_mocap, self.quat))
        return p_root, q_root

    def inverse(self) -> Tuple[np.ndarray, np.ndarray]:
        """The root -> mocap-rigid-body transform, for the reverse lookup."""
        q_inv = q_conj(self.quat)
        return -q_apply(q_inv, self.pos), q_inv

    def translate_local(self, delta: np.ndarray) -> None:
        self.pos += np.asarray(delta, dtype=np.float64).reshape(3)

    def translate_world(self, delta_world: np.ndarray, q_mocap: np.ndarray) -> None:
        self.pos += q_apply_inv(q_mocap, delta_world)

    def rotate_local(self, delta_quat: np.ndarray) -> None:
        """Pre-multiply: the correction is expressed in the mocap body frame."""
        self.quat = q_normalize(q_mul(delta_quat, self.quat))

    def __repr__(self) -> str:
        r, p, y = self.rpy()
        return (
            f"RootOffset(pos=[{self.pos[0]:+.4f}, {self.pos[1]:+.4f}, {self.pos[2]:+.4f}], "
            f"rpy_deg=[{np.degrees(r):+.3f}, {np.degrees(p):+.3f}, {np.degrees(y):+.3f}])"
        )


def load_root_offset(source: Any) -> RootOffset:
    """
    Load the offset written by deploy/b2w_mocap_root_calibration.py.

    Accepts the path to that YAML, an already-loaded dict, or a RootOffset
    (returned unchanged). Quaternion form is preferred; an `..._rpy` entry is
    accepted as a fallback so a hand-written offset is also usable.

    `None` means "no calibration": the identity offset, which makes the mocap
    rigid body's own frame the root frame. A path that is given but missing is
    still an error, so a typo is not silently downgraded to identity.
    """
    if source is None:
        return RootOffset.identity()

    if isinstance(source, RootOffset):
        return source

    if isinstance(source, dict):
        data: Dict[str, Any] = source
        origin = "<dict>"
    else:
        path = os.path.abspath(str(source))
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"mocap root offset file not found: {path}. Run "
                "deploy/b2w_mocap_root_calibration.py to produce it."
            )
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        origin = path

    if not isinstance(data, dict):
        raise ValueError(f"mocap root offset in {origin} is not a mapping")

    pos = data.get("mocap_root_offset_pos")
    if pos is None:
        raise ValueError(f"{origin} has no 'mocap_root_offset_pos'")

    quat = data.get("mocap_root_offset_quat_wxyz")
    if quat is not None:
        return RootOffset(pos, quat)

    rpy = data.get("mocap_root_offset_rpy")
    if rpy is None:
        raise ValueError(
            f"{origin} has neither 'mocap_root_offset_quat_wxyz' nor "
            "'mocap_root_offset_rpy'"
        )
    return RootOffset.from_rpy(pos, rpy)
