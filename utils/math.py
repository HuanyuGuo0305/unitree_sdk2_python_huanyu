import numpy as np


def quat_unique_wxyz(q: np.ndarray) -> np.ndarray:
    """Ensure quaternion has non-negative w for uniqueness."""
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    return (-q if q[0] < 0.0 else q).astype(np.float32)


def quat_normalize_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize quaternion in wxyz format."""
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    n = float(np.linalg.norm(q))
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def quat_conjugate_wxyz(q: np.ndarray) -> np.ndarray:
    """Conjugate of wxyz quaternion."""
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float32)


quat_conj_wxyz = quat_conjugate_wxyz


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both in wxyz."""
    q1 = np.asarray(q1, dtype=np.float32).reshape(4,)
    q2 = np.asarray(q2, dtype=np.float32).reshape(4,)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_apply_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz)."""
    q = quat_normalize_wxyz(q)
    v = np.asarray(v, dtype=np.float32).reshape(3,)
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul_wxyz(quat_mul_wxyz(q, qv), quat_conjugate_wxyz(q))[1:4]


def quat_apply_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by inverse(q)."""
    q = quat_normalize_wxyz(q)
    return quat_apply_wxyz(quat_conjugate_wxyz(q), v)


def quat_rotate_inverse_numpy(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """
    Rotate vector by inverse quaternion.

    Computes:
        v' = q^{-1} * v * q

    Args:
        quat: quaternion [w, x, y, z] (unit, wxyz)
        vec:  vector [x, y, z]

    Returns:
        Rotated vector in same frame.
    """
    quat = quat_unique_wxyz(np.asarray(quat, dtype=np.float32).reshape(4,))
    vec = np.asarray(vec, dtype=np.float32).reshape(3,)
    return quat_apply_inverse_wxyz(quat, vec).astype(np.float32)


def euler_xyz_from_quat_wxyz(q: np.ndarray):
    """Return roll, pitch, yaw from wxyz quaternion."""
    q = quat_normalize_wxyz(q)
    w, x, y, z = q

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return float(roll), float(pitch), float(yaw)


def quat_from_yaw_wxyz(yaw: float) -> np.ndarray:
    """Yaw-only quaternion in wxyz."""
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize a vector."""
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def quat_from_rotmat_wxyz(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert rotation matrix to quaternion (wxyz)."""
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0.0:
        s = np.sqrt(max(tr + 1.0, eps)) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (m00 > m11) and (m00 > m22):
        s = np.sqrt(max(1.0 + m00 - m11 - m22, eps)) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(max(1.0 + m11 - m00 - m22, eps)) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(max(1.0 + m22 - m00 - m11, eps)) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float32)
    return quat_unique_wxyz(quat_normalize_wxyz(q, eps))


def quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float, eps: float = 1e-8) -> np.ndarray:
    """Slerp between two quaternions q0 -> q1, both in wxyz."""
    q0 = quat_normalize_wxyz(q0, eps)
    q1 = quat_normalize_wxyz(q1, eps)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))

    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return quat_unique_wxyz(quat_normalize_wxyz(out, eps))

    omega = float(np.arccos(dot))
    sin_omega = max(float(np.sin(omega)), eps)

    w0 = float(np.sin((1.0 - t) * omega) / sin_omega)
    w1 = float(np.sin(t * omega) / sin_omega)
    out = w0 * q0 + w1 * q1
    return quat_unique_wxyz(quat_normalize_wxyz(out, eps))


def quat_angle_wxyz(q0: np.ndarray, q1: np.ndarray) -> float:
    """Relative rotation angle in rad using shortest path."""
    q0 = quat_normalize_wxyz(q0)
    q1 = quat_normalize_wxyz(q1)
    dot = abs(float(np.dot(q0, q1)))
    dot = float(np.clip(dot, 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def quat_from_keypoints_lb(kp0: np.ndarray, kp1: np.ndarray, kp2: np.ndarray, dx: float, dz: float) -> np.ndarray:
    """
    Recover orientation from keypoints defined in LB:
      kp1 = kp0 + R*[dx,0,0]
      kp2 = kp0 + R*[0,0,dz]
    """
    x_axis = normalize((kp1 - kp0) / max(dx, 1e-8))
    z_axis = normalize((kp2 - kp0) / max(dz, 1e-8))
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns
    return quat_from_rotmat_wxyz(R)