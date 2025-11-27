import numpy as np


def quat_rotate_inverse_numpy(quat, vec):
    """
    Rotate vector by inverse quaternion (NumPy implementation).
    Args:
        quat: quaternion [w, x, y, z] - shape (4,)
        vec: vector [x, y, z] - shape (3,)
    Returns:
        Rotated vector - shape (3,)
    """
    qw, qx, qy, qz = quat
    vx, vy, vz = vec
    
    # Quaternion conjugate rotation: q^-1 * v * q
    # Simplified formula for rotating a vector
    t_x = 2.0 * (qy * vz - qz * vy)
    t_y = 2.0 * (qz * vx - qx * vz)
    t_z = 2.0 * (qx * vy - qy * vx)
    
    rx = vx + qw * t_x + qy * t_z - qz * t_y
    ry = vy + qw * t_y + qz * t_x - qx * t_z
    rz = vz + qw * t_z + qx * t_y - qy * t_x
    
    return np.array([rx, ry, rz], dtype=np.float32)