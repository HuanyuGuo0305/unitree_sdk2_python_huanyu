from dataclasses import dataclass

import numpy as np


@dataclass
class RGBFrame:
    image_bgr: np.ndarray
    timestamp_s: float


@dataclass
class DepthFrame:
    image_raw: np.ndarray
    timestamp_s: float
    depth_scale: float
    sequence: int


@dataclass
class GroundPlane:
    # Plane in Base frame:
    #     normal_base^T p_base + d_base = 0
    normal_base: np.ndarray
    d_base: float
    valid: bool
    inlier_ratio: float
    timestamp_s: float


@dataclass
class ObjectState:
    position_base: np.ndarray
    valid: bool
    mode: str                  # "RGB_GROUND", "RGBD", "INVALID"
    timestamp_s: float
    num_depth_points: int = 0
