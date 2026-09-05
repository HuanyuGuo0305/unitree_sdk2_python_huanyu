from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from perception.data_types import GroundPlane


K_RGB = np.array(
    [
        [722.48999675, 0.0, 954.61641613],
        [0.0, 720.65884236, 544.58259268],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

D_RGB = np.array(
    [
        0.01388045,
        -0.06327465,
        0.00097771,
        0.00019967,
        0.01539046,
    ],
    dtype=np.float64,
)

T_BASE_RGB = np.array(
    [0.3993, 0.0, -0.01576],
    dtype=np.float64,
)


def _rpy_to_rotmat(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [[1, 0, 0], [0, cr, -sr], [0, sr, cr]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]],
        dtype=np.float64,
    )

    return rz @ ry @ rx


R_BASE_RGB = _rpy_to_rotmat(
    -1.5708,
    0.0,
    -1.5708,
)


@dataclass
class RGBGroundLocalization:
    valid: bool
    position_base: np.ndarray
    range_m: float


class RGBGroundLocalizer:
    """
    RGB_GROUND mode.

    Use HSV center_uv to form an RGB ray, transform it into Base,
    then intersect it directly with the plane 3.5 cm above live ground.

    Ground plane:
        n^T p + d = 0

    Target-height plane:
        n^T p + d = object_height_m
    """

    def __init__(
        self,
        object_height_m: float = 0.035,
        min_range_m: float = 0.10,
        max_range_m: float = 5.0,
    ):
        self.object_height_m = float(object_height_m)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)

    def localize(
        self,
        center_uv: tuple[float, float],
        ground_plane: GroundPlane,
    ) -> RGBGroundLocalization:

        if not ground_plane.valid:
            return self._invalid()

        u, v = center_uv

        if not (np.isfinite(u) and np.isfinite(v)):
            return self._invalid()

        # Raw RGB pixel -> normalized optical ray.
        pixel = np.array(
            [[[u, v]]],
            dtype=np.float64,
        )

        xy = cv2.undistortPoints(
            pixel,
            K_RGB,
            D_RGB,
        )[0, 0]

        ray_rgb = np.array(
            [xy[0], xy[1], 1.0],
            dtype=np.float64,
        )
        ray_rgb /= np.linalg.norm(ray_rgb)

        # T_{Base<-RGB}: ray and camera origin in Base.
        ray_base = R_BASE_RGB @ ray_rgb
        ray_base /= np.linalg.norm(ray_base)

        origin_base = T_BASE_RGB

        n = ground_plane.normal_base
        d = float(ground_plane.d_base)

        denom = float(n @ ray_base)

        if abs(denom) < 1e-5:
            return self._invalid()

        distance = (
            self.object_height_m
            - float(n @ origin_base + d)
        ) / denom

        if (
            not np.isfinite(distance)
            or distance < self.min_range_m
            or distance > self.max_range_m
        ):
            return self._invalid()

        position_base = (
            origin_base
            + distance * ray_base
        )

        return RGBGroundLocalization(
            valid=True,
            position_base=position_base,
            range_m=float(distance),
        )

    @staticmethod
    def _invalid() -> RGBGroundLocalization:
        return RGBGroundLocalization(
            valid=False,
            position_base=np.full(
                3,
                np.nan,
                dtype=np.float64,
            ),
            range_m=float("nan"),
        )