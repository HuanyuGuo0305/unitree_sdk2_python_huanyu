from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from perception.ground_plane_estimator import (
    FX_D,
    FY_D,
    CX_D,
    CY_D,
    R_BASE_DEPTH,
    T_BASE_DEPTH,
)
from perception.rgb_ground_localizer import (
    K_RGB,
    D_RGB,
)


# Measured ChArUco extrinsic: T_{RGB<-Depth}
R_RGB_DEPTH = np.array(
    [
        [0.99998724, -0.00456467, 0.00216254],
        [0.00166047, 0.70142670, 0.71273966],
        [-0.00477028, -0.71272698, 0.70142533],
    ],
    dtype=np.float64,
)

T_RGB_DEPTH = np.array(
    [-0.02538479, -0.08406990, 0.00193342],
    dtype=np.float64,
)


@dataclass
class RGBDLocalization:
    valid: bool
    position_base: np.ndarray
    num_mask_points: int
    num_inliers: int
    spread_m: float


class RGBDObjectLocalizer:
    """
    Near mode:
        Depth -> RGB projection
        -> HSV inner-mask selection
        -> median + MAD
        -> direct real RGBD object point in Base.

    No artificial z offset is applied in RGBD mode.
    """

    def __init__(
        self,
        stride: int = 2,
        min_points: int = 50,
        min_depth_m: float = 0.15,
        max_depth_m: float = 2.5,
        mad_scale: float = 3.5,
        max_spread_m: float = 0.30,
    ):
        self.stride = int(stride)
        self.min_points = int(min_points)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.mad_scale = float(mad_scale)
        self.max_spread_m = float(max_spread_m)

    def localize(
        self,
        depth_raw: np.ndarray,
        depth_scale: float,
        inner_mask: np.ndarray,
    ) -> RGBDLocalization:

        h_d, w_d = depth_raw.shape
        h_rgb, w_rgb = inner_mask.shape
        s = self.stride

        # 1) Depth pixels -> p_Depth.
        vv, uu = np.mgrid[
            0:h_d:s,
            0:w_d:s,
        ]

        z = (
            depth_raw[0:h_d:s, 0:w_d:s].astype(np.float64)
            * float(depth_scale)
        )

        valid = (
            np.isfinite(z)
            & (z > self.min_depth_m)
            & (z < self.max_depth_m)
        )

        if not np.any(valid):
            return self._invalid()

        u_d = uu[valid].astype(np.float64)
        v_d = vv[valid].astype(np.float64)
        z = z[valid]

        x = (u_d - CX_D) / FX_D * z
        y = (v_d - CY_D) / FY_D * z

        points_depth = np.column_stack(
            (x, y, z)
        )

        # 2) p_Depth -> p_RGB.
        points_rgb = (
            points_depth @ R_RGB_DEPTH.T
            + T_RGB_DEPTH
        )

        front = points_rgb[:, 2] > 0.02

        points_rgb = points_rgb[front]
        points_depth = points_depth[front]

        if len(points_rgb) == 0:
            return self._invalid()

        # 3) p_RGB -> raw RGB pixels.
        projected, _ = cv2.projectPoints(
            points_rgb.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            K_RGB,
            D_RGB,
        )

        uv = projected.reshape(-1, 2)

        u = np.rint(
            uv[:, 0]
        ).astype(np.int32)

        v = np.rint(
            uv[:, 1]
        ).astype(np.int32)

        inside = (
            (u >= 0)
            & (u < w_rgb)
            & (v >= 0)
            & (v < h_rgb)
        )

        u = u[inside]
        v = v[inside]
        points_depth = points_depth[inside]

        # 4) HSV inner-mask selection.
        select = (
            inner_mask[v, u] > 0
        )

        points = points_depth[select]

        num_mask_points = len(points)

        if num_mask_points < self.min_points:
            return self._invalid(
                num_mask_points=num_mask_points,
            )

        # 5) Median + radial MAD filtering.
        center0 = np.median(
            points,
            axis=0,
        )

        radius = np.linalg.norm(
            points - center0,
            axis=1,
        )

        r_med = float(
            np.median(radius)
        )

        r_mad = float(
            np.median(
                np.abs(radius - r_med)
            )
        )

        threshold = (
            r_med
            + self.mad_scale
            * max(r_mad, 0.005)
        )

        inliers = points[
            radius <= threshold
        ]

        num_inliers = len(inliers)

        if num_inliers < self.min_points:
            return self._invalid(
                num_mask_points=num_mask_points,
                num_inliers=num_inliers,
            )

        position_depth = np.median(
            inliers,
            axis=0,
        )

        spread_m = float(
            np.median(
                np.linalg.norm(
                    inliers - position_depth,
                    axis=1,
                )
            )
        )

        if spread_m > self.max_spread_m:
            return self._invalid(
                num_mask_points=num_mask_points,
                num_inliers=num_inliers,
            )

        # Direct real RGBD point.
        position_base = (
            R_BASE_DEPTH @ position_depth
            + T_BASE_DEPTH
        )

        return RGBDLocalization(
            valid=True,
            position_base=position_base,
            num_mask_points=num_mask_points,
            num_inliers=num_inliers,
            spread_m=spread_m,
        )

    @staticmethod
    def _invalid(
        num_mask_points: int = 0,
        num_inliers: int = 0,
    ) -> RGBDLocalization:
        return RGBDLocalization(
            valid=False,
            position_base=np.full(
                3,
                np.nan,
                dtype=np.float64,
            ),
            num_mask_points=int(
                num_mask_points
            ),
            num_inliers=int(
                num_inliers
            ),
            spread_m=float("nan"),
        )
