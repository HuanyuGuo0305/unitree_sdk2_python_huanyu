from __future__ import annotations

import math

import numpy as np

from perception.data_types import DepthFrame, GroundPlane


# D430I factory intrinsics, 848x480.
FX_D = 430.12475586
FY_D = 430.12475586
CX_D = 430.40548706
CY_D = 233.43594360


# B2WZ1 URDF: T_{Base<-Depth}
T_BASE_DEPTH = np.array(
    [0.42161, 0.025, 0.061851],
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


R_BASE_DEPTH = _rpy_to_rotmat(
    -2.3562,
    0.0,
    -1.5708,
)


class GroundPlaneEstimator:
    """
    D430I depth -> live ground plane in Base frame.

    Plane:
        n^T p + d = 0
    """

    def __init__(
        self,
        stride: int = 6,
        min_depth_m: float = 0.15,
        max_depth_m: float = 2.5,
        distance_threshold_m: float = 0.015,
        ransac_iterations: int = 300,
        max_tilt_deg: float = 50.0,
        min_inlier_ratio: float = 0.30,
    ):
        self.stride = int(stride)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.distance_threshold_m = float(
            distance_threshold_m
        )
        self.ransac_iterations = int(
            ransac_iterations
        )
        self.min_nz = math.cos(
            math.radians(max_tilt_deg)
        )
        self.min_inlier_ratio = float(
            min_inlier_ratio
        )

        self._rng = np.random.default_rng(0)

    def estimate(
        self,
        frame: DepthFrame,
    ) -> GroundPlane:

        depth_m = (
            frame.image_raw.astype(np.float64)
            * float(frame.depth_scale)
        )

        points = self._depth_to_base_points(
            depth_m
        )

        if len(points) < 100:
            return self._invalid(frame.timestamp_s)

        result = self._fit_ransac(points)

        if result is None:
            return self._invalid(frame.timestamp_s)

        normal, d, inlier_mask = result
        inlier_ratio = float(
            np.mean(inlier_mask)
        )

        if inlier_ratio < self.min_inlier_ratio:
            return self._invalid(frame.timestamp_s)

        return GroundPlane(
            normal_base=normal,
            d_base=float(d),
            valid=True,
            inlier_ratio=inlier_ratio,
            timestamp_s=frame.timestamp_s,
        )

    def _depth_to_base_points(
        self,
        depth_m: np.ndarray,
    ) -> np.ndarray:

        h, w = depth_m.shape
        s = self.stride

        vv, uu = np.mgrid[
            0:h:s,
            0:w:s,
        ]

        z = depth_m[
            0:h:s,
            0:w:s,
        ]

        valid = (
            np.isfinite(z)
            & (z > self.min_depth_m)
            & (z < self.max_depth_m)
        )

        if not np.any(valid):
            return np.empty(
                (0, 3),
                dtype=np.float64,
            )

        u = uu[valid].astype(np.float64)
        v = vv[valid].astype(np.float64)
        z = z[valid]

        x = (u - CX_D) / FX_D * z
        y = (v - CY_D) / FY_D * z

        points_depth = np.column_stack(
            (x, y, z)
        )

        points_base = (
            points_depth @ R_BASE_DEPTH.T
            + T_BASE_DEPTH
        )

        # Broad ROI.
        roi = (
            (points_base[:, 0] > 0.02)
            & (points_base[:, 0] < 3.0)
            & (np.abs(points_base[:, 1]) < 1.5)
            & (points_base[:, 2] > -1.5)
            & (points_base[:, 2] < 0.30)
        )

        return points_base[roi]

    def _fit_ransac(
        self,
        points: np.ndarray,
    ):
        best_mask = None
        best_count = 0

        for _ in range(self.ransac_iterations):
            ids = self._rng.choice(
                len(points),
                size=3,
                replace=False,
            )

            p0, p1, p2 = points[ids]

            normal = np.cross(
                p1 - p0,
                p2 - p0,
            )

            norm = np.linalg.norm(normal)
            if norm < 1e-8:
                continue

            normal /= norm

            if normal[2] < 0:
                normal = -normal

            if normal[2] < self.min_nz:
                continue

            d = -float(normal @ p0)

            # Ground below Base origin.
            if d < 0.02 or d > 1.5:
                continue

            dist = np.abs(
                points @ normal + d
            )

            mask = (
                dist < self.distance_threshold_m
            )

            count = int(
                np.count_nonzero(mask)
            )

            if count > best_count:
                best_count = count
                best_mask = mask

        if best_mask is None:
            return None

        inliers = points[best_mask]
        if len(inliers) < 50:
            return None

        center = np.mean(
            inliers,
            axis=0,
        )

        _, _, vt = np.linalg.svd(
            inliers - center,
            full_matrices=False,
        )

        normal = vt[-1]
        normal /= np.linalg.norm(normal)

        if normal[2] < 0:
            normal = -normal

        if normal[2] < self.min_nz:
            return None

        d = -float(normal @ center)

        dist = np.abs(
            points @ normal + d
        )

        final_mask = (
            dist < self.distance_threshold_m
        )

        return normal, d, final_mask

    @staticmethod
    def _invalid(
        timestamp_s: float,
    ) -> GroundPlane:
        return GroundPlane(
            normal_base=np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float64,
            ),
            d_base=float("nan"),
            valid=False,
            inlier_ratio=0.0,
            timestamp_s=float(timestamp_s),
        )
