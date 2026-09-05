from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HSVDetection:
    valid: bool
    mask: np.ndarray
    inner_mask: np.ndarray
    area_px: int
    bbox_xywh: tuple[int, int, int, int]
    center_uv: tuple[float, float]
    ground_uv: tuple[float, float]


class RedObjectSegmenter:
    """
    Simple HSV detector for the red/orange-red plush.
    """

    def __init__(
        self,
        h1_min: int = 0,
        h1_max: int = 18,
        h2_min: int = 170,
        h2_max: int = 179,
        s_min: int = 100,
        v_min: int = 70,
        min_area_px: int = 1500,
    ):
        self.lower1 = np.array(
            [h1_min, s_min, v_min],
            dtype=np.uint8,
        )
        self.upper1 = np.array(
            [h1_max, 255, 255],
            dtype=np.uint8,
        )

        self.lower2 = np.array(
            [h2_min, s_min, v_min],
            dtype=np.uint8,
        )
        self.upper2 = np.array(
            [h2_max, 255, 255],
            dtype=np.uint8,
        )

        self.min_area_px = int(min_area_px)

        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        )
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (9, 9),
        )
        self.kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        )

    def detect(self, image_bgr: np.ndarray) -> HSVDetection:
        hsv = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2HSV,
        )

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.lower1, self.upper1),
            cv2.inRange(hsv, self.lower2, self.upper2),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel_open,
            iterations=1,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            self.kernel_close,
            iterations=1,
        )

        (
            num_labels,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        if num_labels <= 1:
            return self._invalid(mask)

        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = int(np.argmax(areas)) + 1
        best_area = int(stats[best_label, cv2.CC_STAT_AREA])

        if best_area < self.min_area_px:
            return self._invalid(mask)

        object_mask = np.zeros_like(mask)
        object_mask[labels == best_label] = 255

        x = int(stats[best_label, cv2.CC_STAT_LEFT])
        y = int(stats[best_label, cv2.CC_STAT_TOP])
        w = int(stats[best_label, cv2.CC_STAT_WIDTH])
        h = int(stats[best_label, cv2.CC_STAT_HEIGHT])

        center_u = float(centroids[best_label, 0])
        center_v = float(centroids[best_label, 1])

        inner_mask = cv2.erode(
            object_mask,
            self.kernel_erode,
            iterations=1,
        )

        # Robust bottom-center pixel for RGB_GROUND.
        ys, _ = np.nonzero(object_mask)
        v_max = int(np.max(ys))

        band_h = max(
            3,
            int(round(0.08 * h)),
        )

        bottom_ys = ys[
            ys >= (v_max - band_h)
        ]

        ground_u = center_u
        ground_v = float(
            np.percentile(bottom_ys, 95.0)
        )

        return HSVDetection(
            valid=True,
            mask=object_mask,
            inner_mask=inner_mask,
            area_px=best_area,
            bbox_xywh=(x, y, w, h),
            center_uv=(center_u, center_v),
            ground_uv=(ground_u, ground_v),
        )

    @staticmethod
    def _invalid(mask: np.ndarray) -> HSVDetection:
        empty = np.zeros_like(mask)
        nan = float("nan")

        return HSVDetection(
            valid=False,
            mask=empty,
            inner_mask=empty,
            area_px=0,
            bbox_xywh=(0, 0, 0, 0),
            center_uv=(nan, nan),
            ground_uv=(nan, nan),
        )
