from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from perception.camera_rgb import RGBCamera
from perception.camera_depth import DepthCamera
from perception.data_types import ObjectState
from perception.hsv_segmenter import RedObjectSegmenter
from perception.ground_plane_estimator import GroundPlaneEstimator
from perception.rgb_ground_localizer import RGBGroundLocalizer
from perception.rgbd_object_localizer import RGBDObjectLocalizer


class PerceptionThread:
    """
    Two-mode perception.

    RGB_GROUND:
        HSV center ray intersects the plane 3.5 cm above live ground.

    RGBD:
        HSV-selected real RGBD points -> median + MAD -> Base point.

    Switching:
        RGB_GROUND -> RGBD after 3 consecutive valid RGBD frames.
        RGBD -> RGB_GROUND after 3 consecutive invalid RGBD frames.
    """

    def __init__(
        self,
        depth_host: str = "192.168.123.164",
        depth_port: int = 50010,
        rate_hz: float = 10.0,
        rgb_ground_object_height_m: float = 0.035,
        rgbd_enter_frames: int = 3,
        rgbd_exit_frames: int = 3,
        max_rgb_depth_dt_s: float = 0.10,
        min_rgbd_mask_points: int = 80,
    ):
        self.period_s = 1.0 / float(rate_hz)

        self.rgb_camera = RGBCamera()

        self.depth_camera = DepthCamera(
            host=depth_host,
            port=depth_port,
        )

        self.segmenter = RedObjectSegmenter()
        self.ground_estimator = GroundPlaneEstimator()

        self.rgb_ground_localizer = RGBGroundLocalizer(
            object_height_m=rgb_ground_object_height_m,
        )

        self.rgbd_localizer = RGBDObjectLocalizer()

        self.rgbd_enter_frames = int(rgbd_enter_frames)
        self.rgbd_exit_frames = int(rgbd_exit_frames)
        self.max_rgb_depth_dt_s = float(max_rgb_depth_dt_s)
        self.min_rgbd_mask_points = int(min_rgbd_mask_points)

        self._preferred_mode = "RGB_GROUND"
        self._rgbd_good_count = 0
        self._rgbd_bad_count = 0

        self._latest: Optional[ObjectState] = None
        self._lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return

        # ChannelFactoryInitialize(...) must already be called by main.
        self.rgb_camera.start()
        self.depth_camera.start()

        self._running = True

        self._thread = threading.Thread(
            target=self._run,
            name="PerceptionThread",
            daemon=True,
        )

        self._thread.start()

        print("[PerceptionThread] started")

    def stop(self) -> None:
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=2.0)

        self.depth_camera.stop()
        self.rgb_camera.stop()

        self._thread = None

        print("[PerceptionThread] stopped")

    def get_latest(self) -> Optional[ObjectState]:
        """Non-blocking."""
        with self._lock:
            return self._latest

    def _publish(
        self,
        position_base: np.ndarray,
        mode: str,
        num_depth_points: int = 0,
    ) -> None:
        state = ObjectState(
            position_base=position_base.copy(),
            valid=True,
            mode=mode,
            timestamp_s=time.monotonic(),
            num_depth_points=int(num_depth_points),
        )

        with self._lock:
            self._latest = state

    def _publish_invalid(self) -> None:
        state = ObjectState(
            position_base=np.full(
                3,
                np.nan,
                dtype=np.float64,
            ),
            valid=False,
            mode="INVALID",
            timestamp_s=time.monotonic(),
            num_depth_points=0,
        )

        with self._lock:
            self._latest = state

    def _run(self) -> None:
        while self._running:
            t0 = time.monotonic()

            try:
                self._step()

            except Exception as exc:
                print(f"[PerceptionThread] error: {exc}")
                self._publish_invalid()

            elapsed = time.monotonic() - t0

            time.sleep(
                max(
                    0.0,
                    self.period_s - elapsed,
                )
            )

    def _step(self) -> None:
        rgb = self.rgb_camera.get_latest()
        depth = self.depth_camera.get_latest()

        if rgb is None or depth is None:
            self._publish_invalid()
            return

        detection = self.segmenter.detect(
            rgb.image_bgr
        )

        if not detection.valid:
            self._publish_invalid()
            return

        # ------------------------------------------------------------
        # Ground plane
        # ------------------------------------------------------------

        ground = self.ground_estimator.estimate(
            depth
        )

        # ------------------------------------------------------------
        # RGB_GROUND candidate
        #
        # New method: use HSV center_uv directly.
        # ------------------------------------------------------------

        rgb_ground = None

        if ground.valid:
            rgb_ground = (
                self.rgb_ground_localizer.localize(
                    detection.center_uv,
                    ground,
                )
            )

        rgb_ground_valid = (
            rgb_ground is not None
            and rgb_ground.valid
        )

        # ------------------------------------------------------------
        # RGBD candidate
        # ------------------------------------------------------------

        rgbd = None

        rgb_depth_dt = abs(
            rgb.timestamp_s
            - depth.timestamp_s
        )

        if rgb_depth_dt <= self.max_rgb_depth_dt_s:
            rgbd = self.rgbd_localizer.localize(
                depth_raw=depth.image_raw,
                depth_scale=depth.depth_scale,
                inner_mask=detection.inner_mask,
            )

        rgbd_valid = (
            rgbd is not None
            and rgbd.valid
            and rgbd.num_mask_points
            >= self.min_rgbd_mask_points
        )

        # ------------------------------------------------------------
        # Mode hysteresis
        # ------------------------------------------------------------

        if self._preferred_mode == "RGB_GROUND":

            if rgbd_valid:
                self._rgbd_good_count += 1
            else:
                self._rgbd_good_count = 0

            if (
                self._rgbd_good_count
                >= self.rgbd_enter_frames
            ):
                self._preferred_mode = "RGBD"
                self._rgbd_bad_count = 0

        else:

            if rgbd_valid:
                self._rgbd_bad_count = 0
            else:
                self._rgbd_bad_count += 1

            if (
                self._rgbd_bad_count
                >= self.rgbd_exit_frames
            ):
                self._preferred_mode = "RGB_GROUND"
                self._rgbd_good_count = 0

        # ------------------------------------------------------------
        # Publish
        # ------------------------------------------------------------

        if self._preferred_mode == "RGBD":

            if rgbd_valid:
                self._publish(
                    rgbd.position_base,
                    "RGBD",
                    rgbd.num_inliers,
                )
                return

            # Short hold during exit hysteresis.
            if (
                self._rgbd_bad_count
                < self.rgbd_exit_frames
                and self._latest is not None
                and self._latest.valid
            ):
                return

        if rgb_ground_valid:
            self._publish(
                rgb_ground.position_base,
                "RGB_GROUND",
                0,
            )
            return

        # Rare fallback.
        if rgbd_valid:
            self._publish(
                rgbd.position_base,
                "RGBD",
                rgbd.num_inliers,
            )
            return

        self._publish_invalid()