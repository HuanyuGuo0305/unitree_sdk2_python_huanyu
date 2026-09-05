from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np

from unitree_sdk2py.b2.front_video.front_video_client import FrontVideoClient

from perception.data_types import RGBFrame


class RGBCamera:
    """
    Background producer for the B2W front optical RGB camera.

    ChannelFactoryInitialize(...) must be called once by the main program.
    Only the latest successfully decoded frame is kept.
    """

    def __init__(self, timeout_s: float = 3.0):
        self.timeout_s = float(timeout_s)

        self._client: Optional[FrontVideoClient] = None
        self._latest: Optional[RGBFrame] = None
        self._lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.num_ok = 0
        self.num_rpc_fail = 0
        self.num_decode_fail = 0

    def start(self) -> None:
        if self._running:
            return

        client = FrontVideoClient()
        client.SetTimeout(self.timeout_s)
        client.Init()

        self._client = client
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="RGBCameraThread",
            daemon=True,
        )
        self._thread.start()

        print("[RGBCamera] started")

    def _run(self) -> None:
        while self._running:
            try:
                code, data = self._client.GetImageSample()

                if code != 0:
                    self.num_rpc_fail += 1
                    continue

                encoded = np.frombuffer(bytes(data), dtype=np.uint8)
                image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

                if image is None:
                    self.num_decode_fail += 1
                    continue

                frame = RGBFrame(
                    image_bgr=image,
                    timestamp_s=time.monotonic(),
                )

                with self._lock:
                    self._latest = frame

                self.num_ok += 1

            except Exception as exc:
                if self._running:
                    print(f"[RGBCamera] error: {exc}")

    def get_latest(self) -> Optional[RGBFrame]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=self.timeout_s + 1.0)

        self._thread = None
        self._client = None

        print(
            "[RGBCamera] stopped | "
            f"ok={self.num_ok}, "
            f"rpc_fail={self.num_rpc_fail}, "
            f"decode_fail={self.num_decode_fail}"
        )
