import sys
import time
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.b2.front_video.front_video_client import FrontVideoClient
from unitree_sdk2py.b2.back_video.back_video_client import BackVideoClient


@dataclass
class RGBFrame:
    image_bgr: np.ndarray
    timestamp_s: float
    sequence: int


class OpticalCamera:
    def __init__(self, camera: str):
        if camera == "front":
            client_cls = FrontVideoClient
        elif camera == "back":
            client_cls = BackVideoClient
        else:
            raise ValueError(camera)

        self.camera = camera

        # IMPORTANT:
        # Construct DDS/RPC client here, in main thread.
        self.client = client_cls()
        self.client.SetTimeout(3.0)
        self.client.Init()

        self._latest = None
        self._running = False
        self._thread = None

        self.frame_count = 0
        self.fail_count = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name=f"{self.camera}_camera",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=4.0)

    def get_latest(self):
        return self._latest

    def _loop(self):
        seq = 0

        while self._running:
            # Worker thread ONLY does RPC calls.
            code, data = self.client.GetImageSample()

            if code != 0:
                self.fail_count += 1
                continue

            buf = np.frombuffer(
                bytes(data),
                dtype=np.uint8,
            )

            image = cv2.imdecode(
                buf,
                cv2.IMREAD_COLOR,
            )

            if image is None:
                self.fail_count += 1
                continue

            timestamp_s = time.monotonic()

            seq += 1
            self.frame_count += 1

            self._latest = RGBFrame(
                image_bgr=image,
                timestamp_s=timestamp_s,
                sequence=seq,
            )

def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python3 -m perception.test_dual_optical_camera "
            "<network_interface>"
        )
        return

    # Important: initialize DDS only once in main process.
    ChannelFactoryInitialize(0, sys.argv[1])

    front = OpticalCamera("front")
    back = OpticalCamera("back")

    front.start()
    back.start()

    t0 = time.monotonic()
    last_print = t0

    try:
        while True:
            now = time.monotonic()

            if now - last_print >= 1.0:
                ff = front.get_latest()
                bf = back.get_latest()

                elapsed = now - t0

                if ff is None:
                    front_info = "NONE"
                else:
                    front_age_ms = (
                        now - ff.timestamp_s
                    ) * 1000.0

                    front_info = (
                        f"seq={ff.sequence} "
                        f"shape={ff.image_bgr.shape} "
                        f"age={front_age_ms:.1f}ms "
                        f"fps={front.frame_count / elapsed:.1f}"
                    )

                if bf is None:
                    back_info = "NONE"
                else:
                    back_age_ms = (
                        now - bf.timestamp_s
                    ) * 1000.0

                    back_info = (
                        f"seq={bf.sequence} "
                        f"shape={bf.image_bgr.shape} "
                        f"age={back_age_ms:.1f}ms "
                        f"fps={back.frame_count / elapsed:.1f}"
                    )

                print(
                    f"FRONT: {front_info} "
                    f"fail={front.fail_count}"
                )
                print(
                    f"BACK : {back_info} "
                    f"fail={back.fail_count}"
                )
                print("-" * 100)

                last_print = now

            time.sleep(0.01)

    except KeyboardInterrupt:
        pass

    finally:
        front.stop()
        back.stop()


if __name__ == "__main__":
    main()