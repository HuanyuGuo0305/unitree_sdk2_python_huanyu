import sys
import time

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
)

from perception.perception_thread import PerceptionThread


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python3 -m perception.test_perception_thread "
            "<network_interface>"
        )
        return

    interface = sys.argv[1]

    ChannelFactoryInitialize(
        0,
        interface,
    )

    perception = PerceptionThread(
        rate_hz=10.0,
        rgb_ground_object_height_m=0.035,
        rgbd_enter_frames=3,
        rgbd_exit_frames=3,
        max_rgb_depth_dt_s=0.10,
        min_rgbd_mask_points=80,
    )

    try:
        perception.start()
        time.sleep(1.5)

        while True:
            time.sleep(0.1)

            state = perception.get_latest()

            if state is None:
                print("NO STATE")
                continue

            age_ms = (
                time.monotonic()
                - state.timestamp_s
            ) * 1000.0

            if not state.valid:
                print(
                    f"INVALID age={age_ms:.1f} ms"
                )
                continue

            p = state.position_base

            print(
                f"{state.mode:10s} "
                f"p_Base=["
                f"{p[0]:+.3f}, "
                f"{p[1]:+.3f}, "
                f"{p[2]:+.3f}] "
                f"depth_pts={state.num_depth_points:4d} "
                f"age={age_ms:5.1f} ms"
            )

    except KeyboardInterrupt:
        pass

    finally:
        perception.stop()


if __name__ == "__main__":
    main()