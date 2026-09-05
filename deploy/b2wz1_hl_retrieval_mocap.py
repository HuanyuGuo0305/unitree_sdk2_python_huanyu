"""
B2WZ1 hierarchical retrieval sim2real, driven by OptiTrack motion capture.

This is deploy/b2wz1_hl_retrieval_sim2real.py with the entire sensing stack
swapped out. NO camera perception, NO AprilTag, NO rear visual odometry, and
no developer-PC UDP server: the object position, the retrieval target and the
B2W base pose all come straight from Motive.

Everything downstream of sensing is INHERITED UNCHANGED from the validated
controller -- the ONNX policies, the observation layout and histories, the
PLB/EE geometry, the Z1 DCMotor gripper law, the grasp proxy, the safe-hold
and damping-protection paths, the startup sequence and the MuJoCo debug
visualizer. That is deliberate: subclassing keeps a single copy of the 6000
lines that are hard to get right, and makes the mocap change reviewable as
what it actually is -- a different source for three quantities.

What is replaced
----------------
    self.perception                     MocapPerceptionSystem instead of
                                        CRLB2WPerceptionSystem
    anchor_vo_base_height_...()         mocap height is absolute, so it is
                                        verified rather than anchored
    setup()                             prints a mocap banner and skips the
                                        developer-server reporting

Three Motive rigid bodies are required (as currently named on this system):

    "B2"          the robot. With the calibrated mocap->root offset this
                  gives the exact base_link pose, hence the base height and
                  the world->base transform used for the two targets.
    "octopus"     the object to retrieve.
    "retrieval"   the retrieval target.

The mocap->root offset comes from deploy/b2w_mocap_root_calibration.py; point
`mocap_root_offset_path` at the YAML it writes. Without a correct offset the
robot's own frame is wrong and every derived quantity is wrong with it, so
the script refuses to start rather than guessing.

Run (from the repository root):

    python3 deploy/b2wz1_hl_retrieval_mocap.py \
        enxa0cec819e15f \
        deploy/configs/b2wz1_hl_retrieval_mocap.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DEPLOY_DIR, ".."))

# The project root makes `utils` importable; the deploy directory makes the
# camera controller importable as a module, which running this file directly
# would do anyway but importing it (tests, tooling) would not.
for _p in (_PROJECT_ROOT, _DEPLOY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from utils.command_helper import InitLowCmd
from utils.mocap_perception import MocapBodySelector, MocapPerceptionSystem

from b2wz1_hl_retrieval_sim2real import B2WZ1HierarchicalRetrievalController


class B2WZ1MocapRetrievalController(B2WZ1HierarchicalRetrievalController):
    """Hierarchical retrieval controller sensing through OptiTrack."""

    def __init__(self, cfg_path: str, network_interface: str) -> None:
        # The base constructor builds a CRLB2WPerceptionSystem, but that
        # constructor only stores configuration -- every socket, camera and
        # thread is created in initialize(), which is never reached here
        # because self.perception is replaced immediately below.
        super().__init__(cfg_path=cfg_path, network_interface=network_interface)

        # The base class reads this to decide whether base height comes from
        # the perception snapshot. Under mocap that is always what we want.
        if not self.vo_base_height_enabled:
            raise ValueError(
                "vo_base_height_enabled must be true for the mocap "
                "deployment: it is what makes the controller take base height "
                "from the perception snapshot, which mocap measures directly."
            )

        self.perception = self._build_mocap_perception()

        # Rolling timing trace, dumped when protection trips. The z1_controller
        # declares the arm lost after repeated 20 ms UDP recv timeouts, so what
        # matters is whether THIS process stalls the machine for that long.
        self._trace_len = int(self.cfg.get("trace_length_steps", 1500))
        self._trace = np.zeros((self._trace_len, 6), dtype=np.float64)
        self._trace_i = 0
        self._trace_n = 0
        self._trace_t0 = 0.0
        self._last_step_start = 0.0
        self._arm_max_stall_s = 0.0

        # Live health reporting, so a drop can be watched happening rather
        # than only reconstructed afterwards.
        self._health_print_s = float(self.cfg.get("health_print_s", 1.0))
        self._last_health_print = 0.0
        self._window_max_dur = 0.0
        self._window_max_gap = 0.0
        self._window_arm_stall = 0.0
        self._last_fsm = None
        self._slow_step_warns = 0

        # Frozen-reading watchdog. The z1_controller keeps serving its last
        # known state while it retries a lost arm link, and only flips the FSM
        # to PASSIVE after its disconnect counter expires -- measured at 2.4 s
        # on this robot. The policy spends that whole time driving on dead
        # feedback, and the gripper DCMotor law saturates against a stale angle.
        # Detecting the freeze directly trips protection immediately instead.
        self._freeze_trip_s = float(self.cfg.get("arm_reading_freeze_trip_s", 0.30))
        self._freeze_warn_s = float(self.cfg.get("arm_reading_freeze_warn_s", 0.12))
        self._last_q_seen = None
        self._freeze_started = 0.0
        self._freeze_warned = False

        # Arm joint-limit clamp. The Z1 SDK's jointProtect() does NOT clamp
        # position -- an out-of-range command passes straight through it -- and
        # the MuJoCo model the policy trained against gives joint 2 five
        # degrees MORE travel than the real arm has (170 vs 165). So the policy
        # can legitimately command range the hardware does not have, jam the
        # stop, and stall the motor at clipped torque. Nothing else in the
        # stack prevents that.
        self._arm_limit_margin = float(self.cfg.get("arm_limit_margin_rad", 0.02))
        self._arm_limits = None          # filled in at connect, from the SDK
        self._arm_clamp_counts = np.zeros(6, dtype=np.int64)
        self._arm_clamp_worst = np.zeros(6, dtype=np.float64)
        self._arm_clamp_warned = 0

        # Arm tracking error. Distinct from the limit clamp: the clamp catches
        # a target the hardware cannot reach, this catches the arm not being
        # where it was told. With the soft runtime gains (kp 3.0-3.5) gravity
        # can drag the arm well past its target -- 36 deg was observed as the
        # robot went down -- and far enough drags it into its mechanical stop,
        # which is what faults the arm and drops the link.
        self._arm_track_warn_rad = float(self.cfg.get("arm_tracking_warn_rad", 0.35))
        self._arm_track_worst = 0.0
        self._window_track = 0.0
        self._arm_track_warned = 0

        # 500 Hz arm PD thread. See _start_arm_thread().
        self._arm_command_hz = float(self.cfg.get("arm_command_hz", 500.0))
        self._arm_thread_enabled = bool(
            self.cfg.get("arm_thread_enabled", True)
        ) and self._arm_command_hz > (1.0 / self.control_dt) + 1e-9
        self._arm_thread: Optional[threading.Thread] = None
        self._arm_thread_stop = threading.Event()
        self._arm_thread_error: Optional[str] = None
        self._arm_target_lock = threading.Lock()
        self._arm_target_q = self.default_arm_pos.copy()
        self._arm_target_gripper = float(self.gripper_open_pos)
        self._arm_target_stamp = 0.0
        self._arm_achieved_hz = 0.0
        self._arm_stale_warn_s = float(
            self.cfg.get("arm_target_stale_warn_s", 0.2)
        )

        # Background monitor: keeps the MuJoCo window live during every phase,
        # not just the policy loop. See _monitor_loop().
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._monitor_warned = False
        self._last_control_push = 0.0
        self._monitor_hz = float(self.cfg.get("visualizer_monitor_hz", 30.0))
        self._monitor_takeover_s = float(
            self.cfg.get("visualizer_monitor_takeover_s", 0.25)
        )
        self._monitor_enabled = bool(
            self.cfg.get("visualizer_monitor_enabled", True)
        )

        # The base class validated this contract against the perception object
        # it built; re-run it against the replacement so the two can never
        # drift apart silently.
        for attr in (
            "initialize",
            "start",
            "stop",
            "get_latest_snapshot",
            "anchor_vo_base_height",
            "get_vo_base_height_snapshot",
            "reset_vo_base_height_estimator",
        ):
            if not callable(getattr(self.perception, attr, None)):
                raise TypeError(
                    f"{type(self.perception).__name__} is missing the required "
                    f"perception API {attr!r}."
                )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _selector(self, key: str, default_name: str) -> MocapBodySelector:
        cfg = self.cfg
        return MocapBodySelector(
            label=key,
            name=cfg.get(f"mocap_{key}_name", default_name),
            rb_id=cfg.get(f"mocap_{key}_id"),
            offset_local=cfg.get(f"mocap_{key}_offset_local", [0.0, 0.0, 0.0]),
            max_age_s=float(cfg.get(f"mocap_{key}_max_age_s", 0.25)),
        )

    def _build_mocap_perception(self) -> MocapPerceptionSystem:
        cfg = self.cfg

        offset_path = cfg.get("mocap_root_offset_path")
        inline_offset = cfg.get("mocap_root_offset")
        if offset_path:
            root_offset = self._resolve_path(str(offset_path))
        elif inline_offset:
            root_offset = inline_offset
        else:
            # Optional. Without it the Motive rigid body's own frame is used
            # as the robot root, which is enough to bring the system up but
            # leaves the base pose wrong by whatever that asset's pivot and
            # axes happen to be. See the warning printed in setup().
            root_offset = None

        return MocapPerceptionSystem(
            body=self._selector("body", "B2"),
            object_body=self._selector("object", "octopus"),
            retrieval_body=self._selector("retrieval", "retrieval"),
            root_offset=root_offset,
            ground_z=self.ground_z,
            local_ip=str(cfg.get("mocap_local_ip", "")),
            server_ip=cfg.get("mocap_server_ip") or None,
            multicast_group=str(cfg.get("mocap_multicast_group", "239.255.42.99")),
            data_port=int(cfg.get("mocap_data_port", 1511)),
            command_port=int(cfg.get("mocap_command_port", 1510)),
            join_multicast=bool(cfg.get("mocap_join_multicast", True)),
            up_axis=str(cfg.get("mocap_up_axis", "auto")),
            startup_timeout_s=float(cfg.get("mocap_timeout_s", 10.0)),
        )

    # ------------------------------------------------------------------
    # Base height
    # ------------------------------------------------------------------

    def anchor_vo_base_height_before_policy_start(self) -> None:
        """
        Verify the measured base height instead of anchoring an estimator.

        The VO version had to pin a drifting relative translation to a known
        standing height, and could only do that once, at policy start. Mocap
        measures the base pose absolutely in a gravity-aligned world frame, so
        there is nothing to anchor -- but this is still the right moment to
        prove the height is live and sane before the policy is handed control,
        so the same full-DEFAULT hold is kept while waiting.
        """
        print(
            "[BASE-HEIGHT] Mocap height is absolute; verifying it is live "
            "before policy start..."
        )

        tolerance = float(self.cfg.get("mocap_base_height_sanity_tolerance_m", 0.25))
        expected = float(self.base_height_anchor_m)

        # The check catches a bad offset or a wrong ground_z. How hard it bites
        # depends on whether there is a calibration to be wrong in the first
        # place: with one, a mismatch means something is broken and the run
        # stops; without one, the height is EXPECTED to be off and stopping
        # would just block a deliberately uncalibrated bring-up. A tolerance
        # of zero or less disables the check outright.
        calibrated = self.perception.root_offset_calibrated
        enforce = tolerance > 0.0
        fatal = enforce and calibrated

        last_print = 0.0

        while True:
            # Keep the exact startup hold the base class uses while waiting.
            self._write_b2w_pose_cmd_policy(
                self.default_b2w_pos_policy,
                use_pd_gains=True,
            )
            self.send_b2w_cmd()
            self.z1.hold_pose_lowcmd(
                self.default_arm_pos.copy(),
                self._gripper_training_to_sdk(self.gripper_open_pos),
            )

            self._read_all_sensors_once()

            state = self.perception.get_vo_base_height_snapshot()
            self.last_base_height_state = state

            height_m = state.get("height_m")
            valid = bool(state.get("valid", False)) and height_m is not None

            if valid and np.isfinite(float(height_m)):
                height_m = float(height_m)
                error = abs(height_m - expected)

                if enforce and error > tolerance:
                    message = (
                        f"Mocap base height {height_m:.4f} m differs from the "
                        f"expected DEFAULT standing height {expected:.4f} m by "
                        f"{error:.4f} m (tolerance {tolerance:.3f} m)."
                    )
                    if fatal:
                        raise RuntimeError(
                            message
                            + " Check the mocap->root offset calibration and "
                            "ground_z, or raise "
                            "mocap_base_height_sanity_tolerance_m if the robot "
                            "really is standing at this height."
                        )
                    print(
                        "[BASE-HEIGHT][WARN] "
                        + message
                        + " Running UNCALIBRATED, so this is expected: the "
                        "policy will observe a base height that is off by "
                        "roughly this much, which shifts the commanded arm "
                        "target. Calibrate to remove it."
                    )

                self.base_height = height_m
                print(
                    "[BASE-HEIGHT] MOCAP | "
                    f"h={self.base_height:.4f} m | "
                    f"expected~{expected:.4f} m | "
                    f"err={error:.4f} m | "
                    f"frame={state.get('epoch')} | "
                    f"age={state.get('age_ms')} ms"
                )
                return

            now = time.monotonic()
            if now - last_print >= self.perception_status_print_s:
                print(
                    "[BASE-HEIGHT-WAIT] "
                    f"valid={int(bool(state.get('valid', False)))} | "
                    f"reason={state.get('reason')} | "
                    f"age={state.get('age_ms')} ms"
                )
                last_print = now

            time.sleep(self.control_dt)

    # ------------------------------------------------------------------
    # MuJoCo visualization
    # ------------------------------------------------------------------

    def _configure_visualizer(self) -> None:
        """
        Switch the debug visualizer into absolute-world mode.

        The camera/VO deployment could only ever draw a robot-centric scene:
        it knew the base height but not where the robot was, so the root sat
        on the world Z axis and the object and target were drawn around it.
        Mocap measures all three absolutely, so the whole scene can be drawn
        where it physically is -- which is what makes the render worth
        watching while driving.
        """
        if self.visualizer is None:
            return

        self.visualizer.configure_world_view(
            show_world_origin=bool(
                self.cfg.get("visualizer_show_world_origin", True)
            ),
            camera_follow=bool(self.cfg.get("visualizer_camera_follow", True)),
            world_origin_axis_len=float(
                self.cfg.get("visualizer_world_origin_axis_len", 0.4)
            ),
            camera_follow_smoothing=float(
                self.cfg.get("visualizer_camera_follow_smoothing", 0.15)
            ),
        )

        self._viz_show_mocap_pivots = bool(
            self.cfg.get("visualizer_show_mocap_pivots", True)
        )
        self._viz_pivot_radius = float(
            self.cfg.get("visualizer_mocap_pivot_radius", 0.022)
        )

    def _mocap_pivot_markers(
        self, snap: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Raw Motive pivots for the three assets, as world-frame spheres."""
        if not getattr(self, "_viz_show_mocap_pivots", False):
            return None

        if snap is None:
            snap = self.last_perception_snapshot
        if not snap:
            return None

        markers = []
        for key, rgba in (
            ("body", [1.0, 1.0, 1.0, 0.85]),
            ("object", [0.2, 1.0, 0.4, 0.55]),
            ("retrieval", [1.0, 0.3, 1.0, 0.55]),
        ):
            channel = snap.get(key) or {}
            if not channel.get("valid"):
                continue
            pos = channel.get("position_world")
            if pos is None:
                continue
            markers.append(
                {"pos": pos, "rgba": rgba, "radius": self._viz_pivot_radius}
            )

        return markers or None

    def _push_visualizer_state(self, task: Dict[str, Any]) -> None:
        """
        Draw the scene where it actually is.

        Two things change versus the inherited robot-centric push:

        `base_pos_w`  places the root at its measured world position instead
                      of on the world Z axis.

        `base_quat_wxyz` becomes the MOCAP orientation rather than the IMU's.
                      That is required for the scene to be self-consistent:
                      the object and target are drawn by rotating their
                      base-frame positions back out to world, and the IMU's
                      yaw is arbitrary, so using it would spin both targets
                      around the robot by an unknown angle. The policy itself
                      is untouched and still observes the IMU.

        If the body is not currently tracked there is no world pose to draw
        with, so this leaves the inherited robot-centric push alone rather
        than pinning the robot at a stale position.
        """
        super()._push_visualizer_state(task)

        if self.visualizer is None:
            return

        # Tells the monitor thread that the control loop is driving the view.
        self._last_control_push = time.monotonic()

        body = (self.last_perception_snapshot or {}).get("body") or {}
        if not body.get("valid"):
            return

        root_pos_w = body.get("root_position_world")
        root_quat_w = body.get("root_quat_wxyz")
        if root_pos_w is None or root_quat_w is None:
            return

        self.visualizer.update_state(
            base_pos_w=np.asarray(root_pos_w, dtype=np.float64),
            base_quat_wxyz=np.asarray(root_quat_w, dtype=np.float32),
            mocap_points=self._mocap_pivot_markers(),
        )

    # ------------------------------------------------------------------
    # Arm joint-limit clamp
    # ------------------------------------------------------------------

    def _load_arm_limits(self) -> None:
        """Take the limits from the SDK: they are what the hardware has."""
        arm_model = getattr(self.z1, "arm_model", None)
        if arm_model is None:
            print("[ARM-LIMIT][WARN] Z1 arm model unavailable; targets will NOT be clamped.")
            return
        try:
            lo = np.asarray(arm_model.getJointQMin(), dtype=np.float64).reshape(6)
            hi = np.asarray(arm_model.getJointQMax(), dtype=np.float64).reshape(6)
        except Exception as exc:  # noqa: BLE001
            print(f"[ARM-LIMIT][WARN] could not read SDK limits ({exc!r}); no clamping.")
            return

        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)) and np.all(hi > lo)):
            print("[ARM-LIMIT][WARN] SDK limits look wrong; no clamping.")
            return

        m = self._arm_limit_margin
        self._arm_limits = (lo + m, hi - m)
        print(
            f"[ARM-LIMIT] clamping arm targets to the SDK range, {m:.3f} rad inside:"
        )
        for i in range(6):
            print(f"              joint{i+1}: [{lo[i]+m:+.4f}, {hi[i]-m:+.4f}] rad "
                  f"= [{np.degrees(lo[i]+m):+7.1f}, {np.degrees(hi[i]-m):+7.1f}] deg")

    def _clamp_arm_target(self) -> None:
        """
        Clamp self.arm_target in place, and say so when it bites.

        Repeated clamping is not just a safety net firing -- it means the
        policy is commanding somewhere the hardware cannot go, which is a
        sim2real gap worth knowing about rather than silently absorbing.
        """
        if self._arm_limits is None:
            return
        lo, hi = self._arm_limits
        target = np.asarray(self.arm_target, dtype=np.float64).reshape(6)
        clamped = np.clip(target, lo, hi)
        over = np.abs(clamped - target)
        hit = over > 1.0e-9
        if np.any(hit):
            self._arm_clamp_counts[hit] += 1
            self._arm_clamp_worst = np.maximum(self._arm_clamp_worst, over)
            if self._arm_clamp_warned < 10:
                self._arm_clamp_warned += 1
                j = int(np.argmax(over))
                print(
                    f"[ARM-LIMIT] joint{j+1} target {target[j]:+.4f} clamped to "
                    f"{clamped[j]:+.4f} rad (by {np.degrees(over[j]):.2f} deg). The "
                    "policy is commanding past the hardware range."
                )
            self.arm_target[:] = clamped.astype(self.arm_target.dtype)

    # ------------------------------------------------------------------
    # 500 Hz arm PD thread
    # ------------------------------------------------------------------

    def _start_arm_thread(self) -> None:
        """
        Run the Z1 PD at `arm_command_hz` while the policy stays at 50 Hz.

        The policy still decides the arm target once per 50-Hz step; this
        thread just re-evaluates the PD law against that held target ten times
        more often. Nothing about the policy, its observations or its action
        scaling changes.

        Why it helps: the Z1 lowcmd loop is native 500 Hz. Feeding it a
        setpoint only every 20 ms leaves the firmware sitting on a stale
        command for most of its own cycles, which measures as a large fixed
        tracking lag. Re-evaluating the same target at 2 ms removes that.

        Two things follow from the fact that one Z1 packet carries the arm and
        the gripper together:

          - the gripper's DCMotor tau law is evaluated at 500 Hz too. That is
            CLOSER to training, not further from it: IsaacLab evaluates
            actuator models every physics step, not every policy step.
          - this thread owns Z1 comms while it runs, so the 50-Hz loop must
            not also call read_state() (see _read_all_sensors_once).

        It runs ONLY during the policy loop. Startup moves, damping protection
        and leg recovery drive the arm themselves, and would fight it.
        """
        if not self._arm_thread_enabled or self._arm_thread is not None:
            return

        with self._arm_target_lock:
            self._arm_target_q = self.arm_target.copy()
            self._arm_target_gripper = float(self.gripper_target)
            self._arm_target_stamp = time.monotonic()

        self._arm_thread_error = None
        self._arm_thread_stop.clear()
        self._arm_thread = threading.Thread(
            target=self._arm_loop, name="z1-arm-500hz", daemon=True
        )
        self._arm_thread.start()
        print(
            f"[Z1-RATE] Arm PD thread started at {self._arm_command_hz:g} Hz "
            f"(policy stays at {1.0 / self.control_dt:.0f} Hz)."
        )

    def _stop_arm_thread(self) -> None:
        # getattr, not attribute access: this runs from run()'s finally, and
        # raising here on a partially constructed controller would MASK the
        # exception that actually ended the run.
        if getattr(self, "_arm_thread", None) is None:
            return
        self._arm_thread_stop.set()
        self._arm_thread.join(timeout=1.0)
        self._arm_thread = None
        print(
            f"[Z1-RATE] Arm PD thread stopped "
            f"(achieved {self._arm_achieved_hz:.0f} Hz)."
        )

    @property
    def _arm_thread_running(self) -> bool:
        return self._arm_thread is not None and self._arm_thread.is_alive()

    def _publish_arm_target(self) -> None:
        with self._arm_target_lock:
            self._arm_target_q = self.arm_target.copy()
            self._arm_target_gripper = float(self.gripper_target)
            self._arm_target_stamp = time.monotonic()

    def _arm_loop(self) -> None:
        # ONE guard around the entire loop. Anything that raises in here --
        # the command, the FSM read, the bookkeeping -- must surface as a
        # thread error that trips protection. A thread that dies quietly
        # leaves the arm uncommanded with nobody noticing.
        try:
            self._arm_loop_body()
        except Exception as exc:  # noqa: BLE001
            self._arm_thread_error = repr(exc)
            print(f"[Z1-RATE][ERROR] arm thread failed: {exc!r}")

    def _arm_loop_body(self) -> None:
        period = 1.0 / max(self._arm_command_hz, 1.0)
        next_deadline = time.perf_counter()
        count = 0
        window_start = time.perf_counter()
        warned_stale = False
        last_send = 0.0

        while not self._arm_thread_stop.is_set():
            with self._arm_target_lock:
                q_target = self._arm_target_q.copy()
                gripper_target = self._arm_target_gripper
                stamp = self._arm_target_stamp

            # A stale target means the policy loop stopped feeding us. Keep
            # holding the last one rather than going quiet: under position_pd
            # the firmware would hold anyway, but under dcmotor the arm is
            # held up ONLY by this torque stream and dropping it would let the
            # arm fall. The outer loop is what decides to enter protection.
            if not warned_stale and time.monotonic() - stamp > self._arm_stale_warn_s:
                warned_stale = True
                print(
                    "[Z1-RATE][WARN] arm target is stale; holding the last one. "
                    "The policy loop is not keeping up."
                )

            self.z1.track_target_pd_runtime_once(
                q_target=q_target,
                gripper_q_target_training=gripper_target,
                use_startup_gains=False,
            )

            # The arm thread sees the FSM at its own rate, so it notices a drop
            # to PASSIVE sooner than the 50 Hz loop can. Printing only on
            # CHANGE keeps this off the hot path.
            fsm = self.z1.get_fsm_state()
            if fsm != self._last_fsm:
                elapsed = time.perf_counter() - (self._trace_t0 or time.perf_counter())
                print(
                    f"\n[FSM] t={elapsed:7.2f}s  {self._last_fsm} -> {fsm}"
                    + ("   *** ARM LEFT LOWCMD ***" if self._last_fsm is not None else "")
                )
                self._last_fsm = fsm

            now = time.perf_counter()

            # A live encoder read is never bit-identical twice running: sensor
            # noise always moves the last digits. All six joints AND the
            # gripper repeating exactly means the value is a cache. A held arm
            # measured a 99 ms worst-case quiet window, so the trip threshold
            # sits well clear of that.
            q_now = np.asarray(self.z1.q, dtype=np.float64).reshape(6)
            grip_now = float(self.z1.gripper_q)
            same = (
                self._last_q_seen is not None
                and np.array_equal(q_now, self._last_q_seen[0])
                and grip_now == self._last_q_seen[1]
            )
            if same:
                if self._freeze_started == 0.0:
                    self._freeze_started = now
                frozen_for = now - self._freeze_started
                if frozen_for >= self._freeze_warn_s and not self._freeze_warned:
                    self._freeze_warned = True
                    print(
                        f"\n[FREEZE][WARN] Z1 reading has not changed for "
                        f"{frozen_for*1e3:.0f} ms (fsm still {self.z1.get_fsm_state()}). "
                        "The arm link may be dropping."
                    )
                if frozen_for >= self._freeze_trip_s:
                    print(
                        f"\n[FREEZE] Z1 reading frozen at {np.round(q_now, 4).tolist()} "
                        f"for {frozen_for*1e3:.0f} ms while the FSM still reports "
                        f"{self.z1.get_fsm_state()}."
                    )
                    raise RuntimeError(
                        f"Z1 joint reading frozen for {frozen_for*1e3:.0f} ms "
                        "(stale cache; the arm link is down even though the FSM "
                        "has not caught up yet)"
                    )
            else:
                self._freeze_started = 0.0
                self._freeze_warned = False
                self._last_q_seen = (q_now.copy(), grip_now)

            count += 1
            if now - window_start >= 1.0:
                self._arm_achieved_hz = count / (now - window_start)
                count = 0
                window_start = now
                # A rate the loop cannot hold is worth seeing at once: it means
                # this process is saturated, which is also what starves the
                # z1_controller and drops the arm link.
                if self._arm_achieved_hz < 0.85 * self._arm_command_hz:
                    print(
                        f"[Z1-RATE][WARN] only {self._arm_achieved_hz:.0f} Hz of the "
                        f"requested {self._arm_command_hz:g} Hz; the machine is not "
                        "keeping up. Lower arm_command_hz."
                    )

            # Worst gap between consecutive arm sends. If this exceeds the
            # controller's 20 ms recv timeout, that alone explains a dropped
            # link.
            if last_send > 0.0:
                stall = now - last_send
                self._arm_max_stall_s = max(self._arm_max_stall_s, stall)
                self._window_arm_stall = max(self._window_arm_stall, stall)
                if stall > 0.020:
                    print(f"[ARM-STALL] {stall*1e3:.0f} ms gap between arm sends "
                          f"-- past the controller's 20 ms UDP timeout")
            last_send = now

            next_deadline += period
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                # Fell behind: re-anchor rather than burst.
                next_deadline = time.perf_counter()

    # ------------------------------------------------------------------
    # Hooks into the inherited control flow
    # ------------------------------------------------------------------

    def send_policy_targets(self) -> None:
        """Hand the arm target to the 500 Hz thread instead of sending it here."""
        # Before either path: the target must be reachable on this hardware.
        self._clamp_arm_target()

        if not self._arm_thread_running:
            super().send_policy_targets()
            return

        self._write_b2w_rl_cmd()
        self.send_b2w_cmd()
        self._publish_arm_target()

    def _read_all_sensors_once(self) -> None:
        """
        Skip the Z1 read while the arm thread owns comms.

        track_target_pd_runtime_once() already refreshes z1.q / qd / gripper_q
        after every one of its 500 sends per second, so a read here would add
        a second round trip per policy step and contend for the comms lock for
        no new information.
        """
        self._read_b2w_sensors_once()
        if not self._arm_thread_running:
            self.z1.read_state()

    def prime_first_hl_and_ll(self):
        # Priming still commands the arm directly; the thread takes over only
        # once the policy loop is about to begin.
        result = super().prime_first_hl_and_ll()
        self._start_arm_thread()
        return result

    def step_after_prime(self):
        if self._arm_thread_enabled and self._arm_thread is not None \
                and not self._arm_thread_running:
            # Trip regardless of whether an error was recorded. A thread that
            # is simply gone is just as dangerous as one that reported why,
            # and the arm is no longer being commanded either way.
            error = self._arm_thread_error or "no error recorded (thread vanished)"
            self._stop_arm_thread()
            return False, f"Z1 arm command thread died: {error}"

        now = time.perf_counter()
        if self._trace_t0 == 0.0:
            self._trace_t0 = now
        gap = (now - self._last_step_start) if self._last_step_start else 0.0
        self._last_step_start = now

        ok, reason = super().step_after_prime()

        done = time.perf_counter()
        snap = self.last_perception_snapshot or {}
        body = snap.get("body") or {}
        self._trace[self._trace_i] = (
            now - self._trace_t0,          # t since first step
            done - now,                    # how long this step took
            gap,                           # actual loop period
            self._arm_achieved_hz,
            float(body.get("age_ms") or -1.0),
            float(getattr(self.z1.get_fsm_state(), "value", -1)),
        )
        self._trace_i = (self._trace_i + 1) % self._trace_len
        self._trace_n = min(self._trace_n + 1, self._trace_len)

        # How far the arm actually is from where it was told to be.
        try:
            err = np.abs(
                np.asarray(self.arm_target, dtype=np.float64).reshape(6)
                - np.asarray(self.z1.q, dtype=np.float64).reshape(6)
            )
            worst = float(err.max())
            j = int(err.argmax())
            self._window_track = max(self._window_track, worst)
            self._arm_track_worst = max(self._arm_track_worst, worst)
            if worst > self._arm_track_warn_rad and self._arm_track_warned < 15:
                self._arm_track_warned += 1
                print(
                    f"[ARM-TRACK] joint{j+1} is {np.degrees(worst):.1f} deg from its "
                    f"target (cmd {self.arm_target[j]:+.3f}, meas {self.z1.q[j]:+.3f}). "
                    "The arm is not holding -- gravity can drag it into its stop."
                )
        except Exception:  # noqa: BLE001 - diagnostics must never break the loop
            pass

        step_dur = done - now
        self._window_max_dur = max(self._window_max_dur, step_dur)
        self._window_max_gap = max(self._window_max_gap, gap)

        # Precursor: one slow step is not a fault, but it is the thing that
        # grows into a dropped link, so say so the moment it appears.
        if step_dur > 0.5 * self.control_dt and self._slow_step_warns < 20:
            self._slow_step_warns += 1
            print(f"[SLOW-STEP] t={now - self._trace_t0:7.2f}s  compute "
                  f"{step_dur*1e3:5.1f} ms of the {self.control_dt*1e3:.0f} ms budget")

        if now - self._last_health_print >= self._health_print_s:
            self._last_health_print = now
            age = (self.last_perception_snapshot or {}).get("body", {}).get("age_ms")
            print(
                f"[HEALTH] t={now - self._trace_t0:7.2f}s | "
                f"compute max {self._window_max_dur*1e3:5.1f} ms | "
                f"period max {self._window_max_gap*1e3:5.1f} ms | "
                f"arm {self._arm_achieved_hz:4.0f} Hz stall {self._window_arm_stall*1e3:4.0f} ms | "
                f"fsm {self.z1.get_fsm_state()} | "
                f"mocap {age if age is not None else -1:.0f} ms | "
                f"armerr {np.degrees(self._window_track):4.1f} deg"
                + (f" | LIMIT-HIT {self._arm_clamp_counts.sum()}"
                   if self._arm_clamp_counts.any() else "")
            )
            self._window_max_dur = 0.0
            self._window_max_gap = 0.0
            self._window_arm_stall = 0.0
            self._window_track = 0.0

        if not ok:
            self._dump_trace(reason)
            # Damping protection drives the arm itself, so release comms first.
            self._stop_arm_thread()
        return ok, reason

    def _dump_trace(self, reason: str) -> None:
        """
        What was this process doing in the seconds before protection tripped?

        The controller drops the arm after repeated 20 ms UDP recv timeouts, so
        the question is whether we stalled the machine that long. Loop periods
        and step durations answer it directly; if they are clean, the stall was
        not ours and the link itself is suspect.
        """
        if self._trace_n == 0:
            return

        idx = (np.arange(self._trace_n) + self._trace_i - self._trace_n) % self._trace_len
        tr = self._trace[idx]
        t, dur, gap, arm_hz, mocap_age, fsm = (tr[:, k] for k in range(6))

        print()
        print("=" * 100)
        print(f"[TRACE] protection tripped after {t[-1]:.1f} s: {reason}")
        print("=" * 100)

        # The controller drops the arm after repeated 20 ms UDP recv timeouts.
        # What can cause that is us HOLDING THE CPU that long -- not the loop
        # period, which is 20 ms at 50 Hz by design and spends most of it
        # asleep, leaving the controller free to run.
        LIMIT = 0.020
        budget = 0.8 * self.control_dt          # compute time saturating the step
        overrun = 1.5 * self.control_dt         # loop period that actually slipped

        over_compute = dur > budget
        over_period = gap > overrun
        print(f"  50 Hz loop period : mean {gap[1:].mean()*1e3:6.2f} ms | "
              f"max {gap.max()*1e3:6.1f} ms | {int(over_period.sum())} steps over "
              f"{overrun*1e3:.0f} ms (nominal {self.control_dt*1e3:.0f})")
        print(f"  step compute time : mean {dur.mean()*1e3:6.2f} ms | "
              f"max {dur.max()*1e3:6.1f} ms | {int(over_compute.sum())} steps over "
              f"{budget*1e3:.0f} ms   <- this is the CPU we hold")
        print(f"  arm thread rate   : min {arm_hz[arm_hz>0].min() if np.any(arm_hz>0) else 0:6.0f} Hz | "
              f"requested {self._arm_command_hz:g} Hz")
        print(f"  arm send stall    : worst {self._arm_max_stall_s*1e3:6.1f} ms "
              f"(controller times out at {LIMIT*1e3:.0f} ms)")
        print(f"  arm tracking err  : worst {np.degrees(self._arm_track_worst):.1f} deg "
              f"between commanded and measured")
        if self._arm_track_worst > self._arm_track_warn_rad:
            print("                      the arm was not holding its target. With the soft")
            print("                      runtime gains gravity drags it, and far enough drags")
            print("                      it into its stop -- which faults the arm.")

        if self._arm_clamp_counts.any():
            print(f"  arm limit clamps  : {self._arm_clamp_counts.tolist()} per joint | "
                  f"worst overshoot {np.degrees(self._arm_clamp_worst.max()):.2f} deg")
            print("                      the policy commanded past the hardware range; "
                  "an arm jammed")
            print("                      against its stop at clipped torque is a likely "
                  "cause of an arm-side fault.")

        valid_age = mocap_age[mocap_age >= 0]
        if valid_age.size:
            print(f"  mocap age         : mean {valid_age.mean():6.1f} ms | max {valid_age.max():6.1f} ms")

        print()
        print("  last 25 steps before the trip:")
        print(f"    {'t s':>8} {'period ms':>10} {'compute ms':>11} {'arm Hz':>8} "
              f"{'mocap ms':>9} {'fsm':>5}")
        for k in range(max(0, self._trace_n - 25), self._trace_n):
            flag = "  <-- OVER" if (dur[k] > budget or gap[k] > overrun) else ""
            print(f"    {t[k]:8.2f} {gap[k]*1e3:10.2f} {dur[k]*1e3:11.2f} "
                  f"{arm_hz[k]:8.0f} {mocap_age[k]:9.1f} {int(fsm[k]):5d}{flag}")

        print()
        if self._arm_max_stall_s > LIMIT or over_compute.sum() or over_period.sum():
            print(f"  READ: this process stalled past the controller's {LIMIT*1e3:.0f} ms UDP")
            print("        timeout. That is very likely why the arm link dropped. Shed load:")
            print("        visualizer_enabled: false, lower arm_command_hz, or give")
            print("        z1_controller real-time priority (chrt -f 80 ...).")
        else:
            print(f"  READ: timing stayed inside the controller's {LIMIT*1e3:.0f} ms budget")
            print("        throughout. The stall was NOT ours -- suspect the arm's own bus:")
            print("        cabling, connectors, EMI, or the controller process being starved")
            print("        by something else on the machine.")
        print("=" * 100)

    # ------------------------------------------------------------------
    # Background monitor
    # ------------------------------------------------------------------

    def start_monitor(self) -> None:
        """
        Bring the MuJoCo window up at setup and keep it fed in every phase.

        The inherited controller only pushes render state from the 50-Hz
        policy step, so the window would open just before the main loop and
        go static the moment the run left it -- exactly when you most want to
        look at it: the startup holds, the wait for A, and damping protection
        after a fault.

        This thread samples telemetry that is live regardless of phase --
        `low_state` is refreshed by its DDS callback, and the mocap snapshot
        by the NatNet thread -- so the view keeps updating no matter what the
        control code is doing. It reads only; nothing here mutates controller
        state, so it cannot perturb the policy.

        While the policy loop IS pushing, the monitor stands down (see
        `_monitor_takeover_s`), because those pushes carry strictly more --
        the decoded EE command and the resolved task state.
        """
        if self.visualizer is None or not self._monitor_enabled:
            return
        if self._monitor_thread is not None:
            return

        self.visualizer.start()

        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="mocap-scene-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        print(
            f"[B2WZ1-MOCAP] Scene monitor running at {self._monitor_hz:g} Hz "
            "(live through startup, policy and damping)."
        )

    def stop_monitor(self) -> None:
        # Same reasoning as _stop_arm_thread(): never raise from teardown.
        if getattr(self, "_monitor_thread", None) is None:
            return
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def _monitor_loop(self) -> None:
        period = 1.0 / max(self._monitor_hz, 1.0)

        while not self._monitor_stop.is_set():
            loop_start = time.perf_counter()

            try:
                idle = time.monotonic() - self._last_control_push
                if idle >= self._monitor_takeover_s:
                    state = self._monitor_state()
                    if state is not None:
                        self.visualizer.push_state(**state)
            except Exception as exc:  # noqa: BLE001
                # A debug view must never be able to take down a run.
                if not self._monitor_warned:
                    self._monitor_warned = True
                    print(f"[MONITOR][WARN] disabled after error: {exc!r}")
                    return

            remaining = period - (time.perf_counter() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)

    def _monitor_state(self) -> Optional[Dict[str, Any]]:
        """
        Build a render snapshot from phase-independent telemetry.

        Joint angles come straight from `low_state` rather than from the
        controller's cached copies, because not every startup phase refreshes
        those, whereas the DDS callback always does.
        """
        low = self.low_state
        if low is None or getattr(low, "tick", 0) == 0:
            return None

        motors = low.motor_state
        joints = np.empty(self.num_b2w_dof, dtype=np.float32)
        for policy_idx in range(self.num_b2w_dof):
            joints[policy_idx] = motors[
                self.hardware_to_policy_joint_indices[policy_idx]
            ].q

        raw_q = low.imu_state.quaternion
        quat = np.array(
            [raw_q[0], raw_q[1], raw_q[2], raw_q[3]], dtype=np.float32
        )
        norm = float(np.linalg.norm(quat))
        quat = (
            quat / norm
            if norm > 1.0e-8
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )

        state: Dict[str, Any] = {
            "base_quat_wxyz": quat,
            "base_height": float(self.base_height),
            "ground_z": float(self.ground_z),
            "b2w_joint_pos_policy": joints,
            "z1_q": np.asarray(self.z1.q, dtype=np.float32).copy(),
            "gripper_q_training": float(self._get_gripper_q_training()),
            "grasp_confidence_proxy": bool(self.grasp_confidence_proxy),
        }

        snap = self.perception.get_latest_snapshot()

        body = snap.get("body") or {}
        if body.get("valid"):
            state["base_pos_w"] = np.asarray(
                body["root_position_world"], dtype=np.float64
            )
            state["base_quat_wxyz"] = np.asarray(
                body["root_quat_wxyz"], dtype=np.float32
            )
            height = (snap.get("base_height") or {}).get("height_m")
            if height is not None:
                state["base_height"] = float(height)

        obj = snap.get("object") or {}
        if obj.get("valid"):
            state["object_pos_base"] = np.asarray(
                obj["position_base"], dtype=np.float32
            )

        ret = snap.get("retrieval") or {}
        if ret.get("valid"):
            state["retrieval_pos_base"] = np.asarray(
                ret["retrieval_target_base"], dtype=np.float32
            )

        state["mocap_points"] = self._mocap_pivot_markers(snap)
        return state

    # ------------------------------------------------------------------
    # Setup / run
    # ------------------------------------------------------------------

    def setup(self) -> None:
        print("=" * 108)
        print("B2WZ1 HIERARCHICAL RETRIEVAL SIM2REAL  --  MOCAP SENSING")
        print("=" * 108)
        print(f"Low policy       : {self.low_policy_path}")
        print(f"High policy      : {self.high_policy_path}")
        print(
            f"LL               : {1.0 / self.control_dt:.1f} Hz | "
            f"obs={self.ll_obs_dim} | act={self.ll_action_dim}"
        )
        print(
            f"HL               : {1.0 / self.hl_control_dt:.1f} Hz | "
            f"obs={self.hl_obs_dim} | act={self.hl_action_dim}"
        )
        print(f"LL/HL ratio      : {self.ll_steps_per_hl_step}")
        print("Sensing          : OPTITRACK MOCAP (no camera, no AprilTag, no VO)")

        perception = self.perception
        print(f"  {perception.body.describe()}")
        print(f"  {perception.object_body.describe()}")
        print(f"  {perception.retrieval_body.describe()}")

        offset = perception.root_offset
        roll, pitch, yaw = offset.rpy()
        print(
            "  root offset    : "
            f"pos={np.round(offset.pos, 4).tolist()} m | "
            f"rpy={np.round(np.degrees([roll, pitch, yaw]), 3).tolist()} deg | "
            + ("CALIBRATED" if perception.root_offset_calibrated else "IDENTITY")
        )
        print(f"  ground_z       : {self.ground_z:.4f} m")

        if not perception.root_offset_calibrated:
            print("-" * 108)
            print(
                "[WARN] Running with NO mocap->root calibration. The Motive "
                "rigid body's own frame is being used as the"
            )
            print(
                "       robot root, so the base pose is off by wherever Motive "
                "put that asset's pivot and axes."
            )
            print(
                "         - base_height is the ASSET's height, not the robot's. "
                "It feeds the PLB frame, so the"
            )
            print(
                "           commanded end-effector target is shifted by the "
                "same error."
            )
            print(
                "         - object and retrieval positions are expressed in the "
                "ASSET's frame. If that frame was"
            )
            print(
                "           defined aligned with the robot body this is small; "
                "any misalignment rotates or"
            )
            print("           translates both targets relative to the robot.")
            print(
                "       Fix with:  python3 deploy/b2w_mocap_root_calibration.py "
                "deploy/configs/b2w_mocap_root_calibration.yaml"
            )
            print("       then set mocap_root_offset_path to the YAML it writes.")
            print("-" * 108)

        if self.z1_arm_runtime_mode == "dcmotor":
            print("Arm runtime      : DCMOTOR (zero firmware gains, external tau_f)")
        else:
            print("Arm runtime      : POSITION_PD (Z1 firmware position loop)")

        print(
            "Gripper runtime  : DCMOTOR | "
            f"q_train = q_sdk - {self.z1.gripper_q_offset:.5f}"
        )
        if self._arm_thread_enabled:
            print(
                f"Arm command rate : {self._arm_command_hz:g} Hz PD thread "
                f"(policy target updated at {1.0 / self.control_dt:.0f} Hz)"
            )
        else:
            print(
                f"Arm command rate : {1.0 / self.control_dt:.0f} Hz, in the policy loop"
            )
        print(
            "Grasp proxy mode : "
            + (
                "COMMAND_ASSUMED (first CLOSE -> proxy=1, latched)"
                if self.grasp_proxy_mode == "command_assumed"
                else "HEURISTIC (gripper stall + hysteresis)"
            )
        )
        print(
            "Perception fail  : safe hold, then damping after "
            f"{self.perception_fault_timeout_s:.3f}s"
        )
        print("Base height      : MOCAP (absolute, no anchoring)")
        print(
            "Damping recovery : "
            + (
                "ENABLED | A in protection -> legs only, "
                f"{self.damping_recover_leg_duration_s:.1f}s (arm NOT moved)"
                if self.damping_leg_recovery_enabled
                else "DISABLED"
            )
        )
        print("=" * 108)

        self.wait_for_low_state()
        InitLowCmd(self.low_cmd)

        self.z1.connect()
        self._read_all_sensors_once()

        self._load_arm_limits()

        print("[B2WZ1-MOCAP] Connecting to Motive...")
        self.perception.initialize()
        self.perception.start()
        print("[B2WZ1-MOCAP] Mocap perception running.")

        self._configure_visualizer()
        self.start_monitor()

    def run(self) -> None:
        """Inherited run, with the monitor torn down on every exit path."""
        try:
            super().run()
        finally:
            self._stop_arm_thread()
            self.stop_monitor()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B2WZ1 hierarchical retrieval sim2real with mocap sensing."
    )
    parser.add_argument(
        "net",
        type=str,
        help="Unitree network interface, e.g. enxa0cec819e15f",
    )
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="deploy/configs/b2wz1_hl_retrieval_mocap.yaml",
        help="Path to the mocap hierarchical sim2real YAML.",
    )
    args = parser.parse_args()

    # Exactly once, as in the camera deployment.
    ChannelFactoryInitialize(0, args.net)

    controller = B2WZ1MocapRetrievalController(
        cfg_path=args.config,
        network_interface=args.net,
    )
    controller.run()


if __name__ == "__main__":
    main()
