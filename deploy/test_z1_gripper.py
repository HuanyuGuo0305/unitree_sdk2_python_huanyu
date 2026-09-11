#!/usr/bin/env python3
"""
Z1 gripper close/open test on the firmware position PD, one kp/kd pair per run.

The gripper is driven the way deploy/b2wz1_hl_retrieval_abs_uan_mocap.py drives
it: Z1ArmAdapter with z1_gripper_runtime_mode="position_pd", so z1_ctrl closes
the gripper loop at its own rate with z1_gripper_kp / z1_gripper_kd, and the
target is a STEP in training coordinates (closed = 0, open = -pi/2), like the
binary HL gripper action. The arm only holds the pose it had at connect.

With the gripper gains from --kp / --kd:
    switch to them while holding OPEN for 1 s            (not scored)
    --cycles times: step CLOSE, hold --close-s; step OPEN, hold --open-s

CLOSE defaults to gripper_close_pos from the YAML; --close_angle overrides it
(0 = fully closed, -0.78 rad ~ -45 deg).

Each scored step reports:
    t90        time to cover 90 % of the travel
    settle     time after which |q - target| stays within 0.03 rad
    overshoot  peak excursion past the target
    short      final shortfall (+ = stopped short of the target, - = past it)
    peak|qd|, peak|tau|
    hold|tau|  mean |tau| over the last 0.25 s: the torque held against a stop

Traces and metrics are written to Log/z1_gripper_test/<timestamp>_kp<kp>_kd<kd>/.

The run stops when:
    the Z1 leaves LOWCMD, or lowstate stops changing  -> nothing more is sent
    the gripper stalls: |tau| >= --stall-tau while |qd| < 0.2 rad/s for 0.3 s
                                                      -> relax at the measured
                                                         position, then park

z1_ctrl must be running. Example:

    python3 deploy/test_z1_gripper.py --kp 20 --kd 10000 --cycles 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import yaml

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_DEPLOY_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.z1_helper import Z1ArmAdapter  # noqa: E402


DEFAULT_CONFIG = os.path.join(
    _DEPLOY_DIR, "configs", "b2wz1_hl_retrieval_abs_uan_mocap.yaml"
)
LOG_ROOT = os.path.join(PROJECT_ROOT, "Log", "z1_gripper_test")

PREFLIGHT_S = 0.5      # hold before any gripper motion; a dead link shows up here
GAIN_SWITCH_S = 1.0    # hold at OPEN after the gain switch, not scored
PARK_S = 0.5           # hold after the YAML gripper gains are restored
SETTLE_TOL_RAD = 0.03
TAIL_S = 0.25          # end-of-hold window for `short` and `hold|tau|`
FREEZE_S = 0.10        # lowstate bit-identical this long = arm link frozen
STALL_QD = 0.2         # rad/s
STALL_S = 0.30
MAX_HOLD_NUDGE_RAD = 0.10  # same startup jump Z1ArmAdapter.connect() accepts

TRACE_KEYS = ("t", "gripper_q_sdk", "gripper_qd", "gripper_tau", "arm_q")
SEG_META_KEYS = ("name", "cycle", "kp", "kd", "target", "q_start", "t_start", "complete")


class GripperAbort(RuntimeError):
    """Stop the run. arm_alive=False means the Z1 must not be commanded."""

    def __init__(self, msg: str, arm_alive: bool) -> None:
        super().__init__(msg)
        self.arm_alive = arm_alive


def sleep_until(deadline: float) -> float:
    """Sleep to `deadline`. Returns how late we already were, 0.0 if on time."""
    remaining = deadline - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)
        return 0.0
    return -remaining


def scalar(value) -> float:
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(v[0]) if v.size else float("nan")


def fmt(value: Optional[float], spec: str = ".3f", unit: str = "") -> str:
    return "-" if value is None else format(value, spec) + unit


def mean_str(values: List[Optional[float]], spec: str = ".3f") -> str:
    """Mean of the known values; '*' marks that some cycles had none."""
    known = [v for v in values if v is not None]
    if not known:
        return "-"
    text = format(float(np.mean(known)), spec)
    return text + "*" if len(known) < len(values) else text


def step_metrics(seg: Dict) -> Dict:
    """Step-response numbers for one close/open step, training coordinates."""
    t = seg["t"] - seg["t_start"]
    q = seg["gripper_q_train"]
    target, q_start = seg["target"], seg["q_start"]
    tail = t >= t[-1] - TAIL_S

    t90 = None
    overshoot = 0.0
    travel = target - q_start
    if abs(travel) > 1e-3:
        progress = (q - q_start) / travel
        reached = np.flatnonzero(progress >= 0.9)
        t90 = float(t[reached[0]]) if reached.size else None
        overshoot = float(max(0.0, progress.max() - 1.0) * abs(travel))

    outside = np.flatnonzero(np.abs(q - target) > SETTLE_TOL_RAD)
    if outside.size == 0:
        settle = float(t[0])
    elif outside[-1] + 1 < t.size:
        settle = float(t[outside[-1] + 1])
    else:
        settle = None  # still outside the band when the hold ended

    return {
        "t90_s": t90,
        "settle_s": settle,
        "overshoot_rad": overshoot,
        "short_rad": float(np.mean(seg["direction"] * (target - q[tail]))),
        "peak_qd": float(np.max(np.abs(seg["gripper_qd"]))),
        "peak_tau": float(np.max(np.abs(seg["gripper_tau"]))),
        "hold_tau": float(np.mean(np.abs(seg["gripper_tau"][tail]))),
        "max_tick_ms": float(np.max(np.diff(seg["t"])) * 1e3) if t.size > 1 else None,
    }


class GripperGainTest:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

        # This test is about the firmware position PD, whatever the YAML says.
        cfg["z1_gripper_runtime_mode"] = "position_pd"
        cfg["z1_arm_runtime_mode"] = "position_pd"
        # The adapter debug line fires every 100 sends, i.e. 5x/s at 500 Hz.
        cfg["z1_debug_print"] = False
        self.cfg = cfg

        self.z1 = Z1ArmAdapter(cfg, PROJECT_ROOT)
        self.lowcmd_state = self.z1.unitree_arm_interface.ArmFSMState.LOWCMD
        self.default_kp = float(self.z1.gripper_kp)
        self.default_kd = float(self.z1.gripper_kd)

        self.kp = self.default_kp if args.kp is None else float(args.kp)
        self.kd = self.default_kd if args.kd is None else float(args.kd)
        if min(self.kp, self.kd) < 0.0:
            raise SystemExit("[GRIP] gains must be >= 0.")

        self.close_pos = float(
            cfg.get("gripper_close_pos", 0.0) if args.close_pos is None else args.close_pos
        )
        self.open_pos = float(
            cfg.get("gripper_open_pos", -0.5 * np.pi) if args.open_pos is None else args.open_pos
        )
        lo = self.z1.gripper_travel_min_training
        hi = self.z1.gripper_travel_max_training
        for flag, pos in (("--close_angle", self.close_pos), ("--open-pos", self.open_pos)):
            if not lo - 1e-6 <= pos <= hi + 1e-6:
                raise SystemExit(
                    f"[GRIP] {flag} {pos:+.4f} is outside the gripper travel "
                    f"[{lo:+.4f}, {hi:+.4f}] rad; the adapter would silently clamp it."
                )
        if abs(self.close_pos - self.open_pos) < 1e-3:
            raise SystemExit("[GRIP] close and open positions coincide.")
        # +1 if closing increases q; signs `short`.
        self.close_dir = float(np.sign(self.close_pos - self.open_pos))

        self.loop_hz = float(
            cfg.get("hardware_command_hz", 500.0) if args.loop_hz is None else args.loop_hz
        )
        if self.loop_hz <= 0.0:
            raise SystemExit("[GRIP] --loop-hz must be > 0.")
        if args.cycles < 1 or min(args.close_s, args.open_s) <= TAIL_S:
            raise SystemExit(f"[GRIP] need --cycles >= 1 and holds longer than {TAIL_S} s.")
        self.dt = 1.0 / self.loop_hz
        self.use_startup_gains = args.arm_gains == "startup"

        self.segments: List[Dict] = []
        self.q_hold: Optional[np.ndarray] = None
        self.session_dir = ""
        self.t_session = 0.0
        self._last_sig = None
        self._fresh_t = 0.0

    # -- plan -------------------------------------------------------------

    def print_plan(self) -> None:
        a = self.args
        kp_scale = float(self.cfg.get("z1_kp_protocol_scale", 25.6))
        kd_scale = float(self.cfg.get("z1_kd_protocol_scale", 0.0128))
        total_s = PREFLIGHT_S + GAIN_SWITCH_S + a.cycles * (a.close_s + a.open_s) + PARK_S
        stall = (
            f"|tau| >= {a.stall_tau:g} Nm with |qd| < {STALL_QD:g} rad/s for {STALL_S:g} s"
            if a.stall_tau > 0.0
            else "DISABLED"
        )
        print("\n========== Z1 gripper position-PD gain test ==========")
        print(f"config     : {a.config}")
        print(
            f"gripper    : kp={self.kp:g} kd={self.kd:g}  (~{self.kp * kp_scale:.0f} Nm/rad, "
            f"~{self.kd * kd_scale:.1f} Nm*s/rad at the arm protocol scale)"
        )
        print(
            f"targets    : close {self.close_pos:+.4f} rad ({np.degrees(self.close_pos):+.1f} deg), "
            f"open {self.open_pos:+.4f} rad ({np.degrees(self.open_pos):+.1f} deg) "
            f"(training; sdk = training {self.z1.gripper_q_offset:+.5f})"
        )
        print(
            f"schedule   : {GAIN_SWITCH_S:g} s at open after the gain switch, then {a.cycles} x "
            f"[close {a.close_s:g} s -> open {a.open_s:g} s], ~{total_s:.0f} s total"
        )
        print(f"loop       : {self.loop_hz:g} Hz; arm holds its connect pose on {a.arm_gains} gains")
        print(f"stall stop : {stall}")
        print(
            f"afterwards : gripper held where it ends, gains restored to "
            f"kp={self.default_kp:g} kd={self.default_kd:g}"
        )

    # -- top level ----------------------------------------------------------

    def run(self) -> None:
        self.print_plan()
        if not self.args.yes:
            print(
                "\nThe gripper will close and open repeatedly while the arm holds its "
                "current pose.\nKeep hands clear of the jaws. If the Z1 drops to "
                "PASSIVE the arm goes limp, so park or support it first."
            )
            if input("Type 'go' to start: ").strip().lower() != "go":
                print("[GRIP] aborted before any motion.")
                return

        self.z1.connect()
        self.q_hold = self.z1.q.copy()
        self._require_holdable_pose()

        self.session_dir = os.path.join(
            LOG_ROOT, f"{time.strftime('%Y%m%d_%H%M%S')}_kp{self.kp:g}_kd{self.kd:g}"
        )
        self.t_session = self._fresh_t = time.perf_counter()

        abort_reason: Optional[str] = None
        try:
            self._run_segment("preflight", self._measured_gripper_target(), PREFLIGHT_S)
            self._run_trial()
        except GripperAbort as e:
            abort_reason = str(e)
            print(f"\n[GRIP][ABORT] {e}")
            if e.arm_alive:
                self._park()
            else:
                self._print_dead_arm_help()
        except KeyboardInterrupt:
            abort_reason = "interrupted by user"
            print("\n[GRIP] interrupted by user.")
            self._park()
        except Exception as e:  # noqa: BLE001 - relax the gripper before re-raising
            abort_reason = f"{type(e).__name__}: {e}"
            print(f"\n[GRIP][ERROR] {abort_reason}")
            self._park()
            raise
        else:
            self._park()
        finally:
            self._print_summary(abort_reason)
            self._save(abort_reason)

    def _run_trial(self) -> None:
        a = self.args
        print(f"\n[GRIP] ===== gripper kp={self.kp:g} kd={self.kd:g} =====")
        self._apply_gripper_gains(self.kp, self.kd)
        self._run_segment("gain_switch", self.open_pos, GAIN_SWITCH_S)
        self._require_gripper_gains_sent()
        steps = (("close", self.close_pos, a.close_s), ("open", self.open_pos, a.open_s))
        for cycle in range(a.cycles):
            for name, target, duration_s in steps:
                self._report_segment(self._run_segment(name, target, duration_s, cycle=cycle))

    # -- hardware -----------------------------------------------------------

    def _apply_gripper_gains(self, kp: float, kd: float) -> None:
        self.z1.gripper_kp = float(kp)
        self.z1.gripper_kd = float(kd)
        # set_control_gain() only compares the ARM gains against its cache, so
        # invalidate it; the new gripper gains go out with the next packet.
        self.z1._last_applied_kp = None
        self.z1._last_applied_kd = None

    def _require_gripper_gains_sent(self) -> None:
        """Read back the gripper slot of the SDK command, which is what z1_ctrl receives."""
        kp_sent = float(self.z1.lowcmd.kp[-1])
        kd_sent = float(self.z1.lowcmd.kd[-1])
        if not (np.isclose(kp_sent, self.kp) and np.isclose(kd_sent, self.kd)):
            raise GripperAbort(
                f"the SDK command carries gripper kp={kp_sent:g} kd={kd_sent:g}, not the "
                f"requested kp={self.kp:g} kd={self.kd:g}, so the steps would not test them.",
                arm_alive=True,
            )
        print(f"[GRIP] SDK command carries gripper kp={kp_sent:g} kd={kd_sent:g}")

    def _measured_gripper_target(self) -> float:
        """The current gripper position as a command: zero PD error, ~zero torque."""
        return float(
            np.clip(
                self.z1.get_gripper_q_training(),
                self.z1.gripper_travel_min_training,
                self.z1.gripper_travel_max_training,
            )
        )

    def _require_holdable_pose(self) -> None:
        """Hold the pose jointProtect() will actually command, if it is close enough.

        Every adapter send clamps q into the SDK range. A folded Z1 at rest sits a
        few hundredths of a rad past it (j2 < 0, j3 > 0), so the first command
        nudges the arm back inside; anything beyond MAX_HOLD_NUDGE_RAD is refused.
        """
        try:
            q_safe, _ = self.z1.arm_model.jointProtect(
                self.q_hold.copy(), np.zeros(6, dtype=np.float32)
            )
        except Exception:  # noqa: BLE001 - the adapter then sends q unclamped too
            return
        q_safe = np.asarray(q_safe, dtype=np.float32).reshape(6)
        step = q_safe - self.q_hold
        clamped = np.flatnonzero(np.abs(step) > 1e-4)
        if clamped.size == 0:
            return
        detail = ", ".join(
            f"joint{j + 1} {self.q_hold[j]:+.3f} -> {q_safe[j]:+.3f}" for j in clamped
        )
        if float(np.max(np.abs(step))) > MAX_HOLD_NUDGE_RAD:
            raise SystemExit(
                "[GRIP] the arm is parked too far outside the SDK joint range to hold "
                f"({detail} rad, limit {MAX_HOLD_NUDGE_RAD} rad). Bring it back inside "
                "the range with the Z1 controller first."
            )
        print(f"[GRIP] arm rests just outside the SDK range; the hold nudges it back in: {detail} rad")
        self.q_hold = q_safe

    def _run_segment(
        self,
        name: str,
        target: float,
        duration_s: float,
        cycle: Optional[int] = None,
    ) -> Dict:
        """Hold one gripper target and record every tick. cycle=None: unscored."""
        z1 = self.z1
        stall_tau = self.args.stall_tau
        seg: Dict = {
            "name": name,
            "cycle": cycle,
            "kp": float(z1.gripper_kp),
            "kd": float(z1.gripper_kd),
            "target": float(target),
            "q_start": float(z1.get_gripper_q_training()),
            "direction": self.close_dir if name == "close" else -self.close_dir,
            "complete": False,
        }
        self.segments.append(seg)
        rec: Dict[str, list] = {key: [] for key in TRACE_KEYS}
        where = f"'{name}' at kp={seg['kp']:g} kd={seg['kd']:g}"

        n = max(1, int(round(duration_s * self.loop_hz)))
        stall_t: Optional[float] = None
        t0 = time.perf_counter()
        seg["t_start"] = t0 - self.t_session
        try:
            for i in range(n):
                z1.track_target_pd_runtime_once(
                    q_target=self.q_hold,
                    gripper_q_target_training=target,
                    use_startup_gains=self.use_startup_gains,
                )
                now = time.perf_counter()
                tau = scalar(z1.arm.lowstate.getGripperTau())

                rec["t"].append(now - self.t_session)
                rec["gripper_q_sdk"].append(z1.gripper_q)
                rec["gripper_qd"].append(z1.gripper_qd)
                rec["gripper_tau"].append(tau)
                rec["arm_q"].append(z1.q.copy())

                fsm = z1.get_fsm_state()
                if fsm != self.lowcmd_state:
                    raise GripperAbort(f"Z1 left LOWCMD (fsm={fsm}) during {where}.", arm_alive=False)

                # A live encoder never repeats bit-identically for long; a frozen
                # lowstate is the SDK replaying the last packet of a dead link.
                sig = (z1.q.tobytes(), z1.qd.tobytes(), z1.tau.tobytes(), z1.gripper_q, z1.gripper_qd, tau)
                if sig != self._last_sig:
                    self._last_sig, self._fresh_t = sig, now
                elif now - self._fresh_t >= FREEZE_S:
                    raise GripperAbort(
                        f"lowstate frozen for {(now - self._fresh_t) * 1e3:.0f} ms during "
                        f"{where}: the arm link is down.",
                        arm_alive=False,
                    )

                stalled = stall_tau > 0.0 and abs(tau) >= stall_tau and abs(z1.gripper_qd) < STALL_QD
                if not stalled:
                    stall_t = None
                elif stall_t is None:
                    stall_t = now
                elif now - stall_t >= STALL_S:
                    raise GripperAbort(
                        f"gripper stalled at |tau|={abs(tau):.1f} Nm for {STALL_S:g} s during "
                        f"{where}: q={z1.get_gripper_q_training():+.3f}, target {target:+.3f} rad. "
                        "Raise --stall-tau (0 disables) if this torque is acceptable.",
                        arm_alive=True,
                    )

                sleep_until(t0 + (i + 1) * self.dt)
            seg["complete"] = True
        finally:
            for key, values in rec.items():
                seg[key] = np.asarray(values, dtype=np.float64)
            seg["gripper_q_train"] = seg["gripper_q_sdk"] - z1.gripper_q_offset
            if cycle is not None and seg["t"].size:
                seg["metrics"] = step_metrics(seg)
        return seg

    def _park(self) -> None:
        """Restore the YAML gripper gains, unless the arm is already limp.

        The gripper is held where it is: any other target would be driven
        there on the restored, typically much stiffer, gains.
        """
        try:
            if self.z1.get_fsm_state() != self.lowcmd_state:
                # Commanding a PASSIVE arm does nothing but block on UDP.
                self._print_dead_arm_help()
                return
            gripper_target = self._measured_gripper_target()
            self._apply_gripper_gains(self.default_kp, self.default_kd)
            n = max(1, int(round(PARK_S * self.loop_hz)))
            t0 = time.perf_counter()
            for i in range(n):
                self.z1.track_target_pd_runtime_once(
                    q_target=self.q_hold,
                    gripper_q_target_training=gripper_target,
                    use_startup_gains=self.use_startup_gains,
                )
                sleep_until(t0 + (i + 1) * self.dt)
            print(
                f"[GRIP] parked: gripper gains back to kp={self.default_kp:g} "
                f"kd={self.default_kd:g}, gripper held at {gripper_target:+.3f} rad."
            )
        except KeyboardInterrupt:
            print("[GRIP][WARN] park interrupted; the test gripper gains may still be applied.")
        except Exception as e:  # noqa: BLE001
            print(f"[GRIP][ERROR] park failed: {type(e).__name__}: {e}")

    def _print_dead_arm_help(self) -> None:
        print(
            f"[GRIP] The Z1 cannot be commanded (last reported fsm={self.z1.get_fsm_state()}); nothing "
            "more is sent. If it went PASSIVE it is limp, so support it.\n"
            "[GRIP] Read the cause in the z1_ctrl terminal (arm link timeout, motor lost "
            "connection, overheat), then restart z1_ctrl before rerunning."
        )

    # -- reporting ----------------------------------------------------------

    def _report_segment(self, seg: Dict) -> None:
        m = seg["metrics"]
        print(
            f"[GRIP] kp={seg['kp']:g} kd={seg['kd']:g} cycle {seg['cycle'] + 1}/{self.args.cycles} "
            f"{seg['name']:<5} | t90={fmt(m['t90_s'], unit='s')} "
            f"settle={fmt(m['settle_s'], unit='s')} "
            f"overshoot={m['overshoot_rad']:.3f} short={m['short_rad']:+.3f} rad | "
            f"peak|qd|={m['peak_qd']:.2f} rad/s | peak|tau|={m['peak_tau']:.1f} "
            f"hold|tau|={m['hold_tau']:.1f} Nm | max_dt={fmt(m['max_tick_ms'], '.1f', 'ms')}"
        )

    def _print_summary(self, abort_reason: Optional[str]) -> None:
        done = [s for s in self.segments if s["complete"] and "metrics" in s]
        if done:
            rows = []
            for name in ("close", "open"):
                ms = [s["metrics"] for s in done if s["name"] == name]
                if not ms:
                    continue

                def col(key: str, spec: str = ".3f") -> str:
                    return mean_str([m[key] for m in ms], spec)

                rows.append(
                    f"{name:<5} {len(ms):>2} {col('t90_s'):>8} {col('settle_s'):>9} "
                    f"{col('overshoot_rad'):>8} {col('short_rad', '+.3f'):>10} "
                    f"{col('peak_qd', '.2f'):>7} {col('peak_tau', '.1f'):>7} "
                    f"{col('hold_tau', '.1f'):>9}"
                )
            print(
                f"\n========== gripper kp={self.kp:g} kd={self.kd:g}: "
                "mean over complete cycles =========="
            )
            print(
                f"{'step':<5} {'n':>2} {'t90[s]':>8} {'settle[s]':>9} {'ovs[rad]':>8} "
                f"{'short[rad]':>10} {'pk|qd|':>7} {'pk|tau|':>7} {'hold|tau|':>9}"
            )
            print("\n".join(rows))
            if any("*" in row for row in rows):
                print("  * some cycles never got there; the mean covers the rest.")
        if abort_reason:
            print(f"[GRIP] run stopped early: {abort_reason}")

    def _save(self, abort_reason: Optional[str]) -> None:
        segs = [s for s in self.segments if s["t"].size]
        if not segs:
            return
        os.makedirs(self.session_dir, exist_ok=True)

        def cat(key: str) -> np.ndarray:
            return np.concatenate([s[key] for s in segs])

        seg_id = np.concatenate(
            [np.full(s["t"].size, i, dtype=np.int32) for i, s in enumerate(segs)]
        )
        seg_target = np.array([s["target"] for s in segs])
        trace_path = os.path.join(self.session_dir, "trace.npz")
        np.savez_compressed(
            trace_path,
            t=cat("t"),
            seg_id=seg_id,
            gripper_q_des_train=seg_target[seg_id],
            gripper_q_train=cat("gripper_q_train"),
            gripper_q_sdk=cat("gripper_q_sdk"),
            gripper_qd=cat("gripper_qd"),
            gripper_tau=cat("gripper_tau"),
            arm_q=cat("arm_q"),
            seg_name=np.array([s["name"] for s in segs]),
            seg_kp=np.array([s["kp"] for s in segs]),
            seg_kd=np.array([s["kd"] for s in segs]),
            seg_cycle=np.array([-1 if s["cycle"] is None else s["cycle"] for s in segs]),
            seg_target=seg_target,
            seg_t_start=np.array([s["t_start"] for s in segs]),
            seg_complete=np.array([s["complete"] for s in segs]),
        )

        summary = {
            "config": os.path.abspath(self.args.config),
            "args": vars(self.args),
            "abort_reason": abort_reason,
            "loop_hz": self.loop_hz,
            "arm_q_hold": self.q_hold.tolist(),
            "gripper_q_offset": self.z1.gripper_q_offset,
            "close_pos_train": self.close_pos,
            "open_pos_train": self.open_pos,
            "yaml_gripper_gains": {"kp": self.default_kp, "kd": self.default_kd},
            "gripper_gains": {"kp": self.kp, "kd": self.kd},
            "segments": [
                dict({key: s[key] for key in SEG_META_KEYS}, metrics=s.get("metrics"))
                for s in segs
            ],
        }
        summary_path = os.path.join(self.session_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"[GRIP] wrote {trace_path}")
        print(f"[GRIP] wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Z1 gripper close/open test on the firmware position PD, one kp/kd pair per run."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="YAML the Z1 adapter is built from (gripper offset, travel, arm gains).",
    )
    parser.add_argument(
        "--kp", type=float, default=None,
        help="Raw firmware gripper kp. Default: z1_gripper_kp from the YAML.",
    )
    parser.add_argument(
        "--kd", type=float, default=None,
        help="Raw firmware gripper kd. Default: z1_gripper_kd from the YAML.",
    )
    parser.add_argument("--cycles", type=int, default=3, help="Close/open cycles.")
    parser.add_argument("--close-s", type=float, default=2.0, help="Hold after each close step [s].")
    parser.add_argument("--open-s", type=float, default=2.0, help="Hold after each open step [s].")
    parser.add_argument(
        "--close_angle", "--close-pos", dest="close_pos", type=float, default=None,
        help="Close target, training coordinates [rad]: 0 = fully closed, -0.78 ~ -45 deg. "
             "Default: gripper_close_pos.",
    )
    parser.add_argument(
        "--open-pos", type=float, default=None,
        help="Open target, training coordinates [rad]. Default: gripper_open_pos.",
    )
    parser.add_argument(
        "--loop-hz", type=float, default=None,
        help="Command rate [Hz]. Default: hardware_command_hz from the YAML, else 500.",
    )
    parser.add_argument(
        "--arm-gains", choices=("startup", "runtime"), default="startup",
        help="Firmware gains the arm holds its pose with.",
    )
    parser.add_argument(
        "--stall-tau", type=float, default=25.0,
        help="Stop when |gripper tau| stays at or above this [Nm] while not moving. 0 disables.",
    )
    parser.add_argument("-y", "--yes", action="store_true", help="Skip the pre-motion confirmation.")
    args = parser.parse_args()

    np.set_printoptions(precision=3, suppress=True)
    GripperGainTest(args).run()


if __name__ == "__main__":
    main()
