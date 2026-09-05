"""
Z1 hardware data collection for an Unsupervised Actuator Net (UAN).

Step 1 of reproducing "Bridging the Sim-to-Real Gap for Athletic
Loco-Manipulation" (arXiv:2502.10894) on this robot: drive the real Z1 with
excitation signals that cover the state space, and log the transitions
{(s_t, tau_t, s_t+1)} the UAN is trained against.

WHAT IS COMMANDED
-----------------
Exactly the paper's three action sequences (Section II-A.2):

    square waves   one joint at a time, the other five held at home,
                   12 amplitude x frequency combinations, ~50 s per joint
    sine waves     same sweep, same 12 combinations, ~50 s per joint
    gaussian noise all six joints simultaneously, each joint redrawing its
                   target after its own uniform 5-400 ms hold, ~5 min

ACTUATION
---------
The collector uses only the deployed Z1 position-PD path:

    position_pd   The Z1 firmware closes the position loop with
                  arm_kps_runtime / arm_kds_runtime and tau_f = 0. The
                  script sends q_des at the collection loop rate. The log
                  carries q_des plus the firmware gains; a replaying sim
                  uses its own physical PD gains, e.g.
                      tau = Kp_sim * (q_des - q) - Kd_sim * qd,
                  and the UAN learns the remaining sim-to-real residual.

OUTPUT
------
    <output_root>/<timestamp>/metadata.json
    <output_root>/<timestamp>/square_sine_log.pkl
    <output_root>/<timestamp>/noise_log.pkl
    <output_root>/<timestamp>/*.npz                 (same data, convenient)

The .pkl files use the schema that athletic-loco-manipulation's
omniisaacgymenvs.utils.utils.load_hardware_data() reads, with the same file
names it expects, so they can be pointed at directly:

    data['arm_pd_tau_targets'] : q_des (N,6), gripperQ_des (N,), kp (N,7),
                                 kd (N,7), timestamp (N,) microseconds
    data['arm_control_data']   : q (N,7), qd (N,7), tau_est (N,7),
                                 timestamp (N,) microseconds
    data['uan_meta']           : PD gains, segment table, config echo

Column 6 of the arm_control_data arrays is the gripper, matching the 7-DOF
layout that loader assumes.

SETUP THIS SCRIPT ASSUMES
-------------------------
    - Z1 on a fixed stand, base LEVEL and world-aligned. The replaying sim
      must fix the arm base at identity orientation to match the gravity
      direction seen here.
    - Nothing in the gripper; it is held at a constant position target
      through the firmware servo and is never excited.
    - A clear workspace. The sweep is large and fast.

Run (from the repository root):

    # inspect the whole plan, touch no hardware
    python3 deploy/z1_uan_data_collection.py --dry-run

    # collect
    python3 deploy/z1_uan_data_collection.py

    # one phase only
    python3 deploy/z1_uan_data_collection.py --phases noise

Ctrl+C at any time ramps the arm back to home under firmware gains and
saves everything recorded so far.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

_DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DEPLOY_DIR, ".."))

for _p in (_PROJECT_ROOT, _DEPLOY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import yaml

from utils.z1_helper import Z1ArmAdapter


NUM_ARM_JOINTS = 6
ACTUATOR_MODE = "position_pd"

# Hard limits the Z1 SDK itself enforces, read from Z1Model.getJointQMin() /
# getJointQMax() / getJointSpeedMax(). These are NOT the limits in the
# paper's URDF -- that file is a modified model and is looser on joints 3,
# 4 and 5 -- and the SDK's are the ones that matter, because ArmModel's
# jointProtect() silently clamps every command into this box. A target
# outside it does not reach the arm; it becomes a step to the boundary.
# Verified against the live model at connect().
Z1_Q_LOWER = np.array([-2.618, 0.0, -2.886, -1.518, -1.344, -2.793])
Z1_Q_UPPER = np.array([2.618, 2.88, 0.0, 1.518, 1.344, 2.793])
Z1_QD_MAX = 3.142


class SafetyAbort(RuntimeError):
    """A safety monitor tripped; the session must unwind."""


# ---------------------------------------------------------------------
# Excitation plan
# ---------------------------------------------------------------------


@dataclass
class Segment:
    """One contiguous block of the excitation plan."""

    name: str
    kind: str                      # "square" | "sine" | "noise" | "transition"
    joint: int                     # excited joint, or -1 for all / none
    amplitude: float               # rad, 0 for noise and transitions
    frequency_hz: float            # 0 for noise and transitions
    num_ticks: int                 # length in target-update ticks
    start_tick: int = 0

    def as_dict(self) -> Dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "joint": int(self.joint),
            "amplitude": float(self.amplitude),
            "frequency_hz": float(self.frequency_hz),
            "start_tick": int(self.start_tick),
            "num_ticks": int(self.num_ticks),
        }


@dataclass
class Plan:
    """A precomputed target trajectory plus the segment table describing it."""

    phase: str
    target_update_hz: float
    targets: np.ndarray            # (K, 6) rad
    seg_ids: np.ndarray            # (K,) int
    segments: List[Segment] = field(default_factory=list)

    @property
    def num_ticks(self) -> int:
        return int(self.targets.shape[0])

    @property
    def duration_s(self) -> float:
        return self.num_ticks / self.target_update_hz


def _cosine_blend(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Smooth (zero end-slope) blend from a to b over n ticks, excluding a."""
    if n <= 0:
        return np.zeros((0, a.shape[0]), dtype=np.float64)
    tau = (np.arange(1, n + 1, dtype=np.float64) / float(n))[:, None]
    s = 0.5 * (1.0 - np.cos(np.pi * tau))
    return a[None, :] * (1.0 - s) + b[None, :] * s


class PlanBuilder:
    """Turns the YAML excitation description into concrete target sequences."""

    def __init__(self, cfg: Dict) -> None:
        u = cfg["uan"]
        self.home = np.asarray(u["home_q"], dtype=np.float64).reshape(NUM_ARM_JOINTS)
        self.q_min = np.asarray(u["q_min"], dtype=np.float64).reshape(NUM_ARM_JOINTS)
        self.q_max = np.asarray(u["q_max"], dtype=np.float64).reshape(NUM_ARM_JOINTS)
        self.global_target_hz = float(u["target_update_hz"])
        self.ss = u["square_sine"]
        self.noise = u["noise"]

        if np.any(self.q_min >= self.q_max):
            raise ValueError("uan.q_min must be strictly below uan.q_max.")
        if np.any(self.q_min < Z1_Q_LOWER) or np.any(self.q_max > Z1_Q_UPPER):
            bad = np.flatnonzero(
                (self.q_min < Z1_Q_LOWER) | (self.q_max > Z1_Q_UPPER)
            )
            detail = "\n".join(
                f"    joint{j + 1}: box [{self.q_min[j]:+.3f}, "
                f"{self.q_max[j]:+.3f}] vs SDK "
                f"[{Z1_Q_LOWER[j]:+.3f}, {Z1_Q_UPPER[j]:+.3f}]"
                for j in bad
            )
            raise ValueError(
                "uan.q_min/q_max fall outside the limits the Z1 SDK enforces, "
                "so jointProtect() would clamp those targets and the "
                "excitation would not be what the log says:\n" + detail
            )
        if np.any(self.home < self.q_min) or np.any(self.home > self.q_max):
            raise ValueError("uan.home_q is outside the soft box uan.q_min/q_max.")

    # -- amplitude resolution ------------------------------------------

    def _torque_gain(self, shape: str, freq_idx: int) -> float:
        """Measured peak torque per (kp_effective * amplitude) for this cell.

        Purely empirical, because the static model kp*2A is wrong twice over:

          - Overshoot. The loop is very lightly damped, so the joint sails
            past the target and the NEXT edge starts from further away than
            2A. Measured peak error reached 1.3x the commanded step.
          - Resonance. Joint2's natural frequency sits near the top of the
            sweep, so at 1.2 Hz a sine that needs 0.8x the static torque at
            0.25 Hz needs 1.7x. This is why sine cells -- which used to have
            no torque budget at all -- were the ones that peaked at 67.5 Nm
            on a 60 Nm joint.

        Values come from measuring peak torque per cell on real runs.
        """
        key = "square_torque_gain" if shape == "square" else "sine_torque_gain"
        gains = self.ss[key]
        return float(gains[min(freq_idx, len(gains) - 1)])

    def _max_amplitude(
        self, joint: int, shape: str, freq_hz: float, freq_idx: int
    ) -> float:
        """Largest amplitude this (joint, shape, frequency) may use, in rad."""
        amp = float(self.ss["amplitude_max"][joint])

        # The torque budget applies to BOTH shapes. Sine cells near resonance
        # are just as capable of exceeding the joint rating as square edges.
        kp_eff = float(self.ss["kp_effective"][joint])
        budget = float(self.ss["torque_budget"][joint])
        gain = self._torque_gain(shape, freq_idx)
        if kp_eff > 0.0 and gain > 0.0:
            amp = min(amp, budget / (kp_eff * gain))

        if shape != "square":
            # A sine's peak speed is 2*pi*f*A; cap it so the fast, wide
            # corner of the sweep does not command absurd velocities. A
            # square's commanded speed is unbounded whatever A is, so the
            # cap is meaningless there and square_amplitude_scale is the
            # knob instead.
            max_speed = self.ss.get("max_peak_speed", None)
            if max_speed is not None and freq_hz > 0.0:
                amp = min(amp, float(max_speed) / (2.0 * np.pi * freq_hz))

        # Keep home +/- amp inside the soft box.
        headroom = min(
            self.q_max[joint] - self.home[joint],
            self.home[joint] - self.q_min[joint],
        )
        return float(max(0.0, min(amp, headroom)))

    def _resolve_amplitude(
        self, joint: int, shape: str, frac: float, freq_hz: float, freq_idx: int
    ) -> float:
        """Amplitude for one sweep cell, in rad.

        amplitude_fracs are fractions of what is *achievable* at this
        frequency, not of a fixed ceiling. Otherwise the speed cap collapses
        every high-amplitude cell at high frequency onto the same value and
        a third of the sweep becomes duplicate data.
        """
        return float(frac) * self._max_amplitude(joint, shape, freq_hz, freq_idx)

    # -- phases ---------------------------------------------------------

    def build_square_sine(self) -> Plan:
        hz = self.global_target_hz
        dt = 1.0 / hz
        seg_len = int(round(float(self.ss["segment_duration_s"]) * hz))
        trans_len = int(round(float(self.ss["transition_s"]) * hz))

        blocks: List[np.ndarray] = []
        segments: List[Segment] = []
        seg_ids: List[np.ndarray] = []
        cursor = 0
        prev_target = self.home.copy()

        def push(block: np.ndarray, seg: Segment) -> None:
            nonlocal cursor, prev_target
            if block.shape[0] == 0:
                return
            seg.start_tick = cursor
            seg.num_ticks = int(block.shape[0])
            blocks.append(block)
            seg_ids.append(np.full(block.shape[0], len(segments), dtype=np.int32))
            segments.append(seg)
            cursor += block.shape[0]
            prev_target = block[-1].copy()

        for joint in self.ss["joints"]:
            joint = int(joint)
            for shape in self.ss["wave_shapes"]:
                for frac in self.ss["amplitude_fracs"]:
                    for freq_idx, freq in enumerate(self.ss["frequencies_hz"]):
                        freq = float(freq)
                        amp = self._resolve_amplitude(
                            joint, shape, frac, freq, freq_idx
                        )

                        t = np.arange(seg_len, dtype=np.float64) * dt
                        if shape == "square":
                            phase = np.mod(freq * t, 1.0)
                            wave = np.where(phase < 0.5, 1.0, -1.0)
                        elif shape == "sine":
                            wave = np.sin(2.0 * np.pi * freq * t)
                        else:
                            raise ValueError(f"Unknown wave shape {shape!r}.")

                        block = np.repeat(
                            self.home[None, :], seg_len, axis=0
                        )
                        block[:, joint] += amp * wave

                        # Blend from wherever the previous segment ended into
                        # this segment's first commanded target.
                        push(
                            _cosine_blend(prev_target, block[0], trans_len),
                            Segment(
                                name=f"transition->j{joint}_{shape}"
                                f"_a{frac:g}_f{freq:g}",
                                kind="transition",
                                joint=joint,
                                amplitude=0.0,
                                frequency_hz=0.0,
                                num_ticks=trans_len,
                            ),
                        )
                        push(
                            block,
                            Segment(
                                name=f"j{joint}_{shape}_a{frac:g}_f{freq:g}",
                                kind=shape,
                                joint=joint,
                                amplitude=amp,
                                frequency_hz=freq,
                                num_ticks=seg_len,
                            ),
                        )

        # Settle back at home so the phase ends where it started.
        push(
            _cosine_blend(prev_target, self.home, trans_len),
            Segment(
                name="transition->home",
                kind="transition",
                joint=-1,
                amplitude=0.0,
                frequency_hz=0.0,
                num_ticks=trans_len,
            ),
        )

        targets = np.clip(np.concatenate(blocks, axis=0), self.q_min, self.q_max)
        return Plan(
            phase="square_sine",
            target_update_hz=hz,
            targets=targets,
            seg_ids=np.concatenate(seg_ids, axis=0),
            segments=segments,
        )

    def build_noise(self) -> Plan:
        hz = float(self.noise.get("target_update_hz") or self.global_target_hz)
        dt = 1.0 / hz
        n = int(round(float(self.noise["duration_s"]) * hz))
        trans_len = int(round(float(self.noise["transition_s"]) * hz))

        # Size the excursion from the SAME torque budget as the sweep. The
        # noise phase had none, which is why it tripped joint2 at 62 Nm on a
        # 60 Nm joint within 2.5 s while the whole square/sine sweep stayed
        # clear. Here the arm lags the target almost completely, so the
        # tracking error IS the excursion and torque ~ kp_eff * excursion:
        # measured peak |err| of [0.52, 0.62, 0.48, 0.41, 0.52, 0.70] against
        # a budget-allowed [0.54, 0.44, 0.46, 0.44, 0.55, 0.49].
        kp_eff = np.asarray(self.ss["kp_effective"], dtype=np.float64)
        budget = np.asarray(self.ss["torque_budget"], dtype=np.float64)
        # Per joint, not one number for all six. Measured gains span
        # 0.80 (joint4) to 2.56 (joint6): joint6 has the lowest stiffness and
        # the least inertia, so it overshoots hardest, and a shared gain of
        # 1.3 let it reach 30.2 Nm on a 30 Nm joint.
        gain = np.asarray(
            self.noise["torque_gain"], dtype=np.float64
        ).reshape(NUM_ARM_JOINTS)
        excursion = np.where(
            kp_eff > 0.0, budget / np.maximum(kp_eff * gain, 1e-9), np.inf
        )
        # Never leave the soft box either.
        excursion = np.minimum(
            excursion,
            np.minimum(self.q_max - self.home, self.home - self.q_min),
        )
        self.noise_excursion = excursion
        sigma = float(self.noise["sigma_frac"]) * excursion
        lo_box = self.home - excursion
        hi_box = self.home + excursion
        hold_lo = float(self.noise["hold_ms_min"]) * 1e-3
        hold_hi = float(self.noise["hold_ms_max"]) * 1e-3
        # Expressed in rad/s so it means the same thing at any target rate.
        speed_clip = self.noise.get("max_target_speed", None)
        step_clip = None
        if speed_clip is not None:
            step_clip = (
                np.asarray(speed_clip, dtype=np.float64).reshape(NUM_ARM_JOINTS)
                * dt
            )

        rng = np.random.default_rng(int(self.noise["seed"]))

        targets = np.zeros((n, NUM_ARM_JOINTS), dtype=np.float64)
        emitted = self.home.copy()
        held = self.home.copy()
        next_resample = rng.uniform(hold_lo, hold_hi, size=NUM_ARM_JOINTS)

        for i in range(n):
            t = i * dt
            due = t >= next_resample
            if np.any(due):
                draw = rng.normal(self.home, sigma)
                held = np.where(due, np.clip(draw, lo_box, hi_box), held)
                next_resample = np.where(
                    due,
                    t + rng.uniform(hold_lo, hold_hi, size=NUM_ARM_JOINTS),
                    next_resample,
                )

            if step_clip is None:
                emitted = held.copy()
            else:
                emitted = emitted + np.clip(held - emitted, -step_clip, step_clip)
            targets[i] = emitted

        targets = np.clip(targets, lo_box, hi_box)

        lead_in = _cosine_blend(self.home, targets[0], trans_len)
        lead_out = _cosine_blend(targets[-1], self.home, trans_len)

        segments = [
            Segment("transition->noise", "transition", -1, 0.0, 0.0, trans_len, 0),
            Segment("noise_all_joints", "noise", -1, 0.0, 0.0, n, trans_len),
            Segment(
                "transition->home", "transition", -1, 0.0, 0.0, trans_len,
                trans_len + n,
            ),
        ]
        seg_ids = np.concatenate(
            [
                np.zeros(trans_len, dtype=np.int32),
                np.ones(n, dtype=np.int32),
                np.full(trans_len, 2, dtype=np.int32),
            ]
        )
        return Plan(
            phase="noise",
            target_update_hz=hz,
            targets=np.concatenate([lead_in, targets, lead_out], axis=0),
            seg_ids=seg_ids,
            segments=segments,
        )


# ---------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------


class Recorder:
    """Preallocated per-step buffers, so the 500 Hz loop never allocates."""

    _ARM_FIELDS = ("q_des", "q", "qd", "tau_est")
    _SCALAR_FIELDS = ("t", "gripper_q_des", "gripper_q", "gripper_qd", "gripper_tau")

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.n = 0
        self._arm = {
            f: np.zeros((self.capacity, NUM_ARM_JOINTS), dtype=np.float64)
            for f in self._ARM_FIELDS
        }
        self._scalar = {
            f: np.zeros(self.capacity, dtype=np.float64) for f in self._SCALAR_FIELDS
        }
        self._seg = np.zeros(self.capacity, dtype=np.int32)

    def append(
        self,
        t: float,
        q_des: np.ndarray,
        q: np.ndarray,
        qd: np.ndarray,
        tau_est: np.ndarray,
        gripper_q_des: float,
        gripper_q: float,
        gripper_qd: float,
        gripper_tau: float,
        seg_id: int,
    ) -> None:
        if self.n >= self.capacity:
            return
        i = self.n
        self._arm["q_des"][i] = q_des
        self._arm["q"][i] = q
        self._arm["qd"][i] = qd
        self._arm["tau_est"][i] = tau_est
        self._scalar["t"][i] = t
        self._scalar["gripper_q_des"][i] = gripper_q_des
        self._scalar["gripper_q"][i] = gripper_q
        self._scalar["gripper_qd"][i] = gripper_qd
        self._scalar["gripper_tau"][i] = gripper_tau
        self._seg[i] = seg_id
        self.n = i + 1

    def arm(self, name: str) -> np.ndarray:
        return self._arm[name][: self.n]

    def scalar(self, name: str) -> np.ndarray:
        return self._scalar[name][: self.n]

    def seg_ids(self) -> np.ndarray:
        return self._seg[: self.n]


# ---------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------


class Z1UANDataCollector:
    def __init__(self, cfg_path: str, cli: argparse.Namespace) -> None:
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        self.cfg_path = os.path.abspath(cfg_path)
        self.cli = cli
        self.u = self.cfg["uan"]
        self.safety = self.u["safety"]

        # Command rate and log rate are deliberately different. The Z1
        # lowcmd link runs at 500 Hz and wants to be fed at 500 Hz, but the
        # arm publishes state at about half that, so logging every send just
        # duplicates rows. Safety runs at the command rate for the fastest
        # possible reaction; recording runs at the state rate.
        self.loop_hz = float(self.u["loop_hz"])
        self.loop_dt = 1.0 / self.loop_hz
        self.log_hz = float(self.u["log_hz"])
        ratio = self.loop_hz / self.log_hz
        self.log_decim = int(round(ratio))
        if self.log_decim < 1 or abs(ratio - self.log_decim) > 1e-9:
            raise ValueError(
                f"uan.loop_hz ({self.loop_hz:g}) must be an integer multiple "
                f"of uan.log_hz ({self.log_hz:g})."
            )
        self.log_dt = 1.0 / self.log_hz

        self.home = np.asarray(
            self.u["home_q"], dtype=np.float64
        ).reshape(NUM_ARM_JOINTS)
        self.gripper_hold_q = float(self.u["gripper_hold_q"])

        self.q_min = np.asarray(self.u["q_min"], dtype=np.float64)
        self.q_max = np.asarray(self.u["q_max"], dtype=np.float64)
        self.q_margin = float(self.safety["q_limit_margin"])
        self.q_lower = Z1_Q_LOWER.copy()
        self.q_upper = Z1_Q_UPPER.copy()
        self.liveness_samples = int(self.safety["liveness_samples"])
        self.liveness_min_distinct = int(self.safety["liveness_min_distinct"])
        self.home_tolerance = float(self.u["home_tolerance"])
        self.home_move_max_speed = float(self.u["home_move_max_speed"])
        self.max_abs_qd = float(self.safety["max_abs_qd"])
        self.max_abs_qd_hard = float(self.safety["max_abs_qd_hard"])
        self.qd_hold_steps = int(
            round(float(self.safety["qd_hold_s"]) * self.loop_hz)
        )
        self.max_abs_tau = np.asarray(
            self.safety["max_abs_tau_est"], dtype=np.float64
        ).reshape(NUM_ARM_JOINTS)
        self.tau_hold_steps = int(
            round(float(self.safety["tau_hold_s"]) * self.loop_hz)
        )
        self.stale_hold_steps = int(
            round(float(self.safety["stale_hold_s"]) * self.loop_hz)
        )
        self.rated_torque = np.asarray(
            self.safety["rated_torque"], dtype=np.float64
        ).reshape(NUM_ARM_JOINTS)
        self.overtorque_budget_s = float(self.safety["overtorque_budget_s"])
        self._overtorque = np.zeros(NUM_ARM_JOINTS)
        self._overtorque_decay = float(
            np.exp(-self.loop_dt / float(self.safety["overtorque_decay_s"]))
        )
        self.max_track_err = float(self.safety["max_tracking_error"])
        self.track_err_hold_steps = int(
            round(float(self.safety["tracking_error_hold_s"]) * self.loop_hz)
        )

        self.phases: List[str] = list(cli.phases or ["square_sine", "noise"])
        for p in self.phases:
            if p not in ("square_sine", "noise"):
                raise ValueError(f"Unknown phase {p!r}.")

        self.builder = PlanBuilder(self.cfg)
        self.plans: Dict[str, Plan] = {}
        if "square_sine" in self.phases:
            self.plans["square_sine"] = self.builder.build_square_sine()
        if "noise" in self.phases:
            self.plans["noise"] = self.builder.build_noise()

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(
            _PROJECT_ROOT, self.u["output_root"], stamp
        )

        self.z1: Optional[Z1ArmAdapter] = None
        self._lowcmd_state = None
        self._track_err_streak = 0
        self._tau_streak = 0
        self._qd_streak = 0
        self._tau_saturated_steps = 0
        self._stale_streak = 0
        self._stale_total = 0
        self._last_state: Optional[np.ndarray] = None

    # -- plan reporting -------------------------------------------------

    def print_plan(self) -> None:
        print("=" * 78)
        print("Z1 UAN data collection plan")
        print("=" * 78)
        print(f"config          : {self.cfg_path}")
        print(f"session dir     : {self.session_dir}")
        print(f"actuation       : {ACTUATOR_MODE} (firmware PD only)")
        print(f"phases          : {self.phases}")
        print(f"command rate    : {self.loop_hz:g} Hz")
        print(
            f"log rate        : {self.log_hz:g} Hz "
            f"(every {self.log_decim} command)"
        )
        print(f"home_q          : {np.round(self.home, 3)}")
        print(f"soft q_min      : {np.round(self.q_min, 3)}")
        print(f"soft q_max      : {np.round(self.q_max, 3)}")
        print(f"gripper hold    : {self.gripper_hold_q:.3f} (raw SDK, no payload)")

        total = 0.0
        for name in self.phases:
            plan = self.plans[name]
            excited = [s for s in plan.segments if s.kind != "transition"]
            print("-" * 78)
            print(
                f"phase {name!r}: {plan.duration_s:7.1f} s  "
                f"({plan.num_ticks} ticks @ {plan.target_update_hz:g} Hz, "
                f"{len(plan.segments)} segments, {len(excited)} excited)"
            )
            speeds = np.abs(np.diff(plan.targets, axis=0)) * plan.target_update_hz
            print(
                "  commanded target speed: "
                f"max {np.max(speeds):6.2f} rad/s, "
                f"p99 {np.percentile(speeds, 99):5.2f} rad/s"
            )
            print(
                "  target range per joint: "
                f"min {np.round(plan.targets.min(axis=0), 2)} "
                f"max {np.round(plan.targets.max(axis=0), 2)}"
            )
            if name == "square_sine":
                self._print_square_sine_table(plan)
            else:
                exc = getattr(self.builder, "noise_excursion", None)
                if exc is not None:
                    kp = np.asarray(self.builder.ss["kp_effective"])
                    g = np.asarray(self.builder.noise["torque_gain"])
                    print(
                        "  excursion bound per joint: "
                        f"{np.round(exc, 3)} rad"
                    )
                    print(
                        "  predicted peak torque    : "
                        f"{np.round(kp * exc * g, 1)} Nm "
                        f"(rated {np.round(self.rated_torque, 0)})"
                    )
            total += plan.duration_s

        print("-" * 78)
        total_motion = total
        print(f"total           : {total_motion / 60.0:.1f} min of motion, plus home moves")
        print(
            "samples         : "
            f"~{int(total_motion * self.log_hz):,} logged rows "
            f"({int(total_motion * self.loop_hz):,} commands)"
        )
        print("=" * 78)

    def _print_square_sine_table(self, plan: Plan) -> None:
        """Per joint / shape, the amplitude span and peak speed after caps."""
        rows: Dict[str, List[Segment]] = {}
        for s in plan.segments:
            if s.kind not in ("square", "sine"):
                continue
            rows.setdefault(f"j{s.joint} {s.kind}", []).append(s)
        for key in sorted(rows):
            segs = rows[key]
            amps = np.array([x.amplitude for x in segs])
            speeds = np.array(
                [2.0 * np.pi * x.frequency_hz * x.amplitude for x in segs]
            )
            print(
                f"    {key:<10s} {len(segs):2d} cells, "
                f"amp {amps.min():.3f}-{amps.max():.3f} rad, "
                f"sine-equivalent peak speed <= {speeds.max():.2f} rad/s"
            )

    # -- hardware -------------------------------------------------------

    def connect(self) -> None:
        self.z1 = Z1ArmAdapter(cfg=self.cfg, project_root=_PROJECT_ROOT)
        self.z1.connect()
        self.z1.read_state()
        self._lowcmd_state = self.z1.unitree_arm_interface.ArmFSMState.LOWCMD
        print(f"[UAN] connected. q = {np.round(self.z1.q, 3)}")
        self._read_sdk_limits()
        self._require_live_state()
        self._require_startable_pose()

    def _require_live_state(self) -> None:
        """Confirm the SDK is publishing FRESH state before trusting the pose.

        When the z1_controller stops publishing, the SDK keeps handing back
        the last packet it received, so q/qd look like a perfectly plausible
        pose that simply is not where the arm is any more. Every downstream
        check then reasons about a stale number -- including the startable
        pose gate, which would blame the arm's position for what is really a
        dead link.

        A live arm is easy to tell apart: encoder LSB is ~1e-5 rad, so even a
        stationary one produced 238 distinct readings in 500 consecutive rows
        on the good runs. A frozen one repeats byte-identically (2550 rows in
        a row on the run that faulted).
        """
        assert self.z1 is not None
        seen = []
        for _ in range(self.liveness_samples):
            self.z1.read_state()
            seen.append(
                tuple(np.round(self.z1.q, 6)) + tuple(np.round(self.z1.qd, 6))
            )
            time.sleep(self.log_dt)

        distinct = len(set(seen))
        if distinct >= self.liveness_min_distinct:
            print(
                f"[UAN] state is live ({distinct} distinct readings in "
                f"{len(seen)} samples)."
            )
            return

        raise SafetyAbort(
            f"the Z1 returned only {distinct} distinct reading(s) in "
            f"{len(seen)} samples over "
            f"{len(seen) * self.log_dt:.2f}s, so the reported pose is STALE, "
            "not current.\n"
            f"    reported q = {np.round(self.z1.q, 3)}\n"
            "  The z1_controller has stopped publishing; the SDK is replaying "
            "its last packet.\n"
            "  Restart the z1_controller (and power-cycle the arm if that "
            "does not help), then rerun. Do not trust the pose above until "
            "this reads live."
        )

    def _read_sdk_limits(self) -> None:
        """Take the joint limits from the live model rather than trusting
        the module constants, and complain if they disagree."""
        assert self.z1 is not None
        try:
            lo = np.asarray(self.z1.arm_model.getJointQMin(), dtype=np.float64)
            hi = np.asarray(self.z1.arm_model.getJointQMax(), dtype=np.float64)
        except Exception as e:  # noqa: BLE001
            print(f"[UAN][WARN] could not read SDK joint limits ({e}); "
                  "using the built-in values.")
            return
        if not (np.allclose(lo, Z1_Q_LOWER) and np.allclose(hi, Z1_Q_UPPER)):
            print("[UAN][WARN] SDK joint limits differ from the built-in ones:")
            print(f"[UAN][WARN]   sdk lower = {np.round(lo, 3)}")
            print(f"[UAN][WARN]   sdk upper = {np.round(hi, 3)}")
            print("[UAN][WARN] using the SDK values.")
        self.q_lower, self.q_upper = lo, hi
        if np.any(self.q_min < lo) or np.any(self.q_max > hi):
            raise SafetyAbort(
                "uan.q_min/q_max fall outside the live SDK joint limits "
                f"[{np.round(lo, 3)}, {np.round(hi, 3)}]; jointProtect() "
                "would clamp the excitation."
            )

    def _require_startable_pose(self) -> None:
        """Refuse to move if the arm is parked outside the SDK's legal range.

        This is deliberately the SDK limit, not the excitation soft box. The
        box is where *targets* may go during the sweep; it excludes the arm's
        own zero pose, so gating startup on it would reject a perfectly
        healthy arm parked at its zero pose.

        What actually matters at startup is jointProtect(): it clamps every
        command into the legal range, so if the arm is parked outside it the
        first command becomes a step of however far outside it is. That is
        how a collapsed arm turns a move-to-home into a protection trip.
        """
        assert self.z1 is not None
        q = self.z1.q.astype(np.float64)
        lo = self.q_lower - self.q_margin
        hi = self.q_upper + self.q_margin
        bad = np.flatnonzero((q < lo) | (q > hi))
        if bad.size == 0:
            return
        detail = "\n".join(
            f"    joint{j + 1}: q={q[j]:+.3f} outside the SDK range "
            f"[{self.q_lower[j]:+.3f}, {self.q_upper[j]:+.3f}]"
            for j in bad
        )
        raise SafetyAbort(
            "the arm is parked outside the range the Z1 SDK will accept "
            "commands in, so it cannot be moved to home safely:\n" + detail
            + f"\n    full reading: q = {np.round(q, 3)}\n"
            "  jointProtect() clamps every command into that range, so the "
            "first hold command would be a step of the distance shown.\n"
            "\n"
            "  If the arm is PHYSICALLY somewhere reasonable and only the "
            "numbers look wrong, the encoder zero is off, not the pose. The "
            "state is live (checked above), so the offset is in the arm's "
            "own reference: put it in the Z1's defined start pose and "
            "re-run the controller's calibration, then rerun this.\n"
            "  Otherwise the arm really is out of range -- bring it back "
            "with the Z1 controller. backToStart() is NOT safe for that "
            "here: it targets all zeros, the arm lying flat, and drives it "
            "into the bench."
        )

    def _move_to_home(
        self, duration_s: Optional[float] = None, verify: bool = True
    ) -> None:
        """Slow supervised ramp to home under firmware startup gains."""
        assert self.z1 is not None
        self.z1.read_state()
        q0 = self.z1.q.astype(np.float64).copy()

        # Give a long move proportionally longer, so distance never turns
        # into speed.
        distance = float(np.max(np.abs(self.home - q0)))
        duration_s = float(
            duration_s if duration_s is not None else self.u["home_move_s"]
        )
        duration_s = max(duration_s, distance / self.home_move_max_speed)
        steps = max(1, int(round(duration_s / self.loop_dt)))

        print(
            f"[UAN] moving to home over {duration_s:.1f}s: "
            f"{np.round(q0, 3)} -> {np.round(self.home, 3)}"
        )
        t0 = time.perf_counter()
        for i in range(steps):
            s = 0.5 * (1.0 - np.cos(np.pi * (i + 1) / steps))
            q_cmd = (1.0 - s) * q0 + s * self.home
            self.z1.hold_pose_lowcmd(
                q_cmd=q_cmd.astype(np.float32),
                gripper_q_cmd=self.gripper_hold_q,
            )
            _sleep_until(t0 + (i + 1) * self.loop_dt)

        self.z1.read_state()
        err = self.home - self.z1.q
        max_err = float(np.max(np.abs(err)))
        print(f"[UAN] at home. |err| max = {max_err:.4f} rad")

        # Starting the sweep from somewhere other than home makes every
        # planned target a large step, so never just report the error.
        if verify and max_err > self.home_tolerance:
            raise SafetyAbort(
                f"move to home did not arrive: |err| max = {max_err:.4f} rad "
                f"> {self.home_tolerance} rad (err = {np.round(err, 3)}, "
                f"FSM = {self.z1.get_fsm_state()}). The arm is not tracking; "
                "check that it is still in LOWCMD and free to move."
            )

    def _engage(self) -> None:
        """Switch to the deployed firmware position-PD loop and verify it holds."""
        assert self.z1 is not None
        self.z1.arm_runtime_mode = ACTUATOR_MODE
        print(
            "[UAN] actuation=position_pd  firmware PD "
            f"kp={np.round(self.z1.arm_kps_runtime, 2)} "
            f"kd={np.round(self.z1.arm_kds_runtime, 1)}"
        )

        # Walk the firmware gains from startup to runtime while holding home,
        # so the softer runtime loop takes over without a jolt.
        settle = max(1, int(round(0.5 / self.loop_dt)))
        t0 = time.perf_counter()
        for i in range(settle):
            self.z1.track_target_pd_once(
                q_target=self.home.astype(np.float32),
                gripper_q_target=self.gripper_hold_q,
                use_startup_gains=False,
            )
            _sleep_until(t0 + (i + 1) * self.loop_dt)

    def _disengage(self) -> None:
        """Hold the current pose with the firmware position loop."""
        if self.z1 is None:
            return
        self.z1.read_state()
        q_now = self.z1.q.copy()
        self.z1.arm_runtime_mode = ACTUATOR_MODE
        for _ in range(20):
            self.z1.hold_pose_lowcmd(
                q_cmd=q_now, gripper_q_cmd=self.gripper_hold_q
            )
            time.sleep(self.loop_dt)

    def _send_step(self, q_des: np.ndarray) -> None:
        """Send one position target through the Z1 firmware PD loop."""
        assert self.z1 is not None
        self.z1.track_target_pd_once(
            q_target=np.asarray(q_des, dtype=np.float32).reshape(NUM_ARM_JOINTS),
            gripper_q_target=self.gripper_hold_q,
            use_startup_gains=False,
        )

    def _check_safety(self, q_des: np.ndarray) -> None:
        assert self.z1 is not None
        q = self.z1.q.astype(np.float64)
        qd = self.z1.qd.astype(np.float64)
        tau = self.z1.tau.astype(np.float64)

        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            raise SafetyAbort(f"non-finite state: q={q} qd={qd}")

        # The arm faulting looks like nothing at all: the SDK keeps handing
        # back the LAST state it received, so q/qd/tau freeze at plausible
        # values and every other check stays happy while the arm is limp.
        # Catching the freeze is the only way to notice promptly.
        state = np.concatenate((q, qd, tau))
        if self._last_state is not None and np.array_equal(state, self._last_state):
            self._stale_streak += 1
            self._stale_total += 1
            if self._stale_streak > self.stale_hold_steps:
                raise SafetyAbort(
                    "Z1 state stopped updating for "
                    f"{self._stale_streak * self.loop_dt:.2f}s -- the arm has "
                    f"most likely faulted out of LOWCMD. Last live state: "
                    f"q={np.round(q, 3)} qd={np.round(qd, 2)} "
                    f"tau={np.round(tau, 1)}"
                )
        else:
            self._stale_streak = 0
        self._last_state = state

        fsm = self.z1.get_fsm_state()
        if self._lowcmd_state is not None and fsm != self._lowcmd_state:
            raise SafetyAbort(f"Z1 left LOWCMD: FSM is now {fsm}")

        # A square edge legitimately flings the joint past its rated speed
        # for a few milliseconds -- the SDK's pi rad/s is a clamp on
        # *commanded* qd, not a ceiling on what the arm physically reaches.
        # So a moderate overspeed only aborts when it persists, and only a
        # frankly impossible speed aborts on the spot.
        if np.any(np.abs(qd) > self.max_abs_qd_hard):
            raise SafetyAbort(
                f"joint speed {np.round(qd, 2)} exceeds the hard limit "
                f"{self.max_abs_qd_hard} rad/s"
            )
        if np.any(np.abs(qd) > self.max_abs_qd):
            self._qd_streak += 1
            if self._qd_streak > self.qd_hold_steps:
                raise SafetyAbort(
                    f"joint speed {np.round(qd, 2)} stayed above "
                    f"{self.max_abs_qd} rad/s for "
                    f"{self.qd_hold_steps * self.loop_dt:.2f}s"
                )
        else:
            self._qd_streak = 0

        # A square edge saturates the actuator by design: the firmware hits
        # its own torque clip for a few milliseconds and the replaying sim
        # clips identically, so that is signal, not a fault. Only *sustained*
        # saturation means something is wrong -- a stall or a collision.
        # Exceeding the joint rating at all now means the amplitude model is
        # wrong, because cells are sized to 75% of rating. The excursions that
        # preceded the last fault were 20 ms spikes to 67.5 Nm on a 60 Nm
        # joint -- far too brief for any hold-based check -- so this one is
        # instantaneous.
        if np.any(np.abs(tau) > self.rated_torque):
            j = int(np.argmax(np.abs(tau) - self.rated_torque))
            raise SafetyAbort(
                f"joint{j + 1} torque {tau[j]:+.1f} Nm exceeded its "
                f"{self.rated_torque[j]:.0f} Nm rating. Cells are sized to "
                "75% of rating, so this means the amplitude model is wrong "
                "for this joint -- re-measure its torque gain rather than "
                "raising the limit."
            )

        # Secondary backstop for repeated moderate excursions that never
        # individually reach the rating. Note this would NOT have caught the
        # 20260905_142932 fault: that run accumulated only 0.13 s above rating
        # across four minutes. The amplitude budget is the real defence.
        over = np.maximum(np.abs(tau) - self.rated_torque, 0.0)
        self._overtorque *= self._overtorque_decay
        self._overtorque += np.where(over > 0.0, self.loop_dt, 0.0)
        if np.any(self._overtorque > self.overtorque_budget_s):
            j = int(np.argmax(self._overtorque))
            raise SafetyAbort(
                f"joint{j + 1} has spent {self._overtorque[j]:.3f}s above its "
                f"{self.rated_torque[j]:.0f} Nm rating within the decay "
                f"window (budget {self.overtorque_budget_s}s). The arm's own "
                "protection trips on accumulated overcurrent, so back the "
                "amplitudes off rather than raising this."
            )

        if np.any(np.abs(tau) > self.max_abs_tau):
            self._tau_streak += 1
            self._tau_saturated_steps += 1
            if self._tau_streak > self.tau_hold_steps:
                raise SafetyAbort(
                    f"measured torque {np.round(tau, 1)} exceeded "
                    f"{self.max_abs_tau} Nm for "
                    f"{self.tau_hold_steps * self.loop_dt:.2f}s"
                )
        else:
            self._tau_streak = 0

        lo = self.q_min - self.q_margin
        hi = self.q_max + self.q_margin
        if np.any(q < lo) or np.any(q > hi):
            raise SafetyAbort(
                f"joint position {np.round(q, 3)} left the soft box "
                f"[{np.round(lo, 3)}, {np.round(hi, 3)}]"
            )

        # A square-wave edge is a large tracking error by design, so this
        # only trips when the error stays large -- a stall, a collision, or
        # lost torque authority.
        if np.max(np.abs(np.asarray(q_des, dtype=np.float64) - q)) > self.max_track_err:
            self._track_err_streak += 1
            if self._track_err_streak > self.track_err_hold_steps:
                raise SafetyAbort(
                    f"tracking error above {self.max_track_err} rad for "
                    f"{self.track_err_hold_steps * self.loop_dt:.2f}s "
                    f"(q_des={np.round(q_des, 3)} q={np.round(q, 3)})"
                )
        else:
            self._track_err_streak = 0

    # -- the collection loop --------------------------------------------

    def run_phase(self, plan: Plan):
        """Run one phase. Returns (recorder, abort_exception_or_None)."""
        assert self.z1 is not None
        n_steps = int(round(plan.duration_s * self.loop_hz))
        rec = Recorder(capacity=n_steps // self.log_decim + 16)
        tick_per_step = plan.target_update_hz / self.loop_hz
        last_tick = plan.num_ticks - 1

        print("-" * 78)
        print(
            f"[UAN] actuation={ACTUATOR_MODE} phase={plan.phase}: "
            f"{n_steps:,} commands @ {self.loop_hz:g} Hz, "
            f"{n_steps // self.log_decim:,} logged @ {self.log_hz:g} Hz, "
            f"{plan.duration_s / 60.0:.1f} min"
        )
        self._track_err_streak = 0
        self._tau_streak = 0
        self._qd_streak = 0
        self._tau_saturated_steps = 0
        self._stale_streak = 0
        self._stale_total = 0
        self._last_state = None
        self._overtorque[:] = 0.0

        t0 = time.perf_counter()
        next_report = 10.0
        aborted: Optional[BaseException] = None

        try:
            self._run_ticks(plan, rec, n_steps, tick_per_step, last_tick, t0,
                            next_report)
        except (SafetyAbort, KeyboardInterrupt) as e:
            # Never throw away what was already recorded: a trip at minute 16
            # of a 17 minute sweep still leaves 16 usable minutes.
            aborted = e

        self._report_phase(rec, aborted)
        return rec, aborted

    def _run_ticks(
        self,
        plan: Plan,
        rec: Recorder,
        n_steps: int,
        tick_per_step: float,
        last_tick: int,
        t0: float,
        next_report: float,
    ) -> None:
        assert self.z1 is not None
        for i in range(n_steps):
            tick = min(int(i * tick_per_step), last_tick)
            q_des = plan.targets[tick]

            self._send_step(q_des)
            t = time.perf_counter() - t0

            if i % self.log_decim == 0:
                rec.append(
                    t=t,
                    q_des=q_des,
                    q=self.z1.q,
                    qd=self.z1.qd,
                    tau_est=self.z1.tau,
                    gripper_q_des=self.gripper_hold_q,
                    gripper_q=self.z1.gripper_q,
                    gripper_qd=self.z1.gripper_qd,
                    gripper_tau=self._read_gripper_tau(),
                    seg_id=int(plan.seg_ids[tick]),
                )

            # Always at the command rate, never decimated: a fault must be
            # caught on the step it happens, not on the next logged one.
            self._check_safety(q_des)

            if t >= next_report:
                seg = plan.segments[int(plan.seg_ids[tick])]
                print(
                    f"  [{t:6.1f}/{plan.duration_s:6.1f}s] {seg.name:<34s} "
                    f"|err|={np.max(np.abs(q_des - self.z1.q)):.3f} "
                    f"|qd|={np.max(np.abs(self.z1.qd)):5.2f} "
                    f"|tau|={np.max(np.abs(self.z1.tau)):5.1f}"
                )
                next_report += 10.0

            # The SDK's sendRecv() is itself paced by the 500 Hz z1_controller,
            # so this usually returns immediately. Interval statistics are
            # computed from the recorded timestamps in _report_phase().
            _sleep_until(t0 + (i + 1) * self.loop_dt)

    def _report_phase(self, rec: Recorder, aborted: Optional[BaseException]) -> None:
        if rec.n == 0:
            print("[UAN] phase produced no samples.")
            return
        elapsed = float(rec.scalar("t")[-1])
        if rec.n < 2 or elapsed <= 0.0:
            print(
                f"[UAN] phase {'ABORTED' if aborted else 'done'}: {rec.n} "
                "sample(s) -- too short to characterise."
            )
            return
        print(
            f"[UAN] phase {'ABORTED' if aborted else 'done'}: {rec.n:,} samples "
            f"over {elapsed:.1f}s, logged at {rec.n / elapsed:.1f} Hz "
            f"(commanded at {rec.n * self.log_decim / elapsed:.1f} Hz)"
        )
        st = self._timing_stats(rec)
        if st is not None:
            print(
                f"[UAN]   step interval: mean {st['mean_ms']:.3f} ms, "
                f"p99 {st['p99_ms']:.3f}, max {st['max_ms']:.3f} "
                f"({st['hiccups']} over {1.5 * self.log_dt * 1e3:.1f} ms)"
            )
        dup = self._duplicate_fraction(rec)
        if dup is not None:
            print(
                f"[UAN]   duplicated state: {dup * 100:.1f}% of logged rows "
                f"(effective state rate ~{rec.n / elapsed * (1.0 - dup):.0f} Hz)"
            )
        print(
            "[UAN]   peak |qd| per joint "
            f"{np.round(np.abs(rec.arm('qd')).max(axis=0), 2)} rad/s "
            f"(sustained limit {self.max_abs_qd}, hard {self.max_abs_qd_hard})"
        )
        sat = self._saturation_stats(rec)
        print(
            f"[UAN]   torque above {np.round(self.max_abs_tau, 0)} Nm: "
            f"{sat['steps']:,} steps ({sat['fraction'] * 100:.3f}%), "
            f"peak |tau| per joint {np.round(sat['peak'], 1)}"
        )

    def _timing_stats(self, rec: Recorder) -> Optional[Dict]:
        """Per-step interval, which is what actually matters.

        Not drift against a fixed schedule: the loop is paced by the SDK, so
        a steady 2.004 ms step accumulates tens of ms of 'lateness' over a
        long phase while being perfectly regular.
        """
        if rec.n < 2:
            return None
        d = np.diff(rec.scalar("t")) * 1e3
        return {
            "mean_ms": float(d.mean()),
            "p99_ms": float(np.percentile(d, 99)),
            "max_ms": float(d.max()),
            "hiccups": int((d > 1.5 * self.log_dt * 1e3).sum()),
        }

    def _duplicate_fraction(self, rec: Recorder) -> Optional[float]:
        """Share of steps where the SDK returned the previous state again.

        The Z1 publishes state more slowly than we can send, so a nonzero
        value is normal; it tells you the honest state rate of the dataset.
        """
        if rec.n < 2:
            return None
        d = np.abs(np.diff(rec.arm("q"), axis=0)).sum(1)
        d += np.abs(np.diff(rec.arm("qd"), axis=0)).sum(1)
        d += np.abs(np.diff(rec.arm("tau_est"), axis=0)).sum(1)
        return float((d == 0).mean())

    def _saturation_stats(self, rec: Recorder) -> Dict:
        tau = np.abs(rec.arm("tau_est"))
        over = (tau > self.max_abs_tau).any(axis=1)
        return {
            "steps": int(over.sum()),
            "fraction": float(over.mean()) if rec.n else 0.0,
            "peak": tau.max(axis=0) if rec.n else np.zeros(NUM_ARM_JOINTS),
        }

    def _read_gripper_tau(self) -> float:
        assert self.z1 is not None
        try:
            v = np.asarray(
                self.z1.arm.lowstate.getGripperTau(), dtype=np.float64
            ).reshape(-1)
            return float(v[0]) if v.size else 0.0
        except Exception:
            return 0.0

    # -- saving ----------------------------------------------------------

    def _save_phase(self, plan: Plan, rec: Recorder) -> str:
        out_dir = self.session_dir
        os.makedirs(out_dir, exist_ok=True)
        stem = "square_sine_log" if plan.phase == "square_sine" else "noise_log"

        assert self.z1 is not None
        n = rec.n
        t_us = np.round(rec.scalar("t") * 1e6).astype(np.int64)

        kp6 = np.asarray(self.z1.arm_kps_runtime, dtype=np.float64)
        kd6 = np.asarray(self.z1.arm_kds_runtime, dtype=np.float64)
        grip_kp, grip_kd = self.z1.gripper_kp, self.z1.gripper_kd

        kp7 = np.tile(np.append(kp6, grip_kp), (n, 1))
        kd7 = np.tile(np.append(kd6, grip_kd), (n, 1))

        def with_gripper(arm: np.ndarray, grip: np.ndarray) -> np.ndarray:
            return np.hstack((arm, grip[:, None]))

        data = {
            "arm_pd_tau_targets": {
                "q_des": rec.arm("q_des"),
                "gripperQ_des": rec.scalar("gripper_q_des"),
                "kp": kp7,
                "kd": kd7,
                "timestamp": t_us,
            },
            "arm_control_data": {
                "q": with_gripper(rec.arm("q"), rec.scalar("gripper_q")),
                "qd": with_gripper(rec.arm("qd"), rec.scalar("gripper_qd")),
                "tau_est": with_gripper(
                    rec.arm("tau_est"), rec.scalar("gripper_tau")
                ),
                "timestamp": t_us,
            },
            "uan_meta": self._phase_meta(plan, rec),
        }

        pkl_path = os.path.join(out_dir, f"{stem}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        np.savez_compressed(
            os.path.join(out_dir, f"{stem}.npz"),
            t=rec.scalar("t"),
            q_des=rec.arm("q_des"),
            q=rec.arm("q"),
            qd=rec.arm("qd"),
            tau_est=rec.arm("tau_est"),
            gripper_q_des=rec.scalar("gripper_q_des"),
            gripper_q=rec.scalar("gripper_q"),
            gripper_qd=rec.scalar("gripper_qd"),
            gripper_tau=rec.scalar("gripper_tau"),
            seg_id=rec.seg_ids(),
            kp=kp6,
            kd=kd6,
        )

        print(f"[UAN] saved {pkl_path}  ({n:,} samples)")
        return pkl_path

    def _phase_meta(self, plan: Plan, rec: Recorder) -> Dict:
        assert self.z1 is not None
        seg_ids = rec.seg_ids()
        boundaries = (
            np.flatnonzero(np.diff(seg_ids) != 0) + 1 if rec.n > 1 else np.array([])
        )
        return {
            "mode": ACTUATOR_MODE,
            "phase": plan.phase,
            "num_samples": int(rec.n),
            "loop_hz": self.loop_hz,
            "log_hz": self.log_hz,
            "timing": self._timing_stats(rec),
            "duplicate_state_fraction": self._duplicate_fraction(rec),
            "torque_saturation": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self._saturation_stats(rec).items()
            },
            "target_update_hz": plan.target_update_hz,
            "home_q": self.home.tolist(),
            "gripper_hold_q": self.gripper_hold_q,
            "gripper_excited": False,
            "payload": "none",
            "mount": "standalone, base level and world-aligned",
            "arm_kps_runtime": np.asarray(self.z1.arm_kps_runtime).tolist(),
            "arm_kds_runtime": np.asarray(self.z1.arm_kds_runtime).tolist(),
            "torque_note": (
                "tau_est is the arm's own torque estimate. No torque is "
                "commanded: the firmware closes the position loop, so a "
                "replaying sim must reconstruct the nominal torque from "
                "q_des with its own Kp_sim/Kd_sim."
            ),
            "columns": "arm_control_data arrays are [j1..j6, gripper]",
            "segments": [s.as_dict() for s in plan.segments],
            "segment_boundaries": boundaries.astype(int).tolist(),
            "config_path": self.cfg_path,
            "paper": "arXiv:2502.10894 Section II-A.2",
        }

    def _write_session_metadata(self, results: Dict) -> None:
        os.makedirs(self.session_dir, exist_ok=True)
        meta = {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "config_path": self.cfg_path,
            "mode": ACTUATOR_MODE,
            "phases": self.phases,
            "results": results,
            "config": self.cfg,
            "sdk_q_lower": np.asarray(self.q_lower).tolist(),
            "sdk_q_upper": np.asarray(self.q_upper).tolist(),
        }
        path = os.path.join(self.session_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"[UAN] wrote {path}")

    # -- top level -------------------------------------------------------

    def run(self) -> None:
        self.print_plan()

        if self.cli.dry_run:
            self._dump_dry_run()
            return

        if not self.cli.yes:
            print(
                "\nThe arm will move through the full sweep above. Confirm the "
                "workspace is clear,\nthe base is bolted down and level, and "
                "the gripper is empty."
            )
            if input("Type 'go' to start: ").strip().lower() != "go":
                print("[UAN] aborted before any motion.")
                return

        self.connect()
        results: Dict = {}

        try:
            self._move_to_home()
            self._engage()
            for phase in self.phases:
                plan = self.plans[phase]
                rec, aborted = self.run_phase(plan)
                results[phase] = {
                    "path": self._save_phase(plan, rec) if rec.n else None,
                    "num_samples": int(rec.n),
                    "complete": aborted is None,
                    "abort_reason": None if aborted is None else str(aborted),
                }
                if aborted is not None:
                    raise aborted
            self._disengage()
            self._move_to_home()

        except KeyboardInterrupt:
            print("\n[UAN] interrupted by user.")
            self._emergency_recover()
        except SafetyAbort as e:
            print(f"\n[UAN][SAFETY ABORT] {e}")
            self._emergency_recover()
        except Exception as e:  # noqa: BLE001 - the arm must be parked first
            print(f"\n[UAN][ERROR] {type(e).__name__}: {e}")
            self._emergency_recover()
            raise
        finally:
            self._write_session_metadata(results)

        print("\n[UAN] session complete.")
        print(f"[UAN] data in {self.session_dir}")

    def _emergency_recover(self) -> None:
        """Restore the firmware position loop and park the arm at home."""
        if self.z1 is None:
            return
        try:
            fsm = self.z1.get_fsm_state()
            if fsm != self._lowcmd_state:
                # Commanding a PASSIVE arm does nothing but block on UDP; the
                # arm is already limp and there is nothing to recover.
                print(
                    f"[UAN] the Z1 is in {fsm}, not LOWCMD -- it has already "
                    "gone limp and is not accepting commands."
                )
                print(
                    "[UAN] Support the arm, then power-cycle it (or use the "
                    "Z1 controller) to bring it back to a safe pose inside "
                    "the soft box before rerunning."
                )
                return

            print("[UAN] restoring firmware hold at the measured pose...")
            self._disengage()
            self._move_to_home(
                duration_s=max(4.0, float(self.u["home_move_s"])),
                verify=False,
            )
            print("[UAN] arm parked at home.")
        except KeyboardInterrupt:
            print("[UAN][WARN] recovery interrupted; the arm may be unparked.")
        except Exception as e:  # noqa: BLE001
            print(f"[UAN][ERROR] recovery failed: {type(e).__name__}: {e}")
            print("[UAN] STOP THE ARM MANUALLY.")

    def _dump_dry_run(self) -> None:
        out_dir = self.session_dir + "_dryrun"
        os.makedirs(out_dir, exist_ok=True)
        for name, plan in self.plans.items():
            path = os.path.join(out_dir, f"plan_{name}.npz")
            np.savez_compressed(
                path,
                targets=plan.targets,
                seg_ids=plan.seg_ids,
                target_update_hz=plan.target_update_hz,
                segment_names=np.array([s.name for s in plan.segments]),
            )
            print(f"[UAN][dry-run] wrote {path}")
        print("[UAN][dry-run] no hardware was touched.")


def _sleep_until(t_target: float) -> float:
    """Sleep to t_target. Returns how late we already were, in seconds."""
    now = time.perf_counter()
    late = now - t_target
    if late >= 0.0:
        return late
    remaining = -late
    if remaining > 0.0008:
        time.sleep(remaining - 0.0005)
    while time.perf_counter() < t_target:
        pass
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect Z1 hardware data for unsupervised actuator net training."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.path.join(_DEPLOY_DIR, "configs", "z1_uan_data_collection.yaml"),
        help="Path to the collection YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the excitation plan without touching the arm.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=None,
        metavar="PHASE",
        help="Override the phases to run: square_sine and/or noise.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the pre-motion confirmation.",
    )
    args = parser.parse_args()

    np.set_printoptions(precision=3, suppress=True)
    Z1UANDataCollector(cfg_path=args.config, cli=args).run()


if __name__ == "__main__":
    main()