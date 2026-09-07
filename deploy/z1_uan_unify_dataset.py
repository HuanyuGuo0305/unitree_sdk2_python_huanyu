#!/usr/bin/env python3
"""
Synchronize the per-phase Z1 UAN hardware logs into one unified dataset.

deploy/z1_uan_data_collection.py writes one log per excitation phase:

    <output_root>/<timestamp>/square_sine_log.pkl/.npz
    <output_root>/<timestamp>/noise_log.pkl/.npz
    <output_root>/<timestamp>/multi_pose_coupled_log.pkl/.npz

Each phase restarts its own clock at t=0 and its own seg_id at 0, so the three
files cannot simply be concatenated: the timeline would run backwards and
segments from different phases would collide. This tool merges them into a
single dataset with one monotonic clock, globally unique segment ids, and a
row-level index saying where every sample came from.

The merge is LOSSLESS. Nothing is dropped, resampled or filtered -- transitions
and duplicated state rows are flagged, not removed, so the training side
decides what to keep.

Output (default <root>/unified/):

    uan_dataset.pkl     same arm_pd_tau_targets / arm_control_data / uan_meta
                        schema as a phase log, plus a uan_index block
    uan_dataset.npz     flat arrays, same field names as a phase .npz
    uan_dataset.json    metadata only, for eyeballing without loading arrays

Row-level index fields (uan_index in the .pkl, and top level in the .npz):

    session_id      index into uan_meta["sessions"]
    phase_id        index into uan_meta["phases"]
    seg_id          segment id within its own phase log (as recorded)
    seg_uid         globally unique segment id across the whole dataset
    is_transition   True on blend/settle/center-move rows, i.e. not excitation
    is_dup_state    True when the arm republished the previous state unchanged

Usage:

    python deploy/z1_uan_unify_dataset.py                       # Log/z1_uan_v2
    python deploy/z1_uan_unify_dataset.py --root Log/z1_uan
    python deploy/z1_uan_unify_dataset.py --sessions 20260907_174934
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

# Phase log stems, in the order the collector runs them. A session is merged in
# this order regardless of how metadata.json lists it, so two sessions merged
# together stay comparable.
PHASE_STEMS = (
    ("square_sine", "square_sine_log"),
    ("noise", "noise_log"),
    ("multi_pose_coupled", "multi_pose_coupled_log"),
)

# Segment kinds that are not excitation: blends between segments, the settle
# hold at a new center, and the slow center-to-center move. Kept in the dataset
# but flagged, because they are dominated by near-zero-velocity samples.
TRANSITION_KINDS = frozenset({"transition", "settle_center", "center_transition"})

SESSION_RE = re.compile(r"^\d{8}_\d{6}$")

NUM_ARM_JOINTS = 6

# Fields that MUST agree across every phase and session being merged. Merging
# across a change in any of them would describe two different plants.
CRITICAL_META = ("mode", "log_hz", "columns", "arm_kps_runtime", "arm_kds_runtime")


class MergeError(RuntimeError):
    pass


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------


def find_sessions(root: str, wanted: Optional[List[str]]) -> List[str]:
    if not os.path.isdir(root):
        raise MergeError(f"Dataset root does not exist: {root}")
    names = sorted(
        d
        for d in os.listdir(root)
        if SESSION_RE.match(d) and os.path.isdir(os.path.join(root, d))
    )
    if wanted:
        missing = [w for w in wanted if w not in names]
        if missing:
            raise MergeError(
                f"Requested session(s) not found under {root}: {', '.join(missing)}"
            )
        names = [n for n in names if n in wanted]
    if not names:
        raise MergeError(f"No timestamped session directories under {root}.")
    return names


def load_phase(session_dir: str, stem: str) -> Optional[Dict]:
    """Load one phase log. The .pkl is canonical; the .npz supplies seg_id."""
    pkl_path = os.path.join(session_dir, f"{stem}.pkl")
    npz_path = os.path.join(session_dir, f"{stem}.npz")
    if not os.path.isfile(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    for key in ("arm_pd_tau_targets", "arm_control_data", "uan_meta"):
        if key not in data:
            raise MergeError(f"{pkl_path} is missing the {key!r} block.")

    n = int(data["uan_meta"]["num_samples"])
    for block in ("arm_pd_tau_targets", "arm_control_data"):
        for name, arr in data[block].items():
            if np.shape(arr)[0] != n:
                raise MergeError(
                    f"{pkl_path}: {block}/{name} has {np.shape(arr)[0]} rows, "
                    f"expected {n}."
                )

    # seg_id only exists in the .npz. Without it the segment table cannot be
    # tied back to rows, so this is fatal rather than a warning.
    if not os.path.isfile(npz_path):
        raise MergeError(f"Missing {npz_path}; it holds the per-row seg_id.")
    with np.load(npz_path) as npz:
        seg_id = np.asarray(npz["seg_id"], dtype=np.int64)
        t = np.asarray(npz["t"], dtype=np.float64)
    if seg_id.shape[0] != n or t.shape[0] != n:
        raise MergeError(f"{npz_path} has a different row count than {pkl_path}.")

    data["seg_id"] = seg_id
    data["t"] = t
    data["pkl_path"] = pkl_path
    return data


# ---------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------


def check_compatible(ref: Dict, ref_label: str, cur: Dict, cur_label: str) -> None:
    """Refuse to merge logs that describe different plants."""
    for key in CRITICAL_META:
        a, b = ref.get(key), cur.get(key)
        if isinstance(a, list) or isinstance(b, list):
            same = np.allclose(np.asarray(a, dtype=np.float64),
                               np.asarray(b, dtype=np.float64))
        else:
            same = a == b
        if not same:
            raise MergeError(
                f"Refusing to merge: {key} differs between {ref_label} ({a}) and "
                f"{cur_label} ({b}). Logs collected under different gains or rates "
                f"describe different plants and must stay separate datasets."
            )


def duplicate_state_mask(q: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """True where the arm republished the previous state byte-identically.

    The Z1 publishes state slower than the collector sends commands, so some
    logged rows repeat the previous packet. They carry no new information about
    the plant, but dropping them would break the fixed-dt assumption -- so they
    are flagged here and left in place.
    """
    n = q.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n < 2:
        return mask
    same = (
        np.all(np.diff(q, axis=0) == 0.0, axis=1)
        & np.all(np.diff(qd, axis=0) == 0.0, axis=1)
        & np.all(np.diff(tau, axis=0) == 0.0, axis=1)
    )
    mask[1:] = same
    return mask


# ---------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------


def merge(
    root: str,
    session_names: List[str],
    phase_gap_s: float,
    session_gap_s: float,
) -> Tuple[Dict, Dict]:
    q_des, grip_des, kp, kd = [], [], [], []
    q, qd, tau_est = [], [], []
    t_unified, t_phase = [], []
    session_ids, phase_ids, seg_ids, seg_uids = [], [], [], []
    is_transition, is_dup = [], []

    segments: List[Dict] = []
    sessions_meta: List[Dict] = []
    phase_names: List[str] = []
    ref_meta: Optional[Dict] = None
    ref_label = ""

    t_cursor = 0.0
    row_cursor = 0
    seg_uid_cursor = 0
    first_session = True

    for s_idx, name in enumerate(session_names):
        session_dir = os.path.join(root, name)
        meta_path = os.path.join(session_dir, "metadata.json")
        session_meta = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                session_meta = json.load(f)

        if not first_session:
            t_cursor += session_gap_s
        first_session = False

        session_entry = {
            "name": name,
            "dir": os.path.abspath(session_dir),
            "created": session_meta.get("created"),
            "config_path": session_meta.get("config_path"),
            "phases": [],
        }
        session_first_phase = True

        for phase, stem in PHASE_STEMS:
            log = load_phase(session_dir, stem)
            if log is None:
                continue

            pmeta = log["uan_meta"]
            label = f"{name}/{phase}"
            if ref_meta is None:
                ref_meta, ref_label = pmeta, label
            else:
                check_compatible(ref_meta, ref_label, pmeta, label)

            if phase not in phase_names:
                phase_names.append(phase)
            phase_id = phase_names.index(phase)

            n = int(pmeta["num_samples"])
            tp = log["t"]
            if n > 1 and not np.all(np.diff(tp) > 0.0):
                raise MergeError(f"{label}: recorded time is not increasing.")

            # The collector rests at home between phases without logging, so the
            # real wall-clock gap is unrecorded. Insert the nominal rest instead
            # of pretending the phases are contiguous.
            if not session_first_phase:
                t_cursor += phase_gap_s
            session_first_phase = False

            t_unified.append(tp + t_cursor)
            t_phase.append(tp)
            t_cursor += float(tp[-1]) if n else 0.0

            tgt, ctl = log["arm_pd_tau_targets"], log["arm_control_data"]
            q_des.append(np.asarray(tgt["q_des"], dtype=np.float64))
            grip_des.append(np.asarray(tgt["gripperQ_des"], dtype=np.float64))
            kp.append(np.asarray(tgt["kp"], dtype=np.float64))
            kd.append(np.asarray(tgt["kd"], dtype=np.float64))
            q_i = np.asarray(ctl["q"], dtype=np.float64)
            qd_i = np.asarray(ctl["qd"], dtype=np.float64)
            tau_i = np.asarray(ctl["tau_est"], dtype=np.float64)
            q.append(q_i)
            qd.append(qd_i)
            tau_est.append(tau_i)

            seg_i = log["seg_id"]
            session_ids.append(np.full(n, s_idx, dtype=np.int32))
            phase_ids.append(np.full(n, phase_id, dtype=np.int32))
            seg_ids.append(seg_i.astype(np.int32))
            seg_uids.append((seg_i + seg_uid_cursor).astype(np.int32))
            is_dup.append(duplicate_state_mask(q_i, qd_i, tau_i))

            # Segment table: per-phase entries, re-keyed to global rows.
            phase_segments = pmeta.get("segments", [])
            kinds = np.array(
                [str(s.get("kind", "")) for s in phase_segments], dtype=object
            )
            trans_flag = np.zeros(n, dtype=bool)
            for local_id in range(len(phase_segments)):
                rows = np.flatnonzero(seg_i == local_id)
                seg = dict(phase_segments[local_id])
                is_trans = str(seg.get("kind", "")) in TRANSITION_KINDS
                if is_trans and rows.size:
                    trans_flag[rows] = True
                seg.update(
                    {
                        "seg_uid": seg_uid_cursor + local_id,
                        "seg_id": local_id,
                        "session": name,
                        "session_id": s_idx,
                        "phase": phase,
                        "phase_id": phase_id,
                        "is_transition": bool(is_trans),
                        "num_rows": int(rows.size),
                        "row_start": int(row_cursor + rows[0]) if rows.size else -1,
                        "row_end": int(row_cursor + rows[-1] + 1) if rows.size else -1,
                    }
                )
                segments.append(seg)

            # Any seg_id with no entry in the table would silently lose its
            # transition flag, so surface it instead of guessing.
            unknown = np.setdiff1d(np.unique(seg_i), np.arange(len(phase_segments)))
            if unknown.size:
                raise MergeError(
                    f"{label}: rows reference seg_id {unknown.tolist()} which the "
                    f"segment table ({len(phase_segments)} entries) does not define."
                )
            is_transition.append(trans_flag)

            session_entry["phases"].append(
                {
                    "phase": phase,
                    "phase_id": phase_id,
                    "source": os.path.abspath(log["pkl_path"]),
                    "num_samples": n,
                    "num_segments": len(phase_segments),
                    "duration_s": round(float(tp[-1]) if n else 0.0, 3),
                    "row_start": int(row_cursor),
                    "row_end": int(row_cursor + n),
                    "t_start": round(float(t_unified[-1][0]), 3),
                    "t_end": round(float(t_unified[-1][-1]), 3),
                    "target_update_hz": pmeta.get("target_update_hz"),
                    "duplicate_state_fraction": pmeta.get("duplicate_state_fraction"),
                    "timing": pmeta.get("timing"),
                    "excitation_kinds": sorted(
                        {k for k in kinds.tolist() if k not in TRANSITION_KINDS}
                    ),
                }
            )
            row_cursor += n
            seg_uid_cursor += len(phase_segments)

        if not session_entry["phases"]:
            raise MergeError(f"Session {name} contains no readable phase logs.")
        sessions_meta.append(session_entry)

    if ref_meta is None:
        raise MergeError("Nothing to merge.")

    cat = lambda parts, dt=None: (
        np.concatenate(parts).astype(dt) if dt else np.concatenate(parts)
    )
    t_all = cat(t_unified)
    if not np.all(np.diff(t_all) > 0.0):
        raise MergeError("Unified time is not strictly increasing; gaps are too small.")

    arrays = {
        "t": t_all,
        "t_phase": cat(t_phase),
        "q_des": cat(q_des),
        "gripper_q_des": cat(grip_des),
        "kp": cat(kp),
        "kd": cat(kd),
        "q": cat(q),
        "qd": cat(qd),
        "tau_est": cat(tau_est),
        "session_id": cat(session_ids, np.int32),
        "phase_id": cat(phase_ids, np.int32),
        "seg_id": cat(seg_ids, np.int32),
        "seg_uid": cat(seg_uids, np.int32),
        "is_transition": cat(is_transition, bool),
        "is_dup_state": cat(is_dup, bool),
    }

    n_total = arrays["t"].shape[0]
    n_excite = int((~arrays["is_transition"]).sum())
    n_usable = int((~arrays["is_transition"] & ~arrays["is_dup_state"]).sum())

    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator": "deploy/z1_uan_unify_dataset.py",
        "unified": True,
        "root": os.path.abspath(root),
        "num_samples": n_total,
        "num_sessions": len(sessions_meta),
        "num_segments": len(segments),
        "duration_s": round(float(t_all[-1] - t_all[0]), 3),
        "excitation_samples": n_excite,
        "excitation_duration_s": round(n_excite / float(ref_meta["log_hz"]), 1),
        "usable_samples": n_usable,
        "duplicate_state_fraction": round(float(arrays["is_dup_state"].mean()), 6),
        "transition_fraction": round(float(arrays["is_transition"].mean()), 6),
        # Carried over unchanged from the phase logs, having been verified
        # identical across every merged phase.
        "mode": ref_meta["mode"],
        "log_hz": ref_meta["log_hz"],
        "loop_hz": ref_meta.get("loop_hz"),
        "columns": ref_meta["columns"],
        "arm_kps_runtime": ref_meta["arm_kps_runtime"],
        "arm_kds_runtime": ref_meta["arm_kds_runtime"],
        "home_q": ref_meta.get("home_q"),
        "gripper_hold_q": ref_meta.get("gripper_hold_q"),
        "gripper_excited": ref_meta.get("gripper_excited"),
        "payload": ref_meta.get("payload"),
        "mount": ref_meta.get("mount"),
        "torque_note": ref_meta.get("torque_note"),
        "paper": ref_meta.get("paper"),
        "phases": phase_names,
        "sessions": sessions_meta,
        "segments": segments,
        "time_base": (
            "t is a synthetic monotonic timeline: each phase keeps its own "
            f"recorded clock (t_phase) offset by a nominal {phase_gap_s:.1f}s "
            f"inter-phase and {session_gap_s:.1f}s inter-session gap. The real "
            "wall-clock gaps are unlogged (the arm rests at home unlogged), so "
            "t is for ordering and plotting -- integrate within a phase only."
        ),
        "index_note": (
            "Rows are contiguous within a phase. Drop blends and settles with "
            "~is_transition; drop republished state packets with ~is_dup_state."
        ),
    }
    return arrays, meta


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------


def write_outputs(arrays: Dict, meta: Dict, out_dir: str, stem: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []

    n = arrays["t"].shape[0]
    ts_us = np.round(arrays["t"] * 1e6).astype(np.int64)
    ts_phase_us = np.round(arrays["t_phase"] * 1e6).astype(np.int64)

    # Same three-block schema as a phase log, so an existing reader works
    # unchanged, plus the row-level provenance index.
    data = {
        "arm_pd_tau_targets": {
            "q_des": arrays["q_des"],
            "gripperQ_des": arrays["gripper_q_des"],
            "kp": arrays["kp"],
            "kd": arrays["kd"],
            "timestamp": ts_us,
            "timestamp_phase": ts_phase_us,
        },
        "arm_control_data": {
            "q": arrays["q"],
            "qd": arrays["qd"],
            "tau_est": arrays["tau_est"],
            "timestamp": ts_us,
            "timestamp_phase": ts_phase_us,
        },
        "uan_index": {
            k: arrays[k]
            for k in (
                "session_id",
                "phase_id",
                "seg_id",
                "seg_uid",
                "is_transition",
                "is_dup_state",
            )
        },
        "uan_meta": meta,
    }

    pkl_path = os.path.join(out_dir, f"{stem}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    written.append(pkl_path)

    npz_path = os.path.join(out_dir, f"{stem}.npz")
    np.savez_compressed(npz_path, **arrays)
    written.append(npz_path)

    json_path = os.path.join(out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    written.append(json_path)

    assert n == arrays["q"].shape[0]
    return written


def print_summary(arrays: Dict, meta: Dict) -> None:
    hz = float(meta["log_hz"])
    print()
    print("=" * 72)
    print("Unified Z1 UAN dataset")
    print("=" * 72)
    print(f"  sessions            : {meta['num_sessions']}")
    print(f"  phases              : {', '.join(meta['phases'])}")
    print(f"  samples             : {meta['num_samples']:,} @ {hz:g} Hz")
    print(f"  segments            : {meta['num_segments']:,}")
    print(
        f"  excitation          : {meta['excitation_samples']:,} rows "
        f"({meta['excitation_duration_s'] / 60.0:.1f} min)"
    )
    print(
        f"  usable (no dup)     : {meta['usable_samples']:,} rows "
        f"({meta['usable_samples'] / hz / 60.0:.1f} min)"
    )
    print(f"  transitions         : {meta['transition_fraction'] * 100:.1f}% of rows")
    print(f"  duplicated state    : {meta['duplicate_state_fraction'] * 100:.1f}% of rows")
    print(f"  gains kp            : {meta['arm_kps_runtime']}")
    print(f"  gains kd            : {meta['arm_kds_runtime']}")
    print()
    print(f"  {'session':<18}{'phase':<22}{'rows':>10}{'min':>8}{'rows [start,end)':>22}")
    for s in meta["sessions"]:
        for p in s["phases"]:
            span = f"[{p['row_start']:,}, {p['row_end']:,})"
            print(
                f"  {s['name']:<18}{p['phase']:<22}{p['num_samples']:>10,}"
                f"{p['num_samples'] / hz / 60.0:>8.1f}{span:>22}"
            )
    print()
    print("  per-joint |q| range and peak |tau| over excitation rows:")
    keep = ~arrays["is_transition"]
    q = arrays["q"][keep][:, :NUM_ARM_JOINTS]
    tau = arrays["tau_est"][keep][:, :NUM_ARM_JOINTS]
    qd = arrays["qd"][keep][:, :NUM_ARM_JOINTS]
    for j in range(NUM_ARM_JOINTS):
        print(
            f"    j{j + 1}  q [{q[:, j].min():+.3f}, {q[:, j].max():+.3f}] rad   "
            f"|qd|max {np.abs(qd[:, j]).max():5.2f} rad/s   "
            f"|tau|max {np.abs(tau[:, j]).max():6.2f} Nm"
        )
    print()


# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Merge per-phase Z1 UAN logs into one unified dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", default="Log/z1_uan_v2",
                    help="Dataset root holding timestamped session directories.")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="Session directory names to merge (default: all).")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: <root>/unified).")
    ap.add_argument("--stem", default="uan_dataset",
                    help="Output file stem.")
    ap.add_argument("--inter-phase-gap-s", type=float, default=10.0,
                    help="Nominal unlogged rest inserted between phases on the "
                         "unified clock; match uan.inter_phase_rest_s.")
    ap.add_argument("--inter-session-gap-s", type=float, default=60.0,
                    help="Nominal gap inserted between sessions.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Merge and report, but write nothing.")
    args = ap.parse_args()

    try:
        sessions = find_sessions(args.root, args.sessions)
        arrays, meta = merge(
            args.root, sessions, args.inter_phase_gap_s, args.inter_session_gap_s
        )
    except MergeError as e:
        raise SystemExit(f"[UAN] merge failed: {e}")

    print_summary(arrays, meta)

    if args.dry_run:
        print("  --dry-run: nothing written.")
        return

    out_dir = args.out_dir or os.path.join(args.root, "unified")
    for path in write_outputs(arrays, meta, out_dir, args.stem):
        print(f"[UAN] wrote {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
