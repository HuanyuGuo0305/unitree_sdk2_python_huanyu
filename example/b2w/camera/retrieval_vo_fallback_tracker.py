#!/usr/bin/env python3
"""
AprilTag retrieval visual <-> Rear stereo VO fallback tracker.

Production placement
--------------------
CRL PC.

Inputs
------
1) B2AprilTagDetector
       full physical Tag pose T_B_FROM_TAG from FRONT/BACK B2 optical camera
2) RearVOUdpReceiver
       Rear stereo VO pose T_V_FROM_B(t), converted into CRL monotonic time

Transform convention
--------------------
T_A_FROM_B maps B -> A.

When AprilTag is visible at t0:

    T_V_FROM_TAG =
        T_V_FROM_B(t0)
        @ T_B_FROM_TAG(t0)

The complete PHYSICAL Tag pose is anchored, not only Tag center and not the
virtual retrieval target.

When AprilTag is visually lost:

    T_B_FROM_TAG(t) =
        inv(T_V_FROM_B(t))
        @ T_V_FROM_TAG

Then the retrieval target is always derived from the current physical Tag pose:

    p_retrieval^B(t) =
        T_B_FROM_TAG(t) @ [0, 0, retrieval_z, 1]

Safety
------
- FRONT/BACK AprilTag visual always overrides VO fallback.
- No anchor => no fabricated fallback.
- VO invalid => fallback invalid.
- Developer VO server session change => old Tag anchor discarded.
- VO epoch change => old Tag anchor discarded.
- Temporary VO invalidity within the same session/epoch does not destroy the
  anchor; fallback can resume if VO recovers without an epoch reset.
- All provider reads are latest-value / nonblocking.
"""

from __future__ import annotations

import argparse
import copy
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


SOURCE_VISUAL_FRONT = "VISUAL_FRONT"
SOURCE_VISUAL_BACK = "VISUAL_BACK"
SOURCE_VO = "RETRIEVAL_VO_FALLBACK"

DEFAULT_VISUAL_MAX_AGE_S = 0.200
DEFAULT_VO_MAX_AGE_MS = 250.0
DEFAULT_LOOKUP_MAX_GAP_MS = 150.0
DEFAULT_FUSION_HZ = 30.0
DEFAULT_TRACKER_MAX_AGE_MS = 250.0

DEFAULT_RETRIEVAL_TARGET_TAG = np.array(
    [0.0, 0.0, 1.0],
    dtype=np.float64,
)


def _as_T(
    value: Any,
) -> Optional[np.ndarray]:
    if value is None:
        return None

    try:
        T = np.asarray(
            value,
            dtype=np.float64,
        ).reshape(4, 4)
    except Exception:
        return None

    if not np.isfinite(T).all():
        return None

    if not np.allclose(
        T[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
    ):
        return None

    return T


def make_T_B_FROM_TAG(
    tag_center_base: Any,
    R_base_tag: Any,
) -> Optional[np.ndarray]:
    if (
        tag_center_base is None
        or R_base_tag is None
    ):
        return None

    try:
        t = np.asarray(
            tag_center_base,
            dtype=np.float64,
        ).reshape(3)

        R = np.asarray(
            R_base_tag,
            dtype=np.float64,
        ).reshape(3, 3)
    except Exception:
        return None

    if (
        not np.isfinite(t).all()
        or not np.isfinite(R).all()
    ):
        return None

    T = np.eye(
        4,
        dtype=np.float64,
    )

    T[:3, :3] = R
    T[:3, 3] = t

    return T


def derived_tag_outputs(
    T_B_FROM_TAG: np.ndarray,
    retrieval_target_tag: np.ndarray,
) -> Dict[str, Any]:
    T = np.asarray(
        T_B_FROM_TAG,
        dtype=np.float64,
    ).reshape(4, 4)

    p_ret_tag = np.asarray(
        retrieval_target_tag,
        dtype=np.float64,
    ).reshape(3)

    R = T[:3, :3]
    t = T[:3, 3]

    p_ret_b = (
        R @ p_ret_tag
        + t
    )

    return {
        "T_base_tag": [
            [
                float(x)
                for x in row
            ]
            for row in T
        ],

        "tag_center_base": [
            float(x)
            for x in t
        ],

        "R_base_tag": [
            [
                float(x)
                for x in row
            ]
            for row in R
        ],

        "tag_z_base": [
            float(x)
            for x in R[:, 2]
        ],

        "retrieval_target_base": [
            float(x)
            for x in p_ret_b
        ],
    }


class RetrievalVOFallbackTracker:
    """
    Asynchronous latest-value retrieval tracker.

    tag_provider must expose:
        get_latest_snapshot(max_age_s=...)

    vo_receiver must expose:
        get_latest_snapshot(max_age_ms=...)
        lookup_T_V_FROM_B(
            crl_monotonic_ns,
            epoch=...,
            max_gap_ms=...,
        )
    """

    def __init__(
        self,
        tag_provider: Any,
        vo_receiver: Any,
        retrieval_target_tag: np.ndarray =
            DEFAULT_RETRIEVAL_TARGET_TAG,
        visual_max_age_s: float =
            DEFAULT_VISUAL_MAX_AGE_S,
        vo_max_age_ms: float =
            DEFAULT_VO_MAX_AGE_MS,
        lookup_max_gap_ms: float =
            DEFAULT_LOOKUP_MAX_GAP_MS,
        fusion_hz: float =
            DEFAULT_FUSION_HZ,
    ):
        self.tag_provider = tag_provider
        self.vo_receiver = vo_receiver

        self.retrieval_target_tag = (
            np.asarray(
                retrieval_target_tag,
                dtype=np.float64,
            ).reshape(3)
        )

        self.visual_max_age_s = float(
            visual_max_age_s
        )

        self.vo_max_age_ms = float(
            vo_max_age_ms
        )

        self.lookup_max_gap_ms = float(
            lookup_max_gap_ms
        )

        self.fusion_hz = float(
            fusion_hz
        )

        if self.fusion_hz <= 0.0:
            raise ValueError(
                "fusion_hz must be > 0"
            )

        self._lock = threading.Lock()
        self._latest: Optional[
            Dict[str, Any]
        ] = None

        self._stop_event = threading.Event()
        self._thread: Optional[
            threading.Thread
        ] = None

        self._anchor_T_V_FROM_TAG: Optional[
            np.ndarray
        ] = None

        self._anchor_session_id: Optional[
            str
        ] = None

        self._anchor_epoch: Optional[
            int
        ] = None

        self._anchor_visual_host_ns: Optional[
            int
        ] = None

        self._anchor_visual_source: Optional[
            str
        ] = None

        self._anchor_count = 0

        self._last_visual_key: Optional[
            Tuple[int, str]
        ] = None

    def start(
        self,
    ) -> None:
        """
        Start only the fusion thread.

        The caller owns lifecycle of tag_provider and vo_receiver.
        """
        if self._thread is not None:
            raise RuntimeError(
                "RetrievalVOFallbackTracker already started"
            )

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._worker,
            name="retrieval-vo-fallback-tracker",
            daemon=True,
        )

        self._thread.start()

    def stop(
        self,
    ) -> None:
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(
                timeout=2.0
            )

        self._thread = None

    def clear_anchor(
        self,
    ) -> None:
        with self._lock:
            self._clear_anchor_locked()

    def _clear_anchor_locked(
        self,
    ) -> None:
        self._anchor_T_V_FROM_TAG = None
        self._anchor_session_id = None
        self._anchor_epoch = None
        self._anchor_visual_host_ns = None
        self._anchor_visual_source = None

    def _anchor_snapshot_locked(
        self,
    ) -> Dict[str, Any]:
        return {
            "anchor_valid":
                self._anchor_T_V_FROM_TAG
                is not None,

            "anchor_session_id":
                self._anchor_session_id,

            "anchor_epoch":
                self._anchor_epoch,

            "anchor_visual_host_monotonic_ns":
                self._anchor_visual_host_ns,

            "anchor_visual_source":
                self._anchor_visual_source,

            "T_V_FROM_TAG_anchor": (
                None
                if self._anchor_T_V_FROM_TAG
                is None
                else [
                    [
                        float(x)
                        for x in row
                    ]
                    for row
                    in self._anchor_T_V_FROM_TAG
                ]
            ),

            "anchor_count":
                int(
                    self._anchor_count
                ),
        }

    def get_latest_snapshot(
        self,
        max_age_ms: Optional[
            float
        ] = DEFAULT_TRACKER_MAX_AGE_MS,
    ) -> Optional[Dict[str, Any]]:
        """
        Nonblocking latest-value tracker read.
        """
        with self._lock:
            if self._latest is None:
                return None

            snap = copy.deepcopy(
                self._latest
            )

        now_ns = time.monotonic_ns()

        age_ms = (
            now_ns
            - int(
                snap[
                    "tracker_crl_monotonic_ns"
                ]
            )
        ) / 1e6

        snap[
            "tracker_age_ms"
        ] = float(
            age_ms
        )

        if (
            max_age_ms is not None
            and age_ms
            > float(
                max_age_ms
            )
        ):
            snap[
                "valid"
            ] = False

            snap[
                "reason"
            ] = "TRACKER_STALE"

        return snap

    def update_once(
        self,
    ) -> Dict[str, Any]:
        """
        One nonblocking fusion step.
        """
        now_ns = time.monotonic_ns()

        visual = (
            self.tag_provider
            .get_latest_snapshot(
                max_age_s=
                    self.visual_max_age_s
            )
        )

        vo = (
            self.vo_receiver
            .get_latest_snapshot(
                max_age_ms=
                    self.vo_max_age_ms
            )
        )

        visual_valid = bool(
            visual is not None
            and visual.get(
                "valid",
                False,
            )
        )

        vo_valid = bool(
            vo is not None
            and vo.get(
                "valid",
                False,
            )
        )

        session_id = (
            None
            if vo is None
            or vo.get(
                "session_id"
            )
            is None
            else str(
                vo[
                    "session_id"
                ]
            )
        )

        vo_epoch = (
            None
            if vo is None
            or vo.get(
                "epoch"
            )
            is None
            else int(
                vo[
                    "epoch"
                ]
            )
        )

        # Session and epoch are hard world-frame boundaries.
        boundary_reason = None

        with self._lock:
            if (
                self._anchor_T_V_FROM_TAG
                is not None
            ):
                if (
                    session_id is not None
                    and self._anchor_session_id
                    is not None
                    and session_id
                    != self._anchor_session_id
                ):
                    self._clear_anchor_locked()
                    boundary_reason = "VO_SESSION_CHANGED"

                elif (
                    vo_epoch is not None
                    and self._anchor_epoch
                    is not None
                    and vo_epoch
                    != self._anchor_epoch
                ):
                    self._clear_anchor_locked()
                    boundary_reason = "VO_EPOCH_CHANGED"

        if visual_valid:
            result = (
                self._handle_visual(
                    visual=visual,
                    vo=vo,
                    vo_valid=vo_valid,
                    session_id=session_id,
                    vo_epoch=vo_epoch,
                    now_ns=now_ns,
                )
            )
        else:
            if boundary_reason is not None:
                result = self._make_result(
                    valid=False,
                    source=None,
                    reason=boundary_reason,
                    T_B_FROM_TAG=None,
                    visual=visual,
                    vo=vo,
                    now_ns=now_ns,
                )
            else:
                result = (
                    self._handle_fallback(
                        visual=visual,
                        vo=vo,
                        vo_valid=vo_valid,
                        session_id=session_id,
                        vo_epoch=vo_epoch,
                        now_ns=now_ns,
                    )
                )

        with self._lock:
            result.update(
                self._anchor_snapshot_locked()
            )

            self._latest = copy.deepcopy(
                result
            )

        return copy.deepcopy(
            result
        )

    def _handle_visual(
        self,
        visual: Dict[str, Any],
        vo: Optional[Dict[str, Any]],
        vo_valid: bool,
        session_id: Optional[str],
        vo_epoch: Optional[int],
        now_ns: int,
    ) -> Dict[str, Any]:
        T_B_FROM_TAG = (
            make_T_B_FROM_TAG(
                visual.get(
                    "tag_center_base"
                ),
                visual.get(
                    "R_base_tag"
                ),
            )
        )

        capture_ns = visual.get(
            "capture_host_monotonic_ns"
        )

        camera_source = str(
            visual.get(
                "source",
                "",
            )
        ).upper()

        if camera_source == "FRONT":
            output_source = (
                SOURCE_VISUAL_FRONT
            )
        elif camera_source == "BACK":
            output_source = (
                SOURCE_VISUAL_BACK
            )
        else:
            output_source = (
                "VISUAL_UNKNOWN"
            )

        if (
            T_B_FROM_TAG is None
            or capture_ns is None
        ):
            with self._lock:
                self._clear_anchor_locked()

            return self._make_result(
                valid=False,
                source=None,
                reason=
                    "VISUAL_TAG_GEOMETRY_INVALID",
                T_B_FROM_TAG=None,
                visual=visual,
                vo=vo,
                now_ns=now_ns,
            )

        capture_ns = int(
            capture_ns
        )

        visual_key = (
            capture_ns,
            camera_source,
        )

        with self._lock:
            is_new_visual = (
                self._last_visual_key
                != visual_key
            )

            if is_new_visual:
                self._last_visual_key = (
                    visual_key
                )

        anchor_refresh_ok = False

        if (
            vo_valid
            and session_id is not None
            and vo_epoch is not None
        ):
            T_V_FROM_B_t0 = (
                self.vo_receiver
                .lookup_T_V_FROM_B(
                    capture_ns,
                    epoch=vo_epoch,
                    max_gap_ms=
                        self.lookup_max_gap_ms,
                )
            )

            T_V_FROM_B_t0 = _as_T(
                T_V_FROM_B_t0
            )

            if T_V_FROM_B_t0 is not None:
                T_V_FROM_TAG = (
                    T_V_FROM_B_t0
                    @ T_B_FROM_TAG
                )

                with self._lock:
                    self._anchor_T_V_FROM_TAG = (
                        T_V_FROM_TAG.copy()
                    )

                    self._anchor_session_id = str(
                        session_id
                    )

                    self._anchor_epoch = int(
                        vo_epoch
                    )

                    self._anchor_visual_host_ns = (
                        capture_ns
                    )

                    self._anchor_visual_source = (
                        camera_source
                    )

                    self._anchor_count += 1

                anchor_refresh_ok = True

        if (
            not anchor_refresh_ok
            and is_new_visual
        ):
            # New visual truth could not be synchronized to the VO world.
            # Do not retain an older, inconsistent fallback anchor.
            with self._lock:
                self._clear_anchor_locked()

        # Visual always wins.
        return self._make_result(
            valid=True,
            source=output_source,
            reason=(
                "VALID_VISUAL_ANCHORED"
                if anchor_refresh_ok
                else "VALID_VISUAL_NO_VO_ANCHOR"
            ),
            T_B_FROM_TAG=T_B_FROM_TAG,
            visual=visual,
            vo=vo,
            now_ns=now_ns,
        )

    def _handle_fallback(
        self,
        visual: Optional[
            Dict[str, Any]
        ],
        vo: Optional[
            Dict[str, Any]
        ],
        vo_valid: bool,
        session_id: Optional[str],
        vo_epoch: Optional[int],
        now_ns: int,
    ) -> Dict[str, Any]:
        with self._lock:
            anchor_T_V_FROM_TAG = (
                None
                if self._anchor_T_V_FROM_TAG
                is None
                else self._anchor_T_V_FROM_TAG.copy()
            )

            anchor_session_id = (
                self._anchor_session_id
            )

            anchor_epoch = (
                self._anchor_epoch
            )

        if anchor_T_V_FROM_TAG is None:
            return self._make_result(
                valid=False,
                source=None,
                reason="NO_TAG_ANCHOR",
                T_B_FROM_TAG=None,
                visual=visual,
                vo=vo,
                now_ns=now_ns,
            )

        if not vo_valid:
            return self._make_result(
                valid=False,
                source=None,
                reason=(
                    "VO_NOT_VALID:"
                    + str(
                        None
                        if vo is None
                        else vo.get(
                            "reason"
                        )
                    )
                ),
                T_B_FROM_TAG=None,
                visual=visual,
                vo=vo,
                now_ns=now_ns,
            )

        if (
            session_id is None
            or anchor_session_id is None
            or session_id
            != anchor_session_id
        ):
            with self._lock:
                self._clear_anchor_locked()

            return self._make_result(
                valid=False,
                source=None,
                reason=
                    "VO_SESSION_MISMATCH",
                T_B_FROM_TAG=None,
                visual=visual,
                vo=vo,
                now_ns=now_ns,
            )

        if (
            vo_epoch is None
            or anchor_epoch is None
            or int(
                vo_epoch
            )
            != int(
                anchor_epoch
            )
        ):
            with self._lock:
                self._clear_anchor_locked()

            return self._make_result(
                valid=False,
                source=None,
                reason=
                    "VO_EPOCH_MISMATCH",
                T_B_FROM_TAG=None,
                visual=visual,
                vo=vo,
                now_ns=now_ns,
            )

        T_V_FROM_B = _as_T(
            vo.get(
                "T_V_FROM_B"
            )
        )

        if T_V_FROM_B is None:
            return self._make_result(
                valid=False,
                source=None,
                reason="VO_POSE_INVALID",
                T_B_FROM_TAG=None,
                visual=visual,
                vo=vo,
                now_ns=now_ns,
            )

        T_B_FROM_V = np.linalg.inv(
            T_V_FROM_B
        )

        T_B_FROM_TAG = (
            T_B_FROM_V
            @ anchor_T_V_FROM_TAG
        )

        return self._make_result(
            valid=True,
            source=SOURCE_VO,
            reason=
                "VALID_VO_FALLBACK",
            T_B_FROM_TAG=
                T_B_FROM_TAG,
            visual=visual,
            vo=vo,
            now_ns=now_ns,
        )

    def _make_result(
        self,
        valid: bool,
        source: Optional[str],
        reason: str,
        T_B_FROM_TAG: Optional[
            np.ndarray
        ],
        visual: Optional[
            Dict[str, Any]
        ],
        vo: Optional[
            Dict[str, Any]
        ],
        now_ns: int,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "valid": bool(
                valid
            ),

            "source": source,

            "reason": str(
                reason
            ),

            "tracker_crl_monotonic_ns":
                int(
                    now_ns
                ),

            "state_crl_monotonic_ns": (
                int(
                    visual[
                        "capture_host_monotonic_ns"
                    ]
                )
                if (
                    source
                    in (
                        SOURCE_VISUAL_FRONT,
                        SOURCE_VISUAL_BACK,
                        "VISUAL_UNKNOWN",
                    )
                    and visual is not None
                    and visual.get(
                        "capture_host_monotonic_ns"
                    )
                    is not None
                )
                else (
                    int(
                        vo[
                            "capture_crl_monotonic_ns"
                        ]
                    )
                    if (
                        source == SOURCE_VO
                        and vo is not None
                        and vo.get(
                            "capture_crl_monotonic_ns"
                        )
                        is not None
                    )
                    else None
                )
            ),

            "T_base_tag": None,
            "tag_center_base": None,
            "R_base_tag": None,
            "tag_z_base": None,
            "retrieval_target_base": None,

            "visual_valid": bool(
                visual is not None
                and visual.get(
                    "valid",
                    False,
                )
            ),

            "visual_source": (
                None
                if visual is None
                else visual.get(
                    "source"
                )
            ),

            "visual_reason": (
                None
                if visual is None
                else visual.get(
                    "reason"
                )
            ),

            "visual_age_ms": (
                None
                if visual is None
                else visual.get(
                    "age_ms"
                )
            ),

            "visual_capture_crl_monotonic_ns": (
                None
                if visual is None
                else visual.get(
                    "capture_host_monotonic_ns"
                )
            ),

            "front_valid": bool(
                visual is not None
                and visual.get(
                    "front_valid",
                    False,
                )
            ),

            "back_valid": bool(
                visual is not None
                and visual.get(
                    "back_valid",
                    False,
                )
            ),

            "reprojection_rmse_px": (
                None
                if visual is None
                else visual.get(
                    "reprojection_rmse_px"
                )
            ),

            "vo_valid": bool(
                vo is not None
                and vo.get(
                    "valid",
                    False,
                )
            ),

            "vo_reason": (
                None
                if vo is None
                else vo.get(
                    "reason"
                )
            ),

            "vo_session_id": (
                None
                if vo is None
                else vo.get(
                    "session_id"
                )
            ),

            "vo_epoch": (
                None
                if vo is None
                else vo.get(
                    "epoch"
                )
            ),

            "vo_age_ms": (
                None
                if vo is None
                else vo.get(
                    "age_ms"
                )
            ),

            "vo_capture_crl_monotonic_ns": (
                None
                if vo is None
                else vo.get(
                    "capture_crl_monotonic_ns"
                )
            ),

            "vo_transport_ms": (
                None
                if vo is None
                else vo.get(
                    "transport_ms"
                )
            ),
        }

        if (
            valid
            and T_B_FROM_TAG
            is not None
        ):
            result.update(
                derived_tag_outputs(
                    T_B_FROM_TAG,
                    self.retrieval_target_tag,
                )
            )

        return result

    def _worker(
        self,
    ) -> None:
        period_s = (
            1.0
            / self.fusion_hz
        )

        next_t = time.monotonic()

        while not self._stop_event.is_set():
            try:
                self.update_once()
            except BaseException as exc:
                now_ns = (
                    time.monotonic_ns()
                )

                with self._lock:
                    fatal = {
                        "valid": False,
                        "source": None,
                        "reason":
                            "TRACKER_EXCEPTION",
                        "fatal_error":
                            repr(
                                exc
                            ),
                        "T_base_tag": None,
                        "tag_center_base": None,
                        "R_base_tag": None,
                        "tag_z_base": None,
                        "retrieval_target_base":
                            None,
                        "tracker_crl_monotonic_ns":
                            int(
                                now_ns
                            ),
                    }

                    fatal.update(
                        self._anchor_snapshot_locked()
                    )

                    self._latest = fatal

            next_t += period_s

            delay = (
                next_t
                - time.monotonic()
            )

            if delay > 0.0:
                self._stop_event.wait(
                    delay
                )
            else:
                next_t = (
                    time.monotonic()
                )


# =============================================================================
# Standalone CRL hardware test
# =============================================================================

def _resolve_repo_root(
    file_path: Path,
) -> Path:
    """
    Expected placement:
        <repo>/example/b2w/camera/retrieval_vo_fallback_tracker.py
    """
    p = file_path.resolve()

    if len(
        p.parents
    ) >= 4:
        candidate = p.parents[3]

        if (
            candidate
            / "example"
            / "b2w"
        ).exists():
            return candidate

    # Fallback to current working directory if invoked from repo root.
    cwd = Path.cwd().resolve()

    if (
        cwd
        / "example"
        / "b2w"
    ).exists():
        return cwd

    raise RuntimeError(
        "Could not resolve Unitree SDK repository root"
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "interface",
        help=(
            "Unitree DDS interface, "
            "e.g. enxa0cec819e15f"
        ),
    )

    parser.add_argument(
        "--developer-host",
        default="192.168.123.164",
    )

    parser.add_argument(
        "--data-port",
        type=int,
        default=50020,
    )

    parser.add_argument(
        "--sync-port",
        type=int,
        default=50021,
    )

    parser.add_argument(
        "--tag-id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--tag-size",
        type=float,
        default=0.195,
    )

    parser.add_argument(
        "--retrieval-z",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--visual-max-age",
        type=float,
        default=
            DEFAULT_VISUAL_MAX_AGE_S,
    )

    parser.add_argument(
        "--vo-max-age-ms",
        type=float,
        default=
            DEFAULT_VO_MAX_AGE_MS,
    )

    parser.add_argument(
        "--front-intrinsics",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--back-intrinsics",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--print-period",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()

    # Local imports keep tracker class hardware/network independent.
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
    )

    from b2_apriltag_detector import (
        B2AprilTagDetector,
        resolve_back_intrinsics,
    )

    from rear_vo_udp_receiver import (
        RearVOUdpReceiver,
    )

    repo_root = _resolve_repo_root(
        Path(__file__)
    )

    if args.front_intrinsics is None:
        front_intrinsics = (
            repo_root
            / "extrinsic_calib"
            / "front_rgb"
            / "front_intrinsics.npz"
        )
    else:
        front_intrinsics = (
            args.front_intrinsics
            .expanduser()
        )

        if not front_intrinsics.is_absolute():
            front_intrinsics = (
                repo_root
                / front_intrinsics
            )

    front_intrinsics = (
        front_intrinsics.resolve()
    )

    if not front_intrinsics.exists():
        raise FileNotFoundError(
            "FRONT intrinsics not found: "
            f"{front_intrinsics}"
        )

    back_intrinsics = (
        resolve_back_intrinsics(
            repo_root,
            args.back_intrinsics,
        )
    )

    retrieval_target_tag = np.array(
        [
            0.0,
            0.0,
            float(
                args.retrieval_z
            ),
        ],
        dtype=np.float64,
    )

    ChannelFactoryInitialize(
        0,
        args.interface,
    )

    detector = B2AprilTagDetector(
        repo_root=repo_root,
        front_intrinsics=
            front_intrinsics,
        back_intrinsics=
            back_intrinsics,
        tag_id=args.tag_id,
        tag_size_m=args.tag_size,
        retrieval_target_tag=
            retrieval_target_tag,
        visual_max_age_s=
            args.visual_max_age,
    )

    receiver = RearVOUdpReceiver(
        developer_host=
            args.developer_host,
        data_port=
            args.data_port,
        sync_port=
            args.sync_port,
    )

    tracker = RetrievalVOFallbackTracker(
        tag_provider=detector,
        vo_receiver=receiver,
        retrieval_target_tag=
            retrieval_target_tag,
        visual_max_age_s=
            args.visual_max_age,
        vo_max_age_ms=
            args.vo_max_age_ms,
    )

    print("=" * 112)
    print(
        "APRILTAG RETRIEVAL VISUAL <-> REAR VO FALLBACK TRACKER"
    )
    print("=" * 112)
    print(
        "CRL PC"
    )
    print(
        "B2 FRONT/BACK : visual full physical Tag pose"
    )
    print(
        "Rear VO       :",
        f"{args.developer_host}:"
        f"{args.data_port}",
    )
    print(
        "Tag ID/size   :",
        f"{args.tag_id} / "
        f"{args.tag_size:.3f} m",
    )
    print(
        "Retrieval Tag :",
        retrieval_target_tag,
    )
    print(
        "Visual        : always overrides VO"
    )
    print(
        "Session/epoch : hard fallback boundaries"
    )
    print(
        "IMPORTANT     : stop standalone rear_vo_udp_receiver.py first"
    )
    print("=" * 112)

    print(
        "[INIT] B2 camera RPC clients..."
    )

    detector.initialize_clients()

    detector.start()
    receiver.start()
    tracker.start()

    print(
        "[INIT] detector + VO receiver + tracker started"
    )

    stop_requested = False

    def on_signal(
        signum,
        frame,
    ):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(
        signal.SIGINT,
        on_signal,
    )

    signal.signal(
        signal.SIGTERM,
        on_signal,
    )

    last_print = 0.0

    try:
        while not stop_requested:
            now = time.monotonic()

            if (
                now - last_print
                >= args.print_period
            ):
                snap = (
                    tracker
                    .get_latest_snapshot()
                )

                if snap is None:
                    print(
                        "[WAIT] tracker not ready"
                    )

                elif snap["valid"]:
                    p_tag = snap[
                        "tag_center_base"
                    ]

                    p_ret = snap[
                        "retrieval_target_base"
                    ]

                    print(
                        "[VALID] "
                        f"source="
                        f"{snap['source']:<21s} "
                        f"Tag_B="
                        f"[{p_tag[0]:+.3f},"
                        f"{p_tag[1]:+.3f},"
                        f"{p_tag[2]:+.3f}]m "
                        f"Ret_B="
                        f"[{p_ret[0]:+.3f},"
                        f"{p_ret[1]:+.3f},"
                        f"{p_ret[2]:+.3f}]m "
                        f"anchor="
                        f"{int(snap['anchor_valid'])} "
                        f"a_epoch="
                        f"{snap['anchor_epoch']} "
                        f"vo_epoch="
                        f"{snap['vo_epoch']} "
                        f"session="
                        f"{str(snap['vo_session_id'])[:8]}"
                    )

                else:
                    print(
                        "[INVALID] "
                        f"reason={snap['reason']} "
                        f"visual="
                        f"{snap.get('visual_reason')} "
                        f"vo="
                        f"{snap.get('vo_reason')} "
                        f"anchor="
                        f"{int(snap.get('anchor_valid', False))} "
                        f"a_epoch="
                        f"{snap.get('anchor_epoch')} "
                        f"vo_epoch="
                        f"{snap.get('vo_epoch')} "
                        f"session="
                        f"{str(snap.get('vo_session_id'))[:8]}"
                    )

                last_print = now

            time.sleep(
                0.01
            )

    finally:
        tracker.stop()
        receiver.stop()
        detector.stop()

    print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
