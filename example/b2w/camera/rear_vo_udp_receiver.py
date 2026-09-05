#!/usr/bin/env python3
"""
CRL-side receiver for the integrated developer perception server.

Receives two independent packet streams on one UDP port:

    VO_STATE
        Rear stereo VO T_V_FROM_B
        + short time history
        + developer->CRL clock conversion

    OBJECT_STATE
        Already-fused Front RGB-D / Rear VO object position
        in Base coordinates.

Backward compatibility
----------------------
get_latest_snapshot()
    still returns the latest VO snapshot.

lookup_T_V_FROM_B()
    unchanged; used by RetrievalVOFallbackTracker.

New API
-------
get_latest_object_snapshot()
    returns the fused object state.

anchor_base_height(projected_gravity_b, height_m=...)
    explicitly anchors VO-derived Base height at a known physical height.

get_base_height_snapshot(projected_gravity_b=...)
    returns the latest VO-derived Base height.  Same-epoch updates require only
    VO translation; projected_gravity_b is used when a VO session/epoch changes
    so height can be re-anchored continuously in the new VO frame.

All reads are latest-value and nonblocking.

Clock exchange
--------------
t0 = CRL send
t1 = DEV receive
t2 = DEV send
t3 = CRL receive

offset(dev - crl)
  = ((t1 - t0) + (t2 - t3)) / 2

Convert:
    t_crl ~= t_dev - offset
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import socket
import sys
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


PROTOCOL_VERSION = 1
MAGIC = "B2W_REAR_VO"

DEFAULT_DATA_PORT = 50020
DEFAULT_SYNC_PORT = 50021

DEFAULT_SYNC_PERIOD_S = 0.25
DEFAULT_SYNC_WINDOW = 64
DEFAULT_SYNC_BEST_SAMPLES = 8

DEFAULT_HISTORY_SECONDS = 10.0
DEFAULT_MAX_AGE_MS = 250.0
DEFAULT_OBJECT_MAX_AGE_MS = 300.0
DEFAULT_LOOKUP_MAX_GAP_MS = 150.0
DEFAULT_BASE_HEIGHT_M = 0.6017


def quat_wxyz_from_rotmat(
    R: np.ndarray,
) -> np.ndarray:
    R = np.asarray(
        R,
        dtype=np.float64,
    ).reshape(3, 3)

    tr = float(
        np.trace(R)
    )

    if tr > 0.0:
        s = math.sqrt(
            tr + 1.0
        ) * 2.0

        q = np.array(
            [
                0.25 * s,
                (
                    R[2, 1]
                    - R[1, 2]
                ) / s,
                (
                    R[0, 2]
                    - R[2, 0]
                ) / s,
                (
                    R[1, 0]
                    - R[0, 1]
                ) / s,
            ],
            dtype=np.float64,
        )

    elif (
        R[0, 0] > R[1, 1]
        and R[0, 0] > R[2, 2]
    ):
        s = math.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0

        q = np.array(
            [
                (
                    R[2, 1]
                    - R[1, 2]
                ) / s,
                0.25 * s,
                (
                    R[0, 1]
                    + R[1, 0]
                ) / s,
                (
                    R[0, 2]
                    + R[2, 0]
                ) / s,
            ],
            dtype=np.float64,
        )

    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0

        q = np.array(
            [
                (
                    R[0, 2]
                    - R[2, 0]
                ) / s,
                (
                    R[0, 1]
                    + R[1, 0]
                ) / s,
                0.25 * s,
                (
                    R[1, 2]
                    + R[2, 1]
                ) / s,
            ],
            dtype=np.float64,
        )

    else:
        s = math.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0

        q = np.array(
            [
                (
                    R[1, 0]
                    - R[0, 1]
                ) / s,
                (
                    R[0, 2]
                    + R[2, 0]
                ) / s,
                (
                    R[1, 2]
                    + R[2, 1]
                ) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )

    q /= max(
        np.linalg.norm(q),
        1e-12,
    )

    return q


def rotmat_from_quat_wxyz(
    q: np.ndarray,
) -> np.ndarray:
    q = np.asarray(
        q,
        dtype=np.float64,
    ).reshape(4)

    q /= max(
        np.linalg.norm(q),
        1e-12,
    )

    w, x, y, z = q

    return np.array(
        [
            [
                1.0 - 2.0 * (y*y + z*z),
                2.0 * (x*y - z*w),
                2.0 * (x*z + y*w),
            ],
            [
                2.0 * (x*y + z*w),
                1.0 - 2.0 * (x*x + z*z),
                2.0 * (y*z - x*w),
            ],
            [
                2.0 * (x*z - y*w),
                2.0 * (y*z + x*w),
                1.0 - 2.0 * (x*x + y*y),
            ],
        ],
        dtype=np.float64,
    )


def slerp_wxyz(
    q0: np.ndarray,
    q1: np.ndarray,
    alpha: float,
) -> np.ndarray:
    q0 = np.asarray(
        q0,
        dtype=np.float64,
    ).reshape(4)

    q1 = np.asarray(
        q1,
        dtype=np.float64,
    ).reshape(4)

    q0 /= max(
        np.linalg.norm(q0),
        1e-12,
    )

    q1 /= max(
        np.linalg.norm(q1),
        1e-12,
    )

    dot = float(
        np.dot(
            q0,
            q1,
        )
    )

    if dot < 0.0:
        q1 = -q1
        dot = -dot

    dot = float(
        np.clip(
            dot,
            -1.0,
            1.0,
        )
    )

    if dot > 0.9995:
        q = (
            (1.0 - alpha) * q0
            + alpha * q1
        )

        q /= max(
            np.linalg.norm(q),
            1e-12,
        )

        return q

    theta = math.acos(
        dot
    )

    s = math.sin(
        theta
    )

    a = (
        math.sin(
            (1.0 - alpha)
            * theta
        )
        / s
    )

    b = (
        math.sin(
            alpha
            * theta
        )
        / s
    )

    return (
        a * q0
        + b * q1
    )


class ClockOffsetEstimator:
    """
    Estimate developer_monotonic - crl_monotonic.

    Uses a rolling set of NTP samples, selects the K lowest RTT samples,
    then takes median offset among those samples.
    """

    def __init__(
        self,
        max_samples: int =
            DEFAULT_SYNC_WINDOW,
        best_samples: int =
            DEFAULT_SYNC_BEST_SAMPLES,
    ):
        self.max_samples = int(
            max_samples
        )

        self.best_samples = int(
            best_samples
        )

        self.samples: Deque[
            Tuple[int, int]
        ] = deque(
            maxlen=
                self.max_samples
        )

        self.offset_dev_minus_crl_ns: Optional[
            int
        ] = None

        self.best_rtt_ns: Optional[
            int
        ] = None

        self.offset_spread_ns: Optional[
            int
        ] = None

    def add_sample(
        self,
        t0_crl_ns: int,
        t1_dev_ns: int,
        t2_dev_ns: int,
        t3_crl_ns: int,
    ) -> None:
        rtt_ns = (
            (t3_crl_ns - t0_crl_ns)
            - (t2_dev_ns - t1_dev_ns)
        )

        if rtt_ns < 0:
            return

        offset_ns = int(
            round(
                (
                    (t1_dev_ns - t0_crl_ns)
                    + (t2_dev_ns - t3_crl_ns)
                )
                / 2.0
            )
        )

        self.samples.append(
            (
                int(
                    rtt_ns
                ),
                int(
                    offset_ns
                ),
            )
        )

        ranked = sorted(
            self.samples,
            key=lambda x:
                x[0],
        )

        k = min(
            self.best_samples,
            len(
                ranked
            ),
        )

        best = ranked[:k]

        offsets = np.asarray(
            [
                x[1]
                for x in best
            ],
            dtype=np.int64,
        )

        self.offset_dev_minus_crl_ns = int(
            np.median(
                offsets
            )
        )

        self.best_rtt_ns = int(
            best[0][0]
        )

        if len(
            offsets
        ) >= 2:
            self.offset_spread_ns = int(
                np.max(
                    np.abs(
                        offsets
                        - self.offset_dev_minus_crl_ns
                    )
                )
            )
        else:
            self.offset_spread_ns = 0

    @property
    def ready(
        self,
    ) -> bool:
        return (
            self.offset_dev_minus_crl_ns
            is not None
            and len(
                self.samples
            )
            >= 3
        )

    def dev_to_crl_ns(
        self,
        dev_ns: int,
    ) -> Optional[int]:
        if (
            self.offset_dev_minus_crl_ns
            is None
        ):
            return None

        return int(
            int(
                dev_ns
            )
            - self.offset_dev_minus_crl_ns
        )

    def snapshot(
        self,
    ) -> Dict[str, Any]:
        return {
            "ready": bool(
                self.ready
            ),

            "samples": int(
                len(
                    self.samples
                )
            ),

            "offset_dev_minus_crl_ms": (
                None
                if self.offset_dev_minus_crl_ns
                is None
                else (
                    self.offset_dev_minus_crl_ns
                    / 1e6
                )
            ),

            "best_rtt_ms": (
                None
                if self.best_rtt_ns
                is None
                else (
                    self.best_rtt_ns
                    / 1e6
                )
            ),

            "offset_spread_ms": (
                None
                if self.offset_spread_ns
                is None
                else (
                    self.offset_spread_ns
                    / 1e6
                )
            ),
        }


class VOBaseHeightEstimator:
    """
    Stateful VO-derived Base-height estimator.

    The estimator does NOT assume that any raw VO axis is vertical.

    At an explicit physical anchor:
        up_B = -projected_gravity_B
        up_V = R_V_FROM_B(anchor) @ up_B

    Then:
        h(t) = h_anchor
               + up_V.T @ (
                   p_V_FROM_B(t)
                   - p_V_FROM_B(anchor)
                 )

    For the current VO session/epoch, up_V is frozen.  This is the same
    gravity-projected estimator validated on the real B2W with a
    DEFAULT -> SQUAT -> DEFAULT cycle.

    If the developer session or VO epoch changes:
      - preserve last valid physical height,
      - anchor the first valid pose of the new epoch at that height,
      - recompute up_V from the new T_V_FROM_B and current B2 IMU gravity.

    Motion that occurs while VO is unavailable remains unobservable.
    """

    def __init__(
        self,
        nominal_height_m: float =
            DEFAULT_BASE_HEIGHT_M,
    ) -> None:
        self.nominal_height_m = float(
            nominal_height_m
        )

        self.anchored = False

        self.session_id: Optional[
            str
        ] = None

        self.epoch: Optional[
            int
        ] = None

        self.anchor_pos_v: Optional[
            np.ndarray
        ] = None

        self.anchor_position_b0: Optional[
            np.ndarray
        ] = None

        self.up_v: Optional[
            np.ndarray
        ] = None

        self.anchor_height_m = (
            self.nominal_height_m
        )

        self.last_height_m = (
            self.nominal_height_m
        )

        self.last_delta_h_m = 0.0

        self.last_height_b0z_m: Optional[
            float
        ] = None

        self.reanchor_count = 0

    @staticmethod
    def _normalize_vec3(
        value: Any,
    ) -> Optional[np.ndarray]:
        try:
            v = np.asarray(
                value,
                dtype=np.float64,
            ).reshape(
                3,
            )
        except Exception:
            return None

        if not np.isfinite(
            v
        ).all():
            return None

        n = float(
            np.linalg.norm(
                v
            )
        )

        if (
            not np.isfinite(
                n
            )
            or n < 1.0e-12
        ):
            return None

        return (
            v
            / n
        )

    @staticmethod
    def _extract_pose(
        vo: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        value = vo.get(
            "T_V_FROM_B"
        )

        if value is None:
            return None

        try:
            T = np.asarray(
                value,
                dtype=np.float64,
            ).reshape(
                4,
                4,
            )
        except Exception:
            return None

        if not np.isfinite(
            T
        ).all():
            return None

        return T

    @staticmethod
    def _extract_position_b0(
        vo: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        value = vo.get(
            "position_B0_m"
        )

        if value is None:
            return None

        try:
            p = np.asarray(
                value,
                dtype=np.float64,
            ).reshape(
                3,
            )
        except Exception:
            return None

        if not np.isfinite(
            p
        ).all():
            return None

        return p

    @classmethod
    def _make_up_v(
        cls,
        T_v_from_b: np.ndarray,
        projected_gravity_b: Any,
    ) -> Optional[np.ndarray]:
        gravity_b = (
            cls._normalize_vec3(
                projected_gravity_b
            )
        )

        if gravity_b is None:
            return None

        up_b = -gravity_b

        up_v = (
            np.asarray(
                T_v_from_b[
                    :3,
                    :3,
                ],
                dtype=np.float64,
            )
            @ up_b
        )

        return (
            cls._normalize_vec3(
                up_v
            )
        )

    def reset(
        self,
        nominal_height_m: Optional[
            float
        ] = None,
    ) -> None:
        if nominal_height_m is not None:
            self.nominal_height_m = float(
                nominal_height_m
            )

        self.anchored = False
        self.session_id = None
        self.epoch = None
        self.anchor_pos_v = None
        self.anchor_position_b0 = None
        self.up_v = None

        self.anchor_height_m = (
            self.nominal_height_m
        )

        self.last_height_m = (
            self.nominal_height_m
        )

        self.last_delta_h_m = 0.0
        self.last_height_b0z_m = None
        self.reanchor_count = 0

    def anchor(
        self,
        vo: Dict[str, Any],
        projected_gravity_b: Any,
        height_m: Optional[
            float
        ] = None,
    ) -> Dict[str, Any]:
        if not bool(
            vo.get(
                "valid",
                False,
            )
        ):
            return self._make_snapshot(
                valid=False,
                reason=(
                    "ANCHOR_VO_"
                    + str(
                        vo.get(
                            "reason",
                            "INVALID",
                        )
                    )
                ),
                vo=vo,
            )

        T = self._extract_pose(
            vo
        )

        if T is None:
            return self._make_snapshot(
                valid=False,
                reason=
                    "ANCHOR_BAD_T_V_FROM_B",
                vo=vo,
            )

        up_v = self._make_up_v(
            T,
            projected_gravity_b,
        )

        if up_v is None:
            return self._make_snapshot(
                valid=False,
                reason=
                    "ANCHOR_BAD_GRAVITY_B",
                vo=vo,
            )

        target_height = (
            self.nominal_height_m
            if height_m is None
            else float(
                height_m
            )
        )

        if not np.isfinite(
            target_height
        ):
            return self._make_snapshot(
                valid=False,
                reason=
                    "ANCHOR_BAD_HEIGHT",
                vo=vo,
            )

        p_b0 = (
            self._extract_position_b0(
                vo
            )
        )

        self.session_id = str(
            vo.get(
                "session_id",
                "",
            )
        )

        self.epoch = int(
            vo.get(
                "epoch",
                -1,
            )
        )

        self.anchor_pos_v = (
            T[
                :3,
                3,
            ].copy()
        )

        self.anchor_position_b0 = (
            None
            if p_b0 is None
            else p_b0.copy()
        )

        self.up_v = (
            up_v.copy()
        )

        self.anchor_height_m = (
            target_height
        )

        self.last_height_m = (
            target_height
        )

        self.last_delta_h_m = 0.0
        self.last_height_b0z_m = (
            target_height
        )

        self.anchored = True

        return self._make_snapshot(
            valid=True,
            reason="ANCHORED",
            vo=vo,
            reanchored=False,
        )

    def update(
        self,
        vo: Dict[str, Any],
        projected_gravity_b: Any =
            None,
    ) -> Dict[str, Any]:
        if not self.anchored:
            return self._make_snapshot(
                valid=False,
                reason=
                    "NOT_ANCHORED",
                vo=vo,
            )

        if not bool(
            vo.get(
                "valid",
                False,
            )
        ):
            return self._make_snapshot(
                valid=False,
                reason=(
                    "VO_"
                    + str(
                        vo.get(
                            "reason",
                            "INVALID",
                        )
                    )
                ),
                vo=vo,
            )

        T = self._extract_pose(
            vo
        )

        if T is None:
            return self._make_snapshot(
                valid=False,
                reason=
                    "BAD_T_V_FROM_B",
                vo=vo,
            )

        session_id = str(
            vo.get(
                "session_id",
                "",
            )
        )

        epoch = int(
            vo.get(
                "epoch",
                -1,
            )
        )

        changed_session = (
            self.session_id
            is not None
            and session_id
            != self.session_id
        )

        changed_epoch = (
            self.epoch
            is not None
            and epoch
            != self.epoch
        )

        if (
            changed_session
            or changed_epoch
        ):
            up_v = self._make_up_v(
                T,
                projected_gravity_b,
            )

            if up_v is None:
                return self._make_snapshot(
                    valid=False,
                    reason=(
                        "REANCHOR_SESSION_NEEDS_GRAVITY"
                        if changed_session
                        else "REANCHOR_EPOCH_NEEDS_GRAVITY"
                    ),
                    vo=vo,
                )

            # Preserve the physical height accumulated before VO reset.
            self.anchor_height_m = float(
                self.last_height_m
            )

            self.session_id = (
                session_id
            )

            self.epoch = epoch

            self.anchor_pos_v = (
                T[
                    :3,
                    3,
                ].copy()
            )

            self.up_v = (
                up_v.copy()
            )

            p_b0 = (
                self._extract_position_b0(
                    vo
                )
            )

            self.anchor_position_b0 = (
                None
                if p_b0 is None
                else p_b0.copy()
            )

            self.last_delta_h_m = 0.0

            self.last_height_b0z_m = (
                self.last_height_m
            )

            self.reanchor_count += 1

            return self._make_snapshot(
                valid=True,
                reason=(
                    "REANCHOR_SESSION"
                    if changed_session
                    else "REANCHOR_EPOCH"
                ),
                vo=vo,
                reanchored=True,
            )

        if (
            self.anchor_pos_v
            is None
            or self.up_v is None
        ):
            return self._make_snapshot(
                valid=False,
                reason=
                    "INTERNAL_NOT_ANCHORED",
                vo=vo,
            )

        p_v = (
            T[
                :3,
                3,
            ]
        )

        delta_p_v = (
            p_v
            - self.anchor_pos_v
        )

        delta_h_m = float(
            np.dot(
                self.up_v,
                delta_p_v,
            )
        )

        height_m = float(
            self.anchor_height_m
            + delta_h_m
        )

        if not np.isfinite(
            height_m
        ):
            return self._make_snapshot(
                valid=False,
                reason=
                    "NONFINITE_HEIGHT",
                vo=vo,
            )

        self.last_delta_h_m = (
            delta_h_m
        )

        self.last_height_m = (
            height_m
        )

        p_b0 = (
            self._extract_position_b0(
                vo
            )
        )

        self.last_height_b0z_m = None

        if (
            p_b0 is not None
            and self.anchor_position_b0
            is not None
        ):
            self.last_height_b0z_m = float(
                self.anchor_height_m
                + (
                    p_b0[2]
                    - self.anchor_position_b0[
                        2
                    ]
                )
            )

        return self._make_snapshot(
            valid=True,
            reason="OK",
            vo=vo,
            reanchored=False,
        )

    def _make_snapshot(
        self,
        valid: bool,
        reason: str,
        vo: Optional[
            Dict[str, Any]
        ] = None,
        reanchored: bool = False,
    ) -> Dict[str, Any]:
        vo = (
            {}
            if vo is None
            else vo
        )

        return {
            "anchored": bool(
                self.anchored
            ),

            "valid": bool(
                valid
            ),

            "reason": str(
                reason
            ),

            "height_m": float(
                self.last_height_m
            ),

            "delta_h_m": float(
                self.last_delta_h_m
            ),

            "height_b0z_m": (
                None
                if self.last_height_b0z_m
                is None
                else float(
                    self.last_height_b0z_m
                )
            ),

            "nominal_height_m": float(
                self.nominal_height_m
            ),

            "anchor_height_m": float(
                self.anchor_height_m
            ),

            "up_V": (
                None
                if self.up_v is None
                else (
                    self.up_v
                    .astype(
                        np.float64
                    )
                    .tolist()
                )
            ),

            "session_id": (
                self.session_id
            ),

            "epoch": (
                self.epoch
            ),

            "reanchored": bool(
                reanchored
            ),

            "reanchor_count": int(
                self.reanchor_count
            ),

            "vo_valid": bool(
                vo.get(
                    "valid",
                    False,
                )
            ),

            "vo_reason": str(
                vo.get(
                    "reason",
                    "NO_VO_STATE",
                )
            ),

            "vo_age_ms": (
                vo.get(
                    "age_ms"
                )
            ),

            "vo_seq": (
                vo.get(
                    "seq"
                )
            ),

            "vo_session_id": (
                vo.get(
                    "session_id"
                )
            ),

            "vo_epoch": (
                vo.get(
                    "epoch"
                )
            ),

            "vo_transport_ms": (
                vo.get(
                    "transport_ms"
                )
            ),
        }


class RearVOUdpReceiver:
    """
    Backward-compatible name.

    VO API:
        get_latest_snapshot()
        lookup_T_V_FROM_B()

    Object API:
        get_latest_object_snapshot()

    VO Base-height API:
        anchor_base_height()
        get_base_height_snapshot()
        reset_base_height_estimator()
    """

    def __init__(
        self,
        developer_host: str,
        data_bind_host: str =
            "0.0.0.0",
        data_port: int =
            DEFAULT_DATA_PORT,
        sync_port: int =
            DEFAULT_SYNC_PORT,
        sync_period_s: float =
            DEFAULT_SYNC_PERIOD_S,
        history_seconds: float =
            DEFAULT_HISTORY_SECONDS,
    ):
        self.developer_host = (
            developer_host
        )

        self.data_bind_host = (
            data_bind_host
        )

        self.data_port = int(
            data_port
        )

        self.sync_port = int(
            sync_port
        )

        self.sync_period_s = float(
            sync_period_s
        )

        self.history_seconds = float(
            history_seconds
        )

        self.clock = (
            ClockOffsetEstimator()
        )

        self._stop_event = (
            threading.Event()
        )

        self._data_thread: Optional[
            threading.Thread
        ] = None

        self._sync_thread: Optional[
            threading.Thread
        ] = None

        self._data_socket: Optional[
            socket.socket
        ] = None

        self._sync_socket: Optional[
            socket.socket
        ] = None

        self._lock = (
            threading.Lock()
        )

        self._latest_vo: Optional[
            Dict[str, Any]
        ] = None

        self._latest_object: Optional[
            Dict[str, Any]
        ] = None

        self._history: Deque[
            Dict[str, Any]
        ] = deque()

        self._session_id: Optional[
            str
        ] = None

        self.rx_packets = 0
        self.vo_rx_packets = 0
        self.object_rx_packets = 0

        self.bad_packets = 0
        self.out_of_order = 0
        self.vo_out_of_order = 0
        self.object_out_of_order = 0
        self.session_resets = 0

        self.last_vo_seq: Optional[
            int
        ] = None

        self.last_object_seq: Optional[
            int
        ] = None

        # --------------------------------------------------------------
        # VO-derived Base-height estimator.
        #
        # Separate lock avoids coupling its state machine to the UDP packet
        # receiver lock.  The estimator is updated from nonblocking public reads.
        # --------------------------------------------------------------
        self._base_height_lock = (
            threading.Lock()
        )

        self._base_height_estimator = (
            VOBaseHeightEstimator(
                nominal_height_m=
                    DEFAULT_BASE_HEIGHT_M
            )
        )

        self._ping_seq = 0

    def start(
        self,
    ) -> None:
        if (
            self._data_thread
            is not None
            or self._sync_thread
            is not None
        ):
            return

        data_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        data_sock.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1,
        )

        data_sock.bind(
            (
                self.data_bind_host,
                self.data_port,
            )
        )

        data_sock.settimeout(
            0.2
        )

        sync_sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        # Ephemeral CRL source port.
        sync_sock.bind(
            (
                "0.0.0.0",
                0,
            )
        )

        sync_sock.settimeout(
            0.2
        )

        self._data_socket = (
            data_sock
        )

        self._sync_socket = (
            sync_sock
        )

        self._stop_event.clear()

        self._data_thread = (
            threading.Thread(
                target=
                    self._data_loop,
                name=
                    "perception-udp-data",
                daemon=True,
            )
        )

        self._sync_thread = (
            threading.Thread(
                target=
                    self._sync_loop,
                name=
                    "perception-clock-sync",
                daemon=True,
            )
        )

        self._data_thread.start()
        self._sync_thread.start()

    def stop(
        self,
    ) -> None:
        self._stop_event.set()

        for thread in (
            self._data_thread,
            self._sync_thread,
        ):
            if thread is not None:
                thread.join(
                    timeout=2.0
                )

        self._data_thread = None
        self._sync_thread = None

        for sock in (
            self._data_socket,
            self._sync_socket,
        ):
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

        self._data_socket = None
        self._sync_socket = None

    def get_clock_snapshot(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(
                self.clock.snapshot()
            )

    def reset_base_height_estimator(
        self,
        nominal_height_m: Optional[
            float
        ] = None,
    ) -> Dict[str, Any]:
        """
        Reset the VO Base-height state machine.

        This does NOT reset/restart VO itself.

        Typical deployment:
            reset once before establishing a new policy-start anchor.
        """
        with self._base_height_lock:
            self._base_height_estimator.reset(
                nominal_height_m=
                    nominal_height_m
            )

            return copy.deepcopy(
                self._base_height_estimator
                ._make_snapshot(
                    valid=False,
                    reason="RESET",
                    vo=None,
                )
            )

    def anchor_base_height(
        self,
        projected_gravity_b: Any,
        height_m: float =
            DEFAULT_BASE_HEIGHT_M,
        max_age_ms: float =
            DEFAULT_MAX_AGE_MS,
    ) -> Dict[str, Any]:
        """
        Explicitly anchor physical Base height to the CURRENT fresh VO pose.

        Parameters
        ----------
        projected_gravity_b:
            B2 IMU gravity direction expressed in Base frame.
            Same quantity used by the hierarchical policy
            (approximately [0,0,-1] when level).

        height_m:
            Known Base-origin height at this instant.
            Deployment default is 0.6017 m, but the caller should choose the
            physical anchor time explicitly (normally after B2W reaches DEFAULT
            and immediately before hierarchical policy state/history init).

        max_age_ms:
            Freshness requirement for the anchor VO sample.

        Returns
        -------
        Snapshot dict with valid/reason/height_m/up_V/session/epoch.
        """
        vo = self.get_latest_snapshot(
            max_age_ms=
                max_age_ms
        )

        if vo is None:
            with self._base_height_lock:
                return copy.deepcopy(
                    self._base_height_estimator
                    ._make_snapshot(
                        valid=False,
                        reason=
                            "ANCHOR_NO_VO_STATE",
                        vo=None,
                    )
                )

        with self._base_height_lock:
            snap = (
                self._base_height_estimator
                .anchor(
                    vo=vo,
                    projected_gravity_b=
                        projected_gravity_b,
                    height_m=
                        height_m,
                )
            )

            return copy.deepcopy(
                snap
            )

    def get_base_height_snapshot(
        self,
        projected_gravity_b: Any =
            None,
        max_age_ms: float =
            DEFAULT_MAX_AGE_MS,
    ) -> Dict[str, Any]:
        """
        Nonblocking VO-derived Base-height read/update.

        Same VO session/epoch:
            projected_gravity_b is not required; height uses the frozen up_V
            established at the physical anchor.

        New developer session / VO epoch:
            projected_gravity_b is required ONCE to establish the new frame's
            up_V while preserving last physical height continuously.

        VO invalid/stale:
            returns valid=False and freezes the last valid height_m.

        The method never falls back to 0.6017 during runtime after anchoring.
        """
        vo = self.get_latest_snapshot(
            max_age_ms=
                max_age_ms
        )

        if vo is None:
            vo_for_update = {
                "valid": False,
                "reason":
                    "NO_VO_STATE",
                "age_ms": None,
            }
        else:
            vo_for_update = vo

        with self._base_height_lock:
            snap = (
                self._base_height_estimator
                .update(
                    vo=
                        vo_for_update,
                    projected_gravity_b=
                        projected_gravity_b,
                )
            )

            return copy.deepcopy(
                snap
            )

    def get_latest_snapshot(
        self,
        max_age_ms: float =
            DEFAULT_MAX_AGE_MS,
    ) -> Optional[Dict[str, Any]]:
        """
        Backward-compatible VO latest-value read.
        """
        with self._lock:
            if self._latest_vo is None:
                return None

            snap = copy.deepcopy(
                self._latest_vo
            )

        capture_crl_ns = snap.get(
            "capture_crl_monotonic_ns"
        )

        if capture_crl_ns is None:
            snap["valid"] = False
            snap["reason"] = (
                "CLOCK_NOT_READY"
            )

            snap["age_ms"] = None
            return snap

        age_ms = (
            time.monotonic_ns()
            - int(
                capture_crl_ns
            )
        ) / 1e6

        snap["age_ms"] = float(
            age_ms
        )

        if (
            age_ms
            > float(
                max_age_ms
            )
        ):
            snap["valid"] = False
            snap["reason"] = (
                "STALE"
            )

        return snap

    def get_latest_object_snapshot(
        self,
        max_age_ms: float =
            DEFAULT_OBJECT_MAX_AGE_MS,
    ) -> Optional[Dict[str, Any]]:
        """
        Latest fused object state from the developer PC.

        Staleness is gated by CRL receive age, so the already-fused object
        channel does not depend on clock synchronization being ready.
        Converted developer timestamps are still exposed when clock sync exists.
        """
        with self._lock:
            if (
                self._latest_object
                is None
            ):
                return None

            snap = copy.deepcopy(
                self._latest_object
            )

        now_ns = (
            time.monotonic_ns()
        )

        recv_ns = int(
            snap[
                "recv_crl_monotonic_ns"
            ]
        )

        rx_age_ms = (
            now_ns
            - recv_ns
        ) / 1e6

        snap[
            "age_ms"
        ] = float(
            rx_age_ms
        )

        state_crl_ns = snap.get(
            "state_crl_monotonic_ns"
        )

        if state_crl_ns is None:
            snap[
                "source_age_ms"
            ] = None
        else:
            snap[
                "source_age_ms"
            ] = float(
                (
                    now_ns
                    - int(
                        state_crl_ns
                    )
                )
                / 1e6
            )

        if (
            rx_age_ms
            > float(
                max_age_ms
            )
        ):
            snap["valid"] = False
            snap["reason"] = (
                "STALE"
            )

        return snap

    def lookup_T_V_FROM_B(
        self,
        crl_monotonic_ns: int,
        epoch: Optional[int] = None,
        max_gap_ms: float =
            DEFAULT_LOOKUP_MAX_GAP_MS,
    ) -> Optional[np.ndarray]:
        """
        Lookup/interpolate T_V_FROM_B at a CRL-clock timestamp.
        """
        target = int(
            crl_monotonic_ns
        )

        with self._lock:
            hist = [
                {
                    "session_id":
                        x[
                            "session_id"
                        ],

                    "epoch":
                        int(
                            x[
                                "epoch"
                            ]
                        ),

                    "capture_crl_monotonic_ns":
                        int(
                            x[
                                "capture_crl_monotonic_ns"
                            ]
                        ),

                    "T_V_FROM_B":
                        np.asarray(
                            x[
                                "T_V_FROM_B"
                            ],
                            dtype=np.float64,
                        ).reshape(
                            4,
                            4,
                        ).copy(),
                }
                for x
                in self._history
            ]

            active_session = (
                self._session_id
            )

        hist = [
            x
            for x in hist
            if x[
                "session_id"
            ]
            == active_session
        ]

        if epoch is not None:
            hist = [
                x
                for x in hist
                if x[
                    "epoch"
                ]
                == int(
                    epoch
                )
            ]

        if not hist:
            return None

        hist.sort(
            key=lambda x:
                x[
                    "capture_crl_monotonic_ns"
                ]
        )

        first_t = hist[0][
            "capture_crl_monotonic_ns"
        ]

        last_t = hist[-1][
            "capture_crl_monotonic_ns"
        ]

        if target <= first_t:
            if (
                first_t
                - target
            ) / 1e6 > max_gap_ms:
                return None

            return hist[0][
                "T_V_FROM_B"
            ].copy()

        if target >= last_t:
            if (
                target
                - last_t
            ) / 1e6 > max_gap_ms:
                return None

            return hist[-1][
                "T_V_FROM_B"
            ].copy()

        for a, b in zip(
            hist[:-1],
            hist[1:],
        ):
            ta = a[
                "capture_crl_monotonic_ns"
            ]

            tb = b[
                "capture_crl_monotonic_ns"
            ]

            if not (
                ta
                <= target
                <= tb
            ):
                continue

            if (
                a["epoch"]
                != b["epoch"]
            ):
                return None

            span_ns = (
                tb
                - ta
            )

            if span_ns <= 0:
                return a[
                    "T_V_FROM_B"
                ].copy()

            if (
                span_ns
                / 1e6
                > 2.0
                * max_gap_ms
            ):
                return None

            alpha = (
                target
                - ta
            ) / span_ns

            Ta = a[
                "T_V_FROM_B"
            ]

            Tb = b[
                "T_V_FROM_B"
            ]

            t = (
                (1.0 - alpha)
                * Ta[:3, 3]
                + alpha
                * Tb[:3, 3]
            )

            qa = (
                quat_wxyz_from_rotmat(
                    Ta[
                        :3,
                        :3,
                    ]
                )
            )

            qb = (
                quat_wxyz_from_rotmat(
                    Tb[
                        :3,
                        :3,
                    ]
                )
            )

            q = slerp_wxyz(
                qa,
                qb,
                float(
                    alpha
                ),
            )

            T = np.eye(
                4,
                dtype=np.float64,
            )

            T[:3, :3] = (
                rotmat_from_quat_wxyz(
                    q
                )
            )

            T[:3, 3] = t

            return T

        return None

    def _clear_for_new_session(
        self,
        session_id: str,
    ) -> None:
        self._session_id = (
            session_id
        )

        self.last_vo_seq = None
        self.last_object_seq = None

        self._latest_vo = None
        self._latest_object = None

        self._history.clear()

        self.session_resets += 1

    def _convert_dev_ns_locked(
        self,
        value: Any,
    ) -> Optional[int]:
        if value is None:
            return None

        try:
            return (
                self.clock
                .dev_to_crl_ns(
                    int(
                        value
                    )
                )
            )

        except Exception:
            return None

    def _accept_session_locked(
        self,
        session_id: str,
    ) -> None:
        if (
            self._session_id
            != session_id
        ):
            self._clear_for_new_session(
                session_id
            )

    def _handle_vo_packet_locked(
        self,
        msg: Dict[str, Any],
        recv_crl_ns: int,
    ) -> None:
        session_id = str(
            msg.get(
                "session_id",
                "",
            )
        )

        self._accept_session_locked(
            session_id
        )

        seq = int(
            msg.get(
                "seq",
                -1,
            )
        )

        if (
            self.last_vo_seq
            is not None
            and seq
            <= self.last_vo_seq
        ):
            self.vo_out_of_order += 1
            self.out_of_order += 1
            return

        self.last_vo_seq = seq

        self.rx_packets += 1
        self.vo_rx_packets += 1

        capture_dev_ns = msg.get(
            "capture_dev_monotonic_ns"
        )

        sent_dev_ns = msg.get(
            "sent_dev_monotonic_ns"
        )

        capture_crl_ns = (
            self._convert_dev_ns_locked(
                capture_dev_ns
            )
        )

        sent_crl_ns = (
            self._convert_dev_ns_locked(
                sent_dev_ns
            )
        )

        transport_ms = (
            None
            if sent_crl_ns is None
            else (
                recv_crl_ns
                - sent_crl_ns
            ) / 1e6
        )

        sample = {
            "session_id":
                session_id,

            "seq":
                seq,

            "epoch": int(
                msg.get(
                    "epoch",
                    -1,
                )
            ),

            "valid": bool(
                msg.get(
                    "valid",
                    False,
                )
            ),

            "reason": str(
                msg.get(
                    "reason",
                    "UNKNOWN",
                )
            ),

            "capture_dev_monotonic_ns":
                capture_dev_ns,

            "sent_dev_monotonic_ns":
                sent_dev_ns,

            "capture_crl_monotonic_ns":
                capture_crl_ns,

            "sent_crl_monotonic_ns":
                sent_crl_ns,

            "recv_crl_monotonic_ns":
                int(
                    recv_crl_ns
                ),

            "transport_ms":
                transport_ms,

            "rs_timestamp_ms":
                msg.get(
                    "rs_timestamp_ms"
                ),

            "T_V_FROM_B":
                msg.get(
                    "T_V_FROM_B"
                ),

            "position_B0_m":
                msg.get(
                    "position_B0_m"
                ),

            "yaw_B0_deg":
                msg.get(
                    "yaw_B0_deg"
                ),

            "stereo_points":
                msg.get(
                    "stereo_points"
                ),

            "temporal_tracks":
                msg.get(
                    "temporal_tracks"
                ),

            "pnp_inliers":
                msg.get(
                    "pnp_inliers"
                ),

            "pnp_inlier_ratio":
                msg.get(
                    "pnp_inlier_ratio"
                ),

            "median_reprojection_px":
                msg.get(
                    "median_reprojection_px"
                ),

            "consecutive_failures":
                msg.get(
                    "consecutive_failures"
                ),

            "clock":
                self.clock.snapshot(),
        }

        self._latest_vo = (
            sample
        )

        if (
            sample["valid"]
            and sample[
                "T_V_FROM_B"
            ]
            is not None
            and capture_crl_ns
            is not None
            and self.clock.ready
        ):
            self._history.append(
                copy.deepcopy(
                    sample
                )
            )

            cutoff = (
                int(
                    capture_crl_ns
                )
                - int(
                    self.history_seconds
                    * 1e9
                )
            )

            while (
                self._history
                and int(
                    self._history[0][
                        "capture_crl_monotonic_ns"
                    ]
                )
                < cutoff
            ):
                self._history.popleft()

    def _handle_object_packet_locked(
        self,
        msg: Dict[str, Any],
        recv_crl_ns: int,
    ) -> None:
        session_id = str(
            msg.get(
                "session_id",
                "",
            )
        )

        self._accept_session_locked(
            session_id
        )

        seq = int(
            msg.get(
                "seq",
                -1,
            )
        )

        if (
            self.last_object_seq
            is not None
            and seq
            <= self.last_object_seq
        ):
            self.object_out_of_order += 1
            self.out_of_order += 1
            return

        self.last_object_seq = (
            seq
        )

        self.rx_packets += 1
        self.object_rx_packets += 1

        state_dev_ns = msg.get(
            "state_dev_monotonic_ns"
        )

        tracker_dev_ns = msg.get(
            "tracker_dev_monotonic_ns"
        )

        sent_dev_ns = msg.get(
            "sent_dev_monotonic_ns"
        )

        state_crl_ns = (
            self._convert_dev_ns_locked(
                state_dev_ns
            )
        )

        tracker_crl_ns = (
            self._convert_dev_ns_locked(
                tracker_dev_ns
            )
        )

        sent_crl_ns = (
            self._convert_dev_ns_locked(
                sent_dev_ns
            )
        )

        transport_ms = (
            None
            if sent_crl_ns is None
            else (
                recv_crl_ns
                - sent_crl_ns
            ) / 1e6
        )

        sample = {
            "session_id":
                session_id,

            "seq":
                seq,

            "valid": bool(
                msg.get(
                    "valid",
                    False,
                )
            ),

            "source":
                msg.get(
                    "source"
                ),

            "reason": str(
                msg.get(
                    "reason",
                    "UNKNOWN",
                )
            ),

            "position_base":
                msg.get(
                    "position_base"
                ),

            "p_base":
                msg.get(
                    "p_base"
                ),

            "state_dev_monotonic_ns":
                state_dev_ns,

            "tracker_dev_monotonic_ns":
                tracker_dev_ns,

            "sent_dev_monotonic_ns":
                sent_dev_ns,

            "state_crl_monotonic_ns":
                state_crl_ns,

            "tracker_crl_monotonic_ns":
                tracker_crl_ns,

            "sent_crl_monotonic_ns":
                sent_crl_ns,

            "recv_crl_monotonic_ns":
                int(
                    recv_crl_ns
                ),

            "transport_ms":
                transport_ms,

            "anchor_valid": bool(
                msg.get(
                    "anchor_valid",
                    False,
                )
            ),

            "anchor_epoch":
                msg.get(
                    "anchor_epoch"
                ),

            "anchor_count":
                msg.get(
                    "anchor_count"
                ),

            "anchor_visual_sequence":
                msg.get(
                    "anchor_visual_sequence"
                ),

            "anchor_visual_host_monotonic_ns":
                msg.get(
                    "anchor_visual_host_monotonic_ns"
                ),

            "anchor_point_v":
                msg.get(
                    "anchor_point_v"
                ),

            "visual_valid": bool(
                msg.get(
                    "visual_valid",
                    False,
                )
            ),

            "visual_reason":
                msg.get(
                    "visual_reason"
                ),

            "visual_sequence":
                msg.get(
                    "visual_sequence"
                ),

            "visual_age_ms":
                msg.get(
                    "visual_age_ms"
                ),

            "visual_host_monotonic_ns":
                msg.get(
                    "visual_host_monotonic_ns"
                ),

            "visual_area_px":
                msg.get(
                    "visual_area_px"
                ),

            "visual_depth_valid_pixels":
                msg.get(
                    "visual_depth_valid_pixels"
                ),

            "vo_valid": bool(
                msg.get(
                    "vo_valid",
                    False,
                )
            ),

            "vo_reason":
                msg.get(
                    "vo_reason"
                ),

            "vo_epoch":
                msg.get(
                    "vo_epoch"
                ),

            "vo_age_ms":
                msg.get(
                    "vo_age_ms"
                ),

            "vo_host_monotonic_ns":
                msg.get(
                    "vo_host_monotonic_ns"
                ),

            "clock":
                self.clock.snapshot(),
        }

        self._latest_object = (
            sample
        )

    def _data_loop(
        self,
    ) -> None:
        assert (
            self._data_socket
            is not None
        )

        while not self._stop_event.is_set():
            try:
                raw, peer = (
                    self._data_socket
                    .recvfrom(
                        65535
                    )
                )

            except socket.timeout:
                continue

            except OSError:
                break

            recv_crl_ns = (
                time.monotonic_ns()
            )

            try:
                msg = json.loads(
                    raw.decode(
                        "utf-8"
                    )
                )

            except Exception:
                self.bad_packets += 1
                continue

            if (
                msg.get(
                    "magic"
                )
                != MAGIC
                or int(
                    msg.get(
                        "version",
                        -1,
                    )
                )
                != PROTOCOL_VERSION
            ):
                self.bad_packets += 1
                continue

            packet_type = (
                msg.get(
                    "type"
                )
            )

            with self._lock:
                try:
                    if (
                        packet_type
                        == "VO_STATE"
                    ):
                        self._handle_vo_packet_locked(
                            msg,
                            recv_crl_ns,
                        )

                    elif (
                        packet_type
                        == "OBJECT_STATE"
                    ):
                        self._handle_object_packet_locked(
                            msg,
                            recv_crl_ns,
                        )

                    else:
                        self.bad_packets += 1

                except Exception:
                    self.bad_packets += 1

    def _sync_loop(
        self,
    ) -> None:
        assert (
            self._sync_socket
            is not None
        )

        target = (
            self.developer_host,
            self.sync_port,
        )

        while not self._stop_event.is_set():
            self._ping_seq += 1

            ping_seq = int(
                self._ping_seq
            )

            t0_crl_ns = (
                time.monotonic_ns()
            )

            ping = {
                "magic": MAGIC,
                "version":
                    PROTOCOL_VERSION,
                "type": "CLOCK_PING",
                "ping_seq":
                    ping_seq,
                "t0_crl_ns":
                    int(
                        t0_crl_ns
                    ),
            }

            raw = json.dumps(
                ping,
                separators=(",", ":"),
            ).encode("utf-8")

            try:
                self._sync_socket.sendto(
                    raw,
                    target,
                )

                pong_raw, peer = (
                    self._sync_socket
                    .recvfrom(
                        4096
                    )
                )

                t3_crl_ns = (
                    time.monotonic_ns()
                )

                pong = json.loads(
                    pong_raw.decode(
                        "utf-8"
                    )
                )

                if (
                    pong.get(
                        "magic"
                    )
                    != MAGIC
                    or int(
                        pong.get(
                            "version",
                            -1,
                        )
                    )
                    != PROTOCOL_VERSION
                    or pong.get(
                        "type"
                    )
                    != "CLOCK_PONG"
                    or int(
                        pong.get(
                            "ping_seq",
                            -1,
                        )
                    )
                    != ping_seq
                ):
                    raise RuntimeError(
                        "bad clock pong"
                    )

                with self._lock:
                    self.clock.add_sample(
                        t0_crl_ns=
                            int(
                                pong[
                                    "t0_crl_ns"
                                ]
                            ),

                        t1_dev_ns=
                            int(
                                pong[
                                    "t1_dev_ns"
                                ]
                            ),

                        t2_dev_ns=
                            int(
                                pong[
                                    "t2_dev_ns"
                                ]
                            ),

                        t3_crl_ns=
                            int(
                                t3_crl_ns
                            ),
                    )

            except (
                socket.timeout,
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
            ):
                pass

            self._stop_event.wait(
                self.sync_period_s
            )


def _fmt_num(
    value: Any,
    digits: int = 1,
) -> str:
    if value is None:
        return "None"

    try:
        return (
            f"{float(value):.{digits}f}"
        )

    except Exception:
        return str(
            value
        )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--developer-host",
        default=
            "192.168.123.164",
        help=(
            "Developer PC address."
        ),
    )

    parser.add_argument(
        "--data-bind-host",
        default="0.0.0.0",
    )

    parser.add_argument(
        "--data-port",
        type=int,
        default=
            DEFAULT_DATA_PORT,
    )

    parser.add_argument(
        "--sync-port",
        type=int,
        default=
            DEFAULT_SYNC_PORT,
    )

    parser.add_argument(
        "--print-period",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-age-ms",
        type=float,
        default=
            DEFAULT_MAX_AGE_MS,
        help=(
            "VO stale threshold."
        ),
    )

    parser.add_argument(
        "--object-max-age-ms",
        type=float,
        default=
            DEFAULT_OBJECT_MAX_AGE_MS,
    )

    args = parser.parse_args()

    receiver = RearVOUdpReceiver(
        developer_host=
            args.developer_host,
        data_bind_host=
            args.data_bind_host,
        data_port=
            args.data_port,
        sync_port=
            args.sync_port,
    )

    print("=" * 116)
    print(
        "CRL INTEGRATED VO + OBJECT UDP RECEIVER"
    )
    print("=" * 116)
    print(
        "developer sync target :",
        f"{args.developer_host}:"
        f"{args.sync_port}",
    )
    print(
        "data bind             :",
        f"{args.data_bind_host}:"
        f"{args.data_port}",
    )
    print(
        "packet types          :",
        "VO_STATE + OBJECT_STATE",
    )
    print(
        "VO timestamps         :",
        "converted to CRL monotonic domain",
    )
    print(
        "object stale gate     :",
        "CRL receive age; independent of clock readiness",
    )
    print(
        "controller behavior   :",
        "nonblocking latest/history only",
    )
    print("=" * 116)

    receiver.start()

    last_print = 0.0

    try:
        while True:
            now = (
                time.monotonic()
            )

            if (
                now
                - last_print
                < args.print_period
            ):
                time.sleep(
                    0.01
                )
                continue

            last_print = now

            clock = (
                receiver
                .get_clock_snapshot()
            )

            vo = (
                receiver
                .get_latest_snapshot(
                    max_age_ms=
                        args.max_age_ms
                )
            )

            obj = (
                receiver
                .get_latest_object_snapshot(
                    max_age_ms=
                        args.object_max_age_ms
                )
            )

            if vo is None:
                vo_txt = (
                    "VO=WAIT"
                )
            else:
                vo_txt = (
                    "VO="
                    f"seq={vo['seq']} "
                    f"e={vo['epoch']} "
                    f"v={int(bool(vo['valid']))} "
                    f"{vo['reason']} "
                    f"age={_fmt_num(vo.get('age_ms'))}ms"
                )

            if obj is None:
                obj_txt = (
                    "OBJ=WAIT"
                )
            elif obj.get(
                "valid",
                False,
            ):
                p = obj.get(
                    "position_base"
                )

                if p is None:
                    ptxt = "None"
                else:
                    ptxt = (
                        f"[{p[0]:+.3f},"
                        f"{p[1]:+.3f},"
                        f"{p[2]:+.3f}]"
                    )

                obj_txt = (
                    "OBJ="
                    f"{obj.get('source')} "
                    f"p_B={ptxt} "
                    f"a={int(bool(obj.get('anchor_valid', False)))} "
                    f"e={obj.get('anchor_epoch')} "
                    f"age={_fmt_num(obj.get('age_ms'))}ms"
                )
            else:
                obj_txt = (
                    "OBJ=INVALID:"
                    f"{obj.get('reason')} "
                    f"age={_fmt_num(obj.get('age_ms'))}ms"
                )

            print(
                "[RX] "
                f"{vo_txt} | "
                f"{obj_txt} | "
                f"clock="
                f"{int(bool(clock['ready']))} "
                f"rtt="
                f"{_fmt_num(clock['best_rtt_ms'], 3)}ms "
                f"spread="
                f"{_fmt_num(clock['offset_spread_ms'], 3)}ms "
                f"rx(vo/obj)="
                f"{receiver.vo_rx_packets}/"
                f"{receiver.object_rx_packets} "
                f"bad={receiver.bad_packets}"
            )

    except KeyboardInterrupt:
        print()
        print("Ctrl-C")

    finally:
        receiver.stop()

        print(
            "Stopped. "
            f"rx(vo/obj)="
            f"{receiver.vo_rx_packets}/"
            f"{receiver.object_rx_packets} "
            f"bad={receiver.bad_packets} "
            f"ooo(vo/obj)="
            f"{receiver.vo_out_of_order}/"
            f"{receiver.object_out_of_order} "
            f"session_resets="
            f"{receiver.session_resets}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())