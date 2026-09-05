#!/usr/bin/env python3
"""
Final CRL-side unified perception interface for B2W.

Architecture
============

Developer PC
------------
Front D435i
    -> FrontObjectPerception
    -> ObjectVOFallbackTracker
                         ^
                         |
Rear D435i -> ONE RearStereoVO
                         |
                         +-> VO_STATE UDP
Object tracker ------------> OBJECT_STATE UDP


CRL PC
------
B2 FRONT/BACK optical cameras
    -> B2AprilTagDetector
                         \
                          -> RetrievalVOFallbackTracker
RearVOUdpReceiver ---------/

RearVOUdpReceiver
    -> fused OBJECT_STATE latest value

This module exposes both channels as ONE nonblocking snapshot:

    obs = perception.get_latest_snapshot()

    obs["object"]["valid"]
    obs["object"]["source"]
    obs["object"]["position_base"]

    obs["retrieval"]["valid"]
    obs["retrieval"]["source"]
    obs["retrieval"]["retrieval_target_base"]
    obs["retrieval"]["T_base_tag"]

No RealSense waits, camera RPC, AprilTag detection, VO work, or socket I/O
occurs inside get_latest_snapshot(). Those operations live in background
producer threads owned by the existing detector/receiver/tracker modules.

Transform convention
====================
T_A_FROM_B maps B -> A:

    p_A = R_A_FROM_B @ p_B + t_A_FROM_B

Retrieval fallback anchors the COMPLETE PHYSICAL Tag pose:

    T_V_FROM_TAG =
        T_V_FROM_B(t0)
        @ T_B_FROM_TAG(t0)

and during visual dropout:

    T_B_FROM_TAG(t) =
        inv(T_V_FROM_B(t))
        @ T_V_FROM_TAG

Object fallback is already fused on the developer PC:

    p_object^V =
        T_V_FROM_B(t0)
        @ p_object^B(t0)

    p_object^B(t) =
        T_B_FROM_V(t)
        @ p_object^V

Safety semantics
================
- Object visual always overrides object VO fallback.
- AprilTag visual always overrides retrieval VO fallback.
- No valid anchor -> no fabricated fallback.
- VO epoch change invalidates old anchors.
- Developer perception-server session change invalidates old Tag anchors.
- Network staleness invalidates received object/VO state.
- Object VO fallback assumes the invisible object is stationary in the world.
"""

from __future__ import annotations

import argparse
import copy
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


DEFAULT_OBJECT_MAX_AGE_MS = 300.0
DEFAULT_RETRIEVAL_MAX_AGE_MS = 250.0
DEFAULT_VO_MAX_AGE_MS = 250.0
DEFAULT_VISUAL_MAX_AGE_S = 0.200

DEFAULT_TAG_ID = 0
DEFAULT_TAG_SIZE_M = 0.195

DEFAULT_RETRIEVAL_TARGET_TAG = np.array(
    [0.0, 0.0, 1.0],
    dtype=np.float64,
)


def _deepcopy_or_none(
    value: Any,
) -> Any:
    if value is None:
        return None
    return copy.deepcopy(value)


class B2WPerceptionAggregator:
    """
    Lightweight nonblocking aggregator.

    Required providers
    ------------------
    receiver:
        get_latest_object_snapshot(max_age_ms=...)
        get_latest_snapshot(max_age_ms=...)      # VO
        get_base_height_snapshot(projected_gravity_b=..., max_age_ms=...)
        get_clock_snapshot()

    retrieval_tracker:
        get_latest_snapshot(max_age_ms=...)

    This class starts no thread and owns no hardware. It only performs
    latest-value reads and dictionary composition.
    """

    def __init__(
        self,
        receiver: Any,
        retrieval_tracker: Any,
        object_max_age_ms: float =
            DEFAULT_OBJECT_MAX_AGE_MS,
        retrieval_max_age_ms: float =
            DEFAULT_RETRIEVAL_MAX_AGE_MS,
        vo_max_age_ms: float =
            DEFAULT_VO_MAX_AGE_MS,
    ):
        self.receiver = receiver
        self.retrieval_tracker = retrieval_tracker

        self.object_max_age_ms = float(
            object_max_age_ms
        )

        self.retrieval_max_age_ms = float(
            retrieval_max_age_ms
        )

        self.vo_max_age_ms = float(
            vo_max_age_ms
        )

    @staticmethod
    def _normalize_object(
        snap: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if snap is None:
            return {
                "valid": False,
                "source": None,
                "reason": "NO_OBJECT_STATE",
                "position_base": None,
                "age_ms": None,
                "source_age_ms": None,
                "anchor_valid": False,
                "anchor_epoch": None,
                "vo_epoch": None,
                "session_id": None,
            }

        return {
            "valid": bool(
                snap.get(
                    "valid",
                    False,
                )
            ),

            "source":
                snap.get(
                    "source"
                ),

            "reason": str(
                snap.get(
                    "reason",
                    "UNKNOWN",
                )
            ),

            "position_base":
                _deepcopy_or_none(
                    snap.get(
                        "position_base"
                    )
                ),

            # Compatibility alias.
            "p_base":
                _deepcopy_or_none(
                    snap.get(
                        "p_base"
                    )
                ),

            "age_ms":
                snap.get(
                    "age_ms"
                ),

            "source_age_ms":
                snap.get(
                    "source_age_ms"
                ),

            "session_id":
                snap.get(
                    "session_id"
                ),

            "anchor_valid": bool(
                snap.get(
                    "anchor_valid",
                    False,
                )
            ),

            "anchor_epoch":
                snap.get(
                    "anchor_epoch"
                ),

            "anchor_count":
                snap.get(
                    "anchor_count"
                ),

            "visual_valid": bool(
                snap.get(
                    "visual_valid",
                    False,
                )
            ),

            "visual_reason":
                snap.get(
                    "visual_reason"
                ),

            "visual_area_px":
                snap.get(
                    "visual_area_px"
                ),

            "visual_depth_valid_pixels":
                snap.get(
                    "visual_depth_valid_pixels"
                ),

            "vo_valid": bool(
                snap.get(
                    "vo_valid",
                    False,
                )
            ),

            "vo_reason":
                snap.get(
                    "vo_reason"
                ),

            "vo_epoch":
                snap.get(
                    "vo_epoch"
                ),

            "transport_ms":
                snap.get(
                    "transport_ms"
                ),

            "state_crl_monotonic_ns":
                snap.get(
                    "state_crl_monotonic_ns"
                ),

            "recv_crl_monotonic_ns":
                snap.get(
                    "recv_crl_monotonic_ns"
                ),

            "seq":
                snap.get(
                    "seq"
                ),
        }

    @staticmethod
    def _normalize_retrieval(
        snap: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if snap is None:
            return {
                "valid": False,
                "source": None,
                "reason": "NO_RETRIEVAL_STATE",
                "retrieval_target_base": None,
                "T_base_tag": None,
                "tag_center_base": None,
                "R_base_tag": None,
                "tag_z_base": None,
                "tracker_age_ms": None,
                "anchor_valid": False,
                "anchor_epoch": None,
                "anchor_session_id": None,
                "vo_epoch": None,
                "vo_session_id": None,
            }

        return {
            "valid": bool(
                snap.get(
                    "valid",
                    False,
                )
            ),

            "source":
                snap.get(
                    "source"
                ),

            "reason": str(
                snap.get(
                    "reason",
                    "UNKNOWN",
                )
            ),

            "retrieval_target_base":
                _deepcopy_or_none(
                    snap.get(
                        "retrieval_target_base"
                    )
                ),

            "T_base_tag":
                _deepcopy_or_none(
                    snap.get(
                        "T_base_tag"
                    )
                ),

            "tag_center_base":
                _deepcopy_or_none(
                    snap.get(
                        "tag_center_base"
                    )
                ),

            "R_base_tag":
                _deepcopy_or_none(
                    snap.get(
                        "R_base_tag"
                    )
                ),

            "tag_z_base":
                _deepcopy_or_none(
                    snap.get(
                        "tag_z_base"
                    )
                ),

            "tracker_age_ms":
                snap.get(
                    "tracker_age_ms"
                ),

            "state_crl_monotonic_ns":
                snap.get(
                    "state_crl_monotonic_ns"
                ),

            "anchor_valid": bool(
                snap.get(
                    "anchor_valid",
                    False,
                )
            ),

            "anchor_epoch":
                snap.get(
                    "anchor_epoch"
                ),

            "anchor_session_id":
                snap.get(
                    "anchor_session_id"
                ),

            "anchor_count":
                snap.get(
                    "anchor_count"
                ),

            "visual_valid": bool(
                snap.get(
                    "visual_valid",
                    False,
                )
            ),

            "visual_source":
                snap.get(
                    "visual_source"
                ),

            "visual_reason":
                snap.get(
                    "visual_reason"
                ),

            "visual_age_ms":
                snap.get(
                    "visual_age_ms"
                ),

            "front_valid": bool(
                snap.get(
                    "front_valid",
                    False,
                )
            ),

            "back_valid": bool(
                snap.get(
                    "back_valid",
                    False,
                )
            ),

            "reprojection_rmse_px":
                snap.get(
                    "reprojection_rmse_px"
                ),

            "vo_valid": bool(
                snap.get(
                    "vo_valid",
                    False,
                )
            ),

            "vo_reason":
                snap.get(
                    "vo_reason"
                ),

            "vo_epoch":
                snap.get(
                    "vo_epoch"
                ),

            "vo_session_id":
                snap.get(
                    "vo_session_id"
                ),

            "vo_age_ms":
                snap.get(
                    "vo_age_ms"
                ),

            "vo_transport_ms":
                snap.get(
                    "vo_transport_ms"
                ),
        }

    @staticmethod
    def _normalize_base_height(
        snap: Optional[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Normalize the VO-derived physical Base-height channel.

        IMPORTANT:
        Its validity is intentionally independent from object/retrieval validity.
        """
        if snap is None:
            return {
                "anchored": False,
                "valid": False,
                "reason":
                    "NO_BASE_HEIGHT_STATE",
                "height_m": None,
                "delta_h_m": None,
                "height_b0z_m": None,
                "anchor_height_m": None,
                "nominal_height_m": None,
                "up_V": None,
                "session_id": None,
                "epoch": None,
                "reanchor_count": 0,
                "vo_age_ms": None,
                "vo_seq": None,
            }

        return {
            "anchored": bool(
                snap.get(
                    "anchored",
                    False,
                )
            ),

            "valid": bool(
                snap.get(
                    "valid",
                    False,
                )
            ),

            "reason": str(
                snap.get(
                    "reason",
                    "UNKNOWN",
                )
            ),

            "height_m":
                snap.get(
                    "height_m"
                ),

            "delta_h_m":
                snap.get(
                    "delta_h_m"
                ),

            "height_b0z_m":
                snap.get(
                    "height_b0z_m"
                ),

            "anchor_height_m":
                snap.get(
                    "anchor_height_m"
                ),

            "nominal_height_m":
                snap.get(
                    "nominal_height_m"
                ),

            "up_V":
                _deepcopy_or_none(
                    snap.get(
                        "up_V"
                    )
                ),

            "session_id":
                snap.get(
                    "session_id"
                ),

            "epoch":
                snap.get(
                    "epoch"
                ),

            "reanchored": bool(
                snap.get(
                    "reanchored",
                    False,
                )
            ),

            "reanchor_count": int(
                snap.get(
                    "reanchor_count",
                    0,
                )
                or 0
            ),

            "vo_valid": bool(
                snap.get(
                    "vo_valid",
                    False,
                )
            ),

            "vo_reason":
                snap.get(
                    "vo_reason"
                ),

            "vo_age_ms":
                snap.get(
                    "vo_age_ms"
                ),

            "vo_seq":
                snap.get(
                    "vo_seq"
                ),

            "vo_session_id":
                snap.get(
                    "vo_session_id"
                ),

            "vo_epoch":
                snap.get(
                    "vo_epoch"
                ),

            "vo_transport_ms":
                snap.get(
                    "vo_transport_ms"
                ),
        }

    @staticmethod
    def _normalize_vo(
        snap: Optional[Dict[str, Any]],
        base_height_snap: Optional[
            Dict[str, Any]
        ] = None,
    ) -> Dict[str, Any]:
        if snap is None:
            return {
                "valid": False,
                "reason": "NO_VO_STATE",
                "session_id": None,
                "epoch": None,
                "age_ms": None,
                "T_V_FROM_B": None,
                "base_height":
                    B2WPerceptionAggregator
                    ._normalize_base_height(
                        base_height_snap
                    ),
            }

        return {
            "valid": bool(
                snap.get(
                    "valid",
                    False,
                )
            ),

            "reason": str(
                snap.get(
                    "reason",
                    "UNKNOWN",
                )
            ),

            "session_id":
                snap.get(
                    "session_id"
                ),

            "epoch":
                snap.get(
                    "epoch"
                ),

            "age_ms":
                snap.get(
                    "age_ms"
                ),

            "T_V_FROM_B":
                _deepcopy_or_none(
                    snap.get(
                        "T_V_FROM_B"
                    )
                ),

            "position_B0_m":
                _deepcopy_or_none(
                    snap.get(
                        "position_B0_m"
                    )
                ),

            "yaw_B0_deg":
                snap.get(
                    "yaw_B0_deg"
                ),

            "transport_ms":
                snap.get(
                    "transport_ms"
                ),

            "seq":
                snap.get(
                    "seq"
                ),

            "base_height":
                B2WPerceptionAggregator
                ._normalize_base_height(
                    base_height_snap
                ),
        }

    def get_latest_snapshot(
        self,
        projected_gravity_b: Any =
            None,
    ) -> Dict[str, Any]:
        """
        Fully nonblocking HL-facing read.

        "valid" means BOTH object and retrieval are valid at this instant.
        The two channels remain independently valid/invalid and should always
        be inspected individually by downstream code.
        """
        now_ns = time.monotonic_ns()

        object_raw = (
            self.receiver
            .get_latest_object_snapshot(
                max_age_ms=
                    self.object_max_age_ms
            )
        )

        retrieval_raw = (
            self.retrieval_tracker
            .get_latest_snapshot(
                max_age_ms=
                    self.retrieval_max_age_ms
            )
        )

        vo_raw = (
            self.receiver
            .get_latest_snapshot(
                max_age_ms=
                    self.vo_max_age_ms
            )
        )

        base_height_raw = (
            self.receiver
            .get_base_height_snapshot(
                projected_gravity_b=
                    projected_gravity_b,
                max_age_ms=
                    self.vo_max_age_ms,
            )
        )

        clock = (
            self.receiver
            .get_clock_snapshot()
        )

        object_state = (
            self._normalize_object(
                object_raw
            )
        )

        retrieval_state = (
            self._normalize_retrieval(
                retrieval_raw
            )
        )

        vo_state = (
            self._normalize_vo(
                vo_raw,
                base_height_snap=
                    base_height_raw,
            )
        )

        object_valid = bool(
            object_state[
                "valid"
            ]
        )

        retrieval_valid = bool(
            retrieval_state[
                "valid"
            ]
        )

        all_valid = (
            object_valid
            and retrieval_valid
        )

        invalid_channels = []

        if not object_valid:
            invalid_channels.append(
                "object"
            )

        if not retrieval_valid:
            invalid_channels.append(
                "retrieval"
            )

        return {
            "valid":
                bool(
                    all_valid
                ),

            "all_required_valid":
                bool(
                    all_valid
                ),

            "invalid_channels":
                invalid_channels,

            "snapshot_crl_monotonic_ns":
                int(
                    now_ns
                ),

            "object":
                object_state,

            "retrieval":
                retrieval_state,

            # VO is included for diagnostics/health and future consumers.
            # HL object/retrieval positions should use the fused channels above.
            "vo":
                vo_state,

            # Convenience alias.  This channel is NOT part of the existing
            # object && retrieval perception-valid gate.
            "base_height":
                copy.deepcopy(
                    vo_state[
                        "base_height"
                    ]
                ),

            "clock": {
                "ready": bool(
                    clock.get(
                        "ready",
                        False,
                    )
                ),

                "samples":
                    clock.get(
                        "samples"
                    ),

                "best_rtt_ms":
                    clock.get(
                        "best_rtt_ms"
                    ),

                "offset_spread_ms":
                    clock.get(
                        "offset_spread_ms"
                    ),

                "offset_dev_minus_crl_ms":
                    clock.get(
                        "offset_dev_minus_crl_ms"
                    ),
            },
        }


class CRLB2WPerceptionSystem:
    """
    Production CRL owner/wrapper.

    Owns exactly one instance of:
        B2AprilTagDetector
        RearVOUdpReceiver
        RetrievalVOFallbackTracker
        B2WPerceptionAggregator

    It does NOT own either RealSense camera; they live on the developer PC.
    """

    def __init__(
        self,
        interface: str,
        repo_root: Path,
        developer_host: str =
            "192.168.123.164",
        data_port: int = 50020,
        sync_port: int = 50021,
        tag_id: int =
            DEFAULT_TAG_ID,
        tag_size_m: float =
            DEFAULT_TAG_SIZE_M,
        retrieval_target_tag: np.ndarray =
            DEFAULT_RETRIEVAL_TARGET_TAG,
        visual_max_age_s: float =
            DEFAULT_VISUAL_MAX_AGE_S,
        object_max_age_ms: float =
            DEFAULT_OBJECT_MAX_AGE_MS,
        retrieval_max_age_ms: float =
            DEFAULT_RETRIEVAL_MAX_AGE_MS,
        vo_max_age_ms: float =
            DEFAULT_VO_MAX_AGE_MS,
        front_intrinsics: Optional[Path] =
            None,
        back_intrinsics: Optional[Path] =
            None,
        initialize_channel_factory: bool =
            True,
    ):
        self.interface = str(
            interface
        )

        self.repo_root = (
            Path(
                repo_root
            )
            .expanduser()
            .resolve()
        )

        self.developer_host = str(
            developer_host
        )

        self.data_port = int(
            data_port
        )

        self.sync_port = int(
            sync_port
        )

        self.tag_id = int(
            tag_id
        )

        self.tag_size_m = float(
            tag_size_m
        )

        self.retrieval_target_tag = (
            np.asarray(
                retrieval_target_tag,
                dtype=np.float64,
            ).reshape(3)
        )

        self.visual_max_age_s = float(
            visual_max_age_s
        )

        self.object_max_age_ms = float(
            object_max_age_ms
        )

        self.retrieval_max_age_ms = float(
            retrieval_max_age_ms
        )

        self.vo_max_age_ms = float(
            vo_max_age_ms
        )

        self.front_intrinsics_override = (
            None
            if front_intrinsics is None
            else Path(
                front_intrinsics
            ).expanduser()
        )

        self.back_intrinsics_override = (
            None
            if back_intrinsics is None
            else Path(
                back_intrinsics
            ).expanduser()
        )

        self.initialize_channel_factory = bool(
            initialize_channel_factory
        )

        self.detector = None
        self.receiver = None
        self.retrieval_tracker = None
        self.aggregator = None

        self._initialized = False
        self._started = False

    def initialize(
        self,
    ) -> None:
        """
        Initialize Unitree channel factory (optionally) and B2 camera RPC clients.

        Call this from the main thread before start().
        """
        if self._initialized:
            return

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

        from retrieval_vo_fallback_tracker import (
            RetrievalVOFallbackTracker,
        )

        if self.initialize_channel_factory:
            ChannelFactoryInitialize(
                0,
                self.interface,
            )

        if (
            self.front_intrinsics_override
            is None
        ):
            front_intrinsics = (
                self.repo_root
                / "extrinsic_calib"
                / "front_rgb"
                / "front_intrinsics.npz"
            )
        else:
            front_intrinsics = (
                self.front_intrinsics_override
            )

            if (
                not front_intrinsics
                .is_absolute()
            ):
                front_intrinsics = (
                    self.repo_root
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
                self.repo_root,
                self.back_intrinsics_override,
            )
        )

        self.detector = (
            B2AprilTagDetector(
                repo_root=
                    self.repo_root,
                front_intrinsics=
                    front_intrinsics,
                back_intrinsics=
                    back_intrinsics,
                tag_id=
                    self.tag_id,
                tag_size_m=
                    self.tag_size_m,
                retrieval_target_tag=
                    self.retrieval_target_tag,
                visual_max_age_s=
                    self.visual_max_age_s,
            )
        )

        # RPC clients are intentionally initialized in this/main thread.
        self.detector.initialize_clients()

        self.receiver = (
            RearVOUdpReceiver(
                developer_host=
                    self.developer_host,
                data_port=
                    self.data_port,
                sync_port=
                    self.sync_port,
            )
        )

        self.retrieval_tracker = (
            RetrievalVOFallbackTracker(
                tag_provider=
                    self.detector,
                vo_receiver=
                    self.receiver,
                retrieval_target_tag=
                    self.retrieval_target_tag,
                visual_max_age_s=
                    self.visual_max_age_s,
                vo_max_age_ms=
                    self.vo_max_age_ms,
            )
        )

        self.aggregator = (
            B2WPerceptionAggregator(
                receiver=
                    self.receiver,
                retrieval_tracker=
                    self.retrieval_tracker,
                object_max_age_ms=
                    self.object_max_age_ms,
                retrieval_max_age_ms=
                    self.retrieval_max_age_ms,
                vo_max_age_ms=
                    self.vo_max_age_ms,
            )
        )

        self._initialized = True

    def start(
        self,
    ) -> None:
        if not self._initialized:
            raise RuntimeError(
                "Call initialize() before start()."
            )

        if self._started:
            return

        detector_started = False
        receiver_started = False
        retrieval_started = False

        try:
            # Start raw producers before starting the fusion tracker.
            self.detector.start()
            detector_started = True

            self.receiver.start()
            receiver_started = True

            self.retrieval_tracker.start()
            retrieval_started = True

            self._started = True

        except BaseException:
            if retrieval_started:
                self.retrieval_tracker.stop()

            if receiver_started:
                self.receiver.stop()

            if detector_started:
                self.detector.stop()

            raise

    def stop(
        self,
    ) -> None:
        # Stop downstream consumer before upstream producers.
        if self.retrieval_tracker is not None:
            try:
                self.retrieval_tracker.stop()
            except Exception:
                pass

        if self.receiver is not None:
            try:
                self.receiver.stop()
            except Exception:
                pass

        if self.detector is not None:
            try:
                self.detector.stop()
            except Exception:
                pass

        self._started = False

    def get_latest_snapshot(
        self,
        projected_gravity_b: Any =
            None,
    ) -> Dict[str, Any]:
        """
        Nonblocking unified HL-facing snapshot.

        projected_gravity_b is optional during an unchanged VO epoch.
        Pass the current B2 IMU projected gravity during runtime so a VO
        session/epoch reset can re-anchor Base height continuously.
        """
        if (
            not self._initialized
            or self.aggregator
            is None
        ):
            raise RuntimeError(
                "Perception system is not initialized."
            )

        return (
            self.aggregator
            .get_latest_snapshot(
                projected_gravity_b=
                    projected_gravity_b
            )
        )

    def anchor_vo_base_height(
        self,
        projected_gravity_b: Any,
        height_m: float =
            0.6017,
    ) -> Dict[str, Any]:
        """
        Anchor VO-derived Base height at the current fresh VO pose.

        Intended deployment timing:
            B2W reached DEFAULT
            -> perception ready / user start accepted
            -> anchor height
            -> initialize policy state/history
        """
        if (
            not self._initialized
            or self.receiver is None
        ):
            raise RuntimeError(
                "Perception system is not initialized."
            )

        return (
            self.receiver
            .anchor_base_height(
                projected_gravity_b=
                    projected_gravity_b,
                height_m=
                    float(
                        height_m
                    ),
                max_age_ms=
                    self.vo_max_age_ms,
            )
        )

    def get_vo_base_height_snapshot(
        self,
        projected_gravity_b: Any =
            None,
    ) -> Dict[str, Any]:
        if (
            not self._initialized
            or self.receiver is None
        ):
            raise RuntimeError(
                "Perception system is not initialized."
            )

        return (
            self.receiver
            .get_base_height_snapshot(
                projected_gravity_b=
                    projected_gravity_b,
                max_age_ms=
                    self.vo_max_age_ms,
            )
        )

    def reset_vo_base_height_estimator(
        self,
        nominal_height_m: Optional[
            float
        ] = None,
    ) -> Dict[str, Any]:
        if (
            not self._initialized
            or self.receiver is None
        ):
            raise RuntimeError(
                "Perception system is not initialized."
            )

        return (
            self.receiver
            .reset_base_height_estimator(
                nominal_height_m=
                    nominal_height_m
            )
        )

    def get_object_snapshot(
        self,
    ) -> Dict[str, Any]:
        return (
            self.get_latest_snapshot()
            ["object"]
        )

    def get_retrieval_snapshot(
        self,
    ) -> Dict[str, Any]:
        return (
            self.get_latest_snapshot()
            ["retrieval"]
        )


def resolve_repo_root(
    file_path: Path,
) -> Path:
    """
    Expected deployment:
        <repo>/example/b2w/camera/b2w_perception.py
    """
    p = file_path.resolve()

    # b2w_perception.py
    # camera/
    # b2w/
    # example/
    # repo/
    if len(
        p.parents
    ) >= 4:
        candidate = (
            p.parents[3]
        )

        if (
            candidate
            / "example"
            / "b2w"
        ).exists():
            return candidate

    cwd = (
        Path.cwd()
        .resolve()
    )

    if (
        cwd
        / "example"
        / "b2w"
    ).exists():
        return cwd

    raise RuntimeError(
        "Could not resolve Unitree SDK repository root."
    )


def _fmt_point(
    p: Any,
) -> str:
    if p is None:
        return "None"

    try:
        a = np.asarray(
            p,
            dtype=np.float64,
        ).reshape(3)

        return (
            f"[{a[0]:+.3f},"
            f"{a[1]:+.3f},"
            f"{a[2]:+.3f}]"
        )

    except Exception:
        return str(
            p
        )


def main(
) -> int:
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
        default=
            "192.168.123.164",
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
        default=
            DEFAULT_TAG_ID,
    )

    parser.add_argument(
        "--tag-size",
        type=float,
        default=
            DEFAULT_TAG_SIZE_M,
    )

    parser.add_argument(
        "--retrieval-z",
        type=float,
        default=1.0,
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
        "--object-max-age-ms",
        type=float,
        default=
            DEFAULT_OBJECT_MAX_AGE_MS,
    )

    parser.add_argument(
        "--retrieval-max-age-ms",
        type=float,
        default=
            DEFAULT_RETRIEVAL_MAX_AGE_MS,
    )

    parser.add_argument(
        "--vo-max-age-ms",
        type=float,
        default=
            DEFAULT_VO_MAX_AGE_MS,
    )

    parser.add_argument(
        "--visual-max-age",
        type=float,
        default=
            DEFAULT_VISUAL_MAX_AGE_S,
    )

    parser.add_argument(
        "--print-period",
        type=float,
        default=0.20,
    )

    args = (
        parser.parse_args()
    )

    repo_root = (
        resolve_repo_root(
            Path(
                __file__
            )
        )
    )

    retrieval_target_tag = (
        np.array(
            [
                0.0,
                0.0,
                float(
                    args.retrieval_z
                ),
            ],
            dtype=np.float64,
        )
    )

    perception = (
        CRLB2WPerceptionSystem(
            interface=
                args.interface,
            repo_root=
                repo_root,
            developer_host=
                args.developer_host,
            data_port=
                args.data_port,
            sync_port=
                args.sync_port,
            tag_id=
                args.tag_id,
            tag_size_m=
                args.tag_size,
            retrieval_target_tag=
                retrieval_target_tag,
            visual_max_age_s=
                args.visual_max_age,
            object_max_age_ms=
                args.object_max_age_ms,
            retrieval_max_age_ms=
                args.retrieval_max_age_ms,
            vo_max_age_ms=
                args.vo_max_age_ms,
            front_intrinsics=
                args.front_intrinsics,
            back_intrinsics=
                args.back_intrinsics,
            initialize_channel_factory=
                True,
        )
    )

    print("=" * 122)
    print(
        "B2W FINAL UNIFIED PERCEPTION"
    )
    print("=" * 122)
    print(
        "CRL interface      :",
        args.interface,
    )
    print(
        "Developer server   :",
        f"{args.developer_host}:"
        f"{args.data_port}",
    )
    print(
        "Object channel     :",
        "Front RGB-D visual / shared Rear VO fallback",
    )
    print(
        "Retrieval channel  :",
        "B2 FRONT/BACK AprilTag / Rear VO fallback",
    )
    print(
        "Retrieval Tag xyz  :",
        retrieval_target_tag,
    )
    print(
        "HL interface       :",
        "nonblocking get_latest_snapshot()",
    )
    print(
        "IMPORTANT          :",
        "do not run standalone rear_vo_udp_receiver.py or b2_apriltag_detector.py",
    )
    print("=" * 122)

    print(
        "[INIT] Unitree channel + B2 camera clients..."
    )

    perception.initialize()

    print(
        "[INIT] starting detector + receiver + retrieval tracker..."
    )

    perception.start()

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

            obs = (
                perception
                .get_latest_snapshot()
            )

            obj = (
                obs[
                    "object"
                ]
            )

            ret = (
                obs[
                    "retrieval"
                ]
            )

            vo = (
                obs[
                    "vo"
                ]
            )

            clock = (
                obs[
                    "clock"
                ]
            )

            print(
                "[PERCEPTION] "
                f"all={int(obs['valid'])} | "
                f"OBJ="
                f"{int(obj['valid'])}:"
                f"{obj['source']} "
                f"p_B={_fmt_point(obj['position_base'])} "
                f"a={int(obj['anchor_valid'])} "
                f"e={obj['anchor_epoch']} | "
                f"RET="
                f"{int(ret['valid'])}:"
                f"{ret['source']} "
                f"p_B={_fmt_point(ret['retrieval_target_base'])} "
                f"a={int(ret['anchor_valid'])} "
                f"e={ret['anchor_epoch']} | "
                f"VO="
                f"{int(vo['valid'])}:"
                f"{vo['reason']} "
                f"e={vo['epoch']} | "
                f"clock="
                f"{int(clock['ready'])}"
            )

    finally:
        perception.stop()

    print("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())