#!/usr/bin/env python3
"""
Production B2W dual-optical-camera AprilTag detector.

Purpose
-------
Use BOTH built-in B2 optical cameras:

    B2 Front optical
    B2 Back optical

to detect AprilTag 36h11 ID 0, estimate the PHYSICAL Tag pose in Base,
and construct the virtual retrieval target:

    p_Retrieval^Tag = [0, 0, +1.0] m

The detector is VISUAL ONLY. It intentionally contains no VO fallback.
VO will be layered on top later.

Transform convention
--------------------
T_A_FROM_B maps B -> A:

    p_A = R_A_FROM_B @ p_B + t_A_FROM_B

For each camera C:

    T_BASE_FROM_TAG =
        T_BASE_FROM_C @ T_C_FROM_TAG

Physical Tag center:

    p_tag_base = T_BASE_FROM_TAG[:3, 3]

Retrieval target:

    p_retrieval_base =
        R_BASE_FROM_TAG @ [0, 0, 1] + p_tag_base

Important semantics
-------------------
1. The PHYSICAL Tag center is the visual source of truth and future VO anchor.
2. The 1 m retrieval point is a derived HL target only.
3. No camera RPC or AprilTag detection is ever performed in a controller loop.
4. Front/Back camera RPC runs in producer threads.
5. AprilTag detection runs in a separate processor thread.
6. Readers use get_latest_snapshot(), which is nonblocking latest-value access.
7. No temporal median is applied in Base frame; that would lag while B2 moves.

Standalone:
    python3 example/b2w/camera/b2_apriltag_detector.py enxa0cec819e15f

Integration:
    ChannelFactoryInitialize(...) must already have been called.
    detector = B2AprilTagDetector(...)
    detector.initialize_clients()  # main thread
    detector.start()
    snap = detector.get_latest_snapshot()
    ...
    detector.stop()
"""

from __future__ import annotations

import argparse
import copy
import math
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.b2.front_video.front_video_client import FrontVideoClient
from unitree_sdk2py.b2.back_video.back_video_client import BackVideoClient


# =============================================================================
# Frozen AprilTag configuration
# =============================================================================

TAG_ID = 0
TAG_SIZE_M = 0.195
ARUCO_DICT_ID = cv2.aruco.DICT_APRILTAG_36h11

RETRIEVAL_TARGET_TAG = np.array(
    [0.0, 0.0, 1.0],
    dtype=np.float64,
)


# =============================================================================
# Frozen B2 optical-camera -> Base extrinsics from URDF
# =============================================================================
#
# Front:
#   xyz = [+0.3993, 0, -0.01576]
#   rpy = [-1.5708, 0, -1.5708]
#
# Back:
#   xyz = [-0.39143, 0, -0.026131]
#   rpy = [-1.5708, 0, +1.5708]
#
# URDF:
#   R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
# =============================================================================

FRONT_RPY_BASE = np.array(
    [-1.5708, 0.0, -1.5708],
    dtype=np.float64,
)

FRONT_T_BASE = np.array(
    [+0.3993, 0.0, -0.01576],
    dtype=np.float64,
)

BACK_RPY_BASE = np.array(
    [-1.5708, 0.0, +1.5708],
    dtype=np.float64,
)

BACK_T_BASE = np.array(
    [-0.39143, 0.0, -0.026131],
    dtype=np.float64,
)


# =============================================================================
# Production validity defaults
# =============================================================================

DEFAULT_MAX_RMSE_PX = 2.0
DEFAULT_VISUAL_MAX_AGE_S = 0.200

# If both cameras are fresh, newer data wins if capture times differ by more
# than this. Otherwise lower reprojection RMSE wins.
SOURCE_FRESHNESS_TIE_S = 0.050

# Reject very tiny detections even if ArUco decoded an ID.
DEFAULT_MIN_TAG_AREA_PX2 = 100.0

# Tag front normal should point approximately from Tag toward camera.
# facing_cos = dot(Tag +Z in camera, unit(Tag -> camera))
DEFAULT_MIN_FACING_COS = 0.10


# =============================================================================
# Geometry helpers
# =============================================================================

def rpy_to_rotmat(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    cr = math.cos(roll)
    sr = math.sin(roll)

    cp = math.cos(pitch)
    sp = math.sin(pitch)

    cy = math.cos(yaw)
    sy = math.sin(yaw)

    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr,  cr],
        ],
        dtype=np.float64,
    )

    Ry = np.array(
        [
            [ cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )

    Rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy,  cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return Rz @ Ry @ Rx


def make_transform(
    rpy: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    T = np.eye(
        4,
        dtype=np.float64,
    )

    T[:3, :3] = rpy_to_rotmat(
        float(rpy[0]),
        float(rpy[1]),
        float(rpy[2]),
    )

    T[:3, 3] = np.asarray(
        t,
        dtype=np.float64,
    ).reshape(3)

    return T


T_BASE_FROM_FRONT = make_transform(
    FRONT_RPY_BASE,
    FRONT_T_BASE,
)

T_BASE_FROM_BACK = make_transform(
    BACK_RPY_BASE,
    BACK_T_BASE,
)


# =============================================================================
# Intrinsics
# =============================================================================

def _npz_has_camera_intrinsics(path: Path) -> bool:
    try:
        with np.load(path) as data:
            return "K" in data.files and "D" in data.files
    except Exception:
        return False


def resolve_back_intrinsics(
    repo_root: Path,
    explicit: Optional[Path],
) -> Path:
    """
    Prefer an explicitly supplied file.

    Otherwise discover the existing calibrated B2 BACK intrinsic file without
    inventing numerical values. The repository already used calibrated BACK
    intrinsics in the earlier back/dual AprilTag tests; this resolver locates
    that file by content (K,D) and path/name.

    If discovery is ambiguous, fail loudly and require --back-intrinsics.
    """
    if explicit is not None:
        path = explicit.expanduser()

        if not path.is_absolute():
            path = repo_root / path

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"BACK intrinsics not found: {path}"
            )

        if not _npz_has_camera_intrinsics(path):
            raise RuntimeError(
                f"BACK intrinsics file lacks K/D: {path}"
            )

        return path

    preferred = [
        repo_root / "extrinsic_calib" / "back_rgb" / "back_intrinsics.npz",
        repo_root / "extrinsic_calib" / "back_rgb" / "rgb_intrinsics.npz",
        repo_root / "extrinsic_calib" / "back_rgb" / "intrinsics.npz",
        repo_root / "extrinsic_calib" / "back_intrinsics.npz",
    ]

    for path in preferred:
        if path.exists() and _npz_has_camera_intrinsics(path):
            return path.resolve()

    candidates = []

    root = repo_root / "extrinsic_calib"

    if root.exists():
        for path in root.rglob("*.npz"):
            text = str(path).lower()

            if "back" not in text:
                continue

            if not _npz_has_camera_intrinsics(path):
                continue

            score = 0

            if "back_rgb" in text:
                score += 100

            if "intrinsic" in path.name.lower():
                score += 50

            if path.name.lower() == "back_intrinsics.npz":
                score += 100

            candidates.append(
                (score, path.resolve())
            )

    if not candidates:
        raise FileNotFoundError(
            "Could not auto-discover calibrated B2 BACK intrinsics.\n"
            "Run:\n"
            "  find extrinsic_calib -type f -name '*.npz' -print\n"
            "then pass:\n"
            "  --back-intrinsics <path-to-tested-back-intrinsics.npz>"
        )

    candidates.sort(
        key=lambda item: (-item[0], str(item[1]))
    )

    top_score = candidates[0][0]

    top = [
        path
        for score, path in candidates
        if score == top_score
    ]

    if len(top) != 1:
        formatted = "\n".join(
            f"  {p}"
            for p in top
        )

        raise RuntimeError(
            "BACK intrinsic auto-discovery is ambiguous.\n"
            "Candidates:\n"
            f"{formatted}\n"
            "Pass --back-intrinsics explicitly."
        )

    return top[0]


def load_intrinsics(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]:
    with np.load(path) as data:
        if "K" not in data.files:
            raise RuntimeError(
                f"K missing from {path}"
            )

        if "D" not in data.files:
            raise RuntimeError(
                f"D missing from {path}"
            )

        K = np.asarray(
            data["K"],
            dtype=np.float64,
        ).reshape(3, 3)

        D = np.asarray(
            data["D"],
            dtype=np.float64,
        ).reshape(-1)

        image_size = None

        if "image_size" in data.files:
            s = np.asarray(
                data["image_size"]
            ).reshape(-1)

            if len(s) >= 2:
                image_size = (
                    int(s[0]),
                    int(s[1]),
                )

    return K, D, image_size


# =============================================================================
# Camera / AprilTag helpers
# =============================================================================

def decode_image(
    data: Any,
) -> Optional[np.ndarray]:
    if data is None:
        return None

    buf = np.frombuffer(
        bytes(data),
        dtype=np.uint8,
    )

    if buf.size == 0:
        return None

    return cv2.imdecode(
        buf,
        cv2.IMREAD_COLOR,
    )


def polygon_area_px2(
    image_corners: np.ndarray,
) -> float:
    pts = np.asarray(
        image_corners,
        dtype=np.float64,
    ).reshape(4, 2)

    return float(
        abs(
            cv2.contourArea(
                pts.astype(np.float32)
            )
        )
    )


def compute_reprojection_rmse(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        D,
    )

    projected = projected.reshape(-1, 2)

    image_points = np.asarray(
        image_points,
        dtype=np.float64,
    ).reshape(-1, 2)

    error = projected - image_points

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    error * error,
                    axis=1,
                )
            )
        )
    )


def solve_tag_pose_ippe_square(
    image_corners: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    tag_size_m: float,
    min_facing_cos: float,
) -> Optional[Dict[str, Any]]:
    """
    Solve T_CAMERA_FROM_TAG.

    IPPE has a planar two-solution ambiguity. We do NOT choose only by RMSE.
    For a visible Tag front face, Tag +Z must point from the Tag approximately
    toward the camera.

        facing_cos =
            dot(
                Tag +Z expressed in camera,
                unit vector Tag -> camera
            )

    Candidate must have:
        Tag center z > 0
        facing_cos >= min_facing_cos

    Among valid candidates, choose minimum reprojection RMSE.
    """
    half = float(tag_size_m) / 2.0

    object_points = np.array(
        [
            [-half, +half, 0.0],
            [+half, +half, 0.0],
            [+half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )

    image_points = np.asarray(
        image_corners,
        dtype=np.float64,
    ).reshape(4, 2)

    result = cv2.solvePnPGeneric(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=K,
        distCoeffs=D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not bool(result[0]):
        return None

    rvecs = result[1]
    tvecs = result[2]

    candidates = []

    for rvec, tvec in zip(
        rvecs,
        tvecs,
    ):
        rvec = np.asarray(
            rvec,
            dtype=np.float64,
        ).reshape(3, 1)

        tvec = np.asarray(
            tvec,
            dtype=np.float64,
        ).reshape(3, 1)

        p_cam_tag = tvec.reshape(3)

        if p_cam_tag[2] <= 0.0:
            continue

        R_CAM_FROM_TAG, _ = cv2.Rodrigues(
            rvec
        )

        tag_z_cam = R_CAM_FROM_TAG[:, 2]

        tag_to_camera = -p_cam_tag

        norm = float(
            np.linalg.norm(
                tag_to_camera
            )
        )

        if norm <= 1e-9:
            continue

        tag_to_camera /= norm

        facing_cos = float(
            np.dot(
                tag_z_cam,
                tag_to_camera,
            )
        )

        if facing_cos < min_facing_cos:
            continue

        rmse = compute_reprojection_rmse(
            object_points,
            image_points,
            rvec,
            tvec,
            K,
            D,
        )

        candidates.append(
            (
                rmse,
                -facing_cos,
                R_CAM_FROM_TAG,
                p_cam_tag,
                rvec,
                tvec,
                facing_cos,
            )
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    (
        rmse,
        _,
        R_CAM_FROM_TAG,
        p_cam_tag,
        rvec,
        tvec,
        facing_cos,
    ) = candidates[0]

    return {
        "R_camera_tag": R_CAM_FROM_TAG,
        "t_camera_tag": p_cam_tag,
        "rvec": rvec,
        "tvec": tvec,
        "reprojection_rmse_px": float(rmse),
        "facing_cos": float(facing_cos),
    }


def make_aruco_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICT_ID
    )

    if hasattr(
        cv2.aruco,
        "ArucoDetector",
    ):
        parameters = cv2.aruco.DetectorParameters()

        detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        )

        return (
            lambda gray:
            detector.detectMarkers(
                gray
            )
        )

    # Old OpenCV fallback.
    if hasattr(
        cv2.aruco,
        "DetectorParameters_create",
    ):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    return (
        lambda gray:
        cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )
    )


# =============================================================================
# B2 dual-camera detector
# =============================================================================

class B2AprilTagDetector:
    """
    Dual B2 optical-camera, asynchronous latest-value AprilTag detector.

    Lifecycle:
        1. ChannelFactoryInitialize(...) outside this class.
        2. detector = B2AprilTagDetector(...)
        3. detector.initialize_clients()      # main thread
        4. detector.start()
        5. detector.get_latest_snapshot()     # nonblocking
        6. detector.stop()
    """

    def __init__(
        self,
        repo_root: Path,
        front_intrinsics: Path,
        back_intrinsics: Path,
        tag_id: int = TAG_ID,
        tag_size_m: float = TAG_SIZE_M,
        retrieval_target_tag: np.ndarray = RETRIEVAL_TARGET_TAG,
        max_rmse_px: float = DEFAULT_MAX_RMSE_PX,
        visual_max_age_s: float = DEFAULT_VISUAL_MAX_AGE_S,
        min_tag_area_px2: float = DEFAULT_MIN_TAG_AREA_PX2,
        min_facing_cos: float = DEFAULT_MIN_FACING_COS,
    ):
        self.repo_root = repo_root.resolve()

        self.front_intrinsics_path = (
            front_intrinsics.resolve()
        )

        self.back_intrinsics_path = (
            back_intrinsics.resolve()
        )

        (
            self.K_front,
            self.D_front,
            self.front_image_size,
        ) = load_intrinsics(
            self.front_intrinsics_path
        )

        (
            self.K_back,
            self.D_back,
            self.back_image_size,
        ) = load_intrinsics(
            self.back_intrinsics_path
        )

        self.tag_id = int(tag_id)
        self.tag_size_m = float(tag_size_m)

        self.retrieval_target_tag = np.asarray(
            retrieval_target_tag,
            dtype=np.float64,
        ).reshape(3)

        self.max_rmse_px = float(
            max_rmse_px
        )

        self.visual_max_age_s = float(
            visual_max_age_s
        )

        self.min_tag_area_px2 = float(
            min_tag_area_px2
        )

        self.min_facing_cos = float(
            min_facing_cos
        )

        self._detect_markers = make_aruco_detector()

        self._front_client = None
        self._back_client = None

        self._clients_initialized = False

        self._stop_event = threading.Event()

        self._frame_lock = threading.Lock()

        self._latest_frame = {
            "FRONT": None,
            "BACK": None,
        }

        self._frame_seq = {
            "FRONT": 0,
            "BACK": 0,
        }

        self._last_processed_frame_seq = {
            "FRONT": -1,
            "BACK": -1,
        }

        self._measurement_lock = threading.Lock()

        self._measurement = {
            "FRONT": None,
            "BACK": None,
        }

        self._stats_lock = threading.Lock()

        self._stats = {
            "FRONT": {
                "rpc_fail": 0,
                "decode_fail": 0,
                "frames": 0,
                "detections": 0,
                "valid_poses": 0,
            },
            "BACK": {
                "rpc_fail": 0,
                "decode_fail": 0,
                "frames": 0,
                "detections": 0,
                "valid_poses": 0,
            },
        }

        self._threads = []

        self._camera_shape_checked = {
            "FRONT": False,
            "BACK": False,
        }

        self._fatal_lock = threading.Lock()
        self._fatal_error = None

    def initialize_clients(
        self,
        timeout_s: float = 3.0,
    ) -> None:
        """
        Initialize Unitree RPC clients in the caller/main thread.

        This matches the already-tested B2 dual-camera deployment pattern.
        """
        if self._clients_initialized:
            return

        front = FrontVideoClient()
        front.SetTimeout(
            float(timeout_s)
        )
        front.Init()

        back = BackVideoClient()
        back.SetTimeout(
            float(timeout_s)
        )
        back.Init()

        self._front_client = front
        self._back_client = back

        self._clients_initialized = True

    def start(self) -> None:
        if not self._clients_initialized:
            raise RuntimeError(
                "Call initialize_clients() before start()."
            )

        if self._threads:
            raise RuntimeError(
                "Detector already started."
            )

        self._stop_event.clear()

        front_thread = threading.Thread(
            target=self._camera_producer,
            args=("FRONT", self._front_client),
            name="b2-front-camera-producer",
            daemon=True,
        )

        back_thread = threading.Thread(
            target=self._camera_producer,
            args=("BACK", self._back_client),
            name="b2-back-camera-producer",
            daemon=True,
        )

        processor_thread = threading.Thread(
            target=self._processor,
            name="b2-apriltag-processor",
            daemon=True,
        )

        self._threads = [
            front_thread,
            back_thread,
            processor_thread,
        ]

        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop_event.set()

        for thread in self._threads:
            thread.join(
                timeout=4.0
            )

        self._threads = []

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return copy.deepcopy(
                self._stats
            )

    def get_camera_snapshot(
        self,
        camera: str,
        max_age_s: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        camera = camera.upper()

        if camera not in (
            "FRONT",
            "BACK",
        ):
            raise ValueError(
                f"Unknown camera: {camera}"
            )

        with self._measurement_lock:
            measurement = self._measurement[
                camera
            ]

            if measurement is None:
                return None

            snap = copy.deepcopy(
                measurement
            )

        now_ns = time.monotonic_ns()

        age_s = (
            now_ns
            - int(
                snap[
                    "capture_host_monotonic_ns"
                ]
            )
        ) / 1e9

        snap["age_ms"] = float(
            age_s * 1000.0
        )

        limit = (
            self.visual_max_age_s
            if max_age_s is None
            else float(max_age_s)
        )

        if age_s > limit:
            snap["valid"] = False
            snap["reason"] = "STALE"

        return snap

    def get_latest_snapshot(
        self,
        max_age_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Nonblocking unified visual result.

        Selection:
          - discard invalid/stale measurements
          - if only one camera valid, use it
          - if both valid:
              * if one capture is >50 ms newer, use newer
              * otherwise use lower reprojection RMSE

        Physical Tag center remains present separately from derived retrieval.
        """
        limit = (
            self.visual_max_age_s
            if max_age_s is None
            else float(max_age_s)
        )

        front = self.get_camera_snapshot(
            "FRONT",
            limit,
        )

        back = self.get_camera_snapshot(
            "BACK",
            limit,
        )

        candidates = [
            s
            for s in (front, back)
            if s is not None
            and bool(
                s.get(
                    "valid",
                    False,
                )
            )
        ]

        selected = None

        if len(candidates) == 1:
            selected = candidates[0]

        elif len(candidates) == 2:
            f = next(
                s
                for s in candidates
                if s["source"] == "FRONT"
            )

            b = next(
                s
                for s in candidates
                if s["source"] == "BACK"
            )

            dt_s = (
                int(
                    f[
                        "capture_host_monotonic_ns"
                    ]
                )
                - int(
                    b[
                        "capture_host_monotonic_ns"
                    ]
                )
            ) / 1e9

            if abs(dt_s) > SOURCE_FRESHNESS_TIE_S:
                selected = (
                    f
                    if dt_s > 0.0
                    else b
                )

            else:
                selected = min(
                    (f, b),
                    key=lambda s: (
                        float(
                            s[
                                "reprojection_rmse_px"
                            ]
                        ),
                        -float(
                            s[
                                "tag_area_px2"
                            ]
                        ),
                    ),
                )

        result = {
            "valid": selected is not None,
            "source": (
                selected["source"]
                if selected is not None
                else None
            ),
            "reason": (
                "VALID"
                if selected is not None
                else "NO_FRESH_VISUAL_TAG"
            ),

            "tag_id": self.tag_id,
            "tag_size_m": self.tag_size_m,

            "front_valid": bool(
                front is not None
                and front.get(
                    "valid",
                    False,
                )
            ),

            "back_valid": bool(
                back is not None
                and back.get(
                    "valid",
                    False,
                )
            ),

            "front_age_ms": (
                None
                if front is None
                else front.get(
                    "age_ms"
                )
            ),

            "back_age_ms": (
                None
                if back is None
                else back.get(
                    "age_ms"
                )
            ),

            "front": front,
            "back": back,

            "tag_center_base": None,
            "R_base_tag": None,
            "tag_z_base": None,
            "retrieval_target_base": None,

            "capture_host_monotonic_ns": None,
            "receive_host_monotonic_ns": None,
            "rpc_duration_ms": None,

            "reprojection_rmse_px": None,
            "facing_cos": None,
            "tag_area_px2": None,
        }

        if selected is not None:
            for key in (
                "tag_center_base",
                "R_base_tag",
                "tag_z_base",
                "retrieval_target_base",
                "capture_host_monotonic_ns",
                "receive_host_monotonic_ns",
                "rpc_duration_ms",
                "reprojection_rmse_px",
                "facing_cos",
                "tag_area_px2",
            ):
                result[key] = copy.deepcopy(
                    selected[key]
                )

            result["age_ms"] = float(
                selected["age_ms"]
            )

        else:
            result["age_ms"] = None

        with self._fatal_lock:
            result["fatal_error"] = (
                None
                if self._fatal_error is None
                else repr(
                    self._fatal_error
                )
            )

        return result

    def _increment_stat(
        self,
        camera: str,
        key: str,
        amount: int = 1,
    ) -> None:
        with self._stats_lock:
            self._stats[camera][key] += int(
                amount
            )

    def _camera_producer(
        self,
        camera: str,
        client: Any,
    ) -> None:
        """
        Camera RPC producer.

        Timestamp semantics:
            t_before = host monotonic immediately before GetImageSample()
            t_after  = host monotonic immediately after RPC response
            capture_host_monotonic_ns = midpoint(t_before, t_after)

        The midpoint is the best available host-side approximation for later
        D435i VO pose-history lookup because the Unitree image RPC does not
        expose a hardware capture timestamp here.
        """
        while not self._stop_event.is_set():
            t_before_ns = time.monotonic_ns()

            try:
                code, data = client.GetImageSample()
            except Exception:
                self._increment_stat(
                    camera,
                    "rpc_fail",
                )
                continue

            t_after_ns = time.monotonic_ns()

            if code != 0:
                self._increment_stat(
                    camera,
                    "rpc_fail",
                )
                continue

            image = decode_image(
                data
            )

            if image is None:
                self._increment_stat(
                    camera,
                    "decode_fail",
                )
                continue

            self._increment_stat(
                camera,
                "frames",
            )

            capture_ns = (
                t_before_ns
                + t_after_ns
            ) // 2

            rpc_duration_ms = (
                t_after_ns
                - t_before_ns
            ) / 1e6

            with self._frame_lock:
                seq = self._frame_seq[
                    camera
                ]

                self._frame_seq[
                    camera
                ] += 1

                self._latest_frame[
                    camera
                ] = {
                    "image": image,
                    "frame_seq": int(seq),

                    "capture_host_monotonic_ns":
                        int(
                            capture_ns
                        ),

                    "receive_host_monotonic_ns":
                        int(
                            t_after_ns
                        ),

                    "rpc_duration_ms":
                        float(
                            rpc_duration_ms
                        ),
                }

    def _processor(self) -> None:
        """
        Process only the newest unprocessed frame from each camera.

        Producers overwrite old camera frames. Therefore processor backlog can
        never grow and the detector naturally behaves as latest-value vision.
        """
        cameras = (
            "FRONT",
            "BACK",
        )

        try:
            while not self._stop_event.is_set():
                did_work = False

                for camera in cameras:
                    with self._frame_lock:
                        item = self._latest_frame[
                            camera
                        ]

                        if item is None:
                            continue

                        seq = int(
                            item[
                                "frame_seq"
                            ]
                        )

                        if (
                            seq
                            == self._last_processed_frame_seq[
                                camera
                            ]
                        ):
                            continue

                        # Copy image reference; producer replaces dictionary,
                        # not the existing ndarray.
                        frame_item = dict(
                            item
                        )

                    self._last_processed_frame_seq[
                        camera
                    ] = seq

                    did_work = True

                    measurement = self._process_one(
                        camera,
                        frame_item,
                    )

                    with self._measurement_lock:
                        self._measurement[
                            camera
                        ] = measurement

                if not did_work:
                    time.sleep(
                        0.001
                    )

        except BaseException as exc:
            with self._fatal_lock:
                self._fatal_error = exc

    def _check_image_shape(
        self,
        camera: str,
        image: np.ndarray,
        expected: Optional[Tuple[int, int]],
    ) -> None:
        if self._camera_shape_checked[
            camera
        ]:
            return

        h, w = image.shape[:2]

        if expected is not None:
            expected_w, expected_h = (
                int(expected[0]),
                int(expected[1]),
            )

            if (
                w != expected_w
                or h != expected_h
            ):
                raise RuntimeError(
                    f"{camera} image is {w}x{h}, "
                    f"but calibrated intrinsics expect "
                    f"{expected_w}x{expected_h}"
                )

        self._camera_shape_checked[
            camera
        ] = True

    def _process_one(
        self,
        camera: str,
        frame_item: Dict[str, Any],
    ) -> Dict[str, Any]:
        if camera == "FRONT":
            K = self.K_front
            D = self.D_front
            image_size = self.front_image_size
            T_BASE_FROM_CAMERA = (
                T_BASE_FROM_FRONT
            )
        else:
            K = self.K_back
            D = self.D_back
            image_size = self.back_image_size
            T_BASE_FROM_CAMERA = (
                T_BASE_FROM_BACK
            )

        image = frame_item[
            "image"
        ]

        self._check_image_shape(
            camera,
            image,
            image_size,
        )

        measurement = {
            "valid": False,
            "reason": "NO_TAG",
            "source": camera,

            "frame_seq": int(
                frame_item[
                    "frame_seq"
                ]
            ),

            "capture_host_monotonic_ns":
                int(
                    frame_item[
                        "capture_host_monotonic_ns"
                    ]
                ),

            "receive_host_monotonic_ns":
                int(
                    frame_item[
                        "receive_host_monotonic_ns"
                    ]
                ),

            "rpc_duration_ms":
                float(
                    frame_item[
                        "rpc_duration_ms"
                    ]
                ),

            "tag_id": self.tag_id,

            "tag_center_camera": None,
            "tag_center_base": None,

            "R_camera_tag": None,
            "R_base_tag": None,

            "tag_z_base": None,

            "retrieval_target_base": None,

            "reprojection_rmse_px": None,
            "facing_cos": None,
            "tag_area_px2": None,
        }

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        corners, ids, _ = self._detect_markers(
            gray
        )

        if ids is None:
            return measurement

        ids_flat = np.asarray(
            ids
        ).reshape(-1)

        matches = np.where(
            ids_flat == self.tag_id
        )[0]

        if len(matches) == 0:
            return measurement

        target_index = int(
            matches[0]
        )

        self._increment_stat(
            camera,
            "detections",
        )

        target_corners = np.asarray(
            corners[target_index],
            dtype=np.float64,
        ).reshape(4, 2)

        area_px2 = polygon_area_px2(
            target_corners
        )

        measurement[
            "tag_area_px2"
        ] = float(
            area_px2
        )

        if area_px2 < self.min_tag_area_px2:
            measurement[
                "reason"
            ] = "TAG_TOO_SMALL"

            return measurement

        pose = solve_tag_pose_ippe_square(
            target_corners,
            K,
            D,
            self.tag_size_m,
            self.min_facing_cos,
        )

        if pose is None:
            measurement[
                "reason"
            ] = "PNP_FAILED_OR_BAD_FACING"

            return measurement

        rmse = float(
            pose[
                "reprojection_rmse_px"
            ]
        )

        measurement[
            "reprojection_rmse_px"
        ] = rmse

        measurement[
            "facing_cos"
        ] = float(
            pose[
                "facing_cos"
            ]
        )

        if rmse > self.max_rmse_px:
            measurement[
                "reason"
            ] = "RMSE_REJECTED"

            return measurement

        R_CAMERA_TAG = np.asarray(
            pose[
                "R_camera_tag"
            ],
            dtype=np.float64,
        ).reshape(3, 3)

        t_CAMERA_TAG = np.asarray(
            pose[
                "t_camera_tag"
            ],
            dtype=np.float64,
        ).reshape(3)

        T_CAMERA_FROM_TAG = np.eye(
            4,
            dtype=np.float64,
        )

        T_CAMERA_FROM_TAG[
            :3,
            :3,
        ] = R_CAMERA_TAG

        T_CAMERA_FROM_TAG[
            :3,
            3,
        ] = t_CAMERA_TAG

        T_BASE_FROM_TAG = (
            T_BASE_FROM_CAMERA
            @ T_CAMERA_FROM_TAG
        )

        R_BASE_TAG = (
            T_BASE_FROM_TAG[
                :3,
                :3,
            ]
        )

        p_tag_base = (
            T_BASE_FROM_TAG[
                :3,
                3,
            ]
        )

        tag_z_base = (
            R_BASE_TAG[:, 2]
        )

        p_retrieval_base = (
            R_BASE_TAG
            @ self.retrieval_target_tag
            + p_tag_base
        )

        measurement.update(
            {
                "valid": True,
                "reason": "VALID",

                "tag_center_camera": [
                    float(x)
                    for x in t_CAMERA_TAG
                ],

                "tag_center_base": [
                    float(x)
                    for x in p_tag_base
                ],

                "R_camera_tag": [
                    [
                        float(x)
                        for x in row
                    ]
                    for row in R_CAMERA_TAG
                ],

                "R_base_tag": [
                    [
                        float(x)
                        for x in row
                    ]
                    for row in R_BASE_TAG
                ],

                "tag_z_base": [
                    float(x)
                    for x in tag_z_base
                ],

                "retrieval_target_base": [
                    float(x)
                    for x in p_retrieval_base
                ],
            }
        )

        self._increment_stat(
            camera,
            "valid_poses",
        )

        return measurement


# =============================================================================
# Standalone CLI
# =============================================================================

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
        "--tag-id",
        type=int,
        default=TAG_ID,
    )

    parser.add_argument(
        "--tag-size",
        type=float,
        default=TAG_SIZE_M,
    )

    parser.add_argument(
        "--retrieval-z",
        type=float,
        default=1.0,
        help=(
            "Retrieval target offset along Tag +Z in meters."
        ),
    )

    parser.add_argument(
        "--max-rmse",
        type=float,
        default=DEFAULT_MAX_RMSE_PX,
    )

    parser.add_argument(
        "--max-age",
        type=float,
        default=DEFAULT_VISUAL_MAX_AGE_S,
        help=(
            "Maximum visual measurement age in seconds."
        ),
    )

    parser.add_argument(
        "--min-tag-area",
        type=float,
        default=DEFAULT_MIN_TAG_AREA_PX2,
    )

    parser.add_argument(
        "--min-facing-cos",
        type=float,
        default=DEFAULT_MIN_FACING_COS,
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
        help=(
            "Calibrated B2 BACK intrinsics .npz. "
            "If omitted, auto-discover under extrinsic_calib."
        ),
    )

    parser.add_argument(
        "--print-period",
        type=float,
        default=0.20,
    )

    args = parser.parse_args()

    # File is expected at:
    #   <repo>/example/b2w/camera/b2_apriltag_detector.py
    repo_root = Path(
        __file__
    ).resolve().parents[3]

    if args.front_intrinsics is None:
        front_intrinsics = (
            repo_root
            / "extrinsic_calib"
            / "front_rgb"
            / "front_intrinsics.npz"
        )
    else:
        front_intrinsics = (
            args.front_intrinsics.expanduser()
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
            f"FRONT intrinsics not found: "
            f"{front_intrinsics}"
        )

    back_intrinsics = resolve_back_intrinsics(
        repo_root,
        args.back_intrinsics,
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
        front_intrinsics=front_intrinsics,
        back_intrinsics=back_intrinsics,
        tag_id=args.tag_id,
        tag_size_m=args.tag_size,
        retrieval_target_tag=retrieval_target_tag,
        max_rmse_px=args.max_rmse,
        visual_max_age_s=args.max_age,
        min_tag_area_px2=args.min_tag_area,
        min_facing_cos=args.min_facing_cos,
    )

    np.set_printoptions(
        precision=6,
        suppress=True,
    )

    print("=" * 104)
    print("B2W DUAL OPTICAL APRILTAG DETECTOR -- PRODUCTION VISUAL SOURCE")
    print("=" * 104)
    print("DDS interface          :", args.interface)
    print("Tag family             : tag36h11")
    print("Tag ID                 :", args.tag_id)
    print(
        "Tag size               :",
        f"{args.tag_size:.3f} m",
    )
    print(
        "Retrieval in Tag       :",
        retrieval_target_tag,
    )
    print(
        "Visual max age         :",
        f"{args.max_age:.3f} s",
    )
    print(
        "Max reprojection RMSE  :",
        f"{args.max_rmse:.3f} px",
    )
    print(
        "Min Tag area           :",
        f"{args.min_tag_area:.1f} px^2",
    )
    print(
        "Min facing cosine      :",
        f"{args.min_facing_cos:.3f}",
    )
    print(
        "FRONT intrinsics       :",
        front_intrinsics,
    )
    print(
        "BACK intrinsics        :",
        back_intrinsics,
    )
    print()
    print(
        "Physical Tag center    : "
        "VISUAL SOURCE OF TRUTH / future VO anchor"
    )
    print(
        "Virtual retrieval      : "
        "derived HL target only"
    )
    print(
        "Camera RPC in HL/LL    : NEVER"
    )
    print(
        "AprilTag detect in HL/LL: NEVER"
    )
    print("=" * 104)

    print(
        "[INIT] Initializing FrontVideoClient "
        "in main thread..."
    )

    detector.initialize_clients()

    print(
        "[INIT] FrontVideoClient + BackVideoClient ready."
    )

    detector.start()

    print(
        "[INIT] Front/Back producers + "
        "AprilTag processor started."
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
                snap = detector.get_latest_snapshot(
                    max_age_s=args.max_age
                )

                if snap[
                    "fatal_error"
                ] is not None:
                    print(
                        "[FATAL] processor error:",
                        snap[
                            "fatal_error"
                        ],
                    )
                    break

                if snap["valid"]:
                    p_tag = snap[
                        "tag_center_base"
                    ]

                    p_ret = snap[
                        "retrieval_target_base"
                    ]

                    z_tag = snap[
                        "tag_z_base"
                    ]

                    print(
                        "[VALID] "
                        f"source={snap['source']:<5s} "
                        f"age={snap['age_ms']:5.1f}ms "
                        f"rmse="
                        f"{snap['reprojection_rmse_px']:.3f}px "
                        f"front={int(snap['front_valid'])} "
                        f"back={int(snap['back_valid'])}"
                    )

                    print(
                        "        Base<-Tag       = "
                        f"[{p_tag[0]:+.3f}, "
                        f"{p_tag[1]:+.3f}, "
                        f"{p_tag[2]:+.3f}] m"
                    )

                    print(
                        "        Tag +Z in Base  = "
                        f"[{z_tag[0]:+.3f}, "
                        f"{z_tag[1]:+.3f}, "
                        f"{z_tag[2]:+.3f}]"
                    )

                    print(
                        "        Base<-Retrieval = "
                        f"[{p_ret[0]:+.3f}, "
                        f"{p_ret[1]:+.3f}, "
                        f"{p_ret[2]:+.3f}] m"
                    )

                else:
                    print(
                        "[INVALID] "
                        f"reason={snap['reason']} "
                        f"front={int(snap['front_valid'])} "
                        f"back={int(snap['back_valid'])} "
                        f"front_age={snap['front_age_ms']} "
                        f"back_age={snap['back_age_ms']}"
                    )

                last_print = now

            time.sleep(
                0.01
            )

    finally:
        detector.stop()

        stats = detector.get_stats()

        print()
        print("=" * 104)
        print("FINAL CAMERA STATS")
        print("=" * 104)
        print("FRONT:", stats["FRONT"])
        print("BACK :", stats["BACK"])
        print("=" * 104)
        print("Stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
