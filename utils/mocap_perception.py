"""
Mocap-backed replacement for the camera/VO perception system.

`MocapPerceptionSystem` presents the SAME interface that
`b2w_perception.CRLB2WPerceptionSystem` presents to
`B2WZ1HierarchicalRetrievalController`, so the hierarchical retrieval
controller can be pointed at OptiTrack without any change to the policy,
observation, actuation or safety paths:

    initialize() / start() / stop()
    get_latest_snapshot(projected_gravity_b=None) -> dict
    anchor_vo_base_height(...) / get_vo_base_height_snapshot()
    reset_vo_base_height_estimator(...)

Three Motive rigid bodies replace the whole AprilTag + RealSense + rear-VO
stack:

    body        the B2 itself. Combined with the calibrated mocap->root
                offset it gives the exact base_link pose, and therefore both
                the base height and the world->base transform.
    object      the object to retrieve (the "octopus" asset).
    retrieval   the retrieval target.

What the controller actually consumes is object and retrieval expressed in
the ROBOT BASE frame, plus a scalar base height, so the maths is just:

    p_root^W, q_root^W = root_offset applied to the body rigid body
    base_height        = p_root^W.z - ground_z
    p_x^base           = R(q_root^W)^T . (p_x^W - p_root^W)

Both the object and the retrieval target support a constant offset expressed
in their own rigid-body frame, because Motive's pivot for an asset sits
wherever the marker cluster's centroid landed rather than at the physically
meaningful point (the graspable centre of the object, the actual drop point
of the target).

A note on which orientation is used where
-----------------------------------------
The world->base rotation above is the MOCAP orientation, not the IMU's. It
has to be: `p_x^W` is expressed in the mocap world frame, and the IMU's world
frame has an arbitrary yaw, so mixing them would rotate the object by an
unknown angle about Z.

The controller separately keeps using the B2 IMU for `base_quat_wxyz` and
`projected_gravity_b`, exactly as the validated camera-based deployment does.
That is consistent as long as mocap and IMU agree on roll/pitch, which is
precisely what the gravity solver in the calibration tool enforces (it drives
that residual to well under a tenth of a degree). Yaw disagreement is
harmless here, because a position expressed in the base frame is unaffected
by how the base's yaw is labelled.

Failure reporting
-----------------
Nothing is faked, latched or extrapolated. A rigid body that is untracked,
stale or absent from the stream makes its channel invalid with a specific
reason string, and the controller's existing safe-hold-then-damping logic
takes over. That is the same contract the camera system had.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.mocap_frames import RootOffset, load_root_offset, q_apply, q_apply_inv
from utils.natnet_client import NatNetClient, RigidBodySample


def _as_point(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr


class MocapBodySelector:
    """
    How one Motive rigid body is addressed, plus its constant local offset.

    `name` is preferred over `rb_id` because it survives someone renumbering
    the asset in Motive, but it only resolves if the server answers the
    model-definition request, so the id stays as the fallback.
    """

    __slots__ = ("label", "name", "rb_id", "offset_local", "max_age_s")

    def __init__(
        self,
        label: str,
        name: Optional[str] = None,
        rb_id: Optional[int] = None,
        offset_local: Any = (0.0, 0.0, 0.0),
        max_age_s: float = 0.25,
    ) -> None:
        if name is None and rb_id is None:
            raise ValueError(f"{label}: one of name / rb_id must be given")
        self.label = str(label)
        self.name = str(name) if name else None
        self.rb_id = int(rb_id) if rb_id is not None else None
        offset = _as_point(offset_local)
        if offset is None:
            raise ValueError(f"{label}: offset_local must be 3 finite numbers")
        self.offset_local = offset
        self.max_age_s = float(max_age_s)

    def describe(self) -> str:
        target = f"{self.name!r}" if self.name else f"id {self.rb_id}"
        off = np.round(self.offset_local, 4).tolist()
        return f"{self.label}={target} offset_local={off} max_age={self.max_age_s * 1e3:.0f}ms"


class MocapPerceptionSystem:
    """Drop-in perception provider driven entirely by OptiTrack/Motive."""

    def __init__(
        self,
        body: MocapBodySelector,
        object_body: MocapBodySelector,
        retrieval_body: MocapBodySelector,
        root_offset: Any = None,
        ground_z: float = 0.0,
        local_ip: str = "",
        server_ip: Optional[str] = None,
        multicast_group: str = "239.255.42.99",
        data_port: int = 1511,
        command_port: int = 1510,
        join_multicast: bool = True,
        up_axis: str = "auto",
        startup_timeout_s: float = 10.0,
        verbose: bool = True,
    ) -> None:
        self.body = body
        self.object_body = object_body
        self.retrieval_body = retrieval_body
        # None -> identity: the Motive rigid body's own frame is taken as the
        # robot root. Usable without calibrating, at the cost of the base pose
        # being wrong by whatever the asset's pivot and axes happen to be.
        self.root_offset: RootOffset = load_root_offset(root_offset)
        self.root_offset_calibrated = not self.root_offset.is_identity()
        self.ground_z = float(ground_z)
        self.startup_timeout_s = float(startup_timeout_s)
        self.verbose = bool(verbose)

        self.client = NatNetClient(
            server_ip=server_ip,
            local_ip=local_ip,
            multicast_group=multicast_group,
            data_port=data_port,
            command_port=command_port,
            join_multicast=join_multicast,
            up_axis=up_axis,
            verbose=verbose,
        )

        self._initialized = False
        self._started = False
        self._lock = threading.Lock()
        self._resolved: Dict[str, Optional[int]] = {}

    # ------------------------------------------------------------------
    # Lifecycle (mirrors CRLB2WPerceptionSystem)
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[MOCAP-PERCEPTION] {msg}")

    def initialize(self) -> None:
        if self._initialized:
            return

        self.client.start()

        if not self.client.wait_for_first_frame(timeout_s=self.startup_timeout_s):
            raise RuntimeError(
                "No NatNet frames received. Check that Motive's Streaming pane "
                "is enabled with Rigid Bodies checked, that the mocap NIC "
                f"address is right, and that the server is reachable."
            )

        streamed = self.client.rigid_body_ids()
        names = self.client.rigid_body_names()
        self._log(f"streaming rigid bodies: " + (
            ", ".join(f"{i}:{names.get(i, '?')!r}" for i in streamed) or "(none)"
        ))

        missing = []
        for selector in (self.body, self.object_body, self.retrieval_body):
            rb_id = self._resolve(selector, names, streamed)
            self._resolved[selector.label] = rb_id
            if rb_id is None or rb_id not in streamed:
                missing.append(selector.label)
            self._log("  " + selector.describe() + f" -> id {rb_id}")

        if missing:
            raise RuntimeError(
                "These rigid bodies are not being streamed by Motive: "
                + ", ".join(missing)
                + f". Streamed ids are {streamed}. Create/enable the assets in "
                "Motive, or fix the names/ids in the config."
            )

        # Tracking validity is reported per frame and is a runtime condition,
        # not a startup error -- but an asset that is untracked right now will
        # block policy start, so say so plainly here rather than letting the
        # operator watch a silent wait loop.
        untracked = [
            sel.label
            for sel in (self.body, self.object_body, self.retrieval_body)
            if not self._is_tracked(self._resolved[sel.label])
        ]
        if untracked:
            self._log(
                "[WARN] currently UNTRACKED (occluded or not solving in "
                "Motive): " + ", ".join(untracked)
            )

        self._initialized = True

    def start(self) -> None:
        if not self._initialized:
            raise RuntimeError("initialize() must be called before start()")
        self._started = True
        rate = f"{self.client.frame_rate:g} Hz" if self.client.frame_rate else "unknown rate"
        self._log(
            f"running | up_axis={self.client.up_axis} | {rate} | root offset "
            + ("CALIBRATED" if self.root_offset_calibrated else "IDENTITY (uncalibrated)")
        )

    def stop(self) -> None:
        self._started = False
        self.client.stop()

    def _resolve(
        self,
        selector: MocapBodySelector,
        names: Dict[int, str],
        streamed: list,
    ) -> Optional[int]:
        if selector.name:
            for rb_id, rb_name in names.items():
                if rb_name == selector.name:
                    return int(rb_id)
            if selector.rb_id is None:
                self._log(
                    f"[WARN] {selector.label}: name {selector.name!r} not found in "
                    f"the model definitions and no fallback id was configured"
                )
                return None
            self._log(
                f"[WARN] {selector.label}: name {selector.name!r} not resolvable; "
                f"falling back to configured id {selector.rb_id}"
            )
        return selector.rb_id

    def _is_tracked(self, rb_id: Optional[int]) -> bool:
        if rb_id is None:
            return False
        rb = self.client.latest_rigid_body(rb_id)
        return rb is not None and rb.tracking_valid

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _sample(
        self, selector: MocapBodySelector, now: float
    ) -> Tuple[Optional[RigidBodySample], str, Optional[float]]:
        """Fetch one rigid body and classify why it is unusable, if it is."""
        rb_id = self._resolved.get(selector.label)
        if rb_id is None:
            return None, "NOT_CONFIGURED", None

        rb = self.client.latest_rigid_body(rb_id)
        if rb is None:
            return None, "NOT_IN_STREAM", None

        age_ms = 1000.0 * rb.age_s(now)
        if not rb.tracking_valid:
            return None, "UNTRACKED", age_ms
        if rb.age_s(now) > selector.max_age_s:
            return None, "STALE", age_ms
        return rb, "OK", age_ms

    @staticmethod
    def _world_point(rb: RigidBodySample, selector: MocapBodySelector) -> np.ndarray:
        """Rigid-body pose plus its constant offset, in the mocap world frame."""
        return rb.pos + q_apply(rb.quat_wxyz, selector.offset_local)

    def get_latest_snapshot(
        self, projected_gravity_b: Any = None
    ) -> Dict[str, Any]:
        """
        Nonblocking read shaped exactly like the camera system's snapshot.

        `projected_gravity_b` is accepted for interface compatibility and
        deliberately unused: the camera system needed the IMU to turn a VO
        translation into a height, whereas mocap measures the base pose in an
        already gravity-aligned world frame.
        """
        now = time.monotonic()
        frame = self.client.latest_frame()

        body_rb, body_reason, body_age_ms = self._sample(self.body, now)

        if body_rb is None:
            reason = f"BODY_{body_reason}"
            empty = self._invalid_channel(reason, body_age_ms)
            return self._assemble(
                object_state=dict(empty, position_base=None, position_world=None),
                retrieval_state=dict(
                    empty, retrieval_target_base=None, position_world=None
                ),
                base_height_state=self._invalid_height(reason, body_age_ms),
                body_state={
                    "valid": False,
                    "reason": body_reason,
                    "age_ms": body_age_ms,
                    "position_world": None,
                    "quat_wxyz": None,
                    "root_position_world": None,
                    "root_quat_wxyz": None,
                },
                frame=frame,
            )

        # The body is good, so the base frame and height are known.
        p_root_w, q_root_w = self.root_offset.apply(body_rb.pos, body_rb.quat_wxyz)
        base_height = float(p_root_w[2] - self.ground_z)

        object_state = self._target_channel(
            self.object_body, "position_base", p_root_w, q_root_w, now
        )
        retrieval_state = self._target_channel(
            self.retrieval_body, "retrieval_target_base", p_root_w, q_root_w, now
        )

        height_valid = bool(np.isfinite(base_height))
        base_height_state = {
            "valid": height_valid,
            "height_m": base_height if height_valid else None,
            "reason": "OK" if height_valid else "NON_FINITE_HEIGHT",
            "source": "MOCAP",
            # The VO estimator reported a session epoch here; the NatNet frame
            # number is the closest analogue and keeps the controller's
            # existing debug line informative.
            "epoch": int(body_rb.frame_number),
            "age_ms": body_age_ms,
            "ground_z": self.ground_z,
            "root_position_world": p_root_w.tolist(),
        }

        return self._assemble(
            object_state=object_state,
            retrieval_state=retrieval_state,
            base_height_state=base_height_state,
            body_state={
                "valid": True,
                "reason": "OK",
                "age_ms": body_age_ms,
                "position_world": body_rb.pos.tolist(),
                "quat_wxyz": body_rb.quat_wxyz.tolist(),
                "root_position_world": p_root_w.tolist(),
                "root_quat_wxyz": q_root_w.tolist(),
                "mean_marker_error_m": body_rb.mean_marker_error,
            },
            frame=frame,
        )

    def _target_channel(
        self,
        selector: MocapBodySelector,
        base_key: str,
        p_root_w: np.ndarray,
        q_root_w: np.ndarray,
        now: float,
    ) -> Dict[str, Any]:
        rb, reason, age_ms = self._sample(selector, now)
        if rb is None:
            state = self._invalid_channel(reason, age_ms)
            state[base_key] = None
            state["position_world"] = None
            return state

        p_world = self._world_point(rb, selector)
        p_base = q_apply_inv(q_root_w, p_world - p_root_w)

        return {
            "valid": True,
            "source": f"MOCAP:{selector.name or selector.rb_id}",
            "reason": "OK",
            base_key: p_base.tolist(),
            "position_world": p_world.tolist(),
            "age_ms": age_ms,
            "source_age_ms": age_ms,
            "mean_marker_error_m": rb.mean_marker_error,
        }

    @staticmethod
    def _invalid_channel(reason: str, age_ms: Optional[float]) -> Dict[str, Any]:
        return {
            "valid": False,
            "source": None,
            "reason": reason,
            "age_ms": age_ms,
            "source_age_ms": age_ms,
        }

    @staticmethod
    def _invalid_height(reason: str, age_ms: Optional[float]) -> Dict[str, Any]:
        return {
            "valid": False,
            "height_m": None,
            "reason": reason,
            "source": "MOCAP",
            "epoch": None,
            "age_ms": age_ms,
        }

    def _assemble(
        self,
        object_state: Dict[str, Any],
        retrieval_state: Dict[str, Any],
        base_height_state: Dict[str, Any],
        body_state: Dict[str, Any],
        frame: Any,
    ) -> Dict[str, Any]:
        object_valid = bool(object_state["valid"])
        retrieval_valid = bool(retrieval_state["valid"])
        all_valid = object_valid and retrieval_valid

        invalid_channels = []
        if not object_valid:
            invalid_channels.append("object")
        if not retrieval_valid:
            invalid_channels.append("retrieval")

        return {
            "valid": all_valid,
            "all_required_valid": all_valid,
            "invalid_channels": invalid_channels,
            "snapshot_crl_monotonic_ns": int(time.monotonic_ns()),
            "object": object_state,
            "retrieval": retrieval_state,
            "base_height": base_height_state,
            "body": body_state,
            "mocap": {
                "frame_number": int(frame.frame_number) if frame is not None else None,
                "frames_received": self.client.frame_count,
                "parse_errors": self.client.parse_error_count,
                "up_axis": self.client.up_axis,
                "server": self.client.server_info.get("name"),
            },
        }

    # ------------------------------------------------------------------
    # Base-height API
    #
    # The camera deployment had to anchor a drifting VO translation against a
    # known standing height. Mocap measures the base pose absolutely in a
    # gravity-aligned frame, so there is nothing to anchor and nothing to
    # reset -- these exist so the controller's VO-era code path stays
    # satisfied, and they report the true measured height.
    # ------------------------------------------------------------------

    def anchor_vo_base_height(
        self,
        projected_gravity_b: Any = None,
        height_m: Any = None,
    ) -> Dict[str, Any]:
        state = self.get_latest_snapshot()["base_height"]
        return dict(state, anchor_requested_height_m=height_m)

    def get_vo_base_height_snapshot(
        self, projected_gravity_b: Any = None
    ) -> Dict[str, Any]:
        return self.get_latest_snapshot()["base_height"]

    def reset_vo_base_height_estimator(self, nominal_height_m: Any = None) -> None:
        return None
