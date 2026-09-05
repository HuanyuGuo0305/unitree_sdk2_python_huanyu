"""
Minimal, dependency-free NatNet client for OptiTrack Motive.

Verified against the Motive 3.3.4.1 server on this network, which reports
NatNet 4.2.0.0.

Only the part of the stream this repository needs is decoded:

    - rigid-body pose  (position + orientation)
    - marker-set marker positions
    - unlabeled ("other") marker positions

Everything after the rigid-body block (skeletons, assets, labeled markers,
force plates, devices, timing) is ignored, so the parser never has to track
the many per-version changes in those sections.

Frame layout
------------
NatNet 4.1 changed the frame layout: every section is now introduced by BOTH
its element count and its size in bytes,

    int32 count, int32 sizeBytes, <sizeBytes of data>

with the pair present even for empty sections. That is a gift for a partial
parser -- an unrecognised section can be stepped over by its own byte count
instead of being decoded. Older servers wrote bare counts with no sizes, so
both layouts are supported and the client picks whichever actually decodes a
rigid body (see `_decode_frame`).

Transport
---------
Motive streams either to a multicast group or, in Unicast mode, straight back
to the port that sent the connect ping. Which one is a server-side setting the
client cannot see, so it simply listens on BOTH: it joins the multicast group
and also reads frame data from its own command socket. No fallback timer, no
transport guessing.

Coordinate frames
-----------------
Motive streams in whatever up-axis its Streaming pane is set to (Y-up by
default). The client converts positions into a Z-up right-handed world
frame with  C = Rx(+90 deg),  p_W = C p_stream, and rotates orientations on
the LEFT by the same C.

Whether Motive additionally re-expresses the rigid-body's own local frame
when its up axis changes does NOT matter here: any residual constant
right-multiplication is a fixed body-frame rotation, which is exactly what
the mocap->root calibration offset absorbs.

Usage
-----
    client = NatNetClient(local_ip="192.168.0.17", server_ip="192.168.0.118",
                          up_axis="y")
    client.start()
    rb = client.latest_rigid_body(1)
    if rb is not None and rb.tracking_valid:
        print(rb.pos, rb.quat_wxyz)
    client.stop()
"""

from __future__ import annotations

import select
import socket
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# NatNet message ids (NatNetTypes.h).
NAT_CONNECT = 0
NAT_SERVERINFO = 1
NAT_REQUEST = 2
NAT_RESPONSE = 3
NAT_REQUEST_MODELDEF = 4
NAT_MODELDEF = 5
NAT_REQUEST_FRAMEOFDATA = 6
NAT_FRAMEOFDATA = 7
NAT_MESSAGESTRING = 8
NAT_DISCONNECT = 9
NAT_KEEPALIVE = 10
NAT_UNRECOGNIZED_REQUEST = 100

DEFAULT_MULTICAST_GROUP = "239.255.42.99"
DEFAULT_DATA_PORT = 1511
DEFAULT_COMMAND_PORT = 1510

# Sanity limits used to reject a mis-aligned parse before it produces garbage
# poses. A real Motive stream is far below all of these.
_MAX_MARKER_SETS = 512
_MAX_MARKERS = 20000
_MAX_RIGID_BODIES = 512

# Rotation taking each possible Motive up axis onto world +Z, as wxyz
# quaternions. Positions use the matching matrix, so positions and
# orientations can never drift apart.
_SQRT_HALF = float(np.sqrt(0.5))
_UP_AXIS_QUAT = {
    "x": np.array([_SQRT_HALF, 0.0, -_SQRT_HALF, 0.0], dtype=np.float64),  # Ry(-90)
    "y": np.array([_SQRT_HALF, _SQRT_HALF, 0.0, 0.0], dtype=np.float64),   # Rx(+90)
    "z": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),                 # identity
}

# NatNet's UpAxis enum, as returned by the "UpAxis" request.
_UP_AXIS_NAMES = {0: "x", 1: "y", 2: "z"}


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two wxyz quaternions, float64."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


_UP_AXIS_MAT = {k: _quat_to_mat(v) for k, v in _UP_AXIS_QUAT.items()}


def _to_zup_pos(p: np.ndarray, up_axis: str) -> np.ndarray:
    """Rotate one point, or an (N, 3) array of them, into the Z-up world."""
    p = np.asarray(p, dtype=np.float64)
    return p @ _UP_AXIS_MAT[up_axis].T


class RigidBodySample:
    """One rigid body from one NatNet frame, already converted to Z-up."""

    __slots__ = (
        "rb_id",
        "pos",
        "quat_wxyz",
        "mean_marker_error",
        "tracking_valid",
        "frame_number",
        "recv_time",
    )

    def __init__(
        self,
        rb_id: int,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        mean_marker_error: float,
        tracking_valid: bool,
        frame_number: int,
        recv_time: float,
    ) -> None:
        self.rb_id = int(rb_id)
        self.pos = np.asarray(pos, dtype=np.float64).reshape(3)
        self.quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        self.mean_marker_error = float(mean_marker_error)
        self.tracking_valid = bool(tracking_valid)
        self.frame_number = int(frame_number)
        self.recv_time = float(recv_time)

    def age_s(self, now: Optional[float] = None) -> float:
        return (time.monotonic() if now is None else float(now)) - self.recv_time

    def __repr__(self) -> str:
        return (
            f"RigidBodySample(id={self.rb_id}, pos={np.round(self.pos, 4).tolist()}, "
            f"quat={np.round(self.quat_wxyz, 4).tolist()}, valid={self.tracking_valid})"
        )


class NatNetFrame:
    """One decoded NatNet frame of data (only the sections this repo uses)."""

    __slots__ = (
        "frame_number",
        "rigid_bodies",
        "marker_sets",
        "unlabeled_markers",
        "recv_time",
    )

    def __init__(
        self,
        frame_number: int,
        rigid_bodies: Dict[int, RigidBodySample],
        marker_sets: Dict[str, np.ndarray],
        unlabeled_markers: np.ndarray,
        recv_time: float,
    ) -> None:
        self.frame_number = int(frame_number)
        self.rigid_bodies = rigid_bodies
        self.marker_sets = marker_sets
        self.unlabeled_markers = unlabeled_markers
        self.recv_time = float(recv_time)


class _ParseError(Exception):
    pass


def _read_cstring(buf: bytes, off: int) -> Tuple[str, int]:
    end = buf.find(b"\0", off)
    if end < 0:
        raise _ParseError("unterminated string")
    return buf[off:end].decode("utf-8", "replace"), end + 1


def _read_i32(buf: bytes, off: int) -> Tuple[int, int]:
    if off + 4 > len(buf):
        raise _ParseError("truncated int32")
    return int(struct.unpack_from("<i", buf, off)[0]), off + 4


def _read_points(buf: bytes, off: int, n: int) -> Tuple[np.ndarray, int]:
    nbytes = 12 * n
    if off + nbytes > len(buf):
        raise _ParseError("truncated marker block")
    pts = np.frombuffer(buf, dtype="<f4", count=3 * n, offset=off)
    return pts.reshape(n, 3).astype(np.float64), off + nbytes


def _parse_marker_sets(buf: bytes, off: int, count: int) -> Tuple[dict, int]:
    marker_sets = {}
    for _ in range(count):
        name, off = _read_cstring(buf, off)
        n_markers, off = _read_i32(buf, off)
        if not 0 <= n_markers <= _MAX_MARKERS:
            raise _ParseError(f"implausible marker count {n_markers}")
        pts, off = _read_points(buf, off, n_markers)
        marker_sets[name] = pts
    return marker_sets, off


def _parse_rigid_bodies(
    buf: bytes,
    off: int,
    count: int,
    frame_number: int,
    up_axis: str,
    recv_time: float,
) -> Tuple[Dict[int, RigidBodySample], int]:
    rigid_bodies: Dict[int, RigidBodySample] = {}
    for _ in range(count):
        rb_id, off = _read_i32(buf, off)
        if off + 34 > len(buf):
            raise _ParseError("truncated rigid body")
        px, py, pz, qx, qy, qz, qw = struct.unpack_from("<7f", buf, off)
        off += 28
        mean_error = float(struct.unpack_from("<f", buf, off)[0])
        off += 4
        params = int(struct.unpack_from("<h", buf, off)[0])
        off += 2

        quat = np.array([qw, qx, qy, qz], dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        # A mis-aligned parse almost always yields a non-unit quaternion; this
        # is what makes the layout probing in `_decode_frame` reliable.
        if not 0.9 < norm < 1.1:
            raise _ParseError(f"non-unit rigid-body quaternion (norm={norm:.4f})")
        quat /= norm

        pos = _to_zup_pos(np.array([px, py, pz], dtype=np.float64), up_axis)
        quat = _quat_mul(_UP_AXIS_QUAT[up_axis], quat)

        rigid_bodies[int(rb_id)] = RigidBodySample(
            rb_id=rb_id,
            pos=pos,
            quat_wxyz=quat,
            mean_marker_error=mean_error,
            tracking_valid=bool(params & 0x01),
            frame_number=frame_number,
            recv_time=recv_time,
        )
    return rigid_bodies, off


def _read_section_header(buf: bytes, off: int) -> Tuple[int, int, int]:
    """NatNet 4.1+ section header: element count followed by size in bytes."""
    count, off = _read_i32(buf, off)
    size, off = _read_i32(buf, off)
    if not 0 <= count <= _MAX_MARKERS:
        raise _ParseError(f"implausible section count {count}")
    if not 0 <= size <= len(buf) - off:
        raise _ParseError(f"implausible section size {size}")
    return count, size, off


def _parse_frame_sized(payload: bytes, up_axis: str, recv_time: float) -> "NatNetFrame":
    """
    NatNet 4.1+ layout, where each section carries its own byte size.

    Marker data is decoded opportunistically: if a marker section does not
    decode, the section is stepped over by its byte size and the frame is
    still returned, because the rigid-body pose is what callers depend on and
    markers are only drawn as a visual aid.
    """
    off = 0
    frame_number, off = _read_i32(payload, off)

    n_sets, size, off = _read_section_header(payload, off)
    try:
        marker_sets, _ = _parse_marker_sets(payload[off:off + size], 0, n_sets)
    except (_ParseError, struct.error, ValueError):
        marker_sets = {}
    off += size

    n_other, size, off = _read_section_header(payload, off)
    try:
        unlabeled, _ = _read_points(payload[off:off + size], 0, n_other)
    except (_ParseError, struct.error, ValueError):
        unlabeled = np.zeros((0, 3), dtype=np.float64)
    off += size

    n_rb, size, off = _read_section_header(payload, off)
    if not 0 <= n_rb <= _MAX_RIGID_BODIES:
        raise _ParseError(f"implausible rigid-body count {n_rb}")
    rigid_bodies, _ = _parse_rigid_bodies(
        payload[off:off + size], 0, n_rb, frame_number, up_axis, recv_time
    )

    marker_sets = {k: _to_zup_pos(v, up_axis) for k, v in marker_sets.items()}
    unlabeled = _to_zup_pos(unlabeled, up_axis)

    return NatNetFrame(frame_number, rigid_bodies, marker_sets, unlabeled, recv_time)


def _parse_frame_unsized(
    payload: bytes,
    legacy_other_markers: bool,
    up_axis: str,
    recv_time: float,
) -> "NatNetFrame":
    """
    Pre-4.1 layout: bare counts, no section sizes.

    `legacy_other_markers` selects whether the unlabeled-marker count follows
    the marker sets, since that block came and went across older versions.
    """
    off = 0
    frame_number, off = _read_i32(payload, off)

    n_sets, off = _read_i32(payload, off)
    if not 0 <= n_sets <= _MAX_MARKER_SETS:
        raise _ParseError(f"implausible marker-set count {n_sets}")
    marker_sets, off = _parse_marker_sets(payload, off, n_sets)

    unlabeled = np.zeros((0, 3), dtype=np.float64)
    if legacy_other_markers:
        n_other, off = _read_i32(payload, off)
        if not 0 <= n_other <= _MAX_MARKERS:
            raise _ParseError(f"implausible other-marker count {n_other}")
        unlabeled, off = _read_points(payload, off, n_other)

    n_rb, off = _read_i32(payload, off)
    if not 0 <= n_rb <= _MAX_RIGID_BODIES:
        raise _ParseError(f"implausible rigid-body count {n_rb}")
    rigid_bodies, off = _parse_rigid_bodies(
        payload, off, n_rb, frame_number, up_axis, recv_time
    )

    marker_sets = {k: _to_zup_pos(v, up_axis) for k, v in marker_sets.items()}
    unlabeled = _to_zup_pos(unlabeled, up_axis)

    return NatNetFrame(frame_number, rigid_bodies, marker_sets, unlabeled, recv_time)


# Frame layouts probed on startup, best (most rigid bodies) wins.
LAYOUT_SIZED = "sized"
LAYOUT_LEGACY_OTHER = "unsized+other"
LAYOUT_NO_OTHER = "unsized"
_ALL_LAYOUTS = (LAYOUT_SIZED, LAYOUT_LEGACY_OTHER, LAYOUT_NO_OTHER)


def _parse_frame_payload(
    payload: bytes,
    layout: str,
    up_axis: str,
    recv_time: float,
) -> "NatNetFrame":
    if layout == LAYOUT_SIZED:
        return _parse_frame_sized(payload, up_axis, recv_time)
    return _parse_frame_unsized(
        payload, layout == LAYOUT_LEGACY_OTHER, up_axis, recv_time
    )


def _parse_modeldef_rigid_body_names(
    payload: bytes, natnet_major: int, natnet_minor: int
) -> Dict[int, str]:
    """
    Rigid-body id -> name map from a NAT_MODELDEF payload.

    On NatNet 4.1+ every description is prefixed with its byte size, so only
    the rigid-body entries need decoding and everything else is skipped
    wholesale. On older servers there is no size to skip by, so parsing stops
    at the first description type whose layout is not known here.
    """
    names: Dict[int, str] = {}
    sized = (natnet_major, natnet_minor) >= (4, 1)

    off = 0
    n_datasets, off = _read_i32(payload, off)
    if not 0 <= n_datasets <= 1024:
        raise _ParseError(f"implausible dataset count {n_datasets}")

    for _ in range(n_datasets):
        if off >= len(payload):
            break
        dtype, off = _read_i32(payload, off)

        if sized:
            size, off = _read_i32(payload, off)
            if not 0 <= size <= len(payload) - off:
                raise _ParseError(f"implausible description size {size}")
            if dtype == 1:  # rigid body
                body = payload[off:off + size]
                name, inner = _read_cstring(body, 0)
                rb_id, _ = _read_i32(body, inner)
                names[int(rb_id)] = name
            off += size
            continue

        if dtype == 0:  # marker set
            _, off = _read_cstring(payload, off)
            n, off = _read_i32(payload, off)
            for _ in range(n):
                _, off = _read_cstring(payload, off)

        elif dtype == 1:  # rigid body
            name, off = _read_cstring(payload, off)
            rb_id, off = _read_i32(payload, off)
            _parent_id, off = _read_i32(payload, off)
            off += 12  # parent-relative offset xyz
            names[int(rb_id)] = name
            if natnet_major >= 3:
                n_markers, off = _read_i32(payload, off)
                if not 0 <= n_markers <= _MAX_MARKERS:
                    raise _ParseError("implausible rigid-body marker count")
                off += 12 * n_markers  # marker positions
                off += 4 * n_markers  # required active labels
                if natnet_major >= 4:
                    for _ in range(n_markers):
                        _, off = _read_cstring(payload, off)

        else:
            # Without a size prefix the remaining types cannot be stepped over.
            break

    return names


class NatNetClient:
    """
    Background NatNet receiver.

    Runs one thread on the data socket. `latest_frame()` / `latest_rigid_body()`
    return the most recent decoded snapshot under a short lock, so a real-time
    control or render loop never blocks on the network.
    """

    def __init__(
        self,
        server_ip: Optional[str] = None,
        local_ip: str = "",
        multicast_group: str = DEFAULT_MULTICAST_GROUP,
        data_port: int = DEFAULT_DATA_PORT,
        command_port: int = DEFAULT_COMMAND_PORT,
        join_multicast: bool = True,
        up_axis: str = "auto",
        verbose: bool = True,
    ) -> None:
        up_axis = str(up_axis).lower()
        if up_axis not in ("auto", "x", "y", "z"):
            raise ValueError(f"up_axis must be auto/x/y/z, got {up_axis!r}")
        self.up_axis_requested = up_axis

        self.server_ip = server_ip
        self.local_ip = local_ip
        self.multicast_group = multicast_group
        self.data_port = int(data_port)
        self.command_port = int(command_port)
        self.join_multicast = bool(join_multicast)
        # Resolved during the handshake when set to "auto"; Motive's factory
        # default is Y-up, so that is the assumption if the server cannot be
        # asked.
        self.up_axis = "y" if up_axis == "auto" else up_axis
        self.frame_rate: Optional[float] = None
        self.verbose = bool(verbose)

        self.server_info: Dict[str, object] = {}
        self.natnet_major = 4
        self.natnet_minor = 1

        self._rb_names: Dict[int, str] = {}
        self._layout: Optional[str] = None

        self._lock = threading.Lock()
        self._frame: Optional[NatNetFrame] = None
        self._frame_count = 0
        self._parse_errors = 0

        self._data_sock: Optional[socket.socket] = None
        self._cmd_sock: Optional[socket.socket] = None
        self._last_keepalive = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return

        self._handshake()

        # Both sockets are read from: the multicast group carries frame data
        # when Motive is in Multicast mode, and the command socket carries it
        # when Motive is in Unicast mode (it streams back to whichever port
        # sent the connect ping). Listening on both removes the need to know
        # or guess the server's transmission setting.
        if self.join_multicast:
            try:
                self._data_sock = self._open_data_socket()
            except OSError as exc:
                self._log(f"could not join multicast ({exc!r}); unicast only")
                self._data_sock = None

        if self._data_sock is None and self._cmd_sock is None:
            raise RuntimeError("no usable NatNet socket (multicast join and handshake both failed)")

        if self._cmd_sock is not None:
            self._log(
                "also listening for unicast frame data on "
                f"{self._cmd_sock.getsockname()}"
            )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="natnet-client", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for sock in (self._data_sock, self._cmd_sock):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._data_sock = None
        self._cmd_sock = None

    def __enter__(self) -> "NatNetClient":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def latest_frame(self) -> Optional[NatNetFrame]:
        with self._lock:
            return self._frame

    def latest_rigid_body(self, rb_id: int) -> Optional[RigidBodySample]:
        with self._lock:
            frame = self._frame
        if frame is None:
            return None
        return frame.rigid_bodies.get(int(rb_id))

    def rigid_body_ids(self) -> List[int]:
        with self._lock:
            frame = self._frame
        return sorted(frame.rigid_bodies.keys()) if frame is not None else []

    def rigid_body_names(self) -> Dict[int, str]:
        return dict(self._rb_names)

    def rigid_body_id_for_name(self, name: str) -> Optional[int]:
        for rb_id, rb_name in self._rb_names.items():
            if rb_name == name:
                return rb_id
        return None

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def parse_error_count(self) -> int:
        with self._lock:
            return self._parse_errors

    def wait_for_first_frame(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self.latest_frame() is not None:
                return True
            time.sleep(0.02)
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[NATNET] {msg}")

    def _open_data_socket(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 0x100000)

        sock.bind(("", self.data_port))
        mreq = struct.pack(
            "=4s4s",
            socket.inet_aton(self.multicast_group),
            socket.inet_aton(self.local_ip or "0.0.0.0"),
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        self._log(
            f"listening multicast {self.multicast_group}:{self.data_port} "
            f"on interface {self.local_ip or '0.0.0.0'}"
        )
        sock.setblocking(False)
        return sock

    def _handshake(self) -> None:
        """
        Ping the command port for server info and rigid-body names.

        Purely informational in multicast mode -- Motive streams regardless --
        so every failure here is logged and ignored. In unicast mode the ping
        is what makes Motive start sending to this client, so a failure is
        reported more loudly.
        """
        try:
            self._cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._cmd_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            self._cmd_sock.bind((self.local_ip or "", 0))
            self._cmd_sock.settimeout(1.0)

            targets = (
                [self.server_ip]
                if self.server_ip
                else [self._broadcast_address()]
            )
            for target in targets:
                if target is None:
                    continue
                self._cmd_sock.sendto(
                    self._build_command(NAT_CONNECT, "Ping"),
                    (target, self.command_port),
                )

            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not self.server_info:
                try:
                    data, addr = self._cmd_sock.recvfrom(65535)
                except socket.timeout:
                    break
                msg_id, payload = self._split_packet(data)
                if msg_id == NAT_SERVERINFO:
                    self._handle_server_info(payload, addr[0])

            if self.server_info:
                self._query_server_settings()
                self._request_model_definitions()
            else:
                self._log(
                    "no NAT_SERVERINFO reply; continuing without server info "
                    "(harmless in multicast mode)"
                )

        except OSError as exc:
            self._log(f"handshake failed ({exc!r}); continuing without it")

    def _broadcast_address(self) -> Optional[str]:
        if not self.local_ip:
            return "255.255.255.255"
        parts = self.local_ip.split(".")
        if len(parts) != 4:
            return "255.255.255.255"
        return ".".join(parts[:3] + ["255"])

    @staticmethod
    def _build_command(command_id: int, command_str: str = "") -> bytes:
        body = command_str.encode("utf-8") + b"\0" if command_str else b""
        return struct.pack("<HH", command_id, len(body)) + body

    @staticmethod
    def _split_packet(data: bytes) -> Tuple[int, bytes]:
        if len(data) < 4:
            return -1, b""
        msg_id, n_bytes = struct.unpack_from("<HH", data, 0)
        return int(msg_id), data[4 : 4 + n_bytes] if n_bytes else data[4:]

    def _handle_server_info(self, payload: bytes, from_ip: str) -> None:
        try:
            name = payload[:256].split(b"\0", 1)[0].decode("utf-8", "replace")
            app_version = tuple(payload[256:260])
            natnet_version = tuple(payload[260:264])
        except Exception as exc:  # noqa: BLE001 - informational only
            self._log(f"could not decode server info: {exc!r}")
            return

        if self.server_ip is None:
            self.server_ip = from_ip

        self.natnet_major = int(natnet_version[0]) or self.natnet_major
        self.natnet_minor = int(natnet_version[1])
        self.server_info = {
            "name": name,
            "app_version": app_version,
            "natnet_version": natnet_version,
            "ip": from_ip,
        }
        self._log(
            f"server {name!r} at {from_ip}, app "
            f"{'.'.join(str(v) for v in app_version)}, NatNet "
            f"{'.'.join(str(v) for v in natnet_version)}"
        )

    def _request_int_setting(self, name: str) -> Optional[bytes]:
        """One NAT_REQUEST round trip. Returns the raw 4-byte reply, or None."""
        if self._cmd_sock is None or not self.server_ip:
            return None
        try:
            self._cmd_sock.sendto(
                self._build_command(NAT_REQUEST, name), (self.server_ip, self.command_port)
            )
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                data, _ = self._cmd_sock.recvfrom(65535)
                msg_id, payload = self._split_packet(data)
                if msg_id == NAT_RESPONSE and len(payload) == 4:
                    return payload
        except (OSError, struct.error):
            pass
        return None

    def _query_server_settings(self) -> None:
        """
        Ask Motive for its up axis and frame rate.

        The up axis is a Streaming-pane setting with no visible effect on the
        wire -- a Y-up and a Z-up stream are both just three floats -- so
        getting it wrong silently rotates the whole world by 90 degrees.
        Asking the server removes that failure mode entirely.
        """
        reply = self._request_int_setting("FrameRate")
        if reply is not None:
            self.frame_rate = float(struct.unpack("<f", reply)[0])

        reply = self._request_int_setting("UpAxis")
        resolved = (
            _UP_AXIS_NAMES.get(int(struct.unpack("<i", reply)[0]))
            if reply is not None
            else None
        )

        if self.up_axis_requested == "auto":
            if resolved is None:
                self._log(
                    "server did not answer the UpAxis query; assuming Y-up "
                    "(Motive's default). Set mocap_up_axis explicitly if the "
                    "robot renders lying on its side."
                )
            else:
                self.up_axis = resolved
        elif resolved is not None and resolved != self.up_axis:
            self._log(
                f"[WARN] configured up axis {self.up_axis!r} disagrees with the "
                f"server's {resolved!r}; using the configured value"
            )

        rate = f"{self.frame_rate:g} Hz" if self.frame_rate else "unknown rate"
        self._log(f"up axis {self.up_axis!r}, {rate}")

    def _request_model_definitions(self) -> None:
        if self._cmd_sock is None or not self.server_ip:
            return
        try:
            self._cmd_sock.sendto(
                self._build_command(NAT_REQUEST_MODELDEF),
                (self.server_ip, self.command_port),
            )
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    data, _ = self._cmd_sock.recvfrom(65535)
                except socket.timeout:
                    return
                msg_id, payload = self._split_packet(data)
                if msg_id != NAT_MODELDEF:
                    continue
                self._rb_names = _parse_modeldef_rigid_body_names(
                    payload, self.natnet_major, self.natnet_minor
                )
                if self._rb_names:
                    pretty = ", ".join(
                        f"{i}:{n!r}" for i, n in sorted(self._rb_names.items())
                    )
                    self._log(f"rigid bodies: {pretty}")
                return
        except (OSError, _ParseError, struct.error, IndexError) as exc:
            self._log(f"model definition request failed ({exc!r}); ids only")

    def _run(self) -> None:
        socks = [s for s in (self._data_sock, self._cmd_sock) if s is not None]
        for sock in socks:
            sock.setblocking(False)

        while not self._stop_event.is_set():
            self._send_keepalive()

            try:
                ready, _, _ = select.select(socks, [], [], 0.2)
            except (OSError, ValueError):
                return

            for sock in ready:
                while True:
                    try:
                        data, _ = sock.recvfrom(65535)
                    except BlockingIOError:
                        break
                    except OSError:
                        return

                    msg_id, payload = self._split_packet(data)
                    if msg_id != NAT_FRAMEOFDATA:
                        continue

                    frame = self._decode_frame(payload)
                    with self._lock:
                        if frame is None:
                            self._parse_errors += 1
                        else:
                            self._frame = frame
                            self._frame_count += 1

    def _send_keepalive(self) -> None:
        """
        Unicast streaming is a subscription: Motive stops sending if the
        client goes quiet, so refresh it once a second. Harmless in
        multicast mode.
        """
        if self._cmd_sock is None or not self.server_ip:
            return
        now = time.monotonic()
        if now - self._last_keepalive < 1.0:
            return
        self._last_keepalive = now
        try:
            self._cmd_sock.sendto(
                self._build_command(NAT_KEEPALIVE), (self.server_ip, self.command_port)
            )
        except OSError:
            pass

    def _decode_frame(self, payload: bytes) -> Optional[NatNetFrame]:
        recv_time = time.monotonic()

        # Once a layout has proven itself, keep using it so the per-frame cost
        # stays a single parse.
        layouts = (self._layout,) if self._layout is not None else _ALL_LAYOUTS

        # Several layouts can "succeed" on the same packet: reading the wrong
        # one usually lands on a later zero word and yields an empty but
        # structurally valid frame. Whichever layout recovers MORE rigid bodies
        # is the correct one -- so decide on that, and do not commit to a
        # layout until a frame with an actual rigid body in it has been seen.
        best_layout: Optional[str] = None
        best_frame: Optional[NatNetFrame] = None

        for layout in layouts:
            try:
                frame = _parse_frame_payload(payload, layout, self.up_axis, recv_time)
            except (_ParseError, struct.error, ValueError, IndexError):
                continue

            if best_frame is None or len(frame.rigid_bodies) > len(best_frame.rigid_bodies):
                best_layout, best_frame = layout, frame

        if best_frame is None:
            return None

        if self._layout is None and best_frame.rigid_bodies:
            self._layout = best_layout
            self._log(f"frame layout: {best_layout}")

        return best_frame


def main() -> None:
    """`python3 -m utils.natnet_client --local-ip 192.168.0.17` stream dump."""
    import argparse

    parser = argparse.ArgumentParser(description="NatNet stream monitor")
    parser.add_argument("--local-ip", default="", help="NIC facing the mocap net")
    parser.add_argument("--server-ip", default=None, help="Motive host (optional)")
    parser.add_argument("--multicast-group", default=DEFAULT_MULTICAST_GROUP)
    parser.add_argument("--data-port", type=int, default=DEFAULT_DATA_PORT)
    parser.add_argument("--command-port", type=int, default=DEFAULT_COMMAND_PORT)
    parser.add_argument("--no-multicast", action="store_true",
                        help="skip the multicast join; unicast (command socket) only")
    parser.add_argument("--up-axis", default="auto", choices=["auto", "x", "y", "z"])
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    client = NatNetClient(
        server_ip=args.server_ip,
        local_ip=args.local_ip,
        multicast_group=args.multicast_group,
        data_port=args.data_port,
        command_port=args.command_port,
        join_multicast=not args.no_multicast,
        up_axis=args.up_axis,
    )
    client.start()
    try:
        if not client.wait_for_first_frame(timeout_s=5.0):
            print("[NATNET] no frames received -- check Motive streaming settings")
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            frame = client.latest_frame()
            if frame is not None:
                print(f"--- frame {frame.frame_number} ---")
                for rb_id in sorted(frame.rigid_bodies):
                    print("   ", frame.rigid_bodies[rb_id])
                for name, pts in frame.marker_sets.items():
                    print(f"    markerset {name!r}: {len(pts)} markers")
                print(f"    unlabeled: {len(frame.unlabeled_markers)}")
            time.sleep(0.5)
    finally:
        client.stop()


if __name__ == "__main__":
    main()
