from __future__ import annotations

import socket
import struct
import threading
import time
import zlib
from typing import Optional

import numpy as np

from perception.data_types import DepthFrame


class DepthCamera:
    """
    CRL-side receiver for raw D430I depth streamed from .164.

    Only the latest frame is kept.
    get_latest() is non-blocking.
    """

    MAGIC = b"DPT1"
    HEADER_STRUCT = struct.Struct("!4sIHHIf")

    def __init__(
        self,
        host: str = "192.168.123.164",
        port: int = 50010,
    ):
        self.host = host
        self.port = int(port)

        self._latest: Optional[DepthFrame] = None
        self._lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        self.num_ok = 0
        self.num_errors = 0
        self.num_dropped = 0
        self._last_sequence: Optional[int] = None

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="DepthCameraThread",
            daemon=True,
        )
        self._thread.start()

        print(
            f"[DepthCamera] started receiver "
            f"{self.host}:{self.port}"
        )

    @staticmethod
    def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
        chunks = []
        remaining = nbytes

        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Depth stream connection closed")

            chunks.append(chunk)
            remaining -= len(chunk)

        return b"".join(chunks)

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(2.0)

        print(
            f"[DepthCamera] connecting to "
            f"{self.host}:{self.port} ..."
        )
        sock.connect((self.host, self.port))
        print("[DepthCamera] connected")

        return sock

    def _receive_stream(self, sock: socket.socket) -> None:
        while self._running:
            header = self._recv_exact(
                sock,
                self.HEADER_STRUCT.size,
            )

            (
                magic,
                sequence,
                height,
                width,
                payload_size,
                depth_scale,
            ) = self.HEADER_STRUCT.unpack(header)

            if magic != self.MAGIC:
                raise RuntimeError(f"Bad depth magic: {magic}")

            if payload_size <= 0 or payload_size > 10_000_000:
                raise RuntimeError(
                    f"Bad depth payload size: {payload_size}"
                )

            payload = self._recv_exact(sock, payload_size)
            raw = zlib.decompress(payload)

            expected = int(height) * int(width) * 2
            if len(raw) != expected:
                raise RuntimeError(
                    f"Bad depth raw size: {len(raw)} != {expected}"
                )

            depth = np.frombuffer(
                raw,
                dtype=np.uint16,
            ).reshape(
                int(height),
                int(width),
            ).copy()

            if self._last_sequence is not None:
                expected_seq = (self._last_sequence + 1) & 0xFFFFFFFF
                if sequence != expected_seq:
                    self.num_dropped += (
                        sequence - expected_seq
                    ) & 0xFFFFFFFF

            self._last_sequence = int(sequence)

            frame = DepthFrame(
                image_raw=depth,
                timestamp_s=time.monotonic(),
                depth_scale=float(depth_scale),
                sequence=int(sequence),
            )

            with self._lock:
                self._latest = frame

            self.num_ok += 1

    def _run(self) -> None:
        while self._running:
            try:
                sock = self._connect()
                self._sock = sock
                self._receive_stream(sock)

            except (
                ConnectionError,
                ConnectionRefusedError,
                socket.timeout,
                OSError,
                RuntimeError,
                zlib.error,
            ) as exc:
                if self._running:
                    self.num_errors += 1
                    print(f"[DepthCamera] stream error: {exc}")
                    time.sleep(1.0)

            finally:
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                self._sock = None

    def get_latest(self) -> Optional[DepthFrame]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._running = False

        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

            try:
                self._sock.close()
            except OSError:
                pass

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        self._thread = None
        self._sock = None

        print(
            "[DepthCamera] stopped | "
            f"ok={self.num_ok}, "
            f"errors={self.num_errors}, "
            f"dropped={self.num_dropped}"
        )
