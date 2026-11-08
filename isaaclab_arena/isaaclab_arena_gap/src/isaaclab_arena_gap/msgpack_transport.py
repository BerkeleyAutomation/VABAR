"""Arena-independent transport for the GaP observation/action protocol."""

from __future__ import annotations

import math
import socket
import struct
import time
from typing import Any

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

_LEN = struct.Struct(">I")
_DEFAULT_MAX_FRAME_BYTES = 64 * 1024 * 1024


class GapProtocolError(ConnectionError):
    """The GaP peer sent or received a malformed protocol frame."""


class GapMsgpackClient:
    """Length-prefixed msgpack client for GaP's local request/reply server."""

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout_s: float = 90.0,
        io_timeout_s: float | None = 10.0,
        max_frame_bytes: int = _DEFAULT_MAX_FRAME_BYTES,
    ) -> None:
        if not math.isfinite(connect_timeout_s) or connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be positive")
        if io_timeout_s is not None and (
            not math.isfinite(io_timeout_s) or io_timeout_s <= 0
        ):
            raise ValueError("io_timeout_s must be positive or None")
        if max_frame_bytes <= 0:
            raise ValueError("max_frame_bytes must be positive")

        self._host = host
        self._port = port
        self._connect_timeout_s = float(connect_timeout_s)
        self._io_timeout_s = io_timeout_s
        self._max_frame_bytes = int(max_frame_bytes)
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        """Connect within one wall-clock deadline, retrying while GaP starts."""
        if self._sock is not None:
            return

        deadline = time.monotonic() + self._connect_timeout_s
        last_err: OSError | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock = None
            try:
                sock = socket.create_connection(
                    (self._host, self._port),
                    timeout=min(10.0, remaining),
                )
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(self._io_timeout_s)
                self._sock = sock
                return
            except OSError as exc:
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
                last_err = exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(1.0, remaining))

        raise ConnectionError(
            f"GapMsgpackClient: could not connect to GaP server {self._host}:{self._port} "
            f"within {self._connect_timeout_s:g}s. Last error: {last_err}"
        )

    def _connected_socket(self) -> socket.socket:
        if self._sock is None:
            raise ConnectionError("GapMsgpackClient: connect() must be called first")
        return self._sock

    def _recv_exactly(self, size: int) -> bytes:
        sock = self._connected_socket()
        buf = bytearray()
        while len(buf) < size:
            chunk = sock.recv(size - len(buf))
            if not chunk:
                raise ConnectionError(
                    "GapMsgpackClient: GaP server closed the connection"
                )
            buf.extend(chunk)
        return bytes(buf)

    def send_obs(self, frame: dict[str, Any]) -> None:
        payload = msgpack.packb(frame, use_bin_type=True)
        if len(payload) > self._max_frame_bytes:
            raise GapProtocolError(
                f"GaP observation frame is {len(payload)} bytes; "
                f"limit is {self._max_frame_bytes}"
            )
        self._connected_socket().sendall(_LEN.pack(len(payload)) + payload)

    def recv_action(self) -> dict[str, Any]:
        (size,) = _LEN.unpack(self._recv_exactly(_LEN.size))
        if size > self._max_frame_bytes:
            raise GapProtocolError(
                f"GaP action frame declares {size} bytes; limit is {self._max_frame_bytes}"
            )
        payload = self._recv_exactly(size)
        try:
            action = msgpack.unpackb(payload, raw=False)
        except (ValueError, msgpack.exceptions.UnpackException) as exc:
            raise GapProtocolError("GaP action frame contains invalid msgpack") from exc
        if not isinstance(action, dict):
            raise GapProtocolError(
                f"GaP action frame must decode to a mapping, got {type(action).__name__}"
            )
        return action

    def round_trip(self, frame: dict[str, Any]) -> dict[str, Any]:
        t0 = time.perf_counter()
        self.send_obs(frame)
        t1 = time.perf_counter()
        action = self.recv_action()
        # Perf-trace observability only (read by GapRemotePolicy when
        # CAP_PERF_TRACE_DIR is set): send = serialize+write, recv = GaP
        # think time + transit.
        self.last_send_s = t1 - t0
        self.last_recv_s = time.perf_counter() - t1
        return action

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
