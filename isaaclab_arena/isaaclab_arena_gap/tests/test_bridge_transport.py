"""Public wire contract between the Arena policy and the GaP process."""

from __future__ import annotations

import struct

import msgpack
import numpy as np
import pytest
from isaaclab_arena_gap.msgpack_transport import GapMsgpackClient, GapProtocolError


class _Socket:
    def __init__(self, incoming: bytes = b"") -> None:
        self.incoming = bytearray(incoming)
        self.sent = bytearray()
        self.closed = False

    def recv(self, size: int) -> bytes:
        count = min(size, 3, len(self.incoming))
        if count == 0:
            return b""
        value = bytes(self.incoming[:count])
        del self.incoming[:count]
        return value

    def sendall(self, value: bytes) -> None:
        self.sent.extend(value)

    def close(self) -> None:
        self.closed = True


def _frame(value) -> bytes:
    payload = msgpack.packb(value, use_bin_type=True)
    return struct.pack(">I", len(payload)) + payload


def test_one_observation_action_round_trip_preserves_numpy_payloads() -> None:
    action = {
        "left": {
            "joint_pos": np.arange(7, dtype=np.float32),
            "gripper": np.float32(0.25),
        }
    }
    socket = _Socket(_frame(action))
    client = GapMsgpackClient("127.0.0.1", 9000)
    client._sock = socket

    observation = {"rgb": np.zeros((2, 3, 3), dtype=np.uint8)}
    received = client.round_trip(observation)

    assert np.array_equal(received["left"]["joint_pos"], action["left"]["joint_pos"])
    (size,) = struct.unpack(">I", socket.sent[:4])
    assert size == len(socket.sent) - 4
    sent = msgpack.unpackb(socket.sent[4:], raw=False)
    assert np.array_equal(sent["rgb"], observation["rgb"])


@pytest.mark.parametrize(
    "incoming, message",
    [
        (struct.pack(">I", 9), "declares 9 bytes"),
        (_frame([1, 2, 3]), "must decode to a mapping"),
        (struct.pack(">I", 1) + b"\xc1", "invalid msgpack"),
    ],
)
def test_malformed_or_oversized_actions_fail_closed(
    incoming: bytes, message: str
) -> None:
    client = GapMsgpackClient("127.0.0.1", 9000, max_frame_bytes=8)
    client._sock = _Socket(incoming)

    with pytest.raises(GapProtocolError, match=message):
        client.recv_action()


def test_transport_requires_connection_and_finite_positive_limits() -> None:
    client = GapMsgpackClient("127.0.0.1", 9000)
    with pytest.raises(ConnectionError, match=r"connect\(\)"):
        client.send_obs({})
    for kwargs in (
        {"connect_timeout_s": 0},
        {"connect_timeout_s": float("nan")},
        {"io_timeout_s": float("inf")},
        {"max_frame_bytes": 0},
    ):
        with pytest.raises(ValueError):
            GapMsgpackClient("127.0.0.1", 9000, **kwargs)


def test_peer_close_is_reported_as_a_connection_failure() -> None:
    client = GapMsgpackClient("127.0.0.1", 9000)
    client._sock = _Socket()

    with pytest.raises(ConnectionError, match="closed the connection"):
        client.recv_action()
