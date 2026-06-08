from __future__ import annotations

import socket
import struct
from dataclasses import dataclass


MAGIC = b"NSGD"
HEADER_FORMAT = "!4sIQI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp_ms: int
    image_bytes: bytes


def encode_packet(frame_id: int, timestamp_ms: int, image_bytes: bytes) -> bytes:
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative")
    if timestamp_ms < 0:
        raise ValueError("timestamp_ms must be non-negative")
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"image_bytes exceeds {MAX_IMAGE_BYTES} bytes")

    header = struct.pack(HEADER_FORMAT, MAGIC, frame_id, timestamp_ms, len(image_bytes))
    return header + image_bytes


def parse_header(header_bytes: bytes) -> tuple[int, int, int]:
    if len(header_bytes) != HEADER_SIZE:
        raise ValueError(f"header must be {HEADER_SIZE} bytes")

    magic, frame_id, timestamp_ms, image_len = struct.unpack(HEADER_FORMAT, header_bytes)
    if magic != MAGIC:
        raise ValueError("Invalid frame magic")
    if image_len <= 0 or image_len > MAX_IMAGE_BYTES:
        raise ValueError(f"Invalid image length: {image_len}")
    return frame_id, timestamp_ms, image_len


def recvall(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = conn.recv(remaining)
        if chunk == b"":
            raise ConnectionError("Socket closed before enough bytes were received")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_packet(conn: socket.socket, frame_id: int, timestamp_ms: int, image_bytes: bytes) -> None:
    conn.sendall(encode_packet(frame_id, timestamp_ms, image_bytes))


def recv_packet(conn: socket.socket) -> FramePacket:
    header = recvall(conn, HEADER_SIZE)
    frame_id, timestamp_ms, image_len = parse_header(header)
    image_bytes = recvall(conn, image_len)
    return FramePacket(frame_id=frame_id, timestamp_ms=timestamp_ms, image_bytes=image_bytes)
