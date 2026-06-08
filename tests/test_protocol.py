import socket
import threading

import pytest

from src.common.protocol import HEADER_SIZE, encode_packet, parse_header, recv_packet


def test_encode_parse_header_round_trip():
    image_bytes = b"jpeg-bytes"
    packet = encode_packet(frame_id=7, timestamp_ms=123456789, image_bytes=image_bytes)

    frame_id, timestamp_ms, image_len = parse_header(packet[:HEADER_SIZE])

    assert frame_id == 7
    assert timestamp_ms == 123456789
    assert image_len == len(image_bytes)
    assert packet[HEADER_SIZE:] == image_bytes


def test_parse_header_rejects_invalid_magic():
    packet = bytearray(encode_packet(frame_id=1, timestamp_ms=2, image_bytes=b"x"))
    packet[0:4] = b"BAD!"

    with pytest.raises(ValueError, match="Invalid frame magic"):
        parse_header(bytes(packet[:HEADER_SIZE]))


def test_recv_packet_handles_split_tcp_chunks():
    sender, receiver = socket.socketpair()
    packet = encode_packet(frame_id=3, timestamp_ms=99, image_bytes=b"abcdef")

    def write_chunks():
        try:
            sender.sendall(packet[:2])
            sender.sendall(packet[2:9])
            sender.sendall(packet[9:])
        finally:
            sender.close()

    writer = threading.Thread(target=write_chunks)
    writer.start()

    try:
        result = recv_packet(receiver)
    finally:
        receiver.close()
        writer.join(timeout=2)

    assert result.frame_id == 3
    assert result.timestamp_ms == 99
    assert result.image_bytes == b"abcdef"
