import pytest

from src.common.protocol import HEADER_SIZE, encode_packet, parse_header, recv_packet


class ChunkedConnection:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def recv(self, size):
        if not self.chunks:
            return b""
        chunk = self.chunks.pop(0)
        if len(chunk) > size:
            self.chunks.insert(0, chunk[size:])
            return chunk[:size]
        return chunk


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
    packet = encode_packet(frame_id=3, timestamp_ms=99, image_bytes=b"abcdef")
    receiver = ChunkedConnection([packet[:2], packet[2:9], packet[9:]])

    result = recv_packet(receiver)

    assert result.frame_id == 3
    assert result.timestamp_ms == 99
    assert result.image_bytes == b"abcdef"
