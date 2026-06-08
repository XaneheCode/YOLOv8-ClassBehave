import numpy as np

from src.common.image_codec import decode_jpeg, encode_jpeg, resize_to_width


def test_encode_decode_jpeg_round_trip_shape():
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    frame[:, :, 1] = 180

    data = encode_jpeg(frame, quality=80)
    decoded = decode_jpeg(data)

    assert isinstance(data, bytes)
    assert decoded.shape == frame.shape
    assert decoded.dtype == np.uint8


def test_resize_to_width_keeps_aspect_ratio():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = resize_to_width(frame, target_width=50)

    assert resized.shape == (25, 50, 3)
