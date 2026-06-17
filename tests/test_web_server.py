import numpy as np
from pathlib import Path

from src.common.types import Detection, DetectionAssessment
from src.web.server import (
    assessment_to_payload,
    detection_from_payload,
    detection_to_payload,
    decode_data_url_image,
    encode_frame_data_url,
)


def test_data_url_image_round_trips_png_frame():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = [10, 20, 30]

    data_url = encode_frame_data_url(frame, image_format="png")
    decoded = decode_data_url_image(data_url)

    assert data_url.startswith("data:image/png;base64,")
    assert decoded.shape == frame.shape
    assert decoded[0, 0].tolist() == [10, 20, 30]


def test_assessment_payload_uses_display_label_and_alarm_state():
    assessment = DetectionAssessment(
        detection=Detection("Useing-Phone", 0.91, (10, 20, 80, 140)),
        status="abnormal",
        is_abnormal=True,
        is_alarm=True,
        reason="Useing-Phone",
        duration_seconds=3.25,
    )

    payload = assessment_to_payload(assessment)

    assert payload["label"] == "Useing-Phone"
    assert payload["displayLabel"] == "使用手机"
    assert payload["status"] == "abnormal"
    assert payload["isAlarm"] is True
    assert payload["bbox"] == [10, 20, 80, 140]


def test_detection_payload_round_trips_browser_payload():
    detection = Detection("person", 0.8765, (1, 2, 30, 40))

    payload = detection_to_payload(detection)
    restored = detection_from_payload(payload)

    assert payload["source"] == "yolo"
    assert payload["displayLabel"] == "person"
    assert payload["bbox"] == [1, 2, 30, 40]
    assert restored == Detection("person", 0.8765, (1, 2, 30, 40))


def test_web_sender_exposes_camera_device_selection():
    root = Path(__file__).resolve().parents[1]
    html = (root / "web-dashboard" / "index.html").read_text(encoding="utf-8")
    script = (root / "web-dashboard" / "app.js").read_text(encoding="utf-8")

    assert 'id="cameraSelect"' in html
    assert 'id="refreshCameras"' in html
    assert "enumerateDevices" in script
    assert "deviceId" in script
