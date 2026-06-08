from pathlib import Path

import pytest

from scripts.download_roboflow_dataset import api_url, require_api_key, verify_yolov8_dataset


def test_api_url_uses_student_activity_defaults_shape():
    url = api_url("studentactivity", "new-student-classroom-activity-2", 2, "yolov8")

    assert url == "https://api.roboflow.com/studentactivity/new-student-classroom-activity-2/2/yolov8"


def test_require_api_key_exits_with_clear_message():
    with pytest.raises(SystemExit, match="ROBOFLOW_API_KEY"):
        require_api_key(None)


def test_verify_yolov8_dataset_accepts_test_images(tmp_path: Path):
    dataset = tmp_path / "dataset"
    images = dataset / "test" / "images"
    images.mkdir(parents=True)
    (dataset / "data.yaml").write_text("names: ['phone', 'sleep', 'study']\n", encoding="utf-8")
    (images / "sample.jpg").write_bytes(b"fake")

    verify_yolov8_dataset(dataset)


def test_verify_yolov8_dataset_rejects_missing_test_images(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("names: []\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="test"):
        verify_yolov8_dataset(dataset)

