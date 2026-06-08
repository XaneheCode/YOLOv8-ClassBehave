from pathlib import Path

import pytest

from scripts.offline_test_images import classify_behaviour_label, iter_images, write_prediction_rows


def test_iter_images_returns_sorted_images_with_limit(tmp_path: Path):
    (tmp_path / "b.jpg").write_bytes(b"b")
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "note.txt").write_text("ignored", encoding="utf-8")

    images = iter_images(tmp_path, limit=1)

    assert [path.name for path in images] == ["a.png"]


def test_iter_images_rejects_empty_directory(tmp_path: Path):
    with pytest.raises(SystemExit, match="No images"):
        iter_images(tmp_path, limit=0)


def test_write_prediction_rows_creates_csv(tmp_path: Path):
    csv_path = tmp_path / "predictions.csv"

    write_prediction_rows(
        csv_path,
        [
            {
                "image": "sample.jpg",
                "label": "sleep",
                "confidence": "0.9000",
                "x1": "1.0",
                "y1": "2.0",
                "x2": "3.0",
                "y2": "4.0",
                "behaviour_status": "abnormal",
                "abnormal_candidate": "true",
            }
        ],
    )

    text = csv_path.read_text(encoding="utf-8")

    assert "sample.jpg,sleep,0.9000" in text
    assert "abnormal,true" in text


def test_classify_behaviour_label_uses_course_rules():
    assert classify_behaviour_label("sleep") == ("abnormal", True)
    assert classify_behaviour_label("Using_phone") == ("abnormal", True)
    assert classify_behaviour_label("upright") == ("normal", False)
    assert classify_behaviour_label("unknown") == ("unknown", False)
