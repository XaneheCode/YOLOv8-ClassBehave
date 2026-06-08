from pathlib import Path

from scripts.prepare_yolo_data_yaml import build_yolo_data_yaml, parse_names


def test_parse_names_reads_roboflow_names_line(tmp_path: Path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "nc: 2\n"
        "names: ['sleep', 'upright']\n",
        encoding="utf-8",
    )

    assert parse_names(data_yaml) == ["sleep", "upright"]


def test_build_yolo_data_yaml_uses_absolute_dataset_paths(tmp_path: Path):
    dataset = tmp_path / "Student Behaviour Detection.v6i.yolov8"
    (dataset / "train" / "images").mkdir(parents=True)
    (dataset / "valid" / "images").mkdir(parents=True)
    (dataset / "test" / "images").mkdir(parents=True)
    (dataset / "data.yaml").write_text(
        "train: ../train/images\n"
        "val: ../valid/images\n"
        "test: ../test/images\n"
        "nc: 12\n"
        "names: ['Using_phone', 'bend', 'book', 'bow_head', 'hand-raising', 'phone', "
        "'raise_head', 'reading', 'sleep', 'turn_head', 'upright', 'writing']\n",
        encoding="utf-8",
    )

    content = build_yolo_data_yaml(dataset)

    assert f"train: {str((dataset / 'train' / 'images').resolve()).replace(chr(92), '/')}" in content
    assert f"val: {str((dataset / 'valid' / 'images').resolve()).replace(chr(92), '/')}" in content
    assert f"test: {str((dataset / 'test' / 'images').resolve()).replace(chr(92), '/')}" in content
    assert "nc: 12" in content
    assert "Using_phone" in content
    assert "writing" in content
