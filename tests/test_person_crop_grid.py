import numpy as np

from src.backend.person_crop_grid import build_person_crop_grid
from src.common.types import Detection


def test_build_person_crop_grid_numbers_crops_and_preserves_source_boxes():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    detections = [
        Detection(label="person", confidence=0.91, bbox=(10, 20, 30, 60)),
        Detection(label="person", confidence=0.82, bbox=(70, 10, 100, 50)),
    ]

    grid = build_person_crop_grid(frame, detections, tile_size=(64, 64), columns=2, padding_ratio=0.1)

    assert grid.image.shape == (64, 128, 3)
    assert [crop.person_id for crop in grid.crops] == [1, 2]
    assert [crop.detection.bbox for crop in grid.crops] == [(10, 20, 30, 60), (70, 10, 100, 50)]
    assert grid.source_by_id[1].bbox == (10, 20, 30, 60)
    assert grid.source_by_id[2].bbox == (70, 10, 100, 50)
    assert int(grid.image.sum()) > 0


def test_build_person_crop_grid_limits_people_and_expands_clipped_boxes():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    detections = [
        Detection(label="person", confidence=0.91, bbox=(0, 0, 20, 20)),
        Detection(label="person", confidence=0.82, bbox=(50, 50, 79, 79)),
    ]

    grid = build_person_crop_grid(frame, detections, tile_size=(32, 32), columns=1, padding_ratio=0.5, max_people=1)

    assert grid.image.shape == (32, 32, 3)
    assert len(grid.crops) == 1
    assert grid.crops[0].crop_bbox == (0, 0, 30, 30)
