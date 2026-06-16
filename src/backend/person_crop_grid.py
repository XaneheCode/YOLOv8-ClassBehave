from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import cv2
import numpy as np

from src.common.types import Detection


@dataclass(frozen=True)
class PersonCrop:
    person_id: int
    detection: Detection
    crop_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PersonCropGrid:
    image: np.ndarray
    crops: list[PersonCrop]
    source_by_id: dict[int, Detection]


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(round(width * padding_ratio))
    pad_y = int(round(height * padding_ratio))

    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(image_width - 1, x2 + pad_x),
        min(image_height - 1, y2 + pad_y),
    )


def _fit_crop_to_tile(crop: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
    tile = np.full((tile_height, tile_width, 3), 245, dtype=np.uint8)
    crop_height, crop_width = crop.shape[:2]
    if crop_width <= 0 or crop_height <= 0:
        return tile

    scale = min(tile_width / crop_width, tile_height / crop_height)
    resized_width = max(1, int(round(crop_width * scale)))
    resized_height = max(1, int(round(crop_height * scale)))
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    offset_x = (tile_width - resized_width) // 2
    offset_y = (tile_height - resized_height) // 2
    tile[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return tile


def _draw_person_id(tile: np.ndarray, person_id: int) -> None:
    text = str(person_id)
    cv2.rectangle(tile, (0, 0), (38, 28), (0, 102, 204), -1)
    cv2.putText(tile, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def build_person_crop_grid(
    frame: np.ndarray,
    detections: list[Detection],
    *,
    tile_size: tuple[int, int] = (224, 224),
    columns: int = 4,
    padding_ratio: float = 0.2,
    max_people: int = 30,
) -> PersonCropGrid:
    tile_width, tile_height = tile_size
    selected = detections[:max_people]
    if not selected:
        return PersonCropGrid(
            image=np.full((tile_height, tile_width, 3), 245, dtype=np.uint8),
            crops=[],
            source_by_id={},
        )

    height, width = frame.shape[:2]
    columns = max(1, columns)
    rows = ceil(len(selected) / columns)
    grid = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    crops: list[PersonCrop] = []

    for index, detection in enumerate(selected, start=1):
        crop_bbox = expand_bbox(detection.bbox, image_width=width, image_height=height, padding_ratio=padding_ratio)
        x1, y1, x2, y2 = crop_bbox
        crop = frame[y1:y2, x1:x2]
        tile = _fit_crop_to_tile(crop, tile_width=tile_width, tile_height=tile_height)
        _draw_person_id(tile, index)

        row = (index - 1) // columns
        col = (index - 1) % columns
        grid[
            row * tile_height : (row + 1) * tile_height,
            col * tile_width : (col + 1) * tile_width,
        ] = tile
        cv2.rectangle(
            grid,
            (col * tile_width, row * tile_height),
            ((col + 1) * tile_width - 1, (row + 1) * tile_height - 1),
            (180, 180, 180),
            1,
        )
        crops.append(PersonCrop(person_id=index, detection=detection, crop_bbox=crop_bbox))

    return PersonCropGrid(
        image=grid,
        crops=crops,
        source_by_id={crop.person_id: crop.detection for crop in crops},
    )
