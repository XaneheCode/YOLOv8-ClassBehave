from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])


@dataclass(frozen=True)
class AlarmState:
    is_alarm: bool
    suspicious: bool
    duration_seconds: float
    reason: str
    abnormal_count: int = 0
    abnormal_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectionAssessment:
    detection: Detection
    status: str
    is_abnormal: bool
    is_alarm: bool
    reason: str
    duration_seconds: float
