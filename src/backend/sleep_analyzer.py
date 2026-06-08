from __future__ import annotations

from src.common.types import AlarmState, Detection


class SleepAnalyzer:
    def __init__(
        self,
        threshold_seconds: float = 3.0,
        min_confidence: float = 0.45,
        wide_person_ratio: float = 1.45,
    ) -> None:
        if threshold_seconds <= 0:
            raise ValueError("threshold_seconds must be positive")
        self.threshold_seconds = threshold_seconds
        self.min_confidence = min_confidence
        self.wide_person_ratio = wide_person_ratio
        self._suspicious_since: float | None = None

    def update(self, detections: list[Detection], now_seconds: float) -> AlarmState:
        suspicious, reason = self._is_suspicious(detections)
        if not suspicious:
            self._suspicious_since = None
            return AlarmState(False, False, 0.0, "normal")

        if self._suspicious_since is None:
            self._suspicious_since = now_seconds
            return AlarmState(False, True, 0.0, reason)

        duration = max(0.0, now_seconds - self._suspicious_since)
        return AlarmState(duration >= self.threshold_seconds, True, duration, reason)

    def _is_suspicious(self, detections: list[Detection]) -> tuple[bool, str]:
        for detection in detections:
            label = detection.label.lower()
            if detection.confidence < self.min_confidence:
                continue

            if label in {"sleep", "desk_sleep", "head_down", "lying"}:
                return True, "sleep_label"

            if label == "person" and detection.height > 0:
                width_height_ratio = detection.width / detection.height
                if width_height_ratio >= self.wide_person_ratio:
                    return True, "wide_person_box"

        return False, "normal"
