from __future__ import annotations

from src.common.types import AlarmState, Detection, DetectionAssessment


ABNORMAL_LABELS = {"Useing-Phone", "Head-down", "Sleeping"}
NORMAL_LABELS = {"Hand-raise", "Reading", "Writing"}
LABEL_DISPLAY_NAMES = {
    "Hand-raise": "举手",
    "Reading": "看书",
    "Writing": "写字",
    "Useing-Phone": "使用手机",
    "Head-down": "低头",
    "Sleeping": "睡觉",
}

_ABNORMAL_BY_LOWER = {label.lower(): label for label in ABNORMAL_LABELS}
_NORMAL_BY_LOWER = {label.lower(): label for label in NORMAL_LABELS}
_DISPLAY_BY_LOWER = {label.lower(): display for label, display in LABEL_DISPLAY_NAMES.items()}


def display_label(label: str) -> str:
    return _DISPLAY_BY_LOWER.get(label.lower(), label)


class BehaviourAnalyzer:
    def __init__(self, threshold_seconds: float = 3.0, min_confidence: float = 0.35) -> None:
        if threshold_seconds <= 0:
            raise ValueError("threshold_seconds must be positive")
        self.threshold_seconds = threshold_seconds
        self.min_confidence = min_confidence
        self._abnormal_since: dict[str, float] = {}

    def update(
        self,
        detections: list[Detection],
        now_seconds: float,
    ) -> tuple[list[DetectionAssessment], AlarmState]:
        active_labels = self._active_abnormal_labels(detections)
        self._reset_inactive_labels(active_labels)

        for label in active_labels:
            self._abnormal_since.setdefault(label, now_seconds)

        durations = {
            label: round(max(0.0, now_seconds - started_at), 2)
            for label, started_at in self._abnormal_since.items()
            if label in active_labels
        }
        alarm_labels = {label for label, duration in durations.items() if duration >= self.threshold_seconds}
        assessments = [self._assess_detection(detection, durations, alarm_labels) for detection in detections]

        abnormal_assessments = [assessment for assessment in assessments if assessment.is_abnormal]
        abnormal_labels = tuple(dict.fromkeys(assessment.reason for assessment in abnormal_assessments))
        suspicious = bool(abnormal_assessments)
        is_alarm = bool(alarm_labels)
        duration_seconds = max(durations.values(), default=0.0)
        reason = "multi_behaviour_abnormal" if is_alarm else "behaviour_suspicious" if suspicious else "normal"

        return assessments, AlarmState(
            is_alarm=is_alarm,
            suspicious=suspicious,
            duration_seconds=duration_seconds,
            reason=reason,
            abnormal_count=len(abnormal_assessments),
            abnormal_labels=abnormal_labels,
        )

    def _active_abnormal_labels(self, detections: list[Detection]) -> set[str]:
        labels: set[str] = set()
        for detection in detections:
            if detection.confidence < self.min_confidence:
                continue
            canonical = _ABNORMAL_BY_LOWER.get(detection.label.lower())
            if canonical is not None:
                labels.add(canonical)
        return labels

    def _reset_inactive_labels(self, active_labels: set[str]) -> None:
        for label in list(self._abnormal_since):
            if label not in active_labels:
                del self._abnormal_since[label]

    def _assess_detection(
        self,
        detection: Detection,
        durations: dict[str, float],
        alarm_labels: set[str],
    ) -> DetectionAssessment:
        if detection.confidence < self.min_confidence:
            return DetectionAssessment(detection, "ignored", False, False, "low_confidence", 0.0)

        label_lower = detection.label.lower()
        abnormal_label = _ABNORMAL_BY_LOWER.get(label_lower)
        if abnormal_label is not None:
            return DetectionAssessment(
                detection=detection,
                status="abnormal",
                is_abnormal=True,
                is_alarm=abnormal_label in alarm_labels,
                reason=abnormal_label,
                duration_seconds=durations.get(abnormal_label, 0.0),
            )

        normal_label = _NORMAL_BY_LOWER.get(label_lower)
        if normal_label is not None:
            return DetectionAssessment(detection, "normal", False, False, normal_label, 0.0)

        return DetectionAssessment(detection, "unknown", False, False, detection.label, 0.0)
