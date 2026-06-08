from src.backend.sleep_analyzer import SleepAnalyzer
from src.common.types import Detection


def test_explicit_sleep_label_alarms_after_threshold():
    analyzer = SleepAnalyzer(threshold_seconds=3.0)
    detections = [Detection(label="sleep", confidence=0.91, bbox=(10, 10, 80, 50))]

    first = analyzer.update(detections, now_seconds=10.0)
    second = analyzer.update(detections, now_seconds=12.0)
    third = analyzer.update(detections, now_seconds=13.1)

    assert first.suspicious is True
    assert first.is_alarm is False
    assert second.is_alarm is False
    assert third.is_alarm is True
    assert third.reason == "sleep_label"


def test_normal_person_does_not_alarm():
    analyzer = SleepAnalyzer(threshold_seconds=3.0)
    detections = [Detection(label="person", confidence=0.88, bbox=(10, 10, 60, 180))]

    result = analyzer.update(detections, now_seconds=20.0)

    assert result.suspicious is False
    assert result.is_alarm is False
    assert result.reason == "normal"


def test_wide_person_box_is_suspicious_for_desk_sleep():
    analyzer = SleepAnalyzer(threshold_seconds=1.0, wide_person_ratio=1.3)
    detections = [Detection(label="person", confidence=0.9, bbox=(10, 10, 160, 90))]

    analyzer.update(detections, now_seconds=1.0)
    result = analyzer.update(detections, now_seconds=2.2)

    assert result.is_alarm is True
    assert result.reason == "wide_person_box"


def test_normal_frame_resets_suspicious_timer():
    analyzer = SleepAnalyzer(threshold_seconds=3.0)
    sleep = [Detection(label="sleep", confidence=0.91, bbox=(10, 10, 80, 50))]
    normal = [Detection(label="person", confidence=0.91, bbox=(10, 10, 60, 180))]

    analyzer.update(sleep, now_seconds=1.0)
    reset = analyzer.update(normal, now_seconds=2.0)
    later = analyzer.update(sleep, now_seconds=5.0)

    assert reset.suspicious is False
    assert later.is_alarm is False
    assert later.duration_seconds == 0.0
