from src.backend.behaviour_analyzer import ABNORMAL_LABELS, NORMAL_LABELS, BehaviourAnalyzer
from src.common.types import Detection


def test_abnormal_and_normal_detections_are_assessed_independently():
    analyzer = BehaviourAnalyzer(threshold_seconds=3.0, min_confidence=0.35)
    detections = [
        Detection(label="Useing-Phone", confidence=0.91, bbox=(10, 10, 80, 60)),
        Detection(label="Hand-raise", confidence=0.88, bbox=(90, 10, 140, 120)),
    ]

    assessments, alarm = analyzer.update(detections, now_seconds=10.0)

    assert assessments[0].is_abnormal is True
    assert assessments[0].status == "abnormal"
    assert assessments[1].is_abnormal is False
    assert assessments[1].status == "normal"
    assert alarm.suspicious is True
    assert alarm.is_alarm is False
    assert alarm.abnormal_count == 1
    assert alarm.abnormal_labels == ("Useing-Phone",)


def test_abnormal_and_normal_yolov8_detections_are_assessed_independently():
    analyzer = BehaviourAnalyzer(threshold_seconds=3.0, min_confidence=0.35)
    detections = [
        Detection(label="Sleeping", confidence=0.91, bbox=(10, 10, 80, 60)),
        Detection(label="Writing", confidence=0.88, bbox=(90, 10, 140, 120)),
    ]

    assessments, alarm = analyzer.update(detections, now_seconds=10.0)

    assert assessments[0].is_abnormal is True
    assert assessments[0].status == "abnormal"
    assert assessments[0].reason == "Sleeping"
    assert assessments[1].is_abnormal is False
    assert assessments[1].status == "normal"
    assert assessments[1].reason == "Writing"
    assert alarm.suspicious is True
    assert alarm.is_alarm is False
    assert alarm.abnormal_count == 1
    assert alarm.abnormal_labels == ("Sleeping",)


def test_abnormal_label_alarms_after_threshold():
    analyzer = BehaviourAnalyzer(threshold_seconds=2.0, min_confidence=0.35)
    detections = [Detection(label="Head-down", confidence=0.9, bbox=(10, 10, 50, 50))]

    analyzer.update(detections, now_seconds=1.0)
    result_assessments, alarm = analyzer.update(detections, now_seconds=3.2)

    assert result_assessments[0].is_alarm is True
    assert alarm.is_alarm is True
    assert alarm.reason == "multi_behaviour_abnormal"
    assert alarm.duration_seconds == 2.2


def test_missing_abnormal_label_resets_its_timer():
    analyzer = BehaviourAnalyzer(threshold_seconds=2.0, min_confidence=0.35)
    phone = [Detection(label="Useing-Phone", confidence=0.9, bbox=(10, 10, 50, 50))]
    normal = [Detection(label="Reading", confidence=0.9, bbox=(10, 10, 50, 90))]

    analyzer.update(phone, now_seconds=1.0)
    analyzer.update(normal, now_seconds=2.0)
    _, alarm = analyzer.update(phone, now_seconds=4.0)

    assert alarm.is_alarm is False
    assert alarm.duration_seconds == 0.0


def test_low_confidence_abnormal_detection_is_ignored():
    analyzer = BehaviourAnalyzer(threshold_seconds=2.0, min_confidence=0.35)
    detections = [Detection(label="Sleeping", confidence=0.2, bbox=(10, 10, 50, 50))]

    assessments, alarm = analyzer.update(detections, now_seconds=1.0)

    assert assessments[0].status == "ignored"
    assert assessments[0].is_abnormal is False
    assert alarm.suspicious is False


def test_declared_label_sets_match_yolov8_six_class_model():
    assert ABNORMAL_LABELS == {"Useing-Phone", "Head-down", "Sleeping"}
    assert NORMAL_LABELS == {"Hand-raise", "Reading", "Writing"}
