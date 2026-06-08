from src.backend.detector import result_to_detections


class FakeBox:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = [xyxy]
        self.conf = [conf]
        self.cls = [cls]


class FakeResult:
    names = {0: "person", 1: "sleep"}

    def __init__(self):
        self.boxes = [
            FakeBox([1.2, 2.8, 30.1, 50.9], 0.88, 0),
            FakeBox([5.0, 6.0, 40.0, 24.0], 0.92, 1),
        ]


def test_result_to_detections_converts_boxes():
    detections = result_to_detections(FakeResult())

    assert len(detections) == 2
    assert detections[0].label == "person"
    assert detections[0].confidence == 0.88
    assert detections[0].bbox == (1, 3, 30, 51)
    assert detections[1].label == "sleep"
    assert detections[1].bbox == (5, 6, 40, 24)
