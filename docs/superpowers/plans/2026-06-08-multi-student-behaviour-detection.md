# Multi Student Behaviour Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the backend from whole-frame sleeping alarms to per-detection classroom behaviour analysis with multiple normal and abnormal states.

**Architecture:** Keep the existing TCP image transport and YOLO detector. Add a focused behaviour analyzer that classifies each `Detection` as normal, abnormal, or ignored, and returns one frame-level alarm summary for screenshot/logging.

**Tech Stack:** Python 3.12, OpenCV, Ultralytics YOLOv8, pytest, PowerShell packaging script.

---

## File Structure

- Modify `src/common/types.py`: add per-detection assessment dataclasses and extend `AlarmState` with abnormal count/labels.
- Create `src/backend/behaviour_analyzer.py`: map YOLO labels to normal/abnormal states and track abnormal label duration.
- Modify `src/backend/app.py`: use the new analyzer, draw red/green boxes per detection, and write expanded alarm CSV rows.
- Modify `scripts/offline_test_images.py`: mark multi-behaviour abnormal candidates instead of only `sleep_candidate`.
- Create `scripts/prepare_yolo_data_yaml.py`: generate a local absolute-path YOLO config for datasets with spaces or Roboflow relative paths.
- Modify `scripts/package_backend.ps1`: package the new multi-state model name by default.
- Modify `README.md`: document multi-state training, offline testing, backend launch, and packaging.
- Add/update tests in `tests/`: cover behaviour analyzer, overlay colours, CSV fields, offline CSV fields, and config generation.

---

### Task 1: Per-Detection Behaviour Analyzer

**Files:**
- Modify: `src/common/types.py`
- Create: `src/backend/behaviour_analyzer.py`
- Test: `tests/test_behaviour_analyzer.py`

- [ ] **Step 1: Write failing behaviour analyzer tests**

```python
from src.backend.behaviour_analyzer import ABNORMAL_LABELS, BehaviourAnalyzer, NORMAL_LABELS
from src.common.types import Detection


def test_abnormal_and_normal_detections_are_assessed_independently():
    analyzer = BehaviourAnalyzer(threshold_seconds=3.0, min_confidence=0.35)
    detections = [
        Detection(label="sleep", confidence=0.91, bbox=(10, 10, 80, 60)),
        Detection(label="upright", confidence=0.88, bbox=(90, 10, 140, 120)),
    ]

    assessments, alarm = analyzer.update(detections, now_seconds=10.0)

    assert assessments[0].is_abnormal is True
    assert assessments[0].status == "abnormal"
    assert assessments[1].is_abnormal is False
    assert assessments[1].status == "normal"
    assert alarm.suspicious is True
    assert alarm.is_alarm is False
    assert alarm.abnormal_count == 1
    assert alarm.abnormal_labels == ("sleep",)


def test_abnormal_label_alarms_after_threshold():
    analyzer = BehaviourAnalyzer(threshold_seconds=2.0, min_confidence=0.35)
    detections = [Detection(label="phone", confidence=0.9, bbox=(10, 10, 50, 50))]

    analyzer.update(detections, now_seconds=1.0)
    result_assessments, alarm = analyzer.update(detections, now_seconds=3.2)

    assert result_assessments[0].is_alarm is True
    assert alarm.is_alarm is True
    assert alarm.reason == "multi_behaviour_abnormal"
    assert alarm.duration_seconds == 2.2


def test_missing_abnormal_label_resets_its_timer():
    analyzer = BehaviourAnalyzer(threshold_seconds=2.0, min_confidence=0.35)
    phone = [Detection(label="phone", confidence=0.9, bbox=(10, 10, 50, 50))]
    normal = [Detection(label="upright", confidence=0.9, bbox=(10, 10, 50, 90))]

    analyzer.update(phone, now_seconds=1.0)
    analyzer.update(normal, now_seconds=2.0)
    _, alarm = analyzer.update(phone, now_seconds=4.0)

    assert alarm.is_alarm is False
    assert alarm.duration_seconds == 0.0


def test_declared_label_sets_match_course_rule():
    assert {"Using_phone", "phone", "sleep", "bend", "bow_head", "turn_head"} <= ABNORMAL_LABELS
    assert {"upright", "reading", "writing", "book", "hand-raising", "raise_head"} <= NORMAL_LABELS
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_behaviour_analyzer.py -q
```

Expected: import failure because `src.backend.behaviour_analyzer` does not exist.

- [ ] **Step 3: Implement minimal types and analyzer**

Add to `src/common/types.py`:

```python
@dataclass(frozen=True)
class DetectionAssessment:
    detection: Detection
    status: str
    is_abnormal: bool
    is_alarm: bool
    reason: str
    duration_seconds: float
```

Extend `AlarmState` with defaults:

```python
abnormal_count: int = 0
abnormal_labels: tuple[str, ...] = ()
```

Create `BehaviourAnalyzer` with `NORMAL_LABELS`, `ABNORMAL_LABELS`, per-label timers, low-confidence `ignored` status, and `update()` returning `(list[DetectionAssessment], AlarmState)`.

- [ ] **Step 4: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_behaviour_analyzer.py -q
```

Expected: all behaviour analyzer tests pass.

---

### Task 2: Backend Overlay and Alarm Logging

**Files:**
- Modify: `src/backend/app.py`
- Modify: `tests/test_backend_app.py`

- [ ] **Step 1: Write/update backend app tests**

Update `tests/test_backend_app.py` so `draw_overlay()` receives `DetectionAssessment` objects. Add a test that samples rectangle border pixels and confirms abnormal boxes are red while normal boxes are green.

Expected key assertions:

```python
assert tuple(output[10, 10]) == (0, 0, 255)
assert tuple(output[10, 70]) == (0, 180, 0)
```

Update `append_alarm()` expected CSV header:

```python
[
    "frame_id",
    "timestamp_ms",
    "reason",
    "duration_seconds",
    "abnormal_count",
    "abnormal_labels",
    "image_path",
]
```

- [ ] **Step 2: Run backend app tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_backend_app.py -q
```

Expected: failures because app still expects global `AlarmState` colouring and old CSV fields.

- [ ] **Step 3: Update backend app implementation**

Use:

```python
from src.backend.behaviour_analyzer import BehaviourAnalyzer
from src.common.types import AlarmState, DetectionAssessment
```

Change `draw_overlay()` to iterate assessments:

```python
for assessment in assessments:
    detection = assessment.detection
    color = (0, 0, 255) if assessment.is_abnormal else (0, 180, 0)
```

In `run_backend()`, replace `SleepAnalyzer` with `BehaviourAnalyzer`:

```python
assessments, alarm = analyzer.update(detections, now_seconds=now)
overlay = draw_overlay(frame, assessments, alarm, fps=fps, latency_ms=latency_ms)
```

Expand `append_alarm()` arguments and CSV row with abnormal count/labels.

- [ ] **Step 4: Run backend app tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_backend_app.py -q
```

Expected: backend app tests pass.

---

### Task 3: Offline Test CSV and YOLO Data Config Script

**Files:**
- Modify: `scripts/offline_test_images.py`
- Modify: `tests/test_offline_test_images.py`
- Create: `scripts/prepare_yolo_data_yaml.py`
- Create: `tests/test_prepare_yolo_data_yaml.py`

- [ ] **Step 1: Update offline CSV test**

Replace `sleep_candidate` with:

```python
"behaviour_status": "abnormal",
"abnormal_candidate": "true",
```

For normal labels, expect `behaviour_status` to be `normal` and `abnormal_candidate` to be `false`.

- [ ] **Step 2: Add data YAML generation test**

Test helper output contains absolute `train`, `val`, `test`, `nc`, and the 12 class names from the source `data.yaml`.

- [ ] **Step 3: Run tests and confirm failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_offline_test_images.py tests\test_prepare_yolo_data_yaml.py -q
```

Expected: failures until scripts are updated/created.

- [ ] **Step 4: Implement scripts**

`offline_test_images.py` imports `ABNORMAL_LABELS` and `NORMAL_LABELS`, writes fields:

```python
fieldnames = ["image", "label", "confidence", "x1", "y1", "x2", "y2", "behaviour_status", "abnormal_candidate"]
```

`prepare_yolo_data_yaml.py` reads `names:` from the Roboflow `data.yaml` with `ast.literal_eval`, then writes an absolute-path config.

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_offline_test_images.py tests\test_prepare_yolo_data_yaml.py -q
```

Expected: tests pass.

---

### Task 4: Train and Offline-Test the Multi-State Model

**Files:**
- Runtime output: `tmp/student-behaviour-detection-abs.yaml`
- Runtime output: `output/training/student_behaviour_yolov8n_e3/`
- Runtime output: `output/offline_test/student-behaviour-custom-e3/`
- Create: `docs/course-evidence/student-behaviour-custom-e3-test.md`

- [ ] **Step 1: Generate absolute YOLO config**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_yolo_data_yaml.py --dataset "datasets\Student Behaviour Detection.v6i.yolov8" --output tmp\student-behaviour-detection-abs.yaml
```

Expected: generated config points to absolute local `train/images`, `valid/images`, and `test/images`.

- [ ] **Step 2: Train first multi-state model**

Run:

```powershell
.\.venv\Scripts\yolo.exe detect train data=tmp\student-behaviour-detection-abs.yaml model=yolov8n.pt epochs=3 imgsz=320 batch=8 device=cpu workers=0 project=output\training name=student_behaviour_yolov8n_e3 exist_ok=True
```

Expected: `output\training\student_behaviour_yolov8n_e3\weights\best.pt` exists.

- [ ] **Step 3: Run validation if training command did not already report metrics clearly**

Run:

```powershell
.\.venv\Scripts\yolo.exe detect val data=tmp\student-behaviour-detection-abs.yaml model=output\training\student_behaviour_yolov8n_e3\weights\best.pt imgsz=320 batch=8 device=cpu workers=0
```

Expected: record overall precision, recall, mAP50, and per-class values.

- [ ] **Step 4: Offline test on test/images**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\offline_test_images.py --dataset "datasets\Student Behaviour Detection.v6i.yolov8" --model output\training\student_behaviour_yolov8n_e3\weights\best.pt --output-dir output\offline_test\student-behaviour-custom-e3 --limit 0 --conf 0.25
```

Expected: annotated images and `predictions.csv` are created.

- [ ] **Step 5: Write course evidence summary**

Document dataset size, validation metrics, prediction counts, and output path in `docs/course-evidence/student-behaviour-custom-e3-test.md`.

---

### Task 5: README, Packaging, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `scripts/package_backend.ps1`
- Test: full test suite
- Runtime output: `dist/backend-student-sleep-server.zip`

- [ ] **Step 1: Update README**

Document:

- multi-state dataset path
- config generation command
- training command
- offline test command
- backend launch command using `output\training\student_behaviour_yolov8n_e3\weights\best.pt`
- red/green box behaviour

- [ ] **Step 2: Update package script**

Default model path:

```powershell
output\training\student_behaviour_yolov8n_e3\weights\best.pt
```

Packaged model path:

```powershell
models\student_behaviour_yolov8n_best.pt
```

Backend start command inside package:

```powershell
.\.venv\Scripts\python.exe -m src.backend.app --host 0.0.0.0 --port 5001 --model models\student_behaviour_yolov8n_best.pt
```

- [ ] **Step 3: Run all tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Rebuild backend package**

Run:

```powershell
.\scripts\package_backend.ps1
```

Expected: `dist\backend-student-sleep-server.zip` contains the new multi-state model and backend files.

- [ ] **Step 5: Inspect final status**

Run:

```powershell
git status --short --branch
```

Expected: only intended source, docs, script, and test files changed; generated `datasets`, `output`, `tmp`, and `dist` remain ignored.
