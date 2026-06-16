# Person YOLO + VLM Behaviour Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new pipeline where YOLO only detects people, the backend crops and numbers each person, and the vision model classifies each numbered crop into the existing six classroom behaviours.

**Architecture:** Keep the current six-class YOLO pipeline as a fallback. Add a person-only mode driven by environment/config: YOLO loads a COCO person model, filters detections to `person`, builds a numbered crop grid, sends the grid to the existing vision-provider path, parses `{id, label}` results, then maps labels back onto the original person boxes for overlay, counting, and alarm logic.

**Tech Stack:** Python, PyQt6, OpenCV, Ultralytics YOLO, existing OpenAI-compatible/DashScope vision interface, pytest.

---

### Task 1: Person Detection Filter

**Files:**
- Modify: `src/backend/detector.py`
- Test: `tests/test_detector.py`

- [ ] Add optional label filtering to `result_to_detections` and `YoloDetector`.
- [ ] Verify `person` filtering keeps COCO class 0 and drops other classes.

### Task 2: Person Crop Grid

**Files:**
- Create: `src/backend/person_crop_grid.py`
- Test: `tests/test_person_crop_grid.py`

- [ ] Create `PersonCrop`, `PersonCropGrid`, and `build_person_crop_grid`.
- [ ] Expand person boxes with padding before cropping.
- [ ] Number each crop visually and preserve `id -> original Detection` mapping.

### Task 3: VLM Numbered-Crop Classification

**Files:**
- Modify: `src/backend/qwen_analysis.py`
- Test: `tests/test_qwen_analysis.py`

- [ ] Add a prompt for numbered person crops.
- [ ] Parse `{id, label, confidence, status}` responses.
- [ ] Convert valid numbered responses back to detections using original person boxes.

### Task 4: Backend GUI Integration

**Files:**
- Modify: `src/backend/gui_app.py`
- Test: `tests/test_backend_gui.py`

- [ ] Add a mode selector with `六类YOLO` and `人体YOLO+大模型`.
- [ ] Default the new mode to person detection with `yolov8s.pt`.
- [ ] In the new mode, display person boxes immediately and replace labels after the vision model returns.

### Task 5: Documentation and Verification

**Files:**
- Modify: `README.md`

- [ ] Document the new mode and recommended model.
- [ ] Run `pytest tests/test_detector.py tests/test_person_crop_grid.py tests/test_qwen_analysis.py tests/test_backend_gui.py -q`.
