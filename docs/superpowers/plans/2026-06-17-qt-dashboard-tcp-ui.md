# Qt Dashboard TCP UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the browser dashboard information architecture to the local Qt front-end and back-end programs while keeping the custom TCP packet protocol and Qt worker threads as the official dual-machine path.

**Architecture:** The existing `CameraSenderWorker`, `BackendReceiverWorker`, `LocalMediaWorker`, `QwenWorker`, and `src.common.protocol` remain the runtime boundary. Only the Qt window composition and styling change: both local programs get a left navigation rail, top status area, panel-based controls, large preview/monitor stage, metrics, logs, and protocol badges that explicitly expose the `NSGD` TCP design.

**Tech Stack:** PyQt6, existing OpenCV capture/encoding, existing socket TCP frame protocol, existing pytest Qt offscreen tests.

---

### Task 1: Add Tests for the New Local Qt Shell

**Files:**
- Modify: `tests/test_frontend_gui.py`
- Modify: `tests/test_backend_gui.py`

- [ ] **Step 1: Write the failing tests**

Add assertions that the front-end and back-end windows expose the dashboard shell, sidebar, and TCP protocol badge while preserving existing control attributes.

```python
assert window.centralWidget().objectName() == "qtDashboardShell"
assert window.sidebar.objectName() == "dashboardSidebar"
assert "NSGD" in window.protocol_badge.text()
assert "TCP" in window.protocol_badge.text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_frontend_gui.py::test_frontend_gui_defaults tests\test_backend_gui.py::test_backend_gui_defaults -q
```

Expected: failures showing missing `sidebar` or `protocol_badge`.

### Task 2: Create a Shared Qt Dashboard Theme

**Files:**
- Create: `src/common/qt_dashboard_theme.py`

- [ ] **Step 1: Implement the shared theme helpers**

Create reusable PyQt6 helpers:

```python
from PyQt6 import QtCore, QtWidgets

DASHBOARD_STYLESHEET = """..."""

def apply_dashboard_style(widget: QtWidgets.QWidget) -> None:
    widget.setStyleSheet(DASHBOARD_STYLESHEET)

def make_sidebar(title: str, subtitle: str, active_label: str, secondary_labels: list[str], note: str) -> QtWidgets.QFrame:
    ...

def make_panel(title: str, subtitle: str = "") -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
    ...

def make_metric_card(label: str, value_widget: QtWidgets.QWidget) -> QtWidgets.QFrame:
    ...
```

The stylesheet must keep a quiet classroom-monitoring palette, panel radius at 8px or less, and avoid changing runtime logic.

- [ ] **Step 2: Run import check**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\common\qt_dashboard_theme.py
```

Expected: exit code 0.

### Task 3: Redesign the Qt Front-End Sender Window

**Files:**
- Modify: `src/frontend/gui_client.py`
- Test: `tests/test_frontend_gui.py`

- [ ] **Step 1: Replace only `CameraClientWindow._build_ui` composition**

Keep these attributes and methods unchanged for tests and worker wiring:

```python
self.host_edit
self.port_spin
self.camera_spin
self.width_spin
self.fps_spin
self.quality_spin
self.preview_label
self.status_label
self.fps_label
self.resolution_label
self.frame_count_label
self.jpeg_size_label
self.start_button
self.stop_button
```

Create a two-column shell:

```text
sidebar: ClassBehave Sender / 发送端 active / 协议说明
main: topbar + preview panel + control panel + metric cards
```

Add `self.protocol_badge = QLabel("NSGD TCP 帧包")` and keep the existing `start_sender`, `stop_sender`, `_update_preview`, and `_update_metrics` logic.

- [ ] **Step 2: Run the front-end GUI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_frontend_gui.py -q
```

Expected: all tests pass.

### Task 4: Redesign the Qt Back-End Monitor Window

**Files:**
- Modify: `src/backend/gui_app.py`
- Test: `tests/test_backend_gui.py`

- [ ] **Step 1: Replace only `BackendMonitorWindow._build_ui` composition**

Keep these attributes and methods unchanged:

```python
self.host_edit
self.port_spin
self.mode_combo
self.model_edit
self.alarm_spin
self.output_edit
self.video_label
self.status_label
self.alarm_label
self.fps_label
self.resolution_label
self.frame_count_label
self.latency_label
self.counts_text
self.log_text
self.image_test_button
self.video_test_button
self.stop_test_button
self.start_button
self.stop_button
```

Create a Web-dashboard-like back-end shell:

```text
sidebar: ClassBehave Backend / 实时分析 active / 大模型 / 日志设置 / NSGD TCP
main: topbar + listen controls + monitor panel + insight panel + logs
```

Add `self.protocol_badge = QLabel("NSGD TCP 监听 0.0.0.0:5001")` and preserve the existing socket worker wiring in `start_backend`.

- [ ] **Step 2: Run the back-end GUI tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_backend_gui.py -q
```

Expected: all tests pass.

### Task 5: Update Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Clarify the official dual-machine path**

Update the Qt GUI sections to state:

```text
正式双机联调使用本地 Qt 程序，界面采用 Web 控制台风格，但底层仍使用自定义 NSGD TCP 帧包协议。
```

Mention that the browser Web console remains the HTTP demonstration path.

### Task 6: Final Verification

**Files:**
- No production edits.

- [ ] **Step 1: Run focused GUI tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_frontend_gui.py tests\test_backend_gui.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax checks**

```powershell
.\.venv\Scripts\python.exe -m py_compile src\common\qt_dashboard_theme.py src\frontend\gui_client.py src\backend\gui_app.py
```

Expected: exit code 0.

- [ ] **Step 3: Capture offscreen window screenshots**

Use PyQt6 offscreen rendering to save:

```text
output/playwright/qt-frontend-dashboard.png
output/playwright/qt-backend-dashboard.png
```

Expected: screenshots show sidebar, panels, protocol badge, controls, and preview area.
