from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from src.backend.app import DEFAULT_MODEL_PATH, append_alarm, behaviour_counts, draw_overlay, frame_status_text
from src.backend.behaviour_analyzer import BehaviourAnalyzer, display_label
from src.backend.detector import YoloDetector
from src.backend.person_crop_grid import build_person_crop_grid
from src.backend.qwen_analysis import (
    QwenAnalysisError,
    call_person_crop_vision,
    load_qwen_settings,
    should_use_qwen_for_scene,
)
from src.common.types import AlarmState, Detection, DetectionAssessment


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / "web-dashboard"
DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
MODE_PERSON_VLM = "person_vlm"
MODE_PERSON_ONLY = "person_only"
MODE_BEHAVIOUR_YOLO = "behaviour_yolo"
PERSON_MODEL_PATH = "yolov8s.pt"
WEB_ALARM_DIR = REPO_ROOT / "output" / "web-alarms"


def decode_data_url_image(data_url: str) -> np.ndarray:
    text = data_url.strip()
    if "," in text and text.lower().startswith("data:"):
        _, encoded = text.split(",", 1)
    else:
        encoded = text

    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("图片数据不是有效的 base64") from exc

    buffer = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("无法解码浏览器上传的图片")
    return frame


def encode_frame_data_url(frame: np.ndarray, image_format: str = "jpeg", quality: int = 90) -> str:
    normalized = image_format.strip().lower().lstrip(".")
    if normalized in {"jpg", "jpeg"}:
        extension = ".jpg"
        mime = "image/jpeg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    elif normalized == "png":
        extension = ".png"
        mime = "image/png"
        params = []
    else:
        raise ValueError(f"不支持的图片格式：{image_format}")

    ok, buffer = cv2.imencode(extension, frame, params)
    if not ok:
        raise ValueError("图片编码失败")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def assessment_to_payload(assessment: DetectionAssessment) -> dict[str, Any]:
    detection = assessment.detection
    return {
        "label": detection.label,
        "displayLabel": display_label(detection.label),
        "confidence": detection.confidence,
        "bbox": list(detection.bbox),
        "status": assessment.status,
        "isAbnormal": assessment.is_abnormal,
        "isAlarm": assessment.is_alarm,
        "reason": assessment.reason,
        "durationSeconds": round(assessment.duration_seconds, 2),
    }


def detection_to_payload(detection: Detection, source: str = "yolo") -> dict[str, Any]:
    return {
        "source": source,
        "label": detection.label,
        "displayLabel": display_label(detection.label),
        "confidence": detection.confidence,
        "bbox": list(detection.bbox),
    }


def detection_from_payload(payload: dict[str, Any]) -> Detection:
    bbox = payload.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        raise ValueError("检测框 bbox 必须包含 4 个坐标")
    return Detection(
        label=str(payload.get("label") or "person"),
        confidence=float(payload.get("confidence") or 0.0),
        bbox=tuple(int(round(float(value))) for value in bbox),
    )


def alarm_to_payload(alarm: AlarmState) -> dict[str, Any]:
    return {
        "isAlarm": alarm.is_alarm,
        "suspicious": alarm.suspicious,
        "durationSeconds": round(alarm.duration_seconds, 2),
        "reason": alarm.reason,
        "abnormalCount": alarm.abnormal_count,
        "abnormalLabels": list(alarm.abnormal_labels),
        "text": frame_status_text(alarm),
    }


def _confidence_score(value: str) -> float:
    return {
        "high": 0.9,
        "medium": 0.65,
        "low": 0.4,
        "unknown": 0.5,
    }.get(str(value).lower(), 0.5)


def vision_result_to_detections(result: Any) -> list[Detection]:
    detections: list[Detection] = []
    for person in result.people:
        if len(person.bbox) != 4:
            continue
        detections.append(
            Detection(
                label=person.label,
                confidence=_confidence_score(person.confidence),
                bbox=tuple(int(value) for value in person.bbox),
            )
        )
    return detections


class DashboardRuntime:
    def __init__(self) -> None:
        self._detectors: dict[tuple[str, str], YoloDetector] = {}
        self._yolo_analyzer = BehaviourAnalyzer()
        self._vision_analyzer = BehaviourAnalyzer()
        self._yolo_alarm_seconds = 3.0
        self._vision_alarm_seconds = 3.0
        self._started_at = time.time()
        self._yolo_frame_count = 0
        self._vision_frame_count = 0
        self._last_alarm_frames: dict[str, int] = {}

    def _detector_for(self, mode: str, model_path: str) -> YoloDetector:
        allowed_labels = {"person"} if mode in {MODE_PERSON_VLM, MODE_PERSON_ONLY} else None
        key = (mode, model_path)
        if key not in self._detectors:
            self._detectors[key] = YoloDetector(model_path=model_path, allowed_labels=allowed_labels)
        return self._detectors[key]

    def _yolo_analyzer_for(self, alarm_seconds: float) -> BehaviourAnalyzer:
        if abs(alarm_seconds - self._yolo_alarm_seconds) > 0.001:
            self._yolo_alarm_seconds = alarm_seconds
            self._yolo_analyzer = BehaviourAnalyzer(threshold_seconds=alarm_seconds)
        return self._yolo_analyzer

    def _vision_analyzer_for(self, alarm_seconds: float) -> BehaviourAnalyzer:
        if abs(alarm_seconds - self._vision_alarm_seconds) > 0.001:
            self._vision_alarm_seconds = alarm_seconds
            self._vision_analyzer = BehaviourAnalyzer(threshold_seconds=alarm_seconds)
        return self._vision_analyzer

    def analyze_yolo(self, frame: np.ndarray, options: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        mode = str(options.get("mode") or MODE_PERSON_VLM)
        alarm_seconds = float(options.get("alarmSeconds") or 3.0)
        model_path = str(options.get("modelPath") or (DEFAULT_MODEL_PATH if mode == MODE_BEHAVIOUR_YOLO else PERSON_MODEL_PATH))

        detector = self._detector_for(mode, model_path)
        detections = detector.detect(frame)
        analyzer = self._yolo_analyzer_for(alarm_seconds)
        assessments, alarm = analyzer.update(detections, now_seconds=time.time())

        self._yolo_frame_count += 1
        return self._analysis_payload(
            stage="yolo",
            frame=frame,
            assessments=assessments,
            alarm=alarm,
            started=started,
            frame_count=self._yolo_frame_count,
            extra={
                "mode": mode,
                "modelPath": model_path,
                "yoloDetections": [detection_to_payload(detection) for detection in detections],
                "targetCount": len([assessment for assessment in assessments if assessment.status != "ignored"]),
            },
        )

    def analyze_vision(self, frame: np.ndarray, detections: list[Detection], options: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        alarm_seconds = float(options.get("alarmSeconds") or 3.0)
        use_vision = bool(options.get("useVision", True))
        vision_payload = self._run_vision_analysis(frame, detections, use_vision)

        vision_detections: list[Detection] = []
        if vision_payload.get("rawResult") is not None:
            vision_detections = vision_result_to_detections(vision_payload["rawResult"])
            vision_payload.pop("rawResult", None)

        analyzer = self._vision_analyzer_for(alarm_seconds)
        assessments, alarm = analyzer.update(vision_detections, now_seconds=time.time())

        self._vision_frame_count += 1
        return self._analysis_payload(
            stage="vision",
            frame=frame,
            assessments=assessments,
            alarm=alarm,
            started=started,
            frame_count=self._vision_frame_count,
            extra={
                "mode": MODE_PERSON_VLM,
                "modelPath": PERSON_MODEL_PATH,
                "targetCount": len(vision_detections),
                "vision": vision_payload,
            },
        )

    def analyze(self, frame: np.ndarray, options: dict[str, Any]) -> dict[str, Any]:
        return self.analyze_yolo(frame, options)

    def _analysis_payload(
        self,
        *,
        stage: str,
        frame: np.ndarray,
        assessments: list[DetectionAssessment],
        alarm: AlarmState,
        started: float,
        frame_count: int,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        elapsed = max(0.001, time.time() - self._started_at)
        fps = frame_count / elapsed
        latency_ms = int((time.time() - started) * 1000)
        overlay = draw_overlay(frame, assessments, alarm, fps=fps, latency_ms=latency_ms)
        height, width = frame.shape[:2]
        alarm_record = self._save_alarm_if_needed(stage, overlay, alarm, frame_count)
        payload = {
            "ok": True,
            "stage": stage,
            "frame": {
                "width": width,
                "height": height,
                "fps": round(fps, 1),
                "latencyMs": latency_ms,
                "frameCount": frame_count,
            },
            "alarm": alarm_to_payload(alarm),
            "counts": behaviour_counts(assessments),
            "detections": [assessment_to_payload(assessment) for assessment in assessments],
            "overlayImage": encode_frame_data_url(overlay, image_format="jpeg", quality=88),
            "alarmRecord": alarm_record,
        }
        payload.update(extra)
        return payload

    def _save_alarm_if_needed(
        self,
        stage: str,
        overlay: np.ndarray,
        alarm: AlarmState,
        frame_count: int,
    ) -> dict[str, Any] | None:
        if not alarm.is_alarm or self._last_alarm_frames.get(stage) == frame_count:
            return None

        timestamp_ms = int(time.time() * 1000)
        output_dir = WEB_ALARM_DIR / stage
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"alarm_{stage}_{frame_count}_{timestamp_ms}.jpg"
        csv_path = output_dir / "alarms.csv"
        cv2.imwrite(str(image_path), overlay)
        append_alarm(
            csv_path,
            frame_count,
            timestamp_ms,
            alarm.reason,
            alarm.duration_seconds,
            alarm.abnormal_count,
            alarm.abnormal_labels,
            image_path,
        )
        self._last_alarm_frames[stage] = frame_count
        return {
            "csvPath": str(csv_path),
            "imagePath": str(image_path),
        }

    def _run_vision_analysis(self, frame: np.ndarray, detections: list[Detection], use_vision: bool) -> dict[str, Any]:
        settings = load_qwen_settings()
        grid = build_person_crop_grid(frame, detections, max_people=settings.max_yolo_targets)
        grid_image = encode_frame_data_url(grid.image, image_format="jpeg", quality=88)

        if not detections:
            return {
                "status": "skipped",
                "message": "未检测到人体。",
                "summary": "未检测到人体。",
                "people": [],
                "gridImage": grid_image,
            }
        if not use_vision:
            return {
                "status": "standby",
                "message": "大模型分类已关闭，仅显示人体检测框。",
                "summary": "",
                "people": [],
                "gridImage": grid_image,
            }
        if not settings.api_key:
            return {
                "status": "error",
                "message": "未配置大模型 API Key，已返回 YOLO 人体框。",
                "summary": "",
                "people": [],
                "gridImage": grid_image,
            }
        if not should_use_qwen_for_scene(len(detections), settings.max_yolo_targets):
            return {
                "status": "skipped",
                "message": f"当前人体数量 {len(detections)} 超过上传上限 {settings.max_yolo_targets}。",
                "summary": "",
                "people": [],
                "gridImage": grid_image,
            }

        try:
            result = call_person_crop_vision(grid.image, grid.source_by_id, settings)
        except QwenAnalysisError as exc:
            return {
                "status": "error",
                "message": str(exc),
                "summary": "",
                "people": [],
                "gridImage": grid_image,
            }

        return {
            "status": "completed",
            "message": f"大模型分析完成：{len(result.people)} 个目标。",
            "summary": result.summary,
            "people": [
                {
                    "label": person.label,
                    "displayLabel": display_label(person.label),
                    "status": person.status,
                    "confidence": person.confidence,
                    "bbox": list(person.bbox),
                }
                for person in result.people
            ],
            "gridImage": grid_image,
            "rawResult": result,
        }


class DashboardRequestHandler(BaseHTTPRequestHandler):
    runtime = DashboardRuntime()
    static_dir = STATIC_DIR

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_common_headers()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/config":
            self._send_json(self._config_payload())
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            frame = decode_data_url_image(str(payload.get("image") or ""))
            if parsed.path == "/api/yolo-frame":
                result = self.runtime.analyze_yolo(frame, payload)
            elif parsed.path == "/api/vlm-frame":
                detections = [detection_from_payload(item) for item in payload.get("detections", [])]
                result = self.runtime.analyze_vision(frame, detections, payload)
            elif parsed.path == "/api/analyze-frame":
                result = self.runtime.analyze(frame, payload)
            else:
                self._send_json({"ok": False, "error": "未知接口"}, status=404)
                return
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=500)
            return
        self._send_json(result)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, path: str) -> None:
        if path in {"", "/"}:
            path = "/index.html"
        relative = unquote(path).lstrip("/")
        target = (self.static_dir / relative).resolve()
        root = self.static_dir.resolve()
        if root not in target.parents and target != root:
            self.send_error(403)
            return
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(200)
        self._send_common_headers(content_type=content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _config_payload(self) -> dict[str, Any]:
        settings = load_qwen_settings()
        return {
            "ok": True,
            "defaultMode": MODE_PERSON_VLM,
            "personModel": PERSON_MODEL_PATH,
            "behaviourModel": DEFAULT_MODEL_PATH,
            "vision": {
                "provider": settings.provider,
                "model": settings.model,
                "configured": bool(settings.api_key),
                "intervalSeconds": settings.interval_seconds,
                "maxTargets": settings.max_yolo_targets,
            },
            "api": {
                "yoloFrame": "/api/yolo-frame",
                "vlmFrame": "/api/vlm-frame",
            },
            "labels": [
                {"label": "Hand-raise", "displayLabel": "举手", "status": "normal"},
                {"label": "Reading", "displayLabel": "学习", "status": "normal"},
                {"label": "Writing", "displayLabel": "学习", "status": "normal"},
                {"label": "Useing-Phone", "displayLabel": "使用手机", "status": "abnormal"},
                {"label": "Head-down", "displayLabel": "低头", "status": "abnormal"},
                {"label": "Sleeping", "displayLabel": "睡觉", "status": "abnormal"},
            ],
        }

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_common_headers(content_type="application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_common_headers(self, content_type: str | None = None) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if content_type:
            self.send_header("Content-Type", content_type)

    def log_message(self, format: str, *args: Any) -> None:
        return


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Browser dashboard for classroom behaviour monitoring")
    parser.add_argument("--host", default=DEFAULT_WEB_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_WEB_PORT)
    parser.add_argument("--static-dir", default=str(STATIC_DIR))
    return parser


def run_server(host: str = DEFAULT_WEB_HOST, port: int = DEFAULT_WEB_PORT, static_dir: str | Path = STATIC_DIR) -> None:
    DashboardRequestHandler.static_dir = Path(static_dir)
    server = ThreadingHTTPServer((host, port), DashboardRequestHandler)
    print(f"Web dashboard listening on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_server(host=args.host, port=args.port, static_dir=args.static_dir)


if __name__ == "__main__":
    main()
