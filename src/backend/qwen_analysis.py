from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

import cv2


DEFAULT_QWEN_MODEL = "qwen3.6-flash"
DEFAULT_QWEN_INTERVAL_SECONDS = 10
DEFAULT_CAMERA_INDEX = 0
DEFAULT_QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_QWEN_MAX_YOLO_TARGETS = 3


class QwenAnalysisError(Exception):
    """Raised when Qwen analysis cannot produce a parsed result."""


@dataclass
class QwenPerson:
    bbox: list[int]
    status: str
    confidence: str = "unknown"


@dataclass
class QwenAnalysisResult:
    people: list[QwenPerson]
    summary: str
    raw_text: str


@dataclass
class QwenSettings:
    api_key: str
    model: str
    interval_seconds: int
    camera_index: int = DEFAULT_CAMERA_INDEX
    base_http_api_url: str = DEFAULT_QWEN_API_URL
    use_coordinate_grid: bool = True
    max_yolo_targets: int = DEFAULT_QWEN_MAX_YOLO_TARGETS


def _read_positive_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off"}


def load_qwen_settings() -> QwenSettings:
    return QwenSettings(
        api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
        model=os.getenv("QWEN_VL_MODEL", DEFAULT_QWEN_MODEL).strip() or DEFAULT_QWEN_MODEL,
        interval_seconds=_read_positive_int("QWEN_UPLOAD_INTERVAL_SECONDS", DEFAULT_QWEN_INTERVAL_SECONDS),
        camera_index=_read_positive_int("CAMERA_INDEX", DEFAULT_CAMERA_INDEX),
        base_http_api_url=os.getenv("DASHSCOPE_BASE_HTTP_API_URL", DEFAULT_QWEN_API_URL).strip()
        or DEFAULT_QWEN_API_URL,
        use_coordinate_grid=_read_bool("QWEN_COORDINATE_GRID", True),
        max_yolo_targets=_read_positive_int("QWEN_MAX_YOLO_TARGETS", DEFAULT_QWEN_MAX_YOLO_TARGETS),
    )


def build_qwen_prompt(width: int, height: int, use_coordinate_grid: bool) -> str:
    grid_text = "图像上叠加了淡色坐标网格，网格线旁边的数字就是像素坐标。" if use_coordinate_grid else ""
    return f"""请分析这张课堂监控画面。
图片尺寸：宽 {width} 像素，高 {height} 像素。{grid_text}
只返回一个 JSON 对象，不要返回 Markdown，不要解释。
JSON 格式必须为：
{{
  "people": [
    {{
      "bbox": [x1, y1, x2, y2],
      "status": "自由描述该人物当前状态",
      "confidence": "high|medium|low|unknown"
    }}
  ],
  "summary": "一句话概括画面整体状态"
}}
坐标必须使用上传图片的像素坐标，x1/y1 是左上角，x2/y2 是右下角。
bbox 必须框住人物可见身体区域；如果身体被遮挡，只框住可见的人体部分。
不要框住桌子、椅子、电脑、显示器、杯子、背包或其他物品。
不要只框住手臂、腿、头发或衣物局部。
无法确定人物位置时，不要编造坐标，返回空 people 数组。"""


def prepare_frame_for_qwen(frame, settings: QwenSettings):
    prepared = frame.copy()
    if not settings.use_coordinate_grid:
        return prepared

    height, width = prepared.shape[:2]
    step = max(80, min(width, height) // 6)
    grid_color = (0, 255, 255)
    text_color = (0, 0, 255)

    for x in range(0, width, step):
        cv2.line(prepared, (x, 0), (x, height - 1), grid_color, 1, cv2.LINE_AA)
        cv2.putText(prepared, str(x), (x + 4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    for y in range(0, height, step):
        cv2.line(prepared, (0, y), (width - 1, y), grid_color, 1, cv2.LINE_AA)
        cv2.putText(prepared, str(y), (4, max(18, y + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.rectangle(prepared, (0, 0), (width - 1, height - 1), grid_color, 1)
    cv2.putText(
        prepared,
        f"size {width}x{height}",
        (max(4, width - 150), height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        text_color,
        1,
        cv2.LINE_AA,
    )
    return prepared


def should_upload_frame(now: float, last_upload_at: float | None, interval_seconds: int, in_flight: bool) -> bool:
    if in_flight:
        return False
    if last_upload_at is None:
        return True
    return now - last_upload_at >= interval_seconds


def should_use_qwen_for_scene(target_count: int, max_targets: int) -> bool:
    return target_count <= max_targets


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise QwenAnalysisError("无法解析千问 JSON：返回内容中没有 JSON 对象")
    return stripped[start : end + 1]


def clamp_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_qwen_response(text: str, width: int, height: int) -> QwenAnalysisResult:
    try:
        payload = json.loads(extract_json_object(text))
    except (json.JSONDecodeError, QwenAnalysisError) as exc:
        raise QwenAnalysisError(f"无法解析千问 JSON：{exc}") from exc

    people = []
    for item in payload.get("people", []):
        if not isinstance(item, dict):
            continue
        bbox = clamp_bbox(item.get("bbox"), width=width, height=height)
        if bbox is None:
            continue
        status = str(item.get("status") or "未描述")
        confidence = str(item.get("confidence") or "unknown")
        people.append(QwenPerson(bbox=bbox, status=status, confidence=confidence))

    return QwenAnalysisResult(
        people=people,
        summary=str(payload.get("summary") or ""),
        raw_text=text,
    )


def encode_frame_to_data_url(frame) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise QwenAnalysisError("无法将摄像头画面编码为 JPEG")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_qwen_vision(frame, settings: QwenSettings) -> QwenAnalysisResult:
    if not settings.api_key:
        raise QwenAnalysisError("未配置 DASHSCOPE_API_KEY，无法调用千问")

    try:
        import dashscope
    except ImportError as exc:
        raise QwenAnalysisError("未安装 dashscope，请先安装 requirements.txt 中的依赖") from exc

    dashscope.base_http_api_url = settings.base_http_api_url
    image_url = encode_frame_to_data_url(frame)
    height, width = frame.shape[:2]
    prompt = build_qwen_prompt(width=width, height=height, use_coordinate_grid=settings.use_coordinate_grid)
    messages = [
        {
            "role": "user",
            "content": [
                {"image": image_url},
                {"text": prompt},
            ],
        }
    ]

    response = dashscope.MultiModalConversation.call(
        api_key=settings.api_key,
        model=settings.model,
        messages=messages,
        enable_thinking=False,
    )
    status_code = getattr(response, "status_code", 200)
    if status_code and int(status_code) >= 400:
        code = getattr(response, "code", "UnknownError")
        message = getattr(response, "message", str(response))
        raise QwenAnalysisError(f"千问接口错误：{code}，{message}。当前端点：{settings.base_http_api_url}")
    try:
        text = response.output.choices[0].message.content[0]["text"]
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise QwenAnalysisError(f"千问返回格式异常：{response}") from exc
    return parse_qwen_response(text, width=width, height=height)
