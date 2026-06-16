from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import requests

from src.backend.behaviour_analyzer import LABEL_DISPLAY_NAMES
from src.common.types import Detection


DEFAULT_QWEN_MODEL = "qwen3.6-flash"
DEFAULT_OPENAI_VISION_MODEL = "gpt-5.5"
DEFAULT_QWEN_INTERVAL_SECONDS = 10
DEFAULT_CAMERA_INDEX = 0
DEFAULT_QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1"
DEFAULT_QWEN_MAX_YOLO_TARGETS = 80
DEFAULT_OPENAI_TIMEOUT_SECONDS = 120
DEFAULT_OPENAI_IMAGE_MAX_WIDTH = 640
DEFAULT_OPENAI_REQUEST_ATTEMPTS = 2
QWEN_BEHAVIOUR_LABELS = ("Hand-raise", "Reading", "Writing", "Useing-Phone", "Head-down", "Sleeping")
_QWEN_LABEL_ALIASES = {
    "hand-raise": "Hand-raise",
    "hand_raise": "Hand-raise",
    "handraise": "Hand-raise",
    "raise_hand": "Hand-raise",
    "举手": "Hand-raise",
    "reading": "Reading",
    "read": "Reading",
    "看书": "Reading",
    "阅读": "Reading",
    "writing": "Writing",
    "write": "Writing",
    "写字": "Writing",
    "书写": "Writing",
    "useing-phone": "Useing-Phone",
    "using-phone": "Useing-Phone",
    "using_phone": "Useing-Phone",
    "useing_phone": "Useing-Phone",
    "phone": "Useing-Phone",
    "使用手机": "Useing-Phone",
    "看手机": "Useing-Phone",
    "玩手机": "Useing-Phone",
    "手机屏幕": "Useing-Phone",
    "小屏幕": "Useing-Phone",
    "手持设备": "Useing-Phone",
    "手中设备": "Useing-Phone",
    "移动设备": "Useing-Phone",
    "拿着手机": "Useing-Phone",
    "低头看手机": "Useing-Phone",
    "head-down": "Head-down",
    "head_down": "Head-down",
    "headdown": "Head-down",
    "低头": "Head-down",
    "低头看桌面": "Head-down",
    "sleeping": "Sleeping",
    "sleep": "Sleeping",
    "睡觉": "Sleeping",
    "趴睡": "Sleeping",
}
_QWEN_LABEL_ALIASES.update({label.lower(): label for label in QWEN_BEHAVIOUR_LABELS})
_QWEN_LABEL_ALIASES.update({display: label for label, display in LABEL_DISPLAY_NAMES.items()})


class QwenAnalysisError(Exception):
    """Raised when Qwen analysis cannot produce a parsed result."""


@dataclass
class QwenPerson:
    bbox: list[int]
    label: str
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
    provider: str = "dashscope"
    request_timeout_seconds: int = DEFAULT_OPENAI_TIMEOUT_SECONDS
    max_image_width: int = 0
    openai_image_format: str = "png"


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


def _read_openai_image_format() -> str:
    value = os.getenv("OPENAI_IMAGE_FORMAT", "png").strip().lower().lstrip(".")
    if value in {"jpg", "jpeg"}:
        return "jpeg"
    if value == "png":
        return "png"
    return "png"


def load_qwen_settings() -> QwenSettings:
    provider = os.getenv("VISION_PROVIDER", os.getenv("QWEN_PROVIDER", "dashscope")).strip().lower()
    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        return QwenSettings(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            model=os.getenv("OPENAI_VISION_MODEL", DEFAULT_OPENAI_VISION_MODEL).strip()
            or DEFAULT_OPENAI_VISION_MODEL,
            interval_seconds=_read_positive_int("QWEN_UPLOAD_INTERVAL_SECONDS", DEFAULT_QWEN_INTERVAL_SECONDS),
            camera_index=_read_positive_int("CAMERA_INDEX", DEFAULT_CAMERA_INDEX),
            base_http_api_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_API_URL).strip()
            or DEFAULT_OPENAI_API_URL,
            use_coordinate_grid=_read_bool("QWEN_COORDINATE_GRID", True),
            max_yolo_targets=_read_positive_int("QWEN_MAX_YOLO_TARGETS", DEFAULT_QWEN_MAX_YOLO_TARGETS),
            provider="openai",
            request_timeout_seconds=_read_positive_int("OPENAI_TIMEOUT_SECONDS", DEFAULT_OPENAI_TIMEOUT_SECONDS),
            max_image_width=_read_positive_int("OPENAI_IMAGE_MAX_WIDTH", DEFAULT_OPENAI_IMAGE_MAX_WIDTH),
            openai_image_format=_read_openai_image_format(),
        )

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
      "label": "Hand-raise|Reading|Writing|Useing-Phone|Head-down|Sleeping",
      "status": "简短中文说明该人物当前状态",
      "confidence": "high|medium|low|unknown"
    }}
  ],
  "summary": "一句话概括画面整体状态"
}}
label 必须只能从以上六类中选择；不要输出 standing、sitting、listening、normal、unknown 等其他类别。
六类含义：Hand-raise=举手，Reading=看书，Writing=写字，Useing-Phone=使用手机，Head-down=低头，Sleeping=睡觉。
Useing-Phone 优先：只要看到手机、手持小矩形屏幕、手里拿着手机、双手低头看小屏幕，必须标为 Useing-Phone；不要把看手机的人标为 Head-down、Writing 或 Reading。
如果只是使用台式电脑、笔记本电脑、键盘、鼠标或显示器，不能因为有屏幕就标为 Useing-Phone。
坐标必须使用上传图片的像素坐标，x1/y1 是左上角，x2/y2 是右下角。
bbox 必须框住人物可见身体区域；如果身体被遮挡，只框住可见的人体部分。
不要框住桌子、椅子、电脑、显示器、杯子、背包或其他物品。
不要只框住手臂、腿、头发或衣物局部。
无法确定人物位置时，不要编造坐标，返回空 people 数组。"""


def build_person_crop_prompt(person_ids) -> str:
    id_text = ", ".join(str(person_id) for person_id in person_ids)
    return f"""这是一张由多个课堂人物小图拼成的大图，每个小图左上角有蓝底白字编号。
请只判断编号为 {id_text} 的人物行为。
只返回一个 JSON 对象，不要返回 Markdown，不要解释。
JSON 格式必须为：
{{
  "people": [
    {{
      "id": 1,
      "label": "Hand-raise|Reading|Writing|Useing-Phone|Head-down|Sleeping",
      "status": "简短中文说明该编号人物当前状态",
      "confidence": "high|medium|low|unknown"
    }}
  ],
  "summary": "一句话概括整体状态"
}}
label 必须只能从以上六类中选择；不要输出 standing、sitting、listening、normal、unknown 等其他类别。
六类含义：Hand-raise=举手，Reading=看书，Writing=写字，Useing-Phone=使用手机，Head-down=低头，Sleeping=睡觉。
Useing-Phone 优先：只要看到手机、手持小矩形屏幕、手里拿着手机、双手低头看小屏幕，必须标为 Useing-Phone；不要把看手机的人标为 Head-down、Writing 或 Reading。
如果只是使用台式电脑、笔记本电脑、键盘、鼠标或显示器，不能因为有屏幕就标为 Useing-Phone。
只返回编号和行为类别，不要返回 bbox 或坐标。"""


def prepare_frame_for_qwen(frame, settings: QwenSettings):
    prepared = frame.copy()
    height, width = prepared.shape[:2]
    if settings.max_image_width > 0 and width > settings.max_image_width:
        new_height = max(1, int(round(height * settings.max_image_width / width)))
        prepared = cv2.resize(prepared, (settings.max_image_width, new_height), interpolation=cv2.INTER_AREA)

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


def normalize_qwen_label(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    direct = _QWEN_LABEL_ALIASES.get(text) or _QWEN_LABEL_ALIASES.get(text.lower())
    if direct is not None:
        return direct

    lowered = text.lower()
    for key, label in _QWEN_LABEL_ALIASES.items():
        if key and key.lower() in lowered:
            return label
    return None


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
        label = normalize_qwen_label(item.get("label"))
        status_label = normalize_qwen_label(item.get("status"))
        if status_label == "Useing-Phone":
            label = status_label
        if label is None:
            label = status_label
        if label is None:
            continue
        status = str(item.get("status") or "未描述")
        confidence = str(item.get("confidence") or "unknown")
        people.append(QwenPerson(bbox=bbox, label=label, status=status, confidence=confidence))

    return QwenAnalysisResult(
        people=people,
        summary=str(payload.get("summary") or ""),
        raw_text=text,
    )


def parse_person_crop_response(text: str, source_by_id: dict[int, Detection]) -> QwenAnalysisResult:
    try:
        payload = json.loads(extract_json_object(text))
    except (json.JSONDecodeError, QwenAnalysisError) as exc:
        raise QwenAnalysisError(f"无法解析千问 JSON：{exc}") from exc

    people = []
    for item in payload.get("people", []):
        if not isinstance(item, dict):
            continue
        try:
            person_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        source = source_by_id.get(person_id)
        if source is None:
            continue

        label = normalize_qwen_label(item.get("label"))
        status_label = normalize_qwen_label(item.get("status"))
        if status_label == "Useing-Phone":
            label = status_label
        if label is None:
            label = status_label
        if label is None:
            continue

        people.append(
            QwenPerson(
                bbox=list(source.bbox),
                label=label,
                status=str(item.get("status") or "未描述"),
                confidence=str(item.get("confidence") or "unknown"),
            )
        )

    return QwenAnalysisResult(
        people=people,
        summary=str(payload.get("summary") or ""),
        raw_text=text,
    )


def encode_frame_to_data_url(frame, image_format: str = "jpeg") -> str:
    normalized = image_format.strip().lower().lstrip(".")
    if normalized in {"jpg", "jpeg"}:
        extension = ".jpg"
        mime_type = "image/jpeg"
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    elif normalized == "png":
        extension = ".png"
        mime_type = "image/png"
        encode_params = []
    else:
        raise QwenAnalysisError(f"不支持的图片编码格式：{image_format}")

    ok, buffer = cv2.imencode(extension, frame, encode_params)
    if not ok:
        raise QwenAnalysisError(f"无法将摄像头画面编码为 {normalized.upper()}")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _openai_chat_completions_url(base_http_api_url: str) -> str:
    base = base_http_api_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _extract_openai_message_text(payload: dict[str, Any]) -> str:
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return str(content)


def _openai_attempt_image_formats(settings: QwenSettings) -> list[str]:
    primary = settings.openai_image_format.strip().lower().lstrip(".")
    if primary == "jpg":
        primary = "jpeg"
    if primary not in {"png", "jpeg"}:
        primary = "png"

    formats = [primary]
    if primary == "png":
        formats.append("jpeg")
    while len(formats) < DEFAULT_OPENAI_REQUEST_ATTEMPTS:
        formats.append(primary)
    return formats[:DEFAULT_OPENAI_REQUEST_ATTEMPTS]


def _request_openai_compatible_text(frame, settings: QwenSettings, prompt: str) -> str:
    if not settings.api_key:
        raise QwenAnalysisError("未配置 OPENAI_API_KEY，无法调用 OpenAI 兼容视觉接口")

    url = _openai_chat_completions_url(settings.base_http_api_url)
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    last_error: requests.exceptions.RequestException | None = None
    for image_format in _openai_attempt_image_formats(settings):
        image_url = encode_frame_to_data_url(frame, image_format=image_format)
        payload = {
            "model": settings.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 1200,
        }
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=settings.request_timeout_seconds,
            )
            break
        except requests.exceptions.RequestException as exc:
            last_error = exc
    else:
        raise QwenAnalysisError(
            f"OpenAI 兼容接口网络错误：{last_error}。"
            "视频/直播会在失败后自动冷却再重试；如仍频繁出现，可降低 OPENAI_IMAGE_MAX_WIDTH 或增大 QWEN_UPLOAD_INTERVAL_SECONDS。"
        ) from last_error

    if response.status_code >= 400:
        try:
            error_payload = response.json()
            message = error_payload.get("error", {}).get("message") or str(error_payload)
        except ValueError:
            message = response.text
        raise QwenAnalysisError(
            f"OpenAI 兼容接口错误：HTTP {response.status_code}，{message}。当前端点：{settings.base_http_api_url}"
        )
    try:
        return _extract_openai_message_text(response.json())
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise QwenAnalysisError(f"OpenAI 兼容接口返回格式异常：{response.text}") from exc


def call_openai_compatible_vision(frame, settings: QwenSettings) -> QwenAnalysisResult:
    height, width = frame.shape[:2]
    prompt = build_qwen_prompt(width=width, height=height, use_coordinate_grid=settings.use_coordinate_grid)
    text = _request_openai_compatible_text(frame, settings, prompt)
    return parse_qwen_response(text, width=width, height=height)


def _request_dashscope_text(frame, settings: QwenSettings, prompt: str) -> str:
    if not settings.api_key:
        raise QwenAnalysisError("未配置 DASHSCOPE_API_KEY，无法调用千问")

    try:
        import dashscope
    except ImportError as exc:
        raise QwenAnalysisError("未安装 dashscope，请先安装 requirements.txt 中的依赖") from exc

    dashscope.base_http_api_url = settings.base_http_api_url
    image_url = encode_frame_to_data_url(frame)
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
    return text


def _request_vision_text(frame, settings: QwenSettings, prompt: str) -> str:
    if settings.provider == "openai":
        return _request_openai_compatible_text(frame, settings, prompt)
    return _request_dashscope_text(frame, settings, prompt)


def call_qwen_vision(frame, settings: QwenSettings) -> QwenAnalysisResult:
    height, width = frame.shape[:2]
    prompt = build_qwen_prompt(width=width, height=height, use_coordinate_grid=settings.use_coordinate_grid)
    text = _request_vision_text(frame, settings, prompt)
    return parse_qwen_response(text, width=width, height=height)


def call_person_crop_vision(
    frame,
    source_by_id: dict[int, Detection],
    settings: QwenSettings,
) -> QwenAnalysisResult:
    if not source_by_id:
        return QwenAnalysisResult(people=[], summary="未检测到人体。", raw_text="{}")
    prompt = build_person_crop_prompt(sorted(source_by_id))
    text = _request_vision_text(frame, settings, prompt)
    return parse_person_crop_response(text, source_by_id)
