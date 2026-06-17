import json
import os
import requests
import sys
import types
from unittest import mock

import numpy as np
import pytest

from src.backend.qwen_analysis import (
    QwenAnalysisError,
    QwenAnalysisResult,
    QwenSettings,
    build_person_crop_prompt,
    build_qwen_prompt,
    call_person_crop_vision,
    call_qwen_vision,
    clamp_bbox,
    encode_frame_to_data_url,
    extract_json_object,
    load_qwen_settings,
    parse_person_crop_response,
    parse_qwen_response,
    prepare_frame_for_qwen,
    should_upload_frame,
    should_use_qwen_for_scene,
)
from src.common.types import Detection


def test_extracts_plain_json_object():
    text = '{"people": [], "summary": "empty"}'
    assert extract_json_object(text) == text


def test_extracts_fenced_json_object():
    text = '```json\n{"people": [], "summary": "ok"}\n```'
    assert extract_json_object(text) == '{"people": [], "summary": "ok"}'


def test_parse_valid_people_and_summary():
    raw = json.dumps(
        {
            "people": [
                {
                    "bbox": [-5, 20, 999, 240],
                    "label": "Writing",
                    "status": "正在听讲",
                    "confidence": "high",
                }
            ],
            "summary": "画面中有一名学生。",
        },
        ensure_ascii=False,
    )

    result = parse_qwen_response(raw, width=640, height=360)

    assert len(result.people) == 1
    assert result.people[0].bbox == [0, 20, 639, 240]
    assert result.people[0].label == "Writing"
    assert result.people[0].status == "正在听讲"
    assert result.people[0].confidence == "high"
    assert result.summary == "画面中有一名学生。"
    assert result.raw_text == raw


def test_invalid_json_raises_analysis_error():
    with pytest.raises(QwenAnalysisError, match="无法解析千问 JSON"):
        parse_qwen_response("不是 JSON", width=640, height=360)


def test_invalid_bbox_is_ignored():
    raw = json.dumps(
        {
            "people": [
                {"bbox": [50, 50, 10, 80], "status": "无效框"},
                {"bbox": [10, 20, 30, 40], "label": "Reading", "status": "有效框"},
            ],
            "summary": "mixed",
        },
        ensure_ascii=False,
    )

    result = parse_qwen_response(raw, width=100, height=100)

    assert len(result.people) == 1
    assert result.people[0].bbox == [10, 20, 30, 40]


def test_parse_maps_only_supported_six_class_labels():
    raw = json.dumps(
        {
            "people": [
                {"bbox": [10, 20, 30, 40], "label": "写字", "status": "正在写字"},
                {"bbox": [40, 50, 70, 90], "label": "phone", "status": "看手机"},
                {"bbox": [80, 10, 95, 35], "label": "Standing", "status": "站立"},
            ],
            "summary": "mixed",
        },
        ensure_ascii=False,
    )

    result = parse_qwen_response(raw, width=100, height=100)

    assert [person.label for person in result.people] == ["Writing", "Useing-Phone"]


def test_phone_status_overrides_head_down_label():
    raw = json.dumps(
        {
            "people": [
                {
                    "bbox": [10, 20, 40, 80],
                    "label": "Head-down",
                    "status": "低头看手机，手里拿着小屏幕设备",
                    "confidence": "medium",
                }
            ],
            "summary": "phone",
        },
        ensure_ascii=False,
    )

    result = parse_qwen_response(raw, width=100, height=100)

    assert len(result.people) == 1
    assert result.people[0].label == "Useing-Phone"


def test_clamp_bbox_rejects_bad_shapes():
    assert clamp_bbox([1, 2, 3], width=100, height=100) is None
    assert clamp_bbox(["x", 2, 3, 4], width=100, height=100) is None


def test_should_upload_frame_respects_interval_and_in_flight():
    assert should_upload_frame(now=10.0, last_upload_at=None, interval_seconds=10, in_flight=False) is True
    assert should_upload_frame(now=15.0, last_upload_at=10.0, interval_seconds=10, in_flight=False) is False
    assert should_upload_frame(now=20.1, last_upload_at=10.0, interval_seconds=10, in_flight=False) is True
    assert should_upload_frame(now=25.0, last_upload_at=10.0, interval_seconds=10, in_flight=True) is False


def test_should_use_qwen_only_for_sparse_scenes():
    assert should_use_qwen_for_scene(target_count=0, max_targets=3) is True
    assert should_use_qwen_for_scene(target_count=3, max_targets=3) is True
    assert should_use_qwen_for_scene(target_count=4, max_targets=3) is False


@mock.patch.dict(
    os.environ,
    {
        "DASHSCOPE_API_KEY": "test-key",
        "QWEN_VL_MODEL": "qwen-vl-plus",
        "QWEN_UPLOAD_INTERVAL_SECONDS": "12",
        "QWEN_MAX_YOLO_TARGETS": "2",
    },
    clear=True,
)
def test_load_settings_from_environment():
    settings = load_qwen_settings()

    assert settings.api_key == "test-key"
    assert settings.model == "qwen-vl-plus"
    assert settings.interval_seconds == 12
    assert settings.max_yolo_targets == 2


@mock.patch.dict(os.environ, {}, clear=True)
def test_default_model_is_qwen_flash():
    settings = load_qwen_settings()

    assert settings.model == "qwen3.6-flash"


@mock.patch.dict(
    os.environ,
    {
        "VISION_PROVIDER": "openai",
        "OPENAI_API_KEY": "openai-key",
        "OPENAI_BASE_URL": "https://ai.laodog.top/",
        "OPENAI_VISION_MODEL": "gpt-5.5",
        "OPENAI_TIMEOUT_SECONDS": "150",
        "OPENAI_IMAGE_MAX_WIDTH": "720",
        "QWEN_UPLOAD_INTERVAL_SECONDS": "7",
    },
    clear=True,
)
def test_load_openai_compatible_settings_from_environment():
    settings = load_qwen_settings()

    assert settings.provider == "openai"
    assert settings.api_key == "openai-key"
    assert settings.model == "gpt-5.5"
    assert settings.base_http_api_url == "https://ai.laodog.top/"
    assert settings.interval_seconds == 7
    assert settings.request_timeout_seconds == 150
    assert settings.max_image_width == 720
    assert settings.openai_image_format == "png"


@mock.patch.dict(
    os.environ,
    {
        "VISION_PROVIDER": "openai",
        "OPENAI_API_KEY": "openai-key",
    },
    clear=True,
)
def test_openai_compatible_settings_default_to_smaller_upload_and_longer_timeout():
    settings = load_qwen_settings()

    assert settings.request_timeout_seconds == 120
    assert settings.max_image_width == 640


def test_encode_frame_to_png_data_url():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    data_url = encode_frame_to_data_url(frame, image_format="png")

    assert data_url.startswith("data:image/png;base64,")


def test_prepare_frame_for_openai_resizes_before_grid():
    frame = np.zeros((300, 900, 3), dtype=np.uint8)
    settings = QwenSettings(
        api_key="openai-key",
        model="gpt-5.5",
        interval_seconds=10,
        base_http_api_url="https://ai.laodog.top/",
        use_coordinate_grid=True,
        provider="openai",
        max_image_width=600,
    )

    prepared = prepare_frame_for_qwen(frame, settings)

    assert prepared.shape[:2] == (200, 600)
    assert int(prepared.sum()) > 0


def test_call_qwen_disables_thinking_mode():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    settings = QwenSettings(
        api_key="test-key",
        model="qwen3.6-flash",
        interval_seconds=10,
        base_http_api_url="https://dashscope.aliyuncs.com/api/v1",
    )
    fake_response = types.SimpleNamespace(
        output=types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=[{"text": '{"people": [], "summary": "ok"}'}])
                )
            ]
        )
    )
    fake_dashscope = types.SimpleNamespace(
        base_http_api_url="",
        MultiModalConversation=types.SimpleNamespace(call=mock.Mock(return_value=fake_response)),
    )

    with mock.patch.dict(sys.modules, {"dashscope": fake_dashscope}):
        result = call_qwen_vision(frame, settings)

    assert isinstance(result, QwenAnalysisResult)
    assert result.summary == "ok"
    fake_dashscope.MultiModalConversation.call.assert_called_once()
    kwargs = fake_dashscope.MultiModalConversation.call.call_args.kwargs
    assert kwargs["model"] == "qwen3.6-flash"
    assert kwargs["enable_thinking"] is False
    assert fake_dashscope.base_http_api_url == "https://dashscope.aliyuncs.com/api/v1"


def test_call_qwen_reports_api_status_errors_with_endpoint():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    settings = QwenSettings(
        api_key="test-key",
        model="qwen3.6-flash",
        interval_seconds=10,
        base_http_api_url="https://dashscope.aliyuncs.com/api/v1",
    )
    fake_response = types.SimpleNamespace(
        status_code=401,
        code="InvalidApiKey",
        message="Invalid API-key provided.",
        output=None,
    )
    fake_dashscope = types.SimpleNamespace(
        base_http_api_url="",
        MultiModalConversation=types.SimpleNamespace(call=mock.Mock(return_value=fake_response)),
    )

    with mock.patch.dict(sys.modules, {"dashscope": fake_dashscope}):
        with pytest.raises(QwenAnalysisError) as exc:
            call_qwen_vision(frame, settings)

    assert "InvalidApiKey" in str(exc.value)
    assert "https://dashscope.aliyuncs.com/api/v1" in str(exc.value)


def test_call_openai_compatible_vision_posts_png_data_url(monkeypatch):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    settings = QwenSettings(
        api_key="openai-key",
        model="gpt-5.5",
        interval_seconds=10,
        base_http_api_url="https://ai.laodog.top/",
        provider="openai",
        request_timeout_seconds=150,
    )
    captured = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {"choices": [{"message": {"content": '{"people": [], "summary": "ok"}'}}]}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.backend.qwen_analysis.requests.post", fake_post)

    result = call_qwen_vision(frame, settings)

    assert result.summary == "ok"
    assert captured["url"] == "https://ai.laodog.top/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer openai-key"
    assert captured["json"]["model"] == "gpt-5.5"
    assert captured["timeout"] == 150
    content = captured["json"]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_call_openai_compatible_vision_retries_ssl_network_errors(monkeypatch):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    settings = QwenSettings(
        api_key="openai-key",
        model="gpt-5.5",
        interval_seconds=10,
        base_http_api_url="https://ai.laodog.top/",
        provider="openai",
        request_timeout_seconds=150,
    )
    calls = []

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {"choices": [{"message": {"content": '{"people": [], "summary": "ok"}'}}]}

    data_urls = []

    def fake_post(url, headers, json, timeout):
        calls.append(headers)
        data_urls.append(json["messages"][0]["content"][1]["image_url"]["url"])
        if len(calls) == 1:
            raise requests.exceptions.SSLError("unexpected eof")
        return FakeResponse()

    monkeypatch.setattr("src.backend.qwen_analysis.requests.post", fake_post)

    result = call_qwen_vision(frame, settings)

    assert result.summary == "ok"
    assert len(calls) == 2
    assert calls[0]["Connection"] == "close"
    assert data_urls[0].startswith("data:image/png;base64,")
    assert data_urls[1].startswith("data:image/jpeg;base64,")


def test_call_openai_compatible_vision_reports_network_errors_after_retries(monkeypatch):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    settings = QwenSettings(
        api_key="openai-key",
        model="gpt-5.5",
        interval_seconds=10,
        base_http_api_url="https://ai.laodog.top/",
        provider="openai",
        request_timeout_seconds=150,
    )
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(1)
        raise requests.exceptions.SSLError("unexpected eof")

    monkeypatch.setattr("src.backend.qwen_analysis.requests.post", fake_post)

    with pytest.raises(QwenAnalysisError) as exc:
        call_qwen_vision(frame, settings)

    assert len(calls) == 2
    assert "OpenAI 兼容接口网络错误" in str(exc.value)


def test_parse_person_crop_response_maps_ids_back_to_source_boxes():
    source_by_id = {
        1: Detection(label="person", confidence=0.91, bbox=(10, 20, 30, 60)),
        2: Detection(label="person", confidence=0.82, bbox=(70, 10, 100, 50)),
    }
    raw = json.dumps(
        {
            "people": [
                {"id": 1, "label": "Writing", "status": "正在写字", "confidence": "high"},
                {"id": 2, "label": "看手机", "status": "低头看手机", "confidence": "medium"},
                {"id": 99, "label": "Sleeping", "status": "不存在编号", "confidence": "high"},
            ],
            "summary": "两名学生。",
        },
        ensure_ascii=False,
    )

    result = parse_person_crop_response(raw, source_by_id)

    assert [(person.bbox, person.label) for person in result.people] == [
        ([10, 20, 30, 60], "Writing"),
        ([70, 10, 100, 50], "Useing-Phone"),
    ]


def test_build_person_crop_prompt_requests_numbered_classification():
    prompt = build_person_crop_prompt([1, 2])

    assert "编号为 1, 2" in prompt
    assert '"id": 1' in prompt
    assert "Useing-Phone 优先" in prompt
    assert "不要返回 bbox" in prompt


def test_build_person_crop_prompt_distinguishes_ambiguous_phone_and_study_actions():
    prompt = build_person_crop_prompt([1, 2])

    assert "看不到手机或小屏幕时，不要仅因为低头就标为 Useing-Phone" in prompt
    assert "只有清楚看到笔尖、握笔写字动作或纸面书写区域时，才标 Writing" in prompt
    assert "只是在看纸张、书本、讲义或屏幕资料，没有明显写字动作时，标 Reading" in prompt
    assert "手臂明显高于肩部或头部" in prompt
    assert "趴在桌面、闭眼或明显休息" in prompt


def test_call_person_crop_vision_posts_numbered_crop_prompt(monkeypatch):
    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    settings = QwenSettings(
        api_key="openai-key",
        model="gpt-5.5",
        interval_seconds=10,
        base_http_api_url="https://ai.laodog.top/",
        provider="openai",
        request_timeout_seconds=150,
    )
    source_by_id = {1: Detection(label="person", confidence=0.91, bbox=(10, 20, 30, 60))}
    captured = {}

    class FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"people": [{"id": 1, "label": "Reading", "status": "看书", "confidence": "high"}], "summary": "ok"}'
                        }
                    }
                ]
            }

    def fake_post(url, headers, json, timeout):
        captured["prompt"] = json["messages"][0]["content"][0]["text"]
        return FakeResponse()

    monkeypatch.setattr("src.backend.qwen_analysis.requests.post", fake_post)

    result = call_person_crop_vision(frame, source_by_id, settings)

    assert result.people[0].bbox == [10, 20, 30, 60]
    assert result.people[0].label == "Reading"
    assert "不要返回 bbox" in captured["prompt"]


def test_prompt_includes_dimensions_and_person_box_instruction():
    prompt = build_qwen_prompt(width=640, height=480, use_coordinate_grid=True)

    assert "640" in prompt
    assert "480" in prompt
    assert "坐标网格" in prompt
    assert "人物可见身体" in prompt
    assert "不要框住桌子、椅子、电脑" in prompt
    assert '"label": "Hand-raise|Reading|Writing|Useing-Phone|Head-down|Sleeping"' in prompt
    assert "只能从以上六类中选择" in prompt
    assert "Useing-Phone 优先" in prompt
    assert "不要把看手机的人标为 Head-down" in prompt


def test_prepare_frame_for_qwen_draws_coordinate_grid():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    settings = QwenSettings(
        api_key="test-key",
        model="qwen3.6-flash",
        interval_seconds=10,
        base_http_api_url="https://dashscope.aliyuncs.com/api/v1",
        use_coordinate_grid=True,
    )

    prepared = prepare_frame_for_qwen(frame, settings)

    assert prepared.shape == frame.shape
    assert int(prepared.sum()) > 0


def test_prepare_frame_for_qwen_can_disable_grid():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    settings = QwenSettings(
        api_key="test-key",
        model="qwen3.6-flash",
        interval_seconds=10,
        base_http_api_url="https://dashscope.aliyuncs.com/api/v1",
        use_coordinate_grid=False,
    )

    prepared = prepare_frame_for_qwen(frame, settings)

    assert np.array_equal(prepared, frame)
