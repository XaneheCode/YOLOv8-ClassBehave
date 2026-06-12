import json
import os
import sys
import types
from unittest import mock

import numpy as np
import pytest

from src.backend.qwen_analysis import (
    QwenAnalysisError,
    QwenAnalysisResult,
    QwenSettings,
    build_qwen_prompt,
    call_qwen_vision,
    clamp_bbox,
    extract_json_object,
    load_qwen_settings,
    parse_qwen_response,
    prepare_frame_for_qwen,
    should_upload_frame,
    should_use_qwen_for_scene,
)


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
                {"bbox": [10, 20, 30, 40], "status": "有效框"},
            ],
            "summary": "mixed",
        },
        ensure_ascii=False,
    )

    result = parse_qwen_response(raw, width=100, height=100)

    assert len(result.people) == 1
    assert result.people[0].bbox == [10, 20, 30, 40]


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


def test_prompt_includes_dimensions_and_person_box_instruction():
    prompt = build_qwen_prompt(width=640, height=480, use_coordinate_grid=True)

    assert "640" in prompt
    assert "480" in prompt
    assert "坐标网格" in prompt
    assert "人物可见身体" in prompt
    assert "不要框住桌子、椅子、电脑" in prompt


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
