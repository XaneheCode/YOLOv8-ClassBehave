from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.qwen_analysis import (  # noqa: E402
    DEFAULT_OPENAI_API_URL,
    DEFAULT_OPENAI_IMAGE_MAX_WIDTH,
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
    DEFAULT_QWEN_API_URL,
    DEFAULT_QWEN_INTERVAL_SECONDS,
    DEFAULT_QWEN_MODEL,
    QWEN_BEHAVIOUR_LABELS,
    QwenAnalysisError,
    QwenSettings,
    _request_vision_text,
    build_person_crop_prompt,
    extract_json_object,
    normalize_qwen_label,
)


PHONE_EVIDENCE_KEYWORDS = (
    "手机",
    "小屏幕",
    "小矩形",
    "手持设备",
    "手中设备",
    "电子设备",
    "移动设备",
    "屏幕设备",
    "拿着",
    "持有",
)
PHONE_OVERRIDE_CANDIDATES = {"Head-down", ""}
PHONE_OVERRIDE_PROTECTED = {"Hand-raise", "Sleeping"}
FULL_MATCH_SCORE = 1.4
PARTIAL_HEAD_DOWN_TRUTHS = {"Writing", "Useing-Phone"}
PARTIAL_HEAD_DOWN_SCORE = 0.3
PHONE_AS_LEARNING_SCORE = 0.2
HEAD_DOWN_SLEEPING_NEAR_SCORE = 0.0
LEARNING_LABELS = {"Writing", "Reading"}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(name, value)


def read_positive_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def maybe_resize_for_provider(image: np.ndarray, settings: QwenSettings) -> np.ndarray:
    if settings.provider != "openai" or settings.max_image_width <= 0:
        return image
    height, width = image.shape[:2]
    if width <= settings.max_image_width:
        return image
    new_height = max(1, int(round(height * settings.max_image_width / width)))
    return cv2.resize(image, (settings.max_image_width, new_height), interpolation=cv2.INTER_AREA)


def provider_settings(provider: str) -> QwenSettings:
    if provider == "gpt":
        return QwenSettings(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-5.5").strip() or "gpt-5.5",
            interval_seconds=DEFAULT_QWEN_INTERVAL_SECONDS,
            base_http_api_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_API_URL).strip()
            or DEFAULT_OPENAI_API_URL,
            provider="openai",
            request_timeout_seconds=read_positive_int("OPENAI_TIMEOUT_SECONDS", DEFAULT_OPENAI_TIMEOUT_SECONDS),
            max_image_width=read_positive_int("OPENAI_IMAGE_MAX_WIDTH", DEFAULT_OPENAI_IMAGE_MAX_WIDTH),
            openai_image_format=os.getenv("OPENAI_IMAGE_FORMAT", "png").strip() or "png",
        )
    if provider == "qwen":
        return QwenSettings(
            api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            model=os.getenv("QWEN_VL_MODEL", DEFAULT_QWEN_MODEL).strip() or DEFAULT_QWEN_MODEL,
            interval_seconds=DEFAULT_QWEN_INTERVAL_SECONDS,
            base_http_api_url=os.getenv("DASHSCOPE_BASE_HTTP_API_URL", DEFAULT_QWEN_API_URL).strip()
            or DEFAULT_QWEN_API_URL,
            provider="dashscope",
        )
    raise ValueError(f"Unsupported provider: {provider}")


def parse_people(raw_text: str, target_ids: set[int]) -> tuple[dict[int, dict[str, str]], str]:
    payload = json.loads(extract_json_object(raw_text))
    predictions: dict[int, dict[str, str]] = {}
    for item in payload.get("people", []):
        if not isinstance(item, dict):
            continue
        try:
            person_id = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if person_id not in target_ids:
            continue
        label = normalize_qwen_label(item.get("label"))
        status_label = normalize_qwen_label(item.get("status"))
        if status_label == "Useing-Phone":
            label = status_label
        if label is None:
            label = status_label
        if label not in QWEN_BEHAVIOUR_LABELS:
            continue
        predictions[person_id] = {
            "pred_label": label,
            "pred_confidence": str(item.get("confidence") or "unknown"),
            "pred_note": str(item.get("status") or item.get("note") or ""),
        }
    return predictions, str(payload.get("summary") or "")


def load_ground_truth(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return [
        row
        for row in rows
        if row.get("ignore") != "1" and row.get("initial_label") in QWEN_BEHAVIOUR_LABELS
    ]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluation_label(label: str) -> str:
    return "Learning" if label in LEARNING_LABELS else label


def is_strict_correct(truth_label: str, pred_label: str) -> bool:
    return evaluation_label(truth_label) == evaluation_label(pred_label)


def prediction_soft_score(truth_label: str, pred_label: str) -> float:
    if is_strict_correct(truth_label, pred_label):
        return FULL_MATCH_SCORE
    if truth_label == "Useing-Phone" and evaluation_label(pred_label) == "Learning":
        return PHONE_AS_LEARNING_SCORE
    if {truth_label, pred_label} == {"Head-down", "Sleeping"}:
        return HEAD_DOWN_SLEEPING_NEAR_SCORE
    if truth_label in PARTIAL_HEAD_DOWN_TRUTHS and pred_label == "Head-down":
        return PARTIAL_HEAD_DOWN_SCORE
    return 0.0


def has_phone_evidence(text: str) -> bool:
    return any(keyword in text for keyword in PHONE_EVIDENCE_KEYWORDS)


def _fusion_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["grid_id"]), str(row["person_id"])


def _with_fusion_decision(
    base: dict[str, Any],
    pred_label: str,
    pred_confidence: str,
    pred_note: str,
) -> dict[str, Any]:
    return {
        "provider": "fusion",
        "grid_id": base["grid_id"],
        "person_id": base["person_id"],
        "truth_label": base["truth_label"],
        "pred_label": pred_label,
        "correct": "1" if is_strict_correct(str(base["truth_label"]), pred_label) else "0",
        "soft_score": prediction_soft_score(str(base["truth_label"]), pred_label),
        "pred_confidence": pred_confidence,
        "pred_note": pred_note,
        "summary": "",
        "error": "",
        "raw_response_file": "",
    }


def fuse_provider_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gpt_by_key = {_fusion_key(row): row for row in rows if row["provider"] == "gpt"}
    qwen_by_key = {_fusion_key(row): row for row in rows if row["provider"] == "qwen"}
    fused: list[dict[str, Any]] = []

    for key in sorted(gpt_by_key.keys() & qwen_by_key.keys()):
        gpt_row = gpt_by_key[key]
        qwen_row = qwen_by_key[key]
        gpt_label = str(gpt_row.get("pred_label") or "")
        qwen_label = str(qwen_row.get("pred_label") or "")
        qwen_note = str(qwen_row.get("pred_note") or "")

        if gpt_label == "Useing-Phone":
            fused.append(
                _with_fusion_decision(
                    gpt_row,
                    "Useing-Phone",
                    str(gpt_row.get("pred_confidence") or ""),
                    f"fusion: kept GPT phone; gpt={gpt_label}; qwen={qwen_label}; qwen_note={qwen_note}",
                )
            )
            continue

        if (
            qwen_label == "Useing-Phone"
            and gpt_label in PHONE_OVERRIDE_CANDIDATES
            and gpt_label not in PHONE_OVERRIDE_PROTECTED
            and has_phone_evidence(qwen_note)
        ):
            fused.append(
                _with_fusion_decision(
                    gpt_row,
                    "Useing-Phone",
                    str(qwen_row.get("pred_confidence") or ""),
                    f"fusion: qwen phone evidence override; gpt={gpt_label}; qwen_note={qwen_note}",
                )
            )
            continue

        if gpt_label:
            fused.append(
                _with_fusion_decision(
                    gpt_row,
                    gpt_label,
                    str(gpt_row.get("pred_confidence") or ""),
                    f"fusion: kept GPT; gpt={gpt_label}; qwen={qwen_label}; qwen_note={qwen_note}",
                )
            )
        else:
            fused.append(
                _with_fusion_decision(
                    qwen_row,
                    qwen_label,
                    str(qwen_row.get("pred_confidence") or ""),
                    f"fusion: GPT missing, used Qwen; qwen={qwen_label}; qwen_note={qwen_note}",
                )
            )

    return fused


def run_provider(dataset_dir: Path, provider: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    settings = provider_settings(provider)
    if not settings.api_key:
        raise QwenAnalysisError(f"{provider} API key is not configured")

    raw_dir = dataset_dir / "annotations" / "raw_responses" / provider
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows_by_grid: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        rows_by_grid.setdefault(row["grid_id"], []).append(row)

    output_rows: list[dict[str, Any]] = []
    for index, (grid_id, grid_rows) in enumerate(rows_by_grid.items(), start=1):
        target_ids = {int(row["person_id"]) for row in grid_rows}
        grid_path = dataset_dir / "grids" / f"{grid_id}.png"
        image = maybe_resize_for_provider(imread(grid_path), settings)
        prompt = build_person_crop_prompt(sorted(target_ids))
        print(f"{provider}: {index}/{len(rows_by_grid)} {grid_id} targets={len(target_ids)}")
        try:
            raw_text = _request_vision_text(image, settings, prompt)
            predictions, summary = parse_people(raw_text, target_ids)
            error = ""
        except Exception as exc:  # noqa: BLE001 - batch evaluation should record failures and continue.
            raw_text = ""
            predictions = {}
            summary = ""
            error = str(exc)

        raw_path = raw_dir / f"{grid_id}.txt"
        raw_path.write_text(raw_text or error, encoding="utf-8")

        for row in grid_rows:
            person_id = int(row["person_id"])
            pred = predictions.get(person_id, {})
            pred_label = pred.get("pred_label", "")
            output_rows.append(
                {
                    "provider": provider,
                    "grid_id": grid_id,
                    "person_id": person_id,
                    "truth_label": row["initial_label"],
                    "pred_label": pred_label,
                    "correct": "1" if is_strict_correct(str(row["initial_label"]), pred_label) else "0",
                    "soft_score": prediction_soft_score(str(row["initial_label"]), pred_label),
                    "pred_confidence": pred.get("pred_confidence", ""),
                    "pred_note": pred.get("pred_note", ""),
                    "summary": summary,
                    "error": error,
                    "raw_response_file": str(raw_path),
                }
            )
    return output_rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    providers = sorted({row["provider"] for row in rows})
    summary_rows: list[dict[str, Any]] = []
    for provider in providers:
        provider_rows = [row for row in rows if row["provider"] == provider]
        total = len(provider_rows)
        correct = sum(1 for row in provider_rows if row["correct"] == "1")
        soft_score = round(sum(float(row.get("soft_score") or 0) for row in provider_rows), 4)
        missing = sum(1 for row in provider_rows if not row["pred_label"])
        summary_rows.append(
            {
                "provider": provider,
                "label": "ALL",
                "total": total,
                "correct": correct,
                "accuracy": round(correct / total, 4) if total else 0,
                "soft_score": soft_score,
                "weighted_accuracy": round(soft_score / total, 4) if total else 0,
                "normalized_weighted_accuracy": round(soft_score / (total * FULL_MATCH_SCORE), 4) if total else 0,
                "missing": missing,
            }
        )
        for label in QWEN_BEHAVIOUR_LABELS:
            label_rows = [row for row in provider_rows if row["truth_label"] == label]
            total = len(label_rows)
            correct = sum(1 for row in label_rows if row["correct"] == "1")
            soft_score = round(sum(float(row.get("soft_score") or 0) for row in label_rows), 4)
            missing = sum(1 for row in label_rows if not row["pred_label"])
            summary_rows.append(
                {
                    "provider": provider,
                    "label": label,
                    "total": total,
                    "correct": correct,
                    "accuracy": round(correct / total, 4) if total else 0,
                    "soft_score": soft_score,
                    "weighted_accuracy": round(soft_score / total, 4) if total else 0,
                    "normalized_weighted_accuracy": round(soft_score / (total * FULL_MATCH_SCORE), 4)
                    if total
                    else 0,
                    "missing": missing,
                }
            )
    return summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen/GPT behaviour classification on numbered person grids.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "vlm-behaviour-eval-v1",
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="CSV with initial_label/review_label. Defaults to annotations/initial_human_seed_labels.csv.",
    )
    parser.add_argument("--providers", nargs="+", choices=["qwen", "gpt"], default=["qwen", "gpt"])
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    dataset_dir = args.dataset.resolve()
    truth_path = args.truth or dataset_dir / "annotations" / "initial_human_seed_labels.csv"
    truth_rows = load_ground_truth(truth_path)
    if not truth_rows:
        raise SystemExit(f"No valid truth rows found: {truth_path}")

    all_rows: list[dict[str, Any]] = []
    for provider in args.providers:
        all_rows.extend(run_provider(dataset_dir, provider, truth_rows))
    if {"qwen", "gpt"}.issubset(set(args.providers)):
        all_rows.extend(fuse_provider_rows(all_rows))

    prediction_fields = [
        "provider",
        "grid_id",
        "person_id",
        "truth_label",
        "pred_label",
        "correct",
        "soft_score",
        "pred_confidence",
        "pred_note",
        "summary",
        "error",
        "raw_response_file",
    ]
    summary_fields = [
        "provider",
        "label",
        "total",
        "correct",
        "accuracy",
        "soft_score",
        "weighted_accuracy",
        "normalized_weighted_accuracy",
        "missing",
    ]
    write_csv(dataset_dir / "annotations" / "vlm_predictions_comparison.csv", all_rows, prediction_fields)
    write_csv(dataset_dir / "annotations" / "vlm_accuracy_summary.csv", summarize(all_rows), summary_fields)
    for provider in sorted({row["provider"] for row in all_rows}):
        provider_rows = [row for row in all_rows if row["provider"] == provider]
        write_csv(
            dataset_dir / "annotations" / f"{provider}_predictions.csv",
            [
                {
                    "grid_id": row["grid_id"],
                    "person_id": row["person_id"],
                    "pred_label": row["pred_label"],
                    "pred_confidence": row["pred_confidence"],
                    "pred_note": row["pred_note"],
                    "raw_response_file": row["raw_response_file"],
                }
                for row in provider_rows
            ],
            ["grid_id", "person_id", "pred_label", "pred_confidence", "pred_note", "raw_response_file"],
        )
    print(f"wrote {dataset_dir / 'annotations' / 'vlm_predictions_comparison.csv'}")
    print(f"wrote {dataset_dir / 'annotations' / 'vlm_accuracy_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
