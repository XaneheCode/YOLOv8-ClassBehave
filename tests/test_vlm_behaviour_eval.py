from scripts.run_vlm_behaviour_eval import (
    evaluation_label,
    fuse_provider_rows,
    is_strict_correct,
    prediction_soft_score,
    summarize,
)


def _row(provider, person_id, truth, pred, note="", confidence="medium"):
    return {
        "provider": provider,
        "grid_id": "grid_a",
        "person_id": str(person_id),
        "truth_label": truth,
        "pred_label": pred,
        "correct": "0",
        "pred_confidence": confidence,
        "pred_note": note,
        "summary": "",
        "error": "",
        "raw_response_file": "",
    }


def test_fusion_keeps_gpt_phone_prediction():
    rows = [
        _row("gpt", 1, "Useing-Phone", "Useing-Phone", "手里有手机"),
        _row("qwen", 1, "Useing-Phone", "Writing", "正在写字"),
    ]

    fused = fuse_provider_rows(rows)

    assert fused[0]["provider"] == "fusion"
    assert fused[0]["pred_label"] == "Useing-Phone"
    assert fused[0]["correct"] == "1"


def test_fusion_uses_qwen_phone_for_gpt_head_down_when_note_has_phone_evidence():
    rows = [
        _row("gpt", 1, "Useing-Phone", "Head-down", "低头"),
        _row("qwen", 1, "Useing-Phone", "Useing-Phone", "双手拿着手机小屏幕"),
    ]

    fused = fuse_provider_rows(rows)

    assert fused[0]["pred_label"] == "Useing-Phone"
    assert fused[0]["correct"] == "1"


def test_fusion_rejects_qwen_phone_without_concrete_phone_evidence():
    rows = [
        _row("gpt", 1, "Writing", "Writing", "低头写字"),
        _row("qwen", 1, "Writing", "Useing-Phone", "低头看桌面"),
    ]

    fused = fuse_provider_rows(rows)

    assert fused[0]["pred_label"] == "Writing"
    assert fused[0]["correct"] == "1"


def test_fusion_does_not_override_gpt_writing_with_qwen_phone():
    rows = [
        _row("gpt", 1, "Writing", "Writing", "低头写字"),
        _row("qwen", 1, "Writing", "Useing-Phone", "低头看手机"),
    ]

    fused = fuse_provider_rows(rows)

    assert fused[0]["pred_label"] == "Writing"
    assert fused[0]["correct"] == "1"


def test_fusion_does_not_override_sleeping_or_hand_raise_with_qwen_phone():
    rows = [
        _row("gpt", 1, "Sleeping", "Sleeping", "趴在桌面睡觉"),
        _row("qwen", 1, "Sleeping", "Useing-Phone", "手中疑似手机"),
        _row("gpt", 2, "Hand-raise", "Hand-raise", "举手"),
        _row("qwen", 2, "Hand-raise", "Useing-Phone", "手里疑似手机"),
    ]

    fused = fuse_provider_rows(rows)

    assert [row["pred_label"] for row in fused] == ["Sleeping", "Hand-raise"]
    assert [row["correct"] for row in fused] == ["1", "1"]


def test_evaluation_label_merges_writing_and_reading_as_learning():
    assert evaluation_label("Writing") == "Learning"
    assert evaluation_label("Reading") == "Learning"
    assert evaluation_label("Useing-Phone") == "Useing-Phone"


def test_strict_correct_uses_display_learning_merge():
    assert is_strict_correct("Writing", "Reading") is True
    assert is_strict_correct("Reading", "Writing") is True
    assert is_strict_correct("Writing", "Useing-Phone") is False


def test_prediction_soft_score_uses_relaxed_report_scoring():
    assert prediction_soft_score("Writing", "Writing") == 1.4
    assert prediction_soft_score("Useing-Phone", "Useing-Phone") == 1.4
    assert prediction_soft_score("Writing", "Reading") == 1.4
    assert prediction_soft_score("Reading", "Writing") == 1.4
    assert prediction_soft_score("Writing", "Head-down") == 0.3
    assert prediction_soft_score("Useing-Phone", "Head-down") == 0.3
    assert prediction_soft_score("Useing-Phone", "Writing") == 0.2
    assert prediction_soft_score("Useing-Phone", "Reading") == 0.2
    assert prediction_soft_score("Sleeping", "Head-down") == 0.0
    assert prediction_soft_score("Head-down", "Sleeping") == 0.0
    assert prediction_soft_score("Head-down", "Useing-Phone") == 0.0


def test_summarize_reports_weighted_accuracy():
    rows = [
        _row("gpt", 1, "Writing", "Head-down"),
        _row("gpt", 2, "Useing-Phone", "Head-down"),
        _row("gpt", 3, "Reading", "Head-down"),
        _row("gpt", 4, "Writing", "Writing"),
    ]
    for row in rows:
        row["correct"] = "1" if row["truth_label"] == row["pred_label"] else "0"
        row["soft_score"] = prediction_soft_score(row["truth_label"], row["pred_label"])

    summary = summarize(rows)
    all_row = next(row for row in summary if row["provider"] == "gpt" and row["label"] == "ALL")

    assert all_row["correct"] == 1
    assert all_row["soft_score"] == 2.0
    assert all_row["weighted_accuracy"] == 0.5
    assert all_row["normalized_weighted_accuracy"] == 0.3571
