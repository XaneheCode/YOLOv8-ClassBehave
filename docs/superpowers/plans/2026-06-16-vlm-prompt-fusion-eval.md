# VLM Prompt Fusion Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the six-class VLM behaviour evaluation by tightening the person-crop prompt and adding a reproducible GPT/Qwen fusion baseline.

**Architecture:** Keep raw GPT and Qwen predictions unchanged. Add a deterministic fusion layer in the evaluation script that combines both predictions into a third comparable provider named `fusion`, then summarize all three in the same accuracy CSV. Update the shared person-crop prompt so GUI and offline evaluation use the same clearer behaviour rules. The fusion layer is conservative: Qwen can only override GPT's `Head-down` or missing result with `Useing-Phone` when Qwen's note contains phone-like evidence.

**Tech Stack:** Python 3.12, pytest, OpenCV, existing `src.backend.qwen_analysis` request/parsing utilities.

---

### Task 1: Prompt Rules

**Files:**
- Modify: `src/backend/qwen_analysis.py`
- Test: `tests/test_qwen_analysis.py`

- [ ] Add a test that `build_person_crop_prompt` explicitly distinguishes phone, writing, reading, head-down, sleeping, and hand-raise.
- [ ] Update the prompt text with stricter phone-vs-writing and phone-vs-reading rules.
- [ ] Run the targeted prompt test.

### Task 2: Fusion Strategy

**Files:**
- Modify: `scripts/run_vlm_behaviour_eval.py`
- Test: `tests/test_vlm_behaviour_eval.py`

- [ ] Add tests for deterministic fusion:
  - GPT phone remains phone.
  - Qwen phone can override GPT head-down only when the Qwen note contains phone-like evidence.
  - GPT writing/reading remains unchanged even when Qwen says phone, because the first evaluation showed too many writing-to-phone false positives.
  - Hand-raise and Sleeping are not overridden by phone unless GPT already predicts phone.
- [ ] Implement `fuse_predictions`.
- [ ] Emit `fusion` rows into `vlm_predictions_comparison.csv`.
- [ ] Include fusion in `vlm_accuracy_summary.csv`.

### Task 3: Evaluation Record

**Files:**
- Modify: `docs/course-evidence/vlm-behaviour-eval-v1.md`

- [ ] Re-run the 84-target evaluation after prompt changes.
- [ ] Update the document with GPT, Qwen, and fusion results.
- [ ] Record the main remaining failure modes.
