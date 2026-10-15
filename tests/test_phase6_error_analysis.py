"""
tests/test_phase6_error_analysis.py
=====================================
Regression tests for Phase 6 Error Analysis utilities.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE)

from scripts.experiments.phase6_error_analysis import (
    has_negation,
    is_code_mixed,
    classify_error_patterns,
    confusion_matrix_pairs,
    summarize_errors,
    sensitivity_analysis_emonoba,
)
from src.models.betc import TARGET_COLS


# ── has_negation ──────────────────────────────────────────────────────────────
def test_has_negation_positive():
    text = "আমি ভালো না"
    assert has_negation(text, ["না", "নয়"]) is True


def test_has_negation_negative():
    text = "আমি ভালো আছি"
    assert has_negation(text, ["না", "নয়"]) is False


# ── is_code_mixed ─────────────────────────────────────────────────────────────
def test_is_code_mixed_yes():
    text = "আমি ভালো আছি and I love it very much"
    assert is_code_mixed(text) is True


def test_is_code_mixed_no():
    text = "আমি ভালো আছি"
    assert is_code_mixed(text) is False


# ── classify_error_patterns ───────────────────────────────────────────────────
def test_classify_error_patterns_shapes():
    texts_raw  = ["hi there", "আমি যাব না আর কখনো", "#happy জয় জয়"]
    texts_proc = ["hi there", "আমি যাব না আর কখনো", "happy জয় জয়"]
    neg_vocab = ["না", "নয়"]
    patterns = classify_error_patterns(texts_raw, texts_proc, neg_vocab)
    assert set(patterns.keys()) == {"short_text", "has_negation", "has_emoji", "has_hashtag", "code_mixed"}
    for arr in patterns.values():
        assert len(arr) == 3
        assert arr.dtype == bool


def test_classify_error_patterns_short_text():
    texts_raw  = ["ok", "আমি ভালো আছি এবং সুখী"]
    texts_proc = ["ok", "আমি ভালো আছি এবং সুখী"]
    neg_vocab  = []
    patterns = classify_error_patterns(texts_raw, texts_proc, neg_vocab)
    assert patterns["short_text"][0] == True   # "ok" has 1 word < 5
    assert patterns["short_text"][1] == False  # 5 words, not < 5


def test_classify_error_patterns_hashtag():
    texts_raw  = ["#আনন্দ খুব ভালো", "কোন ট্যাগ নেই"]
    texts_proc = ["আনন্দ খুব ভালো",   "কোন ট্যাগ নেই"]
    neg_vocab  = []
    patterns = classify_error_patterns(texts_raw, texts_proc, neg_vocab)
    assert patterns["has_hashtag"][0] == True
    assert patterns["has_hashtag"][1] == False


# ── confusion_matrix_pairs ────────────────────────────────────────────────────
def test_confusion_matrix_pairs():
    label_names = ["anger", "disgust"]
    Y_true = np.array([[1, 0], [0, 1], [1, 1]])
    Y_pred = np.array([[0, 1], [0, 1], [1, 0]])
    co_occ_true, fp_confusion = confusion_matrix_pairs(Y_true, Y_pred, label_names)
    # Row [1,1] true has both anger and disgust
    assert co_occ_true["anger"]["disgust"] == 1
    assert co_occ_true["disgust"]["anger"] == 1
    # Row [1,0] true=anger, pred=disgust → FP on disgust, actual is anger
    assert fp_confusion["disgust"]["anger"] >= 1


# ── summarize_errors ──────────────────────────────────────────────────────────
def test_summarize_errors():
    n = 20
    Y_true = np.zeros((n, 6), dtype=int)
    Y_pred = np.zeros((n, 6), dtype=int)
    # 5 FP for disgust (col 1): true=0, pred=1
    Y_pred[:5, 1] = 1
    # 3 FN for disgust: true=1, pred=0
    Y_true[10:13, 1] = 1

    df = pd.DataFrame({
        "text": [f"sample {i}" for i in range(n)],
        "processed_text": [f"sample {i}" for i in range(n)],
        "source_dataset": ["DS1"] * n,
        "emonoba_disgust_assumption": [False] * n,
        **{c: Y_true[:, i] for i, c in enumerate(TARGET_COLS)}
    })

    patterns = {
        "short_text": np.zeros(n, dtype=bool),
        "has_negation": np.zeros(n, dtype=bool),
        "has_emoji": np.zeros(n, dtype=bool),
        "has_hashtag": np.zeros(n, dtype=bool),
        "code_mixed": np.zeros(n, dtype=bool),
    }

    fp_df, fn_df, summary = summarize_errors("disgust", Y_true, Y_pred, df, patterns, "disgust")
    assert summary["total_fp"] == 5
    assert summary["total_fn"] == 3
    assert len(fp_df) == 5
    assert len(fn_df) == 3


# ── sensitivity_analysis_emonoba ──────────────────────────────────────────────
def test_sensitivity_analysis_emonoba():
    n = 20
    Y_true = np.zeros((n, 6), dtype=int)
    Y_pred = np.zeros((n, 6), dtype=int)
    # First 10 rows: emonoba assumption True; disgust true and predicted
    Y_true[:10, 1] = 1
    Y_pred[:10, 1] = 1
    # Rows 10–20: emonoba assumption False; disgust true but NOT predicted (FN)
    Y_true[10:, 1] = 1
    Y_pred[10:, 1] = 0

    df = pd.DataFrame({
        "emonoba_disgust_assumption": [True] * 10 + [False] * 10,
        **{c: Y_true[:, i] for i, c in enumerate(TARGET_COLS)}
    })

    result = sensitivity_analysis_emonoba("disgust", Y_true, Y_pred, df)
    assert result["emonoba_assumption_rows"]["f1"] == pytest.approx(1.0)
    assert result["other_rows"]["f1"] == pytest.approx(0.0)
    assert result["emonoba_assumption_rows"]["n"] == 10
    assert result["other_rows"]["n"] == 10
