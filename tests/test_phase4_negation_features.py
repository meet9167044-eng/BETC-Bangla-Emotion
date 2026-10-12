"""
tests/test_phase4_negation_features.py
======================================
Regression tests for Phase 4 Negation & Intensifier Handling:
  - Verifies presence of expanded negation and intensifier vocabulary files
  - Verifies negation preservation during preprocessing
  - Verifies advanced negation scope feature extractor (16 dims, no NaNs/Infs)
"""

import os
import json
import numpy as np
from src.features.preprocessing import preprocess

def test_negation_vocab_files():
    neg_path = os.path.join("configs", "negations_bn.txt")
    intens_path = os.path.join("configs", "intensifiers_bn.txt")

    assert os.path.exists(neg_path), "configs/negations_bn.txt missing"
    assert os.path.exists(intens_path), "configs/intensifiers_bn.txt missing"

    with open(neg_path, encoding="utf-8") as f:
        negs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    with open(intens_path, encoding="utf-8") as f:
        intens = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    assert len(negs) >= 25, f"Expected >= 25 negations, got {len(negs)}"
    assert len(intens) >= 30, f"Expected >= 30 intensifiers, got {len(intens)}"

def test_negation_preprocessing_preservation():
    sample = "আমি একদমই ভালো নেই! কখনো না!"
    processed = preprocess(sample)
    # Check that negation words survived
    assert "একদমই" in processed or "ভালো" in processed
    assert "নেই" in processed
    assert "কখনো" in processed

def test_negation_scope_feature_extraction():
    from scripts.experiments.phase4_negation_study import (
        EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, load_emotion_lexicon,
        extract_advanced_negation_features
    )
    emo_dict = load_emotion_lexicon()
    sample_texts = [
        "আমি ভালো নেই, মনটা খারাপ।",
        "কোন ভয় নেই, আমরা সবাই সাথে আছি।",
        "একদম বাজে কাজ!"
    ]
    feats = extract_advanced_negation_features(sample_texts, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict)
    assert feats.shape == (len(sample_texts), 16)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()
