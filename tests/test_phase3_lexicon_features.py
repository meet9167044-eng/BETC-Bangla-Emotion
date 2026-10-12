"""
tests/test_phase3_lexicon_features.py
=====================================
Regression tests for Phase 3 Lexicon Investigation:
  - Verifies presence and valid formatting of sentiment lexicon
  - Verifies 6-emotion lexicon covers all target emotions
  - Verifies dense emotion lexicon feature extraction (18 dims, no NaNs/Infs)
"""

import os
import json
import numpy as np
from src.models.betc import TARGET_COLS

def test_sentiment_lexicon_format():
    lex_path = os.path.join("configs", "sentiment_lexicon_bn.txt")
    assert os.path.exists(lex_path), "configs/sentiment_lexicon_bn.txt not found"
    
    pos_count = 0
    neg_count = 0
    with open(lex_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(maxsplit=1)
            assert len(parts) == 2, f"Invalid format in lexicon: {line}"
            w, pol = parts
            assert pol in ("pos", "neg"), f"Unknown polarity: {pol}"
            if pol == "pos":
                pos_count += 1
            else:
                neg_count += 1
    assert pos_count > 20, "Too few positive words in lexicon"
    assert neg_count > 20, "Too few negative words in lexicon"

def test_emotion_lexicon_coverage():
    from scripts.experiments.phase3_lexicon_study import EMOTION_LEXICON, extract_emotion_lexicon_features
    for emo in TARGET_COLS:
        assert emo in EMOTION_LEXICON, f"Missing emotion category in EMOTION_LEXICON: {emo}"
        assert len(EMOTION_LEXICON[emo]) >= 5, f"Too few terms for emotion: {emo}"

    sample_texts = [
        "খুব সুন্দর এবং চমৎকার খবর!",
        "ছি ছি কি জঘন্য কাজ!",
        "ভয়ানক বিপদ আসছে সামনে।"
    ]
    vocab_set = {k: set(v) for k, v in EMOTION_LEXICON.items()}
    feats = extract_emotion_lexicon_features(sample_texts, vocab_set)
    assert feats.shape == (len(sample_texts), 18)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()
