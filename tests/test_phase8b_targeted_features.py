"""
tests/test_phase8b_targeted_features.py
=========================================
Unit tests for Phase 8b feature extractors.

Tests are fast (no model fitting) and cover:
  - Group A: Interrogative / Rhetorical Question Markers
  - Group B: Discourse Contrast / Adversative Connectives
  - Group C: Intensifier-Emotion Co-occurrence Bigrams
"""

import numpy as np
import pytest
import sys
import os

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, WORKSPACE)

from scripts.experiments.phase8b_targeted_features import (
    extract_group_A_interrogative,
    extract_group_B_contrast,
    extract_group_C_intensifier_bigrams,
    _BANGLA_WH_WORDS,
    _CONTRAST_CONNECTIVES,
)
from scripts.experiments.phase3_lexicon_study import EMOTION_LEXICON


# ─── Helper ──────────────────────────────────────────────────────────────────
def emo_vocab_set():
    return {k: set(v) for k, v in EMOTION_LEXICON.items()}


# ── Group A Tests ─────────────────────────────────────────────────────────────

class TestGroupAInterrogative:
    def test_output_shape(self):
        texts = ["কেন তুমি এলে না?", "আমি ভালো আছি।"]
        feats = extract_group_A_interrogative(texts)
        assert feats.shape == (2, 8), f"Expected (2,8), got {feats.shape}"

    def test_wh_word_detected(self):
        # Text with clear WH-word কেন (why)
        texts = ["কেন তুমি এলে না?"]
        feats = extract_group_A_interrogative(texts)
        # feature 0: wh_count, feature 1: has_wh
        assert feats[0, 0] >= 1.0, "WH word count should be ≥1"
        assert feats[0, 1] == 1.0, "has_wh should be 1"

    def test_no_wh_in_plain_text(self):
        texts = ["আমি বাড়িতে আছি।"]
        feats = extract_group_A_interrogative(texts)
        assert feats[0, 0] == 0.0, "WH count should be 0 for plain text"
        assert feats[0, 1] == 0.0, "has_wh should be 0 for plain text"

    def test_question_mark_detected(self):
        texts = ["সে কোথায় গেল?"]
        feats = extract_group_A_interrogative(texts)
        # feature 2: qmark rate, feature 3: has_qmark
        assert feats[0, 3] == 1.0, "has_qmark should be 1"

    def test_exclamation_detected(self):
        texts = ["কী আশ্চর্য!"]
        feats = extract_group_A_interrogative(texts)
        # feature 4: has_excl
        assert feats[0, 4] == 1.0, "has_excl should be 1"

    def test_mixed_qexcl_detected(self):
        texts = ["সত্যিই?! এটা কি সম্ভব?!"]
        feats = extract_group_A_interrogative(texts)
        # feature 5: has_qexcl
        assert feats[0, 5] == 1.0, "has_qexcl should be 1 for ?!"

    def test_triple_qmark_detected(self):
        texts = ["এটা সত্যি???"]
        feats = extract_group_A_interrogative(texts)
        # feature 6: has_triple_q
        assert feats[0, 6] == 1.0, "has_triple_q should be 1 for ???"

    def test_ellipsis_detected(self):
        texts = ["সে চলে গেছে..."]
        feats = extract_group_A_interrogative(texts)
        # feature 7: has_ellipsis
        assert feats[0, 7] == 1.0, "has_ellipsis should be 1 for ..."

    def test_no_special_in_neutral(self):
        texts = ["আজকের আবহাওয়া ভালো।"]
        feats = extract_group_A_interrogative(texts)
        # No special punctuation, no WH words
        assert feats[0, 1] == 0.0, "has_wh should be 0"
        assert feats[0, 3] == 0.0, "has_qmark should be 0"
        assert feats[0, 4] == 0.0, "has_excl should be 0"

    def test_dtype_float32(self):
        texts = ["test text"]
        feats = extract_group_A_interrogative(texts)
        assert feats.dtype == np.float32, "Output dtype should be float32"


# ── Group B Tests ─────────────────────────────────────────────────────────────

class TestGroupBContrast:
    def test_output_shape(self):
        texts_raw  = ["আমি রাজি ছিলাম কিন্তু ওদের আচরণ জঘন্য।", "ভালো দিন।"]
        texts_proc = texts_raw
        feats = extract_group_B_contrast(texts_raw, texts_proc, emo_vocab_set())
        assert feats.shape == (2, 10), f"Expected (2,10), got {feats.shape}"

    def test_contrast_connective_detected(self):
        # কিন্তু = but
        texts = ["সে ভালো ছিল কিন্তু পরে পরিবর্তন হয়।"]
        feats = extract_group_B_contrast(texts, texts, emo_vocab_set())
        assert feats[0, 0] >= 1.0, "cc_count should be ≥1"
        assert feats[0, 1] == 1.0, "has_cc should be 1"

    def test_no_contrast_in_simple_sentence(self):
        texts = ["আমি ভালো আছি।"]
        feats = extract_group_B_contrast(texts, texts, emo_vocab_set())
        assert feats[0, 0] == 0.0, "cc_count should be 0"
        assert feats[0, 1] == 0.0, "has_cc should be 0"

    def test_disgust_total_count(self):
        # ঘৃণা is disgust word; should be counted
        disgust_word = list(EMOTION_LEXICON["disgust"])[0]
        texts = [f"সে {disgust_word} করে।"]
        feats = extract_group_B_contrast(texts, texts, emo_vocab_set())
        # feature 7: disgust_total
        assert feats[0, 7] >= 1.0, f"disgust_total should be ≥1 for disgust word: {disgust_word}"

    def test_dtype_float32(self):
        texts = ["test"]
        feats = extract_group_B_contrast(texts, texts, emo_vocab_set())
        assert feats.dtype == np.float32


# ── Group C Tests ─────────────────────────────────────────────────────────────

class TestGroupCIntensifierBigrams:
    def test_output_shape(self):
        texts = ["খুব ভালো লাগছে", "সে চলে গেল"]
        feats = extract_group_C_intensifier_bigrams(texts, emo_vocab_set())
        assert feats.shape == (2, 12), f"Expected (2,12), got {feats.shape}"

    def test_intensifier_emotion_pair_detected(self):
        # "খুব" is intensifier; "ভয়" is a fear word → should register a bigram
        joy_word  = list(EMOTION_LEXICON["joy"])[0]
        texts = [f"খুব {joy_word}"]
        feats = extract_group_C_intensifier_bigrams(texts, emo_vocab_set())
        # features 0-5 are counts per emotion; joy is index 3
        # At least one emotion count should be >0
        assert feats[0, :6].sum() > 0, "Should detect at least one intensifier-emotion bigram"

    def test_no_bigram_if_not_adjacent(self):
        # Intensifier far from emotion word (> 2 tokens away)
        joy_word = list(EMOTION_LEXICON["joy"])[0]
        texts = [f"খুব বড় লম্বা দূরত্বে {joy_word}"]
        feats = extract_group_C_intensifier_bigrams(texts, emo_vocab_set())
        # Joy count (index 3) and binary (index 9) should be 0
        assert feats[0, 3] == 0.0, "Should not detect bigram >2 tokens apart"

    def test_binary_presence(self):
        fear_word = list(EMOTION_LEXICON["fear"])[0]
        texts = [f"ভীষণ {fear_word}"]
        feats = extract_group_C_intensifier_bigrams(texts, emo_vocab_set())
        # Binary flags are at indices 6-11
        # fear binary is index 8 (6 + 2)
        assert feats[0, 8] == 1.0, "Fear binary should be 1 for ভীষণ + fear word"

    def test_dtype_float32(self):
        texts = ["test"]
        feats = extract_group_C_intensifier_bigrams(texts, emo_vocab_set())
        assert feats.dtype == np.float32

    def test_empty_text(self):
        texts = [""]
        feats = extract_group_C_intensifier_bigrams(texts, emo_vocab_set())
        assert feats.shape == (1, 12)
        assert feats[0].sum() == 0.0, "Empty text should produce all zeros"

    def test_multiple_texts(self):
        texts = ["আমি ভালো আছি", "ভীষণ ভয় লাগছে", ""]
        feats = extract_group_C_intensifier_bigrams(texts, emo_vocab_set())
        assert feats.shape == (3, 12)


# ── Lexicon Vocabulary Tests ──────────────────────────────────────────────────

class TestVocabularies:
    def test_bangla_wh_words_nonempty(self):
        assert len(_BANGLA_WH_WORDS) > 0, "WH word list should not be empty"

    def test_contrast_connectives_nonempty(self):
        assert len(_CONTRAST_CONNECTIVES) > 0, "Contrast connective list should not be empty"

    def test_known_wh_words_in_set(self):
        for w in ["কেন", "কী", "কোথায়", "কখন"]:
            assert w in _BANGLA_WH_WORDS, f"Expected WH word '{w}' in set"

    def test_known_contrast_in_set(self):
        for w in ["কিন্তু", "তবে", "যদিও", "তবুও", "বরং"]:
            assert w in _CONTRAST_CONNECTIVES, f"Expected contrast word '{w}' in set"
