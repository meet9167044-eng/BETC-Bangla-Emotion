"""
tests/test_phase5_additional_features.py
========================================
Regression tests for Phase 5 Auxiliary Feature Extractors:
  - 5.1: Character Elongation
  - 5.2: Hashtag Features
  - 5.3: Script Proportions & Code-mixing
  - 5.4: Normalized Text Statistics (TTR)
  - 5.5: Spelling & Normalization Divergence
  - 5.6: Verbal Inflection Markers
"""

import numpy as np
from scripts.experiments.phase5_additional_features import (
    extract_group_5_1_elongation,
    extract_group_5_2_hashtags,
    extract_group_5_3_codemixing,
    extract_group_5_4_textstats,
    extract_group_5_5_spelling_var,
    extract_group_5_6_verbal_inflection
)

def test_group_5_1_elongation():
    texts = ["হাআআআআআ খুব মজা", "সাধারণ বাক্য"]
    f = extract_group_5_1_elongation(texts)
    assert f.shape == (2, 4)
    assert f[0, 0] > 0  # detected elongation in first sample
    assert f[1, 0] == 0 # no elongation in second sample
    assert not np.isnan(f).any()

def test_group_5_2_hashtags():
    texts = ["আজকের দিন #bangla #emotion", "কোন হ্যাশট্যাগ নেই"]
    f = extract_group_5_2_hashtags(texts)
    assert f.shape == (2, 3)
    assert f[0, 0] == 2.0  # 2 hashtags detected
    assert f[1, 0] == 0.0
    assert not np.isnan(f).any()

def test_group_5_3_codemixing():
    texts = ["আমি ভালো আছি and I love it", "শুধুমাত্র বাংলা বাক্য"]
    f = extract_group_5_3_codemixing(texts)
    assert f.shape == (2, 4)
    assert f[0, 3] == 1.0  # is code-mixed
    assert f[1, 3] == 0.0  # not code-mixed
    assert not np.isnan(f).any()

def test_group_5_4_textstats():
    texts = ["এক দুই তিন চার পাঁচ", "এক এক এক এক এক"]
    f = extract_group_5_4_textstats(texts)
    assert f.shape == (2, 4)
    assert f[0, 0] == 1.0  # TTR = 1.0
    assert f[1, 0] == 0.2  # TTR = 0.2
    assert not np.isnan(f).any()

def test_group_5_5_spelling_var():
    raw = ["খাওয়া দাওয়া হয়েছে"]
    proc = ["খাওয়া দাওয়া হয়েছে"]
    f = extract_group_5_5_spelling_var(raw, proc)
    assert f.shape == (1, 3)
    assert not np.isnan(f).any()

def test_group_5_6_verbal_markers():
    texts = ["আমি গেলাম এবং দেখলাম", "আজ বৃষ্টি"]
    f = extract_group_5_6_verbal_inflection(texts)
    assert f.shape == (2, 4)
    assert f[0, 0] > 0.0  # past tense markers detected
    assert not np.isnan(f).any()
