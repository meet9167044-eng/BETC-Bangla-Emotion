"""
tests/test_phase1_expressive_diagnostics.py
===========================================
Regression tests for Phase 1 Expressive Features diagnostics:
  - Verifies no NaNs or Infs
  - Verifies MaxAbsScaler preserves bounds in [0, 1]
  - Verifies row alignment
"""

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler
from src.features.expressive_features import ExpressiveFeatureExtractor, FEATURE_NAMES, extract_one

def test_expressive_no_nans_or_infs():
    sample_texts = [
        "আমি খুব খুশি! 😊",
        "সব শেষ হয়ে গেল... কেন এমন হলো????",
        "",  # Empty string edge case
        "12345!?!?!??!?!?!?!?",
        "হা হা হা হা হা হা হাআআআআআআআআআআআআআআআ",
    ]
    extractor = ExpressiveFeatureExtractor()
    feat_matrix = extractor.transform(sample_texts)
    dense = feat_matrix.toarray()

    assert not np.isnan(dense).any(), "Found NaN in expressive feature extraction"
    assert not np.isinf(dense).any(), "Found Inf in expressive feature extraction"
    assert dense.shape == (len(sample_texts), len(FEATURE_NAMES))

def test_maxabs_scaling_bounds():
    sample_texts = [
        "খুব ছোট বাক্য।",
        "একটি অনেক অনেক অনেক লম্বা বাক্য " * 20,
    ]
    extractor = ExpressiveFeatureExtractor()
    mat = extractor.transform(sample_texts)
    scaler = MaxAbsScaler()
    scaled = scaler.fit_transform(mat).toarray()

    assert (scaled >= 0.0).all() and (scaled <= 1.0).all(), "Scaled features outside [0, 1] range"

def test_row_order_preservation():
    sample_texts = [f"টেক্সট নম্বর {i}" for i in range(10)]
    extractor = ExpressiveFeatureExtractor()
    mat = extractor.transform(sample_texts)
    assert mat.shape[0] == len(sample_texts)
