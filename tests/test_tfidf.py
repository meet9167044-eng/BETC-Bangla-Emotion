import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
tests/test_tfidf.py
====================
BETC Phase 7 — Unit tests for src/features/tfidf.py

Tests use small synthetic corpora. No real split data is used.
Run with:  python tests/test_tfidf.py
       OR: python -m pytest tests/test_tfidf.py -v
"""

import os, tempfile
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features.tfidf import BETCFeatureExtractor, WORD_TFIDF_PARAMS, CHAR_TFIDF_PARAMS


# ─── Synthetic corpora ────────────────────────────────────────────────────────
TRAIN_TEXTS = [
    "আমি খুব রাগান্বিত হয়েছি",
    "সে অনেক ভয় পেয়েছে",
    "আমার মন খুব খারাপ",
    "অনেক আনন্দ হচ্ছে আজকে",
    "ভীষণ ঘৃণা লাগছে এই কথায়",
    "অবাক হয়ে গেলাম একদম",
    "কষ্ট পাচ্ছি না কিন্তু মন ভালো নেই",
    "হাসি পাচ্ছে এই কান্ডে",
]
VAL_TEXTS = [
    "খুব রাগ লাগছে",
    "ভয় না পাওয়াই ভালো",
]
TEST_TEXTS = [
    "আনন্দের কোনো সীমা নেই",
    "অনেক কষ্ট হচ্ছে",
]


def _assert(condition, msg=""):
    if not condition:
        raise AssertionError(msg)


# ─── 1. Basic fit/transform shape ─────────────────────────────────────────────

def test_fit_transform_returns_sparse():
    fx = BETCFeatureExtractor()
    X = fx.fit_transform(TRAIN_TEXTS)
    _assert(sp.issparse(X), "fit_transform must return a sparse matrix")

def test_fit_transform_row_count():
    fx = BETCFeatureExtractor()
    X = fx.fit_transform(TRAIN_TEXTS)
    _assert(X.shape[0] == len(TRAIN_TEXTS),
            f"Expected {len(TRAIN_TEXTS)} rows, got {X.shape[0]}")

def test_combined_dim_at_most_25000():
    """Combined dim <= 25,000 (initial HP cap; may be less if vocab is small)."""
    fx = BETCFeatureExtractor()
    X = fx.fit_transform(TRAIN_TEXTS)
    _assert(X.shape[1] <= 25_000,
            f"Combined dim {X.shape[1]} exceeds 25,000")

def test_transform_val_same_dim():
    fx = BETCFeatureExtractor()
    X_train = fx.fit_transform(TRAIN_TEXTS)
    X_val = fx.transform(VAL_TEXTS)
    _assert(X_val.shape[1] == X_train.shape[1],
            "Val feature dim must match train feature dim")

def test_transform_test_same_dim():
    fx = BETCFeatureExtractor()
    X_train = fx.fit_transform(TRAIN_TEXTS)
    X_test = fx.transform(TEST_TEXTS)
    _assert(X_test.shape[1] == X_train.shape[1],
            "Test feature dim must match train feature dim")

def test_transform_val_row_count():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    X_val = fx.transform(VAL_TEXTS)
    _assert(X_val.shape[0] == len(VAL_TEXTS))

def test_transform_test_row_count():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    X_test = fx.transform(TEST_TEXTS)
    _assert(X_test.shape[0] == len(TEST_TEXTS))


# ─── 2. Leakage prevention (CRITICAL) ────────────────────────────────────────

def test_transform_raises_before_fit():
    """Calling transform() before fit_transform() must raise RuntimeError."""
    fx = BETCFeatureExtractor()
    try:
        fx.transform(VAL_TEXTS)
        _assert(False, "transform() must raise RuntimeError before fitting")
    except RuntimeError:
        pass  # expected

def test_word_tfidf_not_refit_on_val():
    """
    Verify that the word vectorizer vocabulary does not change after transform().
    This is a proxy test for leakage: if vocab changes, fitting happened.
    """
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    vocab_before = frozenset(fx.word_tfidf.vocabulary_.keys())
    fx.transform(VAL_TEXTS)
    vocab_after = frozenset(fx.word_tfidf.vocabulary_.keys())
    _assert(vocab_before == vocab_after,
            "Word vocab must not change after transform() — refitting on val detected!")

def test_char_tfidf_not_refit_on_val():
    """Same leakage check for the character vectorizer."""
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    vocab_before = frozenset(fx.char_tfidf.vocabulary_.keys())
    fx.transform(VAL_TEXTS)
    vocab_after = frozenset(fx.char_tfidf.vocabulary_.keys())
    _assert(vocab_before == vocab_after,
            "Char vocab must not change after transform() — refitting on val detected!")

def test_word_tfidf_not_refit_on_test():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    vocab_before = frozenset(fx.word_tfidf.vocabulary_.keys())
    fx.transform(TEST_TEXTS)
    vocab_after = frozenset(fx.word_tfidf.vocabulary_.keys())
    _assert(vocab_before == vocab_after,
            "Word vocab must not change after transform() on test!")

def test_char_tfidf_not_refit_on_test():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    vocab_before = frozenset(fx.char_tfidf.vocabulary_.keys())
    fx.transform(TEST_TEXTS)
    vocab_after = frozenset(fx.char_tfidf.vocabulary_.keys())
    _assert(vocab_before == vocab_after,
            "Char vocab must not change after transform() on test!")

def test_is_fitted_flag_false_before_fit():
    fx = BETCFeatureExtractor()
    _assert(not fx.is_fitted, "is_fitted must be False before fit_transform()")

def test_is_fitted_flag_true_after_fit():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    _assert(fx.is_fitted, "is_fitted must be True after fit_transform()")

def test_fit_transform_output_is_csr():
    fx = BETCFeatureExtractor()
    X = fx.fit_transform(TRAIN_TEXTS)
    _assert(isinstance(X, sp.csr_matrix), f"Expected csr_matrix, got {type(X)}")

def test_transform_output_is_csr():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    X = fx.transform(VAL_TEXTS)
    _assert(isinstance(X, sp.csr_matrix), f"Expected csr_matrix, got {type(X)}")


# ─── 3. Hyperparameter conformance ───────────────────────────────────────────

def test_word_ngram_range():
    """Word TF-IDF must use ngram_range=(1,2) per PIPELINE_SPEC.md §3.2."""
    fx = BETCFeatureExtractor()
    _assert(fx.word_tfidf.ngram_range == (1, 2),
            f"Expected word ngram_range=(1,2), got {fx.word_tfidf.ngram_range}")

def test_char_ngram_range():
    """Char TF-IDF must use ngram_range=(3,5) per PIPELINE_SPEC.md §3.3."""
    fx = BETCFeatureExtractor()
    _assert(fx.char_tfidf.ngram_range == (3, 5),
            f"Expected char ngram_range=(3,5), got {fx.char_tfidf.ngram_range}")

def test_word_max_features():
    """Word TF-IDF max_features=10,000 per PIPELINE_SPEC.md §3.2."""
    fx = BETCFeatureExtractor()
    _assert(fx.word_tfidf.max_features == 10_000,
            f"Expected word max_features=10000, got {fx.word_tfidf.max_features}")

def test_char_max_features():
    """Char TF-IDF max_features=15,000 per PIPELINE_SPEC.md §3.3."""
    fx = BETCFeatureExtractor()
    _assert(fx.char_tfidf.max_features == 15_000,
            f"Expected char max_features=15000, got {fx.char_tfidf.max_features}")

def test_char_analyzer():
    """Char TF-IDF analyzer='char_wb' per PIPELINE_SPEC.md §3.3."""
    fx = BETCFeatureExtractor()
    _assert(fx.char_tfidf.analyzer == "char_wb",
            f"Expected char analyzer='char_wb', got {fx.char_tfidf.analyzer!r}")

def test_word_strip_accents_none():
    """Word TF-IDF must NOT strip accents (Bangla diacritics must be kept)."""
    fx = BETCFeatureExtractor()
    _assert(fx.word_tfidf.strip_accents is None,
            "strip_accents must be None for Bangla text")

def test_char_strip_accents_none():
    fx = BETCFeatureExtractor()
    _assert(fx.char_tfidf.strip_accents is None,
            "strip_accents must be None for char TF-IDF on Bangla text")


# ─── 4. Feature fusion (hstack) ───────────────────────────────────────────────

def test_fusion_dim_is_word_plus_char():
    """Combined dim must equal word_vocab_size + char_vocab_size."""
    fx = BETCFeatureExtractor()
    X = fx.fit_transform(TRAIN_TEXTS)
    n_word = len(fx.word_tfidf.get_feature_names_out())
    n_char = len(fx.char_tfidf.get_feature_names_out())
    _assert(X.shape[1] == n_word + n_char,
            f"Combined dim {X.shape[1]} != {n_word}+{n_char}={n_word+n_char}")

def test_fusion_not_dense():
    """Feature matrix must stay sparse — no accidental densification."""
    fx = BETCFeatureExtractor()
    X = fx.fit_transform(TRAIN_TEXTS)
    _assert(sp.issparse(X), "Fused feature matrix must remain sparse")

def test_summary_dict_keys():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    s = fx.summary()
    for key in ["fitted", "word_tfidf", "char_tfidf", "combined_dim"]:
        _assert(key in s, f"summary() missing key: {key}")

def test_summary_combined_dim_matches_matrix():
    fx = BETCFeatureExtractor()
    X = fx.fit_transform(TRAIN_TEXTS)
    s = fx.summary()
    _assert(s["combined_dim"] == X.shape[1],
            f"summary combined_dim {s['combined_dim']} != matrix dim {X.shape[1]}")


# ─── 5. Persistence (save / load) ────────────────────────────────────────────

def test_save_creates_files():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    with tempfile.TemporaryDirectory() as tmpdir:
        fx.save(tmpdir)
        _assert(os.path.exists(os.path.join(tmpdir, "word_tfidf.joblib")))
        _assert(os.path.exists(os.path.join(tmpdir, "char_tfidf.joblib")))

def test_save_before_fit_raises():
    fx = BETCFeatureExtractor()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            fx.save(tmpdir)
            _assert(False, "save() must raise RuntimeError before fitting")
        except RuntimeError:
            pass

def test_load_produces_fitted_extractor():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    with tempfile.TemporaryDirectory() as tmpdir:
        fx.save(tmpdir)
        fx2 = BETCFeatureExtractor.load(tmpdir)
        _assert(fx2.is_fitted, "Loaded extractor must be marked as fitted")

def test_loaded_transform_matches_original():
    """After save/load, transform() must produce identical matrices."""
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    X_val_orig = fx.transform(VAL_TEXTS).toarray()
    with tempfile.TemporaryDirectory() as tmpdir:
        fx.save(tmpdir)
        fx2 = BETCFeatureExtractor.load(tmpdir)
        X_val_loaded = fx2.transform(VAL_TEXTS).toarray()
    _assert(np.allclose(X_val_orig, X_val_loaded),
            "Loaded extractor must produce identical transform output")

def test_load_missing_file_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            BETCFeatureExtractor.load(tmpdir)
            _assert(False, "load() must raise FileNotFoundError for missing files")
        except FileNotFoundError:
            pass


# ─── 6. Empty / edge-case texts ──────────────────────────────────────────────

def test_empty_string_in_train_does_not_crash():
    texts = TRAIN_TEXTS + [""]
    fx = BETCFeatureExtractor()
    X = fx.fit_transform(texts)
    _assert(X.shape[0] == len(texts))

def test_empty_string_in_transform_does_not_crash():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    X = fx.transform([""])
    _assert(X.shape[0] == 1)

def test_deterministic_fit_transform():
    """Two BETCFeatureExtractor instances fit on the same data must produce
    identical feature matrices (TF-IDF is deterministic for same data)."""
    fx1 = BETCFeatureExtractor()
    fx2 = BETCFeatureExtractor()
    X1 = fx1.fit_transform(TRAIN_TEXTS).toarray()
    X2 = fx2.fit_transform(TRAIN_TEXTS).toarray()
    _assert(np.allclose(X1, X2), "fit_transform must be deterministic")

def test_feature_names_returns_two_arrays():
    fx = BETCFeatureExtractor()
    fx.fit_transform(TRAIN_TEXTS)
    word_names, char_names = fx.feature_names()
    _assert(len(word_names) > 0, "word feature names must be non-empty")
    _assert(len(char_names) > 0, "char feature names must be non-empty")


# ─── Runner ──────────────────────────────────────────────────────────────────

def _run_all():
    test_fns = [v for k, v in sorted(globals().items())
                if k.startswith("test_") and callable(v)]
    passed = failed = 0
    failures = []
    for fn in test_fns:
        try:
            fn()
            passed += 1
            print("  [PASS] " + fn.__name__)
        except Exception as exc:
            failed += 1
            msg = str(exc).encode('utf-8', errors='replace').decode('utf-8')
            failures.append((fn.__name__, msg))
            print("  [FAIL] " + fn.__name__ + ": " + msg)
    print("\n" + "="*60)
    print("Results: {} passed, {} failed out of {} tests".format(
        passed, failed, passed + failed))
    if failures:
        print("\nFailed tests:")
        for name, msg in failures:
            print("  - " + name + ": " + msg)
    return failed == 0


if __name__ == "__main__":
    print("Running BETC Phase 7 TF-IDF unit tests...")
    ok = _run_all()
    sys.exit(0 if ok else 1)
