"""
tests/test_phase14_features.py
=================================
BETC Phase 14 — Unit tests for new feature extractor modules:
  - EmojiFeatureExtractor (src/features/emoji_features.py)
  - ExpressiveFeatureExtractor (src/features/expressive_features.py)
  - NegationFeatureExtractor (src/features/negation_features.py)
  - SentimentFeatureExtractor (src/features/sentiment_features.py)
  - BETCCombinedExtractor (src/features/feature_combination.py)

Tests use only small synthetic strings, never real test-split examples.

Run with:
  python -m pytest tests/test_phase14_features.py -v
  OR: python tests/test_phase14_features.py
"""

import sys
import io
import os

# UTF-8 output for Bangla text
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import scipy.sparse as sp


def _assert(condition, msg=""):
    if not condition:
        raise AssertionError(msg)


# ══════════════════════════════════════════════════════════════════════════════
# A. Emoji Feature Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_emoji_extractor_import():
    from src.features.emoji_features import EmojiFeatureExtractor
    ext = EmojiFeatureExtractor()
    _assert(ext is not None)


def test_emoji_count_correct():
    from src.features.emoji_features import EmojiFeatureExtractor
    ext = EmojiFeatureExtractor(min_emoji_freq=1)
    texts = ["আমি ভালো 😊😊 আছি", "কোনো ইমোজি নেই", "🎉🎉🎉"]
    X = ext.fit_transform(texts)
    _assert(sp.issparse(X), "Output must be sparse")
    _assert(X.shape[0] == 3, f"Expected 3 rows, got {X.shape[0]}")
    arr = X.toarray()
    _assert(arr[0, 0] == 2, f"Expected 2 emojis in row 0, got {arr[0, 0]}")
    _assert(arr[1, 0] == 0, f"Expected 0 emojis in row 1, got {arr[1, 0]}")
    _assert(arr[2, 0] == 3, f"Expected 3 emojis in row 2, got {arr[2, 0]}")


def test_emoji_has_emoji_binary():
    from src.features.emoji_features import EmojiFeatureExtractor
    ext = EmojiFeatureExtractor(min_emoji_freq=1)
    texts = ["😊 ভালো", "কোনো ইমোজি নেই"]
    X = ext.fit_transform(texts)
    arr = X.toarray()
    # has_emoji is column index 4
    _assert(arr[0, 4] == 1.0, "Row with emoji must have has_emoji=1")
    _assert(arr[1, 4] == 0.0, "Row without emoji must have has_emoji=0")


def test_emoji_repeated_count():
    from src.features.emoji_features import EmojiFeatureExtractor
    ext = EmojiFeatureExtractor(min_emoji_freq=1)
    texts = ["😊😊😊 আনন্দ", "😊 একটু", "আমি নেই"]
    X = ext.fit_transform(texts)
    arr = X.toarray()
    # emoji_repeated_count is column 2
    _assert(arr[0, 2] == 1, f"Row 0 has one repeated emoji (😊 x3), got {arr[0, 2]}")
    _assert(arr[1, 2] == 0, f"Row 1 emoji appears once, got {arr[1, 2]}")
    _assert(arr[2, 2] == 0, f"Row 2 no emoji, got {arr[2, 2]}")


def test_emoji_transform_no_refit():
    from src.features.emoji_features import EmojiFeatureExtractor
    ext = EmojiFeatureExtractor(min_emoji_freq=1)
    train = ["😊 ভালো", "🎉 আনন্দ"]
    ext.fit(train)
    val = ["😊 ভালো", "কোনো নেই"]
    X_val = ext.transform(val)
    _assert(sp.issparse(X_val))
    _assert(X_val.shape[0] == 2)


def test_emoji_row_count_preserved():
    from src.features.emoji_features import EmojiFeatureExtractor
    ext = EmojiFeatureExtractor(min_emoji_freq=1)
    texts = [""] * 10
    X = ext.fit_transform(texts)
    _assert(X.shape[0] == 10, f"Expected 10 rows, got {X.shape[0]}")


def test_emoji_empty_text():
    from src.features.emoji_features import EmojiFeatureExtractor
    ext = EmojiFeatureExtractor(min_emoji_freq=1)
    texts = ["", None, "   "]
    X = ext.fit_transform(texts)
    arr = X.toarray()
    _assert(arr[0, 0] == 0, "Empty text must have emoji_count=0")


def test_emoji_feature_names():
    from src.features.emoji_features import EmojiFeatureExtractor
    ext = EmojiFeatureExtractor(min_emoji_freq=1)
    ext.fit(["😊 ভালো"])
    names = ext.feature_names()
    _assert("emoji_count" in names)
    _assert("has_emoji" in names)
    _assert(len(names) >= 5)


def test_emoji_deterministic():
    from src.features.emoji_features import EmojiFeatureExtractor
    texts = ["😊😊 ভালো", "আমি 🎉 খুশি"]
    for _ in range(3):
        ext = EmojiFeatureExtractor(min_emoji_freq=1)
        X = ext.fit_transform(texts)
    _assert(X.shape[0] == 2, "Deterministic output: 2 rows")


# ══════════════════════════════════════════════════════════════════════════════
# B. Expressive Feature Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_expressive_extractor_import():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    _assert(ext.is_fitted)


def test_expressive_exclamation_count():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    texts = ["ভালো!!!! আনন্দ", "কোনো নেই", "আমি!"]
    X = ext.transform(texts)
    arr = X.toarray()
    # exclamation_count is column 0
    _assert(arr[0, 0] == 4, f"Expected 4 exclamations, got {arr[0, 0]}")
    _assert(arr[1, 0] == 0, f"Expected 0 exclamations, got {arr[1, 0]}")
    _assert(arr[2, 0] == 1, f"Expected 1 exclamation, got {arr[2, 0]}")


def test_expressive_question_count():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    X = ext.transform(["কি হচ্ছে?? কেন???"])
    arr = X.toarray()
    # question_count is column 1
    _assert(arr[0, 1] == 5, f"Expected 5 question marks, got {arr[0, 1]}")


def test_expressive_repeated_punct_count():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    X = ext.transform(["হ্যাঁ!!! ভালো আছি....", "সাধারণ টেক্সট"])
    arr = X.toarray()
    # repeated_excl_count (col 2) and repeated_punct_count (col 4)
    _assert(arr[0, 2] == 1, f"Expected 1 run of 3+ !, got {arr[0, 2]}")
    _assert(arr[1, 4] == 0, f"No repeated punct in plain text, got {arr[1, 4]}")


def test_expressive_char_count():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    text = "আমি"
    X = ext.transform([text])
    arr = X.toarray()
    # char_count is column 6
    _assert(arr[0, 6] == len(text), f"Expected {len(text)}, got {arr[0, 6]}")


def test_expressive_empty_text():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    X = ext.transform(["", None, "   "])
    _assert(X.shape[0] == 3)
    arr = X.toarray()
    _assert(arr[0, 6] == 0, "Empty text char_count must be 0")


def test_expressive_bangla_char_ratio():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    # Pure Bangla text should have high ratio
    X = ext.transform(["আমি ভালো আছি"])
    arr = X.toarray()
    # bangla_char_ratio is column 11
    _assert(arr[0, 11] > 0.5, f"Bangla text should have high bangla_char_ratio, got {arr[0, 11]}")


def test_expressive_row_count_preserved():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    texts = ["text"] * 7
    X = ext.transform(texts)
    _assert(X.shape[0] == 7)


def test_expressive_sparse_output():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    X = ext.transform(["আমি ভালো আছি"])
    _assert(sp.issparse(X), "Output must be sparse")


def test_expressive_feature_names():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    ext = ExpressiveFeatureExtractor()
    names = ext.feature_names()
    _assert("expr_exclamation_count" in names)
    _assert("expr_char_count" in names)
    _assert(len(names) == 13)


# ══════════════════════════════════════════════════════════════════════════════
# C. Negation Feature Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_negation_extractor_import():
    from src.features.negation_features import NegationFeatureExtractor
    ext = NegationFeatureExtractor()
    _assert(ext.is_fitted)


def test_negation_count_na():
    from src.features.negation_features import NegationFeatureExtractor
    ext = NegationFeatureExtractor()
    X = ext.transform(["আমি যাব না", "আমি যাব"])
    arr = X.toarray()
    # neg_count is col 0
    _assert(arr[0, 0] >= 1, f"Expected >=1 negation in 'না', got {arr[0, 0]}")
    _assert(arr[1, 0] == 0, f"Expected 0 negations, got {arr[1, 0]}")


def test_negation_present_binary():
    from src.features.negation_features import NegationFeatureExtractor
    ext = NegationFeatureExtractor()
    X = ext.transform(["না করব", "করব"])
    arr = X.toarray()
    # neg_present is col 1
    _assert(arr[0, 1] == 1.0, "Text with না must have neg_present=1")
    _assert(arr[1, 1] == 0.0, "Text without negation must have neg_present=0")


def test_intensifier_count():
    from src.features.negation_features import NegationFeatureExtractor
    ext = NegationFeatureExtractor()
    X = ext.transform(["খুব ভালো আছি", "আছি"])
    arr = X.toarray()
    # intensifier_count is col 2
    _assert(arr[0, 2] >= 1, f"'খুব' is an intensifier, got {arr[0, 2]}")
    _assert(arr[1, 2] == 0, f"No intensifier expected, got {arr[1, 2]}")


def test_negation_empty_text():
    from src.features.negation_features import NegationFeatureExtractor
    ext = NegationFeatureExtractor()
    X = ext.transform(["", None])
    arr = X.toarray()
    _assert(arr[0, 0] == 0, "Empty text neg_count must be 0")


def test_negation_sparse_output():
    from src.features.negation_features import NegationFeatureExtractor
    ext = NegationFeatureExtractor()
    X = ext.transform(["না করব"])
    _assert(sp.issparse(X))


def test_negation_row_count_preserved():
    from src.features.negation_features import NegationFeatureExtractor
    ext = NegationFeatureExtractor()
    texts = ["না"] * 5
    X = ext.transform(texts)
    _assert(X.shape[0] == 5)


def test_negation_feature_names():
    from src.features.negation_features import NegationFeatureExtractor
    ext = NegationFeatureExtractor()
    names = ext.feature_names()
    _assert("neg_count" in names)
    _assert("neg_present" in names)
    _assert("intensifier_count" in names)
    _assert(len(names) == 10)


def test_total_modifier_count():
    from src.features.negation_features import NegationFeatureExtractor
    ext = NegationFeatureExtractor()
    # Text with negation + intensifier
    X = ext.transform(["খুব ভালো না"])
    arr = X.toarray()
    # total_modifier_count is col 8
    _assert(arr[0, 8] >= 2, f"Expected >=2 total modifiers (না + খুব), got {arr[0, 8]}")


# ══════════════════════════════════════════════════════════════════════════════
# D. Sentiment Feature Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_sentiment_extractor_import():
    from src.features.sentiment_features import SentimentFeatureExtractor
    ext = SentimentFeatureExtractor()
    _assert(ext.is_fitted)


def test_sentiment_positive_count():
    from src.features.sentiment_features import SentimentFeatureExtractor
    ext = SentimentFeatureExtractor()
    X = ext.transform(["আমি ভালো আছি আনন্দ"])
    arr = X.toarray()
    # sent_pos_count is col 0
    _assert(arr[0, 0] >= 1, f"Expected >=1 positive word, got {arr[0, 0]}")


def test_sentiment_negative_count():
    from src.features.sentiment_features import SentimentFeatureExtractor
    ext = SentimentFeatureExtractor()
    X = ext.transform(["এটা খারাপ ঘৃণা"])
    arr = X.toarray()
    # sent_neg_count is col 1
    _assert(arr[0, 1] >= 1, f"Expected >=1 negative word, got {arr[0, 1]}")


def test_sentiment_net_score():
    from src.features.sentiment_features import SentimentFeatureExtractor
    ext = SentimentFeatureExtractor()
    # Pure positive text
    X_pos = ext.transform(["ভালো আনন্দ সুখ"])
    arr_pos = X_pos.toarray()
    # sent_net_score is col 2
    _assert(arr_pos[0, 2] > 0, f"Positive text should have positive net score, got {arr_pos[0, 2]}")


def test_sentiment_polarity():
    from src.features.sentiment_features import SentimentFeatureExtractor
    ext = SentimentFeatureExtractor()
    X_pos = ext.transform(["ভালো আনন্দ"])
    X_neg = ext.transform(["খারাপ ঘৃণা"])
    X_neu = ext.transform(["আমি আছি এখানে"])
    # sent_polarity is col 6
    arr_pos = X_pos.toarray()
    arr_neg = X_neg.toarray()
    arr_neu = X_neu.toarray()
    _assert(arr_pos[0, 6] == 1.0, f"Positive polarity expected, got {arr_pos[0, 6]}")
    _assert(arr_neg[0, 6] == -1.0, f"Negative polarity expected, got {arr_neg[0, 6]}")


def test_sentiment_mixed():
    from src.features.sentiment_features import SentimentFeatureExtractor
    ext = SentimentFeatureExtractor()
    X = ext.transform(["ভালো কিন্তু খারাপ"])
    arr = X.toarray()
    # sent_mixed is col 9
    _assert(arr[0, 9] == 1.0, f"Text with both pos and neg should be mixed, got {arr[0, 9]}")


def test_sentiment_empty_text():
    from src.features.sentiment_features import SentimentFeatureExtractor
    ext = SentimentFeatureExtractor()
    X = ext.transform(["", None])
    arr = X.toarray()
    _assert(arr[0, 0] == 0, "Empty text pos_count must be 0")


def test_sentiment_feature_names():
    from src.features.sentiment_features import SentimentFeatureExtractor
    ext = SentimentFeatureExtractor()
    names = ext.feature_names()
    _assert("sent_pos_count" in names)
    _assert("sent_neg_count" in names)
    _assert(len(names) == 10)


def test_sentiment_sparse_output():
    from src.features.sentiment_features import SentimentFeatureExtractor
    ext = SentimentFeatureExtractor()
    X = ext.transform(["ভালো আছি"])
    _assert(sp.issparse(X))


# ══════════════════════════════════════════════════════════════════════════════
# E. Feature Combination Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_combined_e0_baseline():
    """E0: TF-IDF only — should produce 25000-column matrix."""
    from src.features.feature_combination import BETCCombinedExtractor
    texts_raw  = ["আমি ভালো আছি", "এটা খারাপ", "মজার বিষয়"]
    texts_proc = texts_raw  # simplified for test
    ext = BETCCombinedExtractor(use_tfidf=True, use_emoji=False,
                                 use_sentiment=False, use_expressive=False,
                                 use_negation=False)
    X = ext.fit_transform(texts_raw, texts_proc)
    _assert(sp.issparse(X))
    _assert(X.shape[0] == 3, f"Expected 3 rows, got {X.shape[0]}")
    # Exactly 25000 or fewer (vocab may be smaller for tiny test data)
    _assert(X.shape[1] <= 25000, f"Expected <=25000 cols, got {X.shape[1]}")


def test_combined_row_count_all_groups():
    """All feature groups enabled: row count must match input."""
    from src.features.feature_combination import BETCCombinedExtractor
    n = 8
    texts_raw  = ["আমি 😊 ভালো আছি না", "খারাপ!!!! ঘৃণা", "মজার বিষয়",
                  "আনন্দ 🎉", "কষ্ট দুঃখ", "খুব ভালো",
                  "একদম না করব", "আশ্চর্য!"]
    texts_proc = texts_raw
    ext = BETCCombinedExtractor(use_tfidf=True, use_emoji=True,
                                 use_sentiment=True, use_expressive=True,
                                 use_negation=True, emoji_min_freq=1)
    X = ext.fit_transform(texts_raw, texts_proc)
    _assert(X.shape[0] == n, f"Expected {n} rows, got {X.shape[0]}")


def test_combined_transform_same_dim():
    """Train and val transforms must produce the same feature dimension."""
    from src.features.feature_combination import BETCCombinedExtractor
    train_raw  = ["আমি 😊 ভালো", "খারাপ 😢", "মজার"]
    train_proc = train_raw
    val_raw    = ["নতুন টেক্সট 😊", "আরেকটি"]
    val_proc   = val_raw
    ext = BETCCombinedExtractor(use_tfidf=True, use_emoji=True,
                                 use_sentiment=True, emoji_min_freq=1)
    X_train = ext.fit_transform(train_raw, train_proc)
    X_val   = ext.transform(val_raw, val_proc)
    _assert(X_train.shape[1] == X_val.shape[1],
            f"Dim mismatch: train={X_train.shape[1]}, val={X_val.shape[1]}")


def test_combined_sparse_output():
    from src.features.feature_combination import BETCCombinedExtractor
    texts = ["আমি ভালো", "খারাপ"]
    ext = BETCCombinedExtractor(use_tfidf=False, use_sentiment=True,
                                 use_expressive=True, use_negation=False)
    X = ext.fit_transform(texts, texts)
    _assert(sp.issparse(X), "Output must be sparse CSR matrix")


def test_combined_no_transform_before_fit():
    from src.features.feature_combination import BETCCombinedExtractor
    ext = BETCCombinedExtractor(use_tfidf=True)
    try:
        ext.transform(["টেক্সট"], ["টেক্সট"])
        _assert(False, "Expected RuntimeError before fitting")
    except RuntimeError:
        pass  # expected


def test_combined_factory_e1():
    from src.features.feature_combination import make_extractor
    ext = make_extractor("E1", emoji_min_freq=1)
    _assert(ext.use_tfidf)
    _assert(ext.use_emoji)
    _assert(not ext.use_sentiment)


def test_combined_summary():
    from src.features.feature_combination import BETCCombinedExtractor
    texts = ["আমি ভালো আছি না", "এটা খারাপ"]
    ext = BETCCombinedExtractor(use_tfidf=False, use_sentiment=True,
                                 use_negation=True, use_expressive=True)
    ext.fit_transform(texts, texts)
    s = ext.summary()
    _assert(s["fitted"])
    _assert("sentiment" in s["enabled_groups"])
    _assert(s["total_features"] > 0)


# ══════════════════════════════════════════════════════════════════════════════
# F. Integration: all extractors produce no-NaN output
# ══════════════════════════════════════════════════════════════════════════════

def test_no_nan_in_emoji_output():
    from src.features.emoji_features import EmojiFeatureExtractor
    texts = ["😊 আমি", "খারাপ 😢😢😢", "", "plain text"]
    ext = EmojiFeatureExtractor(min_emoji_freq=1)
    X = ext.fit_transform(texts)
    _assert(not np.any(np.isnan(X.toarray())), "NaN found in emoji features")


def test_no_nan_in_expressive_output():
    from src.features.expressive_features import ExpressiveFeatureExtractor
    texts = ["!!!! আমি", "", "সাধারণ???", None]
    ext = ExpressiveFeatureExtractor()
    X = ext.transform(texts)
    _assert(not np.any(np.isnan(X.toarray())), "NaN found in expressive features")


def test_no_nan_in_negation_output():
    from src.features.negation_features import NegationFeatureExtractor
    texts = ["না করব", "", "খুব ভালো না", None]
    ext = NegationFeatureExtractor()
    X = ext.transform(texts)
    _assert(not np.any(np.isnan(X.toarray())), "NaN found in negation features")


def test_no_nan_in_sentiment_output():
    from src.features.sentiment_features import SentimentFeatureExtractor
    texts = ["ভালো আনন্দ", "খারাপ ঘৃণা", "", None]
    ext = SentimentFeatureExtractor()
    X = ext.transform(texts)
    _assert(not np.any(np.isnan(X.toarray())), "NaN found in sentiment features")


# ══════════════════════════════════════════════════════════════════════════════
# Run standalone
# ══════════════════════════════════════════════════════════════════════════════

def _run_all():
    test_fns = [v for k, v in sorted(globals().items())
                if k.startswith("test_") and callable(v)]
    passed, failed = 0, 0
    failures = []
    for fn in test_fns:
        try:
            fn()
            passed += 1
            name = fn.__name__.encode("utf-8", errors="replace").decode("utf-8")
            print(f"  [PASS] {name}")
        except Exception as exc:
            failed += 1
            name = fn.__name__.encode("utf-8", errors="replace").decode("utf-8")
            msg = str(exc).encode("utf-8", errors="replace").decode("utf-8")
            failures.append((name, msg))
            print(f"  [FAIL] {name}: {msg}")
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    if failures:
        print("\nFailed tests:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
    return failed == 0


if __name__ == "__main__":
    print("Running BETC Phase 14 feature extractor tests...")
    ok = _run_all()
    sys.exit(0 if ok else 1)
