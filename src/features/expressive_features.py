"""
src/features/expressive_features.py
=====================================
BETC — Phase 14 Feature Engineering — Expressive Text Features

Implements punctuation and expressive text feature extraction.
These features capture surface-level emotional expressiveness:
repeated punctuation, elongation, text length, character/word counts, etc.

This extractor is applied to the ORIGINAL (raw) text, before cleaning
removes or collapses punctuation signals. Alternatively, it can be applied
after preprocessing if desired — just pass the appropriate text column.

For BETC, we apply this to the raw 'text' column so that:
  - Repeated exclamations (!!!!!!) are captured before preprocessing collapses them.
  - Elongated characters (আআআআ, hahaaha) are caught before normalization.

Features extracted (no fitting required — all rule-based):
  1.  exclamation_count      — total ! characters
  2.  question_count         — total ? characters
  3.  repeated_excl_count    — runs of 3+ consecutive !
  4.  repeated_ques_count    — runs of 3+ consecutive ?
  5.  repeated_punct_count   — runs of 3+ of any punctuation (.,;:-_।?)
  6.  elongated_word_count   — words with 3+ consecutive identical characters
  7.  char_count             — total character count
  8.  word_count             — whitespace-tokenized word count
  9.  avg_word_len           — average word length (chars)
  10. uppercase_ratio        — ratio of uppercase ASCII chars to total chars (mostly 0 in Bangla)
  11. digit_count            — count of digit characters (ASCII + Bangla digits)
  12. bangla_char_ratio      — fraction of chars in Bangla Unicode range (0980–09FF)
  13. all_caps_word_count    — words that are all uppercase (ASCII; in Bangla-only text = 0)

This module is stateless and purely rule-based. No fitting is needed.
"""

import re
import unicodedata
from typing import List, Sequence

import numpy as np
import scipy.sparse as sp

# ─── Compiled regex patterns ─────────────────────────────────────────────────

_RE_REPEATED_EXCL  = re.compile(r"!{3,}")
_RE_REPEATED_QUES  = re.compile(r"\?{3,}")
_RE_REPEATED_PUNCT = re.compile(r"([?!.,;:\-_।])\1{2,}")
_RE_ELONGATED      = re.compile(r"(.)\1{2,}", re.UNICODE)

# Bangla Unicode block: U+0980 – U+09FF
_BANGLA_RANGE = range(0x0980, 0x0A00)

# Digit chars: ASCII 0-9 + Bangla digits ০-৯ (U+09E6–U+09EF)
_RE_DIGITS = re.compile(r"[\d০-৯]")


FEATURE_NAMES = [
    "expr_exclamation_count",
    "expr_question_count",
    "expr_repeated_excl_count",
    "expr_repeated_ques_count",
    "expr_repeated_punct_count",
    "expr_elongated_word_count",
    "expr_char_count",
    "expr_word_count",
    "expr_avg_word_len",
    "expr_uppercase_ratio",
    "expr_digit_count",
    "expr_bangla_char_ratio",
    "expr_all_caps_word_count",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_one(text: str) -> np.ndarray:
    """
    Extract expressive features from a single text string.

    Parameters
    ----------
    text : str  — raw (pre-cleaning) or preprocessed text.

    Returns
    -------
    np.ndarray, shape (N_FEATURES,), dtype float32
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    words = text.split()
    n_chars = len(text)
    n_words = len(words)

    # Per-feature computation
    excl_count = text.count("!")
    ques_count = text.count("?")
    rep_excl = len(_RE_REPEATED_EXCL.findall(text))
    rep_ques = len(_RE_REPEATED_QUES.findall(text))
    rep_punct = len(_RE_REPEATED_PUNCT.findall(text))

    # Elongated words: words that contain 3+ consecutive identical chars
    elong_count = sum(1 for w in words if _RE_ELONGATED.search(w))

    # Avg word length
    if n_words > 0:
        avg_word_len = sum(len(w) for w in words) / n_words
    else:
        avg_word_len = 0.0

    # Uppercase ratio (relevant for mixed-script or English emphasis)
    n_upper = sum(1 for c in text if c.isupper())
    upper_ratio = n_upper / max(n_chars, 1)

    # Digit count
    digit_count = len(_RE_DIGITS.findall(text))

    # Bangla character ratio
    n_bangla = sum(1 for c in text if ord(c) in _BANGLA_RANGE)
    bangla_ratio = n_bangla / max(n_chars, 1)

    # All-caps words (non-empty, all uppercase ASCII)
    all_caps_words = sum(
        1 for w in words
        if w and w.isupper() and any(c.isalpha() for c in w)
    )

    return np.array([
        excl_count,
        ques_count,
        rep_excl,
        rep_ques,
        rep_punct,
        elong_count,
        n_chars,
        n_words,
        avg_word_len,
        upper_ratio,
        digit_count,
        bangla_ratio,
        all_caps_words,
    ], dtype=np.float32)


class ExpressiveFeatureExtractor:
    """
    Stateless expressive-text feature extractor.

    No fitting is required. Call transform() directly on any split.

    Parameters
    ----------
    None.

    Methods
    -------
    transform(texts) -> sp.csr_matrix
    feature_names()  -> list[str]
    """

    def __init__(self):
        self.is_fitted = True  # stateless — always ready

    def fit(self, texts: Sequence[str]) -> "ExpressiveFeatureExtractor":
        """No-op. Returns self for API compatibility."""
        return self

    def fit_transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Equivalent to transform() since no fitting is needed."""
        return self.transform(texts)

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """
        Extract expressive features from all texts.

        Parameters
        ----------
        texts : list[str] or pd.Series

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_samples, N_FEATURES)
        """
        texts = _coerce_texts(texts)
        mat = np.stack([extract_one(t) for t in texts], axis=0)
        return sp.csr_matrix(mat)

    def feature_names(self) -> List[str]:
        return list(FEATURE_NAMES)

    @property
    def n_features_(self) -> int:
        return N_FEATURES

    def summary(self) -> dict:
        return {
            "fitted": True,
            "n_features": N_FEATURES,
            "feature_names": FEATURE_NAMES,
            "note": "Stateless rule-based extractor. No training data needed.",
        }


# ─── Internal helpers ────────────────────────────────────────────────────────

def _coerce_texts(texts) -> List[str]:
    try:
        import pandas as pd
        if isinstance(texts, pd.Series):
            return texts.fillna("").astype(str).tolist()
    except ImportError:
        pass
    return [str(t) if t is not None else "" for t in texts]
