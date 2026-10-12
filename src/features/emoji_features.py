"""
src/features/emoji_features.py
================================
BETC — Phase 14 Feature Engineering — Emoji Quantification

Implements numeric emoji feature extraction using the `emoji` Python library
(https://github.com/carpedm20/emoji, MIT license, version 2.16.0).

Features extracted (applied to the ORIGINAL text before cleaning):
  1. emoji_count          — total number of emoji characters in the text
  2. emoji_unique_count   — number of distinct emoji types
  3. emoji_repeated_count — emoji that appear ≥ 2 times (encourages duplication signal)
  4. emoji_density        — emoji_count / max(char_count, 1)  (proportion of emojis)
  5. has_emoji            — binary indicator (0/1)
  Per-emoji indicator columns for each emoji present in >= `min_emoji_freq`
  training examples. These are binary 0/1 features.

DESIGN NOTES:
  - This extractor operates on RAW (pre-cleaning) text so that emojis are never
    lost to preprocessing steps. Pass `df['text']` not `df['processed_text']`.
  - The per-emoji binary features are fitted only on training data.
  - Row order is always preserved; the feature matrix rows align 1:1 with the
    input list/series.
  - All features are returned as a scipy.sparse CSR matrix.
  - `emoji.emoji_list()` returns [{emoji:str, match_start:int, match_end:int}].
  - The emoji library's `EMOJI_DATA` dict is used to verify membership.
  - We do NOT map emojis to emotion categories, because:
    (a) the mapping is not reliably deterministic for Bangla social media usage,
    (b) one emoji can carry multiple or opposing emotions contextually.
  - SentiWordNet is not used here; it does not cover emoji or Bangla text.

LEAKAGE SAFETY:
  - `fit()` / `fit_transform()` must be called on TRAINING data only.
  - `transform()` applies the frozen vocab from training.
"""

import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

# Optional import — emoji must be installed separately.
try:
    import emoji as _emoji_lib
    _EMOJI_AVAILABLE = True
except ImportError:
    _EMOJI_AVAILABLE = False


def _require_emoji() -> None:
    if not _EMOJI_AVAILABLE:
        raise ImportError(
            "The 'emoji' library is required for emoji feature extraction.\n"
            "Install with: pip install emoji\n"
            "Source: https://github.com/carpedm20/emoji (MIT license)"
        )


def _extract_emojis(text: str) -> List[str]:
    """Return a list of all emoji strings found in `text` (may contain duplicates)."""
    if not text or not isinstance(text, str):
        return []
    return [entry["emoji"] for entry in _emoji_lib.emoji_list(text)]


class EmojiFeatureExtractor:
    """
    Numeric emoji feature extractor for Bangla text.

    Parameters
    ----------
    min_emoji_freq : int
        Minimum number of training samples an emoji must appear in
        before it gets its own binary indicator column.
        Default: 10  (avoids very sparse indicators for rare emojis).
    include_per_emoji : bool
        Whether to include per-emoji binary indicator columns.
        Default: True.

    Attributes
    ----------
    vocab_ : list[str]
        Sorted list of emoji tokens that survived the min_freq threshold.
    emoji_to_col_ : dict[str, int]
        Maps each vocab emoji to its column index (within the indicator block).
    n_base_features_ : int = 5
        Number of aggregate (non-per-emoji) features.
    is_fitted : bool

    Feature layout (column order):
        [0]  emoji_count
        [1]  emoji_unique_count
        [2]  emoji_repeated_count
        [3]  emoji_density
        [4]  has_emoji
        [5+] per-emoji binary indicators (if include_per_emoji=True)
    """

    N_BASE = 5  # number of aggregate features (always produced)

    BASE_FEATURE_NAMES = [
        "emoji_count",
        "emoji_unique_count",
        "emoji_repeated_count",
        "emoji_density",
        "has_emoji",
    ]

    def __init__(self, min_emoji_freq: int = 10, include_per_emoji: bool = True):
        _require_emoji()
        self.min_emoji_freq = min_emoji_freq
        self.include_per_emoji = include_per_emoji
        self.vocab_: List[str] = []
        self.emoji_to_col_: dict = {}
        self.is_fitted: bool = False

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, texts: Sequence[str]) -> "EmojiFeatureExtractor":
        """
        Fit the per-emoji vocabulary from training texts.

        Parameters
        ----------
        texts : list[str] or pd.Series
            Raw (pre-cleaning) text strings from the TRAINING split only.

        Returns self.
        """
        texts = _coerce_texts(texts)
        if self.include_per_emoji:
            freq: Counter = Counter()
            for t in texts:
                emojis_in_sample = set(_extract_emojis(t))  # count each emoji once per sample
                freq.update(emojis_in_sample)
            self.vocab_ = sorted(
                [em for em, cnt in freq.items() if cnt >= self.min_emoji_freq]
            )
            self.emoji_to_col_ = {em: i for i, em in enumerate(self.vocab_)}
        else:
            self.vocab_ = []
            self.emoji_to_col_ = {}
        self.is_fitted = True
        return self

    def fit_transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Fit on texts and return the feature matrix."""
        self.fit(texts)
        return self.transform(texts)

    # ── Transforming ─────────────────────────────────────────────────────────

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """
        Transform texts into a sparse feature matrix.

        Parameters
        ----------
        texts : list[str] or pd.Series
            Raw (pre-cleaning) text strings.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_samples, n_base + n_vocab)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "EmojiFeatureExtractor must be fitted before calling transform(). "
                "Call fit() or fit_transform() on training data first."
            )
        texts = _coerce_texts(texts)
        n = len(texts)
        n_vocab = len(self.vocab_)
        n_cols = self.N_BASE + n_vocab

        # Build dense base features, then sparse per-emoji indicators
        base = np.zeros((n, self.N_BASE), dtype=np.float32)

        # Per-emoji indicators: sparse COO components
        rows_pe, cols_pe = [], []

        for i, text in enumerate(texts):
            emojis = _extract_emojis(text)
            n_em = len(emojis)
            char_len = len(text) if text else 1

            base[i, 0] = n_em                                     # emoji_count
            base[i, 1] = len(set(emojis))                         # emoji_unique_count
            base[i, 2] = sum(1 for cnt in Counter(emojis).values() if cnt >= 2)  # repeated
            base[i, 3] = n_em / max(char_len, 1)                  # density
            base[i, 4] = 1.0 if n_em > 0 else 0.0                 # has_emoji

            if self.include_per_emoji and n_vocab > 0:
                for em in set(emojis):
                    col_idx = self.emoji_to_col_.get(em)
                    if col_idx is not None:
                        rows_pe.append(i)
                        cols_pe.append(self.N_BASE + col_idx)

        # Assemble sparse matrix
        X_base = sp.csr_matrix(base)
        if self.include_per_emoji and n_vocab > 0 and rows_pe:
            vals_pe = np.ones(len(rows_pe), dtype=np.float32)
            X_pe = sp.csr_matrix(
                (vals_pe, (rows_pe, cols_pe)), shape=(n, n_cols)
            )
            # Combine: base columns are already included in X_pe's zero-filled left block
            # Reconstruct by hstacking
            return sp.hstack(
                [sp.csr_matrix(base), sp.csr_matrix(
                    (vals_pe, ([r for r in rows_pe], [c - self.N_BASE for c in cols_pe])),
                    shape=(n, n_vocab)
                )],
                format="csr"
            )
        else:
            return X_base

    # ── Introspection ────────────────────────────────────────────────────────

    def feature_names(self) -> List[str]:
        """Return list of feature names in column order."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        names = list(self.BASE_FEATURE_NAMES)
        if self.include_per_emoji:
            names += [f"emoji_ind_{em}" for em in self.vocab_]
        return names

    @property
    def n_features_(self) -> int:
        return self.N_BASE + len(self.vocab_)

    def summary(self) -> dict:
        return {
            "fitted": self.is_fitted,
            "vocab_size": len(self.vocab_),
            "min_emoji_freq": self.min_emoji_freq,
            "include_per_emoji": self.include_per_emoji,
            "n_features": self.n_features_ if self.is_fitted else None,
            "feature_names_base": self.BASE_FEATURE_NAMES,
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """Save the fitted extractor to directory."""
        import joblib
        import os
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted extractor.")
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self, str(Path(directory) / "emoji_features.joblib"))

    @classmethod
    def load(cls, directory: str) -> "EmojiFeatureExtractor":
        """Load a fitted extractor from directory."""
        import joblib
        path = Path(directory) / "emoji_features.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Not found: {path}")
        return joblib.load(str(path))


# ─── Internal helpers ────────────────────────────────────────────────────────

def _coerce_texts(texts) -> List[str]:
    """Convert any text-like iterable to a plain Python list of strings."""
    try:
        import pandas as pd
        if isinstance(texts, pd.Series):
            return texts.fillna("").astype(str).tolist()
    except ImportError:
        pass
    return [str(t) if t is not None else "" for t in texts]
