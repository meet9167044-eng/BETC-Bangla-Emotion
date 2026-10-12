"""
src/features/feature_combination.py
=====================================
BETC — Phase 14 Feature Engineering — Feature Combination and Fusion

Provides a unified interface to combine any subset of the BETC feature groups:
  - E0: Base TF-IDF (word + char)
  - E1: + Emoji numeric features
  - E2: + Sentiment features
  - E3: + Expressive text features
  - E4: + Negation/intensifier features
  - E5: (emoji-to-text augmentation — handled in experiment scripts, not here)
  - E6–E9: Combinations of the above

This module handles:
  - Row-order-preserving hstacking of sparse matrices
  - Fit/transform split enforcement (only TF-IDF and emoji per-indicator features
    are trained; all others are stateless)
  - Feature name tracking across all groups
  - Dimension verification before stacking

USAGE EXAMPLE:
  from src.features.feature_combination import BETCCombinedExtractor

  config = {"use_tfidf": True, "use_emoji": True, "use_sentiment": False,
            "use_expressive": False, "use_negation": False}

  extractor = BETCCombinedExtractor(**config)
  X_train = extractor.fit_transform(
      train_raw_texts=df_train["text"],
      train_proc_texts=df_train["processed_text"]
  )
  X_val = extractor.transform(
      raw_texts=df_val["text"],
      proc_texts=df_val["processed_text"]
  )

LEAKAGE SAFETY:
  - All .fit() and .fit_transform() calls use TRAINING data only.
  - Emoji and TF-IDF extractors are fitted on training data; stateless extractors
    (sentiment, expressive, negation) require no fitting.
  - Validation and test splits are always transformed only.
"""

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import scipy.sparse as sp


class BETCCombinedExtractor:
    """
    Unified feature extractor that combines any subset of BETC feature groups.

    Parameters
    ----------
    use_tfidf : bool
        Include word + character TF-IDF features (the E0 baseline).
    use_emoji : bool
        Include numeric emoji features (E1).
    use_sentiment : bool
        Include Bangla sentiment lexicon features (E2).
    use_expressive : bool
        Include expressive/punctuation features (E3).
    use_negation : bool
        Include negation/intensifier/diminisher features (E4).
    emoji_min_freq : int
        Minimum training-sample frequency for per-emoji binary indicators.
    emoji_include_per_emoji : bool
        Whether to include per-emoji binary indicator columns.
    word_tfidf_params : dict, optional
        Override word TF-IDF hyperparameters.
    char_tfidf_params : dict, optional
        Override char TF-IDF hyperparameters.

    Notes
    -----
    - use_tfidf=False: drops the 25,000 TF-IDF features. Useful for testing
      whether auxiliary features alone carry signal, but generally not recommended
      for the main ablation study (always keep E0 as base).
    - Expressive and negation features use RAW text (before cleaning) vs.
      PREPROCESSED text respectively — see docstrings of individual modules.
      This class handles passing the correct text version to each extractor.
    - Raw text = df['text'] column
    - Preprocessed text = df['processed_text'] column
    """

    def __init__(
        self,
        use_tfidf: bool = True,
        use_emoji: bool = False,
        use_sentiment: bool = False,
        use_expressive: bool = False,
        use_negation: bool = False,
        emoji_min_freq: int = 10,
        emoji_include_per_emoji: bool = True,
        word_tfidf_params: Optional[dict] = None,
        char_tfidf_params: Optional[dict] = None,
    ):
        self.use_tfidf = use_tfidf
        self.use_emoji = use_emoji
        self.use_sentiment = use_sentiment
        self.use_expressive = use_expressive
        self.use_negation = use_negation
        self.emoji_min_freq = emoji_min_freq
        self.emoji_include_per_emoji = emoji_include_per_emoji
        self.word_tfidf_params = word_tfidf_params
        self.char_tfidf_params = char_tfidf_params

        self.is_fitted: bool = False
        self._extractors: dict = {}
        self._feature_dims: dict = {}

        self._init_extractors()

    def _init_extractors(self) -> None:
        """Instantiate all enabled extractors."""
        if self.use_tfidf:
            from src.features.tfidf import BETCFeatureExtractor
            kw = {}
            if self.word_tfidf_params:
                kw["word_params"] = self.word_tfidf_params
            if self.char_tfidf_params:
                kw["char_params"] = self.char_tfidf_params
            self._extractors["tfidf"] = BETCFeatureExtractor(**kw)

        if self.use_emoji:
            from src.features.emoji_features import EmojiFeatureExtractor
            self._extractors["emoji"] = EmojiFeatureExtractor(
                min_emoji_freq=self.emoji_min_freq,
                include_per_emoji=self.emoji_include_per_emoji,
            )

        if self.use_sentiment:
            from src.features.sentiment_features import SentimentFeatureExtractor
            self._extractors["sentiment"] = SentimentFeatureExtractor()

        if self.use_expressive:
            from src.features.expressive_features import ExpressiveFeatureExtractor
            self._extractors["expressive"] = ExpressiveFeatureExtractor()

        if self.use_negation:
            from src.features.negation_features import NegationFeatureExtractor
            self._extractors["negation"] = NegationFeatureExtractor()

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit_transform(
        self,
        train_raw_texts: Sequence[str],
        train_proc_texts: Sequence[str],
    ) -> sp.csr_matrix:
        """
        Fit all trainable extractors on TRAINING data and return combined features.

        Parameters
        ----------
        train_raw_texts : list[str] or pd.Series
            Raw (un-preprocessed) text from the TRAINING split. Used for emoji
            and expressive features.
        train_proc_texts : list[str] or pd.Series
            Preprocessed text from the TRAINING split (processed_text column).
            Used for TF-IDF, sentiment, and negation features.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_train, total_features)
        """
        n = len(_coerce(train_raw_texts))
        matrices = []
        self._feature_dims = {}

        if self.use_tfidf:
            X = self._extractors["tfidf"].fit_transform(train_proc_texts)
            matrices.append(X)
            self._feature_dims["tfidf"] = X.shape[1]

        if self.use_emoji:
            X = self._extractors["emoji"].fit_transform(train_raw_texts)
            matrices.append(X)
            self._feature_dims["emoji"] = X.shape[1]

        if self.use_sentiment:
            X = self._extractors["sentiment"].transform(train_proc_texts)
            matrices.append(X)
            self._feature_dims["sentiment"] = X.shape[1]

        if self.use_expressive:
            X = self._extractors["expressive"].transform(train_raw_texts)
            matrices.append(X)
            self._feature_dims["expressive"] = X.shape[1]

        if self.use_negation:
            X = self._extractors["negation"].transform(train_proc_texts)
            matrices.append(X)
            self._feature_dims["negation"] = X.shape[1]

        if not matrices:
            raise ValueError(
                "BETCCombinedExtractor: at least one feature group must be enabled."
            )

        self.is_fitted = True
        combined = sp.hstack(matrices, format="csr")
        _verify_shape(combined, n, "fit_transform")
        return combined

    # ── Transforming ──────────────────────────────────────────────────────────

    def transform(
        self,
        raw_texts: Sequence[str],
        proc_texts: Sequence[str],
    ) -> sp.csr_matrix:
        """
        Transform new data (val or test) using fitted extractors.

        Parameters
        ----------
        raw_texts : list[str] or pd.Series — raw text (for emoji/expressive).
        proc_texts : list[str] or pd.Series — preprocessed text (for TF-IDF/sentiment/negation).

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_samples, total_features)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "BETCCombinedExtractor must be fitted before calling transform(). "
                "Call fit_transform() on training data first."
            )

        n = len(_coerce(raw_texts))
        matrices = []

        if self.use_tfidf:
            matrices.append(self._extractors["tfidf"].transform(proc_texts))

        if self.use_emoji:
            matrices.append(self._extractors["emoji"].transform(raw_texts))

        if self.use_sentiment:
            matrices.append(self._extractors["sentiment"].transform(proc_texts))

        if self.use_expressive:
            matrices.append(self._extractors["expressive"].transform(raw_texts))

        if self.use_negation:
            matrices.append(self._extractors["negation"].transform(proc_texts))

        combined = sp.hstack(matrices, format="csr")
        _verify_shape(combined, n, "transform")
        return combined

    # ── Introspection ─────────────────────────────────────────────────────────

    def feature_names(self) -> List[str]:
        """Return all feature names in column order."""
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform() first.")
        names = []
        if self.use_tfidf:
            w_names, c_names = self._extractors["tfidf"].feature_names()
            names.extend([f"word_{n}" for n in w_names])
            names.extend([f"char_{n}" for n in c_names])
        if self.use_emoji:
            names.extend(self._extractors["emoji"].feature_names())
        if self.use_sentiment:
            names.extend(self._extractors["sentiment"].feature_names())
        if self.use_expressive:
            names.extend(self._extractors["expressive"].feature_names())
        if self.use_negation:
            names.extend(self._extractors["negation"].feature_names())
        return names

    @property
    def n_features_(self) -> int:
        return sum(self._feature_dims.values()) if self._feature_dims else 0

    def summary(self) -> dict:
        dims = dict(self._feature_dims)
        enabled = [k for k in ["tfidf", "emoji", "sentiment", "expressive", "negation"]
                   if getattr(self, f"use_{k}")]
        return {
            "fitted": self.is_fitted,
            "enabled_groups": enabled,
            "feature_dims": dims,
            "total_features": sum(dims.values()) if dims else 0,
        }


# ─── Convenience factory ─────────────────────────────────────────────────────

EXPERIMENT_CONFIGS = {
    "E0": dict(use_tfidf=True,  use_emoji=False, use_sentiment=False,
               use_expressive=False, use_negation=False),
    "E1": dict(use_tfidf=True,  use_emoji=True,  use_sentiment=False,
               use_expressive=False, use_negation=False),
    "E2": dict(use_tfidf=True,  use_emoji=False, use_sentiment=True,
               use_expressive=False, use_negation=False),
    "E3": dict(use_tfidf=True,  use_emoji=False, use_sentiment=False,
               use_expressive=True,  use_negation=False),
    "E4": dict(use_tfidf=True,  use_emoji=False, use_sentiment=False,
               use_expressive=False, use_negation=True),
    # E5 (emoji-to-text augmentation) is handled via text preprocessing in experiments
    "E6": dict(use_tfidf=True,  use_emoji=True,  use_sentiment=True,
               use_expressive=False, use_negation=False),
    "E7": dict(use_tfidf=True,  use_emoji=True,  use_sentiment=True,
               use_expressive=True,  use_negation=False),
    "E8": dict(use_tfidf=True,  use_emoji=True,  use_sentiment=True,
               use_expressive=False, use_negation=True),
    "E9_best": dict(use_tfidf=True,  use_emoji=True,  use_sentiment=True,
                    use_expressive=True,  use_negation=True),
}


def make_extractor(experiment_id: str, **overrides) -> "BETCCombinedExtractor":
    """
    Factory function to create a BETCCombinedExtractor for a given experiment ID.

    Parameters
    ----------
    experiment_id : str  — one of the keys in EXPERIMENT_CONFIGS, e.g. "E1"
    **overrides  : keyword arguments to override defaults

    Returns
    -------
    BETCCombinedExtractor
    """
    if experiment_id not in EXPERIMENT_CONFIGS:
        raise ValueError(
            f"Unknown experiment_id: {experiment_id!r}. "
            f"Valid IDs: {list(EXPERIMENT_CONFIGS.keys())}"
        )
    config = {**EXPERIMENT_CONFIGS[experiment_id], **overrides}
    return BETCCombinedExtractor(**config)


# ─── Internal helpers ────────────────────────────────────────────────────────

def _coerce(texts) -> list:
    try:
        import pandas as pd
        if isinstance(texts, pd.Series):
            return texts.fillna("").astype(str).tolist()
    except ImportError:
        pass
    return [str(t) if t is not None else "" for t in texts]


def _verify_shape(X: sp.csr_matrix, expected_rows: int, stage: str) -> None:
    """Raise ValueError if row count is inconsistent."""
    if X.shape[0] != expected_rows:
        raise ValueError(
            f"BETCCombinedExtractor.{stage}: expected {expected_rows} rows "
            f"but got {X.shape[0]}. Row order has been broken during hstack."
        )
