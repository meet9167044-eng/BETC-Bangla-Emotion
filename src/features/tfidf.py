"""
src/features/tfidf.py
======================
BETC — Bangla Emotion TF-IDF Classifier Chain
Phase 7 — TF-IDF Feature Extraction and Fusion

Implements word-level TF-IDF, character-level TF-IDF, and sparse feature
fusion as specified in PIPELINE_SPEC.md Sections 3.2–3.4.

Hyperparameters (all [INITIAL-HP] — see PIPELINE_SPEC.md Section 6):
  Word TF-IDF:   TfidfVectorizer(ngram_range=(1,2), max_features=10_000)
  Char TF-IDF:   TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5),
                                  max_features=15_000)
  Fusion:        scipy.sparse.hstack → shape (n_samples, 25_000)

Leakage-prevention rules (EVALUATION_PROTOCOL.md Section 3, Rule 1):
  - Both vectorizers are fitted ONLY on the training split.
  - Validation and test splits are transformed with .transform() only.
  - This module never calls .fit() or .fit_transform() on val/test data.

Usage example
-------------
from src.features.tfidf import BETCFeatureExtractor

# Fit (training data only)
fx = BETCFeatureExtractor()
X_train = fx.fit_transform(train_texts)

# Transform (val and test — no fitting)
X_val  = fx.transform(val_texts)
X_test = fx.transform(test_texts)

# Persist
fx.save('artifacts/vectorizers/')

# Reload
fx2 = BETCFeatureExtractor.load('artifacts/vectorizers/')
X_val_check = fx2.transform(val_texts)
"""

import os
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

# ─── Default hyperparameters (INITIAL-HP from PIPELINE_SPEC.md §6) ─────────
WORD_TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    max_features=10_000,
    sublinear_tf=True,    # log(1+tf) — standard practice; not excluded by spec
    strip_accents=None,   # do NOT strip accents — Bangla diacritics are meaningful
    analyzer="word",
    token_pattern=r"(?u)\S+",  # keep all non-whitespace as tokens (incl. Bangla)
)

CHAR_TFIDF_PARAMS = dict(
    analyzer="char_wb",
    ngram_range=(3, 5),
    max_features=15_000,
    sublinear_tf=True,
    strip_accents=None,
)

WORD_VECTORIZER_FILE = "word_tfidf.joblib"
CHAR_VECTORIZER_FILE = "char_tfidf.joblib"


class BETCFeatureExtractor:
    """
    Encapsulates the BETC word + character TF-IDF pipeline.

    State
    -----
    word_tfidf : TfidfVectorizer  (fitted on train; None before fitting)
    char_tfidf : TfidfVectorizer  (fitted on train; None before fitting)
    is_fitted  : bool

    Methods
    -------
    fit_transform(texts)   — fit both vectorizers on texts, return fused matrix
    transform(texts)       — transform texts without refitting (leakage-safe)
    save(directory)        — persist both fitted vectorizers with joblib
    load(directory)        — class method; reload from directory
    feature_names()        — (word_names, char_names) for introspection
    """

    def __init__(
        self,
        word_params: dict = None,
        char_params: dict = None,
    ):
        _word = {**WORD_TFIDF_PARAMS, **(word_params or {})}
        _char = {**CHAR_TFIDF_PARAMS, **(char_params or {})}
        self.word_tfidf = TfidfVectorizer(**_word)
        self.char_tfidf = TfidfVectorizer(**_char)
        self.is_fitted = False
        # Store params for logging
        self._word_params = _word
        self._char_params = _char

    # ── Fitting ────────────────────────────────────────────────────────────

    def fit_transform(self, texts) -> sp.csr_matrix:
        """
        Fit both vectorizers on `texts` and return the fused feature matrix.

        Parameters
        ----------
        texts : list[str] | pd.Series
            Training-split processed texts. Must NEVER include validation or
            test data (EVALUATION_PROTOCOL.md Section 3, Rule 1).

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_train, 25_000)
        """
        texts = _coerce_texts(texts)
        X_word = self.word_tfidf.fit_transform(texts)
        X_char = self.char_tfidf.fit_transform(texts)
        self.is_fitted = True
        return sp.hstack([X_word, X_char], format="csr")

    # ── Transforming (no fitting) ──────────────────────────────────────────

    def transform(self, texts) -> sp.csr_matrix:
        """
        Transform `texts` using the already-fitted vectorizers.

        Raises RuntimeError if called before fit_transform().

        Parameters
        ----------
        texts : list[str] | pd.Series
            Validation or test processed texts.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (n_samples, 25_000)
        """
        if not self.is_fitted:
            raise RuntimeError(
                "BETCFeatureExtractor must be fitted before calling transform(). "
                "Call fit_transform() on training data first."
            )
        texts = _coerce_texts(texts)
        X_word = self.word_tfidf.transform(texts)
        X_char = self.char_tfidf.transform(texts)
        return sp.hstack([X_word, X_char], format="csr")

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """
        Persist both fitted vectorizers to `directory` as joblib files.

        Saved files:
          {directory}/word_tfidf.joblib
          {directory}/char_tfidf.joblib
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted BETCFeatureExtractor.")
        os.makedirs(directory, exist_ok=True)
        word_path = os.path.join(directory, WORD_VECTORIZER_FILE)
        char_path = os.path.join(directory, CHAR_VECTORIZER_FILE)
        joblib.dump(self.word_tfidf, word_path)
        joblib.dump(self.char_tfidf, char_path)

    @classmethod
    def load(cls, directory: str) -> "BETCFeatureExtractor":
        """
        Reload a previously saved BETCFeatureExtractor from `directory`.

        Parameters
        ----------
        directory : str
            Path that contains word_tfidf.joblib and char_tfidf.joblib.

        Returns
        -------
        BETCFeatureExtractor with is_fitted=True
        """
        word_path = os.path.join(directory, WORD_VECTORIZER_FILE)
        char_path = os.path.join(directory, CHAR_VECTORIZER_FILE)
        for p in [word_path, char_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Vectorizer file not found: {p}\n"
                    f"Run Phase 7 feature extraction first."
                )
        instance = cls.__new__(cls)
        instance.word_tfidf = joblib.load(word_path)
        instance.char_tfidf = joblib.load(char_path)
        instance.is_fitted = True
        instance._word_params = WORD_TFIDF_PARAMS
        instance._char_params = CHAR_TFIDF_PARAMS
        return instance

    # ── Introspection ──────────────────────────────────────────────────────

    def feature_names(self):
        """Return (word_feature_names, char_feature_names) for introspection."""
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform() first.")
        return (
            self.word_tfidf.get_feature_names_out(),
            self.char_tfidf.get_feature_names_out(),
        )

    def combined_shape(self, n_samples: int) -> tuple:
        """Return the expected shape of the combined feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("Call fit_transform() first.")
        n_word = len(self.word_tfidf.get_feature_names_out())
        n_char = len(self.char_tfidf.get_feature_names_out())
        return (n_samples, n_word + n_char)

    def summary(self) -> dict:
        """Return a dict summary of vectorizer configuration and vocab sizes."""
        if not self.is_fitted:
            return {"fitted": False}
        n_word = len(self.word_tfidf.get_feature_names_out())
        n_char = len(self.char_tfidf.get_feature_names_out())
        return {
            "fitted": True,
            "word_tfidf": {
                "params": self._word_params,
                "vocab_size": n_word,
            },
            "char_tfidf": {
                "params": self._char_params,
                "vocab_size": n_char,
            },
            "combined_dim": n_word + n_char,
        }


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _coerce_texts(texts) -> list:
    """Convert any text-like iterable to a plain Python list of strings.
    Empty strings are kept as-is (TF-IDF handles them gracefully as zero vectors).
    """
    import pandas as pd
    if isinstance(texts, pd.Series):
        return texts.fillna("").astype(str).tolist()
    return [str(t) if t is not None else "" for t in texts]
