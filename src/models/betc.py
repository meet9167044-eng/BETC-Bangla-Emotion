"""
src/models/betc.py
==================
BETC — Bangla Emotion TF-IDF Classifier Chain (M1)

Implements the full BETC pipeline as specified in PIPELINE_SPEC.md Sections 3.5–3.8.

Architecture:
  Combined 25,000-dim TF-IDF features
    → ClassifierChain (frequency-ordered labels)
      → LogisticRegression(C=1.0, class_weight='balanced') per link
        → Per-class threshold optimization (val set only)
          → Test prediction (one-shot)

Chain label ordering rule (PIPELINE_SPEC.md §3.5 [INITIAL-HP]):
  Descending training-set label frequency (most frequent emotion first).
  The actual order used MUST be logged in configs/betc_full.yaml.

Leakage rules (EVALUATION_PROTOCOL.md §3, PIPELINE_SPEC.md §3.7):
  - Chain fitted on TRAINING data only.
  - Thresholds selected from VALIDATION predicted probabilities only.
  - Test set evaluated exactly once, after all decisions are frozen.
"""

import os
import json
import joblib
import numpy as np
import scipy.sparse as sp

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

TARGET_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

CHAIN_MODEL_FILE  = "betc_full_chain.joblib"
THRESHOLDS_FILE   = "thresholds.json"


def compute_label_order(Y_train: np.ndarray, label_names=None) -> list:
    """
    Compute the chain label order by descending training-set frequency.

    Per PIPELINE_SPEC.md §3.5 [INITIAL-HP]:
    'Default chain order: descending label frequency in the training set
     (most frequent emotion predicted first).'

    Parameters
    ----------
    Y_train     : np.ndarray, shape (n_train, n_labels)
    label_names : list[str], default TARGET_COLS

    Returns
    -------
    order : list[int]  — column indices in descending frequency order
    label_order_names : list[str] — label names in chain order
    label_frequencies : dict {label_name: int}  — raw positive counts
    """
    if label_names is None:
        label_names = TARGET_COLS

    freq = Y_train.sum(axis=0)  # shape (n_labels,)
    order = list(np.argsort(freq)[::-1])  # descending

    label_order_names = [label_names[i] for i in order]
    label_frequencies = {label_names[i]: int(freq[i]) for i in range(len(label_names))}

    return order, label_order_names, label_frequencies


class BETCModel:
    """
    Full BETC Classifier Chain model (M1).

    State
    -----
    chain      : sklearn.multioutput.ClassifierChain  (fitted on train)
    thresholds : dict {label_name: float}  (optimized on val)
    chain_order_indices : list[int]
    chain_order_names   : list[str]
    label_frequencies   : dict {label_name: int}
    is_fitted  : bool

    Methods
    -------
    fit(X_train, Y_train)
        Fits the ClassifierChain on training data.
    optimize_thresholds(X_val, Y_val)
        Sweeps thresholds on validation probabilities. Must be called after fit().
    predict(X)
        Returns binary predictions using stored per-class thresholds.
    predict_proba(X)
        Returns raw probability matrix from the chain.
    save(model_dir, threshold_path)
        Persists fitted chain and thresholds.
    load(model_dir, threshold_path)
        Class method: reloads saved chain and thresholds.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000,
                 solver: str = "saga", random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.chain = None
        self.thresholds = {}
        self.chain_order_indices = None
        self.chain_order_names = None
        self.label_frequencies = None
        self.is_fitted = False

    # ── Fitting ────────────────────────────────────────────────────────────

    def fit(self, X_train: sp.csr_matrix, Y_train: np.ndarray,
            label_names=None) -> "BETCModel":
        """
        Fit the ClassifierChain on TRAINING data only.

        Parameters
        ----------
        X_train     : sparse matrix, shape (n_train, 25000)
        Y_train     : np.ndarray, shape (n_train, 6)
        label_names : list[str] — must match column order of Y_train

        Returns self (for method chaining).
        """
        if label_names is None:
            label_names = TARGET_COLS

        # Compute chain order (descending frequency)
        order, order_names, freqs = compute_label_order(Y_train, label_names)
        self.chain_order_indices = order
        self.chain_order_names   = order_names
        self.label_frequencies   = freqs

        # Build and fit ClassifierChain
        base_clf = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
        )
        self.chain = ClassifierChain(base_clf, order=order, random_state=self.random_state)
        self.chain.fit(X_train, Y_train)
        self.is_fitted = True
        return self

    # ── Threshold optimization (val only) ─────────────────────────────────

    def optimize_thresholds(self, X_val: sp.csr_matrix, Y_val: np.ndarray,
                            label_names=None,
                            tau_min: float = 0.05,
                            tau_max: float = 0.95,
                            tau_step: float = 0.01) -> dict:
        """
        Optimize per-class thresholds on VALIDATION probabilities only.

        PIPELINE_SPEC.md §3.7: 'store the resulting per-class thresholds
        as an artifact — never recompute or "peek" at test-set probabilities
        when choosing thresholds.'

        Parameters
        ----------
        X_val, Y_val : validation features and labels
        tau_min/max/step : sweep range

        Returns thresholds dict (also stored as self.thresholds).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before optimize_thresholds().")

        if label_names is None:
            label_names = TARGET_COLS

        from src.models.evaluate import optimize_thresholds as _opt
        P_val = self.predict_proba(X_val)
        self.thresholds = _opt(P_val, Y_val, tau_min, tau_max, tau_step, label_names)
        return self.thresholds

    # ── Inference ──────────────────────────────────────────────────────────

    def predict_proba(self, X) -> np.ndarray:
        """Return raw probability matrix from the chain, shape (n, 6)."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        return self.chain.predict_proba(X)

    def predict(self, X, label_names=None) -> np.ndarray:
        """
        Return binary predictions using stored per-class thresholds.

        Requires optimize_thresholds() to have been called first.
        """
        if not self.thresholds:
            raise RuntimeError("Call optimize_thresholds() before predict().")
        if label_names is None:
            label_names = TARGET_COLS
        from src.models.evaluate import apply_thresholds
        P = self.predict_proba(X)
        return apply_thresholds(P, self.thresholds, label_names)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, model_dir: str, threshold_path: str = None) -> None:
        """
        Persist the fitted chain (joblib) and thresholds (JSON).

        Parameters
        ----------
        model_dir      : directory to save betc_full_chain.joblib
        threshold_path : path for thresholds JSON (default: artifacts/thresholds.json)
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted BETCModel.")
        os.makedirs(model_dir, exist_ok=True)
        chain_path = os.path.join(model_dir, CHAIN_MODEL_FILE)
        joblib.dump(self.chain, chain_path)

        if threshold_path is not None:
            os.makedirs(os.path.dirname(threshold_path), exist_ok=True)
            with open(threshold_path, "w", encoding="utf-8") as f:
                json.dump(self.thresholds, f, indent=2)

    @classmethod
    def load(cls, model_dir: str, threshold_path: str = None) -> "BETCModel":
        """Reload a saved BETCModel."""
        chain_path = os.path.join(model_dir, CHAIN_MODEL_FILE)
        if not os.path.exists(chain_path):
            raise FileNotFoundError(f"Model file not found: {chain_path}")

        instance = cls.__new__(cls)
        instance.chain = joblib.load(chain_path)
        instance.is_fitted = True
        instance.thresholds = {}
        instance.chain_order_indices = None
        instance.chain_order_names = None
        instance.label_frequencies = None
        instance.C = 1.0
        instance.max_iter = 1000

        if threshold_path and os.path.exists(threshold_path):
            with open(threshold_path, encoding="utf-8") as f:
                instance.thresholds = json.load(f)

        return instance

    def summary(self) -> dict:
        """Return a summary dict for logging/reporting."""
        return {
            "fitted": self.is_fitted,
            "chain_order_names": self.chain_order_names,
            "chain_order_indices": self.chain_order_indices,
            "label_frequencies": self.label_frequencies,
            "thresholds": self.thresholds,
            "base_classifier": "LogisticRegression",
            "hyperparameters": {
                "C": self.C,
                "class_weight": "balanced",
                "max_iter": self.max_iter,
                "solver": self.solver,
            },
        }
