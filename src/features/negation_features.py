"""
src/features/negation_features.py
===================================
BETC — Phase 14 Feature Engineering — Negation and Intensifier Features

Implements rule-based negation and intensifier feature extraction for Bangla text.
These features are derived from the existing configs/negations_bn.txt and
configs/intensifiers_bn.txt word lists already used by the preprocessing pipeline.

Additionally, this module extends those lists with documented Bangla linguistic
vocabulary for diminishers and negation-in-context detection.

SOURCE AND LIMITATIONS:
  - Negation word list: configs/negations_bn.txt (project config, DRAFT status,
    requires human review per RESEARCH_RULES.md Section 3.1)
  - Intensifier word list: configs/intensifiers_bn.txt (same caveat)
  - Diminisher words: compiled from common Bangla linguistic literature examples
    (সামান্য, একটু, কিছুটা, হয়তো, সম্ভবত, খানিকটা). Listed and documented below.
    [STATUS: DRAFT — requires native speaker review before publication]
  - This module does NOT claim to understand Bangla morphology or syntax.
  - Features are bag-of-words presence/count based; no dependency parsing.

FEATURES EXTRACTED:
  1.  neg_count              — count of negation tokens found in text
  2.  neg_present            — binary: at least one negation word (0/1)
  3.  intensifier_count      — count of intensifier tokens
  4.  intensifier_present    — binary: at least one intensifier
  5.  diminisher_count       — count of diminisher tokens
  6.  diminisher_present     — binary: at least one diminisher
  7.  neg_before_verb_count  — count of negation tokens that appear within
                               a 3-word window before common verb-like tokens
                               [NOTE: approximation only; no parser used]
  8.  neg_intensified_count  — negation within 3 tokens of an intensifier
                               (e.g., "একদম না" — "absolutely not")
  9.  total_modifier_count   — neg + intensifier + diminisher count combined
  10. modifier_ratio         — total_modifier_count / max(word_count, 1)

All features are computed on the PREPROCESSED text (processed_text column)
so that the word forms match those in the negation/intensifier config files
after Unicode normalization.

LEAKAGE SAFETY:
  This extractor is stateless and rule-based. It does not learn from training
  data. It may be applied identically to all splits.
"""

import re
from pathlib import Path
from typing import FrozenSet, List, Sequence

import numpy as np
import scipy.sparse as sp

# ─── Locate config files ─────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent      # src/features/
_REPO_ROOT = _HERE.parent.parent             # BETC-Bangla-Emotion/
_CONFIGS_DIR = _REPO_ROOT / "configs"

_NEGATIONS_FILE    = _CONFIGS_DIR / "negations_bn.txt"
_INTENSIFIERS_FILE = _CONFIGS_DIR / "intensifiers_bn.txt"


def _load_word_list(filepath: Path) -> FrozenSet[str]:
    """Load a newline-separated word list, ignoring comment lines and blanks."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Required config file not found: {filepath}\n"
            "See PIPELINE_SPEC.md Section 3.1 and configs/README.md."
        )
    words = set()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line)
    return frozenset(words)


# Load word lists at module import time (same as preprocessing.py pattern)
NEGATION_WORDS    = _load_word_list(_NEGATIONS_FILE)
INTENSIFIER_WORDS = _load_word_list(_INTENSIFIERS_FILE)

# Diminisher words: Bangla vocabulary for hedging / minimizing emotional intensity.
# DRAFT — sourced from common Bangla linguistic references.
# [STATUS: DRAFT — requires native speaker validation]
DIMINISHER_WORDS: FrozenSet[str] = frozenset([
    "সামান্য",      # slight / a little
    "একটু",         # a little bit
    "একটু একটু",   # little by little
    "কিছুটা",       # somewhat
    "খানিকটা",      # somewhat / a bit
    "হয়তো",        # maybe / perhaps
    "সম্ভবত",       # possibly / probably
    "মনে হয়",      # it seems
    "মনে হচ্ছে",   # seems to be
    "কিছুটা কম",   # somewhat less
    "অল্প",         # a little / slight
    "অল্পস্বল্প",  # slight
    "নাহয়",        # otherwise / maybe not
    "মাঝে মাঝে",   # sometimes
    "কখনো কখনো",   # sometimes
    "কখনও কখনও",   # sometimes (variant)
])


FEATURE_NAMES = [
    "neg_count",
    "neg_present",
    "intensifier_count",
    "intensifier_present",
    "diminisher_count",
    "diminisher_present",
    "neg_before_verb_count",
    "neg_intensified_count",
    "total_modifier_count",
    "modifier_ratio",
]

N_FEATURES = len(FEATURE_NAMES)


def _count_matches_in_tokens(tokens: List[str], word_set: FrozenSet[str]) -> int:
    """
    Count occurrences of words from word_set in tokens.
    Handles both single-word and multi-word entries (up to 3 words).
    """
    n = len(tokens)
    count = 0
    i = 0
    while i < n:
        # Try longest match first (3-gram, 2-gram, 1-gram)
        matched = False
        for length in [3, 2, 1]:
            if i + length <= n:
                candidate = " ".join(tokens[i:i + length])
                if candidate in word_set:
                    count += 1
                    i += length
                    matched = True
                    break
        if not matched:
            i += 1
    return count


def extract_one(text: str) -> np.ndarray:
    """
    Extract negation/intensifier features from a single text string.

    Parameters
    ----------
    text : str — preprocessed Bangla text (processed_text column recommended)

    Returns
    -------
    np.ndarray, shape (N_FEATURES,), dtype float32
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    tokens = text.split()
    n_words = max(len(tokens), 1)

    neg_count   = _count_matches_in_tokens(tokens, NEGATION_WORDS)
    intens_count = _count_matches_in_tokens(tokens, INTENSIFIER_WORDS)
    dimin_count  = _count_matches_in_tokens(tokens, DIMINISHER_WORDS)

    # Negation-before-verb approximation: look for negation followed within
    # 3 tokens by common Bangla verb suffixes (করা, হওয়া, যাওয়া, আসা).
    # This is a coarse approximation — no parser is used.
    _COMMON_VERB_TOKENS = frozenset(["করা", "করি", "করে", "করলে", "করব", "করছে",
                                      "হওয়া", "হয়", "হচ্ছে", "হলে", "হবে", "হয়েছে",
                                      "যাওয়া", "যায়", "যাব", "গেছে", "যাচ্ছে",
                                      "আসা", "আসে", "আসব", "এসেছে", "আসছে"])
    neg_before_verb = 0
    for j, tok in enumerate(tokens):
        if tok in NEGATION_WORDS:
            # Look 3 tokens ahead
            window = tokens[j+1: j+4]
            if any(w in _COMMON_VERB_TOKENS for w in window):
                neg_before_verb += 1

    # Negation within 3 tokens of an intensifier
    neg_intensified = 0
    for j, tok in enumerate(tokens):
        if tok in NEGATION_WORDS:
            # Window: 3 tokens before and 3 tokens after
            window = tokens[max(0, j-3): j] + tokens[j+1: j+4]
            if any(w in INTENSIFIER_WORDS for w in window):
                neg_intensified += 1

    total_mod = neg_count + intens_count + dimin_count
    mod_ratio = total_mod / n_words

    return np.array([
        neg_count,
        1.0 if neg_count > 0 else 0.0,
        intens_count,
        1.0 if intens_count > 0 else 0.0,
        dimin_count,
        1.0 if dimin_count > 0 else 0.0,
        neg_before_verb,
        neg_intensified,
        total_mod,
        mod_ratio,
    ], dtype=np.float32)


class NegationFeatureExtractor:
    """
    Stateless rule-based negation and intensifier feature extractor.

    No fitting is required. Call transform() directly on any split.

    Apply to preprocessed text (processed_text) so that Unicode-normalized
    token forms match those in the config files.

    Methods
    -------
    transform(texts) -> sp.csr_matrix, shape (n, N_FEATURES)
    feature_names()  -> list[str]
    """

    def __init__(self):
        self.is_fitted = True

    def fit(self, texts: Sequence[str]) -> "NegationFeatureExtractor":
        """No-op. Returns self for API compatibility."""
        return self

    def fit_transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Equivalent to transform()."""
        return self.transform(texts)

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """
        Extract negation/intensifier features.

        Parameters
        ----------
        texts : list[str] or pd.Series — preprocessed Bangla text.

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
            "negation_vocab_size": len(NEGATION_WORDS),
            "intensifier_vocab_size": len(INTENSIFIER_WORDS),
            "diminisher_vocab_size": len(DIMINISHER_WORDS),
            "note": (
                "Stateless rule-based extractor. Sources: configs/negations_bn.txt, "
                "configs/intensifiers_bn.txt (both DRAFT, require human review). "
                "Diminisher list is independently compiled — DRAFT status. "
                "Verb proximity features are approximations — no parser used."
            ),
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
