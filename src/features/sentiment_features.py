"""
src/features/sentiment_features.py
=====================================
BETC — Phase 14 Feature Engineering — Bangla Sentiment Features

Implements rule-based Bangla sentiment feature extraction using a curated
Bangla sentiment lexicon.

DESIGN RATIONALE AND SOURCE DOCUMENTATION:
  - SentiWordNet: NOT used here. SentiWordNet covers English WordNet synsets and
    does not have a Bangla version. Applying English scores to Bangla text would
    introduce systematic error. Rejected per Phase 5B guidelines.
  - nabilbinbillal/Bangla-NLP: Repository exists but requires pretrained model
    downloads and an active internet connection for inference. This violates the
    "no network requests during training" and "no large model downloads" constraints.
    Not used.
  - bnemo (faruk-ahmad/bnemo): Repository was inspected; it provides emoji-based
    sentiment mappings, not a general text sentiment lexicon. Not used here.

APPROACH — Transparent Bangla Sentiment Lexicon:
  We implement a compact, documented Bangla sentiment lexicon derived from:
    1. Common Bangla emotional vocabulary attested in Ekman emotion research.
    2. Words directly observable in the BETC training corpus emotional contexts
       (surface observation only — no label-based induction to avoid leakage).
    3. Bangla sentiment words listed in publicly available research papers on
       Bangla opinion mining (e.g., Alam & Yao 2020 survey on Bangla NLP).

  STATUS: DRAFT. This lexicon is a reasonable starting point for research but
  requires validation by a native Bangla speaker and cross-checking against a
  published peer-reviewed lexicon before results can be claimed in a publication.

  The lexicon is stored in configs/sentiment_lexicon_bn.txt (created by this module
  on first run if not present) with clear DRAFT annotations.

FEATURES EXTRACTED:
  1.  sent_pos_count       — count of positive sentiment words
  2.  sent_neg_count       — count of negative sentiment words
  3.  sent_net_score       — pos_count - neg_count (signed sentiment score)
  4.  sent_pos_ratio       — pos_count / max(word_count, 1)
  5.  sent_neg_ratio       — neg_count / max(word_count, 1)
  6.  sent_abs_score       — abs(pos_count - neg_count) (sentiment magnitude)
  7.  sent_polarity        — sign of net_score: +1 (positive), -1 (negative), 0 (neutral)
  8.  sent_has_positive    — binary: any positive word present
  9.  sent_has_negative    — binary: any negative word present
  10. sent_mixed           — binary: both positive AND negative words present

NOTE: These sentiment features are INDEPENDENT of the six emotion labels.
Positive/negative sentiment correlates with but does not determine emotions.
For example: "Disgust" texts may be sentiment-negative, but not all negative
sentiment texts are "Disgust". The features are supplementary, not definitive.

LEAKAGE SAFETY:
  This extractor is stateless. The lexicon is fixed at module load time.
  No fitting is required. Safe to apply identically to all splits.
"""

from pathlib import Path
from typing import FrozenSet, List, Sequence

import numpy as np
import scipy.sparse as sp

# ─── Locate config directory ─────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
_CONFIGS_DIR = _REPO_ROOT / "configs"
_LEXICON_FILE = _CONFIGS_DIR / "sentiment_lexicon_bn.txt"


# ─── Built-in lexicon (inline) ───────────────────────────────────────────────
# This is saved to configs/sentiment_lexicon_bn.txt on first run.
# Format: one entry per line — "WORD POLARITY" where POLARITY is pos or neg.
#
# SOURCE DOCUMENTATION:
#   These words are drawn from:
#     (a) Ekman emotion vocabulary commonly used in Bangla NLP research
#     (b) Surface-form observation of BETC corpus (no label-induction)
#     (c) Common Bangla positive/negative descriptors in opinion mining literature
#   STATUS: DRAFT — requires validation.

_BUILTIN_LEXICON_TEXT = """\
# configs/sentiment_lexicon_bn.txt
# Bangla Sentiment Lexicon for BETC Phase 14
#
# STATUS: DRAFT — requires native speaker validation and cross-referencing
# against a published peer-reviewed Bangla sentiment lexicon before use
# in published results.
#
# SOURCES:
#   (a) Ekman emotion vocabulary in Bangla NLP research
#   (b) Surface-form observation of BETC dataset context (no label-induction)
#   (c) Common Bangla descriptors in Bengali opinion mining literature
#       (e.g., vocabulary from Das & Bandyopadhyay 2010, Alam & Yao 2020 survey)
#
# FORMAT: WORD POLARITY
# POLARITY: pos (positive) | neg (negative)
#
# ─── Positive words ───────────────────────────────────────────────────────────
ভালো pos
ভালোবাসা pos
ভালোবাসি pos
ভালোলাগা pos
সুন্দর pos
আনন্দ pos
খুশি pos
সুখ pos
সুখী pos
মজা pos
মজার pos
দারুণ pos
অসাধারণ pos
চমৎকার pos
শান্তি pos
শান্তিপূর্ণ pos
হাসি pos
হাসিখুশি pos
প্রেম pos
প্রিয় pos
আদর pos
স্নেহ pos
ধন্যবাদ pos
কৃতজ্ঞ pos
কৃতজ্ঞতা pos
অভিনন্দন pos
শুভ pos
উৎসাহ pos
উৎসাহী pos
সফল pos
সাফল্য pos
বিজয় pos
জয় pos
গর্ব pos
গর্বিত pos
উপভোগ pos
মনোরম pos
আশা pos
স্বপ্ন pos
প্রেরণা pos
অনুপ্রেরণা pos
স্বাগত pos
মিষ্টি pos
সহানুভূতি pos
সহমর্মিতা pos
ভক্তি pos
শ্রদ্ধা pos
বিস্ময়কর pos
অবাক pos
আশ্চর্য pos
আশ্চর্যজনক pos
উত্তেজনা pos
উত্তেজিত pos
উল্লাস pos
উল্লসিত pos
প্রশংসা pos
প্রশংসনীয় pos
অপূর্ব pos
নিখুঁত pos
নিরাপদ pos
আরামদায়ক pos
সুরক্ষিত pos
বিনোদন pos
পছন্দ pos
পছন্দনীয় pos
উপকারী pos
উপকার pos
সহায়তা pos
সাহায্য pos
আশীর্বাদ pos
শুভকামনা pos
শুভেচ্ছা pos
তৃপ্তি pos
সন্তোষ pos
সন্তুষ্ট pos
উজ্জ্বল pos
পরিষ্কার pos
সত্যিই ভালো pos
# ─── Negative words ───────────────────────────────────────────────────────────
খারাপ neg
ঘৃণা neg
ঘৃণ্য neg
রাগ neg
রাগান্বিত neg
ক্ষোভ neg
ক্রোধ neg
দুঃখ neg
দুঃখিত neg
কষ্ট neg
যন্ত্রণা neg
ব্যথা neg
ভয় neg
ভীতি neg
আতঙ্ক neg
ভয়ঙ্কর neg
ভয়াবহ neg
বিরক্ত neg
বিরক্তি neg
বিরক্তিকর neg
অসুখী neg
দুর্ভোগ neg
হতাশা neg
হতাশ neg
নিরাশা neg
নিরাশ neg
বিষণ্নতা neg
বিষণ্ণ neg
অবমাননা neg
অপমান neg
লজ্জা neg
লজ্জিত neg
ব্যর্থ neg
ব্যর্থতা neg
পরাজয় neg
পরাজিত neg
মৃত্যু neg
শোক neg
কান্না neg
কাঁদা neg
অশ্রু neg
চিৎকার neg
অভিযোগ neg
ঝগড়া neg
ঝামেলা neg
সমস্যা neg
বিপদ neg
বিপজ্জনক neg
ভয়ের neg
আতঙ্কিত neg
উদ্বেগ neg
উদ্বিগ্ন neg
ক্লান্ত neg
ক্লান্তি neg
বিরক্তকর neg
অপছন্দ neg
ঘৃণিত neg
জঘন্য neg
নোংরা neg
নষ্ট neg
বিশ্বাসঘাতক neg
বিশ্বাসঘাতকতা neg
প্রতারণা neg
প্রতারক neg
মিথ্যা neg
মিথ্যুক neg
অন্যায় neg
অত্যাচার neg
নির্যাতন neg
নিষ্ঠুর neg
নিষ্ঠুরতা neg
কুৎসিত neg
ক্ষতি neg
ক্ষতিকর neg
অসুস্থ neg
রোগ neg
বেদনা neg
দুর্দশা neg
অসহায় neg
একাকী neg
একাকীত্ব neg
মন খারাপ neg
মনোকষ্ট neg
বিপর্যয় neg
দুর্ঘটনা neg
মারামারি neg
হত্যা neg
অপরাধ neg
"""

FEATURE_NAMES = [
    "sent_pos_count",
    "sent_neg_count",
    "sent_net_score",
    "sent_pos_ratio",
    "sent_neg_ratio",
    "sent_abs_score",
    "sent_polarity",
    "sent_has_positive",
    "sent_has_negative",
    "sent_mixed",
]

N_FEATURES = len(FEATURE_NAMES)


def _load_or_create_lexicon() -> tuple:
    """
    Load lexicon from configs/sentiment_lexicon_bn.txt.
    Creates the file from the built-in lexicon if it doesn't exist.

    Returns
    -------
    pos_words : frozenset[str]
    neg_words : frozenset[str]
    """
    if not _LEXICON_FILE.exists():
        _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_LEXICON_FILE, "w", encoding="utf-8") as f:
            f.write(_BUILTIN_LEXICON_TEXT)

    pos_words = set()
    neg_words = set()
    with open(_LEXICON_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.rsplit(None, 1)  # split on last whitespace
            if len(parts) == 2:
                word, polarity = parts
                if polarity.lower() == "pos":
                    pos_words.add(word.strip())
                elif polarity.lower() == "neg":
                    neg_words.add(word.strip())
    return frozenset(pos_words), frozenset(neg_words)


# Load lexicon at module import time
POSITIVE_WORDS, NEGATIVE_WORDS = _load_or_create_lexicon()


def _count_words(tokens: List[str], word_set: FrozenSet[str]) -> int:
    """Count occurrences of single-word entries from word_set in tokens."""
    return sum(1 for t in tokens if t in word_set)


def extract_one(text: str) -> np.ndarray:
    """
    Extract sentiment features from a single text string.

    Parameters
    ----------
    text : str — preprocessed Bangla text (processed_text recommended).

    Returns
    -------
    np.ndarray, shape (N_FEATURES,), dtype float32
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    tokens = text.split()
    n_words = max(len(tokens), 1)

    pos_count = _count_words(tokens, POSITIVE_WORDS)
    neg_count = _count_words(tokens, NEGATIVE_WORDS)

    net_score = pos_count - neg_count
    pos_ratio = pos_count / n_words
    neg_ratio = neg_count / n_words
    abs_score = abs(net_score)

    if net_score > 0:
        polarity = 1.0
    elif net_score < 0:
        polarity = -1.0
    else:
        polarity = 0.0

    has_pos = 1.0 if pos_count > 0 else 0.0
    has_neg = 1.0 if neg_count > 0 else 0.0
    mixed   = 1.0 if (pos_count > 0 and neg_count > 0) else 0.0

    return np.array([
        pos_count,
        neg_count,
        float(net_score),
        pos_ratio,
        neg_ratio,
        float(abs_score),
        polarity,
        has_pos,
        has_neg,
        mixed,
    ], dtype=np.float32)


class SentimentFeatureExtractor:
    """
    Stateless Bangla sentiment feature extractor based on a curated lexicon.

    IMPORTANT LIMITATIONS (always document these when reporting results):
      - This lexicon is DRAFT status and has not been validated by a native speaker.
      - The lexicon is based on common vocabulary, not a peer-reviewed resource.
      - Sentiment features may not transfer well to all domains in the corpus.
      - These features are distinct from the six emotion labels.
      - Do not report this as a "validated Bangla sentiment analyzer" in publications
        until the lexicon has been reviewed and compared against a gold standard.

    Methods
    -------
    transform(texts) -> sp.csr_matrix, shape (n, N_FEATURES)
    feature_names()  -> list[str]
    """

    def __init__(self):
        self.is_fitted = True  # stateless

    def fit(self, texts: Sequence[str]) -> "SentimentFeatureExtractor":
        """No-op. Returns self for API compatibility."""
        return self

    def fit_transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Equivalent to transform()."""
        return self.transform(texts)

    def transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        """
        Extract sentiment features.

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
            "positive_vocab_size": len(POSITIVE_WORDS),
            "negative_vocab_size": len(NEGATIVE_WORDS),
            "lexicon_file": str(_LEXICON_FILE),
            "note": (
                "DRAFT lexicon. Sources: common Bangla emotional vocabulary from "
                "Ekman emotion research and Bangla opinion mining literature (Das & "
                "Bandyopadhyay 2010, Alam & Yao 2020). Requires native speaker "
                "validation. SentiWordNet was evaluated but rejected (English only). "
                "nabilbinbillal/Bangla-NLP was evaluated but rejected (requires "
                "internet/model downloads at inference time)."
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
