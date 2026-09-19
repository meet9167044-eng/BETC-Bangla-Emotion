"""
src/features/preprocessing.py
==============================
BETC — Bangla Emotion TF-IDF Classifier Chain
Phase 6 — Bangla Text Preprocessing Pipeline

Implements the deterministic preprocessing function specified in
PIPELINE_SPEC.md Section 3.1 [FINALIZED].

Preprocessing steps (in order):
  1. Null / empty handling — return empty string for None / NaN / empty
  2. URL removal             — strip http/https/www URLs
  3. @mention removal        — strip Twitter/Facebook @mentions
  4. HTML noise removal      — strip common HTML tags and entities
  5. Bangla-specific Unicode normalization — via bnunicodenormalizer (word-level)
  6. Repeated punctuation normalization    — collapse runs of ?,!,. etc.
  7. Whitespace normalization              — collapse any whitespace to single space

What is NOT done (per PIPELINE_SPEC.md):
  - NO stemming or lemmatization
  - NO stopword removal
  - NO synonym replacement / translation
  - NO spell correction
  - NO emoji removal  (PIPELINE_SPEC.md does not specify removal;
                       emojis are preserved as-is — see note below)
  - NO TF-IDF / vocabulary fitting (Phase 7 only)
  - NO label modification

EMOJI NOTE:
  PIPELINE_SPEC.md Section 3.1 says to strip "noise/URL/@mention" and
  preserve negation/intensifier semantics. Emojis are NOT listed as noise
  to strip and have emotional signal in Bangla social-media text.
  They are therefore PRESERVED.
  [RESEARCH DECISION REQUIRED — confirmed as preserve for now]

NEGATION / INTENSIFIER PRESERVATION NOTE:
  The preprocessing function does not strip any words. Because the only
  text modifications are URL removal, @mention removal, HTML removal,
  Unicode normalization, repeated-punctuation collapsing, and whitespace
  collapsing, negation and intensifier words are automatically preserved
  (they are never targeted by any removal step). The configs/negations_bn.txt
  and configs/intensifiers_bn.txt files are loaded and a post-step
  verification confirms they survived in the output.

DETERMINISM:
  This function is stateless and purely rule-based. Given the same input
  string it always produces the same output. There is no randomness,
  corpus-level statistics, or learned component.

LEAKAGE SAFETY:
  This function learns nothing from any split (train, validation, or test).
  It may be applied to all three splits identically.
"""

import re
import unicodedata
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ─── Locate the configs directory ─────────────────────────────────────────────
# Works from any working directory by resolving relative to this file's location.
_HERE = Path(__file__).resolve().parent          # src/features/
_REPO_ROOT = _HERE.parent.parent                 # BETC-Bangla-Emotion/
_CONFIGS_DIR = _REPO_ROOT / "configs"

_NEGATIONS_FILE    = _CONFIGS_DIR / "negations_bn.txt"
_INTENSIFIERS_FILE = _CONFIGS_DIR / "intensifiers_bn.txt"


def _load_word_list(filepath: Path) -> frozenset:
    """Load a newline-separated word list, ignoring comment lines and blanks.

    Each word is normalized through bnunicodenormalizer at load time so that
    the stored forms match what the preprocessing function will produce.
    This ensures verify_word_list_preservation() and tests work correctly
    even when the config file uses a different but visually identical Unicode
    representation (e.g. decomposed য + ় vs precomposed য়).
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Required config file not found: {filepath}\n"
            "Please create this file before running preprocessing. "
            "See PIPELINE_SPEC.md Section 3.1 and configs/README.md."
        )

    # Import normalizer here to avoid circular dependency at module load
    try:
        from bnunicodenormalizer import Normalizer as _Normalizer
        _norm = _Normalizer()

        def _normalize_word(w):
            result = _norm(w)
            normalized = result.get("normalized")
            return normalized if normalized is not None else w
    except ImportError:
        # Fallback if bnunicodenormalizer not installed: return words as-is
        # (preprocessing() itself will also fail, so this is consistent)
        def _normalize_word(w):
            return w

    words = set()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Normalize each word (handles multi-word entries token by token)
                normalized_tokens = [_normalize_word(tok) for tok in line.split()]
                words.add(" ".join(normalized_tokens))
    return frozenset(words)



# Load word lists once at module import time (thread-safe, immutable frozensets).
NEGATION_WORDS    = _load_word_list(_NEGATIONS_FILE)
INTENSIFIER_WORDS = _load_word_list(_INTENSIFIERS_FILE)

# ─── Bangla-specific Unicode normalizer ───────────────────────────────────────
# Load lazily (bnunicodenormalizer can be slow to import) but share the instance.
_bn_normalizer = None

def _get_bn_normalizer():
    """Return (and cache) the bnunicodenormalizer.Normalizer instance."""
    global _bn_normalizer
    if _bn_normalizer is None:
        try:
            from bnunicodenormalizer import Normalizer
            _bn_normalizer = Normalizer()
        except ImportError as exc:
            raise ImportError(
                "bnunicodenormalizer is required for Phase 6 preprocessing.\n"
                "Install with: pip install bnunicodenormalizer"
            ) from exc
    return _bn_normalizer


def _normalize_bangla_unicode(text: str) -> str:
    """
    Apply Bangla-specific Unicode normalization word-by-word using
    bnunicodenormalizer.Normalizer.

    PIPELINE_SPEC.md Section 3.1 [FINALIZED]:
      "Normalize Unicode using a Bangla-specific normalizer (e.g.
       bnunicodenormalizer), not a generic NFC/NFKC pass — generic
       normalization mishandles Bangla conjuncts and reph."

    The normalizer is applied token-by-token. If the normalizer returns
    None for a token (happens when the token is entirely composed of
    invalid/unmappable Unicode), the original token is preserved unchanged
    so that no text is silently dropped.
    """
    normalizer = _get_bn_normalizer()
    tokens = text.split()
    normalized_tokens = []
    for tok in tokens:
        result = normalizer(tok)
        norm = result.get("normalized")
        # Preserve original if normalizer returns None (empty/unmappable token)
        normalized_tokens.append(norm if norm is not None else tok)
    return " ".join(normalized_tokens)


# ─── Compiled regex patterns (compiled once at module load) ───────────────────

# URLs: http/https/ftp with optional www, or bare www.domain.tld
_RE_URL = re.compile(
    r"(?:https?://|ftp://|www\.)\S+",
    re.IGNORECASE
)

# @mentions: any @word (handles Bangla and ASCII usernames)
_RE_MENTION = re.compile(r"@\S+")

# HTML tags: <tag> and </tag> patterns
_RE_HTML_TAG = re.compile(r"<[^>]+>")

# HTML entities: &amp; &lt; &#39; etc.
_RE_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")

# Repeated punctuation: collapse 2+ identical punctuation chars to at most 2
# Covers ?, !, ., ,, ;, :, -, _ and Bangla danda ।
# We preserve up to 2 repetitions to retain expressive emphasis (e.g. !!)
# but collapse !!!!!!! to !!
_RE_REPEATED_PUNCT = re.compile(r"([?!.,;:\-_।])\1{2,}")

# Whitespace: any sequence of whitespace → single space
_RE_WHITESPACE = re.compile(r"\s+")


# ─── Public API ───────────────────────────────────────────────────────────────

def preprocess(text, empty_result: str = "") -> str:
    """
    Apply the BETC deterministic Bangla preprocessing pipeline to a single
    text string.

    Parameters
    ----------
    text : str | None | float (NaN)
        Raw input text. None and NaN-like values are treated as empty.
    empty_result : str
        Value to return when the input is None / NaN / empty string, or when
        all content is removed after cleaning. Default: "".

    Returns
    -------
    str
        Cleaned, Unicode-normalized text. Always a plain Python str.

    Pipeline
    --------
    Step 1: Null / NaN / empty handling
    Step 2: URL removal
    Step 3: @mention removal
    Step 4: HTML tag + entity removal
    Step 5: Bangla-specific Unicode normalization (word-by-word, bnunicodenormalizer)
    Step 6: Repeated punctuation collapsing (2+ → max 2)
    Step 7: Whitespace normalization (any whitespace → single space, strip)

    NOT applied: stemming, lemmatization, stopword removal, emoji removal,
    spell correction, synonym replacement, translation, any corpus-level
    learning. See module docstring for rationale.
    """
    # Step 1: handle null / empty / NaN
    if text is None:
        return empty_result
    # Catch pandas NaN (float) and similar
    if not isinstance(text, str):
        try:
            import math
            if math.isnan(float(text)):
                return empty_result
        except (TypeError, ValueError):
            pass
        text = str(text)
    text = text.strip()
    if not text:
        return empty_result

    # Step 2: URL removal
    text = _RE_URL.sub(" ", text)

    # Step 3: @mention removal
    text = _RE_MENTION.sub(" ", text)

    # Step 4: HTML noise removal (tags then entities)
    text = _RE_HTML_TAG.sub(" ", text)
    text = _RE_HTML_ENTITY.sub(" ", text)

    # Step 5: Bangla-specific Unicode normalization (word-by-word)
    # Only run if there is something to normalize
    text = text.strip()
    if text:
        text = _normalize_bangla_unicode(text)

    # Step 6: Repeated punctuation collapsing  (!!!! → !!)
    text = _RE_REPEATED_PUNCT.sub(r"\1\1", text)

    # Step 7: Whitespace normalization
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text if text else empty_result


def preprocess_series(series, empty_result: str = ""):
    """
    Apply preprocess() to a pandas Series of texts.

    Parameters
    ----------
    series : pd.Series
        Series of raw text strings.
    empty_result : str
        Passed through to preprocess().

    Returns
    -------
    pd.Series
        Series of preprocessed strings (same index as input).
    """
    return series.map(lambda t: preprocess(t, empty_result=empty_result))


def verify_word_list_preservation(text_in: str, text_out: str) -> dict:
    """
    Check that every negation and intensifier word present in text_in
    also appears in text_out (as a substring). Returns a dict with
    'negations_lost' and 'intensifiers_lost' lists.

    Used for unit-testing and the validation report; NOT called at
    runtime during bulk preprocessing (the preprocessing function does
    not remove these words by design).
    """
    def words_in(text):
        return set(text.split())

    in_words  = words_in(text_in)
    out_words = words_in(text_out)

    neg_in   = in_words.intersection(NEGATION_WORDS)
    intens_in = in_words.intersection(INTENSIFIER_WORDS)

    neg_lost   = neg_in   - out_words
    intens_lost = intens_in - out_words

    return {
        "negations_present_in_input":    sorted(neg_in),
        "intensifiers_present_in_input": sorted(intens_in),
        "negations_lost":                sorted(neg_lost),
        "intensifiers_lost":             sorted(intens_lost),
        "all_preserved":                 (len(neg_lost) == 0 and len(intens_lost) == 0),
    }
