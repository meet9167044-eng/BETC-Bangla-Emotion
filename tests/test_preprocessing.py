import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
"""
tests/test_preprocessing.py
============================
BETC Phase 6 — Unit tests for src/features/preprocessing.py

Tests use only small synthetic strings, never real test-split examples.
Run with:   python -m pytest tests/test_preprocessing.py -v
            OR: python tests/test_preprocessing.py  (standalone)
"""

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.features.preprocessing import (
    preprocess,
    preprocess_series,
    verify_word_list_preservation,
    NEGATION_WORDS,
    INTENSIFIER_WORDS,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _assert(condition, msg=""):
    """Minimal assertion helper (works without pytest)."""
    if not condition:
        raise AssertionError(msg)


# ─── 1. Unicode normalization ─────────────────────────────────────────────────

def test_unicode_normalization_preserves_valid_bangla():
    """Valid Bangla text must survive Unicode normalization.
    
    Note: bnunicodenormalizer may change the internal Unicode representation
    of some Bangla characters (e.g. normalizing য় variants) while preserving
    visual/semantic meaning. We check that the output is non-empty and has
    the same number of tokens as the input, not byte-identical token matching.
    """
    text = "আমি গান গাই"
    result = preprocess(text)
    _assert(isinstance(result, str) and len(result) > 0, "Non-empty Bangla must produce non-empty output")
    # Same number of space-separated tokens
    _assert(len(result.split()) == len(text.split()),
            f"Token count changed: {len(text.split())} -> {len(result.split())}")


def test_unicode_normalization_is_applied():
    """Normalization must not crash and must return a non-empty string."""
    text = "ভালো লাগলো"
    result = preprocess(text)
    _assert(isinstance(result, str), "Result must be str")
    _assert(len(result) > 0, "Non-empty input must produce non-empty output")


# ─── 2. URL removal ──────────────────────────────────────────────────────────

def test_url_http_removed():
    result = preprocess("এটা দেখুন http://example.com ভালো")
    _assert("http" not in result, "HTTP URL must be removed")
    _assert("ভালো" in result, "Bangla text must survive URL removal")

def test_url_https_removed():
    result = preprocess("https://www.facebook.com/something ধন্যবাদ")
    _assert("https" not in result)
    _assert("ধন্যবাদ" in result)

def test_url_www_removed():
    result = preprocess("www.youtube.com/watch?v=abc123 মজার ভিডিও")
    _assert("www" not in result)
    _assert("মজার" in result)

def test_url_removal_only_url():
    result = preprocess("http://example.com")
    # After URL removal and stripping, result should be empty
    _assert(result == "", f"Expected empty string, got {result!r}")


# ─── 3. @mention removal ─────────────────────────────────────────────────────

def test_mention_removed():
    result = preprocess("@user123 আপনাকে ধন্যবাদ")
    _assert("@user123" not in result)
    _assert("ধন্যবাদ" in result)

def test_mention_bangla_username():
    result = preprocess("@বাংলা_user কেমন আছেন")
    _assert("@" not in result)
    _assert("কেমন" in result)

def test_mention_only():
    result = preprocess("@onlymention")
    _assert("@" not in result)


# ─── 4. Whitespace normalization ─────────────────────────────────────────────

def test_multiple_spaces_collapsed():
    result = preprocess("আমি   তোমাকে   ভালোবাসি")
    _assert("  " not in result, "Multiple spaces must be collapsed to single")

def test_tabs_and_newlines_normalized():
    result = preprocess("ভালো\t\t লাগলো\nআমার")
    _assert("\t" not in result)
    _assert("\n" not in result)
    _assert("  " not in result)

def test_leading_trailing_whitespace_stripped():
    result = preprocess("  ভালো  ")
    _assert(result == result.strip(), "Leading/trailing spaces must be stripped")


# ─── 5. Bangla text preservation ─────────────────────────────────────────────

def test_bangla_characters_preserved():
    bangla = "বাংলাদেশ একটি সুন্দর দেশ"
    result = preprocess(bangla)
    for word in bangla.split():
        _assert(word in result, f"Bangla word {word!r} must be preserved")

def test_bangla_numbers_preserved():
    text = "১০০ টাকা দিয়েছি"
    result = preprocess(text)
    _assert("১০০" in result)

def test_short_bangla_preserved():
    text = "না"
    result = preprocess(text)
    _assert(result == "না", f"Single negation word must be preserved, got {result!r}")

def test_mixed_bangla_english_preserves_bangla():
    text = "আমি ok আছি"
    result = preprocess(text)
    _assert("আমি" in result)
    _assert("আছি" in result)


# ─── 6. Negation preservation ────────────────────────────────────────────────

def test_negation_na_preserved():
    text = "আমি যাব না"
    result = preprocess(text)
    _assert("না" in result, f"Negation 'না' must be preserved, got {result!r}")

def test_negation_nai_preserved():
    text = "সে নাই"
    result = preprocess(text)
    _assert("নাই" in result)

def test_negation_nei_preserved():
    text = "এখানে কেউ নেই"
    result = preprocess(text)
    _assert("নেই" in result)

def test_negation_noy_preserved():
    """নয় (normalization-aware): the normalizer converts decomposed য+় to precomposed য়.
    We verify that the semantically equivalent normalized form survives."""
    from bnunicodenormalizer import Normalizer
    _n = Normalizer()
    # Normalize the test word itself so we compare apples to apples
    noy_normalized = _n('নয়')['normalized'] or 'নয়'
    text = 'এটা ঠিক নয়'
    result = preprocess(text)
    _assert(noy_normalized in result,
            "Normalized form of negation 'নয়' must be present in output")

def test_all_core_negations_preserved():
    """All four negations from PIPELINE_SPEC.md Section 3.1 must survive.
    
    Uses bnunicodenormalizer to obtain the expected normalized form of each
    negation word before comparing, since normalization may change internal
    Unicode representation (e.g. decomposed vs precomposed forms).
    """
    from bnunicodenormalizer import Normalizer
    _n = Normalizer()
    core_negations = ['না', 'নাই', 'নেই', 'নয়']
    for neg in core_negations:
        norm_neg = _n(neg)['normalized'] or neg
        text = 'আমি ' + neg + ' করব'
        result = preprocess(text)
        _assert(norm_neg in result,
                'Core negation ' + repr(neg) + ' (normalized: ' + repr(norm_neg) + ') was removed by preprocessing')

def test_negation_in_complex_sentence():
    text = "এই কাজ করা একেবারে ঠিক না, মোটেই না"
    result = preprocess(text)
    _assert("না" in result)

def test_verify_word_list_preservation_fn():
    """verify_word_list_preservation must detect no loss for clean input."""
    text_in = "আমি যাব না, খুব ভালো লাগলো"
    text_out = preprocess(text_in)
    info = verify_word_list_preservation(text_in, text_out)
    _assert(info["all_preserved"], f"Word list preservation failed: {info}")


# ─── 7. Intensifier preservation ─────────────────────────────────────────────

def test_intensifier_khub_preserved():
    text = "খুব সুন্দর হয়েছে"
    result = preprocess(text)
    _assert("খুব" in result)

def test_intensifier_onek_preserved():
    text = "অনেক ভালো লাগলো"
    result = preprocess(text)
    _assert("অনেক" in result)

def test_intensifier_ekdom_preserved():
    text = "একদম সঠিক কথা"
    result = preprocess(text)
    _assert("একদম" in result)

def test_all_core_intensifiers_preserved():
    """All three intensifiers from PIPELINE_SPEC.md Section 3.1 must survive."""
    core_intensifiers = ["খুব", "অনেক", "একদম"]
    for intens in core_intensifiers:
        text = f"{intens} ভালো লাগলো"
        result = preprocess(text)
        _assert(intens in result, f"Core intensifier {intens!r} was removed by preprocessing")


# ─── 8. Deterministic output ─────────────────────────────────────────────────

def test_deterministic_same_input():
    text = "আমি খুব ভালো আছি না"
    results = [preprocess(text) for _ in range(5)]
    _assert(len(set(results)) == 1, "preprocess() must be deterministic (5 calls gave different results)")

def test_deterministic_url_and_mention():
    text = "https://example.com @user আমি আছি"
    r1 = preprocess(text)
    r2 = preprocess(text)
    _assert(r1 == r2, "preprocess() must be deterministic on URL+mention input")

def test_deterministic_empty():
    _assert(preprocess("") == preprocess(""))
    _assert(preprocess(None) == preprocess(None))


# ─── 9. Empty / null handling ────────────────────────────────────────────────

def test_none_returns_empty_string():
    result = preprocess(None)
    _assert(result == "", f"None input must return '', got {result!r}")

def test_empty_string_returns_empty():
    result = preprocess("")
    _assert(result == "", f"Empty string must return '', got {result!r}")

def test_whitespace_only_returns_empty():
    result = preprocess("   ")
    _assert(result == "", f"Whitespace-only must return '', got {result!r}")

def test_nan_float_returns_empty():
    import math
    result = preprocess(float("nan"))
    _assert(result == "", f"NaN must return '', got {result!r}")

def test_url_only_returns_empty():
    result = preprocess("https://example.com")
    _assert(result == "", f"URL-only must return '', got {result!r}")

def test_mention_only_returns_empty():
    result = preprocess("@someuser")
    _assert(result == "", f"Mention-only must return '', got {result!r}")

def test_custom_empty_result():
    result = preprocess(None, empty_result="[EMPTY]")
    _assert(result == "[EMPTY]")


# ─── 10. Punctuation / noise handling ────────────────────────────────────────

def test_repeated_exclamations_collapsed():
    text = "ভালো!!!!!!"
    result = preprocess(text)
    _assert("!!!!!!" not in result, "Repeated ! must be collapsed")
    _assert("!!" in result, "Collapsed !! must be present")

def test_repeated_questions_collapsed():
    text = "কেন????"
    result = preprocess(text)
    _assert("????" not in result)
    _assert("??" in result)

def test_repeated_dots_collapsed():
    text = "হ্যাঁ............."
    result = preprocess(text)
    _assert("............." not in result)

def test_single_punct_preserved():
    text = "এটা কি?"
    result = preprocess(text)
    _assert("?" in result, "Single ? must be preserved")

def test_two_repeated_preserved():
    text = "সত্যিই!!"
    result = preprocess(text)
    _assert("!!" in result, "Two consecutive ! must be preserved (collapse only 3+)")

def test_html_tags_removed():
    text = "<b>ভালো</b> লাগলো <br/>"
    result = preprocess(text)
    _assert("<" not in result)
    _assert(">" not in result)
    _assert("ভালো" in result)

def test_html_entities_removed():
    text = "সুন্দর &amp; ভালো"
    result = preprocess(text)
    _assert("&amp;" not in result)
    _assert("ভালো" in result)

def test_emoji_preserved():
    """Emojis are NOT removed — PIPELINE_SPEC.md does not require emoji removal."""
    text = "অনেক ভালো 👌👌"
    result = preprocess(text)
    _assert("👌" in result, "Emojis must be preserved (spec does not require removal)")

def test_bangla_danda_preserved():
    """Bangla full stop (danda ।) must be preserved (single instance)."""
    text = "ধন্যবাদ।"
    result = preprocess(text)
    _assert("।" in result, "Bangla danda must be preserved")

def test_complex_noisy_text():
    """Full end-to-end noisy text with URL, mention, repeated punct, and Bangla."""
    text = "@admin https://t.co/xyz আপনার কাছে না চাই!!! খুব ভালো লাগলো।"
    result = preprocess(text)
    _assert("@" not in result)
    _assert("https" not in result)
    _assert("!!!" not in result)
    _assert("না" in result)
    _assert("খুব" in result)
    _assert("ভালো" in result)


# ─── Run all tests standalone ─────────────────────────────────────────────────

def _run_all():
    """Run all test functions and report results."""
    test_fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    failures = []
    for fn in test_fns:
        try:
            fn()
            passed += 1
            print("  [PASS] " + fn.__name__)
        except Exception as exc:
            failed += 1
            err_msg = str(exc).encode('utf-8', errors='replace').decode('utf-8')
            failures.append((fn.__name__, err_msg))
            print("  [FAIL] " + fn.__name__ + ": " + err_msg)
    print("\n" + "="*60)
    print("Results: {} passed, {} failed out of {} tests".format(passed, failed, passed + failed))
    if failures:
        print("\nFailed tests:")
        for name, msg in failures:
            print("  - " + name + ": " + msg)
    return failed == 0


if __name__ == "__main__":
    print("Running BETC Phase 6 preprocessing unit tests...")
    ok = _run_all()
    sys.exit(0 if ok else 1)
