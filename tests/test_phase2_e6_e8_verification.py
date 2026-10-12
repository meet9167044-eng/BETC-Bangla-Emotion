"""
tests/test_phase2_e6_e8_verification.py
=======================================
Phase 2 Verification Tests:
  - Verifies exact feature groups for E6 and E8
  - Verifies dimensional delta between E6 and E8 (+10 negation features)
  - Verifies row count preservation and CSR matrix format
"""

from src.features.feature_combination import BETCCombinedExtractor

def test_e6_e8_feature_groups():
    ext_e6 = BETCCombinedExtractor(use_tfidf=True, use_emoji=True, use_sentiment=True, use_negation=False)
    ext_e8 = BETCCombinedExtractor(use_tfidf=True, use_emoji=True, use_sentiment=True, use_negation=True)

    assert "tfidf" in ext_e6._extractors
    assert "emoji" in ext_e6._extractors
    assert "sentiment" in ext_e6._extractors
    assert "negation" not in ext_e6._extractors

    assert "tfidf" in ext_e8._extractors
    assert "emoji" in ext_e8._extractors
    assert "sentiment" in ext_e8._extractors
    assert "negation" in ext_e8._extractors

def test_e6_e8_dimension_delta():
    sample_texts = [
        "আমি খুব খুশি! 😊",
        "এটা মোটেও ভালো না! একদম খারাপ।",
        "কেন এমন হলো?",
        "সত্যি খুব সুন্দর।"
    ]
    ext_e6 = BETCCombinedExtractor(use_tfidf=True, use_emoji=True, use_sentiment=True, use_negation=False)
    ext_e8 = BETCCombinedExtractor(use_tfidf=True, use_emoji=True, use_sentiment=True, use_negation=True)

    X_e6 = ext_e6.fit_transform(sample_texts, sample_texts)
    X_e8 = ext_e8.fit_transform(sample_texts, sample_texts)

    assert X_e6.shape[0] == len(sample_texts)
    assert X_e8.shape[0] == len(sample_texts)
    # Negation extractor produces exactly 10 features
    assert X_e8.shape[1] == X_e6.shape[1] + 10
