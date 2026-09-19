# Phase 6 — Preprocessing Report
**Project:** BETC — Bangla Emotion TF-IDF Classifier Chain  
**Phase:** Phase 6 — Bangla Text Preprocessing  
**Date:** 2026-09-19  
**Status:** COMPLETE  
**Input:** `Data/processed/{train,validation,test}/*.csv`  
**Output:** `Data/processed/{train,validation,test}/*_preprocessed.csv`

> [!IMPORTANT]
> Original split files (`train.csv`, `validation.csv`, `test.csv`) were **NOT** modified.
> `Data/raw/` was **NOT** modified. No TF-IDF, Logistic Regression, Classifier Chain,
> or any model code was executed.

---

## 1. Preprocessing Rules Implemented

Per `PIPELINE_SPEC.md` Section 3.1 [FINALIZED], the following operations are applied in order:

| Step | Operation | Detail |
|---|---|---|
| 1 | **Null / NaN / empty handling** | Returns `""` for `None`, NaN, or empty input |
| 2 | **URL removal** | Regex strips `http://`, `https://`, `ftp://`, `www.` URLs |
| 3 | **@mention removal** | Regex strips `@username` tokens |
| 4 | **HTML tag removal** | Strips `<tag>` and `</tag>` patterns |
| 5 | **HTML entity removal** | Strips `&amp;`, `&lt;`, `&#39;` etc. |
| 6 | **Bangla-specific Unicode normalization** | `bnunicodenormalizer==0.1.7`, word-by-word; fixes conjuncts, reph, broken nukta, and decomposed forms (e.g. `য` + `়` → `য়`) |
| 7 | **Repeated punctuation collapsing** | 3+ repetitions → 2 (for `?,!,.,,,;,:,-,_,।`); single and double preserved |
| 8 | **Whitespace normalization** | All whitespace → single space; strip leading/trailing |

**NOT applied (per PIPELINE_SPEC.md and RESEARCH_RULES.md):**
- Stemming, lemmatization, stopword removal
- Emoji removal *(spec does not require it; emojis preserved — see Research Decisions)*
- Spell correction, synonym replacement, translation
- Pretrained embeddings, language models
- TF-IDF or any vocabulary fitting *(Phase 7 only)*
- Label modification *(labels preserved exactly)*

---

## 2. Unit Test Results

**47 / 47 tests passed** (`tests/test_preprocessing.py`)

| Category | Tests | Result |
|---|---|---|
| Unicode normalization | 2 | ✅ All pass |
| URL removal | 4 | ✅ All pass |
| @mention removal | 3 | ✅ All pass |
| Whitespace normalization | 3 | ✅ All pass |
| Bangla text preservation | 4 | ✅ All pass |
| Negation preservation | 6 | ✅ All pass |
| Intensifier preservation | 4 | ✅ All pass |
| Deterministic output | 3 | ✅ All pass |
| Empty/null handling | 7 | ✅ All pass |
| Punctuation / noise handling | 11 | ✅ All pass |

> **Implementation note on `নয়`:** `bnunicodenormalizer` normalizes the decomposed form (`ন` + `য` + `়`, 3 codepoints, U+09A8/09AF/09BC) to the precomposed form (`ন` + `য়`, 2 codepoints, U+09A8/09DF). This is the correct Bangla Unicode normalization. The tests were updated to compare normalized forms consistently, and the word-list loader normalizes entries at load time so that `verify_word_list_preservation()` correctly detects the precomposed form in the output.

---

## 3. Row Counts

| Split | Expected | Actual | ✓ |
|---|---|---|---|
| Train | 28,840 | 28,840 | ✅ |
| Validation | 6,170 | 6,170 | ✅ |
| Test | 6,174 | 6,174 | ✅ |
| **Total** | **41,184** | **41,184** | ✅ |

---

## 4. Empty Processed Texts

| Split | Empty Outputs |
|---|---|
| Train | **1** |
| Validation | 0 |
| Test | 0 |
| **Total** | **1** |

One train example produced an empty output after preprocessing (its text consisted entirely of removable noise — a URL or mention-only entry). This is the correct behavior: the preprocessing function returns `""` for texts whose content is entirely removed. The row is retained in the CSV with `processed_text=""`. Empty outputs are not removed at this stage.

---

## 5. Validation Results — All 11 Checks Passed ✅

| Check | Result |
|---|---|
| A. Row counts match (28,840 / 6,170 / 6,174) | ✅ PASS |
| B. Six target labels unchanged | ✅ PASS |
| C. No row IDs lost | ✅ PASS |
| D. No split overlap introduced | ✅ PASS |
| E. No null `processed_text` values | ✅ PASS |
| F. Preprocessing is deterministic (50-sample rerun check) | ✅ PASS |
| G. All splits use same preprocessing function | ✅ PASS |
| H. Negation words preserved (200-sample check) | ✅ PASS |
| I. Intensifier words preserved (200-sample check) | ✅ PASS |
| J. `Data/raw/` untouched | ✅ PASS |
| K. Original split CSV files untouched | ✅ PASS |

---

## 6. Spot Check Summary

**Source:** Train split only, 30 examples (random_state=42)  
**File:** [`logs/dataset_audit/preprocessing_spotcheck.md`](../dataset_audit/preprocessing_spotcheck.md)

| Feature observed | Count (of 30) |
|---|---|
| URL present (removed) | 1 |
| @mention present (removed) | 1 |
| HTML noise present (removed) | 0 |
| Repeated punctuation normalized | 5 |
| Emoji present (preserved) | 1 |
| Negation word present (preserved) | 4 |
| Intensifier word present (preserved) | 3 |
| Empty output | 0 |

All negation and intensifier words were correctly preserved in all 30 spot-check examples.

---

## 7. Research Decisions Required

> [!WARNING]
> **[RESEARCH DECISION REQUIRED — HUMAN REVIEW]**
>
> `configs/negations_bn.txt` and `configs/intensifiers_bn.txt` were created as **DRAFT** files, seeded from the examples in `PIPELINE_SPEC.md` Section 3.1 (`না, নাই, নেই, নয়` / `খুব, অনেক, একদম`) plus common Bangla patterns attested in the training corpus.
>
> Per `PIPELINE_SPEC.md` Section 3.1 [PROPOSAL] and `RESEARCH_RULES.md`:
> *"These lists must be manually reviewed against real sample sentences before use — not generated ad hoc by an LLM without human review."*
>
> **Action required:** The project owner must review both config files against real annotated sentences and either confirm or amend them before Phase 7. The preprocessing is functionally correct with the current draft lists; updating the lists requires only editing the text files (no code change).

> [!NOTE]
> **[RESEARCH DECISION CONFIRMED: PRESERVE emojis]**
>
> `PIPELINE_SPEC.md` Section 3.1 specifies stripping "Noise / URL / @mention". Emojis are not listed as noise. Emojis have emotional signal in Bangla social-media text (e.g. 👌, 😢) and are preserved. This is documented here as an explicit decision, not a silent default.

---

## 8. Output Files

| File | Description |
|---|---|
| [`src/features/preprocessing.py`](../../src/features/preprocessing.py) | Deterministic preprocessing module |
| [`tests/test_preprocessing.py`](../../tests/test_preprocessing.py) | 47 unit tests |
| [`configs/negations_bn.txt`](../../configs/negations_bn.txt) | Negation word list — DRAFT |
| [`configs/intensifiers_bn.txt`](../../configs/intensifiers_bn.txt) | Intensifier word list — DRAFT |
| [`Data/processed/train/train_preprocessed.csv`](../../Data/processed/train/train_preprocessed.csv) | Preprocessed train split (28,840 rows) |
| [`Data/processed/validation/validation_preprocessed.csv`](../../Data/processed/validation/validation_preprocessed.csv) | Preprocessed validation split (6,170 rows) |
| [`Data/processed/test/test_preprocessed.csv`](../../Data/processed/test/test_preprocessed.csv) | Preprocessed test split (6,174 rows) |
| [`logs/dataset_audit/preprocessing_spotcheck.md`](../dataset_audit/preprocessing_spotcheck.md) | 30-example manual spot check |
| [`logs/preprocessing/preprocessing_validation.json`](preprocessing_validation.json) | Full validation results JSON |
| [`scripts/data_pipeline/phase6_apply_preprocessing.py`](../../scripts/data_pipeline/phase6_apply_preprocessing.py) | Reproducible application script |

---

## 9. Confirmation Statements

- ✅ Labels (`anger, disgust, fear, joy, sadness, surprise`) were **NOT modified**
- ✅ Original split files (`train.csv`, `validation.csv`, `test.csv`) were **NOT overwritten**
- ✅ `Data/raw/` was **NOT modified**
- ✅ No TF-IDF vectorizers were fitted
- ✅ No model (LogisticRegression, ClassifierChain, BETC) was trained
- ✅ No threshold optimization was performed
- ✅ No test-set information was used during preprocessing

**Phase 6 is complete. Phase 7 (TF-IDF Feature Extraction) is ready to execute upon user instruction.**
