# BETC — Phase 6: Disgust and Surprise Error Analysis Report

**Date:** 2026-10-13  
**Dataset:** 41,184 records (28,840 train / 6,170 val / 6,174 test — canonical split)  
**Model:** Phase 5 Champion Pipeline — TF-IDF (25k) + Emoji (52) + 6-Emotion Lexicon (18) + Advanced Negation Scope (16) + Hashtags (3) = **25,089 dims**.  
**Primary Metric:** Val Macro-F1 = `0.4904` | Val Micro-F1 = `0.5821`

---

## 1. Executive Summary

Phase 6 generates validation false-positives (FP) and false-negatives (FN) for **Disgust** and **Surprise**, the two most problematic classes. Key findings:

> [!CAUTION]
> **Critical Discovery:** The EmoNoBa Disgust assumption rows (n=4,750) achieve **F1 = 0.0000** (Precision=0.000, Recall=0.000) on the validation set. Non-EmoNoBa rows achieve **F1 = 0.4508**. The Disgust labels derived from the EmoNoBa assumption are entirely unlearnable with the current feature set.

> [!IMPORTANT]
> **Disgust FP Source Split:** 314 of 492 FPs (63.8%) come from EmoNoBa rows — the model predicts Disgust on EmoNoBa texts but the ground truth (derived from the assumption) says 0, causing cascading FP errors.

---

## 2. Label Frequency (Train)

| Emotion | Train Count | Prevalence |
| :--- | ---: | :---: |
| Joy | 10,368 | 35.9% |
| Anger | 7,895 | 27.4% |
| Sadness | 6,602 | 22.9% |
| Disgust | 2,845 | 9.9% |
| Surprise | 1,749 | 6.1% |
| Fear | 1,252 | 4.3% |

Disgust and Surprise are the 4th and 5th rarest classes, directly causing precision-recall imbalance.

---

## 3. Disgust Error Analysis

### 3.1 FP / FN Counts

| Error Type | Count | Val Disgust F1 |
| :--- | :---: | :---: |
| **False Positives (FP)** | **492** | — |
| **False Negatives (FN)** | **380** | — |
| **Overall Disgust F1** | — | `0.344` |

### 3.2 FP Pattern Breakdown (predicted Disgust, actually NOT Disgust)

| Pattern | FP Count | % of FPs |
| :--- | :---: | :---: |
| Has negation | 340 | ~69.1% |
| Short text (<5 words) | ~25 | ~5.1% |
| Has emoji | ~8 | ~1.6% |
| Code-mixed | 4 | 0.8% |
| Has hashtag | ~3 | ~0.6% |

### 3.3 FN Pattern Breakdown (actually Disgust, predicted as NOT Disgust)

| Pattern | FN Count | % of FNs |
| :--- | :---: | :---: |
| **Has negation** | **245** | **64.5%** |
| Short text (<5 words) | 15 | 3.9% |
| Code-mixed | 7 | 1.8% |
| Has emoji | 2 | 0.5% |
| Has hashtag | 1 | 0.3% |

> **Interpretation:** Negation dominates both FP and FN for Disgust. The model confuses negated anger/sadness sentences with Disgust, and misses genuinely disgusted sentences that contain negation structures.

### 3.4 FP Confusion — What Was the True Label Instead?

| Actual Label (FP Target) | Count |
| :--- | :---: |
| Sadness | 237 |
| Anger | 191 |
| Fear | 36 |
| Surprise | 31 |
| Joy | 29 |

Disgust is most often falsely predicted on **Sadness** and **Anger** texts. This is expected — in the training data, Disgust co-occurs with Anger 145 times and Sadness 21 times.

### 3.5 Source Dataset Breakdown

| Source Dataset | FP | FN |
| :--- | :---: | :---: |
| **EmoNoBa** | **314** (63.8%) | 0 |
| UBMEC | 178 (36.2%) | **338** (89.0%) |
| MONOVAB | 0 | **42** (11.0%) |

> **Critical Insight:** All Disgust FNs come from UBMEC and MONOVAB — these have native Disgust annotations and the model genuinely under-predicts them. EmoNoBa contributes **zero true Disgust labels** to validation, explaining why EmoNoBa rows produce zero F1.

### 3.6 🔬 EmoNoBa Disgust Assumption Sensitivity Analysis

| Row Subset | N | F1 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **EmoNoBa assumption rows** | 4,750 | **0.0000** | 0.0000 | 0.0000 |
| **Other dataset rows** | 1,420 | **0.4508** | 0.5627 | 0.3760 |

The EmoNoBa "Disgust = not-joy-not-love" assumption is **completely inconsistent** with the Disgust distribution in UBMEC and MONOVAB. No Disgust-positive labels appear in the EmoNoBa validation subset, so the model predicts Disgust there (FP) but the assumption says 0.

### 3.7 Qualitative Examples

**FP — Predicted Disgust, Actually Sadness/Anger:**
```
[sadness] তুমরা আওয়ামীলীগ তাই তোমাদের কে আমরা ঘৃণা করি।
          (You are Awami League so we hate you — lexical disgust signal "ঘৃণা" triggers disgust, but label is Sadness)

[anger]   গল্প বলা নিয়ে অদ্ভুত একটা প্রেম। আমরা শুধু একটা গল্প বলতে চেয়েছি
          (Strange love about storytelling... — FP due to ambiguous emotional register)
```

**FN — Actually Disgust, Predicted Otherwise:**
```
[disgust] ক্লাস ১০ এ পড়া একটা ছেলে বা মেয়ে দুনিয়ার কি বুঝে?
          (What does a class 10 student understand about the world? — implicit contempt, no explicit disgust word)

[disgust] যুদ্ধেতো ধ্বংসযজ্ঞ হবেই, ফুলের বাগান করার জন্যতো আর কেউ যুদ্ধ করে না।
          (Of course war destroys things — implicit disgust/contempt requires pragmatic inference)
```

---

## 4. Surprise Error Analysis

### 4.1 FP / FN Counts

| Error Type | Count | Val Surprise F1 |
| :--- | :---: | :---: |
| **False Positives (FP)** | **286** | — |
| **False Negatives (FN)** | **245** | — |
| **Overall Surprise F1** | — | `0.329` |

### 4.2 FP Pattern Breakdown

| Pattern | FP Count | % of FPs |
| :--- | :---: | :---: |
| **Has negation** | **144** | **50.3%** |
| Has emoji | 55 | 19.2% |
| Short text (<5 words) | 28 | 9.8% |
| Code-mixed | 1 | 0.3% |
| Has hashtag | 0 | 0.0% |

### 4.3 FN Pattern Breakdown

| Pattern | FN Count | % of FNs |
| :--- | :---: | :---: |
| **Has negation** | **116** | **47.3%** |
| Short text (<5 words) | 34 | 13.9% |
| Has emoji | 7 | 2.9% |
| Code-mixed | 0 | 0.0% |
| Has hashtag | 1 | 0.4% |

### 4.4 FP Confusion

| Actual Label (FP Target) | Count |
| :--- | :---: |
| Sadness | 88 |
| Anger | 74 |
| Joy | 69 |
| Fear | 54 |
| Disgust | 16 |

Surprise is falsely predicted across all other emotions nearly uniformly, suggesting the model's Surprise threshold is too permissive or features are insufficiently discriminating.

### 4.5 Source Dataset Breakdown

| Source Dataset | FP | FN |
| :--- | :---: | :---: |
| **EmoNoBa** | **208** (72.7%) | **168** (68.6%) |
| UBMEC | 75 (26.2%) | 76 (31.0%) |
| MONOVAB | 3 (1.0%) | 1 (0.4%) |

EmoNoBa dominates both FP and FN for Surprise — the dataset's implicit "surprise = wow/amazement" annotation schema may differ from UBMEC's definition.

### 4.6 Qualitative Examples

**FN — Actually Surprise, Model Missed:**
```
[surprise]        এখন আয় দৈনিক ২ লাখ রুপি!" (Now earning 2 lakh rupees daily — no explicit surprise word)
[joy, surprise]   অনুষ্ঠানটি সভাপতিত্ব করেন... (Formal announcement text — surprise embedded in context)
[anger, surprise] ভিউ কমে যায় কেন?গেলো ঈদেও এমণ দেখলাম। (Why do views go down? — rhetorical surprise)
```

---

## 5. Label Co-occurrence in True Labels

| Focus Label | Co-occurs With | Count |
| :--- | :--- | :---: |
| DISGUST | anger | 145 |
| DISGUST | sadness | 21 |
| DISGUST | surprise | 1 |
| DISGUST | fear | 1 |
| SURPRISE | joy | 24 |
| SURPRISE | anger | 15 |
| SURPRISE | sadness | 15 |
| SURPRISE | fear | 4 |

Disgust co-occurs overwhelmingly with **Anger** in true labels (145 pairs), creating inherent ambiguity that drives FP confusion.

---

## 6. Prioritized Improvement Proposals

| Priority | Target | Finding | Proposed Action | Phase |
| :---: | :--- | :--- | :--- | :---: |
| **P1** 🔴 | Disgust | EmoNoBa rows produce F1=0.000 due to assumption mismatch | Investigate reweighting or excluding EmoNoBa Disgust assumption rows in training | Phase 8 |
| **P2** 🔴 | Disgust | 64.5% of Disgust FNs have negation — model mistakes negated sadness/anger for Disgust | Expand Disgust-specific negation scope features; add neg_disgust penalty term | Phase 7+ |
| **P3** 🟠 | Surprise | Threshold for Surprise too permissive (286 FPs) | Fine-tune Surprise threshold in Phase 7 sweep — likely needs raising | Phase 7 |
| **P4** 🟠 | Disgust+Surprise | FPs dominated by sadness & anger co-occurring texts | Consider per-class C regularization search (lower C for Disgust/Surprise chain links) | Phase 9 |
| **P5** 🟡 | Surprise | 19.2% of Surprise FPs have emoji — emoji falsely signals Surprise | Add Surprise-specific emoji negative filter (e.g., 😂 is Joy not Surprise) | Phase 5.2 ext. |
| **P6** 🟡 | General | Class imbalance: Surprise=1,749 vs Joy=10,368 | Phase 8 class-weight experiments with higher weight for Disgust & Surprise | Phase 8 |

---

## 7. Deliverables

- **Experiment Script:** `scripts/experiments/phase6_error_analysis.py`
- **JSON Report:** `results/ablations/phase6_error_analysis/phase6_error_report.json`
- **FP/FN CSVs:** `results/ablations/phase6_error_analysis/fp_disgust.csv`, `fn_disgust.csv`, `fp_surprise.csv`, `fn_surprise.csv`
- **Unit & Regression Tests:** `tests/test_phase6_error_analysis.py` (10 passed)
