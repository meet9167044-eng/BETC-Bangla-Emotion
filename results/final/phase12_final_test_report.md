# BETC — Phase 12: Final Test Evaluation & Reporting

**Date:** 2026-10-28  
**Champion Pipeline:** `Phase 10 S3 Soft Vote (0.70 LR + 0.30 CNB on 25,101 dims)`  
**Test Set:** Canonical Harmonized Test Set (`6,174` rows)  

---

## Executive Summary

The canonical evaluation was conducted strictly once on the held-out test set using models fitted on training data and per-label decision thresholds frozen from validation.

| Evaluation Split | Pipeline | Macro-F1 | Micro-F1 | Macro-Precision | Macro-Recall | Hamming Loss | Subset Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation (Sanity)** | S3 Champion | `0.5170` | `0.5746` | `0.4595` | `0.6446` | `0.1988` | `0.3357` |
| **Test Set (Held-out)** | **S3 Champion** | `**0.5107**` | `**0.5725**` | `0.4534` | `0.6364` | `0.1988` | `0.3382` |
| **Test Set (Comparison)** | S0 (LR Chain) | `0.4837` | `0.5766` | `0.4355` | `0.5772` | `0.1881` | `0.3612` |
| **Test Set (Baseline)** | E0 (TF-IDF) | `0.4578` | `0.5512` | — | — | — | — |

> **Test Set Improvements:** Champion S3 outperforms the E0 baseline by **+0.0529 Macro-F1** and outperforms the S0 single-model chain by **+0.0270 Macro-F1**.

---

## Per-Label Test Performance Breakdown

| Emotion | Precision | Recall | F1 Score | S0 Test F1 | $\Delta$ vs S0 | Frozen Threshold ($\tau^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anger** | `0.4070` | `0.8109` | `**0.5420**` | `0.5462` | `-0.0042` | `0.286` |
| **Disgust** | `0.2784` | `0.5754` | `**0.3752**` | `0.3476` | `+0.0276` | `0.269` |
| **Fear** | `0.5185` | `0.4164` | `**0.4619**` | `0.3648` | `+0.0971` | `0.313` |
| **Joy** | `0.8041` | `0.8244` | `**0.8141**` | `0.8182` | `-0.0041` | `0.542` |
| **Sadness** | `0.3606` | `0.7859` | `**0.4943**` | `0.4981` | `-0.0038` | `0.205` |
| **Surprise** | `0.3519` | `0.4053` | `**0.3767**` | `0.3274` | `+0.0493` | `0.262` |

---

## Harmonization Assumptions & Limitations

### 1. EmoNoBa Inferred Disgust-Negative Labels
- In the EmoNoBa benchmark subset, disgust was not annotated natively. During harmonization, missing disgust annotations were inferred as negative.
- **Empirical Impact Analysis on Test:**
  - On subsets with native disgust annotation: Disgust F1 = `0.5616` (Precision = `0.5484`, Recall = `0.5754`).
  - On EmoNoBa assumed-negative subset: Disgust Precision = `0.0000` with low false positive rate.

### 2. Canonical Test Set Usage Limitation
- The canonical test set was explored in preliminary research during earlier project development (September 2026).
- To prevent data snooping and overfitting:
  - All feature engineering, classifier selection, and threshold optimizations were strictly performed on the validation split only.
  - The test set was touched exactly once in Phase 12 to generate these final benchmark numbers.

---

## Reproducibility & Saved Artifacts
- Final Pipeline Configuration: `configs/final_model_config.json`
- Calibrated Decision Thresholds: `configs/phase10_thresholds.json`
- Saved Model Weights: `artifacts/final_model/`
  - `model_lr_chain.joblib` (LR ClassifierChain)
  - `model_cnb_chain.joblib` (CNB ClassifierChain)
  - `extractor_base.joblib` (TF-IDF + Emoji Combined Extractor)
  - `scaler_hash.joblib`, `scaler_group_c.joblib`
- Test Predictions: `results/final/test_predictions_champion.csv`

===========================================================================