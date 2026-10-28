# BETC — Bangla Emotion Text Classification

**Bangla Emotion Text Classification (BETC)** is a high-performance, lightweight, multi-label emotion classification framework for Bengali text across six emotion categories: **Anger, Disgust, Fear, Joy, Sadness, and Surprise**.

The framework strictly relies on interpretable feature engineering and dual ensemble classifier chains—achieving competitive performance without requiring heavy deep neural networks or pretrained language models.

---

## 🏆 Final Benchmark Performance (Phase 12 Held-out Test Set)

Evaluated strictly once on the held-out canonical test set (**6,174 samples**) with decision thresholds frozen from validation:

| Pipeline | Test Macro-F1 | Test Micro-F1 | Macro-Precision | Macro-Recall | Hamming Loss | Subset Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0 TF-IDF Baseline** | `0.4578` | `0.5512` | — | — | — | — |
| **S0 LR ClassifierChain** | `0.4837` | `0.5766` | `0.4355` | `0.5772` | `0.1881` | `0.3612` |
| **★ Champion S3 Ensemble** | **`0.5107`** | **`0.5725`** | **`0.4534`** | **`0.6364`** | `0.1988` | `0.3382` |

> **Net Gains**: The Champion pipeline achieves **+0.0529 Macro-F1** over the TF-IDF baseline and **+0.0270 Macro-F1** over the single-model chain on the test set.

### Per-Emotion Test F1 Breakdown

| Emotion | Test F1 | Test Precision | Test Recall | S0 Baseline F1 | Gain ($\Delta$) | Decision Threshold ($\tau^*$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Anger** | **0.5420** | 0.4070 | 0.8109 | 0.5462 | -0.0042 | 0.286 |
| **Disgust** | **0.3752** | 0.2784 | 0.5754 | 0.3476 | **+0.0276** | 0.269 |
| **Fear** | **0.4619** | 0.5185 | 0.4164 | 0.3648 | **+0.0971** | 0.313 |
| **Joy** | **0.8141** | 0.8041 | 0.8244 | 0.8182 | -0.0041 | 0.542 |
| **Sadness** | **0.4943** | 0.3606 | 0.7859 | 0.4981 | -0.0038 | 0.205 |
| **Surprise** | **0.3767** | 0.3519 | 0.4053 | 0.3274 | **+0.0493** | 0.262 |

---

## 🔬 Architecture & Methodology

### 1. Feature Representation (25,101 Dimensions)
- **Word + Character TF-IDF (25,000 dims)**: Word n-grams (1–3) and character n-grams (2–6) capturing morphological variants and word patterns.
- **Emoji Per-Indicator Counts (52 dims)**: Direct occurrences of sentiment and affective emojis.
- **Bangla 6-Emotion Lexicon (6 dims)**: Aggregated counts from curated domain lexicons.
- **Expanded Negation Scope & Polarity Shift (28 dims)**: Detection of explicit negative particles (`না`, `নাই`, `নেই`, `নয়`, etc.), window-based negation scopes, and polar inversions.
- **Hashtag Sentiment (3 dims)**: Normalised frequency and presence of affective hashtags (`MaxAbsScaled`).
- **Group C Intensifier-Emotion Bigrams (12 dims)**: Co-occurrence patterns between Bengali intensifiers (`খুব`, `অনেক`, `ভীষণ`) and emotional lexemes (`MaxAbsScaled`).

### 2. Dual Ensemble Classifier Chains
- **Model A (70% weight)**: `ClassifierChain` with `LogisticRegression(C=1.0, class_weight='balanced', solver='saga')`.
- **Model B (30% weight)**: `ClassifierChain` with `ComplementNB(alpha=1.0)`.
- **Chain Order**: Descending training frequency `[Joy (3), Anger (0), Sadness (4), Disgust (1), Surprise (5), Fear (2)]`.
- **Decision Rule**: Soft-vote probability blending:
  $$P_{\text{final}} = 0.70 \cdot P_{\text{LR}} + 0.30 \cdot P_{\text{CNB}}$$
  Predictions are thresholded using the calibrated per-label threshold vector $\tau^*$.

---

## 📁 Repository Structure & Deliverables

- `configs/`
  - `final_model_config.json`: Master configuration with hyperparameters and metric records.
  - `phase10_thresholds.json`: Calibrated per-label decision thresholds.
- `artifacts/final_model/`
  - `model_lr_chain.joblib`: Serialized Logistic Regression ClassifierChain.
  - `model_cnb_chain.joblib`: Serialized Complement Naive Bayes ClassifierChain.
  - `extractor_base.joblib`: Fitted TF-IDF and Emoji extractor.
  - `scaler_hash.joblib`, `scaler_group_c.joblib`: Fitted scalers for aux features.
- `results/final/`
  - `phase12_final_test_report.md` & `.json`: Canonical test evaluation reports.
  - `test_predictions_champion.csv`: Probabilities, binary predictions, and ground truth for test set.
- `scripts/experiments/`
  - `phase11_multiseed_robustness.py`: Reproducible multi-seed evaluation.
  - `phase12_final_evaluation.py`: Final test evaluation runner.
- `BETC_Phase_by_Phase_Improvement_Plan.md`: Complete audit and log of Phases 0 through 12.

---

## 🚀 Reproduction Instructions

### Run Unit Tests
```bash
python -m pytest tests/ -v
```

### Reproduce Final Test Evaluation
```bash
python scripts/experiments/phase12_final_evaluation.py
```

### Run Multi-Seed Robustness Validation
```bash
python scripts/experiments/phase11_multiseed_robustness.py
```

