# BETC — Phase 9: Individual Classifier Comparison Report

**Date:** 2026-10-28  
**Feature set:** Phase 8b Champion (25,101 dims)  
**C1-LR Reference:** Macro-F1=`0.4921` | Micro-F1=`0.5834`

---

## Strategy Summary

| Classifier | Description | Dims | Val Macro-F1 | Δ Macro | Val Micro-F1 | Δ Micro | Beats C1? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1_LogisticRegression** ✅ | LR C=1.0, balanced, saga (Phase 8b champ) | 25,101 | `0.4921` | `+0.0000` | `0.5834` | `+0.0000` | ⚪ |
| **C2_ComplementNB** | ComplementNB α=1.0 | 25,101 | `0.5049` | `+0.0128` | `0.5589` | `-0.0245` | ❌ |
| **C3_SGD_SVM** | SGD modified_huber (SVM-like), balanced | 25,101 | `0.4644` | `-0.0277` | `0.5615` | `-0.0219` | ❌ |
| **C4_KNN** | KNN k=5, cosine, TruncatedSVD(200) | 200 | `0.3719` | `-0.1202` | `0.4474` | `-0.1360` | ❌ |

## Per-Label F1 Breakdown

| Classifier | anger | disgust | fear | joy | sadness | surprise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1_LogisticRegression** | 0.550 | 0.348 | 0.394 | 0.825 | 0.503 | 0.332 |
| **C2_ComplementNB** | 0.527 | 0.354 | 0.522 | 0.792 | 0.478 | 0.356 |
| **C3_SGD_SVM** | 0.518 | 0.339 | 0.333 | 0.824 | 0.488 | 0.284 |
| **C4_KNN** | 0.475 | 0.270 | 0.181 | 0.679 | 0.427 | 0.198 |

## Selection

**Best individual classifier:** `C1_LogisticRegression`  
**Val Macro-F1:** `0.4921` | **Val Micro-F1:** `0.5834`

> Selection rule: A challenger beats C1 only if Macro-F1 > 0.4921 AND no label drops >0.01 F1 below C1.

## Deliverables
- `scripts/experiments/phase9_classifier_comparison.py`
- `results/ablations/phase9_classifiers/phase9_report.json`
- `tests/test_phase9_classifiers.py`