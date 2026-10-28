# BETC — Phase 10: Classifier Chains vs Ensembles Report

**Date:** 2026-10-28  
**Feature set:** Phase 8b Champion (25,101 dims)  
**S0-LR Reference:** Macro-F1=`0.4921` | Micro-F1=`0.5834`

---

## Strategy Summary

| Strategy | Description | Val Macro-F1 | Δ Macro | Val Micro-F1 | Δ Micro | Beats S0? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S0_LR_Chain** | LR ClassifierChain (Phase 9 champion) | `0.4921` | `+0.0000` | `0.5834` | `+0.0000` | ⚪ Ref |
| **S1_CNB_Chain** | CNB ClassifierChain (Phase 9 unconstrained best) | `0.5049` | `+0.0128` | `0.5589` | `-0.0245` | ❌ |
| **S2_Vote_50_50** | Soft vote: 0.50 LR + 0.50 CNB | `0.5192` | `+0.0271` | `0.5715` | `-0.0119` | ❌ |
| **S3_Vote_LR70_CNB30** ✅ | Soft vote: 0.70 LR + 0.30 CNB | `0.5170` | `+0.0249` | `0.5746` | `-0.0088` | ✅ |
| **S4_Vote_LR30_CNB70** | Soft vote: 0.30 LR + 0.70 CNB | `0.5179` | `+0.0258` | `0.5717` | `-0.0117` | ❌ |
| **S5_LabelWise_Oracle** | Oracle: per-label best (UPPER BOUND, not deployable) | `0.5185` | `+0.0264` | `0.5814` | `-0.0020` | 🔭 Oracle |

## Per-Label F1 Breakdown

| Strategy | anger | disgust | fear | joy | sadness | surprise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0_LR_Chain** | 0.550 | 0.348 | 0.394 | 0.825 | 0.503 | 0.332 |
| **S1_CNB_Chain** | 0.527 | 0.354 | 0.522 | 0.792 | 0.478 | 0.356 |
| **S2_Vote_50_50** | 0.546 | 0.367 | 0.536 | 0.808 | 0.490 | 0.367 |
| **S3_Vote_LR70_CNB30** | 0.544 | 0.363 | 0.516 | 0.818 | 0.499 | 0.362 |
| **S4_Vote_LR30_CNB70** | 0.544 | 0.369 | 0.535 | 0.797 | 0.488 | 0.374 |
| **S5_LabelWise_Oracle** | 0.550 | 0.354 | 0.522 | 0.825 | 0.503 | 0.356 |

## Selection

**Selected strategy:** `S3_Vote_LR70_CNB30`  
**Val Macro-F1:** `0.5170` | **Val Micro-F1:** `0.5746`

> Selection rule: beats S0 if Macro-F1 > 0.4921 AND no label drops >0.01 F1.
> S5 (oracle) is excluded from selection — it uses val performance to select per-label model.

## Deliverables
- `scripts/experiments/phase10_ensemble_comparison.py`
- `results/ablations/phase10_ensemble/phase10_report.json`
- `tests/test_phase10_ensemble.py`