# BETC — Phase 8: Class Imbalance Report

**Date:** 2026-10-15  
**Pipeline:** Phase 5 Champion + Phase 7 thresholds (25,089 dims)  
**Entry Macro-F1:** `0.4913` | Entry Micro-F1: `0.5825`

---

## Strategy Comparison

| Strategy | Description | Macro-F1 | Δ Macro | Micro-F1 | Δ Micro |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **S0_balanced_current** ✅ | class_weight=balanced (Phase 5 champion + Phase 7 thresholds) | `0.4913` | `+0.0000` | `0.5825` | `+0.0000` |
| **S1_no_weighting** | class_weight=None (uniform) | `0.4897` | `-0.0016` | `0.5515` | `-0.0310` |
| **S2_cw_2x_positive** | class_weight={0:1.0, 1:2.0} — 2x positive boost | `0.4805` | `-0.0108` | `0.5868` | `+0.0043` |
| **S3_cw_3x_positive** | class_weight={0:1.0, 1:3.0} — 3x positive boost | `0.4634` | `-0.0279` | `0.5896` | `+0.0071` |
| **S4_cw_1.5x_positive** | class_weight={0:1.0, 1:1.5} — 1.5x positive boost | `0.4927` | `+0.0014` | `0.5762` | `-0.0063` |
| **S5_oversample_fear_surprise_2x** | Oversample Fear & Surprise to 2x; class_weight=balanced | `0.4911` | `-0.0002` | `0.5758` | `-0.0067` |
| **S6_oversample_rare3_2x** | Oversample Fear, Disgust & Surprise to 2x; class_weight=balanced | `0.4906` | `-0.0007` | `0.5752` | `-0.0073` |

## Per-Label F1 Breakdown

| Strategy | anger | disgust | fear | joy | sadness | surprise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0_balanced_current** | 0.549 | 0.345 | 0.393 | 0.826 | 0.503 | 0.331 |
| **S1_no_weighting** | 0.506 | 0.277 | 0.524 | 0.821 | 0.469 | 0.342 |
| **S2_cw_2x_positive** | 0.538 | 0.228 | 0.512 | 0.824 | 0.497 | 0.284 |
| **S3_cw_3x_positive** | 0.533 | 0.258 | 0.426 | 0.812 | 0.506 | 0.245 |
| **S4_cw_1.5x_positive** | 0.531 | 0.245 | 0.557 | 0.823 | 0.482 | 0.319 |
| **S5_oversample_fear_surprise_2x** | 0.541 | 0.317 | 0.422 | 0.823 | 0.503 | 0.342 |
| **S6_oversample_rare3_2x** | 0.533 | 0.345 | 0.416 | 0.823 | 0.502 | 0.324 |

## Selection

**Selected Strategy:** `S0_balanced_current`  
**Val Macro-F1:** `0.4913` | **Val Micro-F1:** `0.5825`

> Regression rule: reject any strategy that drops any label >0.01 F1 below S0.

## Deliverables
- `scripts/experiments/phase8_class_imbalance.py`
- `results/ablations/phase8_class_imbalance/phase8_imbalance_report.json`
- `tests/test_phase8_class_imbalance.py`