# BETC — Phase 8b: Targeted Feature Engineering Report

**Date:** 2026-10-28  
**Baseline (S0):** Phase 5 Champion + Phase 7 τ* | Macro-F1=`0.4913` | Micro-F1=`0.5825`

---

## Strategy Summary

| Config | Description | Dims | Val Macro-F1 | Δ Macro | Val Micro-F1 | Δ Micro | Safe? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0_baseline** | Phase 5 Champ + Phase 7 τ* (reference) | 25,089 | `0.4912` | `-0.0001` | `0.5824` | `-0.0001` | ⚪ |
| **A_interrogative** | Base + Bengali interrogatives & punctuation (8 feats) | 25,097 | `0.4891` | `-0.0022` | `0.5745` | `-0.0080` | ❌ |
| **B_contrast** | Base + Discourse contrast connectives (10 feats) | 25,099 | `0.4891` | `-0.0022` | `0.5823` | `-0.0002` | ❌ |
| **C_intensifier_bigrams** ✅ | Base + Intensifier-emotion co-occ bigrams (12 feats) | 25,101 | `0.4921` | `+0.0008` | `0.5834` | `+0.0009` | ✅ |
| **D_combined_ABC** | Base + Groups A + B + C (30 feats) | 25,119 | `0.4902` | `-0.0011` | `0.5780` | `-0.0045` | ❌ |

## Per-Label F1 Breakdown

| Config | anger | disgust | fear | joy | sadness | surprise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0_baseline** | 0.549 | 0.345 | 0.393 | 0.826 | 0.503 | 0.331 |
| **A_interrogative** | 0.547 | 0.342 | 0.377 | 0.825 | 0.502 | 0.341 |
| **B_contrast** | 0.548 | 0.343 | 0.385 | 0.825 | 0.504 | 0.330 |
| **C_intensifier_bigrams** | 0.550 | 0.348 | 0.394 | 0.825 | 0.503 | 0.332 |
| **D_combined_ABC** | 0.547 | 0.348 | 0.374 | 0.825 | 0.501 | 0.345 |

## Selection

**Selected:** `C_intensifier_bigrams`  
**Val Macro-F1:** `0.4921` | **Val Micro-F1:** `0.5834`

> Selection rule: Macro-F1 must exceed 0.4913 AND no label may drop >0.01 F1 below S0.

## Deliverables
- `scripts/experiments/phase8b_targeted_features.py`
- `results/ablations/phase8b_targeted_features/phase8b_report.json`
- `tests/test_phase8b_targeted_features.py`