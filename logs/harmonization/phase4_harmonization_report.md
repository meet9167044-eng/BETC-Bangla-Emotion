# Phase 4 — Dataset Harmonization Report
**Project:** BETC — Bangla Emotion TF-IDF Classifier Chain  
**Phase:** Phase 4 — Dataset Harmonization  
**Date:** 2026-09-19  
**Status:** COMPLETE — pre-deduplication harmonized corpus created  
**Preceding phase:** Phase 3 (design, approved 2026-09-19)  
**Next phase:** Phase 5 — Deduplication (requires explicit user approval)

> [!IMPORTANT]
> **Data/raw/ was not modified.** All raw files remain byte-for-byte identical. Every output is written to `Data/interim/` and `logs/harmonization/`.

---

## 1. Approved Decisions Applied

| Decision | Approved Strategy | Rule |
|---|---|---|
| EmoNoBa Disgust | Strategy D: `disgust=0` as documented operational assumption; F5 mandatory | DISGUST-D |
| EmoNoBa Love-only rows | LOVE-1: exclude 2,277 rows | LOVE-1 |
| MONOVAB Contempt-only rows | CONTEMPT-1: exclude 2,128 rows | CONTEMPT-1 |
| MONOVAB `enjoyment` | FROZEN → `joy` (semantic harmonization) | ENJOYMENT-JOY |
| UBMEC label conversion | FROZEN → one-hot six-binary | UBMEC-ONE-HOT |
| Target taxonomy | `anger, disgust, fear, joy, sadness, surprise` | FROZEN |

---

## 2. Row Counts

| Source | Raw Rows | Excluded | Included |
|---|---|---|---|
| EmoNoBa (Train+Val+Test) | 22,739 | 2,277 (LOVE-1) | **20,462** |
| UBMEC | 13,436 | 0 | **13,436** |
| MONOVAB | 10,224 | 2,128 (CONTEMPT-1) | **8,096** |
| **Total** | **46,399** | **4,405** | **41,994** |

---

## 3. Exclusion Breakdown

| Exclusion Reason | Dataset | Count | Rule |
|---|---|---|---|
| Love is the only active native label | EmoNoBa | 2,277 | LOVE-1 |
| Contempt is the only active native label | MONOVAB | 2,128 | CONTEMPT-1 |
| Unexpected UBMEC class (not in 6 targets) | UBMEC | 0 | UBMEC-UNEXPECTED-CLASS |
| **Total excluded** | | **4,405** | |

---

## 4. Label Transformations Applied

| Dataset | Native Label | → | Target Label | Rule | Rows Affected |
|---|---|---|---|---|---|
| EmoNoBa | `Joy` | → | `joy` | DIRECT | 20,462 |
| EmoNoBa | `Anger` | → | `anger` | DIRECT | 20,462 |
| EmoNoBa | `Sadness` | → | `sadness` | DIRECT | 20,462 |
| EmoNoBa | `Surprise` | → | `surprise` | DIRECT | 20,462 |
| EmoNoBa | `Fear` | → | `fear` | DIRECT | 20,462 |
| EmoNoBa | `Love` | → | **DROPPED** | FROZEN exclusion | 20,462 |
| EmoNoBa | *(absent)* | → | `disgust=0` (**assumption**) | DISGUST-D | 20,462 |
| UBMEC | `classes` categorical | → | one-hot 6-binary | UBMEC-ONE-HOT | 13,436 |
| MONOVAB | `anger` | → | `anger` | DIRECT | 8,096 |
| MONOVAB | `disgust` | → | `disgust` | DIRECT | 8,096 |
| MONOVAB | `enjoyment` | → | `joy` | ENJOYMENT-JOY FROZEN | 8,096 |
| MONOVAB | `fear` | → | `fear` | DIRECT | 8,096 |
| MONOVAB | `sadness` | → | `sadness` | DIRECT | 8,096 |
| MONOVAB | `surprise` | → | `surprise` | DIRECT | 8,096 |
| MONOVAB | `contempt` | → | **DROPPED** | FROZEN exclusion | 8,096 |

---

## 5. Target Label Statistics (Pre-Deduplication)

| Target Label | Total Positive | Prevalence | EmoNoBa | UBMEC | MONOVAB | Notes |
|---|---|---|---|---|---|---|
| `anger` | 11,457 | 27.28% | 4,478 | 2,480 | 4,499 | Full cross-source coverage |
| `disgust` | 4,146 | 9.87% | **0** *(assumed 0)* | 2,079 | 2,067 | EmoNoBa positives = 0 by Strategy D assumption; true annotated positives = 4,146 |
| `fear` | 1,818 | 4.33% | 401 | 1,361 | 56 | Severely imbalanced; UBMEC dominant |
| `joy` | 15,180 | 36.15% | 10,112 | 3,467 | 1,601 *(via enjoyment)* | Highest prevalence |
| `sadness` | 9,552 | 22.75% | 5,681 | 2,683 | 1,188 | Good cross-source coverage |
| `surprise` | 2,552 | 6.08% | 1,086 | 1,366 | 100 | Low in MONOVAB |

> [!WARNING]
> **Disgust column contains label noise for EmoNoBa rows.** All 20,462 EmoNoBa rows have `disgust=0` by the Strategy D operational assumption, NOT because annotators verified absence of disgust. These rows are flagged `emonoba_disgust_assumption=True` and `f5_eligible=True` in the corpus to enable Experiment F5 sensitivity analysis.

---

## 6. Multi-Label Cardinality Distribution

| Active Labels per Example | Row Count | % |
|---|---|---|
| 1 | 39,401 | 93.82% |
| 2 | 2,479 | 5.90% |
| 3 | 110 | 0.26% |
| 4 | 4 | <0.01% |
| **Mean** | **1.0646** | |

---

## 7. Validation Checks — All Passed ✅

| Check | Result |
|---|---|
| All target label columns contain only 0/1 | ✅ PASS (all 6 columns) |
| No null text values | ✅ PASS (0 null texts) |
| Source datasets are only EmoNoBa / UBMEC / MONOVAB | ✅ PASS |
| All EmoNoBa rows have `emonoba_disgust_assumption=True` | ✅ PASS (20,462 / 20,462) |
| No UBMEC or MONOVAB rows have `emonoba_disgust_assumption=True` | ✅ PASS (0 false positives) |
| All-zero target vector rows | ✅ 0 rows (0.0%) |
| UBMEC one-hot validity (exactly one 1 per row) | ✅ PASS (0 violations) |
| Love-only rows in included corpus | ✅ PASS (0 rows) |

---

## 8. Corpus Schema

The harmonized corpus (`Data/interim/harmonized_pre_dedup.csv`) has the following columns:

| Column | Type | Description |
|---|---|---|
| `source_dataset` | str | `EmoNoBa` / `UBMEC` / `MONOVAB` |
| `source_split` | str | `Train` / `Val` / `Test` (EmoNoBa); `NONE` (others) |
| `source_row_id` | int | Original identifier from source dataset |
| `text` | str | Bangla comment/text |
| `anger` | int (0/1) | Target label |
| `disgust` | int (0/1) | Target label (EmoNoBa rows: 0 by assumption) |
| `fear` | int (0/1) | Target label |
| `joy` | int (0/1) | Target label (MONOVAB: from `enjoyment`) |
| `sadness` | int (0/1) | Target label |
| `surprise` | int (0/1) | Target label |
| `emonoba_disgust_assumption` | bool | `True` for all EmoNoBa included rows; `False` otherwise |
| `f5_eligible` | bool | `True` for EmoNoBa rows; used to flag rows for F5 sensitivity analysis |
| `original_love` | int / NaN | EmoNoBa original Love value (provenance) |
| `original_class` | str / NaN | UBMEC original categorical class string (provenance) |
| `original_contempt` | int / NaN | MONOVAB original contempt value (provenance) |
| `original_enjoyment` | int / NaN | MONOVAB original enjoyment value before →joy mapping (provenance) |
| `domain` | str / NaN | EmoNoBa Domain field (YouTube/Facebook/Twitter) |
| `topic` | str / NaN | EmoNoBa Topic field |

---

## 9. Output Artifact Index

| File | Location | Size / Rows | Description |
|---|---|---|---|
| `harmonized_pre_dedup.csv` | `Data/interim/` | 41,994 rows | **Main harmonized corpus** (pre-dedup) |
| `harmonization_manifest.csv` | `logs/harmonization/` | 46,399 rows (all source rows) | Full row-level decision record |
| `exclusion_log.csv` | `logs/harmonization/` | 4,405 rows | All excluded rows with reasons |
| `transformation_log.csv` | `logs/harmonization/` | 9 entries | All label transformation rules applied |
| `harmonization_validation.json` | `logs/harmonization/` | — | Validation checks + statistics JSON |

---

## 10. Important Notes for Downstream Phases

1. **This corpus is pre-deduplication.** Within-dataset and cross-dataset duplicates have NOT been removed. Do not use this corpus for training until Phase 5 (deduplication) is complete.

2. **EmoNoBa Disgust is an assumption, not an annotation.** The column `emonoba_disgust_assumption=True` and `f5_eligible=True` flags must be preserved through all downstream phases. Experiment F5 requires being able to select/exclude these rows from Disgust-specific evaluation.

3. **UBMEC label zeros mean "not the observed class" in a single-label scheme**, not "annotators confirmed absence." Document this carefully when reporting multi-label evaluation metrics.

4. **MONOVAB enjoyment→joy is a semantic harmonization mapping**, not an exact label identity. The `original_enjoyment` column preserves the original annotation for traceability.

5. **Phase 5 (Deduplication) has NOT started** and requires explicit user approval before execution.
