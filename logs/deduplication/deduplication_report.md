# Phase 5 — Deduplication Report
**Project:** BETC — Bangla Emotion TF-IDF Classifier Chain  
**Phase:** Phase 5 — Deduplication + Conflict Resolution  
**Date:** 2026-09-19  
**Status:** FULLY COMPLETE — DROP resolution applied; all 41,184 rows are unique texts  
**Input:** `Data/interim/harmonized_pre_dedup.csv` (41,994 rows)  
**Final Output:** `Data/interim/harmonized_deduplicated.csv` (41,184 rows)

> [!IMPORTANT]
> `Data/raw/` was not modified. `harmonized_pre_dedup.csv` (Phase 4 output) was not overwritten. All rows removed have provenance records in `logs/deduplication/duplicate_provenance.csv`.

> [!WARNING]
> **98 conflicting duplicate groups require user decision before Phase 6 (splitting) can begin.** Their 251 rows are retained in the output corpus unresolved. See Section 5 and `conflicting_duplicates.csv`.

---

## 1. Before/After Accounting

| Step | Count |
|---|---|
| Phase 4 input rows | **41,994** |
| − Identical-label duplicate rows removed | **559** |
| + Conflicting rows retained unresolved | **251** |
| = **Phase 5 output rows** | **41,435** |
| Reconciliation check | ✅ 41,994 − 559 = 41,435 |

---

## 2. Text Duplicate Summary

| Metric | Value |
|---|---|
| Total rows (Phase 4 input) | 41,994 |
| Unique comparison keys | 41,282 |
| Total duplicate text rows | **712** |
| Texts appearing more than once (unique texts with dups) | **504** |
| Rows in duplicate groups | 1,216 |
| Singleton rows (no duplicate) | 40,778 |

**Comparison key method:** Unicode NFC normalization + whitespace collapse only. Original text column is never modified.

---

## 3. Duplicate Group Classification

| Type | Count | Description |
|---|---|---|
| **Total groups** | **504** | All texts appearing ≥ 2 times |
| **A** — Same source, identical labels | 374 | Within-dataset, same label vector |
| **C** — Cross-source, identical labels | 32 | Texts in multiple datasets with matching labels |
| **E** — EmoNoBa cross-split, identical labels | 12 | Same text across Train/Val/Test (same labels) |
| **B** — Same source, conflicting labels | 57 | Within-dataset, different label vectors |
| **D** — Cross-source, conflicting labels | 41 | Cross-dataset, different label vectors |
| **Identical-label total (A+C+E)** | **406** | → Resolved: canonical selected |
| **Conflicting total (B+D)** | **98** | → Unresolved: require user decision |

---

## 4. Cross-Dataset Duplicate Groups

| Pair | Groups | Notes |
|---|---|---|
| EmoNoBa ↔ UBMEC | 8 | Mix of identical and conflicting |
| EmoNoBa ↔ MONOVAB | 9 | Mix of identical and conflicting |
| UBMEC ↔ MONOVAB | 41 | Dominant cross-source overlap |
| All three datasets | 2 | Same text in EmoNoBa + UBMEC + MONOVAB |
| **Total cross-source** | **60** | |

**Note on literature comparison:** The Phase 1 audit estimated 65 cross-dataset duplicate texts; we observe 60 cross-source duplicate *groups* in the Phase 4 harmonized corpus. The 5-group difference is explained by the Phase 4 exclusions (4,405 rows removed — Love-only and Contempt-only rows that included some of the earlier cross-dataset duplicates).

The literature-reported figure of 665 cross-dataset duplicates ([VERIFY]/[LIT]) is **not comparable** to our count — it refers to the raw 46,035-entry combined pool before any harmonization exclusions, and likely uses a different deduplication method. Our measured figure of **559 identical-label removals** is this project's own independently measured deduplication outcome.

---

## 5. EmoNoBa Cross-Split Duplicates

**12 duplicate groups** where the same text appears across EmoNoBa's own Train / Val / Test splits with identical labels. These represent original EmoNoBa pre-split leakage detected in Phase 1.

| Condition | Result |
|---|---|
| Groups found | 12 |
| Label conflict within these groups | 0 (all identical labels) |
| Action taken | Canonical selected (per deterministic rule); duplicates collapsed |
| Effect | One canonical copy retained; original split info preserved in provenance |

These 12 groups contained 27 rows; 15 were collapsed. The canonical copy preserves the Train-split assignment (Train > Val > Test priority).

---

## 6. Canonical Record Selection Rule

For all 406 identical-label duplicate groups, the canonical record was selected deterministically:

| Priority | Rule |
|---|---|
| 1 | Prefer row with most populated metadata (`domain` + `topic` non-empty → EmoNoBa rows preferred) |
| 2 | Dataset priority: **EmoNoBa > UBMEC > MONOVAB** |
| 3 | Split priority: **Train > Val > Test > NONE** |
| 4 | Smallest `source_row_id` (numeric, ascending) |

**No randomness was used.** The rule is deterministic and fully reproducible.

---

## 7. Unresolved Conflicting Groups — 98 Groups REQUIRE USER DECISION

**98 text groups** have the same text appearing with different six-label vectors in their duplicate rows. These were **not resolved automatically** per the Phase 5 specification.

### Breakdown by Type

| Type | Count | Description |
|---|---|---|
| B — Within-dataset, conflicting | 57 | Same text annotated differently in same dataset |
| D — Cross-dataset, conflicting | 41 | Same text annotated differently across datasets |

### What Is in the Output

All 251 rows from conflicting groups are **retained as-is** in `harmonized_deduplicated.csv`. This means:
- These 251 rows include duplicate texts with different labels
- They are flagged in the provenance log with `conflict_status=CONFLICTING` and `action=RETAINED_UNRESOLVED`
- They are separately logged in `conflicting_duplicates.csv` with all label vectors

### Example Conflict Types Observed

**Type B (within-dataset):** A UBMEC text appearing twice with `joy` in one row and `surprise` in another — genuine annotator disagreement on an ambiguous short text (e.g., single-word expressions like "অসাধারণ" meaning "extraordinary").

**Type D (cross-dataset):** A MONOVAB text also in UBMEC where MONOVAB labelled it `anger=1, disgust=1` and UBMEC labelled it `anger=1` only — different annotation schemes capturing different aspects.

### Resolution Strategies (Awaiting User Approval)

| Option | Description | Rows Removed | Risk |
|---|---|---|---|
| **DROP** | Remove all rows for conflicting texts (safest) | 251 rows removed | Lose 251 examples |
| **UNION** | Merge label vectors with logical OR | 153 rows removed (one canonical per text) | Over-assigns emotions |
| **FIRST** | Keep first occurrence (lowest source_row_id per priority) | 153 rows removed | Arbitrary selection |
| **DATASET-PRIORITY** | Keep the EmoNoBa annotation if present, else UBMEC, else MONOVAB | 153 rows removed | EmoNoBa bias; discards others' annotations |

> [!IMPORTANT]
> **REQUIRES USER DECISION:** Which conflict resolution strategy to apply to the 98 unresolved conflicting groups (251 rows)?

---

## 8. Label Statistics Post-Deduplication

| Target Label | Positive | Prevalence | Change from Phase 4 |
|---|---|---|---|
| `anger` | 11,368 | 27.44% | −89 (−0.78%) |
| `disgust` | 4,104 | 9.90% | −42 (−1.01%) |
| `fear` | 1,802 | 4.35% | −16 (−0.88%) |
| `joy` | 14,853 | 35.85% | −327 (−2.15%) |
| `sadness` | 9,496 | 22.92% | −56 (−0.59%) |
| `surprise` | 2,521 | 6.08% | −31 (−1.21%) |

| Cardinality | Rows | % |
|---|---|---|
| 1 active label | 38,844 | 93.75% |
| 2 active labels | 2,477 | 5.98% |
| 3 active labels | 110 | 0.27% |
| 4 active labels | 4 | <0.01% |
| **Mean** | **1.0654** | |

---

## 9. Source Counts Post-Deduplication

| Source | Phase 4 Rows | Phase 5 Rows | Removed |
|---|---|---|---|
| EmoNoBa | 20,462 | 20,365 | 97 |
| UBMEC | 13,436 | 13,078 | 358 |
| MONOVAB | 8,096 | 7,992 | 104 |
| **Total** | **41,994** | **41,435** | **559** |

---

## 10. Validation Checks — All Passed ✅

| Check | Result |
|---|---|
| All 6 target label columns contain only 0/1 | ✅ PASS |
| No null text values | ✅ PASS |
| No `love` or `contempt` columns present as targets | ✅ PASS |
| `Data/raw/` unchanged | ✅ PASS |
| Phase 4 file (`harmonized_pre_dedup.csv`) unchanged (41,994 rows) | ✅ PASS |
| All removed rows have provenance records | ✅ PASS (559 COLLAPSED_INTO_CANONICAL records) |
| No duplicate texts remain among resolved groups | ✅ PASS (0 remaining resolved dups) |
| Reconciliation: 41,994 − 559 = 41,435 | ✅ PASS |

---

## 11. Output File Index

| File | Location | Rows / Size | Description |
|---|---|---|---|
| `harmonized_deduplicated.csv` | `Data/interim/` | **41,435 rows** | Phase 5 output — deduplicated corpus |
| `duplicate_groups.csv` | `logs/deduplication/` | 504 groups | Summary of every duplicate group |
| `duplicate_provenance.csv` | `logs/deduplication/` | 41,994 rows | Full provenance for every Phase 4 row |
| `conflicting_duplicates.csv` | `logs/deduplication/` | 98 groups | Unresolved conflict groups — all label vectors |
| `deduplication_validation.json` | `logs/deduplication/` | — | Validation checks + statistics JSON |

---

## 12. Conflict Resolution — DROP Strategy (Approved 2026-09-19)

**Decision:** DROP all rows belonging to the 98 conflicting duplicate groups.

### Final Before/After Accounting

| Step | Count |
|---|---|
| Phase 4 input rows | 41,994 |
| − Identical-label duplicate rows removed | 559 |
| − Conflicting duplicate rows removed (DROP) | **251** |
| = **Final deduplicated corpus** | **41,184** |
| Reconciliation check | ✅ 41,994 − 559 − 251 = 41,184 |

### Label Statistics — Final Corpus (41,184 rows)

| Target Label | Positive | Prevalence | EmoNoBa | UBMEC | MONOVAB |
|---|---|---|---|---|---|
| `anger` | 11,279 | 27.39% | — | — | — |
| `disgust` | 4,064 | 9.87% | 0* | — | — |
| `fear` | 1,789 | 4.34% | — | — | — |
| `joy` | 14,811 | 35.96% | — | — | — |
| `sadness` | 9,432 | 22.90% | — | — | — |
| `surprise` | 2,499 | 6.07% | — | — | — |

*EmoNoBa `disgust=0` is the Strategy D operational assumption (F5-eligible rows)

### Source Counts — Final Corpus

| Source | Rows |
|---|---|
| EmoNoBa | 20,352 |
| UBMEC | 12,896 |
| MONOVAB | 7,936 |
| **Total** | **41,184** |

### Cardinality Distribution

| Active Labels | Rows |
|---|---|
| 1 | 38,611 |
| 2 | 2,460 |
| 3 | 109 |
| 4 | 4 |
| **Mean** | **1.0653** |

### Validation — All 9 Checks Passed ✅

| Check | Result |
|---|---|
| Arithmetic (41,435 − 251 = 41,184) | ✅ PASS |
| No duplicate texts remain | ✅ PASS (41,184 unique texts = 41,184 rows) |
| All 6 target label columns binary only | ✅ PASS |
| No `love` or `contempt` target columns | ✅ PASS |
| `emonoba_disgust_assumption` preserved on all EmoNoBa rows | ✅ PASS (20,352 / 20,352) |
| `f5_eligible` preserved on all EmoNoBa rows | ✅ PASS (20,352 / 20,352) |
| `Data/raw/` unchanged | ✅ PASS |
| `harmonized_pre_dedup.csv` (Phase 4) unchanged (41,994 rows) | ✅ PASS |
| `conflicting_duplicates.csv` preserved (98 groups) | ✅ PASS |

### Output Files (Final)

| File | Location | Rows | Description |
|---|---|---|---|
| `harmonized_deduplicated.csv` | `Data/interim/` | **41,184** | **Final modeling corpus** |
| `conflict_resolution_log.csv` | `logs/deduplication/` | 251 rows | DROP evidence — one row per dropped conflicting record |
| `conflicting_duplicates.csv` | `logs/deduplication/` | 98 groups | Preserved unchanged — evidence of conflicts |
| `deduplication_validation_final.json` | `logs/deduplication/` | — | All validation checks JSON |

**Phase 5b (Splitting) is now unblocked.** The final modeling corpus is ready.
