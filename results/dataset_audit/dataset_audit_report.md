# Raw Dataset Audit Report (Phase 1)
**Project:** BETC — Bangla Emotion TF-IDF Classifier Chain  
**Date:** 2026-09-19  
**Status:** PHASE 1 AUDIT COMPLETE — BLOCKING REVIEW REQUIRED BEFORE HARMONIZATION  
**Audit Directory:** `results/dataset_audit/`  
**Governing Documents:** `DOCS/DATASET_SPEC.md`, `DOCS/IMPLEMENTATION_PLAN.md`, `DOCS/RESEARCH_RULES.md`

---

## Executive Summary

As required by **Phase 1** of `IMPLEMENTATION_PLAN.md` and `DATASET_SPEC.md` Section 6, an independent, empirical audit was conducted on all actual raw dataset files present in the repository:
1. **EmoNoBa** (`Data/raw/Emonoba/{Train.csv, Val.csv, Test.csv}`)
2. **UBMEC** (`Data/raw/UBMEC Corpus_Sakib(updated).xlsx`)
3. **MONOVAB** (`Data/raw/MONOVAB (1).csv`)

**Key Findings:**
- **Grand Total Raw Rows:** **46,399 rows** across all raw files.
- **EmoNoBa:** 22,739 rows across 3 split files. Multi-label (1 to 4 active labels). Native labels include `Love`, completely lack `Disgust`. Contains 2,277 rows where `Love` is the sole active emotion.
- **UBMEC:** 13,436 rows in an Excel spreadsheet (`.xlsx`). Strictly single-label across Ekman's 6 basic emotions. Discrepancy with literature: 13,436 rows measured vs. 13,072 cited in literature (+364 rows).
- **MONOVAB:** 10,224 rows in CSV. Multi-label (1 to 4 active labels out of 7 source labels). Uses `enjoyment` for Joy, and contains non-target `contempt` (2,960 instances). Crucially, **2,128 rows** have `contempt` as their only active emotion; discarding `contempt` leaves these rows with zero active target emotions. Discrepancy with literature: 10,224 rows measured vs. 10,244 cited in literature (-20 rows).
- **Cross-dataset overlap:** 65 duplicate texts shared across datasets (10 between EmoNoBa and UBMEC; 12 between EmoNoBa and MONOVAB; 45 between UBMEC and MONOVAB; 2 texts appear in all three).
- **No data alteration:** No rows have been deleted, no text has been modified, no labels have been harmonized, and no model has been trained.

---

## 1. Dataset-Wise Inventory & Structural Specifications

| Dataset Attribute | EmoNoBa | UBMEC | MONOVAB |
| :--- | :--- | :--- | :--- |
| **Actual File Location** | `Data/raw/Emonoba/` | `Data/raw/UBMEC Corpus_Sakib(updated).xlsx` | `Data/raw/MONOVAB (1).csv` |
| **Documented Target Path** | `data/raw/emonoba/` | `data/raw/ubmec/` | `data/raw/monovab/` |
| **File Format** | CSV (`.csv`) | Excel (`.xlsx`), Sheet: `UBMEC` | CSV (`.csv`) |
| **File Size (bytes)** | Train: 3,649,059<br>Val: 418,988<br>Test: 450,627<br>Total: 4,518,674 | 1,116,902 bytes | 1,938,912 bytes |
| **Encoding** | UTF-8 | OpenPyXL / XML | UTF-8 |
| **Total Rows (Measured)** | **22,739** (Train: 18,420; Val: 2,047; Test: 2,272) | **13,436** | **10,224** |
| **Literature Reported Size [VERIFY]** | 22,698 to 22,739 | 13,072 | 10,244 |
| **Discrepancy vs Literature** | Matches upper bound (22,739) exactly | **+364 rows** (exact duplicate count = 364!) | **-20 rows** (10,224 vs 10,244) |
| **Number of Columns** | 11 | 2 | 9 |
| **Text Column** | `Data` | `text` | `comment` |
| **Label Columns** | `Love`, `Joy`, `Surprise`, `Anger`, `Sadness`, `Fear` | `classes` | `anger`, `contempt`, `disgust`, `enjoyment`, `fear`, `sadness`, `surprise` |
| **Metadata Columns** | `ID`, `Topic`, `Domain`, `is_admin` | None | `Unnamed: 0` (index) |
| **Classification Type** | **Multi-label** | **Single-label** | **Multi-label** |

---

## 2. Dataset-Wise Schemas & Data Types

### 2.1 EmoNoBa Schema
| Column Name | Data Type | Null Count | Null % | Description / Content |
| :--- | :--- | :--- | :--- | :--- |
| `ID` | `int64` | 0 | 0.0% | Comment identifier |
| `Data` | `object` (string) | 0 | 0.0% | Raw Bangla text comment |
| `Love` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Non-target emotion |
| `Joy` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |
| `Surprise` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |
| `Anger` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |
| `Sadness` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |
| `Fear` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |
| `Topic` | `object` (string) | 0 | 0.0% | 12 categorical topics (Personal, Politics, etc.) |
| `Domain` | `object` (string) | 0 | 0.0% | 3 social media sources (`Youtube`, `Facebook`, `Twitter`) |
| `is_admin` | `bool` | 0 | 0.0% | Boolean indicator (`false`: 21,516; `true`: 1,223) |

### 2.2 UBMEC Schema
| Column Name | Data Type | Null Count | Null % | Description / Content |
| :--- | :--- | :--- | :--- | :--- |
| `text` | `object` (string) | 0 | 0.0% | Raw Bangla text comment (Row 12039 is int `1`) |
| `classes` | `object` (string) | 0 | 0.0% | Categorical string label (6 Ekman emotions) |

### 2.3 MONOVAB Schema
| Column Name | Data Type | Null Count | Null % | Description / Content |
| :--- | :--- | :--- | :--- | :--- |
| `Unnamed: 0` | `int64` | 0 | 0.0% | Integer row index (0 to 10223) |
| `comment` | `object` (string) | 0 | 0.0% | Raw Bangla text comment |
| `anger` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |
| `contempt` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Non-target emotion |
| `disgust` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |
| `enjoyment` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion (`Joy` equivalent) |
| `fear` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |
| `sadness` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |
| `surprise` | `int64` | 0 | 0.0% | Binary label (0 or 1) — Target emotion |

---

## 3. Label Inventories & Distributions

### 3.1 EmoNoBa Label Distribution (Split-wise and Total)
Native labels: 6 labels (`Love`, `Joy`, `Surprise`, `Anger`, `Sadness`, `Fear`). Note: Completely lacks `Disgust`.

| Native Label | Train (18,420) | Val (2,047) | Test (2,272) | Total Positive | Total Rows | Prevalence % | Status in BETC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Joy** | 8,314 | 942 | 856 | **10,112** | 22,739 | 44.47% | Target Label 4 |
| **Sadness** | 4,569 | 505 | 607 | **5,681** | 22,739 | 24.98% | Target Label 5 |
| **Love** | 3,786 | 414 | 388 | **4,588** | 22,739 | 20.18% | **Non-Target (Excluded)** |
| **Anger** | 3,542 | 388 | 548 | **4,478** | 22,739 | 19.69% | Target Label 1 |
| **Surprise** | 848 | 91 | 147 | **1,086** | 22,739 | 4.78% | Target Label 6 |
| **Fear** | 279 | 32 | 90 | **401** | 22,739 | 1.76% | Target Label 3 |
| *Disgust* | *N/A* | *N/A* | *N/A* | *0 (unannotated)* | 22,739 | *0.00%* | Target Label 2 (Assumed 0) |

**Active Labels Per Example (EmoNoBa):**
- 1 label: 19,214 rows (84.50%)
- 2 labels: 3,445 rows (15.15%)
- 3 labels: 78 rows (0.34%)
- 4 labels: 2 rows (0.01%)
- Mean active labels: **1.1586**
- Minimum active labels: 1; Maximum active labels: 4.
- **Impact of dropping `Love`:** There are **2,277 rows** where `Love` is the ONLY active emotion. Dropping `Love` leaves these 2,277 rows with 0 active target emotions ([0, 0, 0, 0, 0, 0]).

### 3.2 UBMEC Label Distribution
Native labels: 6 categorical strings (`joy`, `sadness`, `anger`, `disgust`, `surprise`, `fear`).

| Native Class | Total Count | Total Rows | Prevalence % | Status in BETC |
| :--- | :--- | :--- | :--- | :--- |
| **joy** | 3,467 | 13,436 | 25.80% | Target Label 4 |
| **sadness** | 2,683 | 13,436 | 19.97% | Target Label 5 |
| **anger** | 2,480 | 13,436 | 18.46% | Target Label 1 |
| **disgust** | 2,079 | 13,436 | 15.47% | Target Label 2 |
| **surprise** | 1,366 | 13,436 | 10.17% | Target Label 6 |
| **fear** | 1,361 | 13,436 | 10.13% | Target Label 3 |

**Active Labels Per Example (UBMEC):**
- Exactly **1 label** per example (100.0%). Strictly single-label.

### 3.3 MONOVAB Label Distribution
Native labels: 7 binary columns (`anger`, `contempt`, `disgust`, `enjoyment`, `fear`, `sadness`, `surprise`).

| Native Label | Positive (1) | Negative (0) | Total Rows | Prevalence % | Status in BETC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **anger** | 4,499 | 5,725 | 10,224 | 44.00% | Target Label 1 |
| **contempt** | 2,960 | 7,264 | 10,224 | 28.95% | **Non-Target (Excluded)** |
| **disgust** | 2,067 | 8,157 | 10,224 | 20.22% | Target Label 2 |
| **enjoyment** | 1,601 | 8,623 | 10,224 | 15.66% | Maps to Target Label 4 (`Joy`) |
| **sadness** | 1,188 | 9,036 | 10,224 | 11.62% | Target Label 5 |
| **surprise** | 100 | 10,124 | 10,224 | 0.98% | Target Label 6 (Severe imbalance) |
| **fear** | 56 | 10,168 | 10,224 | 0.55% | Target Label 3 (Severe imbalance) |

**Active Labels Per Example (MONOVAB):**
- **Across all 7 native labels:**
  - 1 label: 8,202 rows (80.22%)
  - 2 labels: 1,804 rows (17.65%)
  - 3 labels: 211 rows (2.06%)
  - 4 labels: 7 rows (0.07%)
  - Mean active labels: **1.2198**
- **Across the 6 target labels (excluding `contempt`):**
  - **0 labels: 2,128 rows (20.81%)**
  - 1 label: 6,744 rows (65.96%)
  - 2 labels: 1,291 rows (12.63%)
  - 3 labels: 59 rows (0.58%)
  - 4 labels: 2 rows (0.02%)
  - Mean active labels: **0.9303**
  - **Critical Finding:** There are **2,128 rows** where `contempt` was the ONLY positive label. When `contempt` is discarded, these rows become completely negative ([0,0,0,0,0,0]).

---

## 4. Duplicate Statistics

| Dataset | Total Rows | Exact Full-Row Duplicates | Unique Raw Texts | Duplicate Raw Texts | Duplicate Normalized Texts |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EmoNoBa** | 22,739 | 41 | 22,623 | 116 | 125 |
| **UBMEC** | 13,436 | 364 | 13,007 | 429 | 452 |
| **MONOVAB** | 10,224 | 0 (92 excl. index) | 10,096 | 128 | 150 |
| **Combined** | **46,399** | **405** (497 excl. idx) | **45,726** | **673** | **727** |

### Cross-Dataset Overlap:
- EmoNoBa ∩ UBMEC: **10 texts**
- EmoNoBa ∩ MONOVAB: **12 texts**
- UBMEC ∩ MONOVAB: **45 texts**
- Shared across all three: **2 texts**
- Total unique texts across union: **45,645**
- Net cross-dataset duplicate texts: **65**

---

## 5. Missing-Value Statistics

| Dataset | Text Missing / Null | Label Missing / Null | Metadata Missing / Null | Overall Missing |
| :--- | :--- | :--- | :--- | :--- |
| **EmoNoBa** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 |
| **UBMEC** | 0 (0.0%) | 0 (0.0%) | N/A | 0 |
| **MONOVAB** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 |

**Conclusion:** There are **zero NaN or null entries** in any column across all three raw datasets. Every entry is populated.

---

## 6. Text Quality and Language Integrity Audit

| Metric | EmoNoBa (`Data`) | UBMEC (`text`) | MONOVAB (`comment`) |
| :--- | :--- | :--- | :--- |
| **Character Length (Min / Max / Mean / Median)** | 9 / 1,249 / 59.72 / 43.0 | 1 / 3,648 / 109.73 / 86.0 | 2 / 2,741 / 63.18 / 39.0 |
| **Word Count (Min / Max / Mean / Median)** | 2 / 204 / 10.55 / 8.0 | 1 / 552 / 18.79 / 15.0 | 1 / 416 / 10.88 / 7.0 |
| **Pure Bangla (No Latin)** | 22,739 (100.0%) | 12,780 (95.12%) | 10,199 (99.76%) |
| **Mixed Bangla & Latin** | 0 (0.0%) | 655 (4.88%) | 21 (0.21%) |
| **Latin Only (No Bangla)** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| **No Letters (Punctuation / Digits Only)** | 0 (0.0%) | **1 row** (0.01%) | **4 rows** (0.04%) |
| **Contains URLs** | 0 | 48 | 0 |
| **Contains @Mentions** | 3 | 19 | 0 |

---

## 7. Data-Quality Problems Identified

1. **Unusable / Degenerate Texts (Non-Alphabetic):**
   - **UBMEC Row 12039:** Text is an integer `1` with label `sadness`. No words or emotion-bearing characters.
   - **MONOVAB:** 4 rows contain only punctuation marks with no words:
     - Row 5477: `????!!!!!` (labeled `contempt=1`)
     - Row 8180: `!!!!` (labeled `anger=1`)
     - Row 8988: `!!` (labeled `anger=1`)
     - Row 9656: `,,,,` (labeled `anger=1`)
2. **Short / Low-Information Texts:**
   - Single-character texts in UBMEC (e.g. `.` or single words).
3. **Duplicate Texts with Conflicting Annotations:**
   - **UBMEC:** 58 unique texts appear multiple times with **conflicting emotion labels** (e.g., `"অসাধারণ"` appears 4 times, twice labeled `joy` and twice labeled `surprise`). Text at row 8260 appears 14 times with three different labels (`anger`, `sadness`, `disgust`).
   - **MONOVAB:** 33 unique texts appear with conflicting multi-label combinations across identical comments.
   - **EmoNoBa:** 8 unique texts have conflicting annotations across duplicates.
4. **Cross-Split Leakage in Published EmoNoBa Split:**
   - In EmoNoBa's original split, 6 texts appear in both Train and Val, and 8 texts appear in both Train and Test. (This validates our project decision to create our own stratified split after deduplication).
5. **English/Code-Mixed Tokens in UBMEC:**
   - 655 rows in UBMEC contain English/Latin characters mixed with Bangla (e.g., "Love you বাকের ভাই", "s t v", "NID CARD", Facebook usernames).

---

## 8. Contradictions Against Repository Documentation

In accordance with the prompt instruction (*"If the actual dataset structure contradicts the documentation, STOP and report the contradiction instead of silently changing the methodology"*), the following contradictions between actual files and documentation are reported:

1. **Raw Directory and File Placement (`DATASET_SPEC.md` Section 7):**
   - **Documented:**
     - `data/raw/emonoba/`
     - `data/raw/ubmec/`
     - `data/raw/monovab/`
   - **Actual:**
     - `Data/raw/Emonoba/` (Capital `D` and `E`)
     - `Data/raw/UBMEC Corpus_Sakib(updated).xlsx` (File directly in `Data/raw/`, not in a subfolder `ubmec/`)
     - `Data/raw/MONOVAB (1).csv` (File directly in `Data/raw/`, not in a subfolder `monovab/`, with download suffix `(1)`)
2. **UBMEC File Format and Row Count (`DATASET_SPEC.md` Section 2.2):**
   - **Documented:** CSV format expected; reported size ~13,072.
   - **Actual:** Excel file (`.xlsx`), sheet `UBMEC`; contains **13,436 rows** (difference of +364 rows). Interestingly, UBMEC has exactly **364 exact duplicate rows** ($13,436 - 364 = 13,072$ unique rows), explaining the exact origin of the discrepancy!
3. **MONOVAB Column Name and Row Count (`DATASET_SPEC.md` Section 2.3):**
   - **Documented:** 10,244 entries; target labels.
   - **Actual:** 10,224 rows (-20 rows). Joy is named `enjoyment` (not `joy`). Contains `Unnamed: 0` index.
4. **Zero-Label Examples Resulting from Non-Target Label Exclusions:**
   - When non-target labels `Love` (EmoNoBa) and `contempt` (MONOVAB) are dropped, **4,405 rows** (2,277 in EmoNoBa + 2,128 in MONOVAB) become **all-zero label vectors** $[0,0,0,0,0,0]$.

---

## 9. Machine-Readable Audit Artifacts Created

The following machine-readable audit files have been generated and saved under `results/dataset_audit/`:

1. `results/dataset_audit/raw_dataset_summary.json` — Comprehensive summary of all row counts, column counts, text columns, and cross-dataset overlap counts.
2. `results/dataset_audit/emonoba_audit.json` — Complete EmoNoBa split-wise and combined audit (shapes, dtypes, text stats, distributions, domains, topics).
3. `results/dataset_audit/ubmec_audit.json` — Complete UBMEC audit (shapes, sheet name, classes, dtypes, text stats, distributions).
4. `results/dataset_audit/monovab_audit.json` — Complete MONOVAB audit (shapes, dtypes, text stats, 7-label vs 6-label distributions and cardinalities).
5. `results/dataset_audit/cross_dataset_overlap.json` — Exact pairwise and 3-way text intersection counts.
6. `results/dataset_audit/data_quality_anomalies.json` — Precise details and row indices of degenerate texts, code-mixed samples, conflicting-label duplicates, and cross-split leakages.
7. `results/dataset_audit/emonoba_label_distribution.csv` — CSV table of EmoNoBa label counts and prevalence across Train, Val, Test, and Total.
8. `results/dataset_audit/ubmec_label_distribution.csv` — CSV table of UBMEC categorical class counts and prevalence.
9. `results/dataset_audit/monovab_label_distribution.csv` — CSV table of MONOVAB positive/negative counts and prevalence.
