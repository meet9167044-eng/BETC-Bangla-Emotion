# IMPLEMENTATION_PLAN.md

This is the actionable, phase-by-phase build order. **Do not start a
phase until the previous phase's completion criteria are met.** Do not
write model code before Phase 6. This order is itself a
**[FINALIZED]** decision:

```
Research specification
  → Dataset acquisition
  → Dataset audit
  → Label/taxonomy verification
  → Dataset harmonization
  → Deduplication
  → Final dataset construction
  → Reproducible splitting
  → Preprocessing
  → TF-IDF feature extraction
  → Feature fusion
  → Baselines
  → BETC training
  → Validation threshold optimization
  → Ablations
  → Multi-seed experiments
  → Final untouched test evaluation
  → Error analysis
  → Final results / reporting
```

## Repository Structure

```
README.md
PROJECT_SPEC.md
DATASET_SPEC.md
PIPELINE_SPEC.md
EXPERIMENT_PLAN.md
EVALUATION_PROTOCOL.md
IMPLEMENTATION_PLAN.md
RESEARCH_RULES.md

data/
  raw/
    emonoba/
    ubmec/
    monovab/
  interim/
  processed/
    train/
    validation/
    test/

src/
  data/            # acquisition, audit, harmonization, dedup, split code
  features/        # preprocessing, TF-IDF fitting/transform, fusion
  models/          # classifier chain, baselines, threshold optimization
  evaluation/       # metrics, statistical validation
  utils/           # shared helpers (seeding, logging, config loading)

configs/
  label_mapping.yaml
  negations_bn.txt
  intensifiers_bn.txt
  betc_full.yaml
  baselines/
  ablations/

experiments/       # experiment run scripts / definitions, one per EXPERIMENT_PLAN.md ID
results/
  baselines/
  betc_full/
  ablations/
  final/

artifacts/
  splits/
  vectorizers/
  models/
  thresholds/

logs/
  dataset_audit/
  runs/

tests/
notebooks/         # exploration only — not the source of truth for logic
```

**[FINALIZED]** Essential logic (preprocessing, feature extraction,
model training, evaluation) lives in `src/` as importable, testable
modules. Notebooks may call into `src/` for exploration but must never
be the only place a piece of logic exists.

---

## Phase 0 — Repository Scaffolding

- **Objective:** Create the folder structure above and place all 8
  specification files at the repository root.
- **Inputs:** This document set.
- **Tasks:** Create directories; add `.gitkeep` or `README.md` stubs
  where needed; set up a Python environment/dependency file
  (`requirements.txt` or `pyproject.toml`) listing: `scikit-learn`,
  `pandas`, `numpy`, `scipy`, `scikit-multilearn`, `bnlp_toolkit` or
  `bnunicodenormalizer`, `matplotlib`, `seaborn`, `pytest`.
- **Outputs:** Empty but structured repository, dependency file.
- **Validation checks:** Every directory in the tree above exists;
  dependency file installs cleanly in a fresh virtual environment.
- **Completion criteria:** `pip install -r requirements.txt` (or
  equivalent) succeeds; folder tree matches this spec exactly.
- **Do NOT:** Add any model or dataset code yet. Do NOT add any
  pretrained-model dependency (e.g. `transformers`, `sentence-transformers`).
- **Depends on:** Nothing.

## Phase 1 — Dataset Acquisition ✅ COMPLETED

> **Status:** COMPLETE — 2026-09-19  
> All three raw datasets are present under `Data/raw/` and have been verified openable and readable.  
> **Structural note recorded:** Actual file layout differs from the documented spec layout (UBMEC and MONOVAB are flat files under `Data/raw/`, not in source-named subdirectories). This contradiction is documented in `results/dataset_audit/STRUCTURE_CONTRADICTION.md` and `results/dataset_audit/audit_status.json`. No files were modified.

- **Objective:** Obtain the raw EmoNoBa, UBMEC, and MONOVAB files and
  place them, untouched, under `data/raw/`.
- **Inputs:** `DATASET_SPEC.md` Section 2 and Section 6 (Blocking
  Verification Items 1–2).
- **Tasks:** Locate current, working, licensed-for-use sources for each
  dataset; download; record exact source URL and access date in
  `logs/dataset_audit/acquisition.md`.
- **Outputs:** `Data/raw/Emonoba/{Train.csv,Val.csv,Test.csv}`,
  `Data/raw/UBMEC Corpus_Sakib(updated).xlsx`,
  `Data/raw/MONOVAB (1).csv`.
- **Validation checks:** Files open correctly; encoding is confirmed
  UTF-8 (or documented otherwise); no file is modified after download.
- **Completion criteria:** All three raw datasets present and logged
  with provenance. ✅
- **Do NOT:** Modify, clean, relabel, or delete any row in `Data/raw/`
  at this stage.
- **Depends on:** Phase 0.

## Phase 2 — Dataset Audit ✅ COMPLETED

> **Status:** COMPLETE — 2026-09-19  
> All three datasets fully audited. Every `[VERIFY]` item in `DATASET_SPEC.md` Section 2 resolved against actual measured data. Discrepancies against literature documented.  
> **Audit outputs:** `results/dataset_audit/` (11 files including per-dataset JSON audits, label distribution CSVs, cross-dataset overlap JSON, data quality anomalies JSON, and full Markdown report).

- **Objective:** Resolve `DATASET_SPEC.md` Section 6 Blocking
  Verification Items 2, 3, 5, 8, 9.
- **Inputs:** `Data/raw/*`.
- **Tasks:** For each source dataset, document actual file format,
  column names, label encoding, row count, per-label positive count,
  and confirm no Banglish/code-mixed rows are present (or flag and
  exclude them if they are, per `PROJECT_SPEC.md` scope).
- **Actual Outputs:** `results/dataset_audit/dataset_audit_report.md`,
  `results/dataset_audit/emonoba_audit.json`,
  `results/dataset_audit/ubmec_audit.json`,
  `results/dataset_audit/monovab_audit.json`,
  `results/dataset_audit/cross_dataset_overlap.json`,
  `results/dataset_audit/data_quality_anomalies.json`,
  `results/dataset_audit/{emonoba,ubmec,monovab}_label_distribution.csv`.
- **Key findings:** EmoNoBa=22,739 rows (matches literature upper bound);
  UBMEC=13,436 rows (literature says 13,072 — +364 exact duplicate rows
  explain the difference); MONOVAB=10,224 rows (literature says 10,244).
  655 UBMEC rows contain mixed Bangla+Latin script. 5 degenerate texts.
  58 conflicting-label duplicate texts in UBMEC, 33 in MONOVAB, 8 in EmoNoBa.
  65 cross-dataset duplicate texts. EmoNoBa has 0 Latin-character rows.
- **Validation checks:** All `[VERIFY]` claims resolved. ✅
- **Completion criteria:** All three audit files complete. ✅
- **Do NOT:** Proceed to harmonization while any audit item is still
  unresolved.
- **Depends on:** Phase 1.

## Phase 3 — Label/Taxonomy Verification and Harmonization Setup ✅ COMPLETED

> **Status:** COMPLETE — 2026-09-19. All 4 blocking decisions approved by user.
>
> **Approved decisions (2026-09-19):**
> 1. EmoNoBa Disgust — **Strategy D:** `disgust=0` as documented operational assumption; F5 mandatory
> 2. EmoNoBa Love-only rows — **LOVE-1:** exclude 2,277 rows
> 3. MONOVAB Contempt-only rows — **CONTEMPT-1:** exclude 2,128 rows
> 4. MONOVAB `enjoyment` — **FROZEN** mapping to `joy` (semantic harmonization)
> 5. UBMEC one-hot conversion — **FROZEN**
> 6. Target taxonomy — **FROZEN:** `anger, disgust, fear, joy, sadness, surprise`
>
> **Harmonization design outputs:** `results/harmonization/` (6 files)

- **Objective:** Verify the downloaded schemas and implement the already-
  finalized six-label taxonomy: `anger, disgust, fear, joy, sadness,
  surprise`.
- **Inputs:** Phase 2 audit outputs (`results/dataset_audit/`); `DATASET_SPEC.md` Section 3.
- **Tasks:** Confirm actual label columns in each raw file; encode UBMEC's
  single categorical label into six binary target columns; map MONOVAB's
  target labels; exclude non-target `love` and `contempt`; represent
  EmoNoBa `disgust` as 0 under the explicit harmonization assumption
  documented in `DATASET_SPEC.md`; create the checked-in mapping config.
- **Actual Outputs:** `results/harmonization/harmonization_analysis.md`,
  `results/harmonization/label_mapping.csv`,
  `results/harmonization/label_coverage.csv`,
  `results/harmonization/harmonization_strategies.csv`,
  `results/harmonization/source_label_semantics.json`,
  `results/harmonization/harmonization_decisions.json`.
- **Validation checks:** Every native label in every dataset accounted for;
  no label silently dropped. ✅
- **Completion criteria:** All decisions approved by user. ✅
- **Do NOT:** Map `love` to `disgust`; add `love` as a seventh target; or
  describe the EmoNoBa `disgust=0` values as original annotations.
- **Depends on:** Phase 2.

## Phase 4 — Dataset Harmonization ✅ COMPLETED (pre-deduplication)

> **Status:** COMPLETE — 2026-09-19. Harmonization applied; pre-deduplication corpus saved.
> **Note:** Deduplication is separated into Phase 5 per user instruction (stop before deduplication).
> **Outputs:**
> - `Data/interim/harmonized_pre_dedup.csv` — 41,994 rows, 18 columns
> - `logs/harmonization/harmonization_manifest.csv` — 46,399 rows (all source rows)
> - `logs/harmonization/exclusion_log.csv` — 4,405 excluded rows
> - `logs/harmonization/transformation_log.csv` — 9 transformation entries
> - `logs/harmonization/harmonization_validation.json` — validation + statistics
> - `logs/harmonization/phase4_harmonization_report.md` — full report

- **Objective:** Produce a harmonized dataset in `Data/interim/` with all approved
  label mappings applied and all excluded rows logged.
- **Inputs:** `Data/raw/*`; Phase 3 approved decisions.
- **Actual measured results:**
  - EmoNoBa: 22,739 raw → 2,277 excluded (LOVE-1) → **20,462 included**
  - UBMEC: 13,436 raw → 0 excluded → **13,436 included**
  - MONOVAB: 10,224 raw → 2,128 excluded (CONTEMPT-1) → **8,096 included**
  - **Total harmonized (pre-dedup): 41,994 rows**
- **Label statistics (pre-dedup):**
  `anger=11,457 (27.28%)`, `disgust=4,146 (9.87%)*`,
  `fear=1,818 (4.33%)`, `joy=15,180 (36.15%)`,
  `sadness=9,552 (22.75%)`, `surprise=2,552 (6.08%)`
  *EmoNoBa disgust=0 is an assumption for all 20,462 EmoNoBa rows
- **Validation checks:** All 8 validation checks PASSED ✅
  (binary-only labels, no nulls, correct source flags, UBMEC one-hot valid,
  zero all-zero rows, no Love-only rows in included corpus)
- **Deduplication:** NOT YET APPLIED — see Phase 5.
- **Do NOT:** Delete rows silently. EmoNoBa `disgust=0` is a documented assumption.
- **Depends on:** Phase 3.

## Phase 5 — Deduplication ✅ COMPLETED (partial — 1 blocker)

> **Status:** COMPLETE for all objectively resolvable duplicates — 2026-09-19.
> **BLOCKER:** 98 conflicting duplicate groups (251 rows) require user-approved resolution strategy before Phase 5b (splitting) can begin.
>
> **Measured results:**
> - Phase 4 input: 41,994 rows → Phase 5 output: **41,435 rows**
> - 559 identical-label duplicate rows removed (406 groups)
> - 251 conflicting rows retained unresolved (98 groups) — logged in `conflicting_duplicates.csv`
> - Cross-source groups: EmoNoBa↔UBMEC=8, EmoNoBa↔MONOVAB=9, UBMEC↔MONOVAB=41, all-three=2
> - EmoNoBa cross-split groups: 12 (resolved via canonical selection)
> - All 8 validation checks: PASSED
>
> **Scripts:** `scripts/data_pipeline/phase5_dedup.py`  
> **Outputs:** `Data/interim/harmonized_deduplicated.csv` (41,435 rows),
> `logs/deduplication/` (5 files including conflict log)

- **Objective:** Remove duplicate texts; log all decisions; retain unresolvable conflicts for user review.
- **Inputs:** `Data/interim/harmonized_pre_dedup.csv`; Phase 4 provenance.
- **Actual Outputs:** `Data/interim/harmonized_deduplicated.csv`,
  `logs/deduplication/deduplication_report.md`,
  `logs/deduplication/duplicate_groups.csv`,
  `logs/deduplication/duplicate_provenance.csv`,
  `logs/deduplication/conflicting_duplicates.csv`,
  `logs/deduplication/deduplication_validation.json`.
- **Canonical selection rule (deterministic):** Most metadata > EmoNoBa > UBMEC > MONOVAB > Train > Val > Test > smallest row ID.
- **BLOCKER (98 conflicting groups):** Options: DROP all / UNION labels / FIRST occurrence / DATASET-PRIORITY. Requires user decision before splitting.
- **Depends on:** Phase 4.

## Phase 5b — Reproducible Splitting (BLOCKED pending conflict resolution)

> **Status:** NOT STARTED — blocked by 98 unresolved conflicting duplicate groups from Phase 5.

- **Objective:** Create the frozen train/validation/test split.
- **Inputs:** `Data/interim/harmonized_deduplicated.csv` (after conflict resolution);
  `EVALUATION_PROTOCOL.md` Section 1.
- **Tasks:** Apply iterative multi-label-stratified split at 70/15/15;
  save row indices/IDs per split; save per-split label prevalence for
  sanity-checking stratification quality.
- **Outputs:** `data/processed/{train,validation,test}/`;
  `artifacts/splits/split_v1.json`.
- **Validation checks:** No row ID appears in more than one split; every
  target label has a nonzero positive count in every split.
- **Completion criteria:** Split saved and validated.
- **Do NOT:** Re-split later without a new, separately versioned artifact.
- **Depends on:** Phase 5 conflict resolution (user decision required).

## Phase 6 — Preprocessing Implementation

- **Objective:** Implement and unit-test the preprocessing function from
  `PIPELINE_SPEC.md` Section 3.1.
- **Inputs:** `data/processed/train/` (for manual inspection only, not
  fitting); negation/intensifier word lists.
- **Tasks:** Implement Unicode normalization, noise/URL/@mention
  removal, negation/intensifier preservation as a single deterministic
  function in `src/features/preprocessing.py`; manually inspect output
  on 20–30 real sample sentences from the training split.
- **Outputs:** `src/features/preprocessing.py`;
  `tests/test_preprocessing.py`; a manual-inspection note in
  `logs/dataset_audit/preprocessing_spotcheck.md`.
- **Validation checks:** Negation/intensifier words are never stripped;
  function is deterministic (same input → same output); unit tests pass.
- **Completion criteria:** Tests pass; spot-check note confirms
  preprocessing looks correct on real Bangla examples.
- **Do NOT:** Apply this function differently to different splits. Do
  NOT use a generic (non-Bangla-aware) Unicode normalizer.
- **Depends on:** Phase 5.

## Phase 7 — TF-IDF Feature Extraction and Fusion

- **Objective:** Implement word TF-IDF, character TF-IDF, and fusion per
  `PIPELINE_SPEC.md` Sections 3.2–3.4.
- **Inputs:** Preprocessed train/validation/test text.
- **Tasks:** Fit both vectorizers on training text only; transform all
  three splits; fuse with `scipy.sparse.hstack`; save fitted
  vectorizers.
- **Outputs:** `src/features/tfidf.py`; `artifacts/vectorizers/word_tfidf.joblib`,
  `artifacts/vectorizers/char_tfidf.joblib`; feature matrices saved or
  reproducibly regenerable.
- **Validation checks:** Vectorizers are fit-once on train only (unit
  test asserting `.fit()` is never called on val/test data); resulting
  matrix shapes match `(n_samples, 25000)` at initial hyperparameters.
- **Completion criteria:** Feature matrices produced for all three
  splits; leakage unit test passes.
- **Do NOT:** Fit on validation or test data, even "just to check."
- **Depends on:** Phase 6.

## Phase 8 — Baselines

- **Objective:** Implement and run B1–B5 from `EXPERIMENT_PLAN.md`
  Section 1.
- **Inputs:** Fused features from Phase 7; frozen split from Phase 5.
- **Tasks:** Train each baseline model; evaluate per
  `EVALUATION_PROTOCOL.md`; save configs and results.
- **Outputs:** `results/baselines/{B1..B5}.json` (or similar), each
  tagged `category: baseline`.
- **Validation checks:** Each baseline uses the exact same split and
  feature-fitting rules as BETC will.
- **Completion criteria:** All five baseline results saved.
- **Do NOT:** Tune baseline hyperparameters on the test set.
- **Depends on:** Phase 7.

## Phase 9 — BETC Training (M1)

- **Objective:** Train the full BETC pipeline per `PIPELINE_SPEC.md`.
- **Inputs:** Fused features; frozen split; `configs/betc_full.yaml`.
- **Tasks:** Fix and document chain order (descending frequency, per
  `PIPELINE_SPEC.md` 3.5); fit `ClassifierChain`; save fitted model.
- **Outputs:** `artifacts/models/betc_full_chain.joblib`;
  `configs/betc_full.yaml` fully populated.
- **Validation checks:** Chain order used matches what is logged in the
  config; no test data touched.
- **Completion criteria:** Model fit and saved.
- **Do NOT:** Change the architecture (features, chain mechanism, base
  classifier) at this stage — that requires an ablation (Phase 11), not
  a change to M1 itself.
- **Depends on:** Phase 7 (can run in parallel with Phase 8).

## Phase 10 — Validation Threshold Optimization

- **Objective:** Compute and save per-class thresholds per
  `PIPELINE_SPEC.md` Section 3.7.
- **Inputs:** Fitted BETC model from Phase 9; validation split.
- **Tasks:** Get validation `predict_proba`; sweep thresholds per class;
  select F1-maximizing threshold per class; save.
- **Outputs:** `artifacts/thresholds/betc_full_thresholds.json`.
- **Validation checks:** Threshold selection uses only validation data
  (unit test / code review confirms no test-split reference).
- **Completion criteria:** Thresholds saved for every target label.
- **Do NOT:** Use test-set probabilities anywhere in this phase.
- **Depends on:** Phase 9.

## Phase 11 — Ablations

- **Objective:** Run A1–A9 from `EXPERIMENT_PLAN.md` Section 3.
- **Inputs:** Same split, same base infrastructure as M1; one changed
  component per ablation.
- **Tasks:** For each ablation ID, create its config, run, save result
  tagged `category: ablation`.
- **Outputs:** `results/ablations/{A1..A9}.json`; `configs/ablations/*.yaml`.
- **Validation checks:** Each ablation config differs from
  `configs/betc_full.yaml` in exactly the one documented dimension.
- **Completion criteria:** All nine ablation results saved.
- **Do NOT:** Vary more than one component per ablation run. Do NOT
  report an ablation result as if it were M1.
- **Depends on:** Phase 10.

## Phase 12 — Multi-Seed Experiments (F3)

- **Objective:** Run F3 from `EXPERIMENT_PLAN.md` Section 4.
- **Inputs:** Full pipeline (Phases 4–10) re-run per seed.
- **Tasks:** Repeat split → preprocess → features → BETC → thresholds
  → test evaluation for ≥5 seeds; save each seed's full artifact set;
  aggregate mean ± std.
- **Outputs:** `results/final/multiseed_summary.json`;
  per-seed subfolders under `artifacts/` and `results/`.
- **Validation checks:** No seed's result is discarded or excluded
  without a documented, non-performance-based reason (e.g. a crashed
  run is fine to exclude and re-run; a low-scoring run is not).
- **Completion criteria:** ≥5 seed results aggregated with mean ± std
  reported.
- **Do NOT:** Cherry-pick the best-performing seed as "the" result.
- **Depends on:** Phase 11 (informational — F3 re-runs the M1
  configuration specifically, not the ablations).

## Phase 13 — Final Untouched Test Evaluation and Cross-Domain Check (F1, F2)

- **Objective:** Run F1 and F2 from `EXPERIMENT_PLAN.md` Section 4.
- **Inputs:** Test-set predictions already produced in Phases 8, 9–10,
  11, 12.
- **Tasks:** Break down test performance by `source_dataset`/`domain`
  (F1); assemble a comparison table including cited external B6/B7
  numbers, clearly labeled as external and non-identical-protocol (F2).
- **Outputs:** `results/final/cross_domain_breakdown.json`;
  `results/final/benchmark_comparison.md`.
- **Validation checks:** External numbers are visually and structurally
  distinguished (e.g. a separate table or a "source: external" column)
  from this repository's own measured numbers.
- **Completion criteria:** Both reports produced.
- **Do NOT:** Present B6/B7 numbers as if measured under this project's
  own protocol.
- **Depends on:** Phase 12.

## Phase 14 — Error Analysis (F4)

- **Objective:** Run F4 from `EXPERIMENT_PLAN.md` Section 4.
- **Inputs:** Final test predictions from M1 (Phase 10 model, Phase 13
  evaluation).
- **Tasks:** Manually inspect misclassified examples involving
  negation, sarcasm, mixed emotions, short texts, and the lowest-F1
  classes; write up patterns observed.
- **Outputs:** `results/final/error_analysis.md`.
- **Validation checks:** Findings are descriptive/qualitative and do not
  introduce new unverified quantitative claims.
- **Completion criteria:** Error analysis document complete.
- **Depends on:** Phase 13.

## Phase 15 — Final Results and Reporting

- **Objective:** Assemble the final report using only saved artifacts
  and results files.
- **Inputs:** All `results/` and `logs/` content from Phases 1–14.
- **Tasks:** Write the final report/paper draft, citing every number
  from its corresponding results file; explicitly state which results
  are this project's own measurements and which are external/cited.
- **Outputs:** Final report document.
- **Validation checks:** Every number in the report traces to a file in
  `results/` or `logs/`; run the Documentation Consistency Audit
  approach described in the project's spec-creation process against the
  final report as well.
- **Completion criteria:** Report complete and every claim traceable.
- **Do NOT:** Add any number to the report that is not backed by a saved
  artifact.
- **Depends on:** Phase 14.
