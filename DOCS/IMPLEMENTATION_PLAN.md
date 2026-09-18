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

## Phase 1 — Dataset Acquisition

- **Objective:** Obtain the raw EmoNoBa, UBMEC, and MONOVAB files and
  place them, untouched, under `data/raw/`.
- **Inputs:** `DATASET_SPEC.md` Section 2 and Section 6 (Blocking
  Verification Items 1–2).
- **Tasks:** Locate current, working, licensed-for-use sources for each
  dataset; download; record exact source URL and access date in
  `logs/dataset_audit/acquisition.md`.
- **Outputs:** `data/raw/emonoba/`, `data/raw/ubmec/`,
  `data/raw/monovab/` populated; `logs/dataset_audit/acquisition.md`.
- **Validation checks:** Files open correctly; encoding is confirmed
  UTF-8 (or documented otherwise); no file is modified after download.
- **Completion criteria:** All three raw datasets present and logged
  with provenance.
- **Do NOT:** Modify, clean, relabel, or delete any row in `data/raw/`
  at this stage.
- **Depends on:** Phase 0.

## Phase 2 — Dataset Audit

- **Objective:** Resolve `DATASET_SPEC.md` Section 6 Blocking
  Verification Items 2, 3, 5, 8, 9.
- **Inputs:** `data/raw/*`.
- **Tasks:** For each source dataset, document actual file format,
  column names, label encoding, row count, per-label positive count,
  and confirm no Banglish/code-mixed rows are present (or flag and
  exclude them if they are, per `PROJECT_SPEC.md` scope).
- **Outputs:** `logs/dataset_audit/{emonoba,ubmec,monovab}_audit.md`,
  each with actual measured statistics (never copy the `[VERIFY]`
  literature numbers from `DATASET_SPEC.md` as if they were measured).
- **Validation checks:** Every `[VERIFY]` claim in `DATASET_SPEC.md`
  Section 2 has a corresponding measured value in the audit logs, with
  any discrepancy from the literature number explicitly noted.
- **Completion criteria:** All three audit files complete; any
  Banglish/code-mixed rows identified and excluded with a documented
  count.
- **Do NOT:** Proceed to harmonization while any audit item is still
  unresolved.
- **Depends on:** Phase 1.

## Phase 3 — Label/Taxonomy Verification and Harmonization Setup

- **Objective:** Verify the downloaded schemas and implement the already-
  finalized six-label taxonomy: `anger, disgust, fear, joy, sadness,
  surprise`.
- **Inputs:** Phase 2 audit outputs; `DATASET_SPEC.md` Section 3.
- **Tasks:** Confirm actual label columns in each raw file; encode UBMEC's
  single categorical label into six binary target columns; map MONOVAB's
  target labels; exclude non-target `love` and `contempt`; represent
  EmoNoBa `disgust` as 0 under the explicit harmonization assumption
  documented in `DATASET_SPEC.md`; create the checked-in mapping config.
- **Outputs:** `configs/label_mapping.yaml`; a harmonization decision/report
  explicitly recording the EmoNoBa `disgust=0` assumption.
- **Validation checks:** The mapping file accounts for every native
  label in every source dataset — no label is silently dropped without
  being listed in the mapping file's "excluded" section with a reason.
- **Completion criteria:** `configs/label_mapping.yaml` is complete and
  reviewed; the six target labels are present in exactly the fixed order;
  the harmonization report records that EmoNoBa `love` is excluded and its
  missing `disgust` is operationally encoded as 0 as a documented assumption.
- **Do NOT:** Map `love` to `disgust`; add `love` as a seventh target; or
  describe the EmoNoBa `disgust=0` values as original annotations.
- **Depends on:** Phase 2.

## Phase 4 — Dataset Harmonization and Deduplication

- **Objective:** Produce a single harmonized, deduplicated dataset in
  `data/interim/`.
- **Inputs:** `data/raw/*`, `configs/label_mapping.yaml`.
- **Tasks:** Apply the label mapping; preserve `source_dataset` and
  `domain` columns where available; apply comparison-normalization and
  deduplicate across all three sources (`DATASET_SPEC.md` Section 4,
  steps 4–5); log duplicate counts (resolves Blocking Item 7).
- **Outputs:** `data/interim/harmonized_deduplicated.parquet` (or
  equivalent); `logs/dataset_audit/harmonization_report.md` with actual
  measured pre/post counts and per-label prevalence (resolves Blocking
  Item 8).
- **Validation checks:** No row has an ambiguous label (every value is
  either 0, 1, or explicitly "not annotated" — never an implicit blank
  treated as 0); duplicate count matches the harmonization report;
  per-label prevalence table is generated and reviewed for anything that
  suggests contamination (a suspiciously balanced or suspiciously
  identical distribution to the earlier `[VERIFY]` literature numbers
  should be double-checked, not trusted at face value).
- **Completion criteria:** Harmonization report reviewed; interim file
  saved.
- **Do NOT:** Delete rows silently. Do NOT collapse "not annotated" into
  "0" without the justification required by Phase 3's decision.
- **Depends on:** Phase 3.

## Phase 5 — Reproducible Splitting

- **Objective:** Create the frozen train/validation/test split.
- **Inputs:** `data/interim/harmonized_deduplicated.parquet`;
  `EVALUATION_PROTOCOL.md` Section 1.
- **Tasks:** Apply iterative multi-label-stratified split at 70/15/15;
  save row indices/IDs per split; save per-split label prevalence for
  sanity-checking stratification quality.
- **Outputs:** `data/processed/{train,validation,test}/`;
  `artifacts/splits/split_v1.json`.
- **Validation checks:** No row ID appears in more than one split; every
  target label has a nonzero positive count in every split (if not,
  flag and reconsider split ratio/method before proceeding).
- **Completion criteria:** Split saved and validated.
- **Do NOT:** Re-split later without creating a new, separately versioned
  split artifact (e.g. `split_v2.json`) — never overwrite a split
  silently once any experiment has used it.
- **Depends on:** Phase 4.

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
