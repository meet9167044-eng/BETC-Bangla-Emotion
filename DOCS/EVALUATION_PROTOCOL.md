# EVALUATION_PROTOCOL.md

This file defines exactly how data flows through training, validation,
and test so that no experiment in `EXPERIMENT_PLAN.md` leaks information
across splits.

## 1. Split Methodology **[FINALIZED mechanism / INITIAL-HP ratio]**

- Splits are created **once**, immediately after dataset harmonization
  and deduplication (`DATASET_SPEC.md`), and **before** any
  preprocessing, vectorization, or model fitting.
- **[INITIAL-HP]** Split ratio: 70% train / 15% validation / 15% test.
- **[FINALIZED]** Because this is multi-label data with class imbalance,
  use an **iterative multi-label-stratified split**
  (e.g. `scikit-multilearn`'s `iterative_train_test_split`), not a plain
  random split, so rare emotion classes are represented in all three
  splits.
- **[FINALIZED]** Once created, the split must be saved as an artifact
  (`artifacts/splits/{train,val,test}_ids.json` or equivalent row-index
  file) and reused for every experiment (baselines, BETC, ablations,
  final experiments) — every experiment in `EXPERIMENT_PLAN.md` Sections
  1–3 uses the **same** split, so results are comparable. F3
  (multi-seed validation) is the only case where the split itself is
  regenerated per seed, and each seed's split must also be saved.

## 2. What Each Split Is For

| Split | Used for |
|---|---|
| **Train** | Fitting TF-IDF vectorizers, fitting the Classifier Chain / baseline models. |
| **Validation** | Per-class threshold optimization (`PIPELINE_SPEC.md` Section 3.7); any hyperparameter comparison across ablations; model selection. |
| **Test** | Final evaluation only. Touched exactly once per experiment run, after all decisions (features, model, thresholds) are frozen. |

## 3. Leakage-Prevention Rules **[FINALIZED — hard rules]**

1. **Fit TF-IDF vectorizers on the training split only.** Validation and
   test text are transformed with `.transform()` using the already-fit
   vectorizer — never `.fit()` or `.fit_transform()` on validation or
   test data.
2. **Per-class thresholds are selected using validation-set predicted
   probabilities only.** Test-set probabilities must never be used, even
   informally, when choosing a threshold.
3. **The test split is evaluated exactly once per experiment
   configuration.** If a bug is found after test evaluation, the fix
   must be applied and the **entire experiment must be marked as
   re-run**, with the previous test result explicitly superseded and
   logged, not silently overwritten.
4. **No hyperparameter, chain order, or feature configuration may be
   chosen by observing test-set performance.** All such choices are made
   on the validation split (or via cross-validation within the training
   split).
5. **Deduplication and harmonization (`DATASET_SPEC.md`) happen before
   splitting**, so no duplicate of a training example appears in
   validation or test.
6. **Splits are identical across baselines, BETC, and ablations**
   (Section 1), so that metric differences reflect model/feature
   differences, not split differences.

## 4. Metrics — Definitions

All metrics are computed on the binary multi-label prediction matrix
`Ŷ` versus the true binary label matrix `Y`, over the fixed target label
set defined in `DATASET_SPEC.md`. The main BETC matrix is complete 0/1
after the documented harmonization rule. In particular, EmoNoBa
`disgust=0` values are an explicit harmonization assumption rather than
original annotations; the required sensitivity analysis evaluates the
impact of this assumption separately.

- **Macro-F1** — unweighted mean of per-class F1 scores. **[FINALIZED]**
  primary metric, since it does not let frequent classes dominate.
- **Micro-F1** — F1 computed over the pooled true/false
  positive/negative counts across all classes.
- **Per-class F1** — reported individually for every target emotion.
- **Precision and Recall** — overall and per-class.
- **Hamming Loss** — fraction of individual label predictions that are
  incorrect.
- **Jaccard similarity** — per-sample intersection-over-union of
  predicted vs. true label sets, averaged.
- **Subset accuracy (exact match ratio)** — fraction of samples where
  the entire predicted label set exactly matches the true label set.

## 5. Statistical Validation Protocol (for F3 in `EXPERIMENT_PLAN.md`)

- Run the full BETC pipeline (including a fresh split, per Section 1)
  across **[INITIAL-HP]** at least 5 random seeds.
- For each seed, save: the split, the fitted vectorizers, the fitted
  chain, the thresholds, and the resulting metrics — all under a
  seed-specific subdirectory of `artifacts/` and `results/`.
- Report **mean ± standard deviation** for Macro-F1 and Micro-F1 across
  seeds. Do not report a single seed's result as "the" result.
- Where a paired comparison between two configurations is needed (e.g.
  BETC vs. an ablation), use a paired statistical test on per-example
  predictions where feasible, rather than comparing only mean metrics.

## 6. Reporting Format Standard

Every results file must include, at minimum:
- Experiment ID (matching `EXPERIMENT_PLAN.md`).
- Full config used (or a path to the config file).
- Split identifier / seed used.
- All metrics from Section 4.
- Timestamp and code/commit reference if available.
- Explicit category label: `baseline` / `betc_full` / `ablation` /
  `final_experiment` (see `EXPERIMENT_PLAN.md` Section 5).

## 7. No Leakage Exceptions

**[FINALIZED]** There are no exceptions to Section 3. "Just checking"
test-set performance during development, "quickly" fitting a vectorizer
on the full dataset for convenience, or reusing a test-fit vectorizer
"because it's faster" are all leakage and are prohibited — see
`RESEARCH_RULES.md`.
