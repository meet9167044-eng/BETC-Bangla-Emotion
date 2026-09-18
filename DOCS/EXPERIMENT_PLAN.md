# EXPERIMENT_PLAN.md

All experiments run under the identical protocol defined in
`EVALUATION_PROTOCOL.md`, on the same harmonized dataset and splits
defined in `DATASET_SPEC.md`. Results for different experiment
categories must be stored in **separate, clearly labeled result files**
(`results/baselines/`, `results/betc_full/`, `results/ablations/`,
`results/final/`) and must **never** be merged into a single table
without a column identifying which category each row belongs to. See
`RESEARCH_RULES.md`, rule on not mixing baseline and proposed-model
results.

## 1. BASELINES

Purpose: establish what simpler or alternative approaches achieve, under
the exact same features/splits where applicable, so BETC's value (if
any) can be measured honestly.

| ID | Model | Representation | Purpose |
|---|---|---|---|
| B1 | TF-IDF (word only) + Logistic Regression | Word TF-IDF only | Simple lexical baseline |
| B2 | TF-IDF (char only) + Logistic Regression | Char TF-IDF only | Spelling-robust baseline |
| B3 | TF-IDF (combined) + `MultiOutputClassifier` (independent per-label LR) | Word+Char TF-IDF | Label-independence baseline (no chain) |
| B4 | TF-IDF (combined) + Linear SVM | Word+Char TF-IDF | Strong classical baseline |
| B5 | TF-IDF (combined) + Random Forest | Word+Char TF-IDF | Nonlinear classical baseline |
| B6 | Published EmoNoBa AdaBoost result | Hand-crafted features | **[VERIFY]** External classical benchmark — cite, do not reproduce unless raw code/config is available; report as an external reference number, clearly labeled as not produced by this repository |
| B7 | Published EmoNoBa BiLSTM result | Trainable embeddings | **[VERIFY]** External neural benchmark — same rule as B6 |

**[FINALIZED]** B1–B5 must be trained and evaluated by this repository
under the exact same harmonized dataset and split as BETC. B6–B7 are
**external, cited numbers only** — see `RESEARCH_RULES.md` on never
fabricating or silently adapting external results to look like this
project's own measurements.

## 2. FULL BETC

| ID | Model | Description |
|---|---|---|
| M1 | BETC (full) | Combined Word+Char TF-IDF → Classifier Chain (frequency-ordered) → Logistic Regression (C=1.0, balanced) → per-class thresholds, exactly as specified in `PIPELINE_SPEC.md`. |

This is the single canonical "proposed model" run. Its config must be
saved in full (`configs/betc_full.yaml`) and its result stored in
`results/betc_full/`, separately from every baseline and every ablation.

## 3. ABLATIONS

Each ablation changes **exactly one** component of M1 relative to the
full BETC configuration, all else held fixed, so the effect of that one
component can be isolated.

| ID | Ablation | What changes | What stays fixed |
|---|---|---|---|
| A1 | Word-only features | Drop character TF-IDF view | Chain, LR, thresholds |
| A2 | Char-only features | Drop word TF-IDF view | Chain, LR, thresholds |
| A3 | No chain | Replace `ClassifierChain` with `MultiOutputClassifier` (independent per-label) | Features, LR, thresholds |
| A4 | Chain order: correlation-based | Reorder chain by label co-occurrence correlation instead of frequency | Features, LR, thresholds |
| A5 | Chain order: random (averaged) | Random chain order, averaged over ≥5 seeds | Features, LR, thresholds |
| A6 | Flat threshold | Use a fixed 0.5 cutoff instead of per-class optimized thresholds | Features, chain, LR |
| A7 | class_weight ablation | `class_weight=None` instead of `'balanced'` | Features, chain, thresholds |
| A8 | Alternative base classifier: SVM | Linear SVM in place of Logistic Regression inside the chain | Features, chain order, thresholds |
| A9 | Alternative base classifier: Random Forest | Random Forest in place of Logistic Regression inside the chain | Features, chain order, thresholds |

**[EXPERIMENTAL]** All ablation IDs and their exact configs must be
logged before running (`configs/ablations/<id>.yaml`). Ablation results
must never be described as "improvements" or "the final model" — they
exist to isolate component contributions, not to replace M1.

## 4. FINAL EXPERIMENTS

Run only after M1 (Full BETC) and all baselines/ablations above have
completed successfully under `EVALUATION_PROTOCOL.md`.

| ID | Experiment | Purpose |
|---|---|---|
| F1 | Cross-domain / cross-source check | Evaluate whether BETC's performance holds across the different source datasets (EmoNoBa, UBMEC, MONOVAB) and, where available, across EmoNoBa's original domain labels — using the `source_dataset`/`domain` columns preserved during harmonization (`DATASET_SPEC.md` Section 5). |
| F2 | Benchmark comparison | Report BETC's own measured numbers alongside the **cited, external** B6/B7 numbers, under a clearly labeled, non-identical-protocol caveat (since B6/B7 were not necessarily produced on this project's harmonized combined dataset). |
| F3 | Multi-seed statistical validation | Repeat the full M1 pipeline across multiple random seeds (≥5) affecting the split and any stochastic solver behavior; report mean ± standard deviation for Macro-F1 and Micro-F1. No single "best seed" may be reported in isolation — see `RESEARCH_RULES.md` on cherry-picking seeds. |
| F4 | Qualitative error analysis | Manually inspect misclassifications involving negation, sarcasm, mixed emotions, very short comments, and the lowest-F1 emotion classes. |
| F5 | EmoNoBa `disgust` harmonization sensitivity | Re-evaluate the final BETC configuration on an analysis subset that excludes EmoNoBa-derived rows from `disgust`-specific scoring, and report how the harmonization assumption affects the result. This does not change the primary fixed six-label training matrix. |

## 5. Explicit Separation Rule

**[FINALIZED]** Sections 1–4 above must remain in separate result
directories and separate report tables. A results table that mixes rows
from Section 1 (Baselines), Section 2 (Full BETC), Section 3
(Ablations), and Section 4 (Final Experiments) without an explicit
category column is a documentation error and must be fixed before
reporting.

## 6. No Results Exist Yet

**[NO-RESULT]** As of this document's creation, none of B1–B7, M1,
A1–A9, or F1–F5 have been run. Any numeric value that later appears
next to these IDs in this file (if this file is ever edited to include
placeholder numbers) is invalid and must be removed — results belong in
`results/`, not in this planning document.
