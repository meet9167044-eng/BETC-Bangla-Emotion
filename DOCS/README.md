# BETC — Bangla Emotion TF-IDF Classifier Chain

**Status: Specification only. No code, no data, no results exist yet.**

This repository implements a multi-label Bangla emotion detection pipeline
("BETC") using word- and character-level TF-IDF features fused into a single
representation, and a Classifier Chain of Logistic Regression classifiers
with per-class decision thresholds. No pretrained language model, pretrained
embedding, or deep neural encoder is used anywhere in the core pipeline.

This document set is the **source of truth** for an AI coding agent (Claude
Code, Antigravity, Codex, Cursor, or a human contributor) to implement the
project correctly, in the correct order, without redesigning it.

## Document Map

Read in this order before writing any code:

| # | File | Purpose |
|---|---|---|
| 1 | `README.md` | This file. Entry point and reading order. |
| 2 | `PROJECT_SPEC.md` | What we are building, why, scope, and the decision-category legend used throughout every other file. |
| 3 | `DATASET_SPEC.md` | The finalized combined-dataset strategy, harmonization rules, deduplication rules, and every fact about the data that must still be verified. |
| 4 | `PIPELINE_SPEC.md` | The complete BETC architecture: preprocessing, feature extraction, model, math, pseudocode. |
| 5 | `EXPERIMENT_PLAN.md` | Baselines, full BETC, ablations, and final experiments — kept strictly separate. |
| 6 | `EVALUATION_PROTOCOL.md` | Split methodology, leakage rules, metrics, statistical validation. |
| 7 | `IMPLEMENTATION_PLAN.md` | Phase-by-phase build order with completion criteria. This is the file an agent should follow step by step. |
| 8 | `RESEARCH_RULES.md` | Hard constraints. Read this fully before writing or modifying any code. |

## Non-Negotiable Reading Order for an AI Coding Agent

1. `RESEARCH_RULES.md` — understand what you are not allowed to do.
2. `PROJECT_SPEC.md` — understand the decision-category legend.
3. `DATASET_SPEC.md` — understand that the dataset pipeline must be verified
   and harmonized **before** any model code is written. The six target labels are
already fixed; only factual dataset/schema verification remains.
4. `PIPELINE_SPEC.md` and `EXPERIMENT_PLAN.md` — understand the target
   architecture and experiment matrix.
5. `EVALUATION_PROTOCOL.md` — understand how leakage is prevented.
6. `IMPLEMENTATION_PLAN.md` — execute phase by phase, in order, without
   skipping ahead. Do not begin model code (Phase 6 onward) until the
   dataset phases (0–5) pass their validation checks.

## Project Identity

- **Name:** BETC (Bangla Emotion TF-IDF Classifier Chain) — a working name,
  not a published/claimed novel architecture. See `PROJECT_SPEC.md`.
- **Task:** Multi-label Bangla text emotion classification.
- **Target labels (FINALIZED):** exactly six emotions — anger, disgust, fear,
  joy, sadness, surprise. EmoNoBa `love` and MONOVAB `contempt` are not
  output labels; see `DATASET_SPEC.md` for the explicit harmonization rule.
- **Core techniques:** Word TF-IDF + Character TF-IDF (fused) → Classifier
  Chain → Logistic Regression base estimator → per-class threshold
  optimization.
- **Explicitly excluded:** pretrained transformers, pretrained embeddings,
  external LLM/embedding APIs, deep sequence encoders inside BETC itself
  (they may appear only as external baselines for comparison).

## Repository Structure

See `IMPLEMENTATION_PLAN.md` for the full annotated tree. Summary:

```
data/raw/          # untouched original downloads, one subfolder per source dataset
data/interim/       # harmonized, deduplicated, not-yet-split data
data/processed/     # final train/validation/test splits, feature matrices
src/                # all production/research logic (not notebooks)
configs/            # YAML/JSON configs for every run — no magic numbers in code
experiments/        # experiment definitions (baseline / full / ablation / final)
results/            # metrics tables, per-run outputs
artifacts/          # fitted vectorizers, models, thresholds, splits
logs/               # run logs, environment info, seeds
tests/              # unit tests for preprocessing, feature fusion, leakage checks
notebooks/          # exploration only — never the source of truth for logic
```

## Documentation Decisions (Contradiction Resolution)

An earlier stage of this project (recorded in
`EECR/BETC_Bangla_Emotion_Architecture_and_Implementation_Plan.docx`)
specified **EmoNoBa alone** as the dataset. This has been **superseded** by
a later, final decision to use a **combined, harmonized dataset**
(EmoNoBa + UBMEC + MONOVAB) mapped toward Ekman's six basic emotions. Every
file in this spec set reflects the combined-dataset decision. Wherever the
original architecture document assumed EmoNoBa alone, that assumption is
now void; `DATASET_SPEC.md` is authoritative on dataset matters and
overrides the dataset section of the `.docx`. The BETC pipeline
architecture itself (TF-IDF, Classifier Chain, Logistic Regression,
threshold optimization) is **unchanged** and still matches the `.docx`.
