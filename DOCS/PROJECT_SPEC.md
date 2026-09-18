# PROJECT_SPEC.md

## 1. What We Are Building

A multi-label Bangla emotion detection system, named **BETC** (Bangla
Emotion TF-IDF Classifier Chain), that:

1. Ingests short, informal Bangla social-media text.
2. Preprocesses it with Bangla-specific Unicode normalization while
   deliberately preserving negation and intensifier words.
3. Represents each text with two complementary classical feature views —
   word-level TF-IDF and character-level TF-IDF — fused into one feature
   matrix.
4. Predicts a binary vector over six target emotions using a Classifier
   Chain of Logistic Regression classifiers, where each classifier in the
   chain conditions on the labels predicted earlier in a fixed order.
5. Calibrates a separate decision threshold per emotion on a held-out
   validation set, instead of a single global 0.5 cutoff.
6. Is evaluated with Macro-F1, Micro-F1, per-class F1, Hamming Loss,
   Jaccard similarity, and subset accuracy, benchmarked against classical
   and (externally reported) neural baselines.

## 2. Why We Are Building It This Way

- **No pretrained language models or embeddings.** The project is
  deliberately scoped to classical, from-scratch, CPU-feasible feature
  engineering. This is a scope decision, not a technical limitation to be
  "fixed" later.
- **Word + character TF-IDF fusion** exists because Bangla social media
  text has heavy spelling variation, informal typing, and morphological
  inflection that a word-only view misses and a character-only view
  partially recovers.
- **Classifier Chain instead of independent per-label classifiers**
  exists because the six target emotions are not independent —
  co-occurrence between emotions (e.g. anger and disgust) is common in
  the source data, and a chain lets later classifiers condition on
  earlier predictions.
- **Per-class threshold optimization** exists because the label
  distribution is imbalanced across the six emotions, and a single 0.5
  cutoff systematically under-predicts rarer classes.
- **A combined, harmonized dataset** (see `DATASET_SPEC.md`) exists
  because a single Bangla emotion dataset does not have enough labeled
  volume, and prior published work has already validated combining
  EmoNoBa, UBMEC, and MONOVAB toward a shared six-emotion label set.

## 3. Scope

### In scope
- Pure Bangla text (no Banglish / code-mixed text — explicitly excluded
  by prior project decision).
- Multi-label classification over exactly six fixed target emotions:
  **anger, disgust, fear, joy, sadness, surprise**. Source-only labels such
  as EmoNoBa `love` and MONOVAB `contempt` are excluded from the output
  space; see `DATASET_SPEC.md` for harmonization semantics.
- Classical ML only: TF-IDF, Logistic Regression, Classifier Chain, and
  classical baseline models (SVM, Random Forest, MultiOutputClassifier).
- CPU-only training and evaluation.

### Out of scope
- Banglish / code-mixed text.
- Any pretrained Transformer (BERT, Bangla-BERT, IndicBERT, XLM-R, etc.).
- Any pretrained word embedding (Word2Vec, FastText, GloVe) trained on
  external corpora.
- Any external LLM or embedding API call.
- Deep sequence models (LSTM, BiLSTM, CNN encoders) as part of BETC
  itself. They may appear **only** as externally reported baseline
  numbers for comparison — never trained/reproduced as part of this
  repository unless a future phase explicitly authorizes it.

## 4. Decision-Category Legend

Every claim, number, or choice in this document set belongs to exactly
one of the following seven categories. Every other spec file uses this
same legend and tags claims accordingly. An AI coding agent must not
upgrade a claim from a lower-confidence category to a higher one (e.g.
treating an "INITIAL HYPERPARAMETER" as a "FINALIZED DESIGN DECISION")
without an explicit instruction from the project owner.

| Tag | Meaning |
|---|---|
| **[FINALIZED]** | A design decision that is fixed and must not be silently changed. |
| **[INITIAL-HP]** | An initial hyperparameter value — a reasonable starting point, expected to be tuned, not a final number. |
| **[EXPERIMENTAL]** | A variable that is deliberately varied across ablations/experiments — there is no single "correct" value yet. |
| **[VERIFY]** | A dataset fact that is taken from external literature and **must be independently re-confirmed** against the actual downloaded files before being used. Never assume a literature-reported number applies to our own processed data. |
| **[LIT]** | A choice supported by published literature on Bangla emotion detection (cited by dataset/paper name where known). |
| **[PROPOSAL]** | A choice proposed by this project's authors/design discussion that has **not** been validated experimentally or in literature. Must be tested, not assumed correct. |
| **[NO-RESULT]** | Explicitly: no experimental result exists yet for this claim. Any performance number mentioned near a `[NO-RESULT]` tag is illustrative only, not a real measurement. |

**Rule:** Any sentence in any spec file that states a number (dataset
size, F1 score, feature count, etc.) must be traceable to one of these
tags, either inline or via the section it lives in. If a number cannot be
tagged, it must not be written down as fact.

## 5. Success Criteria

There is **no fixed target accuracy or F1 score** for this project. The
success criteria are procedural, not numeric:

1. The combined dataset is correctly harmonized, deduplicated, and
   audited before any model training begins.
2. Train/validation/test splits are leak-free and reproducible.
3. BETC is implemented exactly as specified in `PIPELINE_SPEC.md`.
4. All baselines and ablations in `EXPERIMENT_PLAN.md` are run under the
   identical evaluation protocol in `EVALUATION_PROTOCOL.md`.
5. Results are reported honestly, including cases where BETC does *not*
   outperform a baseline — see `RESEARCH_RULES.md`.
6. Every experimental claim is backed by a saved artifact (metrics file,
   config, seed, model checkpoint/vectorizer) — not just a printed
   number.

## 6. Stakeholders and Audience

This is an individual academic project (course project / thesis-style
work). The primary audience for the final report is an
instructor/supervisor and, potentially, a future publication draft. The
documentation set in this repository should be written so that both a
human reviewer and an AI coding agent can independently verify that the
implementation matches the specification.

## 7. Relationship to the Architecture Document

`BETC_Bangla_Emotion_Architecture_and_Implementation_Plan.docx` (already
produced) is the origin of the BETC architecture (Sections 1–19 of that
document: research objective, architecture overview, math formulation,
step-by-step plan, hyperparameters, pseudocode, experiments, metrics,
baseline table, contribution statement, advantages/failure cases,
complexity analysis, build order, caveat). That document's **pipeline
content is preserved and restated in `PIPELINE_SPEC.md`**. That
document's **dataset section (EmoNoBa alone) is superseded** by
`DATASET_SPEC.md` — see the "Documentation Decisions" note in
`README.md`.
