# RESEARCH_RULES.md

**Read this file in full before writing or modifying any code in this
repository.** These rules apply to any AI coding agent (Claude Code,
Antigravity, Codex, Cursor, or any other automated contributor) and to
human contributors alike. They are not suggestions.

## 1. Architecture Rules

1. **Do not change the BETC architecture** (preprocessing approach,
   word/char TF-IDF configuration, feature fusion method, Classifier
   Chain mechanism, Logistic Regression base estimator, per-class
   threshold optimization) as specified in `PIPELINE_SPEC.md` without
   explicit permission from the project owner recorded in writing (e.g.
   a commit message or an updated spec file with a "Documentation
   Decisions" note, as done in `README.md`).
2. **Do not add pretrained models** of any kind — no pretrained
   Transformers, no pretrained sentence encoders, no pretrained
   Hugging Face checkpoints.
3. **Do not add pretrained embeddings** (Word2Vec, FastText, GloVe, or
   any embedding trained on data outside this project's own training
   split).
4. **Do not call external LLM or embedding APIs** anywhere in the
   pipeline, including for preprocessing, feature generation, or
   evaluation.
5. Deep sequence models (LSTM, BiLSTM, CNN encoders) may appear **only**
   as cited external baseline numbers (`EXPERIMENT_PLAN.md` B6/B7),
   never as code trained inside this repository, unless a future,
   explicit spec change authorizes it.

## 2. Data Integrity Rules

6. **Never leak test information.** TF-IDF vectorizers, chain ordering
   decisions, hyperparameters, and thresholds must be fit/selected using
   only training and/or validation data, exactly as defined in
   `EVALUATION_PROTOCOL.md`.
7. **Never tune on the test set.** No hyperparameter, feature choice, or
   threshold may be selected by observing test-set metrics, even
   informally or "just to check."
8. **Never change labels silently.** Any change to the label mapping
   (`configs/label_mapping.yaml`) must be logged with a reason and must
   go through the Phase 3 decision process in `IMPLEMENTATION_PLAN.md`.
9. **Never fill missing/unannotated labels with 0 silently.** The only
   exception is the explicitly finalized EmoNoBa harmonization rule in
   `DATASET_SPEC.md`: its missing native `disgust` annotation is
   operationally encoded as 0 solely to construct the fixed six-label BETC
   matrix. This assumption must be logged and must never be described as
   an original annotation.
10. **Never delete data silently.** Any row removed (during
    deduplication, Banglish exclusion, or malformed-row cleanup) must be
    counted and logged in the relevant `logs/dataset_audit/` file, with
    the removal criterion stated explicitly.
11. **Raw data (`data/raw/`) is read-only.** No script may overwrite,
    edit, or delete files under `data/raw/` after Phase 1 of
    `IMPLEMENTATION_PLAN.md`.

## 3. Experimental Integrity Rules

12. **Never fabricate results.** Every metric reported anywhere (a
    report, a table, a comment, a docstring) must come from an actual
    run whose output is saved in `results/` or `logs/`. If a result does
    not exist yet, say so explicitly — use the **[NO-RESULT]** tag
    convention from `PROJECT_SPEC.md`.
13. **Never fabricate citations.** Any reference to external literature
    (e.g. dataset statistics, benchmark numbers) must be traceable to an
    actual source. If uncertain about a citation, mark it **[VERIFY]**
    rather than presenting it as confirmed.
14. **Never claim BETC performs better** than a baseline, an ablation
    variant, or an external benchmark **until an actual experiment,
    under the identical protocol in `EVALUATION_PROTOCOL.md`,
    demonstrates it.** Even after such an experiment, report the
    specific numbers and their uncertainty (seed variance), not just a
    qualitative claim.
15. **Never change hyperparameters without logging them.** Every run's
    exact configuration must be saved as a config file
    (`configs/betc_full.yaml`, `configs/ablations/*.yaml`, or
    `configs/baselines/*.yaml`) alongside its result.
16. **Never mix baseline and proposed-model results** in a single
    unlabeled table. Every results table or file must carry a category
    field distinguishing `baseline` / `betc_full` / `ablation` /
    `final_experiment`, per `EXPERIMENT_PLAN.md` Section 5.
17. **Never cherry-pick seeds.** Multi-seed experiments
    (`EXPERIMENT_PLAN.md` F3) must report all seed results and their
    mean ± standard deviation. Excluding a seed's result is only
    acceptable for a documented technical failure (crash, non-
    convergence), never for a low score.
18. **Never claim statistical significance without a test.** A
    difference between two configurations' mean metrics is not
    automatically "significant" — use the paired testing approach in
    `EVALUATION_PROTOCOL.md` Section 5 before making such a claim, and
    state the test used.

## 4. Dataset-Specific Rules

19. **The six target emotions are fixed.** BETC must output exactly
    `anger, disgust, fear, joy, sadness, surprise`. Do not add `love`,
    `contempt`, or any seventh source-specific label.
20. **Do not silently reinterpret source annotations.** EmoNoBa `love` is
    excluded from the target space and is not mapped to `disgust`. Because
    EmoNoBa lacks a native `disgust` annotation, the project uses the
    explicit operational rule `disgust=0` for EmoNoBa rows to construct a
    complete six-column matrix. This is a harmonization assumption, NOT an
    original human annotation, and must be stated in the harmonization
    report and final research write-up.
21. **UBMEC is single-label.** Convert its one categorical emotion into a
    six-column binary representation; do not claim that UBMEC was
    originally multi-label.
22. **MONOVAB non-target labels are excluded.** In particular, `contempt`
    must not become a BETC output label.
23. **Do not treat `[VERIFY]`-tagged literature numbers as confirmed
    facts about this project's own data.** Every such number must be
    independently re-measured from the actual downloaded files
    (`DATASET_SPEC.md` Section 6) before being used in any config,
    report, or code comment as if it were a measured fact.
24. **Do not proceed past a BLOCKING item** listed in `DATASET_SPEC.md`
    Section 6 without resolving it. If an agent encounters a blocking
    item it cannot resolve on its own (e.g. it requires a project-owner
    decision or a license check), it must stop and surface the item
    explicitly rather than guessing or proceeding around it.

## 5. Process Rules

25. **Follow the phase order in `IMPLEMENTATION_PLAN.md` exactly.** Do
    not begin model implementation (Phase 6 onward: preprocessing,
    features, models) before the dataset pipeline (Phases 0–5) has
    passed its validation checks.
26. **Do not proceed to the next phase if the current phase fails its
    completion criteria.** Fix the failure within the current phase
    first.
27. **Every phase's outputs must be saved as artifacts**, not left as
    in-memory objects in a notebook. Essential logic belongs in `src/`,
    not only in `notebooks/`.
28. **Any deviation from this document set must be recorded**, not made
    silently. If an implementer needs to deviate from a spec file (e.g.
    a library API differs from what was assumed), update the relevant
    spec file with a note explaining the change and why, following the
    "Documentation Decisions" pattern used in `README.md`.

## 6. Summary Checklist for Any AI Coding Agent Before Committing Code

- [ ] Does this change alter the BETC architecture? If yes, is it
      explicitly authorized?
- [ ] Does this change introduce any pretrained model, embedding, or
      external API call? If yes, stop — this is prohibited.
- [ ] Does this change touch test data before the intended evaluation
      step? If yes, stop — this is leakage.
- [ ] Does this change alter labels, drop rows, or fill missing values
      without a logged justification? If yes, stop and log it first.
- [ ] Does this change report a number that is not backed by a saved
      artifact? If yes, remove the number or run the experiment first.
- [ ] Does this change mix result categories (baseline/full/ablation/
      final) in one table? If yes, fix the labeling before proceeding.
- [ ] Does this change resolve a BLOCKING dataset item on the agent's
      own authority? If yes, stop and surface it instead.
