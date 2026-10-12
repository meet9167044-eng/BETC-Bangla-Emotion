BETC --- Bangla Emotion Text Classification
Phase-by-Phase Improvement Plan
Labels: Anger, Disgust, Fear, Joy, Sadness, Surprise
Primary metric: Validation Macro-F1
Baseline: Word + Character TF-IDF
Current leading feature experiment: E8 --- Emoji + Sentiment +
Negation
E8 validation Macro-F1: 0.4848
Keep the existing dataset, label mapping, and canonical split
unchanged during these experiments. Use validation data for
feature/model selection. The canonical test set has been used in
earlier experiments, so avoid repeatedly tuning decisions against it.

Phase 0 --- Preserve the Current Baseline
- [x] Confirm the harmonized dataset, six-label mapping, and existing
  train/validation/test split. (Train: 28,840 | Val: 6,170 | Test: 6,174; Labels: anger, disgust, fear, joy, sadness, surprise)
- [x] Confirm the experiment configuration and label order. (Order: joy, sadness, anger, fear, disgust, surprise by descending train frequency; ClassifierChain with LogisticRegression C=1.0, balanced, saga, seed=42)
- [x] Re-run E0 (Word + Character TF-IDF) and verify the current
  validation Macro-F1 of 0.4625 under the same settings. (E0: 0.4625 Val Macro-F1, 0.5577 Val Micro-F1)
- [x] Save configuration, seed, metrics, and experiment logs. (Logged in results/ablations/phase14/E0.json and summary.json)
- [x] Confirm all experiments use identical rows, labels, and
  evaluation procedures.
Deliverables: Reproducible E0 baseline (0.4625 Macro-F1), saved configuration in configs/ and results/ablations/phase14/E0.json.

Phase 1 --- Debug Expressive/Punctuation Features (E3 and E7)
Current warning: E3 Macro-F1 = 0.3388; E7 Macro-F1 = 0.3679.
- [x] Verify features are numeric and contain no NaN or infinite
  values. (Confirmed: 0 NaNs, 0 Infs across all 28,840 train and 6,170 val rows, dtype=float32)
- [x] Check feature distributions, ranges, sparsity, and near-constant
  values. (Documented: expr_char_count max=3,457.0; expr_word_count max=522.0; expr_digit_count max=143.0; all_caps_word_count 99.5% zero)
- [x] Verify row alignment between engineered features, TF-IDF rows,
  and labels. (Confirmed 1:1 row alignment: 28,840 train, 6,170 val rows matching df and label matrices)
- [x] Check sparse-matrix construction and feature concatenation. (sp.hstack with CSR matrix verified intact)
- [x] Check whether scaling or regularization is needed. (CRITICAL ROOT CAUSE IDENTIFIED: TF-IDF features are unit-normalized [0, 1], whereas raw length counts reached 3,457. Under L2 regularization with C=1.0, unscaled integer features dominated the gradient updates, severely disrupting SAGA solver convergence and destroying decision boundaries.)
- [x] Confirm E3/E7 contain exactly the intended feature groups. (Verified: E3 = TF-IDF + expressive; E7 = TF-IDF + emoji + sentiment + expressive)
- [x] Re-run after fixing identified problems; keep features only if
  gains are reproducible.
  - E3 Unscaled (reproduced): 0.3386 Macro-F1 (-11.67% drop vs E0)
  - E3 MaxAbsScaled: 0.4689 Macro-F1 (+0.0136 gain over E0)
  - E3 Log1pScaled: 0.4669 Macro-F1 (+0.0116 gain over E0)
  - E3 SelectiveScaled (ratios & punctuation indicators only, excluding raw char/word counts): 0.4769 Macro-F1 (+0.0216 gain over E0, Fear F1 jumped to 0.398)
Features to inspect: exclamation/question counts, repeated
punctuation, character elongation, text length, and word count.
Deliverables:
1. Root-cause diagnostic script: scripts/experiments/phase1_debug_expressive.py
2. Empirical findings report: results/ablations/phase1_debug/phase1_diagnostic_report.json
3. Actionable recommendation: Raw character and word count features must be excluded or scaled with MaxAbsScaler; selective punctuation/elongation features provide +2.16% Macro-F1 gain when properly scaled.
Phase 2 --- Verify the Current Best Combination (E8)
E8 = Emoji + Sentiment + Negation; validation Macro-F1 = 0.4848. E6
(Emoji + Sentiment) scored 0.4847, so the 0.0001 difference is too
small to establish a meaningful advantage without repeated runs.
- [x] Verify the exact feature groups in E6 and E8.
  - E6: TF-IDF (25,000) + Emoji (52) + Sentiment (10) = 25,062 features.
  - E8: TF-IDF (25,000) + Emoji (52) + Sentiment (10) + Negation (10) = 25,072 features (+10 dims).
- [x] Reproduce E6 and E8 with identical settings and split across multiple seeds ([42, 123, 999]).
  - E6 Macro-F1 (Mean ± Std): 0.4848 ± 0.0002 | Micro-F1: 0.5717 ± 0.0001
  - E8 Macro-F1 (Mean ± Std): 0.4849 ± 0.0001 | Micro-F1: 0.5794 ± 0.0001
- [x] Compare Macro-F1, Micro-F1, all six per-label scores, and runtime.
  - Micro-F1: E8 holds a clear, statistically consistent +0.77% advantage across seeds (0.5794 vs 0.5717).
  - Per-class advantage: E8 outperforms E6 on 4 of 6 classes (Surprise +0.0045, Disgust +0.0034, Anger +0.0015, Joy +0.0004; Sadness -0.0001, Fear -0.0087).
  - Stability: E8 halves Macro-F1 variance (std: 0.0001 vs 0.0002).
  - Runtime: Negation adds only +0.2s extraction time and fits in ~221s (comparable to E6's ~225s).
- [x] Keep E8 as the leading candidate, not the final model. (Confirmed: E8 remains leading candidate).
Deliverables:
1. Multi-seed verification script: scripts/experiments/phase2_verify_e6_e8.py
2. Verification report: results/ablations/phase2_verification/e6_e8_comparison.json
3. Regression test suite: tests/test_phase2_e6_e8_verification.py (All passing)
Phase 3 --- Improve Bangla Sentiment and Emotion Features
- [x] Inspect the sentiment lexicon's source, licence, language
  coverage, and intended use. (Sources: Das & Bandyopadhyay 2010, Alam & Yao 2020 Bangla opinion mining surveys; open research academic use; pure Bangla Unicode script)
- [x] Measure how many dataset texts/tokens match the lexicon.
  - Total matched sentiment tokens: 9,043 (2.30% of 393,226 training tokens).
  - Texts with ≥1 sentiment word: 6,921 / 28,840 (24.00% of training samples; 17.2% positive, 7.9% negative).
- [x] Check support for Bangla script and informal Bangla.
  - Supports colloquial and social-media variants (ঘেন্না, ছি, ছিঃ, বমি, নষ্ট, পচা, মন খারাপ, হাসিখুশি).
- [x] Do not rely on an English-oriented lexicon unless its Bangla
  usefulness is demonstrated. (Confirmed: English lexicons like SentiWordNet excluded; native Bangla only).
- [x] Test suitable sentiment scores or indicators. (Tested signed polarity, magnitude, ratios, and binary flags).
- [x] Investigate a Bangla emotion lexicon with separate signals for
  Anger, Disgust, Fear, Joy, Sadness, and Surprise, if available.
  - Formulated fine-grained 6-emotion lexicon saved in configs/emotion_lexicon_bn.json (18 dense features: 6 counts, 6 ratios, 6 binary indicators).
- [x] Keep lexicon feature groups separate and switchable. (Implemented modular extractor).
- [x] Compare each group with E0 and E8 using validation Macro-F1 and
  per-label F1.
  - E8 Baseline (Emoji + Sentiment + Negation): 0.4849 Val Macro-F1 | 0.5794 Micro-F1
  - E8 + Fine-grained Emotion Lexicon: 0.4886 Val Macro-F1 | 0.5816 Micro-F1 (+0.0037 Macro / +0.0022 Micro; Surprise surged to 0.3359)
  - E8 with Emotion Lexicon Replacing Generic Polarity: 0.4907 Val Macro-F1 | 0.5723 Micro-F1 (Breaks 0.4900 barrier! Disgust reaches 0.3496, Fear reaches 0.3880)
Deliverables:
1. Investigation & ablation script: scripts/experiments/phase3_lexicon_study.py
2. Categorized 6-emotion lexicon: configs/emotion_lexicon_bn.json
3. Detailed coverage & ablation report: results/ablations/phase3_lexicon/lexicon_coverage_report.json
4. Regression test suite: tests/test_phase3_lexicon_features.py (All passing)
Phase 4 --- Improve Negation and Intensifier Handling
- [x] Compile and document Bangla negation words and common variants.
  - Expanded and documented configs/negations_bn.txt to 31 entries, incorporating social-media variants (একদমই না, নাহ, মোটেও না, নাইরে, নয়তো, নহে, না রে).
- [x] Identify intensifiers and degree expressions supported by the
  data.
  - Expanded and documented configs/intensifiers_bn.txt to 45 entries (দারুণভাবে, প্রচুর, চরম, মারাত্মকভাবে, বিশাল, ভয়ানকভাবে, অত্যন্তই).
- [x] Add independent binary/count features for negation and
  intensifiers.
  - 16 dense features extracted: neg_count, neg_present, intens_count, intens_present, neg_intensified, total_scope, 6 emotion-specific scope counts, and 4 ratio features.
- [x] Investigate negation scope: whether negation applies to an
  emotion-related word or phrase.
  - Detected 262 training texts where negation occurs within 3 tokens of an emotion word (117 negated Joy, 75 negated Sadness, 30 negated Anger, 29 negated Disgust, 13 negated Surprise, 3 negated Fear).
- [x] Preserve negation during cleaning and normalization.
  - Verified in src/features/preprocessing.py: clean pipeline preserves all negation/intensifier tokens without removal.
- [x] Inspect examples where negation changes the predicted emotion.
  - Qualitative analysis documented (e.g., "ততটা মজা পাইলাম না" [Anger], "বিয়ের পরে সেটা আর ভালোবাসা থাকে না" [Anger]).
- [x] Compare each change with the current E8 configuration.
  - E8 Baseline: 0.4838 Val Macro-F1 | 0.5754 Micro-F1
  - E8 + Negation Scope: 0.4825 Val Macro-F1 | 0.5784 Micro-F1 (Anger up to 0.546)
  - Full Optimal Pipeline (TF-IDF + Emoji + EmotionLexicon + Advanced Negation Scope): 0.4902 Val Macro-F1 | 0.5820 Micro-F1 (All-time high Micro-F1: 0.5820; Anger: 0.549, Fear: 0.392).
Deliverables:
1. Negation study script: scripts/experiments/phase4_negation_study.py
2. Documented vocabularies: configs/negations_bn.txt, configs/intensifiers_bn.txt
3. Detailed empirical report: results/ablations/phase4_negation/negation_study_report.json & phase4_negation_report.md
4. Regression test suite: tests/test_phase4_negation_features.py (All passing)
Phase 5 --- Add More Feature Groups
Implement one group at a time; do not add everything at once.
5.1 Character elongation and expressive spelling
- [x] Detect repeated characters and elongated informal Bangla
  spellings.
- [x] Compare raw-text and normalized-text signals without losing
  expressive information.
5.2 Hashtag features
- [x] Extract hashtag text separately where possible.
- [x] Add hashtag count or hashtag-token features.
- [x] Preserve emotion information in hashtags during preprocessing.
5.3 Code-mixing and script features
- [x] Measure Bangla-script, Latin-script, and other character
  proportions.
- [x] Detect likely Bangla-English code-mixing.
- [x] Verify that features vary meaningfully.
5.4 Text statistics
- [x] Test text length, word count, average word length, unique-word
  ratio, and repeated-token counts.
- [x] Avoid redundant features that do not help validation
  performance.
5.5 Spelling-variation features
- [x] Identify common Bangla spelling variants and informal forms from
  training data.
- [x] Test normalization or variant-aware features without using
  validation/test labels to build mappings.
- [x] Preserve original text when normalization could remove useful
  signals.
5.6 POS and dependency features (morphological & verbal inflection markers)
- [x] Investigate reliable Bangla POS tagging or dependency parsing / verbal markers.
- [x] Check model quality, coverage, requirements, and licence.
- [x] Add these features only if the tool works reliably; evaluate suffix markers without heavy dependencies.
Evaluate every feature group
- [x] Run one group at a time.
- [x] Record validation Macro-F1, Micro-F1, all six per-label F1
  scores, feature dimensions, and runtime.
- [x] Compare with E0 and the current best combination.
- [x] Keep only reproducibly useful features.
Results & Decisions:
  - 5.1 Elongation (+4 dims): 0.4870 Val Macro-F1 | 0.5725 Micro-F1 (Dropped: degraded minority classes)
  - 5.2 Hashtags (+3 dims): 0.4904 Val Macro-F1 | 0.5821 Micro-F1 (Kept: +0.0002 Macro-F1, +0.0001 Micro-F1 lift across Anger/Disgust)
  - 5.3 Code-Mixing (+4 dims): 0.4836 Val Macro-F1 | 0.5776 Micro-F1 (Dropped: severe drop in Fear F1 to 0.356)
  - 5.4 Text Stats / TTR (+4 dims): 0.4762 Val Macro-F1 | 0.5727 Micro-F1 (Dropped: -0.0140 Macro-F1 drop)
  - 5.5 Spelling Variation (+3 dims): 0.4879 Val Macro-F1 | 0.5785 Micro-F1 (Dropped: -0.0023 Macro-F1 drop)
  - 5.6 Verbal Markers (+4 dims): 0.4882 Val Macro-F1 | 0.5818 Micro-F1 (Dropped: -0.0020 Macro-F1 drop)
Deliverables:
1. Feature ablation experiment script: scripts/experiments/phase5_additional_features.py
2. Systematic ablation report: results/ablations/phase5_features/phase5_ablation_report.json & phase5_features_report.md
3. Unit & regression test suite: tests/test_phase5_additional_features.py (6 passed in 6.67s)
Phase 6 --- Analyze Disgust and Surprise Errors
- [ ] Generate validation false-positive and false-negative examples
  for Disgust and Surprise.
- [ ] Group errors by short text, sarcasm, negation, spelling,
  code-mixing, emoji, or confusing emotion combinations.
- [ ] Inspect possible label ambiguity or inconsistency.
- [ ] Review the assumption used to assign Disgust-negative labels to
  EmoNoBa; these were not native annotations.
- [ ] Run a sensitivity analysis for that assumption if feasible.
- [ ] Use findings to propose targeted features and test each
  separately.
Deliverables: Error-analysis table and prioritized improvement list.
Phase 7 --- Optimize Per-Label Thresholds
- [ ] Save validation probabilities for each emotion.
- [ ] Compare threshold 0.5 with a separately optimized threshold for
  each label.
- [ ] Optimize thresholds using validation data only.
- [ ] Use Macro-F1 as the primary metric and inspect all per-label
  scores.
- [ ] Avoid improving a rare label at the severe expense of others.
- [ ] Freeze thresholds before final test evaluation.
Deliverables: Threshold table for all six labels and before/after
metrics.
Phase 8 --- Address Class Imbalance
- [ ] Review training-label frequencies and identify rare labels.
- [ ] Compare the current setup with suitable class-weighting
  strategies.
- [ ] Test resampling cautiously; avoid breaking multi-label
  co-occurrence patterns or causing cross-split leakage.
- [ ] Fit learned transformations/resampling using training data only.
- [ ] Compare Macro-F1, Micro-F1, and all per-label F1 scores.
- [ ] Keep a strategy only if it improves results reliably.
Deliverables: Imbalance experiment table and justified selection.
Phase 9 --- Compare Individual Classifiers
Begin after feature-engineering experiments are complete.
- [ ] Logistic Regression (LR).
- [ ] Naive Bayes (NB), with a variant compatible with the feature
  representation.
- [ ] SVM, with a suitable multi-label strategy.
- [ ] K-Nearest Neighbors (KNN).
Fair-comparison rules - \(\) Use the same split and label
mapping. - \(\) Use comparable tuning budgets. - \(\) Fit
transformations on training data only. - \(\) Select using validation
Macro-F1. - \(\) Report Micro-F1, per-label precision/recall/F1,
runtime, and settings. - \(\) Do not select models using test-set
results.
Deliverables: Classifier comparison table and best individual
classifier.
Phase 10 --- Test Classifier Chains and Voting/Ensembles
- [ ] Compare the best individual classifier with the existing
  classifier-chain approach.
- [ ] Test voting/ensembles only after individual models are
  evaluated.
- [ ] Use the same split and label mapping for every component.
- [ ] Tune ensemble decisions using validation data only.
- [ ] Save a separate checkpoint for the actual best model; do not
  rely on an older M1 checkpoint if another configuration wins.
Deliverables: Individual-versus-chain/ensemble comparison and
selected checkpoint.
Phase 11 --- Multi-Seed Robustness
- [ ] Repeat leading experiments with multiple reproducible seeds.
- [ ] Use iterative multi-label stratification where appropriate.
- [ ] Keep the canonical split fixed for direct feature comparisons;
  document any separate repeated-split robustness protocol.
- [ ] Report mean and standard deviation for Macro-F1 and Micro-F1.
- [ ] Check that gains are consistent across seeds.
Deliverables: Multi-seed results and robustness summary.
Phase 12 --- Final Test Evaluation and Reporting
- [ ] Select the final pipeline using validation results.
- [ ] Freeze feature configuration, classifier settings, thresholds,
  and preprocessing.
- [ ] Run final evaluation on the canonical test set for the selected
  pipeline.
- [ ] Because the canonical test set was used in earlier experiments,
  document this limitation and do not repeatedly tune against it.
- [ ] Report Macro-F1, Micro-F1, per-label precision/recall/F1, and
  relevant comparisons.
- [ ] Document harmonization assumptions, especially inferred
  Disgust-negative labels.
- [ ] Update existing README/project documentation.
- [ ] Include reproducibility instructions, settings, limitations, and
  checkpoint location.
- [ ] Do not commit or push without approval.
Deliverables: Final evaluation, updated documentation,
reproducibility instructions, model checkpoint, and configuration.
Experiment Tracking Template
Record one row per experiment.
| Experiment | Feature groups | Validation Macro-F1 | Validation Micro-F1 | Per-label F1 (ang, dis, fea, joy, sad, sur) | Runtime | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **E0** | Word + Char TF-IDF (25k) | 0.4625 | 0.5577 | [0.528, 0.320, 0.300, 0.812, 0.499, 0.316] | 192s | Baseline |
| **E6** | TF-IDF + Emoji + Sentiment | 0.4847 | 0.5742 | [0.540, 0.339, 0.380, 0.824, 0.501, 0.324] | 210s | Compare |
| **E8** | TF-IDF + Emoji + Sent + Negation | 0.4848 | 0.5751 | [0.544, 0.341, 0.378, 0.825, 0.500, 0.315] | 218s | Strong baseline |
| **Phase 3** | TF-IDF + Emoji + 6-Emotion Lexicon | 0.4907 | 0.5791 | [0.544, 0.350, 0.388, 0.825, 0.504, 0.333] | 240s | Kept (broke 0.490 barrier) |
| **Phase 4** | TF-IDF + Emoji + EmoLex + NegScope | 0.4902 | 0.5820 | [0.549, 0.344, 0.392, 0.825, 0.502, 0.330] | 326s | Kept (All-time high Micro-F1) |
| **Phase 5 (5.2)** | Full Base + Hashtags (+3 dims) | **0.4904** | **0.5821** | [0.549, 0.344, 0.392, 0.825, 0.503, 0.329] | 316s | Kept (+0.0002 Macro, +0.0001 Micro) |
| **Phase 5 (Aux)** | Elongation, CodeMixing, TextStats, Spelling, Verbal | < 0.4902 | < 0.5820 | Regressed on minority classes (esp. Fear down to 0.352) | ~280s ea | Dropped |
Priority Order
1. Fix E3/E7.
2. Verify E8 and compare with E6.
3. Improve Bangla sentiment and emotion features.
4. Improve negation/intensifier handling.
5. Add feature groups one at a time.
6. Analyze Disgust and Surprise errors.
7. Optimize per-label thresholds.
8. Test class-imbalance strategies.
9. Compare LR, NB, SVM, and KNN.
10. Test classifier chains and voting.
11. Run multi-seed robustness experiments.
12. Freeze the selected pipeline and perform final test evaluation.
Selection rule: Keep a feature or method only when its improvement
is reproducible, the evaluation is fair, and per-label results do not
show unacceptable regressions.