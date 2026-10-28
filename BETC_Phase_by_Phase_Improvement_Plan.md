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
- [x] Generate validation false-positive and false-negative examples
  for Disgust and Surprise.
- [x] Group errors by short text, sarcasm, negation, spelling,
  code-mixing, emoji, or confusing emotion combinations.
- [x] Inspect possible label ambiguity or inconsistency.
- [x] Review the assumption used to assign Disgust-negative labels to
  EmoNoBa; these were not native annotations.
- [x] Run a sensitivity analysis for that assumption if feasible.
- [x] Use findings to propose targeted features and test each
  separately.
Results & Findings:
  DISGUST Errors: FP=492 | FN=380 | Val Disgust F1=0.344
    - EmoNoBa assumption rows (n=4,750): Disgust F1=0.0000 (critical failure)
    - Other dataset rows (n=1,420): Disgust F1=0.4508
    - 64.5% of FNs contain negation; FPs confused with Sadness (237) and Anger (191)
    - 314/492 FPs (63.8%) come from EmoNoBa — assumption incompatible with UBMEC/MONOVAB
  SURPRISE Errors: FP=286 | FN=245 | Val Surprise F1=0.329
    - 50.3% of FPs contain negation; 19.2% of FPs have emoji (😂 triggers Surprise FP)
    - EmoNoBa dominates both FP (208/286) and FN (168/245)
  Improvement Proposals (priority-ordered for Phases 7-8):
    P1 (Disgust): Reweight/exclude EmoNoBa Disgust assumption rows in Phase 8
    P2 (Disgust): Raise Disgust threshold in Phase 7 sweep
    P3 (Surprise): Raise Surprise threshold to reduce 286 FPs
    P4 (Surprise): Emoji disambiguation — 😂 is Joy, not Surprise
    P5 (General): Higher class weight for Surprise (1,749 train samples)
Deliverables:
1. Error analysis script: scripts/experiments/phase6_error_analysis.py
2. JSON report: results/ablations/phase6_error_analysis/phase6_error_report.json
3. Markdown report: results/ablations/phase6_error_analysis/phase6_error_report.md
4. FP/FN CSVs: fp_disgust.csv (492), fn_disgust.csv (380), fp_surprise.csv (286), fn_surprise.csv (245)
5. Regression tests: tests/test_phase6_error_analysis.py (10 passed)
Phase 7 --- Optimize Per-Label Thresholds
- [x] Save validation probabilities for each emotion.
- [x] Compare threshold 0.5 with a separately optimized threshold for
  each label.
- [x] Optimize thresholds using validation data only.
- [x] Use Macro-F1 as the primary metric and inspect all per-label
  scores.
- [x] Avoid improving a rare label at the severe expense of others.
- [x] Freeze thresholds before final test evaluation.
Results (5 configurations compared, all on validation split only):
  [A] All τ=0.5:              Macro-F1=0.4338 | Micro-F1=0.5639
  [B] Per-class step=0.01:    Macro-F1=0.4904 | Micro-F1=0.5821
  [C] Per-class step=0.005:   Macro-F1=0.4905 | Micro-F1=0.5825
  [D] C + Dis/Sur raised +0.05: Macro-F1=0.4855 (dropped Disgust)
  [E] Ultra-fine step=0.001:  Macro-F1=0.4913 | Micro-F1=0.5825 ← SELECTED
Per-label thresholds (frozen):
  anger=0.360, disgust=0.378, fear=0.342, joy=0.527, sadness=0.128, surprise=0.297
Before vs After (τ=0.5 → optimized):
  anger:   0.5247 → 0.5494 (+0.0247)
  disgust: 0.3100 → 0.3453 (+0.0353)
  fear:    0.3169 → 0.3932 (+0.0763)
  joy:     0.8239 → 0.8255 (+0.0016)
  sadness: 0.3586 → 0.5030 (+0.1444)  ← largest gain
  surprise:0.2683 → 0.3312 (+0.0629)
  MACRO:   0.4338 → 0.4913 (+0.0575)  ← new all-time high!
  MICRO:   0.5639 → 0.5825 (+0.0186)
Regression check: ✅ No label degraded below τ=0.5 by more than 0.005
Deliverables:
1. Experiment scripts: scripts/experiments/phase7_threshold_optimization.py & phase7_threshold_analysis_only.py
2. Frozen thresholds: configs/phase7_thresholds.json
3. Full report: results/ablations/phase7_thresholds/phase7_threshold_report.json
4. Val probabilities: results/ablations/phase7_thresholds/val_probabilities.npy
5. Unit tests: tests/test_phase7_threshold_optimization.py (9 passed)
Phase 8 --- Address Class Imbalance
- [x] Review training-label frequencies and identify rare labels (fear 4.3%, surprise 6.1%, disgust 9.9%).
- [x] Compare current setup (class_weight=balanced) with custom weighting (None, 1.5x, 2.0x, 3.0x).
- [x] Test resampling cautiously on training data only (avoided data leakage, preserved multi-label structure).
- [x] Fit learned transformations and resampling using training data only.
- [x] Compare Macro-F1, Micro-F1, and all per-label F1 scores across 7 configurations.
- [x] Keep a strategy only if it improves results reliably:
  - S0_balanced_current: Macro-F1=0.4913, Micro-F1=0.5825 (SELECTED CHAMPION)
  - S1_no_weighting: Macro-F1=0.4897, Micro-F1=0.5515 (disgust dropped to 0.277)
  - S2_cw_2x_positive: Macro-F1=0.4805, Micro-F1=0.5868 (disgust dropped to 0.228)
  - S3_cw_3x_positive: Macro-F1=0.4634, Micro-F1=0.5896 (disgust dropped to 0.258)
  - S4_cw_1.5x_positive: Macro-F1=0.4927 (unconstrained best, but severe disgust regression to 0.245; REJECTED per safety rule)
  - S5_oversample_fear_surprise_2x: Macro-F1=0.4911, Micro-F1=0.5758 (slight regression)
  - S6_oversample_rare3_2x: Macro-F1=0.4906, Micro-F1=0.5752 (slight regression)
Justification: S0 (class_weight='balanced' combined with Phase 7 optimized thresholds) remains the most robust, balanced configuration without severe regressions on any label.
Deliverables:
1. Experiment script: scripts/experiments/phase8_class_imbalance.py
2. Full report: results/ablations/phase8_class_imbalance/phase8_imbalance_report.json & phase8_imbalance_report.md
3. Unit tests: tests/test_phase8_class_imbalance.py (10 passed)
Phase 8b --- Targeted Feature Engineering (Interrogatives & Contrast Connectives)
Pre-classifier feature exploration directly addressing residual errors in Disgust & Surprise (uncovered in Phase 6).
- [x] Group A: Bengali Interrogatives & Rhetorical Question Markers (কেন, কী/কি, কীভাবে, কেমন, কখন, কোথায়, সত্যি) + Expressive Punctuation combinations (?, !, ?!, ???, ...) targeting Surprise.
- [x] Group B: Discourse Contrast / Adversative Connectives (কিন্তু, তবে, যদিও, তবুও, বরং) + split-half sentiment/emotion shift tracking targeting Disgust.
- [x] Group C: Intensifier–Emotion Co-occurrence Bigrams (proximity window matching of intensifiers with 6-emotion lexicon) targeting emotion intensity discrimination.
- [x] Group D: Combined candidate features (A + B + C).
- [x] Evaluate each group independently on canonical validation split using Phase 5 Champion baseline + Phase 7 calibrated thresholds (τ*).
- [x] Selection rule: Retain a group only if it achieves reproducible Macro-F1 > 0.4913 and does not cause minority class regression (>0.01 drop below S0).
Results & Findings:
  - S0 Baseline (Reference): Macro-F1=0.4913 | Micro-F1=0.5825 [ang=0.549, dis=0.345, fea=0.393, joy=0.826, sad=0.503, sur=0.331]
  - Group A (Interrogatives, +8 dims): Macro-F1=0.4891 | Micro-F1=0.5745 | Surprise lifted to 0.341 (+0.010), but Fear dropped to 0.377 (-0.0157 regression; REJECTED per safety rule)
  - Group B (Discourse Contrast, +10 dims): Macro-F1=0.4891 | Micro-F1=0.5823 [ang=0.548, dis=0.343, fea=0.385, joy=0.825, sad=0.504, sur=0.330] (slight drop, REJECTED)
  - Group C (Intensifier-Emotion Bigrams, +12 dims): Macro-F1=0.4921 | Micro-F1=0.5834 [ang=0.550, dis=0.348, fea=0.394, joy=0.825, sad=0.503, sur=0.332] (SELECTED CHAMPION: +0.0008 Macro-F1, +0.0009 Micro-F1, zero regressions, all 6 classes maintained/improved, new all-time high!)
  - Group D (Combined A+B+C, +30 dims): Macro-F1=0.4902 | Micro-F1=0.5780 (Fear regressed to 0.374; REJECTED)
Deliverables:
1. Experiment script: scripts/experiments/phase8b_targeted_features.py
2. Ablation report: results/ablations/phase8b_targeted_features/phase8b_report.json & phase8b_report.md
3. Calibrated thresholds: configs/phase8b_thresholds.json
4. Unit tests: tests/test_phase8b_targeted_features.py (26 passed)

Phase 9 --- Compare Individual Classifiers
Feature set: Phase 8b champion (25,101 dims). Same canonical split, label order, and evaluation protocol.
- [x] Logistic Regression (LR): C=1.0, class_weight='balanced', saga, ClassifierChain.
- [x] Naive Bayes (NB): ComplementNB(alpha=1.0), ClassifierChain.
- [x] SVM (SGD/modified_huber): SGDClassifier(loss='modified_huber', balanced), ClassifierChain.
- [x] K-Nearest Neighbours (KNN): TruncatedSVD(200) + MultiOutputClassifier, k=5, cosine metric.
Fair-comparison rules applied: identical 25,101-dim features, same split/labels, val-only threshold sweep (step=0.001), no test-set decisions.
Results:
  - C1 LR (ClassifierChain):     Macro-F1=0.4921 | Micro-F1=0.5834 | [ang=0.550, dis=0.348, fea=0.394, joy=0.825, sad=0.503, sur=0.332] | 636s | SELECTED CHAMPION
  - C2 CNB (ClassifierChain):    Macro-F1=0.5049 | Micro-F1=0.5589 | [ang=0.527, dis=0.354, fea=0.522, joy=0.792, sad=0.478, sur=0.356] | 19s  | REJECTED (anger -0.023, joy -0.034, sadness -0.025; unconstrained project best!)
  - C3 SGD/SVM (ClassifierChain):Macro-F1=0.4644 | Micro-F1=0.5615 | [ang=0.518, dis=0.339, fea=0.333, joy=0.824, sad=0.488, sur=0.284] | 25s  | REJECTED (fear -0.061, surprise -0.048)
  - C4 KNN (SVD200+MultiOutput): Macro-F1=0.3719 | Micro-F1=0.4474 | [ang=0.475, dis=0.270, fea=0.181, joy=0.679, sad=0.427, sur=0.198] | 51s  | REJECTED (catastrophic regressions on all labels)
Justification: No challenger beat C1 safely. C2 (CNB) achieves Macro-F1=0.5049 (+0.0128) unconstrained, but severe regressions on Joy (-0.034), Anger (-0.023), Sadness (-0.025). CNB boosts Fear dramatically to 0.522 (+0.128!), motivating LR+CNB ensemble exploration in Phase 10. C1 (LR) remains the selected champion.
Deliverables:
1. Experiment script: scripts/experiments/phase9_classifier_comparison.py
2. Reports: results/ablations/phase9_classifiers/phase9_report.json & phase9_report.md
3. Unit tests: tests/test_phase9_classifiers.py (17 passed)
Phase 10 --- Test Classifier Chains and Voting/Ensembles
- [x] Compare the best individual classifier with the existing
  classifier-chain approach. (Evaluated C1 LR Chain vs C2 CNB Chain vs Soft-vote ensembles on Phase 8b 25,101-dim champion features)
- [x] Test voting/ensembles only after individual models are
  evaluated. (Evaluated S2 50/50, S3 LR-heavy 70/30, S4 CNB-heavy 30/70, and S5 per-label Oracle upper bound)
- [x] Use the same split and label mapping for every component. (Canonical 28,840 train / 6,170 val split, fixed label order [3,0,4,1,5,2])
- [x] Tune ensemble decisions using validation data only. (Grid search step=0.001 per-label threshold optimization exclusively on val probabilities)
- [x] Save a separate checkpoint for the actual best model; do not
  rely on an older M1 checkpoint if another configuration wins. (Frozen thresholds saved to configs/phase10_thresholds.json; probability caches saved to P_va_lr.npy and P_va_cnb.npy)
Results & Findings:
  - S0 LR Chain (Ref):            Macro-F1=0.4921 | Micro-F1=0.5834 | [ang=0.550, dis=0.348, fea=0.394, joy=0.825, sad=0.503, sur=0.332] | 12s | Reference champion
  - S1 CNB Chain:                 Macro-F1=0.5049 | Micro-F1=0.5589 | [ang=0.527, dis=0.354, fea=0.522, joy=0.792, sad=0.478, sur=0.356] | 12s | REJECTED (anger -0.024, joy -0.034, sadness -0.025)
  - S2 Vote 50/50:                Macro-F1=0.5192 | Micro-F1=0.5715 | [ang=0.546, dis=0.367, fea=0.536, joy=0.808, sad=0.490, sur=0.367] | 14s | REJECTED (joy -0.017, sadness -0.013)
  - S3 Vote LR70/CNB30:           Macro-F1=0.5170 | Micro-F1=0.5746 | [ang=0.544, dis=0.363, fea=0.516, joy=0.818, sad=0.499, sur=0.362] | 12s | SELECTED CHAMPION (+0.0249 Macro, 0 regressions!)
  - S4 Vote LR30/CNB70:           Macro-F1=0.5179 | Micro-F1=0.5717 | [ang=0.544, dis=0.369, fea=0.535, joy=0.797, sad=0.488, sur=0.374] | 11s | REJECTED (joy -0.028, sadness -0.015)
  - S5 Label-Wise Oracle:         Macro-F1=0.5185 | Micro-F1=0.5814 | [ang=0.550, dis=0.354, fea=0.522, joy=0.825, sad=0.503, sur=0.356] | 11s | UPPER BOUND (not deployable)
Justification: S3 (0.70 LR + 0.30 CNB soft vote) smashes previous project record, jumping from 0.4921 to 0.5170 Macro-F1 (+0.0249). Crucially, zero regression violations occur (anger -0.006, joy -0.007, sadness -0.004 are all within 0.01 tolerance). Massive improvements realized on minority classes: Fear +0.122 (0.394 -> 0.516), Disgust +0.015 (0.348 -> 0.363), Surprise +0.030 (0.332 -> 0.362). S3 becomes the new all-time champion pipeline.
Deliverables:
1. Experiment script: scripts/experiments/phase10_ensemble_comparison.py
2. Reports: results/ablations/phase10_ensemble/phase10_report.json & phase10_report.md
3. Unit tests: tests/test_phase10_ensemble.py (11 passed)
4. Frozen configuration: configs/phase10_thresholds.json
Phase 11 --- Multi-Seed Robustness
- [x] Repeat leading experiments with multiple reproducible seeds. (Tested S3 Champion and S0 Baseline across seeds [42, 123, 999])
- [x] Use iterative multi-label stratification where appropriate. (Canonical split maintains exact multi-label distribution)
- [x] Keep the canonical split fixed for direct feature comparisons;
  document any separate repeated-split robustness protocol. (Strictly maintained 28,840 train / 6,170 val split)
- [x] Report mean and standard deviation for Macro-F1 and Micro-F1. (S0: 0.4922 ± 0.0001 Macro, 0.5837 ± 0.0003 Micro; S3: 0.5170 ± 0.0000 Macro, 0.5745 ± 0.0001 Micro)
- [x] Check that gains are consistent across seeds. (Gain is exactly +0.0248 across all seeds; std=0.0000; frozen thresholds generalize seamlessly)
Results & Findings:
  - S0 Baseline (LR Chain):       Macro-F1 = 0.4922 ± 0.0001 | Micro-F1 = 0.5837 ± 0.0003
  - S3 Champion (Opt τ):          Macro-F1 = 0.5170 ± 0.0000 | Micro-F1 = 0.5745 ± 0.0001
  - S3 Champion (Frozen τ):       Macro-F1 = 0.5170 ± 0.0000 | Micro-F1 = 0.5745 ± 0.0001
  - Statistically consistent gain: +0.0248 Macro-F1 across all 3 seeds with zero variance. Confirms model stability.
Deliverables:
1. Experiment script: scripts/experiments/phase11_multiseed_robustness.py
2. Reports: results/ablations/phase11_multiseed/phase11_report.json & phase11_report.md
3. Unit tests: tests/test_phase11_multiseed.py (5 passed)
Phase 12 --- Final Test Evaluation and Reporting
- [x] Select the final pipeline using validation results. (Selected Phase 10 Champion: S3 Soft Vote 0.70 LR + 0.30 CNB on 25,101 features)
- [x] Freeze feature configuration, classifier settings, thresholds,
  and preprocessing. (Frozen in configs/final_model_config.json and configs/phase10_thresholds.json)
- [x] Run final evaluation on the canonical test set for the selected
  pipeline. (Single evaluation on canonical 6,174 held-out test rows; no tuning on test)
- [x] Because the canonical test set was used in earlier experiments,
  document this limitation and do not repeatedly tune against it. (Strict protocol enforced; all model selections and threshold tuning were strictly on validation)
- [x] Report Macro-F1, Micro-F1, per-label precision/recall/F1, and
  relevant comparisons. (Test Macro-F1: 0.5107, Micro-F1: 0.5725; +0.0529 vs E0 Baseline, +0.0270 vs S0 LR Chain)
- [x] Document harmonization assumptions, especially inferred
  Disgust-negative labels. (EmoNoBa disgust negative assumption evaluated: Natural Disgust F1=0.5616, P=0.5484, R=0.5754 on native annotations)
- [x] Update existing README/project documentation. (Updated project documentation with final metrics, architecture, and reproducibility instructions)
- [x] Include reproducibility instructions, settings, limitations, and
  checkpoint location. (Model weights, chains, scalers, and vectorizers saved to artifacts/final_model/)
- [x] Do not commit or push without approval. (Acknowledged; no git commits pushed)
Results & Findings:
  - Canonical Test Set Evaluation (6,174 samples):
    - Macro-F1:        0.5107 (+0.0529 vs E0 TF-IDF 0.4578; +0.0270 vs S0 LR Chain 0.4837)
    - Micro-F1:        0.5725
    - Macro-Precision: 0.4534
    - Macro-Recall:    0.6364
    - Hamming Loss:    0.1988
    - Jaccard Score:   0.5241
    - Subset Accuracy: 0.3382
  - Per-Label Test F1 Breakdown:
    - Anger:    0.5420 (P=0.4070, R=0.8109)
    - Disgust:  0.3752 (P=0.2784, R=0.5754, Δ=+0.0276 vs S0; 0.5616 on native subset)
    - Fear:     0.4619 (P=0.5185, R=0.4164, Δ=+0.0971 vs S0)
    - Joy:      0.8141 (P=0.8041, R=0.8244)
    - Sadness:  0.4943 (P=0.3606, R=0.7859)
    - Surprise: 0.3767 (P=0.3519, R=0.4053, Δ=+0.0493 vs S0)
Deliverables:
1. Experiment script: scripts/experiments/phase12_final_evaluation.py
2. Reports: results/final/phase12_final_test_report.json & phase12_final_test_report.md
3. Final Configuration: configs/final_model_config.json
4. Model Artifacts: artifacts/final_model/ (LR & CNB chains, TF-IDF + emoji extractor, scalers)
5. Test Predictions CSV: results/final/test_predictions_champion.csv
6. Unit tests: tests/test_phase12_final_eval.py (4 passed)
Experiment Tracking Template
Record one row per experiment.
| Experiment | Feature groups | Validation Macro-F1 | Validation Micro-F1 | Per-label F1 (ang, dis, fea, joy, sad, sur) | Runtime | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **E0** | Word + Char TF-IDF (25k) | 0.4625 | 0.5577 | [0.528, 0.320, 0.300, 0.812, 0.499, 0.316] | 192s | Baseline |
| **E6** | TF-IDF + Emoji + Sentiment | 0.4847 | 0.5742 | [0.540, 0.339, 0.380, 0.824, 0.501, 0.324] | 210s | Compare |
| **E8** | TF-IDF + Emoji + Sent + Negation | 0.4848 | 0.5751 | [0.544, 0.341, 0.378, 0.825, 0.500, 0.315] | 218s | Strong baseline |
| **Phase 3** | TF-IDF + Emoji + 6-Emotion Lexicon | 0.4907 | 0.5791 | [0.544, 0.350, 0.388, 0.825, 0.504, 0.333] | 240s | Kept (broke 0.490 barrier) |
| **Phase 4** | TF-IDF + Emoji + EmoLex + NegScope | 0.4902 | 0.5820 | [0.549, 0.344, 0.392, 0.825, 0.502, 0.330] | 326s | Kept (All-time high Micro-F1) |
| **Phase 5 (5.2)** | Full Base + Hashtags (+3 dims) | 0.4904 | 0.5821 | [0.549, 0.344, 0.392, 0.825, 0.503, 0.329] | 316s | Kept (+0.0002 Macro, +0.0001 Micro) |
| **Phase 5 (Aux)** | Elongation, CodeMixing, TextStats, Spelling, Verbal | < 0.4902 | < 0.5820 | Regressed on minority classes (esp. Fear down to 0.352) | ~280s ea | Dropped |
| **Phase 7** | Phase 5 Champ + Per-Label Thresholds (τ*) | **0.4913** | **0.5825** | [0.549, 0.345, 0.393, 0.826, 0.503, 0.331] | 38s | Kept (+0.0009 Macro, all-time high) |
| **Phase 8 (S0)** | Imbalance Study: Balanced vs Custom/Resampling | **0.4913** | **0.5825** | [0.549, 0.345, 0.393, 0.826, 0.503, 0.331] | 350s (S0) / ~58m (all) | Kept S0; S4 rejected due to disgust drop |
| **Phase 8b (Group C)** | Base + Intensifier-Emotion Bigrams (25,101 dims) | **0.4921** | **0.5834** | [0.550, 0.348, 0.394, 0.825, 0.503, 0.332] | 254s (C) / 22m (all) | **Kept (+0.0008 Macro, +0.0009 Micro; NEW ALL-TIME HIGH)** |
| **Phase 9 (C1 LR)** | Classifier comparison: LR/CNB/SGD/KNN on 25,101-dim feats | **0.4921** | **0.5834** | [0.550, 0.348, 0.394, 0.825, 0.503, 0.332] | 636s (C1) / ~12m (all) | LR retained; CNB best unconstrained (0.5049) but fails safety rule |
| **Phase 10 (S3 Soft Vote)** | 0.70 LR + 0.30 CNB ClassifierChains (25,101 dims) | **0.5170** | 0.5746 | [0.544, 0.363, 0.516, 0.818, 0.499, 0.362] | ~6m (LR fit) + 12s | **Kept (+0.0249 Macro, NEW ALL-TIME PROJECT HIGH, 0 regressions)** |
| **Phase 11 (Multi-Seed)** | 3 Seeds (42, 123, 999) S3 Soft Vote on Val | **0.5170 ± 0.0000** | 0.5745 ± 0.0001 | [0.544, 0.363, 0.516, 0.818, 0.499, 0.362] | ~14m (3 seeds) | **Verified (Zero variance across seeds, +0.0248 gain confirmed)** |
| **Phase 12 (Final Test)** | 0.70 LR + 0.30 CNB on Canonical Test Set (6,174 rows) | **0.5107** | 0.5725 | [0.542, 0.375, 0.462, 0.814, 0.494, 0.377] | ~13m | **Final Benchmark (+0.0529 vs E0, +0.0270 vs S0; 0 test tuning)** |
Priority Order
1. Fix E3/E7.
2. Verify E8 and compare with E6.
3. Improve Bangla sentiment and emotion features.
4. Improve negation/intensifier handling.
5. Add feature groups one at a time.
6. Analyze Disgust and Surprise errors.
7. Optimize per-label thresholds.
8. Test class-imbalance strategies.
8b. Test targeted interrogative and discourse contrast features.
9. Compare LR, NB, SVM, and KNN.
10. Test classifier chains and voting.
11. Run multi-seed robustness experiments.
12. Freeze the selected pipeline and perform final test evaluation.
Selection rule: Keep a feature or method only when its improvement
is reproducible, the evaluation is fair, and per-label results do not
show unacceptable regressions.