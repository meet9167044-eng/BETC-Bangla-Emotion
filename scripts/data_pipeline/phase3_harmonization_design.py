import os, json, csv

OUTPUT_DIR = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion\results\harmonization"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 1. LABEL MAPPING CSV
# =============================================================================
label_mapping = [
    # EmoNoBa
    {
        "dataset": "EmoNoBa",
        "native_label": "Joy",
        "target_label": "joy",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "EmoNoBa Joy is semantically identical to the target Joy/Happiness emotion. Both refer to positive hedonic states. The capitalization difference is a formatting artifact, not a semantic difference.",
        "evidence_source": "EmoNoBa paper (Amin et al., 2023); target taxonomy definition (DATASET_SPEC.md Sec 3)",
        "coverage_positive_rows": 10112,
        "coverage_total_rows": 22739,
        "notes": "Highest-prevalence label in EmoNoBa (44.47%)."
    },
    {
        "dataset": "EmoNoBa",
        "native_label": "Anger",
        "target_label": "anger",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "Anger is one of Ekman's six basic emotions. EmoNoBa Anger directly corresponds to the target Anger class.",
        "evidence_source": "EmoNoBa paper; Ekman (1992)",
        "coverage_positive_rows": 4478,
        "coverage_total_rows": 22739,
        "notes": "19.69% prevalence."
    },
    {
        "dataset": "EmoNoBa",
        "native_label": "Sadness",
        "target_label": "sadness",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "Sadness is one of Ekman's six basic emotions. EmoNoBa Sadness directly corresponds to the target Sadness class.",
        "evidence_source": "EmoNoBa paper; Ekman (1992)",
        "coverage_positive_rows": 5681,
        "coverage_total_rows": 22739,
        "notes": "24.98% prevalence."
    },
    {
        "dataset": "EmoNoBa",
        "native_label": "Surprise",
        "target_label": "surprise",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "Surprise is one of Ekman's six basic emotions. EmoNoBa Surprise directly corresponds to the target Surprise class.",
        "evidence_source": "EmoNoBa paper; Ekman (1992)",
        "coverage_positive_rows": 1086,
        "coverage_total_rows": 22739,
        "notes": "Low prevalence (4.78%); imbalanced class."
    },
    {
        "dataset": "EmoNoBa",
        "native_label": "Fear",
        "target_label": "fear",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "Fear is one of Ekman's six basic emotions. EmoNoBa Fear directly corresponds to the target Fear class.",
        "evidence_source": "EmoNoBa paper; Ekman (1992)",
        "coverage_positive_rows": 401,
        "coverage_total_rows": 22739,
        "notes": "Very low prevalence (1.76%); severely imbalanced."
    },
    {
        "dataset": "EmoNoBa",
        "native_label": "Love",
        "target_label": "NONE",
        "status": "EXCLUDED",
        "mapping_type": "Out-of-taxonomy label",
        "rationale": "Love is not part of Ekman's six basic emotions and is not in the BETC target taxonomy (DATASET_SPEC.md Sec 3, finalized decision). Love is from the Junto Emotion Wheel and has no direct Ekman equivalent. DATASET_SPEC.md explicitly states Love must NOT be mapped to Disgust.",
        "evidence_source": "DATASET_SPEC.md Sec 3 [FINALIZED]; RESEARCH_RULES.md rule 20",
        "coverage_positive_rows": 4588,
        "coverage_total_rows": 22739,
        "notes": "2277 rows have Love as their ONLY active label. Dropping Love without a strategy leaves these rows as all-zero vectors or masks. REQUIRES USER DECISION on treatment. See harmonization_analysis.md."
    },
    {
        "dataset": "EmoNoBa",
        "native_label": "Disgust",
        "target_label": "disgust",
        "status": "UNANNOTATED",
        "mapping_type": "No native annotation exists — target label only",
        "rationale": "EmoNoBa does not include a Disgust annotation column. The target taxonomy requires a Disgust column. This is a missing/unobserved annotation problem, not an annotation of absence. Multiple strategies exist for filling this gap (see harmonization_analysis.md). DATASET_SPEC.md Sec 3 prescribes disgust=0 as the operational rule under a documented harmonization assumption, but this creates label-noise risk.",
        "evidence_source": "DATASET_SPEC.md Sec 3 [FINALIZED harmonization rule]; RESEARCH_RULES.md rule 9",
        "coverage_positive_rows": 0,
        "coverage_total_rows": 22739,
        "notes": "REQUIRES USER DECISION on strategy selection (A/B/C/D). All 22739 EmoNoBa rows are affected. See EmoNoBa Disgust strategy analysis in harmonization_analysis.md."
    },
    # UBMEC
    {
        "dataset": "UBMEC",
        "native_label": "joy",
        "target_label": "joy",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match (single-label to one-hot)",
        "rationale": "UBMEC joy directly matches target Joy. UBMEC is single-label; this row becomes a one-hot binary vector [joy=1, others=0].",
        "evidence_source": "UBMEC dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 3467,
        "coverage_total_rows": 13436,
        "notes": "25.80% of UBMEC rows. Conversion: all other 5 target columns receive 0 (valid for single-label 6-class datasets)."
    },
    {
        "dataset": "UBMEC",
        "native_label": "sadness",
        "target_label": "sadness",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match (single-label to one-hot)",
        "rationale": "UBMEC sadness directly matches target Sadness.",
        "evidence_source": "UBMEC dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 2683,
        "coverage_total_rows": 13436,
        "notes": "19.97% of UBMEC rows."
    },
    {
        "dataset": "UBMEC",
        "native_label": "anger",
        "target_label": "anger",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match (single-label to one-hot)",
        "rationale": "UBMEC anger directly matches target Anger.",
        "evidence_source": "UBMEC dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 2480,
        "coverage_total_rows": 13436,
        "notes": "18.46% of UBMEC rows."
    },
    {
        "dataset": "UBMEC",
        "native_label": "disgust",
        "target_label": "disgust",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match (single-label to one-hot)",
        "rationale": "UBMEC disgust directly matches target Disgust. This is UBMEC's major contribution: it provides Disgust signal missing from EmoNoBa.",
        "evidence_source": "UBMEC dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 2079,
        "coverage_total_rows": 13436,
        "notes": "15.47% of UBMEC rows. Sole dataset with clear Disgust annotation besides MONOVAB."
    },
    {
        "dataset": "UBMEC",
        "native_label": "surprise",
        "target_label": "surprise",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match (single-label to one-hot)",
        "rationale": "UBMEC surprise directly matches target Surprise.",
        "evidence_source": "UBMEC dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 1366,
        "coverage_total_rows": 13436,
        "notes": "10.17% of UBMEC rows."
    },
    {
        "dataset": "UBMEC",
        "native_label": "fear",
        "target_label": "fear",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match (single-label to one-hot)",
        "rationale": "UBMEC fear directly matches target Fear.",
        "evidence_source": "UBMEC dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 1361,
        "coverage_total_rows": 13436,
        "notes": "10.13% of UBMEC rows."
    },
    # MONOVAB
    {
        "dataset": "MONOVAB",
        "native_label": "anger",
        "target_label": "anger",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "MONOVAB anger directly matches target Anger.",
        "evidence_source": "MONOVAB dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 4499,
        "coverage_total_rows": 10224,
        "notes": "44.00% prevalence — dominant label in MONOVAB."
    },
    {
        "dataset": "MONOVAB",
        "native_label": "disgust",
        "target_label": "disgust",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "MONOVAB disgust directly matches target Disgust.",
        "evidence_source": "MONOVAB dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 2067,
        "coverage_total_rows": 10224,
        "notes": "20.22% prevalence."
    },
    {
        "dataset": "MONOVAB",
        "native_label": "enjoyment",
        "target_label": "joy",
        "status": "MAPPED",
        "mapping_type": "Semantic near-equivalence (PROPOSED, not exact taxonomy)",
        "rationale": "Enjoyment is not identical to Joy in all emotion theories, but is the closest available MONOVAB concept to the Ekman Joy category. Enjoyment is a hedonic positive emotion representing pleasure, happiness, and positive affect — the same phenomenological cluster as Joy/Happiness. The MONOVAB authors likely used 'enjoyment' as their label for the Ekman Joy category. The mapping is semantically well-motivated but should not be treated as a certainty.",
        "evidence_source": "MONOVAB dataset documentation; Ekman (1992) on Happiness/Joy; Fredrickson (2001) on positive emotions",
        "coverage_positive_rows": 1601,
        "coverage_total_rows": 10224,
        "notes": "[PROPOSED] — Mapping is reasonable but unverified against original MONOVAB annotation guidelines. REQUIRES USER DECISION on whether to treat as [FROZEN] or [REQUIRES DATA VERIFICATION]. If wrong, 1601 examples get mis-labelled as Joy."
    },
    {
        "dataset": "MONOVAB",
        "native_label": "fear",
        "target_label": "fear",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "MONOVAB fear directly matches target Fear.",
        "evidence_source": "MONOVAB dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 56,
        "coverage_total_rows": 10224,
        "notes": "Extremely low prevalence (0.55%). Severely imbalanced."
    },
    {
        "dataset": "MONOVAB",
        "native_label": "sadness",
        "target_label": "sadness",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "MONOVAB sadness directly matches target Sadness.",
        "evidence_source": "MONOVAB dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 1188,
        "coverage_total_rows": 10224,
        "notes": "11.62% prevalence."
    },
    {
        "dataset": "MONOVAB",
        "native_label": "surprise",
        "target_label": "surprise",
        "status": "DIRECT",
        "mapping_type": "Exact taxonomy match",
        "rationale": "MONOVAB surprise directly matches target Surprise.",
        "evidence_source": "MONOVAB dataset documentation; Ekman (1992)",
        "coverage_positive_rows": 100,
        "coverage_total_rows": 10224,
        "notes": "Very low prevalence (0.98%). Severely imbalanced."
    },
    {
        "dataset": "MONOVAB",
        "native_label": "contempt",
        "target_label": "NONE",
        "status": "EXCLUDED",
        "mapping_type": "Out-of-taxonomy label",
        "rationale": "Contempt is not part of the BETC target taxonomy (DATASET_SPEC.md Sec 3). Some emotion theories list contempt separately or as a variant of disgust/disdain, but the BETC target taxonomy is fixed at six Ekman emotions without contempt. DATASET_SPEC.md Sec 3 explicitly names contempt as a non-target label to discard.",
        "evidence_source": "DATASET_SPEC.md Sec 3 [FINALIZED]; RESEARCH_RULES.md rule 22",
        "coverage_positive_rows": 2960,
        "coverage_total_rows": 10224,
        "notes": "2128 rows have contempt as their ONLY active label. Discarding contempt leaves these rows with zero active target labels. REQUIRES USER DECISION on treatment. See harmonization_analysis.md."
    },
]

with open(os.path.join(OUTPUT_DIR, "label_mapping.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(label_mapping[0].keys()))
    writer.writeheader()
    writer.writerows(label_mapping)

print("label_mapping.csv written.")

# =============================================================================
# 2. LABEL COVERAGE CSV
# =============================================================================
# Shows for each target label, what coverage (positive examples) comes from each source
target_labels = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

label_coverage = [
    {
        "target_label": "anger",
        "emonoba_positive": 4478,
        "emonoba_total": 22739,
        "emonoba_prevalence_pct": 19.69,
        "emonoba_annotation_quality": "Direct human annotation",
        "ubmec_positive": 2480,
        "ubmec_total": 13436,
        "ubmec_prevalence_pct": 18.46,
        "ubmec_annotation_quality": "Direct human annotation (single-label)",
        "monovab_positive": 4499,
        "monovab_total": 10224,
        "monovab_prevalence_pct": 44.00,
        "monovab_annotation_quality": "Direct human annotation",
        "combined_positive_estimate": 11457,
        "combined_total": 46399,
        "combined_prevalence_pct": 24.69,
        "coverage_status": "FULL — all 3 datasets annotate anger",
        "notes": "Dominant label in MONOVAB (44%). Good cross-source coverage."
    },
    {
        "target_label": "disgust",
        "emonoba_positive": 0,
        "emonoba_total": 22739,
        "emonoba_prevalence_pct": 0.0,
        "emonoba_annotation_quality": "NOT ANNOTATED — structural gap in EmoNoBa",
        "ubmec_positive": 2079,
        "ubmec_total": 13436,
        "ubmec_prevalence_pct": 15.47,
        "ubmec_annotation_quality": "Direct human annotation (single-label)",
        "monovab_positive": 2067,
        "monovab_total": 10224,
        "monovab_prevalence_pct": 20.22,
        "monovab_annotation_quality": "Direct human annotation",
        "combined_positive_estimate": 4146,
        "combined_total": 46399,
        "combined_prevalence_pct": 8.94,
        "coverage_status": "PARTIAL — EmoNoBa has no native Disgust; 22739 rows have unknown Disgust status",
        "notes": "CRITICAL GAP. Strategy for EmoNoBa Disgust must be decided before harmonization. Estimated prevalence depends entirely on EmoNoBa strategy chosen."
    },
    {
        "target_label": "fear",
        "emonoba_positive": 401,
        "emonoba_total": 22739,
        "emonoba_prevalence_pct": 1.76,
        "emonoba_annotation_quality": "Direct human annotation",
        "ubmec_positive": 1361,
        "ubmec_total": 13436,
        "ubmec_prevalence_pct": 10.13,
        "ubmec_annotation_quality": "Direct human annotation (single-label)",
        "monovab_positive": 56,
        "monovab_total": 10224,
        "monovab_prevalence_pct": 0.55,
        "monovab_annotation_quality": "Direct human annotation",
        "combined_positive_estimate": 1818,
        "combined_total": 46399,
        "combined_prevalence_pct": 3.92,
        "coverage_status": "FULL but severely imbalanced — very low in EmoNoBa and MONOVAB",
        "notes": "Fear is the rarest target emotion. UBMEC provides the majority of Fear signal. Class-weight balancing will be critical."
    },
    {
        "target_label": "joy",
        "emonoba_positive": 10112,
        "emonoba_total": 22739,
        "emonoba_prevalence_pct": 44.47,
        "emonoba_annotation_quality": "Direct human annotation",
        "ubmec_positive": 3467,
        "ubmec_total": 13436,
        "ubmec_prevalence_pct": 25.80,
        "ubmec_annotation_quality": "Direct human annotation (single-label)",
        "monovab_positive": 1601,
        "monovab_total": 10224,
        "monovab_prevalence_pct": 15.66,
        "monovab_annotation_quality": "MAPPED from 'enjoyment' (PROPOSED, not directly equivalent)",
        "combined_positive_estimate": 15180,
        "combined_total": 46399,
        "combined_prevalence_pct": 32.72,
        "coverage_status": "FULL — all 3 datasets cover joy (MONOVAB via enjoyment mapping)",
        "notes": "MONOVAB uses 'enjoyment' label — mapping to Joy is PROPOSED and requires confirmation. EmoNoBa Joy has highest prevalence of any label in the combined set."
    },
    {
        "target_label": "sadness",
        "emonoba_positive": 5681,
        "emonoba_total": 22739,
        "emonoba_prevalence_pct": 24.98,
        "emonoba_annotation_quality": "Direct human annotation",
        "ubmec_positive": 2683,
        "ubmec_total": 13436,
        "ubmec_prevalence_pct": 19.97,
        "ubmec_annotation_quality": "Direct human annotation (single-label)",
        "monovab_positive": 1188,
        "monovab_total": 10224,
        "monovab_prevalence_pct": 11.62,
        "monovab_annotation_quality": "Direct human annotation",
        "combined_positive_estimate": 9552,
        "combined_total": 46399,
        "combined_prevalence_pct": 20.59,
        "coverage_status": "FULL — all 3 datasets annotate sadness",
        "notes": "Good cross-source coverage with consistent prevalence across datasets."
    },
    {
        "target_label": "surprise",
        "emonoba_positive": 1086,
        "emonoba_total": 22739,
        "emonoba_prevalence_pct": 4.78,
        "emonoba_annotation_quality": "Direct human annotation",
        "ubmec_positive": 1366,
        "ubmec_total": 13436,
        "ubmec_prevalence_pct": 10.17,
        "ubmec_annotation_quality": "Direct human annotation (single-label)",
        "monovab_positive": 100,
        "monovab_total": 10224,
        "monovab_prevalence_pct": 0.98,
        "monovab_annotation_quality": "Direct human annotation",
        "combined_positive_estimate": 2552,
        "combined_total": 46399,
        "combined_prevalence_pct": 5.50,
        "coverage_status": "FULL but imbalanced — very low in MONOVAB",
        "notes": "MONOVAB surprise (0.98%) is severely underrepresented. UBMEC is the dominant Surprise source."
    },
]

with open(os.path.join(OUTPUT_DIR, "label_coverage.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(label_coverage[0].keys()))
    writer.writeheader()
    writer.writerows(label_coverage)

print("label_coverage.csv written.")

# =============================================================================
# 3. HARMONIZATION STRATEGIES CSV
# =============================================================================
strategies = [
    # EmoNoBa Disgust strategies
    {
        "strategy_id": "DISGUST-A",
        "applies_to": "EmoNoBa — Disgust",
        "strategy_name": "Treat missing Disgust as 0 (Negative)",
        "decision_status": "[PROPOSED — DATASET_SPEC.md Sec 3 FINALIZED this as the operational rule, but REQUIRES USER CONFIRMATION before execution]",
        "affected_rows": 22739,
        "usable_rows_for_disgust": 22739,
        "rows_excluded": 0,
        "annotation_accuracy_risk": "HIGH — 22739 examples receive Disgust=0 based on assumption, not annotation. The actual Disgust status of EmoNoBa examples is unknown. Some EmoNoBa examples may genuinely express disgust but were not annotated for it.",
        "training_impact": "All EmoNoBa examples contribute to Disgust=0 training signal. Classifier sees 22739 artificial negatives, biasing it toward predicting Disgust=0 for EmoNoBa-domain texts.",
        "validation_impact": "EmoNoBa val examples used for threshold optimization on Disgust assume 0 is ground truth. If real Disgust exists in those examples, threshold will be miscalibrated.",
        "test_impact": "Sensitivity analysis (Experiment F5) required: re-evaluate with EmoNoBa rows excluded from Disgust scoring.",
        "macro_f1_impact": "Disgust F1 may appear higher than it truly is, as the classifier learns a biased prior. Macro-F1 will over-represent this inflated Disgust F1.",
        "lr_chain_impact": "Logistic Regression receives 22739 extra assumed Disgust=0 examples; class_weight='balanced' will compensate for class imbalance but not for label noise.",
        "advantages": "Simplest approach; uses all 22739 EmoNoBa examples; DATASET_SPEC.md already documents this assumption; enables a complete 6-column training matrix.",
        "disadvantages": "Introduces systematic label noise for Disgust; Disgust classifier trained partly on incorrect negatives; violates RESEARCH_RULES.md rule 9 unless explicitly logged.",
        "implementation_complexity": "LOW",
        "reproducibility": "High — deterministic, but sensitivity analysis (F5) is mandatory to quantify the impact.",
        "requires_user_approval": True
    },
    {
        "strategy_id": "DISGUST-B",
        "applies_to": "EmoNoBa — Disgust",
        "strategy_name": "Use label-coverage mask (treat Disgust as unknown/unobserved for EmoNoBa rows)",
        "decision_status": "[REQUIRES USER DECISION — not currently finalized in DATASET_SPEC.md]",
        "affected_rows": 22739,
        "usable_rows_for_disgust": 23660,
        "rows_excluded": 0,
        "annotation_accuracy_risk": "LOW — Does not introduce false Disgust=0 labels. Correctly represents epistemic uncertainty.",
        "training_impact": "EmoNoBa rows are excluded from Disgust binary cross-entropy loss during training. UBMEC (2079 positive) + MONOVAB (2067 positive) still provide Disgust signal. Requires a partial-label or label-masking training implementation (e.g. mask the Disgust loss term for EmoNoBa examples).",
        "validation_impact": "EmoNoBa val examples excluded from Disgust threshold optimization; threshold derived only from UBMEC/MONOVAB val examples. Smaller effective val set for Disgust.",
        "test_impact": "EmoNoBa test examples excluded from Disgust F1 computation (or reported separately). Test Disgust score reflects only UBMEC/MONOVAB domain examples.",
        "macro_f1_impact": "Macro-F1 computed with Disgust excluded from EmoNoBa evaluation; overall macro may change depending on how excluded label is averaged.",
        "lr_chain_impact": "Standard sklearn ClassifierChain does not natively support partial labels / loss masking. Requires custom wrapper or label-propagation approach. Higher implementation complexity.",
        "advantages": "Scientifically honest; no artificial label noise; Disgust classifier trained on clean annotated data only; produces more reliable Disgust F1 estimate.",
        "disadvantages": "Requires non-standard training code; Disgust classifier has less training data (only 23660 examples vs 46399); UBMEC/MONOVAB domain may not generalize to EmoNoBa domain.",
        "implementation_complexity": "HIGH — requires custom sklearn wrapper for masked labels",
        "reproducibility": "Moderate — mask must be documented and saved with the dataset split",
        "requires_user_approval": True
    },
    {
        "strategy_id": "DISGUST-C",
        "applies_to": "EmoNoBa — Disgust",
        "strategy_name": "Exclude EmoNoBa from Disgust training signal; retain for other 5 target emotions",
        "decision_status": "[REQUIRES USER DECISION]",
        "affected_rows": 22739,
        "usable_rows_for_disgust": 23660,
        "rows_excluded": 0,
        "annotation_accuracy_risk": "LOW — EmoNoBa contributes to 5 clean emotion labels; does not introduce Disgust noise.",
        "training_impact": "Similar to Strategy B at the Disgust link. EmoNoBa rows contribute fully to Anger/Joy/Sadness/Surprise/Fear training. Disgust link trained only on UBMEC+MONOVAB.",
        "validation_impact": "In a ClassifierChain, the Disgust link is trained after other links; if chain order puts Disgust later, it receives chain-augmented features from EmoNoBa rows (without Disgust ground truth). Must ensure EmoNoBa Disgust is masked from loss computation only.",
        "test_impact": "Evaluation of final BETC on EmoNoBa test rows: apply the Disgust predictor (trained on UBMEC+MONOVAB domain) but report a caveat that Disgust scores on EmoNoBa domain are extrapolated.",
        "macro_f1_impact": "Macro-F1 on EmoNoBa test would include a Disgust component from a cross-domain predictor; this is a known source of uncertainty.",
        "lr_chain_impact": "Possible with masked loss in ClassifierChain; same implementation complexity as Strategy B but semantically clearer about what is and isn't ground truth.",
        "advantages": "Most principled approach for a multi-label partial-label scenario; preserves EmoNoBa's valid annotations; does not inflate Disgust metrics artificially.",
        "disadvantages": "Higher implementation complexity; Disgust classifier has smaller training set; extrapolation risk when predicting Disgust for EmoNoBa domain.",
        "implementation_complexity": "HIGH",
        "reproducibility": "Moderate — requires saving which rows contributed to each label's training signal",
        "requires_user_approval": True
    },
    {
        "strategy_id": "DISGUST-D",
        "applies_to": "EmoNoBa — Disgust",
        "strategy_name": "Hybrid: Use Strategy A with mandatory F5 sensitivity analysis; report Disgust results separately for EmoNoBa vs UBMEC/MONOVAB subsets",
        "decision_status": "[PROPOSED — combines DATASET_SPEC.md rule with required sensitivity reporting]",
        "affected_rows": 22739,
        "usable_rows_for_disgust": 46399,
        "rows_excluded": 0,
        "annotation_accuracy_risk": "MEDIUM — same label noise as A, but explicitly quantified and reported",
        "training_impact": "Same as Strategy A; EmoNoBa Disgust=0 assumption used throughout training. F5 experiment (EXPERIMENT_PLAN.md) then re-evaluates excluding EmoNoBa from Disgust scoring.",
        "validation_impact": "Same as Strategy A for primary evaluation; F5 recomputes with EmoNoBa masked.",
        "test_impact": "Full test evaluation reports two Disgust metrics: (1) including EmoNoBa under the assumption, (2) excluding EmoNoBa (F5 sensitivity).",
        "macro_f1_impact": "Primary Macro-F1 uses full matrix (A's approach); supplementary Macro-F1 from F5 shows the impact of removing EmoNoBa from Disgust.",
        "lr_chain_impact": "Same as Strategy A (simple); no custom code needed.",
        "advantages": "Operationally simple; fully compatible with sklearn ClassifierChain; already aligned with DATASET_SPEC.md's documented rule; F5 provides the required scientific honesty check.",
        "disadvantages": "Still introduces label noise; requires careful documentation that Disgust=0 in EmoNoBa is an assumption, not an annotation.",
        "implementation_complexity": "LOW (primary) + MEDIUM (F5 sensitivity analysis)",
        "reproducibility": "High",
        "requires_user_approval": True
    },
    # Love-only EmoNoBa row strategies
    {
        "strategy_id": "LOVE-1",
        "applies_to": "EmoNoBa — Love-only rows (2277 rows)",
        "strategy_name": "Exclude Love-only rows from target-task corpus entirely",
        "decision_status": "[PROPOSED — recommended if zero-label rows are undesirable]",
        "affected_rows": 2277,
        "usable_rows_for_disgust": 0,
        "rows_excluded": 2277,
        "annotation_accuracy_risk": "LOW — removes potentially misleading zero-label examples",
        "training_impact": "2277 fewer training examples. Final combined corpus ≈ 44,122 rows (before dedup). EmoNoBa contribution reduced to 20,462 rows.",
        "validation_impact": "Slightly smaller val set; Love-only examples removed from threshold optimization.",
        "test_impact": "These examples not evaluated; cannot measure BETC performance on Love-expressing text.",
        "macro_f1_impact": "Minor negative impact from smaller training set; avoids artificial all-zero training examples.",
        "lr_chain_impact": "Fewer training examples; less class-imbalance pressure for rare classes.",
        "advantages": "Avoids training on all-zero label vectors; clean corpus; no ambiguity about what these examples teach the model.",
        "disadvantages": "Loses 2277 potentially informative Bangla social-media examples; reduction in corpus size; Love-only examples might still contain other detectable emotional signals.",
        "implementation_complexity": "LOW",
        "reproducibility": "High",
        "requires_user_approval": True
    },
    {
        "strategy_id": "LOVE-2",
        "applies_to": "EmoNoBa — Love-only rows (2277 rows)",
        "strategy_name": "Retain with masked target labels (unknown/unobserved for all 6 targets)",
        "decision_status": "[REQUIRES USER DECISION]",
        "affected_rows": 2277,
        "usable_rows_for_disgust": 2277,
        "rows_excluded": 0,
        "annotation_accuracy_risk": "LOW — correctly represents that these examples were not annotated for the 6 target emotions",
        "training_impact": "2277 examples excluded from all 6 label training signals; their text may still contribute to TF-IDF vocabulary but not to label learning.",
        "validation_impact": "Cannot be used for threshold optimization on any target label.",
        "test_impact": "Cannot be evaluated on any target label; must be reported separately.",
        "macro_f1_impact": "Excluded from F1 computation; reduces effective test set size.",
        "lr_chain_impact": "Requires partial-label support (same complexity as DISGUST-B). High implementation complexity for sklearn pipeline.",
        "advantages": "Most scientifically honest; text is not discarded; Love emotion is preserved in provenance metadata.",
        "disadvantages": "High implementation complexity; examples cannot contribute to F1 evaluation; effectively excluded from meaningful model evaluation.",
        "implementation_complexity": "HIGH",
        "reproducibility": "Moderate",
        "requires_user_approval": True
    },
    {
        "strategy_id": "LOVE-3",
        "applies_to": "EmoNoBa — Love-only rows (2277 rows)",
        "strategy_name": "Retain with all-zero target vector (treat as all-negative for 6 targets)",
        "decision_status": "[REQUIRES USER DECISION — documented assumption required if chosen]",
        "affected_rows": 2277,
        "usable_rows_for_disgust": 2277,
        "rows_excluded": 0,
        "annotation_accuracy_risk": "MEDIUM-HIGH — implicitly claims these texts are negative for Anger, Disgust, Fear, Joy, Sadness, Surprise; this is an inference, not an annotation",
        "training_impact": "2277 all-zero examples in training matrix; classifier sees them as 'none of the six target emotions'. If some of these texts genuinely express Joy or Sadness, this introduces noise.",
        "validation_impact": "All-zero examples contribute to threshold optimization; may push thresholds higher (fewer positives to predict).",
        "test_impact": "Included in evaluation; false negatives predicted as correctly negative by definition.",
        "macro_f1_impact": "Inflates recall for rare classes artificially; may improve subset accuracy metric artifactually.",
        "lr_chain_impact": "Compatible with standard sklearn ClassifierChain; no custom code needed.",
        "advantages": "Operationally simple; no excluded rows; full corpus retained.",
        "disadvantages": "Introduces known label noise; conflates 'not annotated' with 'annotated negative'; violates RESEARCH_RULES.md rule 9 unless explicitly documented.",
        "implementation_complexity": "LOW",
        "reproducibility": "High, but scientific validity is questionable",
        "requires_user_approval": True
    },
    # Contempt-only MONOVAB row strategies
    {
        "strategy_id": "CONTEMPT-1",
        "applies_to": "MONOVAB — Contempt-only rows (2128 rows)",
        "strategy_name": "Exclude Contempt-only rows from target-task corpus",
        "decision_status": "[PROPOSED — recommended for cleanliness]",
        "affected_rows": 2128,
        "usable_rows_for_disgust": 0,
        "rows_excluded": 2128,
        "annotation_accuracy_risk": "LOW — removes ambiguous examples where target emotion status is unknown",
        "training_impact": "2128 fewer training examples. MONOVAB contribution ≈ 8096 rows. Overall corpus ≈ 44,271 rows.",
        "validation_impact": "Smaller MONOVAB val partition.",
        "test_impact": "Cannot evaluate BETC on Contempt-only MONOVAB texts.",
        "macro_f1_impact": "Minor; slightly smaller MONOVAB contribution.",
        "lr_chain_impact": "No impact; standard pipeline.",
        "advantages": "No ambiguity; no all-zero label noise; clean corpus.",
        "disadvantages": "Loses 2128 examples; Contempt examples might co-occur with target emotions that annotators did not record.",
        "implementation_complexity": "LOW",
        "reproducibility": "High",
        "requires_user_approval": True
    },
    {
        "strategy_id": "CONTEMPT-2",
        "applies_to": "MONOVAB — Contempt-only rows (2128 rows)",
        "strategy_name": "Retain with unknown/masked target labels",
        "decision_status": "[REQUIRES USER DECISION]",
        "affected_rows": 2128,
        "usable_rows_for_disgust": 0,
        "rows_excluded": 0,
        "annotation_accuracy_risk": "LOW — correctly represents annotation gap",
        "training_impact": "2128 examples with unknown targets; requires partial-label framework.",
        "validation_impact": "Cannot contribute to threshold optimization.",
        "test_impact": "Cannot be evaluated on target emotions.",
        "macro_f1_impact": "Excluded from F1 computation.",
        "lr_chain_impact": "High implementation complexity; requires masked loss.",
        "advantages": "Scientifically honest; text retained in corpus metadata.",
        "disadvantages": "High complexity; effectively excluded from evaluation anyway.",
        "implementation_complexity": "HIGH",
        "reproducibility": "Moderate",
        "requires_user_approval": True
    },
    {
        "strategy_id": "CONTEMPT-3",
        "applies_to": "MONOVAB — Contempt-only rows (2128 rows)",
        "strategy_name": "Retain with all-zero target vector (treat as all-negative for 6 targets)",
        "decision_status": "[REQUIRES USER DECISION — documented assumption required]",
        "affected_rows": 2128,
        "usable_rows_for_disgust": 2128,
        "rows_excluded": 0,
        "annotation_accuracy_risk": "MEDIUM — contempt and disgust are closely related emotions; some texts labelled contempt-only may actually express disgust; treating them as disgust=0 risks introducing label noise specifically for disgust",
        "training_impact": "2128 all-zero examples; may teach the classifier that Contempt-expressing text is negative for all six target emotions. Potential cross-emotion noise if contempt correlates with disgust.",
        "validation_impact": "Included in threshold optimization with all-zero ground truth.",
        "test_impact": "Included in evaluation; problematic if these texts actually express target emotions.",
        "macro_f1_impact": "Could artificially inflate accuracy for some classes.",
        "lr_chain_impact": "No custom code needed; compatible with standard pipeline.",
        "advantages": "Simple; retains all examples in corpus.",
        "disadvantages": "Semantic overlap between contempt and disgust makes this especially risky; label noise for Disgust is plausible; violates RESEARCH_RULES.md rule 9 unless documented.",
        "implementation_complexity": "LOW",
        "reproducibility": "High, but scientific validity questionable for Disgust label especially",
        "requires_user_approval": True
    },
]

with open(os.path.join(OUTPUT_DIR, "harmonization_strategies.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(strategies[0].keys()))
    writer.writeheader()
    writer.writerows(strategies)

print("harmonization_strategies.csv written.")

# =============================================================================
# 4. SOURCE LABEL SEMANTICS JSON
# =============================================================================
semantics = {
    "taxonomy_system": "Ekman's six basic emotions (1992): anger, disgust, fear, happiness/joy, sadness, surprise",
    "target_taxonomy_status": "[FROZEN]",
    "target_labels": ["anger", "disgust", "fear", "joy", "sadness", "surprise"],
    "datasets": {
        "EmoNoBa": {
            "annotation_system": "Junto Emotion Wheel (variant), 6 categories",
            "domain": "Bangla YouTube/Facebook/Twitter comments (12 topics)",
            "annotation_type": "Multi-label binary",
            "native_labels": {
                "Love": {
                    "target_mapping": None,
                    "taxonomy_origin": "Junto Emotion Wheel (positive emotion cluster)",
                    "ekman_equivalent": "No direct Ekman equivalent; closest: Happiness/Joy but semantically distinct",
                    "status": "EXCLUDED",
                    "reason": "Not in Ekman's six basic emotions; DATASET_SPEC.md explicitly prohibits mapping Love to Disgust or any target label",
                    "prevalence_pct": 20.18
                },
                "Joy": {
                    "target_mapping": "joy",
                    "taxonomy_origin": "Junto Emotion Wheel; equivalent to Ekman Happiness/Joy",
                    "ekman_equivalent": "Joy / Happiness (exact)",
                    "status": "DIRECT",
                    "reason": "Direct semantic match",
                    "prevalence_pct": 44.47
                },
                "Surprise": {
                    "target_mapping": "surprise",
                    "taxonomy_origin": "Ekman six basic emotions",
                    "ekman_equivalent": "Surprise (exact)",
                    "status": "DIRECT",
                    "reason": "Direct semantic match",
                    "prevalence_pct": 4.78
                },
                "Anger": {
                    "target_mapping": "anger",
                    "taxonomy_origin": "Ekman six basic emotions",
                    "ekman_equivalent": "Anger (exact)",
                    "status": "DIRECT",
                    "reason": "Direct semantic match",
                    "prevalence_pct": 19.69
                },
                "Sadness": {
                    "target_mapping": "sadness",
                    "taxonomy_origin": "Ekman six basic emotions",
                    "ekman_equivalent": "Sadness (exact)",
                    "status": "DIRECT",
                    "reason": "Direct semantic match",
                    "prevalence_pct": 24.98
                },
                "Fear": {
                    "target_mapping": "fear",
                    "taxonomy_origin": "Ekman six basic emotions",
                    "ekman_equivalent": "Fear (exact)",
                    "status": "DIRECT",
                    "reason": "Direct semantic match",
                    "prevalence_pct": 1.76
                },
                "Disgust": {
                    "target_mapping": "disgust",
                    "taxonomy_origin": "NOT IN EMONOBA — structural annotation gap",
                    "ekman_equivalent": "Disgust (would be exact if annotated)",
                    "status": "UNANNOTATED",
                    "reason": "EmoNoBa was designed around the Junto Wheel which replaces Disgust with Love; no Disgust annotation was collected for any EmoNoBa example",
                    "prevalence_pct": 0.0
                }
            }
        },
        "UBMEC": {
            "annotation_system": "Ekman's six basic emotions — single-label multi-class",
            "domain": "Bangla Facebook, YouTube, and BNEmo/BEmoC dataset combination",
            "annotation_type": "Single-label categorical (one string class per row)",
            "native_labels": {
                "joy": {"target_mapping": "joy", "status": "DIRECT", "prevalence_pct": 25.80},
                "sadness": {"target_mapping": "sadness", "status": "DIRECT", "prevalence_pct": 19.97},
                "anger": {"target_mapping": "anger", "status": "DIRECT", "prevalence_pct": 18.46},
                "disgust": {"target_mapping": "disgust", "status": "DIRECT", "prevalence_pct": 15.47},
                "surprise": {"target_mapping": "surprise", "status": "DIRECT", "prevalence_pct": 10.17},
                "fear": {"target_mapping": "fear", "status": "DIRECT", "prevalence_pct": 10.13}
            },
            "conversion_note": "Single-label to six-column binary conversion: assign 1 to the observed class, 0 to all other 5 target columns. This is a representation change, not a claim that UBMEC was multi-label. The 0s represent 'not the observed category' in a six-class mutually-exclusive setup. This is valid for single-label classification datasets."
        },
        "MONOVAB": {
            "annotation_system": "Multi-label binary (7 columns), mixing Ekman emotions with contempt",
            "domain": "Bangla social media and online news portals",
            "annotation_type": "Multi-label binary",
            "native_labels": {
                "anger": {"target_mapping": "anger", "status": "DIRECT", "prevalence_pct": 44.00},
                "contempt": {
                    "target_mapping": None,
                    "status": "EXCLUDED",
                    "reason": "Contempt is not in Ekman's six basic emotions and not in BETC target taxonomy; DATASET_SPEC.md Sec 3 explicitly names contempt as excluded; 2128 contempt-only rows need strategy decision",
                    "prevalence_pct": 28.95
                },
                "disgust": {"target_mapping": "disgust", "status": "DIRECT", "prevalence_pct": 20.22},
                "enjoyment": {
                    "target_mapping": "joy",
                    "status": "MAPPED",
                    "semantic_note": "Enjoyment is semantically near-equivalent to Joy/Happiness in most emotion theories. MONOVAB likely used 'enjoyment' for the Ekman Happiness/Joy category. Mapping is well-motivated but not confirmed against original annotation guidelines.",
                    "mapping_certainty": "PROPOSED — [REQUIRES USER DECISION] to freeze",
                    "prevalence_pct": 15.66
                },
                "fear": {"target_mapping": "fear", "status": "DIRECT", "prevalence_pct": 0.55},
                "sadness": {"target_mapping": "sadness", "status": "DIRECT", "prevalence_pct": 11.62},
                "surprise": {"target_mapping": "surprise", "status": "DIRECT", "prevalence_pct": 0.98}
            }
        }
    }
}

with open(os.path.join(OUTPUT_DIR, "source_label_semantics.json"), "w", encoding="utf-8") as f:
    json.dump(semantics, f, indent=2, ensure_ascii=False)

print("source_label_semantics.json written.")

# =============================================================================
# 5. HARMONIZATION DECISIONS JSON
# =============================================================================
decisions = {
    "phase": "Phase 3 — Harmonization Design",
    "date": "2026-09-19",
    "status": "DESIGN ONLY — No data has been modified. Awaiting user review.",
    "frozen_decisions": [
        {
            "decision": "Target taxonomy is exactly: anger, disgust, fear, joy, sadness, surprise",
            "status": "[FROZEN]",
            "source": "DATASET_SPEC.md Sec 3"
        },
        {
            "decision": "EmoNoBa Love is excluded from target space; NOT mapped to Disgust",
            "status": "[FROZEN]",
            "source": "DATASET_SPEC.md Sec 3; RESEARCH_RULES.md rule 20"
        },
        {
            "decision": "MONOVAB contempt is excluded from target space",
            "status": "[FROZEN]",
            "source": "DATASET_SPEC.md Sec 3; RESEARCH_RULES.md rule 22"
        },
        {
            "decision": "UBMEC single categorical label is converted to a six-column one-hot binary vector",
            "status": "[FROZEN]",
            "source": "DATASET_SPEC.md Sec 3; RESEARCH_RULES.md rule 21"
        },
        {
            "decision": "EmoNoBa native labels Joy/Anger/Sadness/Surprise/Fear are DIRECT mappings to target Joy/Anger/Sadness/Surprise/Fear",
            "status": "[FROZEN]",
            "source": "DATASET_SPEC.md Sec 3; direct semantic equivalence"
        },
        {
            "decision": "UBMEC joy/sadness/anger/disgust/surprise/fear are DIRECT mappings to corresponding target labels",
            "status": "[FROZEN]",
            "source": "DATASET_SPEC.md Sec 3; Ekman six emotions; direct semantic equivalence"
        },
        {
            "decision": "MONOVAB anger/disgust/fear/sadness/surprise are DIRECT mappings to corresponding target labels",
            "status": "[FROZEN]",
            "source": "DATASET_SPEC.md Sec 3; direct semantic equivalence"
        }
    ],
    "proposed_decisions": [
        {
            "decision": "MONOVAB enjoyment maps to target Joy",
            "status": "[PROPOSED]",
            "rationale": "Semantic near-equivalence; enjoyment is the hedonic positive emotion in MONOVAB's scheme, corresponding to Ekman's Joy/Happiness",
            "requires_user_approval": True,
            "risk_if_wrong": "1601 examples would have incorrect Joy label"
        },
        {
            "decision": "EmoNoBa Disgust strategy: DISGUST-D (Strategy A + mandatory F5 sensitivity analysis)",
            "status": "[PROPOSED — per DATASET_SPEC.md Sec 3 operational rule]",
            "rationale": "Aligns with DATASET_SPEC.md Sec 3 documented rule while requiring F5 to quantify the assumption's impact. Simplest operationally. Alternative strategies B/C are also viable.",
            "requires_user_approval": True,
            "risk_if_wrong": "Label noise for Disgust if chosen without sensitivity analysis"
        }
    ],
    "requires_user_decision": [
        {
            "issue": "EmoNoBa Disgust strategy",
            "question": "Which strategy (A / B / C / D) should be used for EmoNoBa's unobserved Disgust annotations? DATASET_SPEC.md prescribes A (disgust=0 assumption) but Strategy D is recommended here as it combines A with the required F5 sensitivity analysis.",
            "options": ["DISGUST-A (simple 0-fill)", "DISGUST-B (masked labels)", "DISGUST-C (exclude from disgust training only)", "DISGUST-D (A + mandatory F5 sensitivity)"],
            "blocking": True
        },
        {
            "issue": "EmoNoBa Love-only rows (2277 rows)",
            "question": "How should 2277 EmoNoBa rows where Love is the ONLY active label be handled after Love is excluded from the target taxonomy?",
            "options": ["LOVE-1 (exclude from corpus)", "LOVE-2 (retain with masked labels)", "LOVE-3 (retain as all-zero target vector with documented assumption)"],
            "blocking": True
        },
        {
            "issue": "MONOVAB Contempt-only rows (2128 rows)",
            "question": "How should 2128 MONOVAB rows where Contempt is the ONLY active label be handled after Contempt is excluded from the target taxonomy?",
            "options": ["CONTEMPT-1 (exclude from corpus)", "CONTEMPT-2 (retain with masked labels)", "CONTEMPT-3 (retain as all-zero target vector with documented assumption)"],
            "blocking": True
        },
        {
            "issue": "MONOVAB enjoyment → Joy mapping",
            "question": "Should the MONOVAB 'enjoyment' label be formally frozen as mapping to target 'joy'? This is semantically well-motivated but not confirmed against MONOVAB's original annotation guidelines.",
            "options": ["Freeze the mapping as [FROZEN]", "Keep as [PROPOSED] and verify against MONOVAB documentation", "Reject mapping and treat MONOVAB enjoyment as non-target"],
            "blocking": True
        },
        {
            "issue": "Conflicting duplicate annotations (UBMEC: 58 texts; MONOVAB: 33 texts; EmoNoBa: 8 texts)",
            "question": "What resolution strategy should be applied to identical texts with conflicting emotion labels during Phase 4 deduplication?",
            "options": ["Drop all rows for conflicting texts (safest)", "Take label union (most inclusive)", "Keep first occurrence by row index", "Manual annotation review for disputed labels"],
            "blocking": False,
            "note": "This will be addressed in Phase 4; flagged here for user awareness."
        },
        {
            "issue": "Degenerate texts (5 rows: UBMEC row 12039 = integer '1'; MONOVAB rows 5477, 8180, 8988, 9656 = punctuation only)",
            "question": "Should degenerate/non-alphabetic texts be explicitly dropped during harmonization, or retained?",
            "options": ["Drop and log (recommended)", "Retain as-is"],
            "blocking": False,
            "note": "Minor in scale (5 rows); recommended to drop during Phase 4 dedup/cleanup."
        }
    ],
    "usable_rows_estimates": {
        "note": "Estimates computed from Phase 1 audit. These are pre-deduplication upper bounds.",
        "strategy_combinations": [
            {
                "disgust_strategy": "DISGUST-A or DISGUST-D",
                "love_strategy": "LOVE-1 (exclude Love-only)",
                "contempt_strategy": "CONTEMPT-1 (exclude contempt-only)",
                "emonoba_rows": 20462,
                "ubmec_rows": 13436,
                "monovab_rows": 8096,
                "combined_before_dedup": 42994,
                "cross_dataset_dups_to_remove_estimate": 65,
                "within_dataset_dups_to_remove_estimate": 533,
                "estimated_final_rows": 42396,
                "comment": "Cleanest corpus; recommended combination"
            },
            {
                "disgust_strategy": "DISGUST-A or DISGUST-D",
                "love_strategy": "LOVE-3 (all-zero target vector)",
                "contempt_strategy": "CONTEMPT-1 (exclude contempt-only)",
                "emonoba_rows": 22739,
                "ubmec_rows": 13436,
                "monovab_rows": 8096,
                "combined_before_dedup": 44271,
                "cross_dataset_dups_to_remove_estimate": 65,
                "within_dataset_dups_to_remove_estimate": 533,
                "estimated_final_rows": 43673,
                "comment": "Closest to literature-reported 43676; includes Love-only as all-zero"
            },
            {
                "disgust_strategy": "DISGUST-A or DISGUST-D",
                "love_strategy": "LOVE-3 (all-zero target vector)",
                "contempt_strategy": "CONTEMPT-3 (all-zero target vector)",
                "emonoba_rows": 22739,
                "ubmec_rows": 13436,
                "monovab_rows": 10224,
                "combined_before_dedup": 46399,
                "cross_dataset_dups_to_remove_estimate": 65,
                "within_dataset_dups_to_remove_estimate": 533,
                "estimated_final_rows": 45801,
                "comment": "Maximum row retention; highest label noise risk for Love-only and Contempt-only rows"
            }
        ]
    }
}

with open(os.path.join(OUTPUT_DIR, "harmonization_decisions.json"), "w", encoding="utf-8") as f:
    json.dump(decisions, f, indent=2, ensure_ascii=False)

print("harmonization_decisions.json written.")
print("All machine-readable artifacts generated successfully.")
