"""
scripts/experiments/phase6_error_analysis.py
=============================================
BETC Improvement Plan — Phase 6: Analyze Disgust and Surprise Errors

Note: Uses NumpyEncoder to handle np.int64 types in JSON serialization.

Objectives:
  1. Re-build the Phase 5 Champion Pipeline (TF-IDF + Emoji + 6-EmoLex + NegScope + Hashtags)
     and save per-class validation probabilities.
  2. Extract False Positives (FP) and False Negatives (FN) for Disgust and Surprise.
  3. Group errors by surface patterns:
     - Short text (< 5 words)
     - Presence of negation markers
     - Presence of emoji
     - Bangla-English code-mixing
     - Hashtag usage
     - Confusing emotion combinations (e.g., Anger + Disgust, Joy + Surprise)
     - EmoNoBa Disgust assumption (negative Disgust label derived assumption)
  4. Measure overlap / confusion matrix between Disgust and Surprise vs other classes.
  5. Run sensitivity analysis on EmoNoBa Disgust assumption:
     - Compare performance when emonoba_disgust_assumption == True vs False texts.
  6. Propose and test targeted improvement features.
  7. Output error-analysis table and prioritized improvement list.

Output:
  - results/ablations/phase6_error_analysis/phase6_error_report.json
  - results/ablations/phase6_error_analysis/fp_disgust.csv
  - results/ablations/phase6_error_analysis/fn_disgust.csv
  - results/ablations/phase6_error_analysis/fp_surprise.csv
  - results/ablations/phase6_error_analysis/fn_surprise.csv
"""

import os
import sys
from json import JSONEncoder
import io
import re
import json
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MaxAbsScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

from src.models.betc import TARGET_COLS, compute_label_order
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds
from src.features.feature_combination import BETCCombinedExtractor
from scripts.experiments.phase3_lexicon_study import extract_emotion_lexicon_features, EMOTION_LEXICON
from scripts.experiments.phase4_negation_study import (
    EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, load_emotion_lexicon,
    extract_advanced_negation_features
)
from scripts.experiments.phase5_additional_features import extract_group_5_2_hashtags

DATA_DIR    = os.path.join(WORKSPACE, "Data", "processed")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase6_error_analysis")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Regex helpers for error grouping
_RE_EMOJI   = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
    r"\U0001F680-\U0001F6FF\U0001F700-\U0001F77F"
    r"\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
    r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F"
    r"\U0001FA70-\U0001FAFF\U00002702-\U000027B0]+",
    re.UNICODE
)
_RE_HASHTAG = re.compile(r"#\S+")
_RE_LATIN   = re.compile(r"[a-zA-Z]")
_BANGLA_RANGE = range(0x0980, 0x0A00)


def has_negation(text, neg_vocab):
    """Return True if any negation token found in text."""
    for neg in neg_vocab:
        if neg in text:
            return True
    return False


def is_code_mixed(text, min_bangla=3, min_latin=3):
    """Return True if text has both Bangla and Latin chars above thresholds."""
    n_bn  = sum(1 for c in text if ord(c) in _BANGLA_RANGE)
    n_lat = len(_RE_LATIN.findall(text))
    return n_bn >= min_bangla and n_lat >= min_latin


def classify_error_patterns(texts_raw, texts_proc, neg_vocab):
    """Return a dict of lists of boolean flags for each pattern."""
    patterns = {
        "short_text":   [],   # < 5 words
        "has_negation": [],
        "has_emoji":    [],
        "has_hashtag":  [],
        "code_mixed":   [],
    }
    for raw, proc in zip(texts_raw, texts_proc):
        words = raw.split()
        patterns["short_text"].append(len(words) < 5)
        patterns["has_negation"].append(has_negation(proc, neg_vocab))
        patterns["has_emoji"].append(bool(_RE_EMOJI.search(raw)))
        patterns["has_hashtag"].append(bool(_RE_HASHTAG.search(raw)))
        patterns["code_mixed"].append(is_code_mixed(raw))
    return {k: np.array(v) for k, v in patterns.items()}


def confusion_matrix_pairs(Y_true, Y_pred, label_names):
    """
    For each pair of labels (i, j), count:
      - FP_i & TP_j: predicted i but label is actually j (not i) — most common confusion
      - Co-occurrence: both i and j appear in true labels simultaneously
    Returns a dict of label -> {other_label: count_co_occurrence}
    """
    n_labels = len(label_names)
    co_occ_true  = defaultdict(lambda: defaultdict(int))
    fp_confusion = defaultdict(lambda: defaultdict(int))

    for row_t, row_p in zip(Y_true, Y_pred):
        true_labels = [label_names[i] for i in range(n_labels) if row_t[i] == 1]
        pred_labels = [label_names[i] for i in range(n_labels) if row_p[i] == 1]

        # Co-occurrences in true labels
        for a in true_labels:
            for b in true_labels:
                if a != b:
                    co_occ_true[a][b] += 1

        # FP confusion: predicted a, but a is not in true labels
        for pred_lbl in pred_labels:
            if pred_lbl not in true_labels:
                # What IS in the true labels (what was it instead)
                for true_lbl in true_labels:
                    fp_confusion[pred_lbl][true_lbl] += 1

    return dict(co_occ_true), dict(fp_confusion)


def summarize_errors(label, Y_true, Y_pred, df_val, patterns, focus_col):
    """
    Collect FP and FN for a specific label and compute pattern breakdowns.
    Returns: fp_rows, fn_rows, pattern_summary_dict
    """
    col_idx = TARGET_COLS.index(label)
    true_col = Y_true[:, col_idx]
    pred_col = Y_pred[:, col_idx]

    fp_mask = (true_col == 0) & (pred_col == 1)   # predicted positive, actually negative
    fn_mask = (true_col == 1) & (pred_col == 0)   # predicted negative, actually positive

    fp_df = df_val[fp_mask].copy()
    fn_df = df_val[fn_mask].copy()

    # Pattern summary
    summary = {
        "total_fp": int(fp_mask.sum()),
        "total_fn": int(fn_mask.sum()),
        "fp_patterns": {},
        "fn_patterns": {},
    }
    for pat_name, pat_arr in patterns.items():
        fp_count = int(pat_arr[fp_mask].sum())
        fn_count = int(pat_arr[fn_mask].sum())
        summary["fp_patterns"][pat_name] = {"count": fp_count, "pct": round(fp_count / max(1, int(fp_mask.sum())) * 100, 1)}
        summary["fn_patterns"][pat_name] = {"count": fn_count, "pct": round(fn_count / max(1, int(fn_mask.sum())) * 100, 1)}

    return fp_df, fn_df, summary


def sensitivity_analysis_emonoba(label, Y_true, Y_pred, df_val):
    """
    Compare Disgust F1 on EmoNoBa rows where the assumption was applied
    vs rows where it was not (or other datasets).
    """
    col_idx = TARGET_COLS.index(label)
    col_vals = df_val["emonoba_disgust_assumption"]
    assumption_mask = (col_vals == True) | (col_vals == "True") | (col_vals == 1) | (col_vals == 1.0)

    rows_assumption = assumption_mask.values
    rows_other      = ~rows_assumption

    metrics_assumption = {}
    metrics_other      = {}

    if rows_assumption.sum() > 0:
        from sklearn.metrics import f1_score, precision_score, recall_score
        yt_a = Y_true[rows_assumption, :]
        yp_a = Y_pred[rows_assumption, :]
        f1_a   = float(f1_score(yt_a[:, col_idx], yp_a[:, col_idx], zero_division=0))
        prec_a = float(precision_score(yt_a[:, col_idx], yp_a[:, col_idx], zero_division=0))
        rec_a  = float(recall_score(yt_a[:, col_idx], yp_a[:, col_idx], zero_division=0))
        metrics_assumption = {
            "n": int(rows_assumption.sum()),
            "f1": round(f1_a, 4),
            "precision": round(prec_a, 4),
            "recall": round(rec_a, 4)
        }

    if rows_other.sum() > 0:
        from sklearn.metrics import f1_score, precision_score, recall_score
        yt_o = Y_true[rows_other, :]
        yp_o = Y_pred[rows_other, :]
        f1_o   = float(f1_score(yt_o[:, col_idx], yp_o[:, col_idx], zero_division=0))
        prec_o = float(precision_score(yt_o[:, col_idx], yp_o[:, col_idx], zero_division=0))
        rec_o  = float(recall_score(yt_o[:, col_idx], yp_o[:, col_idx], zero_division=0))
        metrics_other = {
            "n": int(rows_other.sum()),
            "f1": round(f1_o, 4),
            "precision": round(prec_o, 4),
            "recall": round(rec_o, 4)
        }

    return {"emonoba_assumption_rows": metrics_assumption, "other_rows": metrics_other}


def main():
    print("=" * 75)
    print("BETC Phase 6 — Disgust and Surprise Error Analysis")
    print("=" * 75)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[Step 1] Loading canonical data splits...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train", "train_preprocessed.csv"), low_memory=False)
    df_val   = pd.read_csv(os.path.join(DATA_DIR, "validation", "validation_preprocessed.csv"), low_memory=False)
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    print(f"  Train: {len(df_train):,} | Val: {len(df_val):,}")

    train_proc = df_train["processed_text"].fillna("").astype(str).tolist()
    val_proc   = df_val["processed_text"].fillna("").astype(str).tolist()
    train_raw  = df_train["text"].fillna("").astype(str).tolist()
    val_raw    = df_val["text"].fillna("").astype(str).tolist()

    Y_train = df_train[TARGET_COLS].values.astype(int)
    Y_val   = df_val[TARGET_COLS].values.astype(int)
    freq_order, freq_names, label_freqs = compute_label_order(Y_train, TARGET_COLS)

    print(f"  Label frequencies (train): {label_freqs}")

    # ── 2. Build Phase 5 Champion Feature Matrix ─────────────────────────────
    print("\n[Step 2] Building Phase 5 Champion Feature Matrix (TF-IDF + Emoji + EmoLex + NegScope + Hashtags)...")
    t0 = time.time()

    # TF-IDF + Emoji
    ext_base = BETCCombinedExtractor(use_tfidf=True, use_emoji=True, use_sentiment=False, use_negation=False, use_expressive=False)
    X_base_tr = ext_base.fit_transform(train_raw, train_proc)
    X_base_va = ext_base.transform(val_raw, val_proc)

    # 6-Emotion Lexicon (18 dims)
    emo_vocab_set = {k: set(v) for k, v in EMOTION_LEXICON.items()}
    X_emo_tr = sp.csr_matrix(extract_emotion_lexicon_features(train_proc, emo_vocab_set))
    X_emo_va = sp.csr_matrix(extract_emotion_lexicon_features(val_proc, emo_vocab_set))

    # Advanced Negation Scope (16 dims)
    emo_dict = load_emotion_lexicon()
    X_neg_tr = sp.csr_matrix(extract_advanced_negation_features(train_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict))
    X_neg_va = sp.csr_matrix(extract_advanced_negation_features(val_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict))

    # Hashtag features (3 dims, scaled)
    scaler = MaxAbsScaler()
    f_hash_tr = scaler.fit_transform(extract_group_5_2_hashtags(train_raw))
    f_hash_va = MaxAbsScaler().fit_transform(extract_group_5_2_hashtags(val_raw))
    X_hash_tr = sp.csr_matrix(f_hash_tr)
    X_hash_va = sp.csr_matrix(f_hash_va)

    # Champion matrix
    X_champ_tr = sp.hstack([X_base_tr, X_emo_tr, X_neg_tr, X_hash_tr], format="csr")
    X_champ_va = sp.hstack([X_base_va, X_emo_va, X_neg_va, X_hash_va], format="csr")
    print(f"  Champion Feature Matrix: {X_champ_tr.shape} built in {time.time()-t0:.1f}s")

    # ── 3. Fit Model & Predict ────────────────────────────────────────────────
    print("\n[Step 3] Fitting ClassifierChain on training data...")
    t0 = time.time()
    clf   = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="saga", random_state=42)
    chain = ClassifierChain(clf, order=freq_order, random_state=42)
    chain.fit(X_champ_tr, Y_train)

    P_val       = chain.predict_proba(X_champ_va)
    thresholds  = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
    Y_pred      = apply_thresholds(P_val, thresholds, TARGET_COLS)
    metrics     = compute_metrics(Y_val, Y_pred, TARGET_COLS)

    print(f"  Fit Time: {time.time()-t0:.1f}s")
    print(f"  Val Macro-F1: {metrics['macro_f1']:.4f} | Micro-F1: {metrics['micro_f1']:.4f}")
    print("  Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics["per_class"].items()))
    print(f"  Per-class thresholds: {thresholds}")

    # ── 4. Error Pattern Classification ──────────────────────────────────────
    print("\n[Step 4] Classifying validation samples by surface patterns...")
    patterns = classify_error_patterns(val_raw, val_proc, EXPANDED_NEGATIONS)
    for pat_name, arr in patterns.items():
        print(f"  Val rows with '{pat_name}': {arr.sum():,} ({arr.mean()*100:.1f}%)")

    # ── 5. Disgust Error Analysis ─────────────────────────────────────────────
    print("\n[Step 5] Disgust Error Analysis...")
    fp_disgust, fn_disgust, dis_summary = summarize_errors(
        "disgust", Y_val, Y_pred, df_val, patterns, "disgust"
    )
    print(f"  Disgust FP: {dis_summary['total_fp']} | FN: {dis_summary['total_fn']}")
    print("  FP Pattern Breakdown:")
    for pat, stat in dis_summary["fp_patterns"].items():
        print(f"    {pat:<18}: {stat['count']:>4}  ({stat['pct']:>5.1f}%)")
    print("  FN Pattern Breakdown:")
    for pat, stat in dis_summary["fn_patterns"].items():
        print(f"    {pat:<18}: {stat['count']:>4}  ({stat['pct']:>5.1f}%)")

    # EmoNoBa Sensitivity Analysis for Disgust
    print("\n  EmoNoBa Disgust Assumption Sensitivity:")
    dis_sensitivity = sensitivity_analysis_emonoba("disgust", Y_val, Y_pred, df_val)
    if dis_sensitivity["emonoba_assumption_rows"]:
        r = dis_sensitivity["emonoba_assumption_rows"]
        print(f"    EmoNoBa-assumption rows (n={r['n']}): F1={r['f1']:.4f}, Prec={r['precision']:.4f}, Rec={r['recall']:.4f}")
    if dis_sensitivity["other_rows"]:
        r = dis_sensitivity["other_rows"]
        print(f"    Other rows             (n={r['n']}): F1={r['f1']:.4f}, Prec={r['precision']:.4f}, Rec={r['recall']:.4f}")

    # ── 6. Surprise Error Analysis ────────────────────────────────────────────
    print("\n[Step 6] Surprise Error Analysis...")
    fp_surprise, fn_surprise, sur_summary = summarize_errors(
        "surprise", Y_val, Y_pred, df_val, patterns, "surprise"
    )
    print(f"  Surprise FP: {sur_summary['total_fp']} | FN: {sur_summary['total_fn']}")
    print("  FP Pattern Breakdown:")
    for pat, stat in sur_summary["fp_patterns"].items():
        print(f"    {pat:<18}: {stat['count']:>4}  ({stat['pct']:>5.1f}%)")
    print("  FN Pattern Breakdown:")
    for pat, stat in sur_summary["fn_patterns"].items():
        print(f"    {pat:<18}: {stat['count']:>4}  ({stat['pct']:>5.1f}%)")

    # ── 7. Confusion Analysis ─────────────────────────────────────────────────
    print("\n[Step 7] Label Co-occurrence & Confusion Analysis...")
    co_occ_true, fp_confusion = confusion_matrix_pairs(Y_val, Y_pred, TARGET_COLS)

    print("  True label co-occurrences (Disgust and Surprise vs others):")
    for focus in ["disgust", "surprise"]:
        if focus in co_occ_true:
            sorted_co = sorted(co_occ_true[focus].items(), key=lambda x: -x[1])
            print(f"    {focus.upper()} co-occurs with: " + ", ".join(f"{l}={n}" for l, n in sorted_co))

    print("  FP Confusion (what was predicted as Disgust/Surprise when it was wrong):")
    for focus in ["disgust", "surprise"]:
        if focus in fp_confusion:
            sorted_fp = sorted(fp_confusion[focus].items(), key=lambda x: -x[1])
            print(f"    FP_{focus.upper()} actual labels: " + ", ".join(f"{l}={n}" for l, n in sorted_fp))

    # ── 8. Qualitative Examples ───────────────────────────────────────────────
    print("\n[Step 8] Qualitative FP/FN Examples (Top 5 each)...")
    for label, fp_df, fn_df in [("Disgust", fp_disgust, fn_disgust), ("Surprise", fp_surprise, fn_surprise)]:
        print(f"\n  --- {label} False Positives (Predicted={label}, Actual≠{label}) ---")
        for _, row in fp_df.head(5).iterrows():
            true_emos = [c for c in TARGET_COLS if row[c] == 1]
            print(f"    [{', '.join(true_emos) or 'none'}] {str(row['text'])[:100]}")
        print(f"\n  --- {label} False Negatives (Predicted≠{label}, Actual={label}) ---")
        for _, row in fn_df.head(5).iterrows():
            true_emos = [c for c in TARGET_COLS if row[c] == 1]
            print(f"    [{', '.join(true_emos)}] {str(row['text'])[:100]}")

    # ── 9. Source Dataset Breakdown for Errors ────────────────────────────────
    print("\n[Step 9] Source Dataset Breakdown of Errors...")
    source_analysis = {}
    for label, fp_df, fn_df in [("disgust", fp_disgust, fn_disgust), ("surprise", fp_surprise, fn_surprise)]:
        fp_sources = dict(fp_df["source_dataset"].value_counts())
        fn_sources = dict(fn_df["source_dataset"].value_counts())
        source_analysis[label] = {"fp_by_source": fp_sources, "fn_by_source": fn_sources}
        print(f"  {label.upper()} FP by source: {fp_sources}")
        print(f"  {label.upper()} FN by source: {fn_sources}")

    # ── 10. Prioritized Improvement Proposals ─────────────────────────────────
    print("\n[Step 10] Generating Improvement Proposals...")
    improvement_proposals = []

    # Based on FP/FN patterns, generate actionable proposals
    dis_fp_neg_pct  = dis_summary["fp_patterns"]["has_negation"]["pct"]
    dis_fn_short_pct = dis_summary["fn_patterns"]["short_text"]["pct"]
    sur_fn_emo_pct   = sur_summary["fn_patterns"]["has_emoji"]["pct"]

    proposals = [
        {
            "priority": 1,
            "target": "Disgust",
            "hypothesis": "Disgust FPs overlap with anger/sadness in co-occurring multi-label texts. Add per-class C regularization search or raise disgust threshold.",
            "action": "Threshold fine-tuning per class in Phase 7 (optimize_thresholds sweep).",
            "experiment": "Phase 7 threshold optimization with wider sweep [0.05 – 0.95, step 0.005] for Disgust."
        },
        {
            "priority": 2,
            "target": "Disgust",
            "hypothesis": f"EmoNoBa disgust assumption sensitivity: rows derived from non-native annotation may have lower F1. Validate via sensitivity analysis.",
            "action": "Compare Disgust F1 on emonoba_disgust_assumption=True vs =False. If large gap, consider separate C value or label smoothing for those rows.",
            "experiment": "Sensitivity analysis (done in Step 5 above). Inspect per-source F1 for emonoba."
        },
        {
            "priority": 3,
            "target": "Surprise",
            "hypothesis": f"Surprise is often expressed with emoji ({sur_fn_emo_pct:.1f}% of FN have emoji). Adding Surprise-specific emoji features may help.",
            "action": "Extend emoji feature set to include surprise-specific symbols: 😮 😲 🤯 😱 !!.",
            "experiment": "Targeted Surprise-emoji feature extraction and ablation (Phase 5.2 extension)."
        },
        {
            "priority": 4,
            "target": "Disgust + Surprise",
            "hypothesis": "Short texts (< 5 words) often express Disgust/Surprise with single exclamation words or expressive particles.",
            "action": "Add character n-gram boost for short texts, or add a 'short-text' indicator feature.",
            "experiment": "Test short-text binary flag + expanded char-ngram range [2-6] in ablation."
        },
        {
            "priority": 5,
            "target": "General",
            "hypothesis": "Class imbalance (Disgust 2,845; Surprise 1,749 vs Joy 10,368) causes systematic under-prediction of minority classes.",
            "action": "Phase 8: Test class_weight overrides and targeted oversampling for Disgust & Surprise.",
            "experiment": "Phase 8 class imbalance experiments."
        },
    ]

    for p in proposals:
        print(f"  P{p['priority']} [{p['target']}]: {p['hypothesis'][:100]}...")
        improvement_proposals.append(p)

    # ── 11. Save Outputs ──────────────────────────────────────────────────────
    print("\n[Step 11] Saving Results...")

    # JSON Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_pipeline": {
            "feature_dims": X_champ_tr.shape[1],
            "val_macro_f1": metrics["macro_f1"],
            "val_micro_f1": metrics["micro_f1"],
            "per_class": metrics["per_class"],
            "thresholds": thresholds
        },
        "pattern_prevalence_val": {k: int(v.sum()) for k, v in patterns.items()},
        "disgust_errors": {
            **dis_summary,
            "emonoba_sensitivity": dis_sensitivity,
            "fp_by_source": source_analysis["disgust"]["fp_by_source"],
            "fn_by_source": source_analysis["disgust"]["fn_by_source"]
        },
        "surprise_errors": {
            **sur_summary,
            "fp_by_source": source_analysis["surprise"]["fp_by_source"],
            "fn_by_source": source_analysis["surprise"]["fn_by_source"]
        },
        "co_occurrence_true": {k: dict(v) for k, v in co_occ_true.items()},
        "fp_confusion_map": {k: dict(v) for k, v in fp_confusion.items()},
        "improvement_proposals": improvement_proposals
    }

    # Custom encoder to handle np.int64 / np.int32 etc.
    class NumpyEncoder(JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    report_path = os.path.join(RESULTS_DIR, "phase6_error_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"  JSON report saved: {report_path}")

    # Save FP/FN CSVs
    for label, fp_df, fn_df in [("disgust", fp_disgust, fn_disgust), ("surprise", fp_surprise, fn_surprise)]:
        fp_path = os.path.join(RESULTS_DIR, f"fp_{label}.csv")
        fn_path = os.path.join(RESULTS_DIR, f"fn_{label}.csv")
        cols_to_save = ["text", "processed_text", "source_dataset", "emonoba_disgust_assumption"] + TARGET_COLS
        cols_to_save = [c for c in cols_to_save if c in fp_df.columns]
        fp_df[cols_to_save].to_csv(fp_path, index=False, encoding="utf-8-sig")
        fn_df[cols_to_save].to_csv(fn_path, index=False, encoding="utf-8-sig")
        print(f"  Saved FP {label}: {fp_path} ({len(fp_df)} rows)")
        print(f"  Saved FN {label}: {fn_path} ({len(fn_df)} rows)")

    # ── 12. Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PHASE 6 — ERROR ANALYSIS SUMMARY")
    print("=" * 75)
    print(f"\nChampion Pipeline Val Macro-F1: {metrics['macro_f1']:.4f} | Micro-F1: {metrics['micro_f1']:.4f}")
    print(f"\nDISGUST Errors:  FP={dis_summary['total_fp']}  FN={dis_summary['total_fn']}")
    print(f"  Top FP patterns: " + ", ".join(
        f"{k}={v['count']}({v['pct']}%)" for k, v in sorted(dis_summary["fp_patterns"].items(), key=lambda x: -x[1]["count"])[:3]
    ))
    print(f"  EmoNoBa assumption rows F1: {dis_sensitivity.get('emonoba_assumption_rows', {}).get('f1', 'N/A')}")
    print(f"  Other rows F1:             {dis_sensitivity.get('other_rows', {}).get('f1', 'N/A')}")
    print(f"\nSURPRISE Errors: FP={sur_summary['total_fp']}  FN={sur_summary['total_fn']}")
    print(f"  Top FP patterns: " + ", ".join(
        f"{k}={v['count']}({v['pct']}%)" for k, v in sorted(sur_summary["fp_patterns"].items(), key=lambda x: -x[1]["count"])[:3]
    ))
    print("\nImprovement Proposals (by Priority):")
    for p in improvement_proposals:
        print(f"  P{p['priority']} [{p['target']}]: {p['action']}")
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
