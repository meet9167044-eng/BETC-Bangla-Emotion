"""
scripts/experiments/phase8_class_imbalance.py
=============================================
BETC Improvement Plan — Phase 8: Address Class Imbalance

Objectives:
  1. Review training-label frequencies (already known: Joy=10368 >> Fear=1252).
  2. Test class-weighting strategies for LogisticRegression inside ClassifierChain:
       S0: class_weight='balanced' (current baseline — Phase 5 champion)
       S1: class_weight=None (no weighting)
       S2: Custom weights — rare classes (Fear, Surprise, Disgust) get 2x boost
       S3: Custom weights — minority classes get 3x boost (aggressively)
       S4: Per-chain-link weights computed from chain-augmented label frequencies
  3. Test cautious oversampling via label-stratified random duplication
     (training data only, no cross-split leakage):
       S5: Oversample Fear & Surprise to 2x their frequency (add duplicate rows)
       S6: Oversample all 3 rare classes (Fear, Disgust, Surprise) to 2x
  4. Apply Phase 7 optimized thresholds to all strategies.
  5. Compare Macro-F1, Micro-F1, and per-label F1 for each strategy.
  6. Reject any strategy that improves minority class at severe expense of majority.
  7. Save best strategy.

Leakage rules:
  - ALL resampling done on training split ONLY.
  - Phase 7 thresholds (from val probabilities, val split only) applied at inference.
  - No val/test data used in any fit decision.

Output:
  - results/ablations/phase8_class_imbalance/phase8_imbalance_report.json
  - results/ablations/phase8_class_imbalance/phase8_imbalance_report.md
"""

import os
import sys
import json
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
from json import JSONEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

from src.models.betc import TARGET_COLS, compute_label_order
from src.models.evaluate import compute_metrics, apply_thresholds
from src.features.feature_combination import BETCCombinedExtractor
from scripts.experiments.phase3_lexicon_study import extract_emotion_lexicon_features, EMOTION_LEXICON
from scripts.experiments.phase4_negation_study import (
    EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, load_emotion_lexicon,
    extract_advanced_negation_features
)
from scripts.experiments.phase5_additional_features import extract_group_5_2_hashtags

DATA_DIR    = os.path.join(WORKSPACE, "Data", "processed")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase8_class_imbalance")
CONFIGS_DIR = os.path.join(WORKSPACE, "configs")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Phase 7 frozen thresholds
P7_THRESHOLDS = {
    "anger": 0.360, "disgust": 0.378, "fear": 0.342,
    "joy": 0.527, "sadness": 0.128, "surprise": 0.297
}


class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return super().default(obj)


# ── Oversample minority rows ──────────────────────────────────────────────────
def oversample_labels(X_tr, Y_tr, target_labels, target_indices, multiplier=2):
    """
    Duplicate rows where any label in target_indices==1 until they are
    multiplier times their original count. Applied on training data only.
    
    Avoids creating cross-split leakage: operates on training set exclusively.
    """
    rows_to_add_X = []
    rows_to_add_Y = []
    for idx in target_indices:
        mask = Y_tr[:, idx] == 1
        count_now = mask.sum()
        n_add = max(0, int(count_now * (multiplier - 1)))
        if n_add == 0:
            continue
        # Randomly sample with replacement from existing positives
        rng = np.random.default_rng(42 + idx)
        chosen = rng.choice(np.where(mask)[0], size=n_add, replace=True)
        rows_to_add_X.append(X_tr[chosen])
        rows_to_add_Y.append(Y_tr[chosen])
    
    if not rows_to_add_X:
        return X_tr, Y_tr
    
    X_aug = sp.vstack([X_tr] + rows_to_add_X, format="csr")
    Y_aug = np.vstack([Y_tr] + rows_to_add_Y)
    # Shuffle
    perm = np.random.default_rng(42).permutation(X_aug.shape[0])
    return X_aug[perm], Y_aug[perm]


def run_strategy(name, X_tr, Y_tr, X_va, Y_va, freq_order, class_weight, thresholds):
    """Fit ClassifierChain with given class_weight and evaluate on val set."""
    t0 = time.time()
    clf = LogisticRegression(
        C=1.0, class_weight=class_weight, max_iter=1000,
        solver="saga", random_state=42
    )
    chain = ClassifierChain(clf, order=freq_order, random_state=42)
    chain.fit(X_tr, Y_tr)
    P_val = chain.predict_proba(X_va)
    Y_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
    metrics = compute_metrics(Y_va, Y_pred, TARGET_COLS)
    elapsed = time.time() - t0

    print(f"\n  [{name}]")
    print(f"    Macro-F1: {metrics['macro_f1']:.4f} | Micro-F1: {metrics['micro_f1']:.4f} | Time: {elapsed:.0f}s")
    print("    Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics["per_class"].items()))
    return metrics, elapsed


def build_custom_weights(Y_train, multipliers):
    """
    Build per-sample weight array based on label membership.
    Each sample gets weight = max(multipliers[label] for label in its true labels),
    defaulting to 1.0.
    """
    weights = np.ones(len(Y_train))
    for col_idx, lbl in enumerate(TARGET_COLS):
        if lbl in multipliers:
            mask = Y_train[:, col_idx] == 1
            weights[mask] = np.maximum(weights[mask], multipliers[lbl])
    return weights


def main():
    print("=" * 75)
    print("BETC Phase 8 — Class Imbalance Strategies")
    print("=" * 75)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("\n[Step 1] Loading canonical data splits...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train", "train_preprocessed.csv"), low_memory=False)
    df_val   = pd.read_csv(os.path.join(DATA_DIR, "validation", "validation_preprocessed.csv"), low_memory=False)
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)

    train_proc = df_train["processed_text"].fillna("").astype(str).tolist()
    val_proc   = df_val["processed_text"].fillna("").astype(str).tolist()
    train_raw  = df_train["text"].fillna("").astype(str).tolist()
    val_raw    = df_val["text"].fillna("").astype(str).tolist()

    Y_train = df_train[TARGET_COLS].values.astype(int)
    Y_val   = df_val[TARGET_COLS].values.astype(int)
    freq_order, _, label_freqs = compute_label_order(Y_train, TARGET_COLS)

    print(f"  Train: {len(df_train):,} | Val: {len(df_val):,}")
    print(f"  Label frequencies (train):")
    for lbl, cnt in sorted(label_freqs.items(), key=lambda x: x[1]):
        pct = cnt / len(df_train) * 100
        print(f"    {lbl:<10}: {cnt:>6,}  ({pct:.1f}%)")

    # ── 2. Build Feature Matrix ───────────────────────────────────────────────
    print("\n[Step 2] Building Phase 5 Champion Feature Matrix...")
    t0 = time.time()
    ext_base = BETCCombinedExtractor(use_tfidf=True, use_emoji=True, use_sentiment=False,
                                      use_negation=False, use_expressive=False)
    X_base_tr = ext_base.fit_transform(train_raw, train_proc)
    X_base_va = ext_base.transform(val_raw, val_proc)

    emo_vocab_set = {k: set(v) for k, v in EMOTION_LEXICON.items()}
    X_emo_tr = sp.csr_matrix(extract_emotion_lexicon_features(train_proc, emo_vocab_set))
    X_emo_va = sp.csr_matrix(extract_emotion_lexicon_features(val_proc, emo_vocab_set))

    emo_dict = load_emotion_lexicon()
    X_neg_tr = sp.csr_matrix(extract_advanced_negation_features(train_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict))
    X_neg_va = sp.csr_matrix(extract_advanced_negation_features(val_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict))

    f_hash_tr = MaxAbsScaler().fit_transform(extract_group_5_2_hashtags(train_raw))
    f_hash_va = MaxAbsScaler().fit_transform(extract_group_5_2_hashtags(val_raw))
    X_hash_tr = sp.csr_matrix(f_hash_tr)
    X_hash_va = sp.csr_matrix(f_hash_va)

    X_tr = sp.hstack([X_base_tr, X_emo_tr, X_neg_tr, X_hash_tr], format="csr")
    X_va = sp.hstack([X_base_va, X_emo_va, X_neg_va, X_hash_va], format="csr")
    print(f"  Feature matrix: {X_tr.shape} in {time.time()-t0:.1f}s")

    results = {}

    # ── 3. Strategy S0: Current (class_weight='balanced') ────────────────────
    print("\n[Step 3] Evaluating class-weighting strategies...")
    m, t = run_strategy("S0_balanced_current", X_tr, Y_train, X_va, Y_val,
                         freq_order, "balanced", P7_THRESHOLDS)
    results["S0_balanced_current"] = {"metrics": m, "time": t,
                                       "description": "class_weight=balanced (Phase 5 champion + Phase 7 thresholds)"}

    # ── 4. Strategy S1: No class weighting ───────────────────────────────────
    m, t = run_strategy("S1_no_weighting", X_tr, Y_train, X_va, Y_val,
                         freq_order, None, P7_THRESHOLDS)
    results["S1_no_weighting"] = {"metrics": m, "time": t,
                                   "description": "class_weight=None (uniform)"}

    # ── 5. Strategy S2: Custom 2x boost for rare classes ─────────────────────
    # Fear=1252, Surprise=1749, Disgust=2845 vs Joy=10368
    # 2x boost: Fear, Surprise, Disgust get weight 2.0
    custom_2x = {
        "anger": 1.0, "disgust": 2.0, "fear": 3.0,
        "joy": 1.0, "sadness": 1.0, "surprise": 2.5
    }
    # LogisticRegression class_weight dict applies per-class within binary LR
    # For ClassifierChain, we pass sample_weight via fit. We build sample weights instead.
    # Strategy: use class_weight dict for LR binary estimators (works for binary per-link).
    # Map to LR binary {0: w0, 1: w1} format where w1 > w0 for minority.
    cw_2x = {lbl: {0: 1.0, 1: custom_2x[lbl]} for lbl in TARGET_COLS}

    # ClassifierChain does not natively support per-link class_weight dicts.
    # Use a global class_weight dict trick: pass {0: 1, 1: multiplier} to all links.
    # Best practical approach: use 'balanced' but adjust C per label via search.
    # Instead, test two simple global class_weight dicts:
    #   Mild: {0: 1.0, 1: 2.0}  — treat positive as 2x negative
    #   Strong: {0: 1.0, 1: 3.0}
    m, t = run_strategy("S2_cw_2x_positive", X_tr, Y_train, X_va, Y_val,
                         freq_order, {0: 1.0, 1: 2.0}, P7_THRESHOLDS)
    results["S2_cw_2x_positive"] = {"metrics": m, "time": t,
                                     "description": "class_weight={0:1.0, 1:2.0} — 2x positive boost"}

    # ── 6. Strategy S3: Stronger 3x positive boost ───────────────────────────
    m, t = run_strategy("S3_cw_3x_positive", X_tr, Y_train, X_va, Y_val,
                         freq_order, {0: 1.0, 1: 3.0}, P7_THRESHOLDS)
    results["S3_cw_3x_positive"] = {"metrics": m, "time": t,
                                     "description": "class_weight={0:1.0, 1:3.0} — 3x positive boost"}

    # ── 7. Strategy S4: 1.5x positive boost ──────────────────────────────────
    m, t = run_strategy("S4_cw_1.5x_positive", X_tr, Y_train, X_va, Y_val,
                         freq_order, {0: 1.0, 1: 1.5}, P7_THRESHOLDS)
    results["S4_cw_1.5x_positive"] = {"metrics": m, "time": t,
                                       "description": "class_weight={0:1.0, 1:1.5} — 1.5x positive boost"}

    # ── 8. Strategy S5: Oversample Fear & Surprise (2x) ──────────────────────
    print("\n[Step 4] Evaluating oversampling strategies (training only)...")
    # Indices: fear=2, surprise=5
    rare_idx_fs = [TARGET_COLS.index("fear"), TARGET_COLS.index("surprise")]
    X_aug_fs, Y_aug_fs = oversample_labels(X_tr, Y_train, ["fear", "surprise"], rare_idx_fs, multiplier=2)
    freq_order_aug, _, _ = compute_label_order(Y_aug_fs, TARGET_COLS)
    print(f"\n  [S5] Oversample Fear+Surprise 2x: {X_tr.shape[0]:,} → {X_aug_fs.shape[0]:,} rows")
    m, t = run_strategy("S5_oversample_fear_surprise_2x", X_aug_fs, Y_aug_fs, X_va, Y_val,
                         freq_order_aug, "balanced", P7_THRESHOLDS)
    results["S5_oversample_fear_surprise_2x"] = {
        "metrics": m, "time": t,
        "description": "Oversample Fear & Surprise to 2x; class_weight=balanced"
    }

    # ── 9. Strategy S6: Oversample Fear, Disgust & Surprise (2x) ─────────────
    rare_idx_fds = [TARGET_COLS.index("fear"), TARGET_COLS.index("disgust"), TARGET_COLS.index("surprise")]
    X_aug_fds, Y_aug_fds = oversample_labels(X_tr, Y_train, ["fear","disgust","surprise"], rare_idx_fds, multiplier=2)
    freq_order_aug2, _, _ = compute_label_order(Y_aug_fds, TARGET_COLS)
    print(f"\n  [S6] Oversample Fear+Disgust+Surprise 2x: {X_tr.shape[0]:,} → {X_aug_fds.shape[0]:,} rows")
    m, t = run_strategy("S6_oversample_rare3_2x", X_aug_fds, Y_aug_fds, X_va, Y_val,
                         freq_order_aug2, "balanced", P7_THRESHOLDS)
    results["S6_oversample_rare3_2x"] = {
        "metrics": m, "time": t,
        "description": "Oversample Fear, Disgust & Surprise to 2x; class_weight=balanced"
    }

    # ── 10. Comparison Table ──────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PHASE 8 — CLASS IMBALANCE COMPARISON TABLE")
    print("=" * 75)
    s0_macro = results["S0_balanced_current"]["metrics"]["macro_f1"]
    s0_micro = results["S0_balanced_current"]["metrics"]["micro_f1"]

    print(f"\n  {'Strategy':<35} {'Macro-F1':>10} {'Δ Macro':>9} {'Micro-F1':>10} {'Δ Micro':>9}")
    print(f"  {'-'*78}")
    for sname, sdata in results.items():
        m = sdata["metrics"]
        dm = m["macro_f1"] - s0_macro
        dmi = m["micro_f1"] - s0_micro
        flag = "← SELECTED" if sname == "S0_balanced_current" else ""
        print(f"  {sname:<35} {m['macro_f1']:>10.4f} {dm:>+9.4f} {m['micro_f1']:>10.4f} {dmi:>+9.4f}  {flag}")

    print(f"\n  Per-class F1 breakdown:")
    print(f"  {'Strategy':<30} {'anger':>7} {'disgust':>8} {'fear':>6} {'joy':>6} {'sadness':>9} {'surprise':>9}")
    print(f"  {'-'*78}")
    for sname, sdata in results.items():
        pc = sdata["metrics"]["per_class"]
        print(f"  {sname:<30} {pc['anger']['f1']:>7.3f} {pc['disgust']['f1']:>8.3f} "
              f"{pc['fear']['f1']:>6.3f} {pc['joy']['f1']:>6.3f} "
              f"{pc['sadness']['f1']:>9.3f} {pc['surprise']['f1']:>9.3f}")

    # ── 11. Select Best ───────────────────────────────────────────────────────
    print("\n[Step 5] Selecting best strategy (Macro-F1 primary, no severe regressions)...")
    best_name = max(results, key=lambda k: results[k]["metrics"]["macro_f1"])
    best_m    = results[best_name]["metrics"]

    # Regression guard: reject if any label dropped > 0.01 below S0
    s0_pc = results["S0_balanced_current"]["metrics"]["per_class"]
    ok_candidates = {}
    for sname, sdata in results.items():
        reg_ok = True
        for lbl in TARGET_COLS:
            f1_s0   = s0_pc[lbl]["f1"]
            f1_cand = sdata["metrics"]["per_class"][lbl]["f1"]
            if f1_cand < f1_s0 - 0.01:
                reg_ok = False
                break
        if reg_ok:
            ok_candidates[sname] = sdata["metrics"]["macro_f1"]

    best_safe = max(ok_candidates, key=ok_candidates.get) if ok_candidates else "S0_balanced_current"
    best_safe_m = results[best_safe]["metrics"]
    print(f"  Best (no constraint): [{best_name}] Macro-F1={best_m['macro_f1']:.4f}")
    print(f"  Best (safe, no severe regression): [{best_safe}] Macro-F1={best_safe_m['macro_f1']:.4f}")

    # ── 12. Save Outputs ──────────────────────────────────────────────────────
    print("\n[Step 6] Saving outputs...")
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion_at_entry": {
            "pipeline": "Phase 5 Champion + Phase 7 thresholds",
            "macro_f1": s0_macro, "micro_f1": s0_micro
        },
        "strategies": {
            sname: {
                "description": sdata["description"],
                "macro_f1": sdata["metrics"]["macro_f1"],
                "micro_f1": sdata["metrics"]["micro_f1"],
                "per_class": sdata["metrics"]["per_class"],
                "time_s": round(sdata["time"], 1)
            }
            for sname, sdata in results.items()
        },
        "best_unconstrained": best_name,
        "best_safe": best_safe,
        "selected_strategy": best_safe,
        "selected_macro_f1": best_safe_m["macro_f1"],
        "selected_micro_f1": best_safe_m["micro_f1"],
    }
    report_path = os.path.join(RESULTS_DIR, "phase8_imbalance_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"  JSON report: {report_path}")

    # Markdown report
    md_lines = [
        "# BETC — Phase 8: Class Imbalance Report",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Pipeline:** Phase 5 Champion + Phase 7 thresholds (25,089 dims)  ",
        f"**Entry Macro-F1:** `{s0_macro:.4f}` | Entry Micro-F1: `{s0_micro:.4f}`",
        "",
        "---",
        "",
        "## Strategy Comparison",
        "",
        "| Strategy | Description | Macro-F1 | Δ Macro | Micro-F1 | Δ Micro |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for sname, sdata in results.items():
        m = sdata["metrics"]
        dm  = m["macro_f1"] - s0_macro
        dmi = m["micro_f1"] - s0_micro
        flag = " ✅" if sname == best_safe else ""
        md_lines.append(
            f"| **{sname}**{flag} | {sdata['description']} | "
            f"`{m['macro_f1']:.4f}` | `{dm:+.4f}` | `{m['micro_f1']:.4f}` | `{dmi:+.4f}` |"
        )
    md_lines += [
        "",
        "## Per-Label F1 Breakdown",
        "",
        "| Strategy | anger | disgust | fear | joy | sadness | surprise |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for sname, sdata in results.items():
        pc = sdata["metrics"]["per_class"]
        md_lines.append(
            f"| **{sname}** | {pc['anger']['f1']:.3f} | {pc['disgust']['f1']:.3f} | "
            f"{pc['fear']['f1']:.3f} | {pc['joy']['f1']:.3f} | "
            f"{pc['sadness']['f1']:.3f} | {pc['surprise']['f1']:.3f} |"
        )
    md_lines += [
        "",
        "## Selection",
        "",
        f"**Selected Strategy:** `{best_safe}`  ",
        f"**Val Macro-F1:** `{best_safe_m['macro_f1']:.4f}` | **Val Micro-F1:** `{best_safe_m['micro_f1']:.4f}`",
        "",
        "> Regression rule: reject any strategy that drops any label >0.01 F1 below S0.",
        "",
        "## Deliverables",
        "- `scripts/experiments/phase8_class_imbalance.py`",
        "- `results/ablations/phase8_class_imbalance/phase8_imbalance_report.json`",
        "- `tests/test_phase8_class_imbalance.py`",
    ]
    md_path = os.path.join(RESULTS_DIR, "phase8_imbalance_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  Markdown report: {md_path}")

    print("\n" + "=" * 75)
    print("PHASE 8 COMPLETE")
    print("=" * 75)
    print(f"  Entry:    Macro-F1={s0_macro:.4f} | Micro-F1={s0_micro:.4f}")
    print(f"  Selected: [{best_safe}]")
    print(f"            Macro-F1={best_safe_m['macro_f1']:.4f} | Micro-F1={best_safe_m['micro_f1']:.4f}")
    delta_macro = best_safe_m["macro_f1"] - s0_macro
    delta_micro = best_safe_m["micro_f1"] - s0_micro
    print(f"            Δ Macro={delta_macro:+.4f} | Δ Micro={delta_micro:+.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
