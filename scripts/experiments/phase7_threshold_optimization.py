"""
scripts/experiments/phase7_threshold_optimization.py
=====================================================
BETC Improvement Plan — Phase 7: Optimize Per-Label Thresholds

Objectives:
  1. Re-build Phase 5 Champion Pipeline and save validation probabilities.
  2. Compare fixed τ=0.5 vs existing optimize_thresholds (step=0.01).
  3. Run a finer threshold sweep (step=0.005) for Disgust & Surprise
     (Phase 6 flagged these as needing adjustment).
  4. Evaluate the macro-F1 impact of each threshold configuration:
       A. All τ=0.5 (baseline)
       B. Per-class optimized (step=0.01) — current pipeline
       C. Per-class optimized (step=0.005) — fine sweep
       D. Disgust & Surprise thresholds raised by +0.05 over C (FP-reduction)
  5. Inspect per-label F1, precision, recall for each config.
  6. Freeze best thresholds to configs/phase7_thresholds.json.

Leakage rule: ALL threshold decisions use ONLY validation probabilities.

Output:
  - results/ablations/phase7_thresholds/phase7_threshold_report.json
  - results/ablations/phase7_thresholds/val_probabilities.npy
  - configs/phase7_thresholds.json  (frozen threshold file)
"""

import os
import sys
import io
import json
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score, precision_score, recall_score

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
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase7_thresholds")
CONFIGS_DIR = os.path.join(WORKSPACE, "configs")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Threshold sweep utilities ─────────────────────────────────────────────────

def sweep_thresholds(P_val, Y_val, tau_min=0.05, tau_max=0.95, tau_step=0.01, label_names=None):
    """Per-class F1-maximizing threshold sweep on validation probabilities only."""
    if label_names is None:
        label_names = [f"label_{i}" for i in range(Y_val.shape[1])]
    taus = np.round(np.arange(tau_min, tau_max + tau_step / 2, tau_step), 5)
    thresholds = {}
    best_f1s   = {}
    for k, name in enumerate(label_names):
        best_tau = 0.5
        best_f1  = -1.0
        for tau in taus:
            y_pred_k = (P_val[:, k] >= tau).astype(int)
            f1 = float(f1_score(Y_val[:, k], y_pred_k, zero_division=0))
            if f1 > best_f1:
                best_f1  = f1
                best_tau = float(round(tau, 5))
        thresholds[name] = best_tau
        best_f1s[name]   = round(best_f1, 4)
    return thresholds, best_f1s


def eval_with_thresholds(P_val, Y_val, thresholds, label_names=None):
    """Evaluate full metric set given a threshold dict."""
    if label_names is None:
        label_names = TARGET_COLS
    Y_pred = apply_thresholds(P_val, thresholds, label_names)
    return compute_metrics(Y_val, Y_pred, label_names)


def threshold_f1_curve(P_val, Y_val, label, tau_range=None):
    """Return list of (tau, f1) pairs for a single label — for curve analysis."""
    if tau_range is None:
        tau_range = np.round(np.arange(0.05, 0.96, 0.005), 5)
    col_idx = TARGET_COLS.index(label)
    curve = []
    for tau in tau_range:
        y_pred = (P_val[:, col_idx] >= tau).astype(int)
        f1 = float(f1_score(Y_val[:, col_idx], y_pred, zero_division=0))
        curve.append((round(float(tau), 3), round(f1, 4)))
    return curve


def main():
    print("=" * 75)
    print("BETC Phase 7 — Per-Label Threshold Optimization")
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
    freq_order, _, _ = compute_label_order(Y_train, TARGET_COLS)

    # ── 2. Build Phase 5 Champion Feature Matrix ─────────────────────────────
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

    X_champ_tr = sp.hstack([X_base_tr, X_emo_tr, X_neg_tr, X_hash_tr], format="csr")
    X_champ_va = sp.hstack([X_base_va, X_emo_va, X_neg_va, X_hash_va], format="csr")
    print(f"  Champion Feature Matrix: {X_champ_tr.shape} in {time.time()-t0:.1f}s")

    # ── 3. Fit ClassifierChain & Save Validation Probabilities ───────────────
    print("\n[Step 3] Fitting ClassifierChain and saving validation probabilities...")
    t0 = time.time()
    clf   = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="saga", random_state=42)
    chain = ClassifierChain(clf, order=freq_order, random_state=42)
    chain.fit(X_champ_tr, Y_train)
    P_val = chain.predict_proba(X_champ_va)   # shape (6170, 6)
    print(f"  Fit Time: {time.time()-t0:.1f}s | P_val shape: {P_val.shape}")
    print(f"  Probability ranges per class:")
    for i, c in enumerate(TARGET_COLS):
        print(f"    {c:<10}: min={P_val[:,i].min():.3f}  mean={P_val[:,i].mean():.3f}  max={P_val[:,i].max():.3f}")

    # Save probabilities
    prob_path = os.path.join(RESULTS_DIR, "val_probabilities.npy")
    np.save(prob_path, P_val)
    print(f"  Validation probabilities saved: {prob_path}")

    # ── 4. Threshold Configurations ──────────────────────────────────────────
    print("\n[Step 4] Evaluating threshold configurations...")

    # Config A: All τ = 0.5
    thresholds_A = {c: 0.5 for c in TARGET_COLS}
    metrics_A    = eval_with_thresholds(P_val, Y_val, thresholds_A)
    print(f"\n  [A] All τ=0.5 (Baseline)")
    print(f"      Macro-F1: {metrics_A['macro_f1']:.4f} | Micro-F1: {metrics_A['micro_f1']:.4f}")
    print("      Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_A["per_class"].items()))

    # Config B: Per-class optimized (step=0.01) — existing pipeline
    thresholds_B, best_f1s_B = sweep_thresholds(P_val, Y_val, tau_step=0.01, label_names=TARGET_COLS)
    metrics_B = eval_with_thresholds(P_val, Y_val, thresholds_B)
    print(f"\n  [B] Per-class optimized (step=0.01)")
    print(f"      Thresholds: {thresholds_B}")
    print(f"      Macro-F1: {metrics_B['macro_f1']:.4f} | Micro-F1: {metrics_B['micro_f1']:.4f}")
    print("      Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_B["per_class"].items()))

    # Config C: Per-class optimized (finer step=0.005)
    thresholds_C, best_f1s_C = sweep_thresholds(P_val, Y_val, tau_step=0.005, label_names=TARGET_COLS)
    metrics_C = eval_with_thresholds(P_val, Y_val, thresholds_C)
    print(f"\n  [C] Per-class optimized (finer step=0.005)")
    print(f"      Thresholds: {thresholds_C}")
    print(f"      Macro-F1: {metrics_C['macro_f1']:.4f} | Micro-F1: {metrics_C['micro_f1']:.4f}")
    print("      Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_C["per_class"].items()))

    # Config D: Based on C, but raise Disgust & Surprise thresholds +0.05
    # (Phase 6 showed these have excessive FPs — raising threshold cuts FPs)
    thresholds_D = dict(thresholds_C)
    thresholds_D["disgust"]  = min(0.90, round(thresholds_C["disgust"]  + 0.05, 3))
    thresholds_D["surprise"] = min(0.90, round(thresholds_C["surprise"] + 0.05, 3))
    metrics_D = eval_with_thresholds(P_val, Y_val, thresholds_D)
    print(f"\n  [D] Config C + Disgust/Surprise +0.05 (FP-reduction, Phase 6 insight)")
    print(f"      Thresholds: {thresholds_D}")
    print(f"      Macro-F1: {metrics_D['macro_f1']:.4f} | Micro-F1: {metrics_D['micro_f1']:.4f}")
    print("      Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_D["per_class"].items()))

    # Config E: Independent F1-max sweep at ultra-fine step (0.001) per class
    thresholds_E, best_f1s_E = sweep_thresholds(P_val, Y_val, tau_step=0.001, label_names=TARGET_COLS)
    metrics_E = eval_with_thresholds(P_val, Y_val, thresholds_E)
    print(f"\n  [E] Per-class ultra-fine (step=0.001)")
    print(f"      Thresholds: {thresholds_E}")
    print(f"      Macro-F1: {metrics_E['macro_f1']:.4f} | Micro-F1: {metrics_E['micro_f1']:.4f}")
    print("      Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_E["per_class"].items()))

    # ── 5. F1 Curves for Disgust & Surprise ──────────────────────────────────
    print("\n[Step 5] Computing F1-vs-threshold curves for Disgust and Surprise...")
    tau_range = np.round(np.arange(0.05, 0.955, 0.005), 5)
    dis_curve = threshold_f1_curve(P_val, Y_val, "disgust", tau_range)
    sur_curve = threshold_f1_curve(P_val, Y_val, "surprise", tau_range)

    # Find peak for each
    dis_peak = max(dis_curve, key=lambda x: x[1])
    sur_peak = max(sur_curve, key=lambda x: x[1])
    print(f"  Disgust  peak F1={dis_peak[1]:.4f} at τ={dis_peak[0]:.3f}")
    print(f"  Surprise peak F1={sur_peak[1]:.4f} at τ={sur_peak[0]:.3f}")

    # ── 6. Select Best Configuration ─────────────────────────────────────────
    print("\n[Step 6] Selecting best threshold configuration...")
    configs = {
        "A_fixed_0.5":             metrics_A,
        "B_optimized_step0.01":    metrics_B,
        "C_optimized_step0.005":   metrics_C,
        "D_disgust_surprise_boosted": metrics_D,
        "E_ultra_fine_step0.001":  metrics_E,
    }

    best_config_name = max(configs, key=lambda k: configs[k]["macro_f1"])
    best_metrics     = configs[best_config_name]
    best_thresholds_map = {
        "A_fixed_0.5": thresholds_A,
        "B_optimized_step0.01": thresholds_B,
        "C_optimized_step0.005": thresholds_C,
        "D_disgust_surprise_boosted": thresholds_D,
        "E_ultra_fine_step0.001": thresholds_E,
    }
    best_thresholds = best_thresholds_map[best_config_name]

    print(f"\n  Best Configuration: [{best_config_name}]")
    print(f"  Macro-F1: {best_metrics['macro_f1']:.4f} | Micro-F1: {best_metrics['micro_f1']:.4f}")
    print(f"  Thresholds: {best_thresholds}")

    # ── 7. Before/After Comparison Table ─────────────────────────────────────
    print("\n[Step 7] Before (τ=0.5) vs After (best config) comparison...")
    print(f"\n  {'Label':<12} {'τ=0.5 F1':>10} {'Best F1':>10} {'Δ':>8}  {'τ selected':>12}")
    print(f"  {'-'*58}")
    for lbl in TARGET_COLS:
        f1_before = metrics_A["per_class"][lbl]["f1"]
        f1_after  = best_metrics["per_class"][lbl]["f1"]
        delta     = f1_after - f1_before
        tau_sel   = best_thresholds[lbl]
        sign = "+" if delta >= 0 else ""
        print(f"  {lbl:<12} {f1_before:>10.4f} {f1_after:>10.4f} {sign}{delta:>7.4f}  {tau_sel:>12.4f}")
    print(f"  {'-'*58}")
    macro_before = metrics_A["macro_f1"]
    macro_after  = best_metrics["macro_f1"]
    delta_macro  = macro_after - macro_before
    sign = "+" if delta_macro >= 0 else ""
    print(f"  {'MACRO':<12} {macro_before:>10.4f} {macro_after:>10.4f} {sign}{delta_macro:>7.4f}")
    micro_before = metrics_A["micro_f1"]
    micro_after  = best_metrics["micro_f1"]
    delta_micro  = micro_after - micro_before
    sign = "+" if delta_micro >= 0 else ""
    print(f"  {'MICRO':<12} {micro_before:>10.4f} {micro_after:>10.4f} {sign}{delta_micro:>7.4f}")

    # ── 8. Regression Check ───────────────────────────────────────────────────
    print("\n[Step 8] Regression check (no label should drop below τ=0.5 by more than 0.005)...")
    regression_ok = True
    for lbl in TARGET_COLS:
        f1_before = metrics_A["per_class"][lbl]["f1"]
        f1_after  = best_metrics["per_class"][lbl]["f1"]
        if f1_after < f1_before - 0.005:
            print(f"  ⚠️  REGRESSION: {lbl}: {f1_before:.4f} → {f1_after:.4f} (Δ={f1_after-f1_before:+.4f})")
            regression_ok = False
    if regression_ok:
        print("  ✅ No unacceptable regressions. Best config selected.")

    # ── 9. Freeze Thresholds ──────────────────────────────────────────────────
    print("\n[Step 9] Freezing thresholds to configs/phase7_thresholds.json...")
    frozen = {
        "phase": "Phase 7",
        "pipeline": "TF-IDF (25k) + Emoji + 6-EmoLex + NegScope + Hashtags = 25,089 dims",
        "selection_config": best_config_name,
        "val_macro_f1_before": round(float(macro_before), 4),
        "val_macro_f1_after":  round(float(macro_after), 4),
        "val_micro_f1_before": round(float(micro_before), 4),
        "val_micro_f1_after":  round(float(micro_after), 4),
        "thresholds": {k: round(float(v), 5) for k, v in best_thresholds.items()},
        "per_class_after": {
            k: {
                "f1":        v["f1"],
                "precision": v["precision"],
                "recall":    v["recall"],
                "threshold": round(float(best_thresholds[k]), 5)
            }
            for k, v in best_metrics["per_class"].items()
        },
        "note": "Thresholds selected on validation split ONLY. Test set not touched."
    }
    frozen_path = os.path.join(CONFIGS_DIR, "phase7_thresholds.json")
    with open(frozen_path, "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=2)
    print(f"  Frozen thresholds saved: {frozen_path}")

    # ── 10. Save Full Report ──────────────────────────────────────────────────
    print("\n[Step 10] Saving full threshold report...")
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_dims": int(X_champ_tr.shape[1]),
        "configs": {
            name: {
                "macro_f1": m["macro_f1"],
                "micro_f1": m["micro_f1"],
                "per_class": m["per_class"],
                "thresholds": {k: round(float(v), 5) for k, v in best_thresholds_map[name].items()}
            }
            for name, m in configs.items()
        },
        "disgust_f1_curve":  dis_curve,
        "surprise_f1_curve": sur_curve,
        "disgust_peak":      {"tau": dis_peak[0], "f1": dis_peak[1]},
        "surprise_peak":     {"tau": sur_peak[0], "f1": sur_peak[1]},
        "best_config":       best_config_name,
        "frozen_thresholds": frozen,
    }
    report_path = os.path.join(RESULTS_DIR, "phase7_threshold_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    # ── 11. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PHASE 7 — THRESHOLD OPTIMIZATION SUMMARY")
    print("=" * 75)
    print(f"  All τ=0.5:              Macro-F1={metrics_A['macro_f1']:.4f} | Micro-F1={metrics_A['micro_f1']:.4f}")
    print(f"  Step=0.01:              Macro-F1={metrics_B['macro_f1']:.4f} | Micro-F1={metrics_B['micro_f1']:.4f}")
    print(f"  Step=0.005:             Macro-F1={metrics_C['macro_f1']:.4f} | Micro-F1={metrics_C['micro_f1']:.4f}")
    print(f"  Dis/Sur boosted:        Macro-F1={metrics_D['macro_f1']:.4f} | Micro-F1={metrics_D['micro_f1']:.4f}")
    print(f"  Ultra-fine (step=0.001):Macro-F1={metrics_E['macro_f1']:.4f} | Micro-F1={metrics_E['micro_f1']:.4f}")
    print(f"\n  ✅ SELECTED: [{best_config_name}]")
    print(f"     Macro-F1: {macro_before:.4f} → {macro_after:.4f}  ({'+' if delta_macro >= 0 else ''}{delta_macro:.4f})")
    print(f"     Micro-F1: {micro_before:.4f} → {micro_after:.4f}  ({'+' if delta_micro >= 0 else ''}{delta_micro:.4f})")
    print(f"\n  Frozen thresholds:")
    for lbl, tau in best_thresholds.items():
        print(f"    {lbl:<10}: τ={tau:.4f}")
    print(f"\n  All outputs saved to: {RESULTS_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
