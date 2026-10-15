"""
scripts/experiments/phase7_threshold_analysis_only.py
======================================================
Phase 7 — Threshold analysis using pre-saved val_probabilities.npy
(Avoids refitting the chain — uses output from phase7_threshold_optimization.py Step 3)
"""

import os, sys, json, time, numpy as np
from sklearn.metrics import f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

import pandas as pd
from src.models.betc import TARGET_COLS, compute_label_order
from src.models.evaluate import compute_metrics, apply_thresholds
from scripts.experiments.phase7_threshold_optimization import (
    sweep_thresholds, eval_with_thresholds, threshold_f1_curve
)

DATA_DIR    = os.path.join(WORKSPACE, "Data", "processed")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase7_thresholds")
CONFIGS_DIR = os.path.join(WORKSPACE, "configs")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 75)
print("BETC Phase 7 — Threshold Analysis (from saved probabilities)")
print("=" * 75)

# Load val labels
df_val = pd.read_csv(os.path.join(DATA_DIR, "validation", "validation_preprocessed.csv"), low_memory=False)
Y_val  = df_val[TARGET_COLS].values.astype(int)
print(f"\nLoaded Y_val: {Y_val.shape}")

# Load saved probabilities
prob_path = os.path.join(RESULTS_DIR, "val_probabilities.npy")
P_val = np.load(prob_path)
print(f"Loaded P_val: {P_val.shape}")
print("Probability ranges per class:")
for i, c in enumerate(TARGET_COLS):
    print(f"  {c:<10}: min={P_val[:,i].min():.3f}  mean={P_val[:,i].mean():.3f}  max={P_val[:,i].max():.3f}")

# ── Config A: All τ=0.5 ──────────────────────────────────────────────────────
thresholds_A = {c: 0.5 for c in TARGET_COLS}
metrics_A    = eval_with_thresholds(P_val, Y_val, thresholds_A)
print(f"\n[A] All τ=0.5 (Baseline)")
print(f"    Macro-F1: {metrics_A['macro_f1']:.4f} | Micro-F1: {metrics_A['micro_f1']:.4f}")
print("    Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_A["per_class"].items()))

# ── Config B: step=0.01 ──────────────────────────────────────────────────────
thresholds_B, _ = sweep_thresholds(P_val, Y_val, tau_step=0.01, label_names=TARGET_COLS)
metrics_B = eval_with_thresholds(P_val, Y_val, thresholds_B)
print(f"\n[B] Per-class optimized (step=0.01)")
print(f"    Thresholds: {thresholds_B}")
print(f"    Macro-F1: {metrics_B['macro_f1']:.4f} | Micro-F1: {metrics_B['micro_f1']:.4f}")
print("    Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_B["per_class"].items()))

# ── Config C: step=0.005 ─────────────────────────────────────────────────────
thresholds_C, _ = sweep_thresholds(P_val, Y_val, tau_step=0.005, label_names=TARGET_COLS)
metrics_C = eval_with_thresholds(P_val, Y_val, thresholds_C)
print(f"\n[C] Per-class optimized (step=0.005)")
print(f"    Thresholds: {thresholds_C}")
print(f"    Macro-F1: {metrics_C['macro_f1']:.4f} | Micro-F1: {metrics_C['micro_f1']:.4f}")
print("    Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_C["per_class"].items()))

# ── Config D: C + Disgust/Surprise +0.05 ─────────────────────────────────────
thresholds_D = dict(thresholds_C)
thresholds_D["disgust"]  = min(0.90, round(thresholds_C["disgust"]  + 0.05, 3))
thresholds_D["surprise"] = min(0.90, round(thresholds_C["surprise"] + 0.05, 3))
metrics_D = eval_with_thresholds(P_val, Y_val, thresholds_D)
print(f"\n[D] Config C + Disgust/Surprise +0.05 (FP-reduction, Phase 6 insight)")
print(f"    Thresholds: {thresholds_D}")
print(f"    Macro-F1: {metrics_D['macro_f1']:.4f} | Micro-F1: {metrics_D['micro_f1']:.4f}")
print("    Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_D["per_class"].items()))

# ── Config E: step=0.001 (ultra-fine) ────────────────────────────────────────
print("\n[E] Per-class ultra-fine (step=0.001) — computing...")
thresholds_E, _ = sweep_thresholds(P_val, Y_val, tau_step=0.001, label_names=TARGET_COLS)
metrics_E = eval_with_thresholds(P_val, Y_val, thresholds_E)
print(f"    Thresholds: {thresholds_E}")
print(f"    Macro-F1: {metrics_E['macro_f1']:.4f} | Micro-F1: {metrics_E['micro_f1']:.4f}")
print("    Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in metrics_E["per_class"].items()))

# ── F1-vs-Threshold Curves for Disgust & Surprise ────────────────────────────
print("\n[F1 Curves] Disgust & Surprise threshold sensitivity...")
tau_range = np.round(np.arange(0.05, 0.955, 0.005), 5)
dis_curve = threshold_f1_curve(P_val, Y_val, "disgust", tau_range)
sur_curve = threshold_f1_curve(P_val, Y_val, "surprise", tau_range)
dis_peak  = max(dis_curve, key=lambda x: x[1])
sur_peak  = max(sur_curve, key=lambda x: x[1])
print(f"  Disgust  peak: F1={dis_peak[1]:.4f} at τ={dis_peak[0]:.3f}")
print(f"  Surprise peak: F1={sur_peak[1]:.4f} at τ={sur_peak[0]:.3f}")

# ── Select Best ───────────────────────────────────────────────────────────────
configs_map = {
    "A_fixed_0.5":               (metrics_A, thresholds_A),
    "B_optimized_step0.01":      (metrics_B, thresholds_B),
    "C_optimized_step0.005":     (metrics_C, thresholds_C),
    "D_disgust_surprise_boosted":(metrics_D, thresholds_D),
    "E_ultra_fine_step0.001":    (metrics_E, thresholds_E),
}
best_name = max(configs_map, key=lambda k: configs_map[k][0]["macro_f1"])
best_metrics, best_thresholds = configs_map[best_name]

print(f"\n✅ SELECTED: [{best_name}]")
print(f"   Macro-F1: {metrics_A['macro_f1']:.4f} → {best_metrics['macro_f1']:.4f}  ({best_metrics['macro_f1']-metrics_A['macro_f1']:+.4f})")
print(f"   Micro-F1: {metrics_A['micro_f1']:.4f} → {best_metrics['micro_f1']:.4f}  ({best_metrics['micro_f1']-metrics_A['micro_f1']:+.4f})")

# ── Before/After Table ────────────────────────────────────────────────────────
print(f"\n  {'Label':<12} {'τ=0.5 F1':>10} {'Best F1':>10} {'Δ':>8}  {'τ':>8}")
print(f"  {'-'*56}")
for lbl in TARGET_COLS:
    f1_b = metrics_A["per_class"][lbl]["f1"]
    f1_a = best_metrics["per_class"][lbl]["f1"]
    tau  = best_thresholds[lbl]
    print(f"  {lbl:<12} {f1_b:>10.4f} {f1_a:>10.4f} {f1_a-f1_b:>+8.4f}  {tau:>8.4f}")
print(f"  {'-'*56}")
print(f"  {'MACRO':<12} {metrics_A['macro_f1']:>10.4f} {best_metrics['macro_f1']:>10.4f} {best_metrics['macro_f1']-metrics_A['macro_f1']:>+8.4f}")
print(f"  {'MICRO':<12} {metrics_A['micro_f1']:>10.4f} {best_metrics['micro_f1']:>10.4f} {best_metrics['micro_f1']-metrics_A['micro_f1']:>+8.4f}")

# ── Regression Check ──────────────────────────────────────────────────────────
print("\n[Regression Check] No label should drop >0.005 below τ=0.5...")
ok = True
for lbl in TARGET_COLS:
    f1_b = metrics_A["per_class"][lbl]["f1"]
    f1_a = best_metrics["per_class"][lbl]["f1"]
    if f1_a < f1_b - 0.005:
        print(f"  ⚠️  REGRESSION: {lbl}: {f1_b:.4f} → {f1_a:.4f}")
        ok = False
if ok:
    print("  ✅ No unacceptable regressions.")

# ── Freeze Thresholds ─────────────────────────────────────────────────────────
frozen = {
    "phase": "Phase 7",
    "pipeline": "TF-IDF (25k) + Emoji + 6-EmoLex + NegScope + Hashtags = 25,089 dims",
    "selection_config": best_name,
    "val_macro_f1_before": round(float(metrics_A["macro_f1"]), 4),
    "val_macro_f1_after":  round(float(best_metrics["macro_f1"]), 4),
    "val_micro_f1_before": round(float(metrics_A["micro_f1"]), 4),
    "val_micro_f1_after":  round(float(best_metrics["micro_f1"]), 4),
    "thresholds": {k: round(float(v), 5) for k, v in best_thresholds.items()},
    "per_class_after": {
        k: {"f1": v["f1"], "precision": v["precision"], "recall": v["recall"],
            "threshold": round(float(best_thresholds[k]), 5)}
        for k, v in best_metrics["per_class"].items()
    },
    "note": "Thresholds selected on validation split ONLY. Test set not touched."
}
frozen_path = os.path.join(CONFIGS_DIR, "phase7_thresholds.json")
with open(frozen_path, "w", encoding="utf-8") as f:
    json.dump(frozen, f, indent=2)
print(f"\nFrozen thresholds saved: {frozen_path}")

# ── Save Full Report ──────────────────────────────────────────────────────────
report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "configs": {n: {"macro_f1": m["macro_f1"], "micro_f1": m["micro_f1"],
                    "per_class": m["per_class"],
                    "thresholds": {k: round(float(v), 5) for k, v in t.items()}}
                for n, (m, t) in configs_map.items()},
    "disgust_f1_curve":  dis_curve,
    "surprise_f1_curve": sur_curve,
    "disgust_peak":      {"tau": dis_peak[0], "f1": dis_peak[1]},
    "surprise_peak":     {"tau": sur_peak[0], "f1": sur_peak[1]},
    "best_config":       best_name,
    "frozen_thresholds": frozen,
}
report_path = os.path.join(RESULTS_DIR, "phase7_threshold_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(f"Full report saved: {report_path}")

print("\n" + "=" * 75)
print(f"PHASE 7 COMPLETE — Best: [{best_name}]")
print(f"Macro-F1: {metrics_A['macro_f1']:.4f} → {best_metrics['macro_f1']:.4f}")
print("=" * 75)
