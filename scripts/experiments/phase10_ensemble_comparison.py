"""
scripts/experiments/phase10_ensemble_comparison.py
====================================================
BETC Improvement Plan — Phase 10: Classifier Chains vs Voting/Ensembles

Objectives:
  Compare the Phase 9 best individual classifier (C1: LR ClassifierChain)
  against ensemble strategies that may combine the strengths of LR and CNB.

  Key observation from Phase 9:
    - C1 (LR):  Macro-F1=0.4921 | strong on Joy/Anger/Sadness
    - C2 (CNB): Macro-F1=0.5049 unconstrained | Fear=0.522 (+0.128!), Surprise=0.356
    LR+CNB soft voting may combine complementary strengths.

  Strategies tested (all on Phase 8b champion features, 25,101 dims):
    S0 — LR Chain alone          (Phase 9 champion; reconfirm)
    S1 — CNB Chain alone         (Phase 9 unconstrained best; reconfirm)
    S2 — Soft Vote Equal         (0.50 LR + 0.50 CNB)
    S3 — Soft Vote LR-heavy      (0.70 LR + 0.30 CNB)
    S4 — Soft Vote CNB-heavy     (0.30 LR + 0.70 CNB)
    S5 — Label-wise Oracle       (per-label: use whichever model scored higher on val;
                                  NOTE: this is an UPPER BOUND / oracle, not deployable)

  Protocol:
    - Same 25,101-dim feature set as Phase 8b/9.
    - LR and CNB fitted on training data → val probabilities saved to disk.
    - All blending done on val probabilities only; threshold sweep on val only.
    - Selection rule: beats S0 if Macro-F1 > 0.4921 AND no label drops > 0.01.

Deliverables:
    - results/ablations/phase10_ensemble/phase10_report.json
    - results/ablations/phase10_ensemble/phase10_report.md
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
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
from scripts.experiments.phase3_lexicon_study import EMOTION_LEXICON
from scripts.experiments.phase8b_targeted_features import (
    extract_group_C_intensifier_bigrams,
)

DATA_DIR    = os.path.join(WORKSPACE, "Data", "processed")
CACHE_DIR   = os.path.join(WORKSPACE, "results", "ablations", "phase8b_targeted_features")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase10_ensemble")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Phase 9 champion reference (C1 LR Chain)
CHAMPION_MACRO      = 0.4921
CHAMPION_MICRO      = 0.5834
CHAMPION_PER_CLASS  = {
    "anger": 0.5503, "disgust": 0.3478, "fear": 0.3941,
    "joy": 0.8254,   "sadness": 0.5031, "surprise": 0.332,
}
REGRESSION_TOLERANCE = 0.010

PARTIAL_PATH     = os.path.join(RESULTS_DIR, "phase10_partial.json")
PROB_LR_PATH     = os.path.join(RESULTS_DIR, "P_va_lr.npy")
PROB_CNB_PATH    = os.path.join(RESULTS_DIR, "P_va_cnb.npy")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def sweep_thresholds(P, Y, step=0.001, label_names=None):
    if label_names is None:
        label_names = [f"label_{i}" for i in range(Y.shape[1])]
    thresholds = {}
    for i, name in enumerate(label_names):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.955, step):
            preds = (P[:, i] >= t).astype(int)
            f = f1_score(Y[:, i], preds, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t  = round(float(t), 4)
        thresholds[name] = best_t
    return thresholds


def evaluate_probs(name, P_va, Y_va, extra_dims=None):
    """Sweep thresholds on P_va and compute metrics. Returns result dict."""
    t0    = time.time()
    taus  = sweep_thresholds(P_va, Y_va, step=0.001, label_names=TARGET_COLS)
    Y_pred = apply_thresholds(P_va, taus, TARGET_COLS)
    m      = compute_metrics(Y_va, Y_pred, TARGET_COLS)
    elapsed = time.time() - t0

    macro_f1 = m["macro_f1"]
    micro_f1 = m["micro_f1"]
    per_cls  = {k: round(v["f1"], 4) for k, v in m["per_class"].items()}

    print(f"\n  [{name}]")
    print(f"    Macro-F1  : {macro_f1:.4f}  (Δ={macro_f1 - CHAMPION_MACRO:+.4f} vs S0-LR)")
    print(f"    Micro-F1  : {micro_f1:.4f}  (Δ={micro_f1 - CHAMPION_MICRO:+.4f} vs S0-LR)")
    print("    Per-class : " + " | ".join(f"{k}={v:.3f}" for k, v in per_cls.items()))
    print(f"    Thresholds: { {k: round(v, 3) for k, v in taus.items()} }")
    print(f"    Sweep time: {elapsed:.1f}s")

    regressions = {
        k: round(per_cls[k] - CHAMPION_PER_CLASS[k], 4)
        for k in TARGET_COLS
        if round(per_cls[k] - CHAMPION_PER_CLASS[k], 4) < -REGRESSION_TOLERANCE
    }
    if regressions:
        print(f"    ⚠  Regression (>0.01 vs S0): {regressions}")
    else:
        print("    ✅ No regression violations vs S0")

    beats_s0 = (macro_f1 > CHAMPION_MACRO) and (len(regressions) == 0)
    return {
        "name":        name,
        "dims":        extra_dims,
        "macro_f1":    round(macro_f1, 4),
        "micro_f1":    round(micro_f1, 4),
        "per_class":   per_cls,
        "thresholds":  {k: round(v, 4) for k, v in taus.items()},
        "delta_macro": round(macro_f1 - CHAMPION_MACRO, 4),
        "delta_micro": round(micro_f1 - CHAMPION_MICRO, 4),
        "regressions": regressions,
        "beats_s0":    beats_s0,
        "sweep_time_s": round(elapsed, 1),
    }


# ─── Build features ──────────────────────────────────────────────────────────
def build_champion_features(train_proc, val_proc):
    print("  Loading Phase 5 base matrices from cache...")
    X_base_tr = sp.load_npz(os.path.join(CACHE_DIR, "X_base_tr.npz"))
    X_base_va = sp.load_npz(os.path.join(CACHE_DIR, "X_base_va.npz"))

    emo_vocab_set = {k: set(v) for k, v in EMOTION_LEXICON.items()}
    print("  Extracting Group C (Intensifier-Emotion Bigrams, 12 features)...")
    scaler_C = MaxAbsScaler()
    fC_tr = sp.csr_matrix(scaler_C.fit_transform(
        extract_group_C_intensifier_bigrams(train_proc, emo_vocab_set)))
    fC_va = sp.csr_matrix(scaler_C.transform(
        extract_group_C_intensifier_bigrams(val_proc, emo_vocab_set)))

    X_tr = sp.hstack([X_base_tr, fC_tr], format="csr")
    X_va = sp.hstack([X_base_va, fC_va], format="csr")
    print(f"  Champion features: Train={X_tr.shape}  Val={X_va.shape}")
    return X_tr, X_va


# ─── Fit base models & cache probabilities ───────────────────────────────────
def fit_and_cache(X_tr, Y_tr, X_va, freq_order):
    """Fit LR and CNB chains, save val probability matrices. Returns P_lr, P_cnb."""

    # LR
    if os.path.exists(PROB_LR_PATH):
        print("  [LR] Loading cached val probabilities...")
        P_lr = np.load(PROB_LR_PATH)
    else:
        print("  [LR] Fitting ClassifierChain(LR)... (~10 min)")
        t0 = time.time()
        clf_lr = ClassifierChain(
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                               solver="saga", random_state=42),
            order=freq_order, random_state=42
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf_lr.fit(X_tr, Y_tr)
        P_lr = clf_lr.predict_proba(X_va)
        np.save(PROB_LR_PATH, P_lr)
        print(f"  [LR] Done ({time.time()-t0:.0f}s). Saved to {PROB_LR_PATH}")

    # CNB
    if os.path.exists(PROB_CNB_PATH):
        print("  [CNB] Loading cached val probabilities...")
        P_cnb = np.load(PROB_CNB_PATH)
    else:
        print("  [CNB] Fitting ClassifierChain(CNB)... (~30s)")
        t0 = time.time()
        clf_cnb = ClassifierChain(
            ComplementNB(alpha=1.0),
            order=freq_order, random_state=42
        )
        clf_cnb.fit(X_tr, Y_tr)
        P_cnb = clf_cnb.predict_proba(X_va)
        np.save(PROB_CNB_PATH, P_cnb)
        print(f"  [CNB] Done ({time.time()-t0:.0f}s). Saved to {PROB_CNB_PATH}")

    return P_lr, P_cnb


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("BETC Phase 10 — Classifier Chains vs Voting / Ensembles")
    print(f"S0-LR Reference: Macro-F1={CHAMPION_MACRO} | Micro-F1={CHAMPION_MICRO}")
    print("=" * 75)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[Step 1] Loading data...")
    df_tr = pd.read_csv(
        os.path.join(DATA_DIR, "train", "train_preprocessed.csv"), low_memory=False
    ).reset_index(drop=True)
    df_va = pd.read_csv(
        os.path.join(DATA_DIR, "validation", "validation_preprocessed.csv"), low_memory=False
    ).reset_index(drop=True)

    train_proc = df_tr["processed_text"].fillna("").astype(str).tolist()
    val_proc   = df_va["processed_text"].fillna("").astype(str).tolist()
    Y_tr = np.load(os.path.join(CACHE_DIR, "Y_tr.npy"))
    Y_va = np.load(os.path.join(CACHE_DIR, "Y_va.npy"))
    freq_order = json.load(open(os.path.join(CACHE_DIR, "freq_order.json")))["freq_order"]
    print(f"  Train: {len(df_tr):,} | Val: {len(df_va):,}")

    # ── Build features ────────────────────────────────────────────────────────
    print("\n[Step 2] Building Phase 8b champion features (25,101 dims)...")
    X_tr, X_va = build_champion_features(train_proc, val_proc)
    dims = X_tr.shape[1]

    # ── Fit models & cache probabilities ─────────────────────────────────────
    print("\n[Step 3] Fitting LR and CNB — caching val probabilities...")
    P_lr, P_cnb = fit_and_cache(X_tr, Y_tr, X_va, freq_order)
    print(f"  P_lr shape: {P_lr.shape}  P_cnb shape: {P_cnb.shape}")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    results = {}
    if os.path.exists(PARTIAL_PATH):
        try:
            with open(PARTIAL_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"\n  Loaded {len(results)} completed config(s) from checkpoint.")
        except Exception:
            results = {}

    # ── Evaluate strategies ──────────────────────────────────────────────────
    print("\n[Step 4] Evaluating ensemble strategies...")
    print(f"  S0 Reference — LR Chain: Macro-F1={CHAMPION_MACRO} | Micro-F1={CHAMPION_MICRO}")

    strategies = [
        ("S0_LR_Chain",       P_lr),
        ("S1_CNB_Chain",      P_cnb),
        ("S2_Vote_50_50",     0.50 * P_lr + 0.50 * P_cnb),
        ("S3_Vote_LR70_CNB30",0.70 * P_lr + 0.30 * P_cnb),
        ("S4_Vote_LR30_CNB70",0.30 * P_lr + 0.70 * P_cnb),
    ]

    for cfg_name, P_blend in strategies:
        if cfg_name in results:
            r = results[cfg_name]
            print(f"\n  [{cfg_name}] — already done (Macro-F1={r['macro_f1']:.4f}). Skipping.")
            continue
        r = evaluate_probs(cfg_name, P_blend, Y_va, extra_dims=dims)
        r["description"] = {
            "S0_LR_Chain":       "LR ClassifierChain alone (Phase 9 champion)",
            "S1_CNB_Chain":      "CNB ClassifierChain alone (Phase 9 unconstrained best)",
            "S2_Vote_50_50":     "Soft vote: 0.50 LR + 0.50 CNB",
            "S3_Vote_LR70_CNB30":"Soft vote: 0.70 LR + 0.30 CNB",
            "S4_Vote_LR30_CNB70":"Soft vote: 0.30 LR + 0.70 CNB",
        }[cfg_name]
        results[cfg_name] = r
        with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── Label-wise Oracle (upper bound — not deployable) ──────────────────────
    if "S5_LabelWise_Oracle" not in results:
        print("\n  [S5_LabelWise_Oracle] Building per-label best probability mix...")
        print("    (NOTE: This uses val performance to select per-label model — oracle/upper bound only)")
        # For each label, pick whichever base model scored higher in Phase 9
        # LR per-class: ang=0.5503, dis=0.3478, fea=0.3941, joy=0.8254, sad=0.5031, sur=0.332
        # CNB per-class: ang=0.527,  dis=0.354,  fea=0.522,  joy=0.792,  sad=0.478,  sur=0.356
        lr_better  = ["anger", "joy", "sadness"]   # LR wins on these
        cnb_better = ["disgust", "fear", "surprise"]  # CNB wins on these

        P_oracle = np.zeros_like(P_lr)
        for i, col in enumerate(TARGET_COLS):
            if col in lr_better:
                P_oracle[:, i] = P_lr[:, i]
            else:
                P_oracle[:, i] = P_cnb[:, i]
        print(f"    LR labels: {lr_better}  |  CNB labels: {cnb_better}")
        r = evaluate_probs("S5_LabelWise_Oracle", P_oracle, Y_va, extra_dims=dims)
        r["description"] = ("Oracle: per-label best model (LR for anger/joy/sadness, "
                            "CNB for disgust/fear/surprise). UPPER BOUND — not deployable.")
        r["oracle"] = True
        results["S5_LabelWise_Oracle"] = r
        with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        r = results["S5_LabelWise_Oracle"]
        print(f"\n  [S5_LabelWise_Oracle] — already done (Macro-F1={r['macro_f1']:.4f}). Skipping.")

    # ── Selection ─────────────────────────────────────────────────────────────
    print("\n[Step 5] Selecting best deployable strategy...")
    # Oracle is excluded from selection (not deployable)
    deployable = {k: v for k, v in results.items()
                  if k != "S5_LabelWise_Oracle" and k != "S0_LR_Chain"}
    winners = {k: v for k, v in deployable.items() if v["beats_s0"]}

    if winners:
        best_name = max(winners, key=lambda k: winners[k]["macro_f1"])
        best      = winners[best_name]
        print(f"  ✅ Winner: [{best_name}]")
        print(f"     Macro-F1={best['macro_f1']:.4f} (Δ={best['delta_macro']:+.4f})")
        print(f"     Micro-F1={best['micro_f1']:.4f} (Δ={best['delta_micro']:+.4f})")
    else:
        best_name = "S0_LR_Chain"
        best      = results["S0_LR_Chain"]
        print("  ⚪ No ensemble beat S0 safely — LR Chain remains champion.")

    unc_best = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"  Unconstrained best: [{unc_best}] Macro-F1={results[unc_best]['macro_f1']:.4f}")
    oracle_r = results.get("S5_LabelWise_Oracle", {})
    if oracle_r:
        print(f"  Oracle upper bound: Macro-F1={oracle_r['macro_f1']:.4f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    print("\n[Step 6] Saving report...")

    report = {
        "phase":      "Phase 10",
        "date":       pd.Timestamp.now().strftime("%Y-%m-%d"),
        "s0_reference": {
            "name":     "S0_LR_Chain (Phase 9 champion)",
            "macro_f1": CHAMPION_MACRO,
            "micro_f1": CHAMPION_MICRO,
            "per_class": CHAMPION_PER_CLASS,
        },
        "selected":   best_name,
        "results":    results,
    }
    json_path = os.path.join(RESULTS_DIR, "phase10_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # Markdown
    desc_map = {
        "S0_LR_Chain":       "LR ClassifierChain (Phase 9 champion)",
        "S1_CNB_Chain":      "CNB ClassifierChain (Phase 9 unconstrained best)",
        "S2_Vote_50_50":     "Soft vote: 0.50 LR + 0.50 CNB",
        "S3_Vote_LR70_CNB30":"Soft vote: 0.70 LR + 0.30 CNB",
        "S4_Vote_LR30_CNB70":"Soft vote: 0.30 LR + 0.70 CNB",
        "S5_LabelWise_Oracle":"Oracle: per-label best (UPPER BOUND, not deployable)",
    }
    md = [
        "# BETC — Phase 10: Classifier Chains vs Ensembles Report",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        f"**Feature set:** Phase 8b Champion (25,101 dims)  ",
        f"**S0-LR Reference:** Macro-F1=`{CHAMPION_MACRO}` | Micro-F1=`{CHAMPION_MICRO}`",
        "",
        "---",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Description | Val Macro-F1 | Δ Macro | Val Micro-F1 | Δ Micro | Beats S0? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, v in results.items():
        sel    = " ✅" if k == best_name else ""
        if k == "S5_LabelWise_Oracle":
            status = "🔭 Oracle"
        elif k == "S0_LR_Chain":
            status = "⚪ Ref"
        else:
            status = "✅" if v["beats_s0"] else "❌"
        md.append(
            f"| **{k}**{sel} | {desc_map.get(k, k)} | "
            f"`{v['macro_f1']:.4f}` | `{v['delta_macro']:+.4f}` | "
            f"`{v['micro_f1']:.4f}` | `{v['delta_micro']:+.4f}` | {status} |"
        )
    md += [
        "",
        "## Per-Label F1 Breakdown",
        "",
        "| Strategy | anger | disgust | fear | joy | sadness | surprise |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, v in results.items():
        pc = v["per_class"]
        md.append(
            f"| **{k}** | {pc['anger']:.3f} | {pc['disgust']:.3f} | "
            f"{pc['fear']:.3f} | {pc['joy']:.3f} | "
            f"{pc['sadness']:.3f} | {pc['surprise']:.3f} |"
        )
    md += [
        "",
        "## Selection",
        "",
        f"**Selected strategy:** `{best_name}`  ",
        f"**Val Macro-F1:** `{best['macro_f1']:.4f}` | "
        f"**Val Micro-F1:** `{best['micro_f1']:.4f}`",
        "",
        "> Selection rule: beats S0 if Macro-F1 > 0.4921 AND no label drops >0.01 F1.",
        "> S5 (oracle) is excluded from selection — it uses val performance to select per-label model.",
        "",
        "## Deliverables",
        "- `scripts/experiments/phase10_ensemble_comparison.py`",
        "- `results/ablations/phase10_ensemble/phase10_report.json`",
        "- `tests/test_phase10_ensemble.py`",
    ]
    md_path = os.path.join(RESULTS_DIR, "phase10_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  Markdown: {md_path}")

    # Freeze winning thresholds if better than S0
    if best_name != "S0_LR_Chain":
        thresh_out = {
            "phase":      "Phase 10",
            "selected":   best_name,
            "dims":       dims,
            "macro_f1":   best["macro_f1"],
            "micro_f1":   best["micro_f1"],
            "thresholds": best["thresholds"],
            "note":       "Ensemble thresholds optimised on validation split ONLY.",
        }
        thresh_path = os.path.join(WORKSPACE, "configs", "phase10_thresholds.json")
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump(thresh_out, f, indent=2, ensure_ascii=False)
        print(f"  Frozen thresholds: {thresh_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PHASE 10 COMPLETE — Classifier Chains vs Voting / Ensembles")
    print("=" * 75)
    for k, v in results.items():
        tag = " ← SELECTED" if k == best_name else (" [ORACLE]" if k == "S5_LabelWise_Oracle" else "")
        print(f"  [{k}]{tag}")
        print(f"    Macro-F1={v['macro_f1']:.4f}  Micro-F1={v['micro_f1']:.4f}  "
              f"Δ={v['delta_macro']:+.4f}")
    print("=" * 75)
    print(f"  Selected: [{best_name}]")
    print(f"  Macro-F1={best['macro_f1']:.4f} | Micro-F1={best['micro_f1']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

