"""
scripts/experiments/phase9_classifier_comparison.py
======================================================
BETC Improvement Plan — Phase 9: Compare Individual Classifiers

Objectives:
  Evaluate four individual classifiers on the Phase 8b champion feature set
  (25,101 dims: Phase 5 base + Group C Intensifier-Emotion Bigrams) using the
  same canonical validation split, identical label mapping, and comparable
  tuning budgets.

  Classifiers:
    C1 — Logistic Regression (LR)        — current Phase 8b champion baseline
    C2 — Complement Naive Bayes (CNB)    — fast, handles sparse TF-IDF well
    C3 — SGD / SVM-like (modified_huber) — fast SVM with calibrated probs
    C4 — K-Nearest Neighbours (KNN)      — after TruncatedSVD(200) reduction

  Multi-label strategy: ClassifierChain (frequency-ordered) for C1/C2/C3;
                         MultiOutputClassifier for C4 (no chaining needed for
                         distance-based models, and predict_proba is supported).

  Fair-comparison rules:
    - Identical feature set (Phase 8b champion, 25,101 dims)
    - Identical canonical train/val split (no test set touched)
    - Identical label order (freq_order from Phase 8b)
    - Per-class threshold optimisation on val probabilities only (step=0.001)
    - Selection by Val Macro-F1; no test-set decisions
    - Transformations (SVD) fitted on training data only

  Selection rule: A classifier beats C1 (LR) if its Val Macro-F1 > 0.4921
                  AND no label drops > 0.01 F1 below C1.

Deliverables:
    - results/ablations/phase9_classifiers/phase9_report.json
    - results/ablations/phase9_classifiers/phase9_report.md
"""

import os
import sys
import json
import time
import re
import warnings
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.decomposition import TruncatedSVD
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
from scripts.experiments.phase4_negation_study import EXPANDED_INTENSIFIERS
from scripts.experiments.phase8b_targeted_features import (
    extract_group_C_intensifier_bigrams,
)

DATA_DIR    = os.path.join(WORKSPACE, "Data", "processed")
CACHE_DIR   = os.path.join(WORKSPACE, "results", "ablations", "phase8b_targeted_features")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase9_classifiers")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Phase 8b champion reference
CHAMPION_MACRO = 0.4921
CHAMPION_MICRO = 0.5834
CHAMPION_PER_CLASS = {
    "anger": 0.5503, "disgust": 0.3478, "fear": 0.3941,
    "joy": 0.8254,   "sadness": 0.5031, "surprise": 0.332,
}
REGRESSION_TOLERANCE = 0.010

PARTIAL_PATH = os.path.join(RESULTS_DIR, "phase9_partial.json")


# ─── Threshold sweep ──────────────────────────────────────────────────────────
def sweep_thresholds(P, Y, step=0.001, label_names=None):
    """Per-class threshold sweep on validation data."""
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


# ─── Build Phase 8b champion features ────────────────────────────────────────
def build_champion_features(train_proc, val_proc):
    """
    Returns (X_tr, X_va) = Phase 5 base (cached) + Group C (12 dims).
    Base matrix is loaded from disk (saved by Phase 8b step 2).
    Group C is extracted fresh (fast, pure Python).
    """
    print("  Loading Phase 5 base matrices from Phase 8b cache...")
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


# ─── Fit & evaluate one classifier ───────────────────────────────────────────
def run_classifier(name, clf_chain, X_tr, Y_tr, X_va, Y_va, has_proba=True):
    """
    Fit a ClassifierChain or MultiOutputClassifier and evaluate on val.
    Returns metrics dict.
    """
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        clf_chain.fit(X_tr, Y_tr)

    if has_proba:
        raw_proba = clf_chain.predict_proba(X_va)
        if isinstance(raw_proba, list):
            # MultiOutputClassifier returns list of (n_samples, 2) arrays — extract P(class=1)
            P_va = np.column_stack([p[:, 1] for p in raw_proba])
        else:
            P_va = raw_proba
    else:
        # Decision function (LinearSVC fallback) — treat scores as pseudo-probs
        P_va = clf_chain.decision_function(X_va)
        # Sigmoid-normalise to [0,1]
        P_va = 1.0 / (1.0 + np.exp(-P_va))

    taus   = sweep_thresholds(P_va, Y_va, step=0.001, label_names=TARGET_COLS)
    Y_pred = apply_thresholds(P_va, taus, TARGET_COLS)
    m      = compute_metrics(Y_va, Y_pred, TARGET_COLS)
    elapsed = time.time() - t0

    macro_f1 = m["macro_f1"]
    micro_f1 = m["micro_f1"]
    per_cls  = {k: round(v["f1"], 4) for k, v in m["per_class"].items()}

    print(f"\n  [{name}]")
    print(f"    Macro-F1  : {macro_f1:.4f}  (Δ={macro_f1 - CHAMPION_MACRO:+.4f} vs C1-LR)")
    print(f"    Micro-F1  : {micro_f1:.4f}  (Δ={micro_f1 - CHAMPION_MICRO:+.4f} vs C1-LR)")
    print("    Per-class : " + " | ".join(f"{k}={v:.3f}" for k, v in per_cls.items()))
    print(f"    Thresholds: { {k: round(v,3) for k, v in taus.items()} }")
    print(f"    Time      : {elapsed:.0f}s")

    regressions = {
        k: round(per_cls[k] - CHAMPION_PER_CLASS[k], 4)
        for k in TARGET_COLS
        if round(per_cls[k] - CHAMPION_PER_CLASS[k], 4) < -REGRESSION_TOLERANCE
    }
    if regressions:
        print(f"    ⚠  Regression (>0.01 vs C1): {regressions}")
    else:
        print("    ✅ No regression violations vs C1")

    beats_c1 = (macro_f1 > CHAMPION_MACRO) and (len(regressions) == 0)
    return {
        "name":        name,
        "dims":        int(X_tr.shape[1]),
        "macro_f1":    round(macro_f1, 4),
        "micro_f1":    round(micro_f1, 4),
        "per_class":   per_cls,
        "thresholds":  {k: round(v, 4) for k, v in taus.items()},
        "delta_macro": round(macro_f1 - CHAMPION_MACRO, 4),
        "delta_micro": round(micro_f1 - CHAMPION_MICRO, 4),
        "regressions": regressions,
        "beats_c1":    beats_c1,
        "time_s":      round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("BETC Phase 9 — Individual Classifier Comparison")
    print("Feature set: Phase 8b champion (25,101 dims)")
    print(f"C1-LR Reference: Macro-F1={CHAMPION_MACRO} | Micro-F1={CHAMPION_MICRO}")
    print("=" * 75)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[Step 1] Loading data splits...")
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
    print(f"  Train: {len(df_tr):,} | Val: {len(df_va):,} | Chain order: {freq_order}")

    # ── Build champion features ──────────────────────────────────────────────
    print("\n[Step 2] Building Phase 8b champion features (25,101 dims)...")
    X_tr, X_va = build_champion_features(train_proc, val_proc)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    results = {}
    if os.path.exists(PARTIAL_PATH):
        try:
            with open(PARTIAL_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"\n  Loaded {len(results)} completed result(s) from checkpoint.")
        except Exception:
            results = {}

    # ── Classifier definitions ────────────────────────────────────────────────
    print("\n[Step 3] Running classifier comparisons...")
    print(f"  Reference — C1 (LR): Macro-F1={CHAMPION_MACRO} | Micro-F1={CHAMPION_MICRO}")

    # C1 — Logistic Regression (reconfirm champion)
    if "C1_LogisticRegression" not in results:
        print("\n  Running C1 — Logistic Regression (ClassifierChain)...")
        clf_c1 = ClassifierChain(
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                               solver="saga", random_state=42),
            order=freq_order, random_state=42
        )
        results["C1_LogisticRegression"] = run_classifier(
            "C1_LogisticRegression", clf_c1, X_tr, Y_tr, X_va, Y_va
        )
        with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        r = results["C1_LogisticRegression"]
        print(f"\n  [C1_LogisticRegression] — already done (Macro-F1={r['macro_f1']:.4f}). Skipping.")

    # C2 — Complement Naive Bayes (ClassifierChain)
    if "C2_ComplementNB" not in results:
        print("\n  Running C2 — Complement Naive Bayes (ClassifierChain)...")
        clf_c2 = ClassifierChain(
            ComplementNB(alpha=1.0),
            order=freq_order, random_state=42
        )
        results["C2_ComplementNB"] = run_classifier(
            "C2_ComplementNB", clf_c2, X_tr, Y_tr, X_va, Y_va
        )
        with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        r = results["C2_ComplementNB"]
        print(f"\n  [C2_ComplementNB] — already done (Macro-F1={r['macro_f1']:.4f}). Skipping.")

    # C3 — SGDClassifier (modified_huber = SVM-like with calibrated probs)
    if "C3_SGD_SVM" not in results:
        print("\n  Running C3 — SGD/SVM-like modified_huber (ClassifierChain)...")
        clf_c3 = ClassifierChain(
            SGDClassifier(
                loss="modified_huber",
                class_weight="balanced",
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                n_jobs=-1,
            ),
            order=freq_order, random_state=42
        )
        results["C3_SGD_SVM"] = run_classifier(
            "C3_SGD_SVM", clf_c3, X_tr, Y_tr, X_va, Y_va
        )
        with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        r = results["C3_SGD_SVM"]
        print(f"\n  [C3_SGD_SVM] — already done (Macro-F1={r['macro_f1']:.4f}). Skipping.")

    # C4 — KNN (TruncatedSVD 200 dims → MultiOutputClassifier)
    if "C4_KNN" not in results:
        print("\n  Running C4 — KNN (TruncatedSVD 200 dims + MultiOutputClassifier)...")
        print("    Fitting TruncatedSVD(200) on training data...")
        t_svd = time.time()
        svd = TruncatedSVD(n_components=200, random_state=42)
        X_tr_svd = svd.fit_transform(X_tr)
        X_va_svd = svd.transform(X_va)
        print(f"    SVD done: {X_tr_svd.shape} ({time.time()-t_svd:.1f}s)")

        knn_base = KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)
        clf_c4 = MultiOutputClassifier(knn_base, n_jobs=-1)

        results["C4_KNN"] = run_classifier(
            "C4_KNN", clf_c4, X_tr_svd, Y_tr, X_va_svd, Y_va
        )
        with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        r = results["C4_KNN"]
        print(f"\n  [C4_KNN] — already done (Macro-F1={r['macro_f1']:.4f}). Skipping.")

    # ── Selection ─────────────────────────────────────────────────────────────
    print("\n[Step 4] Selecting best classifier...")
    challengers = {k: v for k, v in results.items()
                   if v["beats_c1"] and k != "C1_LogisticRegression"}
    if challengers:
        best_name = max(challengers, key=lambda k: challengers[k]["macro_f1"])
        best      = challengers[best_name]
        print(f"  ✅ A challenger beats C1: [{best_name}]")
        print(f"     Macro-F1={best['macro_f1']:.4f} (Δ={best['delta_macro']:+.4f})")
    else:
        best_name = "C1_LogisticRegression"
        best      = results["C1_LogisticRegression"]
        print("  ⚪ No challenger beat C1 safely — C1 (LR) remains the champion.")

    unc_best = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"  Unconstrained best: [{unc_best}] Macro-F1={results[unc_best]['macro_f1']:.4f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    print("\n[Step 5] Saving report...")

    report = {
        "phase":      "Phase 9",
        "date":       pd.Timestamp.now().strftime("%Y-%m-%d"),
        "champion_reference": {
            "name":     "C1_LogisticRegression (Phase 8b champion)",
            "macro_f1": CHAMPION_MACRO,
            "micro_f1": CHAMPION_MICRO,
            "per_class": CHAMPION_PER_CLASS,
        },
        "selected":   best_name,
        "results":    results,
    }
    json_path = os.path.join(RESULTS_DIR, "phase9_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {json_path}")

    # Markdown
    desc_map = {
        "C1_LogisticRegression": "LR C=1.0, balanced, saga (Phase 8b champ)",
        "C2_ComplementNB":       "ComplementNB α=1.0",
        "C3_SGD_SVM":            "SGD modified_huber (SVM-like), balanced",
        "C4_KNN":                "KNN k=5, cosine, TruncatedSVD(200)",
    }
    md = [
        "# BETC — Phase 9: Individual Classifier Comparison Report",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        f"**Feature set:** Phase 8b Champion (25,101 dims)  ",
        f"**C1-LR Reference:** Macro-F1=`{CHAMPION_MACRO}` | Micro-F1=`{CHAMPION_MICRO}`",
        "",
        "---",
        "",
        "## Strategy Summary",
        "",
        "| Classifier | Description | Dims | Val Macro-F1 | Δ Macro | Val Micro-F1 | Δ Micro | Beats C1? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, v in results.items():
        sel    = " ✅" if k == best_name else ""
        status = "✅" if v["beats_c1"] else ("⚪" if k == "C1_LogisticRegression" else "❌")
        md.append(
            f"| **{k}**{sel} | {desc_map.get(k, k)} | {v['dims']:,} | "
            f"`{v['macro_f1']:.4f}` | `{v['delta_macro']:+.4f}` | "
            f"`{v['micro_f1']:.4f}` | `{v['delta_micro']:+.4f}` | {status} |"
        )
    md += [
        "",
        "## Per-Label F1 Breakdown",
        "",
        "| Classifier | anger | disgust | fear | joy | sadness | surprise |",
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
        f"**Best individual classifier:** `{best_name}`  ",
        f"**Val Macro-F1:** `{best['macro_f1']:.4f}` | **Val Micro-F1:** `{best['micro_f1']:.4f}`",
        "",
        "> Selection rule: A challenger beats C1 only if Macro-F1 > 0.4921 AND "
        "no label drops >0.01 F1 below C1.",
        "",
        "## Deliverables",
        "- `scripts/experiments/phase9_classifier_comparison.py`",
        "- `results/ablations/phase9_classifiers/phase9_report.json`",
        "- `tests/test_phase9_classifiers.py`",
    ]
    md_path = os.path.join(RESULTS_DIR, "phase9_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  Markdown: {md_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PHASE 9 COMPLETE — Individual Classifier Comparison")
    print("=" * 75)
    for k, v in results.items():
        tag = " ← SELECTED" if k == best_name else ""
        print(f"  [{k}]{tag}")
        print(f"    Macro-F1={v['macro_f1']:.4f}  Micro-F1={v['micro_f1']:.4f}  "
              f"Δ={v['delta_macro']:+.4f}  Time={v['time_s']}s")
    print("=" * 75)
    print(f"  Best individual classifier: [{best_name}]")
    print(f"  Macro-F1={best['macro_f1']:.4f} | Micro-F1={best['micro_f1']:.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()

