"""
Phase 8 — Re-run B4 and B5 only
(B1/B2/B3 already saved successfully)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore")

import os, json, datetime, time
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion")
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds, TARGET_COLS

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier

WORKSPACE   = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
FEAT_DIR    = os.path.join(WORKSPACE, "artifacts", "features")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "baselines")
CONFIGS_DIR = os.path.join(WORKSPACE, "configs", "baselines")
RUN_TS      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("Loading feature matrices...")
X_train = sp.load_npz(os.path.join(FEAT_DIR, "X_train.npz"))
X_val   = sp.load_npz(os.path.join(FEAT_DIR, "X_val.npz"))
X_test  = sp.load_npz(os.path.join(FEAT_DIR, "X_test.npz"))
Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))
Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy"))
print(f"Loaded: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")


def _safe(v):
    if isinstance(v, (str, int, float, bool, type(None))): return v
    if isinstance(v, (list, tuple)): return [_safe(i) for i in v]
    if isinstance(v, dict): return {kk: _safe(vv) for kk, vv in v.items()}
    return repr(v)

def _get_hyperparams(clf):
    params = {}
    if hasattr(clf, "estimator"):
        est = clf.estimator
        params["wrapper"] = type(clf).__name__
        params["estimator_type"] = type(est).__name__
        raw = est.get_params(deep=True)
        params["estimator_params"] = {k: _safe(v) for k, v in raw.items()}
    else:
        raw = clf.get_params(deep=False)
        params = {k: _safe(v) for k, v in raw.items()}
    return params


def run_baseline(baseline_id, description, X_tr, X_va, X_te, Y_tr, Y_va, Y_te, model_factory, representation):
    print(f"\n{'='*60}")
    print(f"Running {baseline_id} — {description}")
    print(f"{'='*60}")
    t0 = time.time()

    print(f"  Fitting on train ({X_tr.shape[0]:,} samples)...")
    clf = model_factory()
    clf.fit(X_tr, Y_tr)
    fit_time = time.time() - t0
    print(f"  Fit time: {fit_time:.1f}s")

    t1 = time.time()
    print("  Computing val probabilities for threshold optimization...")
    P_val = clf.predict_proba(X_va)
    if isinstance(P_val, list):
        P_val = np.column_stack([p[:, 1] for p in P_val])

    thresholds = optimize_thresholds(P_val, Y_va, label_names=TARGET_COLS)
    print(f"  Thresholds: { {k: round(v,2) for k,v in thresholds.items()} }")

    Y_val_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
    val_metrics = compute_metrics(Y_va, Y_val_pred, TARGET_COLS)
    print(f"  Val  Macro-F1: {val_metrics['macro_f1']:.4f}  Micro-F1: {val_metrics['micro_f1']:.4f}")

    print("  Evaluating on test split (ONE-SHOT — thresholds frozen from val)...")
    P_test = clf.predict_proba(X_te)
    if isinstance(P_test, list):
        P_test = np.column_stack([p[:, 1] for p in P_test])
    Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)
    test_metrics = compute_metrics(Y_te, Y_test_pred, TARGET_COLS)
    print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}  Micro-F1: {test_metrics['micro_f1']:.4f}")

    total_time = time.time() - t0

    result = {
        "experiment_id": baseline_id,
        "category": "baseline",
        "description": description,
        "representation": representation,
        "run_timestamp": RUN_TS,
        "split_identifier": "split_v1 (artifacts/splits/split_v1.json)",
        "feature_source": "artifacts/features/ (Phase 7 output)",
        "config_file": f"configs/baselines/{baseline_id.lower()}.json",
        "hyperparameters": _get_hyperparams(clf),
        "thresholds": thresholds,
        "threshold_selection": "Per-class F1 maximization on validation set, tau sweep [0.05,0.95] step 0.01",
        "timing_seconds": {
            "fit": round(fit_time, 2),
            "total": round(total_time, 2),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "leakage_declaration": "Thresholds selected from validation probabilities only. Test evaluated exactly once.",
    }

    result_path = os.path.join(RESULTS_DIR, f"{baseline_id.lower()}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    config = {
        "experiment_id": baseline_id, "category": "baseline",
        "description": description, "representation": representation,
        "hyperparameters": _get_hyperparams(clf),
        "thresholds": thresholds, "run_timestamp": RUN_TS,
    }
    config_path = os.path.join(CONFIGS_DIR, f"{baseline_id.lower()}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {result_path}")
    return result


# B4 — Combined + Linear SVM
b4_result = run_baseline(
    baseline_id="B4",
    description="Combined TF-IDF + Linear SVM (CalibratedClassifierCV, MultiOutputClassifier)",
    X_tr=X_train, X_va=X_val, X_te=X_test,
    Y_tr=Y_train, Y_va=Y_val, Y_te=Y_test,
    model_factory=lambda: MultiOutputClassifier(
        CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
            cv=3, method="sigmoid"
        ), n_jobs=-1
    ),
    representation="Word+Char TF-IDF combined (25,000 dims)",
)

# B5 — Combined + Random Forest
b5_result = run_baseline(
    baseline_id="B5",
    description="Combined TF-IDF + Random Forest (MultiOutputClassifier)",
    X_tr=X_train, X_va=X_val, X_te=X_test,
    Y_tr=Y_train, Y_va=Y_val, Y_te=Y_test,
    model_factory=lambda: MultiOutputClassifier(
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               n_jobs=-1, random_state=42),
        n_jobs=1
    ),
    representation="Word+Char TF-IDF combined (25,000 dims)",
)

# ── Now build the final summary from all 5 saved JSONs ────────────────────────
print("\nBuilding final summary from all 5 results...")
all_results = []
for bid in ["b1", "b2", "b3", "b4", "b5"]:
    path = os.path.join(RESULTS_DIR, f"{bid}.json")
    with open(path, encoding="utf-8") as f:
        all_results.append(json.load(f))

print("\n" + "="*72)
print("PHASE 8 BASELINES — FINAL SUMMARY")
print("="*72)
print(f"{'ID':<4} {'Description':<48} {'Val MacF1':>9} {'Test MacF1':>10}")
print("-"*72)
for r in all_results:
    print(f"{r['experiment_id']:<4} "
          f"{r['description'][:47]:<48} "
          f"{r['validation_metrics']['macro_f1']:>9.4f} "
          f"{r['test_metrics']['macro_f1']:>10.4f}")
print("-"*72)

print("\nPer-class Test F1:")
header = f"{'ID':<4}" + "".join(f"{c:>10}" for c in TARGET_COLS)
print(header)
print("-"*72)
for r in all_results:
    row = f"{r['experiment_id']:<4}"
    for c in TARGET_COLS:
        row += f"{r['test_metrics']['per_class'][c]['f1']:>10.4f}"
    print(row)

summary_rows = []
for r in all_results:
    summary_rows.append({
        "experiment_id":    r["experiment_id"],
        "category":         "baseline",
        "description":      r["description"],
        "representation":   r["representation"],
        "run_timestamp":    r["run_timestamp"],
        "val_macro_f1":     r["validation_metrics"]["macro_f1"],
        "val_micro_f1":     r["validation_metrics"]["micro_f1"],
        "val_hamming":      r["validation_metrics"]["hamming_loss"],
        "val_jaccard":      r["validation_metrics"]["jaccard"],
        "val_subset_acc":   r["validation_metrics"]["subset_accuracy"],
        "test_macro_f1":    r["test_metrics"]["macro_f1"],
        "test_micro_f1":    r["test_metrics"]["micro_f1"],
        "test_hamming":     r["test_metrics"]["hamming_loss"],
        "test_jaccard":     r["test_metrics"]["jaccard"],
        "test_subset_acc":  r["test_metrics"]["subset_accuracy"],
        "per_class_test_f1": {k: v["f1"] for k, v in r["test_metrics"]["per_class"].items()},
    })

summary_out = {
    "phase": "Phase 8 — Baselines",
    "category": "baseline",
    "run_timestamp": RUN_TS,
    "split_identifier": "split_v1 (artifacts/splits/split_v1.json)",
    "note_leakage": (
        "All thresholds selected from validation probabilities only. "
        "Test evaluated exactly once per baseline. "
        "No test data was observed during fitting or threshold selection."
    ),
    "note_b6_b7": (
        "B6 (EmoNoBa AdaBoost) and B7 (EmoNoBa BiLSTM) are EXTERNAL CITED "
        "NUMBERS ONLY — not reproduced by this repository. See EXPERIMENT_PLAN.md."
    ),
    "baselines": summary_rows,
}

summary_path = os.path.join(RESULTS_DIR, "baselines_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_out, f, indent=2, ensure_ascii=False)

print(f"\nSaved summary: {summary_path}")
print("\nPhase 8 COMPLETE. DO NOT proceed to Phase 9 without user instruction.")
