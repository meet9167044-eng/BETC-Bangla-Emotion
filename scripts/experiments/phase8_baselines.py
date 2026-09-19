"""
Phase 8 — Baselines B1–B5
==========================
Implements and evaluates the five baselines from EXPERIMENT_PLAN.md Section 1.

Each baseline:
  1. Uses the SAME frozen split as BETC (X_train/val/test from Phase 7)
  2. Fits on TRAINING data only
  3. Optimizes per-class thresholds on VALIDATION predicted probabilities
  4. Evaluates on TEST set exactly ONCE per baseline (final step)
  5. Saves config + results to results/baselines/<ID>.json

Baselines:
  B1: Word-only TF-IDF + Logistic Regression (MultiOutputClassifier)
  B2: Char-only TF-IDF + Logistic Regression (MultiOutputClassifier)
  B3: Combined TF-IDF  + MultiOutputClassifier (LR, no chain)
  B4: Combined TF-IDF  + Linear SVM (MultiOutputClassifier)
  B5: Combined TF-IDF  + Random Forest (MultiOutputClassifier)

EVALUATION_PROTOCOL.md leakage rules strictly followed:
  - No fitting of any kind on val or test
  - Thresholds selected from val probabilities ONLY
  - Test evaluated exactly once per baseline
  - category tag = "baseline" on every result

Note on SVM (B4): LinearSVC does not natively support predict_proba.
We use CalibratedClassifierCV(LinearSVC(...)) for probability estimates,
which is the standard sklearn approach for SVM threshold tuning.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, datetime, time
import numpy as np
import scipy.sparse as sp
import joblib

sys.path.insert(0, r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion")
from src.models.evaluate import (
    compute_metrics, optimize_thresholds, apply_thresholds, TARGET_COLS
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier

WORKSPACE    = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
ARTIFACTS    = os.path.join(WORKSPACE, "artifacts")
FEAT_DIR     = os.path.join(ARTIFACTS, "features")
VEC_DIR      = os.path.join(ARTIFACTS, "vectorizers")
RESULTS_DIR  = os.path.join(WORKSPACE, "results", "baselines")
CONFIGS_DIR  = os.path.join(WORKSPACE, "configs", "baselines")
RUN_TS       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)

# ─── Load feature matrices from Phase 7 ─────────────────────────────────────
print("Loading Phase 7 feature matrices...")
X_train = sp.load_npz(os.path.join(FEAT_DIR, "X_train.npz"))
X_val   = sp.load_npz(os.path.join(FEAT_DIR, "X_val.npz"))
X_test  = sp.load_npz(os.path.join(FEAT_DIR, "X_test.npz"))
Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))
Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy"))

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

# Word-only and char-only slices (B1, B2)
# Word TF-IDF occupies first 10,000 cols; char TF-IDF the remaining 15,000
WORD_DIM = 10_000
CHAR_DIM = 15_000
X_train_word = X_train[:, :WORD_DIM]
X_val_word   = X_val[:,   :WORD_DIM]
X_test_word  = X_test[:,  :WORD_DIM]
X_train_char = X_train[:, WORD_DIM:]
X_val_char   = X_val[:,   WORD_DIM:]
X_test_char  = X_test[:,  WORD_DIM:]

print("Feature slices: word=[:10000], char=[10000:]")


# ─── Baseline runner ─────────────────────────────────────────────────────────

def run_baseline(
    baseline_id: str,
    description: str,
    X_tr, X_va, X_te,
    Y_tr, Y_va, Y_te,
    model_factory,
    representation: str,
):
    """
    Generic baseline runner:
      1. Fit on train
      2. Predict proba on val → optimize thresholds (val only, no leakage)
      3. Apply thresholds to val proba → val metrics
      4. Predict proba on test → apply thresholds → test metrics (one-shot)
      5. Save config + results JSON

    Parameters
    ----------
    baseline_id   : str, e.g. "B1"
    description   : str, human-readable model description
    X_tr, X_va, X_te : feature matrices
    Y_tr, Y_va, Y_te : label matrices
    model_factory : callable() → fitted sklearn estimator
    representation : str, feature description
    """
    print(f"\n{'='*60}")
    print(f"Running {baseline_id} — {description}")
    print(f"{'='*60}")

    t0 = time.time()

    # 1. Fit on training data only
    print(f"  Fitting on train ({X_tr.shape[0]} samples)...")
    clf = model_factory()
    clf.fit(X_tr, Y_tr)
    fit_time = time.time() - t0
    print(f"  Fit time: {fit_time:.1f}s")

    # 2. Val probabilities → threshold optimization (val only, no leakage)
    print("  Computing val probabilities for threshold optimization...")
    t1 = time.time()
    P_val = clf.predict_proba(X_va)
    # MultiOutputClassifier returns list of (n,2) arrays; extract positive class
    if isinstance(P_val, list):
        P_val = np.column_stack([p[:, 1] for p in P_val])
    val_proba_time = time.time() - t1

    print("  Optimizing per-class thresholds on val probabilities...")
    thresholds = optimize_thresholds(P_val, Y_va, label_names=TARGET_COLS)
    print(f"  Thresholds: { {k: round(v,2) for k,v in thresholds.items()} }")

    # 3. Val metrics
    Y_val_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
    val_metrics = compute_metrics(Y_va, Y_val_pred, TARGET_COLS)
    print(f"  Val  Macro-F1: {val_metrics['macro_f1']:.4f}  Micro-F1: {val_metrics['micro_f1']:.4f}")

    # 4. Test metrics — exactly once, after thresholds are frozen
    print("  Evaluating on test split (ONE-SHOT — thresholds frozen from val)...")
    t2 = time.time()
    P_test = clf.predict_proba(X_te)
    if isinstance(P_test, list):
        P_test = np.column_stack([p[:, 1] for p in P_test])
    Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)
    test_time = time.time() - t2
    test_metrics = compute_metrics(Y_te, Y_test_pred, TARGET_COLS)
    print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}  Micro-F1: {test_metrics['micro_f1']:.4f}")

    total_time = time.time() - t0

    # 5. Build result record
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
        "threshold_selection": (
            "Per-class F1 maximization on validation set, "
            "tau sweep [0.05, 0.95] step 0.01 (PIPELINE_SPEC.md §3.7)"
        ),
        "timing_seconds": {
            "fit": round(fit_time, 2),
            "val_proba": round(val_proba_time, 2),
            "test_proba": round(test_time, 2),
            "total": round(total_time, 2),
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "leakage_declaration": (
            "Thresholds selected from validation probabilities only. "
            "Test set evaluated exactly once, after all decisions frozen."
        ),
    }

    # Save result JSON
    result_path = os.path.join(RESULTS_DIR, f"{baseline_id.lower()}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Save config JSON (lightweight — hyperparams + thresholds)
    config = {
        "experiment_id": baseline_id,
        "category": "baseline",
        "description": description,
        "representation": representation,
        "hyperparameters": _get_hyperparams(clf),
        "thresholds": thresholds,
        "run_timestamp": RUN_TS,
    }
    config_path = os.path.join(CONFIGS_DIR, f"{baseline_id.lower()}.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {result_path}")
    return result


def _get_hyperparams(clf) -> dict:
    """Extract hyperparameters from a fitted sklearn estimator for logging.
    Converts any non-JSON-serializable objects (e.g. nested estimators) to str.
    """
    def _safe(v):
        if isinstance(v, (str, int, float, bool, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return [_safe(i) for i in v]
        if isinstance(v, dict):
            return {kk: _safe(vv) for kk, vv in v.items()}
        # For sklearn estimators or other objects — use repr string
        return repr(v)

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


# ─── B1: Word-only TF-IDF + LR ──────────────────────────────────────────────
b1_result = run_baseline(
    baseline_id="B1",
    description="Word-only TF-IDF + Logistic Regression (MultiOutputClassifier)",
    X_tr=X_train_word, X_va=X_val_word, X_te=X_test_word,
    Y_tr=Y_train, Y_va=Y_val, Y_te=Y_test,
    model_factory=lambda: MultiOutputClassifier(
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                           solver="saga", random_state=42),
        n_jobs=-1
    ),
    representation="Word TF-IDF only (ngram=(1,2), max_features=10000)",
)

# ─── B2: Char-only TF-IDF + LR ──────────────────────────────────────────────
b2_result = run_baseline(
    baseline_id="B2",
    description="Char-only TF-IDF + Logistic Regression (MultiOutputClassifier)",
    X_tr=X_train_char, X_va=X_val_char, X_te=X_test_char,
    Y_tr=Y_train, Y_va=Y_val, Y_te=Y_test,
    model_factory=lambda: MultiOutputClassifier(
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                           solver="saga", random_state=42),
        n_jobs=-1
    ),
    representation="Char TF-IDF only (char_wb, ngram=(3,5), max_features=15000)",
)

# ─── B3: Combined TF-IDF + MultiOutputClassifier (no chain) ─────────────────
b3_result = run_baseline(
    baseline_id="B3",
    description="Combined TF-IDF + MultiOutputClassifier (independent per-label LR, no chain)",
    X_tr=X_train, X_va=X_val, X_te=X_test,
    Y_tr=Y_train, Y_va=Y_val, Y_te=Y_test,
    model_factory=lambda: MultiOutputClassifier(
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000,
                           solver="saga", random_state=42),
        n_jobs=-1
    ),
    representation="Word+Char TF-IDF combined (25,000 dims)",
)

# ─── B4: Combined TF-IDF + Linear SVM ───────────────────────────────────────
# LinearSVC has no predict_proba; use CalibratedClassifierCV(cv=3) for proba.
b4_result = run_baseline(
    baseline_id="B4",
    description="Combined TF-IDF + Linear SVM (CalibratedClassifierCV, MultiOutputClassifier)",
    X_tr=X_train, X_va=X_val, X_te=X_test,
    Y_tr=Y_train, Y_va=Y_val, Y_te=Y_test,
    model_factory=lambda: MultiOutputClassifier(
        CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=2000, random_state=42),
            cv=3, method="sigmoid"
        ),
        n_jobs=-1
    ),
    representation="Word+Char TF-IDF combined (25,000 dims)",
)

# ─── B5: Combined TF-IDF + Random Forest ────────────────────────────────────
b5_result = run_baseline(
    baseline_id="B5",
    description="Combined TF-IDF + Random Forest (MultiOutputClassifier)",
    X_tr=X_train, X_va=X_val, X_te=X_test,
    Y_tr=Y_train, Y_va=Y_val, Y_te=Y_test,
    model_factory=lambda: MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
        n_jobs=1  # parallelism handled inside RF
    ),
    representation="Word+Char TF-IDF combined (25,000 dims)",
)

# ─── Summary table ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 8 BASELINES — SUMMARY")
print("="*70)
header = f"{'ID':<4} {'Model':<50} {'Val MacroF1':>11} {'Test MacroF1':>12}"
print(header)
print("-"*70)
for res in [b1_result, b2_result, b3_result, b4_result, b5_result]:
    row = (f"{res['experiment_id']:<4} "
           f"{res['description'][:49]:<50} "
           f"{res['validation_metrics']['macro_f1']:>11.4f} "
           f"{res['test_metrics']['macro_f1']:>12.4f}")
    print(row)
print("-"*70)

# ─── Save combined summary JSON ───────────────────────────────────────────────
summary_rows = []
for res in [b1_result, b2_result, b3_result, b4_result, b5_result]:
    summary_rows.append({
        "experiment_id":    res["experiment_id"],
        "category":         "baseline",
        "description":      res["description"],
        "representation":   res["representation"],
        "run_timestamp":    res["run_timestamp"],
        "val_macro_f1":     res["validation_metrics"]["macro_f1"],
        "val_micro_f1":     res["validation_metrics"]["micro_f1"],
        "val_hamming":      res["validation_metrics"]["hamming_loss"],
        "val_jaccard":      res["validation_metrics"]["jaccard"],
        "val_subset_acc":   res["validation_metrics"]["subset_accuracy"],
        "test_macro_f1":    res["test_metrics"]["macro_f1"],
        "test_micro_f1":    res["test_metrics"]["micro_f1"],
        "test_hamming":     res["test_metrics"]["hamming_loss"],
        "test_jaccard":     res["test_metrics"]["jaccard"],
        "test_subset_acc":  res["test_metrics"]["subset_accuracy"],
        "per_class_test_f1": {k: v["f1"] for k, v in res["test_metrics"]["per_class"].items()},
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

print(f"\nSaved: {summary_path}")
print("\nPhase 8 COMPLETE. DO NOT proceed to Phase 9 without user instruction.")
