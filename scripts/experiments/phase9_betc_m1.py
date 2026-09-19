"""
Phase 9 — BETC Training (M1)
==============================
Full BETC pipeline per PIPELINE_SPEC.md:
  1. Load Phase 7 feature matrices (X_train, X_val, X_test, Y_*)
  2. Compute chain label order (descending training-set frequency)
  3. Fit ClassifierChain(LR(C=1.0, balanced)) on TRAIN only
  4. Optimize per-class thresholds on VAL probabilities only (tau 0.05→0.95)
  5. Evaluate on TEST exactly once (thresholds frozen)
  6. Save model artifacts, thresholds, config, results

EVALUATION_PROTOCOL.md leakage rules strictly followed.
DO NOT PROCEED TO PHASE 10/ABLATIONS without user instruction.
"""

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")  # suppress convergence warnings to stdout

import os, json, datetime, time
import numpy as np
import scipy.sparse as sp
import joblib

sys.path.insert(0, r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion")
from src.models.betc import BETCModel, compute_label_order, TARGET_COLS
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds

WORKSPACE     = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
FEAT_DIR      = os.path.join(WORKSPACE, "artifacts", "features")
MODEL_DIR     = os.path.join(WORKSPACE, "artifacts", "models")
THRESH_PATH   = os.path.join(WORKSPACE, "artifacts", "thresholds.json")
VEC_DIR       = os.path.join(WORKSPACE, "artifacts", "vectorizers")
RESULTS_DIR   = os.path.join(WORKSPACE, "results", "betc_full")
CONFIG_DIR    = os.path.join(WORKSPACE, "configs", "betc_full")
LOG_DIR       = os.path.join(WORKSPACE, "logs", "betc_full")
RUN_TS        = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)

# ─── Step 1: Load feature matrices ───────────────────────────────────────────
print("Loading Phase 7 feature matrices...")
X_train = sp.load_npz(os.path.join(FEAT_DIR, "X_train.npz"))
X_val   = sp.load_npz(os.path.join(FEAT_DIR, "X_val.npz"))
X_test  = sp.load_npz(os.path.join(FEAT_DIR, "X_test.npz"))
Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))
Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy"))
print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")
print(f"Y_train: {Y_train.shape} | Y_val: {Y_val.shape} | Y_test: {Y_test.shape}")

# ─── Step 2: Compute chain label order ───────────────────────────────────────
print("\nComputing chain label order (descending training-set frequency)...")
order_indices, order_names, label_freqs = compute_label_order(Y_train, TARGET_COLS)
print(f"  Label frequencies (train): {label_freqs}")
print(f"  Chain order: {order_names}")

# ─── Step 3: Fit ClassifierChain on TRAIN only ───────────────────────────────
print("\nFitting BETC ClassifierChain on TRAIN split only...")
t0 = time.time()
model = BETCModel(C=1.0, max_iter=2000, solver="saga", random_state=42)
model.fit(X_train, Y_train, label_names=TARGET_COLS)
fit_time = time.time() - t0
print(f"  Fit time: {fit_time:.1f}s")
print(f"  Chain order confirmed: {model.chain_order_names}")

# ─── Step 4: Threshold optimization on VAL only ──────────────────────────────
print("\nOptimizing per-class thresholds on VAL probabilities only...")
t1 = time.time()
thresholds = model.optimize_thresholds(X_val, Y_val, label_names=TARGET_COLS)
threshold_time = time.time() - t1
print(f"  Threshold time: {threshold_time:.1f}s")
print(f"  Thresholds: { {k: round(v,2) for k,v in thresholds.items()} }")

# Val metrics (using frozen thresholds)
P_val = model.predict_proba(X_val)
Y_val_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
val_metrics = compute_metrics(Y_val, Y_val_pred, TARGET_COLS)
print(f"\n  Val  Macro-F1: {val_metrics['macro_f1']:.4f}  "
      f"Micro-F1: {val_metrics['micro_f1']:.4f}  "
      f"Hamming: {val_metrics['hamming_loss']:.4f}")

# ─── Step 5: Test evaluation — EXACTLY ONCE ──────────────────────────────────
print("\nEvaluating on TEST split (ONE-SHOT — thresholds frozen from val)...")
t2 = time.time()
P_test = model.predict_proba(X_test)
Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)
test_time = time.time() - t2
test_metrics = compute_metrics(Y_test, Y_test_pred, TARGET_COLS)
print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}  "
      f"Micro-F1: {test_metrics['micro_f1']:.4f}  "
      f"Hamming: {test_metrics['hamming_loss']:.4f}")
print(f"  Test evaluation time: {test_time:.1f}s")

total_time = time.time() - t0

# Per-class test F1
print("\n  Per-class Test F1:")
for label, vals in test_metrics["per_class"].items():
    print(f"    {label:<10}  F1={vals['f1']:.4f}  P={vals['precision']:.4f}  R={vals['recall']:.4f}")

# ─── Step 6a: Save model and thresholds ──────────────────────────────────────
print("\nSaving model artifacts...")
model.save(MODEL_DIR, THRESH_PATH)
print(f"  Saved: {os.path.join(MODEL_DIR, 'betc_full_chain.joblib')}")
print(f"  Saved: {THRESH_PATH}")

# ─── Step 6b: Save config (betc_full.yaml) ───────────────────────────────────
config = {
    "experiment_id": "M1",
    "category": "betc_full",
    "description": "Full BETC — Combined Word+Char TF-IDF → ClassifierChain (frequency-ordered) → LogisticRegression(C=1.0, balanced) → per-class val thresholds",
    "run_timestamp": RUN_TS,
    "split_identifier": "split_v1 (artifacts/splits/split_v1.json)",
    "feature_source": "artifacts/features/ (Phase 7 output)",
    "features": {
        "word_tfidf": {
            "ngram_range": [1, 2],
            "max_features": 10000,
            "sublinear_tf": True,
            "analyzer": "word",
        },
        "char_tfidf": {
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "max_features": 15000,
            "sublinear_tf": True,
        },
        "fusion": "scipy.sparse.hstack",
        "combined_dim": 25000,
    },
    "model": {
        "type": "ClassifierChain",
        "base_classifier": "LogisticRegression",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 2000,
        "solver": "saga",
        "random_state": 42,
    },
    "chain_label_order": order_names,
    "chain_label_order_indices": order_indices,
    "label_frequencies_train": label_freqs,
    "ordering_rule": "Descending training-set label frequency [INITIAL-HP per PIPELINE_SPEC.md §3.5]",
    "threshold_selection": {
        "method": "Per-class F1 maximization on validation probabilities",
        "tau_range": [0.05, 0.95],
        "tau_step": 0.01,
        "data_used": "Validation split ONLY — test set NOT observed during threshold selection",
    },
    "thresholds": thresholds,
    "artifacts": {
        "chain_model": "artifacts/models/betc_full_chain.joblib",
        "thresholds": "artifacts/thresholds.json",
        "config": "configs/betc_full/betc_full.json",
    },
}

config_path = os.path.join(CONFIG_DIR, "betc_full.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"  Saved config: {config_path}")

# ─── Step 6c: Save results ───────────────────────────────────────────────────
result = {
    "experiment_id": "M1",
    "category": "betc_full",
    "description": config["description"],
    "run_timestamp": RUN_TS,
    "split_identifier": "split_v1 (artifacts/splits/split_v1.json)",
    "feature_source": "artifacts/features/ (Phase 7 output)",
    "config_file": "configs/betc_full/betc_full.json",
    "chain_order": order_names,
    "label_frequencies_train": label_freqs,
    "thresholds": thresholds,
    "threshold_selection": "Per-class F1 max on val probabilities, tau [0.05,0.95] step 0.01",
    "timing_seconds": {
        "fit": round(fit_time, 2),
        "threshold_optimization": round(threshold_time, 2),
        "test_evaluation": round(test_time, 2),
        "total": round(total_time, 2),
    },
    "validation_metrics": val_metrics,
    "test_metrics": test_metrics,
    "leakage_declaration": (
        "ClassifierChain fitted on training split only. "
        "Per-class thresholds selected from validation probabilities only. "
        "Test set evaluated exactly once with frozen thresholds."
    ),
    "note_no_ablations": (
        "This is the canonical M1 result. "
        "Ablations (A1–A9) will be run separately in results/ablations/."
    ),
}

result_path = os.path.join(RESULTS_DIR, "m1.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"  Saved result: {result_path}")

# ─── Final Summary ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("PHASE 9 — BETC (M1) — FINAL RESULTS")
print("="*65)
print(f"  Chain order: {order_names}")
print(f"\n  {'Metric':<25} {'Val':>8} {'Test':>8}")
print(f"  {'-'*42}")
print(f"  {'Macro-F1':<25} {val_metrics['macro_f1']:>8.4f} {test_metrics['macro_f1']:>8.4f}")
print(f"  {'Micro-F1':<25} {val_metrics['micro_f1']:>8.4f} {test_metrics['micro_f1']:>8.4f}")
print(f"  {'Macro-Precision':<25} {val_metrics['macro_precision']:>8.4f} {test_metrics['macro_precision']:>8.4f}")
print(f"  {'Macro-Recall':<25} {val_metrics['macro_recall']:>8.4f} {test_metrics['macro_recall']:>8.4f}")
print(f"  {'Hamming Loss':<25} {val_metrics['hamming_loss']:>8.6f} {test_metrics['hamming_loss']:>8.6f}")
print(f"  {'Jaccard':<25} {val_metrics['jaccard']:>8.4f} {test_metrics['jaccard']:>8.4f}")
print(f"  {'Subset Accuracy':<25} {val_metrics['subset_accuracy']:>8.4f} {test_metrics['subset_accuracy']:>8.4f}")
print(f"\n  Per-class Test F1:")
print(f"  {'Label':<12} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Threshold':>10}")
print(f"  {'-'*48}")
for label in TARGET_COLS:
    pc = test_metrics["per_class"][label]
    tau = thresholds[label]
    print(f"  {label:<12} {pc['f1']:>7.4f} {pc['precision']:>7.4f} {pc['recall']:>7.4f} {tau:>10.2f}")

print("\n  Baselines comparison (Test Macro-F1):")
print(f"  {'B1 (Word+LR)':<30} 0.4825")
print(f"  {'B2 (Char+LR)':<30} 0.4914")
print(f"  {'B3 (Combined+Indep.LR)':<30} 0.5120")
print(f"  {'B4 (Combined+SVM)':<30} 0.4823")
print(f"  {'B5 (Combined+RF)':<30} 0.5150")
print(f"  {'M1 (BETC Chain)':<30} {test_metrics['macro_f1']:.4f}  <-- THIS RUN")

print(f"\n  Total fit+eval time: {total_time:.1f}s")
print("\nPhase 9 COMPLETE. DO NOT proceed to Phase 10 (Ablations) without user instruction.")
