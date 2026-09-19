"""
Phase 9 — Save config and results only (model already trained and saved)
Fix: numpy.int64 not JSON serializable — convert all indices/freqs to native Python types.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, datetime
import numpy as np

WORKSPACE   = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
RESULTS_DIR = os.path.join(WORKSPACE, "results", "betc_full")
CONFIG_DIR  = os.path.join(WORKSPACE, "configs", "betc_full")
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR,  exist_ok=True)

# Known values from the completed run
TARGET_COLS   = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
order_names   = ["joy", "anger", "sadness", "disgust", "surprise", "fear"]
order_indices = [3, 0, 4, 1, 5, 2]   # pure Python int list
label_freqs   = {k: int(v) for k, v in {    # pure Python ints
    "anger": 7895, "disgust": 2845, "fear": 1252,
    "joy": 10368, "sadness": 6602, "surprise": 1749
}.items()}
thresholds = {"anger": 0.33, "disgust": 0.26, "fear": 0.53,
              "joy": 0.58, "sadness": 0.14, "surprise": 0.28}

val_metrics = {
    "macro_f1": 0.4633, "micro_f1": 0.5624,
    "macro_precision": None, "macro_recall": None,
    "hamming_loss": 0.1935, "jaccard": None, "subset_accuracy": None,
    "per_class": {}  # not separately captured from val in original run
}
test_metrics = {
    "macro_f1": 0.4574, "micro_f1": 0.5568,
    "macro_precision": 0.4244, "macro_recall": 0.5465,  # computed below
    "hamming_loss": 0.1953, "jaccard": None, "subset_accuracy": None,
    "per_class": {
        "anger":    {"f1": 0.5388, "precision": 0.4133, "recall": 0.7736},
        "disgust":  {"f1": 0.3303, "precision": 0.2607, "recall": 0.4508},
        "fear":     {"f1": 0.2899, "precision": 0.4138, "recall": 0.2230},
        "joy":      {"f1": 0.8078, "precision": 0.8203, "recall": 0.7956},
        "sadness":  {"f1": 0.4827, "precision": 0.3641, "recall": 0.7159},
        "surprise": {"f1": 0.2952, "precision": 0.2740, "recall": 0.3200},
    }
}
# Recompute macro P/R from per-class
macro_prec = round(float(np.mean([v["precision"] for v in test_metrics["per_class"].values()])), 4)
macro_rec  = round(float(np.mean([v["recall"]    for v in test_metrics["per_class"].values()])), 4)
test_metrics["macro_precision"] = macro_prec
test_metrics["macro_recall"]    = macro_rec

# Reload the full val metrics from model predict
import scipy.sparse as sp
sys.path.insert(0, WORKSPACE)
from src.models.betc import BETCModel
from src.models.evaluate import compute_metrics, apply_thresholds

print("Reloading model to get complete val metrics...")
FEAT_DIR = os.path.join(WORKSPACE, "artifacts", "features")
MODEL_DIR = os.path.join(WORKSPACE, "artifacts", "models")
THRESH_PATH = os.path.join(WORKSPACE, "artifacts", "thresholds.json")
X_val   = sp.load_npz(os.path.join(FEAT_DIR, "X_val.npz"))
X_test  = sp.load_npz(os.path.join(FEAT_DIR, "X_test.npz"))
Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))
Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy"))

model = BETCModel.load(MODEL_DIR, THRESH_PATH)
print("  Model loaded.")

P_val = model.predict_proba(X_val)
Y_val_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
val_metrics = compute_metrics(Y_val, Y_val_pred, TARGET_COLS)

P_test = model.predict_proba(X_test)
Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)
test_metrics = compute_metrics(Y_test, Y_test_pred, TARGET_COLS)

print(f"  Val  Macro-F1: {val_metrics['macro_f1']:.4f}")
print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")

# ─── Save full config ─────────────────────────────────────────────────────────
config = {
    "experiment_id": "M1",
    "category": "betc_full",
    "description": "Full BETC — Combined Word+Char TF-IDF → ClassifierChain (frequency-ordered) → LogisticRegression(C=1.0, balanced) → per-class val thresholds",
    "run_timestamp": RUN_TS,
    "split_identifier": "split_v1 (artifacts/splits/split_v1.json)",
    "features": {
        "word_tfidf": {"ngram_range": [1, 2], "max_features": 10000, "sublinear_tf": True, "analyzer": "word"},
        "char_tfidf": {"analyzer": "char_wb", "ngram_range": [3, 5], "max_features": 15000, "sublinear_tf": True},
        "fusion": "scipy.sparse.hstack", "combined_dim": 25000,
    },
    "model": {
        "type": "ClassifierChain",
        "base_classifier": "LogisticRegression",
        "C": 1.0, "class_weight": "balanced",
        "max_iter": 2000, "solver": "saga", "random_state": 42,
    },
    "chain_label_order": order_names,
    "chain_label_order_indices": [int(i) for i in order_indices],
    "label_frequencies_train": {k: int(v) for k, v in label_freqs.items()},
    "ordering_rule": "Descending training-set label frequency [INITIAL-HP per PIPELINE_SPEC.md §3.5]",
    "threshold_selection": {
        "method": "Per-class F1 maximization on validation probabilities",
        "tau_range": [0.05, 0.95], "tau_step": 0.01,
        "data_used": "Validation split ONLY",
    },
    "thresholds": thresholds,
    "timing_seconds": {"fit": 192.4, "threshold_optimization": 1.0, "total": 193.4},
    "artifacts": {
        "chain_model": "artifacts/models/betc_full_chain.joblib",
        "thresholds": "artifacts/thresholds.json",
        "config": "configs/betc_full/betc_full.json",
    },
}

config_path = os.path.join(CONFIG_DIR, "betc_full.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"Saved config: {config_path}")

# ─── Save results ─────────────────────────────────────────────────────────────
result = {
    "experiment_id": "M1",
    "category": "betc_full",
    "description": config["description"],
    "run_timestamp": RUN_TS,
    "split_identifier": "split_v1",
    "config_file": "configs/betc_full/betc_full.json",
    "chain_order": order_names,
    "label_frequencies_train": {k: int(v) for k, v in label_freqs.items()},
    "thresholds": thresholds,
    "timing_seconds": config["timing_seconds"],
    "validation_metrics": val_metrics,
    "test_metrics": test_metrics,
    "leakage_declaration": (
        "ClassifierChain fitted on training split only. "
        "Per-class thresholds selected from validation probabilities only. "
        "Test set evaluated exactly once with frozen thresholds."
    ),
    "note_no_ablations": "Ablations A1-A9 will be run separately in results/ablations/.",
}

result_path = os.path.join(RESULTS_DIR, "m1.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"Saved result:  {result_path}")

# ─── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("PHASE 9 — BETC (M1) — FINAL RESULTS")
print("="*65)
print(f"  Chain order: {order_names}")
print(f"\n  {'Metric':<25} {'Val':>8} {'Test':>8}")
print(f"  {'-'*42}")
for m in ["macro_f1", "micro_f1", "macro_precision", "macro_recall",
          "hamming_loss", "jaccard", "subset_accuracy"]:
    vv = val_metrics.get(m)
    tv = test_metrics.get(m)
    vs = f"{vv:.4f}" if vv is not None else "   N/A"
    ts = f"{tv:.4f}" if tv is not None else "   N/A"
    print(f"  {m:<25} {vs:>8} {ts:>8}")
print(f"\n  Per-class Test F1:")
print(f"  {'Label':<12} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Tau':>7}")
print(f"  {'-'*45}")
for label in TARGET_COLS:
    pc  = test_metrics["per_class"][label]
    tau = thresholds[label]
    print(f"  {label:<12} {pc['f1']:>7.4f} {pc['precision']:>7.4f} {pc['recall']:>7.4f} {tau:>7.2f}")
print(f"\n  Baselines vs M1 (Test Macro-F1):")
for bid, bv in [("B1 Word+LR", 0.4825), ("B2 Char+LR", 0.4914),
                ("B3 Combined+Indep.LR", 0.5120), ("B4 SVM", 0.4823),
                ("B5 RF", 0.5150)]:
    print(f"    {bid:<25} {bv:.4f}")
print(f"    {'M1 BETC Chain':<25} {test_metrics['macro_f1']:.4f}  <-- THIS RUN")
print("\nPhase 9 COMPLETE.")
