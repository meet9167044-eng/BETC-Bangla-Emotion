"""
Phase 12 — Multi-Seed Validation (F3)
=====================================
Executes the full BETC pipeline (A_BEST configuration) across 5 random seeds.
Evaluates the statistical robustness of the model by measuring the variance
induced by train/val/test splitting and any stochastic solver behaviors.

Pipeline per seed:
1. Split harmonized dataset (70/15/15)
2. Preprocess text
3. Fit TF-IDF on train, transform val/test
4. Fit ClassifierChain (A_BEST) on train
5. Optimize thresholds on val
6. Evaluate on test
7. Save full isolated artifact set

Aggregates all metrics (Mean ± Std Dev) at the end.
"""

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import os, json, datetime, time, gc
import numpy as np
import pandas as pd
import scipy.sparse as sp

import sys
import os

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

from sklearn.model_selection import train_test_split
from src.features.preprocessing import preprocess_series
from src.features.tfidf import BETCFeatureExtractor
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

# Configuration
DATA_PATH = os.path.join(WORKSPACE, "Data", "interim", "harmonized_deduplicated.csv")
ARTIFACTS_DIR = os.path.join(WORKSPACE, "artifacts", "multiseed")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "final")
SEEDS = [42, 0, 123, 100, 999]
TARGET_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
RARE_MIDDLE_ORDER = ["joy", "anger", "fear", "surprise", "disgust", "sadness"]

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load raw dataset once
print(f"Loading dataset from: {DATA_PATH}")
df_full = pd.read_csv(DATA_PATH)
X_raw = df_full['text']
Y_raw = df_full[TARGET_COLS].values

seed_results = []

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"RUNNING SEED: {seed}")
    print(f"{'='*60}")
    t0 = time.time()
    
    # 1. Directories
    seed_artifact_dir = os.path.join(ARTIFACTS_DIR, f"seed_{seed}")
    seed_result_dir = os.path.join(RESULTS_DIR, f"seed_{seed}")
    os.makedirs(seed_artifact_dir, exist_ok=True)
    os.makedirs(seed_result_dir, exist_ok=True)
    
    # 2. Split (70/15/15)
    X_temp, X_test_raw, Y_temp, Y_test = train_test_split(X_raw, Y_raw, test_size=0.15, random_state=seed)
    X_train_raw, X_val_raw, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=(0.15/0.85), random_state=seed)
    
    # Save splits metadata
    with open(os.path.join(seed_artifact_dir, "splits.json"), "w") as f:
        json.dump({"train_size": len(X_train_raw), "val_size": len(X_val_raw), "test_size": len(X_test_raw)}, f)
        
    # 3. Preprocess
    print("  Preprocessing...")
    X_train_clean = preprocess_series(X_train_raw).tolist()
    X_val_clean = preprocess_series(X_val_raw).tolist()
    X_test_clean = preprocess_series(X_test_raw).tolist()
    
    # 4. TF-IDF
    print("  Extracting TF-IDF features...")
    extractor = BETCFeatureExtractor()
    X_train = extractor.fit_transform(X_train_clean)
    X_val = extractor.transform(X_val_clean)
    X_test = extractor.transform(X_test_clean)
    extractor.save(seed_artifact_dir)
    
    sp.save_npz(os.path.join(seed_artifact_dir, "X_train.npz"), X_train)
    sp.save_npz(os.path.join(seed_artifact_dir, "X_val.npz"), X_val)
    sp.save_npz(os.path.join(seed_artifact_dir, "X_test.npz"), X_test)
    np.save(os.path.join(seed_artifact_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(seed_artifact_dir, "Y_val.npy"), Y_val)
    np.save(os.path.join(seed_artifact_dir, "Y_test.npy"), Y_test)
    
    # 5. Train A_BEST
    print("  Training A_BEST Classifier Chain...")
    order_indices = [TARGET_COLS.index(name) for name in RARE_MIDDLE_ORDER]
    base_clf = LogisticRegression(C=1.0, class_weight=None, max_iter=2000, solver="saga", random_state=seed)
    chain = ClassifierChain(base_clf, order=order_indices, random_state=seed)
    
    t_fit = time.time()
    chain.fit(X_train, Y_train)
    print(f"    Fit time: {time.time() - t_fit:.1f}s")
    
    # 6. Optimize Thresholds & Evaluate
    P_val = chain.predict_proba(X_val)
    thresholds = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
    
    with open(os.path.join(seed_artifact_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
        
    P_test = chain.predict_proba(X_test)
    Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)
    
    test_metrics = compute_metrics(Y_test, Y_test_pred, TARGET_COLS)
    print(f"  Seed {seed} Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    
    # 7. Save Seed Results
    result = {
        "seed": seed,
        "total_time_seconds": round(time.time() - t0, 2),
        "test_metrics": test_metrics
    }
    with open(os.path.join(seed_result_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
        
    seed_results.append(result)
    
    # Clear memory
    del X_train, X_val, X_test, chain, extractor
    gc.collect()

# 8. Aggregate Means and Std Devs
print("\n" + "="*60)
print("AGGREGATING METRICS OVER 5 SEEDS")
print("="*60)

metrics_keys = ['macro_f1', 'micro_f1', 'macro_precision', 'macro_recall', 'hamming_loss', 'jaccard', 'subset_accuracy']
summary = {"seeds": SEEDS, "metrics": {}, "per_class": {}}

for key in metrics_keys:
    vals = [res["test_metrics"][key] for res in seed_results]
    summary["metrics"][key] = {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals))
    }
    print(f"{key:<18}: {summary['metrics'][key]['mean']:.4f} ± {summary['metrics'][key]['std']:.4f}")

for label in TARGET_COLS:
    summary["per_class"][label] = {}
    for metric in ['f1', 'precision', 'recall']:
        vals = [res["test_metrics"]["per_class"][label][metric] for res in seed_results]
        summary["per_class"][label][metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }

with open(os.path.join(RESULTS_DIR, "multiseed_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved final summary to results/final/multiseed_summary.json")
