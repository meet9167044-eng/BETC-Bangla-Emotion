"""
Phase 13 — Final Evaluation & Cross-Domain Breakdown
=====================================================
Re-trains A_BEST on seed=42 to align test-set predictions with the raw text
for error analysis, and breaks down the Macro-F1 score by source dataset
and domain.
"""

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import os, json
import numpy as np
import pandas as pd
import scipy.sparse as sp

import sys
import os

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

from sklearn.model_selection import train_test_split
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

# Configuration
TEST_CSV_PATH = os.path.join(WORKSPACE, "Data", "processed", "test", "test.csv")
FEAT_DIR = os.path.join(WORKSPACE, "artifacts", "features")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "final")
TARGET_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
RARE_MIDDLE_ORDER = ["joy", "anger", "fear", "surprise", "disgust", "sadness"]
SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Load Data
print("Loading canonical test set from Data/processed/test/test.csv...")
df_test = pd.read_csv(TEST_CSV_PATH)

print("Loading precomputed Phase 7 TF-IDF matrices (Seed 42)...")
X_train = sp.load_npz(os.path.join(FEAT_DIR, "X_train.npz"))
X_val   = sp.load_npz(os.path.join(FEAT_DIR, "X_val.npz"))
X_test  = sp.load_npz(os.path.join(FEAT_DIR, "X_test.npz"))
Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))
Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy"))

# Verify lengths match
if len(df_test) != X_test.shape[0]:
    raise ValueError(f"Length mismatch: test.csv has {len(df_test)} rows but X_test has {X_test.shape[0]} rows.")

# 2. Train A_BEST
print("Training A_BEST Classifier Chain...")
order_indices = [TARGET_COLS.index(name) for name in RARE_MIDDLE_ORDER]
base_clf = LogisticRegression(C=1.0, class_weight=None, max_iter=2000, solver="saga", random_state=SEED)
chain = ClassifierChain(base_clf, order=order_indices, random_state=SEED)
chain.fit(X_train, Y_train)

# 3. Optimize and Predict
print("Optimizing thresholds and predicting...")
P_val = chain.predict_proba(X_val)
thresholds = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
P_test = chain.predict_proba(X_test)
Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)

# Global Test Macro-F1 check
overall_metrics = compute_metrics(Y_test, Y_test_pred, TARGET_COLS)
print(f"Overall Test Macro-F1: {overall_metrics['macro_f1']:.4f}")

# 4. Construct Predictions DataFrame
for i, label in enumerate(TARGET_COLS):
    df_test[f"true_{label}"] = Y_test[:, i]
    df_test[f"pred_{label}"] = Y_test_pred[:, i]
    
df_test.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
print(f"Saved test_predictions.csv ({len(df_test)} rows)")

# 5. Cross-Domain Breakdown
print("Computing cross-domain performance...")
breakdown = {}

def get_breakdown_metrics(df_subset):
    if len(df_subset) == 0: return None
    y_t = df_subset[[f"true_{l}" for l in TARGET_COLS]].values
    y_p = df_subset[[f"pred_{l}" for l in TARGET_COLS]].values
    return compute_metrics(y_t, y_p, TARGET_COLS)

# By Source
for source in df_test['source_dataset'].dropna().unique():
    subset = df_test[df_test['source_dataset'] == source]
    metrics = get_breakdown_metrics(subset)
    breakdown[f"source_{source}"] = {
        "count": len(subset),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"]
    }

# By EmoNoBa Domain
emonoba_subset = df_test[df_test['source_dataset'] == 'EmoNoBa']
for domain in emonoba_subset['domain'].dropna().unique():
    subset = emonoba_subset[emonoba_subset['domain'] == domain]
    metrics = get_breakdown_metrics(subset)
    breakdown[f"domain_{domain}"] = {
        "count": len(subset),
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"]
    }

with open(os.path.join(RESULTS_DIR, "cross_domain_breakdown.json"), "w") as f:
    json.dump(breakdown, f, indent=2)
print("Saved cross_domain_breakdown.json")

for k, v in breakdown.items():
    print(f"  {k:<20} | N={v['count']:<4} | Macro-F1={v['macro_f1']:.4f}")
