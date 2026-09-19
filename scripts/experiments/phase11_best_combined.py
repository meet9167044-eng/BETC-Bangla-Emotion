"""
Phase 11 — Ablation Group D (A_BEST)
=====================================
Combines the best hyperparameters found in Groups A, B, and C to evaluate
the maximum potential of the ClassifierChain architecture.

Best individual components based on Val Macro-F1:
- C = 1.0 (from A_C1 / default, as C=5.0 degraded when CW=None)
- class_weight = None (from A_CW1, val F1 0.5133)
- chain_order = rare_middle (from A_ORD4, val F1 0.5125)

Combining these:
- C = 1.0
- class_weight = None
- chain_order = rare_middle
"""

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings("ignore")

import os, json, datetime, time
import numpy as np
import scipy.sparse as sp

import sys
import os

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

from src.models.betc import BETCModel, compute_label_order, TARGET_COLS
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

FEAT_DIR = os.path.join(WORKSPACE, "artifacts", "features")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations")

print("Loading Phase 7 feature matrices...")
X_train = sp.load_npz(os.path.join(FEAT_DIR, "X_train.npz"))
X_val   = sp.load_npz(os.path.join(FEAT_DIR, "X_val.npz"))
X_test  = sp.load_npz(os.path.join(FEAT_DIR, "X_test.npz"))
Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))
Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy"))

rare_middle_names = ["joy", "anger", "fear", "surprise", "disgust", "sadness"]
order_indices = [TARGET_COLS.index(name) for name in rare_middle_names]

C_val = 1.0
class_weight_val = None

print(f"Running A_BEST (Group D)")
print(f"C={C_val}, class_weight={class_weight_val}, order={rare_middle_names}")

t0 = time.time()
base_clf = LogisticRegression(C=C_val, class_weight=class_weight_val, max_iter=2000, solver="saga", random_state=42)
chain = ClassifierChain(base_clf, order=order_indices, random_state=42)
chain.fit(X_train, Y_train)
fit_time = time.time() - t0
print(f"Fit time: {fit_time:.1f}s")

P_val = chain.predict_proba(X_val)
thresholds = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
Y_val_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
val_metrics = compute_metrics(Y_val, Y_val_pred, TARGET_COLS)
print(f"Val Macro-F1: {val_metrics['macro_f1']:.4f}")

P_test = chain.predict_proba(X_test)
Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)
test_metrics = compute_metrics(Y_test, Y_test_pred, TARGET_COLS)
print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")

result = {
    "experiment_id": "A_BEST",
    "description": "Group D: Best Combined (C=1.0, CW=None, Order=Rare-middle)",
    "hyperparameters": {"C": C_val, "class_weight": class_weight_val, "chain_order": rare_middle_names, "chain_seed": 42},
    "thresholds": thresholds,
    "validation_metrics": val_metrics,
    "test_metrics": test_metrics,
    "fit_time_seconds": round(fit_time, 2)
}

with open(os.path.join(RESULTS_DIR, "A_BEST.json"), "w") as f:
    json.dump(result, f, indent=2)
print("Saved A_BEST.json")
