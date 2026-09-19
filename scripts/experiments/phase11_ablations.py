"""
Phase 11 — Ablation Experiments (Groups A, B, C)
================================================
Ablation experiments to diagnose M1 underperformance.
- Group A: LR Regularization Strength (C sweep)
- Group B: Chain Order
- Group C: Class Weight

All models fit on X_train.
Thresholds optimized on X_val ONLY.
Evaluated on X_test EXACTLY ONCE.
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

FEAT_DIR = os.path.join(WORKSPACE, "artifacts", "features")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations")
CONFIG_DIR = os.path.join(WORKSPACE, "configs", "ablations")
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

print("Loading Phase 7 feature matrices...")
X_train = sp.load_npz(os.path.join(FEAT_DIR, "X_train.npz"))
X_val   = sp.load_npz(os.path.join(FEAT_DIR, "X_val.npz"))
X_test  = sp.load_npz(os.path.join(FEAT_DIR, "X_test.npz"))
Y_train = np.load(os.path.join(FEAT_DIR, "Y_train.npy"))
Y_val   = np.load(os.path.join(FEAT_DIR, "Y_val.npy"))
Y_test  = np.load(os.path.join(FEAT_DIR, "Y_test.npy"))

# Pre-compute useful things
_, freq_desc_names, _ = compute_label_order(Y_train, TARGET_COLS)
rev_freq_desc_names = freq_desc_names[::-1]

# For correlation order, compute co-occurrence on train
coocc = Y_train.T @ Y_train
# simple greedy: start with most frequent, then pick one with highest co-occurrence to already picked
corr_names = [freq_desc_names[0]]
remaining = list(TARGET_COLS)
remaining.remove(corr_names[0])
while remaining:
    best_next = None
    best_score = -1
    for cand in remaining:
        # sum of co-occurrences with already picked
        score = sum(coocc[TARGET_COLS.index(cand), TARGET_COLS.index(p)] for p in corr_names)
        if score > best_score:
            best_score = score
            best_next = cand
    corr_names.append(best_next)
    remaining.remove(best_next)

alpha_names = sorted(TARGET_COLS)

# Rare middle: freq order is joy, anger, sadness, disgust, surprise, fear
# We want: joy, anger, fear, surprise, disgust, sadness
rare_middle_names = ["joy", "anger", "fear", "surprise", "disgust", "sadness"]

def run_ablation(exp_id, description, C=1.0, class_weight="balanced", chain_order=None, chain_seed=42):
    print(f"\n{'='*70}")
    print(f"Running {exp_id} — {description}")
    print(f"C={C}, CW={class_weight}, Order={chain_order}, Seed={chain_seed}")
    print(f"{'='*70}")
    
    t0 = time.time()
    
    # Map order names to indices
    order_indices = [TARGET_COLS.index(name) for name in chain_order]
    
    # 1. Fit
    from sklearn.multioutput import ClassifierChain
    from sklearn.linear_model import LogisticRegression
    
    base_clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, solver="saga", random_state=42)
    chain = ClassifierChain(base_clf, order=order_indices, random_state=chain_seed)
    chain.fit(X_train, Y_train)
    fit_time = time.time() - t0
    print(f"  Fit time: {fit_time:.1f}s")
    
    # 2. Val threshold
    t1 = time.time()
    P_val = chain.predict_proba(X_val)
    thresholds = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
    Y_val_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
    val_metrics = compute_metrics(Y_val, Y_val_pred, TARGET_COLS)
    val_time = time.time() - t1
    print(f"  Thresholds: { {k: round(v,2) for k,v in thresholds.items()} }")
    print(f"  Val Macro-F1: {val_metrics['macro_f1']:.4f}")
    
    # 3. Test evaluate
    t2 = time.time()
    P_test = chain.predict_proba(X_test)
    Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)
    test_metrics = compute_metrics(Y_test, Y_test_pred, TARGET_COLS)
    test_time = time.time() - t2
    print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    
    total_time = time.time() - t0
    
    # Save config and result
    config = {
        "experiment_id": exp_id,
        "category": "ablation",
        "description": description,
        "run_timestamp": RUN_TS,
        "hyperparameters": {
            "C": C,
            "class_weight": class_weight,
            "chain_order": chain_order,
            "chain_seed": chain_seed
        }
    }
    
    result = {
        "experiment_id": exp_id,
        "category": "ablation",
        "description": description,
        "run_timestamp": RUN_TS,
        "config": config,
        "thresholds": thresholds,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "timing_seconds": {"total": round(total_time, 2)}
    }
    
    cfg_path = os.path.join(CONFIG_DIR, f"{exp_id}.json")
    res_path = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    with open(res_path, "w") as f:
        json.dump(result, f, indent=2)
        
    return result

# ==========================================
# Group A: LR Regularization (C sweep)
# ==========================================
group_a_results = {}
for exp_id, C_val in [("A_C1", 0.1), ("A_C2", 0.5), ("A_C3", 2.0), ("A_C4", 5.0), ("A_C5", 10.0)]:
    group_a_results[exp_id] = run_ablation(
        exp_id, f"Group A: C={C_val}",
        C=C_val, class_weight="balanced", chain_order=freq_desc_names
    )

# Find best C based on VAL
best_c_id = max(group_a_results, key=lambda k: group_a_results[k]['validation_metrics']['macro_f1'])
best_c_val = group_a_results[best_c_id]['config']['hyperparameters']['C']
print(f"\n[GROUP A BEST C: {best_c_val} (Val F1: {group_a_results[best_c_id]['validation_metrics']['macro_f1']:.4f})]")

# ==========================================
# Group B: Chain Order
# ==========================================
group_b_results = {}
orders = {
    "A_ORD1": ("Reverse frequency", rev_freq_desc_names, 42),
    "A_ORD2": ("Correlation-based", corr_names, 42),
    "A_ORD3": ("Alphabetical", alpha_names, 42),
    "A_ORD4": ("Rare-middle", rare_middle_names, 42),
    "A_ORD5": ("Random seed 42", freq_desc_names, 42), # passing freq names, ClassifierChain randomly reorders it based on random_state if order='random'. Wait, no, if we pass order as indices, random_state is ignored for ordering.
}

# Fix for random orders:
random_orders = {}
for seed, name in [(42, "A_ORD5"), (0, "A_ORD6"), (123, "A_ORD7")]:
    np.random.seed(seed)
    rand_ord = list(freq_desc_names)
    np.random.shuffle(rand_ord)
    orders[name] = (f"Random seed {seed}", rand_ord, 42)

for exp_id, (desc, ord_names, seed) in orders.items():
    group_b_results[exp_id] = run_ablation(
        exp_id, f"Group B: {desc}",
        C=1.0, class_weight="balanced", chain_order=ord_names, chain_seed=seed
    )

# Find best Order based on VAL
best_ord_id = max(group_b_results, key=lambda k: group_b_results[k]['validation_metrics']['macro_f1'])
best_ord_names = group_b_results[best_ord_id]['config']['hyperparameters']['chain_order']
print(f"\n[GROUP B BEST ORDER: {best_ord_names} (Val F1: {group_b_results[best_ord_id]['validation_metrics']['macro_f1']:.4f})]")

# ==========================================
# Group C: Class Weight
# ==========================================
group_c_results = {}
group_c_results["A_CW1"] = run_ablation(
    "A_CW1", "Group C: class_weight=None, C=1.0",
    C=1.0, class_weight=None, chain_order=freq_desc_names
)
if best_c_val != 1.0:
    group_c_results["A_CW2"] = run_ablation(
        "A_CW2", f"Group C: class_weight=None, C={best_c_val}",
        C=best_c_val, class_weight=None, chain_order=freq_desc_names
    )

print("\nPhase 11 Groups A, B, C Complete.")
