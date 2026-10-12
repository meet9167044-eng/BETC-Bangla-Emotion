"""
scripts/experiments/phase14_colab_runner.py
============================================
BETC Phase 14 — Google Colab Runner Script

This script is designed to run in Google Colab where the full dataset
and compute resources are available. It:
  1. Verifies the environment setup
  2. Runs all feature ablation experiments E0–E9
  3. Saves results to results/ablations/phase14/

COLAB USAGE:
  1. Upload the BETC repo zip or mount Google Drive
  2. !pip install bnunicodenormalizer emoji scipy scikit-learn pandas numpy
  3. import sys; sys.path.insert(0, '/path/to/BETC-Bangla-Emotion')
  4. exec(open('scripts/experiments/phase14_colab_runner.py').read())

OR simply run:
  !python scripts/experiments/phase14_feature_ablation.py
"""

import sys
import io
import os
import json
import time
import warnings
import datetime
import traceback

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

import numpy as np
import scipy.sparse as sp
import pandas as pd

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

from src.models.betc import TARGET_COLS, compute_label_order
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds
from src.features.feature_combination import BETCCombinedExtractor, EXPERIMENT_CONFIGS

PROC_DIR    = os.path.join(WORKSPACE, "Data", "processed")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase14")
os.makedirs(RESULTS_DIR, exist_ok=True)
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ── Environment check ─────────────────────────────────────────────────────────
print("=" * 60)
print("BETC Phase 14 — Environment Check")
print("=" * 60)
try:
    import emoji
    print(f"  emoji: {emoji.__version__} OK")
except ImportError:
    print("  emoji: NOT INSTALLED — run: pip install emoji")

try:
    from bnunicodenormalizer import Normalizer
    print("  bnunicodenormalizer: OK")
except ImportError:
    print("  bnunicodenormalizer: NOT INSTALLED")

print(f"  numpy: {np.__version__}")
print(f"  scipy: {sp.__version__}")
import sklearn
print(f"  scikit-learn: {sklearn.__version__}")
print(f"  pandas: {pd.__version__}")
print()

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading preprocessed data splits...")
df_train = pd.read_csv(
    os.path.join(PROC_DIR, "train", "train_preprocessed.csv"), low_memory=False
)
df_val = pd.read_csv(
    os.path.join(PROC_DIR, "validation", "validation_preprocessed.csv"), low_memory=False
)
df_test = pd.read_csv(
    os.path.join(PROC_DIR, "test", "test_preprocessed.csv"), low_memory=False
)
print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

train_raw  = df_train["text"].fillna("").astype(str)
train_proc = df_train["processed_text"].fillna("").astype(str)
val_raw    = df_val["text"].fillna("").astype(str)
val_proc   = df_val["processed_text"].fillna("").astype(str)
test_raw   = df_test["text"].fillna("").astype(str)
test_proc  = df_test["processed_text"].fillna("").astype(str)

Y_train = df_train[TARGET_COLS].values.astype(int)
Y_val   = df_val[TARGET_COLS].values.astype(int)
Y_test  = df_test[TARGET_COLS].values.astype(int)

freq_order, freq_names, _ = compute_label_order(Y_train, TARGET_COLS)
print(f"  Chain order: {freq_names}")

# ── Experiment runner ────────────────────────────────────────────────────────

def run_experiment(exp_id, description, C=1.0, max_iter=2000, **feature_flags):
    print(f"\n{'='*60}\n{exp_id}: {description}\n{'='*60}")
    t_start = time.time()

    try:
        # Feature extraction
        t0 = time.time()
        extractor = BETCCombinedExtractor(**feature_flags)
        X_tr = extractor.fit_transform(train_raw, train_proc)
        X_v  = extractor.transform(val_raw, val_proc)
        X_te = extractor.transform(test_raw, test_proc)
        t_feat = time.time() - t0
        print(f"  Feature dim: {X_tr.shape[1]:,} | Extraction: {t_feat:.1f}s")

        # Train
        t1 = time.time()
        base_clf = LogisticRegression(C=C, class_weight="balanced",
                                      max_iter=max_iter, solver="saga", random_state=42)
        chain = ClassifierChain(base_clf, order=freq_order, random_state=42)
        chain.fit(X_tr, Y_train)
        t_fit = time.time() - t1
        print(f"  Model fit: {t_fit:.1f}s")

        # Val thresholds (feature selection)
        P_val = chain.predict_proba(X_v)
        thresholds = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
        Y_val_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
        val_metrics = compute_metrics(Y_val, Y_val_pred, TARGET_COLS)

        # Test (recorded but NOT used for selection)
        P_test = chain.predict_proba(X_te)
        Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)
        test_metrics = compute_metrics(Y_test, Y_test_pred, TARGET_COLS)

        print(f"  Val  Macro-F1: {val_metrics['macro_f1']:.4f}  Micro-F1: {val_metrics['micro_f1']:.4f}")
        print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f} [NOT used for selection]")
        print(f"  Per-class Val F1: " +
              ", ".join(f"{k}={v['f1']:.3f}" for k, v in val_metrics["per_class"].items()))

        result = {
            "experiment_id": exp_id,
            "phase": "phase14_feature_ablation",
            "description": description,
            "run_timestamp": RUN_TS,
            "feature_flags": feature_flags,
            "total_feature_dim": X_tr.shape[1],
            "feature_dims": extractor.summary().get("feature_dims", {}),
            "thresholds": thresholds,
            "validation_metrics": val_metrics,
            "test_metrics": {**test_metrics,
                             "_note": "Recorded for reference only. Not used for feature selection."},
            "timing_seconds": {
                "feature_extraction": round(t_feat, 2),
                "model_fit": round(t_fit, 2),
                "total": round(time.time() - t_start, 2),
            },
            "leakage_declaration": (
                "Feature extractors fitted on training split only. "
                "Thresholds optimized on validation probabilities only."
            ),
            "error": None,
        }
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"  ERROR: {exc}\n{tb}")
        result = {
            "experiment_id": exp_id,
            "description": description,
            "run_timestamp": RUN_TS,
            "error": tb,
            "validation_metrics": None,
            "test_metrics": None,
            "timing_seconds": {"total": round(time.time() - t_start, 2)},
        }

    out_path = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")
    return result


# ── Run all experiments ───────────────────────────────────────────────────────
experiments = [
    ("E0",  "Baseline: word+char TF-IDF (25,000 features)",
     dict(use_tfidf=True, use_emoji=False, use_sentiment=False,
          use_expressive=False, use_negation=False)),
    ("E1",  "E0 + numeric emoji features",
     dict(use_tfidf=True, use_emoji=True, use_sentiment=False,
          use_expressive=False, use_negation=False)),
    ("E2",  "E0 + Bangla sentiment lexicon features",
     dict(use_tfidf=True, use_emoji=False, use_sentiment=True,
          use_expressive=False, use_negation=False)),
    ("E3",  "E0 + expressive/punctuation features",
     dict(use_tfidf=True, use_emoji=False, use_sentiment=False,
          use_expressive=True, use_negation=False)),
    ("E4",  "E0 + negation/intensifier/diminisher features",
     dict(use_tfidf=True, use_emoji=False, use_sentiment=False,
          use_expressive=False, use_negation=True)),
    ("E6",  "E0 + emoji + sentiment",
     dict(use_tfidf=True, use_emoji=True, use_sentiment=True,
          use_expressive=False, use_negation=False)),
    ("E7",  "E0 + emoji + sentiment + expressive",
     dict(use_tfidf=True, use_emoji=True, use_sentiment=True,
          use_expressive=True, use_negation=False)),
    ("E8",  "E0 + emoji + sentiment + negation",
     dict(use_tfidf=True, use_emoji=True, use_sentiment=True,
          use_expressive=False, use_negation=True)),
    ("E9",  "E0 + all feature groups (full combination)",
     dict(use_tfidf=True, use_emoji=True, use_sentiment=True,
          use_expressive=True, use_negation=True)),
]

all_results = {}
for exp_id, desc, flags in experiments:
    all_results[exp_id] = run_experiment(exp_id, desc, C=1.0, max_iter=2000, **flags)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 14 ABLATION SUMMARY (Validation Metrics — used for selection)")
print("=" * 70)
e0_f1 = all_results.get("E0", {}).get("validation_metrics", {})
e0_f1 = e0_f1.get("macro_f1") if e0_f1 else None

print(f"{'ExpID':<8} {'Macro-F1':>10} {'Micro-F1':>10} {'ΔE0':>8} "
      f"{'Disgust':>9} {'Surprise':>9} {'Fear':>8} {'Dim':>8}")
print("-" * 70)

summary_rows = []
for exp_id, (_, desc, _) in zip([e[0] for e in experiments], experiments):
    r = all_results.get(exp_id)
    if not r or not r.get("validation_metrics"):
        print(f"{exp_id:<8} ERROR")
        continue
    vm = r["validation_metrics"]
    m = vm["macro_f1"]
    mi = vm["micro_f1"]
    delta = (m - e0_f1) if e0_f1 is not None else float("nan")
    pc = vm["per_class"]
    d = pc.get("disgust", {}).get("f1", float("nan"))
    s = pc.get("surprise", {}).get("f1", float("nan"))
    f = pc.get("fear",     {}).get("f1", float("nan"))
    dim = r.get("total_feature_dim", 0)
    sign = "+" if delta >= 0 else ""
    print(f"{exp_id:<8} {m:>10.4f} {mi:>10.4f} {sign}{delta:>7.4f} "
          f"{d:>9.4f} {s:>9.4f} {f:>8.4f} {dim:>8,}")
    summary_rows.append({
        "experiment_id": exp_id, "description": desc,
        "val_macro_f1": m, "val_micro_f1": mi,
        "delta_vs_e0": round(delta, 4) if e0_f1 is not None else None,
        "disgust_f1": d, "surprise_f1": s, "fear_f1": f,
        "total_feature_dim": dim,
    })

print("-" * 70)
summary_path = os.path.join(RESULTS_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump({"phase": "phase14_feature_ablation", "run_timestamp": RUN_TS,
               "e0_baseline_val_macro_f1": e0_f1, "results": summary_rows},
              f, indent=2, ensure_ascii=False)
print(f"\nSummary saved: {summary_path}")
print("\nDone. Review summary.json. Select best combination based on VAL Macro-F1.")
print("DO NOT select features based on test metrics.")
