"""
scripts/experiments/phase14_feature_ablation.py
=================================================
BETC — Phase 14 — Feature Engineering Ablation Study (E0–E9)

Runs controlled ablation experiments to evaluate the impact of each new
feature group on validation Macro-F1.

EXPERIMENT DESIGN:
  E0: Existing word + char TF-IDF baseline (25,000 features)
  E1: E0 + numeric emoji features
  E2: E0 + sentiment features
  E3: E0 + expressive text features
  E4: E0 + negation/intensifier features
  E5: E0 + emoji-to-text augmentation (preprocessed text with emoji descriptions)
  E6: E0 + emoji + sentiment
  E7: E0 + emoji + sentiment + expressive
  E8: E0 + emoji + sentiment + negation
  E9: E0 + all feature groups (full combination)

EVALUATION PROTOCOL:
  - All feature extraction fitted on TRAINING data only.
  - Threshold optimization on VALIDATION data only.
  - Test set is NOT evaluated during feature selection (validation only).
  - Model: ClassifierChain(LogisticRegression, C=1.0, class_weight='balanced')
    with frequency-descending chain order (same as M1 reference).
  - Metrics: Validation Macro-F1, Micro-F1, per-class F1.
  - E0 is the reference; all experiments report delta vs. E0.

USAGE:
  python scripts/experiments/phase14_feature_ablation.py
  python scripts/experiments/phase14_feature_ablation.py --exp_ids E0 E1 E2

OUTPUTS:
  - results/ablations/phase14/{exp_id}.json  — per-experiment result
  - results/ablations/phase14/summary.json   — comparison table
  - logs/features/phase14_ablations.log      — experiment log
"""

import sys
import io
import os
import json
import time
import warnings
import argparse
import datetime
import traceback

# UTF-8 output for Bangla text (with line_buffering for real-time progress)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
warnings.filterwarnings("ignore")

# Resolve workspace root
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

# ─── Paths ───────────────────────────────────────────────────────────────────
PROC_DIR     = os.path.join(WORKSPACE, "Data", "processed")
RESULTS_DIR  = os.path.join(WORKSPACE, "results", "ablations", "phase14")
LOG_DIR      = os.path.join(WORKSPACE, "logs", "features")
FEAT_DIR     = os.path.join(WORKSPACE, "artifacts", "features")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ─── Load data splits ────────────────────────────────────────────────────────
print("Loading preprocessed data splits...")
df_train = pd.read_csv(
    os.path.join(PROC_DIR, "train", "train_preprocessed.csv"), low_memory=False
)
df_val = pd.read_csv(
    os.path.join(PROC_DIR, "validation", "validation_preprocessed.csv"), low_memory=False
)
# NOTE: df_test is loaded but NOT used for feature selection.
# Test evaluation happens only in the final evaluation phase.
df_test = pd.read_csv(
    os.path.join(PROC_DIR, "test", "test_preprocessed.csv"), low_memory=False
)

print(f"  Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

# Text columns
train_raw  = df_train["text"].fillna("").astype(str)
train_proc = df_train["processed_text"].fillna("").astype(str)
val_raw    = df_val["text"].fillna("").astype(str)
val_proc   = df_val["processed_text"].fillna("").astype(str)
test_raw   = df_test["text"].fillna("").astype(str)
test_proc  = df_test["processed_text"].fillna("").astype(str)

# Label matrices
Y_train = df_train[TARGET_COLS].values.astype(int)
Y_val   = df_val[TARGET_COLS].values.astype(int)
Y_test  = df_test[TARGET_COLS].values.astype(int)

# Pre-compute chain order (descending frequency, same as M1)
freq_order, freq_names, _ = compute_label_order(Y_train, TARGET_COLS)


def run_experiment(
    exp_id: str,
    description: str,
    use_tfidf: bool = True,
    use_emoji: bool = False,
    use_sentiment: bool = False,
    use_expressive: bool = False,
    use_negation: bool = False,
    C: float = 1.0,
    max_iter: int = 1000,
) -> dict:
    """
    Run a single ablation experiment. Returns result dict.
    Validates on val set only; test metrics computed but NOT used for selection.
    """
    print(f"\n{'='*70}")
    print(f"Experiment {exp_id}: {description}")
    groups = [g for g, v in [("tfidf", use_tfidf), ("emoji", use_emoji),
                              ("sentiment", use_sentiment), ("expressive", use_expressive),
                              ("negation", use_negation)] if v]
    print(f"  Feature groups: {groups}")
    print(f"{'='*70}")

    t_start = time.time()
    error_msg = None

    try:
        # ── 1. Feature extraction ──────────────────────────────────────────────
        t0 = time.time()
        extractor = BETCCombinedExtractor(
            use_tfidf=use_tfidf,
            use_emoji=use_emoji,
            use_sentiment=use_sentiment,
            use_expressive=use_expressive,
            use_negation=use_negation,
        )
        X_train_feat = extractor.fit_transform(train_raw, train_proc)
        X_val_feat   = extractor.transform(val_raw, val_proc)
        X_test_feat  = extractor.transform(test_raw, test_proc)
        t_feat = time.time() - t0

        feat_dim = X_train_feat.shape[1]
        print(f"  Feature dim: {feat_dim:,} | Extraction time: {t_feat:.1f}s")

        # ── 2. Train model ─────────────────────────────────────────────────────
        t1 = time.time()
        base_clf = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            solver="saga",
            random_state=42,
        )
        chain = ClassifierChain(base_clf, order=freq_order, random_state=42)
        chain.fit(X_train_feat, Y_train)
        t_fit = time.time() - t1
        print(f"  Fit time: {t_fit:.1f}s")

        # ── 3. Val threshold optimization ──────────────────────────────────────
        t2 = time.time()
        P_val = chain.predict_proba(X_val_feat)
        thresholds = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
        Y_val_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
        val_metrics = compute_metrics(Y_val, Y_val_pred, TARGET_COLS)
        t_val = time.time() - t2

        print(f"  Val Macro-F1: {val_metrics['macro_f1']:.4f}  "
              f"Val Micro-F1: {val_metrics['micro_f1']:.4f}")
        print(f"  Per-class F1: " +
              ", ".join(f"{k}={v['f1']:.3f}" for k, v in val_metrics["per_class"].items()))

        # ── 4. Test metrics (computed but NOT used for selection) ──────────────
        t3 = time.time()
        P_test = chain.predict_proba(X_test_feat)
        Y_test_pred = apply_thresholds(P_test, thresholds, TARGET_COLS)
        test_metrics = compute_metrics(Y_test, Y_test_pred, TARGET_COLS)
        t_test = time.time() - t3
        print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f} "
              f"[NOTE: not used for feature selection]")

        total_time = time.time() - t_start
        ext_summary = extractor.summary()

        result = {
            "experiment_id": exp_id,
            "phase": "phase14_feature_ablation",
            "category": "feature_engineering_ablation",
            "description": description,
            "run_timestamp": RUN_TS,
            "feature_groups_enabled": groups,
            "hyperparameters": {
                "C": C,
                "class_weight": "balanced",
                "max_iter": max_iter,
                "solver": "saga",
                "chain_order": freq_names,
                "chain_seed": 42,
            },
            "feature_dimensions": ext_summary.get("feature_dims", {}),
            "total_feature_dim": feat_dim,
            "thresholds": thresholds,
            "validation_metrics": val_metrics,
            "test_metrics": {
                **test_metrics,
                "_note": (
                    "Test metrics recorded for reference only. "
                    "Feature selection decisions are based solely on validation metrics."
                ),
            },
            "timing_seconds": {
                "feature_extraction": round(t_feat, 2),
                "model_fit": round(t_fit, 2),
                "val_threshold_opt": round(t_val, 2),
                "test_eval": round(t_test, 2),
                "total": round(total_time, 2),
            },
            "leakage_declaration": (
                "Feature extractors fitted on training split only. "
                "Thresholds optimized on validation probabilities only. "
                "Feature selection based on validation Macro-F1 only."
            ),
            "error": None,
        }

    except Exception as exc:
        total_time = time.time() - t_start
        error_msg = traceback.format_exc()
        print(f"  ERROR in {exp_id}: {exc}")
        print(error_msg)
        result = {
            "experiment_id": exp_id,
            "phase": "phase14_feature_ablation",
            "description": description,
            "run_timestamp": RUN_TS,
            "error": error_msg,
            "validation_metrics": None,
            "test_metrics": None,
            "timing_seconds": {"total": round(total_time, 2)},
        }

    # Save result
    out_path = os.path.join(RESULTS_DIR, f"{exp_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="BETC Phase 14 Feature Ablation Experiments"
    )
    parser.add_argument(
        "--exp_ids",
        nargs="+",
        default=list(EXPERIMENT_CONFIGS.keys()) + ["E9"],
        help="Experiment IDs to run (default: all E0–E9)"
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="LogisticRegression C parameter (default: 1.0)"
    )
    parser.add_argument(
        "--max_iter", type=int, default=1000,
        help="Max iterations for LogisticRegression (default: 1000)"
    )
    return parser.parse_args()


EXPERIMENT_DESCRIPTIONS = {
    "E0": "Baseline: word + char TF-IDF (25,000 features)",
    "E1": "E0 + numeric emoji features",
    "E2": "E0 + Bangla sentiment lexicon features",
    "E3": "E0 + expressive/punctuation features",
    "E4": "E0 + negation/intensifier/diminisher features",
    "E5": "E0 + emoji-to-text augmentation (emoji descriptions appended to text)",
    "E6": "E0 + emoji + sentiment",
    "E7": "E0 + emoji + sentiment + expressive",
    "E8": "E0 + emoji + sentiment + negation",
    "E9": "E0 + emoji + sentiment + expressive + negation (all groups)",
    "E9_best": "E0 + all feature groups (alias for E9)",
}


def main():
    args = parse_args()

    print("=" * 70)
    print(f"BETC Phase 14 — Feature Ablation Study")
    print(f"Run timestamp: {RUN_TS}")
    print(f"Experiments to run: {args.exp_ids}")
    print("=" * 70)

    results = {}
    e0_macro_f1 = None

    for exp_id in args.exp_ids:
        # Skip E5 — emoji-to-text requires text augmentation at preprocessing level
        if exp_id == "E5":
            print(f"\nSkipping {exp_id}: emoji-to-text augmentation requires "
                  "preprocessing-level text modification. Run separately.")
            continue

        # Get config
        config_key = exp_id if exp_id in EXPERIMENT_CONFIGS else "E9_best"
        if exp_id == "E9":
            config = {**EXPERIMENT_CONFIGS["E9_best"]}
        elif exp_id in EXPERIMENT_CONFIGS:
            config = {**EXPERIMENT_CONFIGS[exp_id]}
        else:
            print(f"  Unknown experiment ID: {exp_id}, skipping.")
            continue

        desc = EXPERIMENT_DESCRIPTIONS.get(exp_id, exp_id)
        result = run_experiment(
            exp_id=exp_id,
            description=desc,
            C=args.C,
            max_iter=args.max_iter,
            **config,
        )
        results[exp_id] = result

        if exp_id == "E0" and result.get("validation_metrics"):
            e0_macro_f1 = result["validation_metrics"]["macro_f1"]

    # If E0 was not in the current run, try to load it from disk
    if e0_macro_f1 is None:
        e0_file = os.path.join(RESULTS_DIR, "E0.json")
        if os.path.exists(e0_file):
            try:
                with open(e0_file, "r", encoding="utf-8") as f:
                    e0_data = json.load(f)
                if e0_data.get("validation_metrics"):
                    e0_macro_f1 = e0_data["validation_metrics"]["macro_f1"]
            except Exception:
                pass

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 14 ABLATION SUMMARY (Validation Metrics)")
    print("=" * 70)
    print(f"{'ExpID':<8} {'Macro-F1':>10} {'Micro-F1':>10} {'ΔE0':>8} "
          f"{'Disgust':>9} {'Surprise':>9} {'Fear':>9} {'Dim':>8}")
    print("-" * 70)

    summary_rows = []
    for exp_id, result in results.items():
        if result.get("validation_metrics") is None:
            print(f"{exp_id:<8} {'ERROR':>10}")
            continue
        vm = result["validation_metrics"]
        macro = vm["macro_f1"]
        micro = vm["micro_f1"]
        delta = (macro - e0_macro_f1) if e0_macro_f1 is not None else float("nan")
        pc = vm["per_class"]
        disgust  = pc.get("disgust",  {}).get("f1", float("nan"))
        surprise = pc.get("surprise", {}).get("f1", float("nan"))
        fear     = pc.get("fear",     {}).get("f1", float("nan"))
        dim      = result.get("total_feature_dim", 0)

        sign = "+" if delta >= 0 else ""
        print(f"{exp_id:<8} {macro:>10.4f} {micro:>10.4f} {sign}{delta:>7.4f} "
              f"{disgust:>9.4f} {surprise:>9.4f} {fear:>9.4f} {dim:>8,}")

        summary_rows.append({
            "experiment_id": exp_id,
            "description": result.get("description", ""),
            "val_macro_f1": macro,
            "val_micro_f1": micro,
            "delta_vs_e0": round(delta, 4) if e0_macro_f1 is not None else None,
            "disgust_f1": disgust,
            "surprise_f1": surprise,
            "fear_f1": fear,
            "total_feature_dim": dim,
            "error": result.get("error"),
        })

    print("-" * 70)

    # Save summary
    summary = {
        "phase": "phase14_feature_ablation",
        "run_timestamp": RUN_TS,
        "e0_baseline_val_macro_f1": e0_macro_f1,
        "evaluation_note": (
            "Feature selection based on validation Macro-F1 only. "
            "Test metrics in individual result files are recorded for reference. "
            "Do not re-evaluate test set to select features."
        ),
        "results": summary_rows,
    }
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")
    print("\nPhase 14 feature ablation complete.")
    print("Next step: inspect summary.json and identify best feature combination.")
    print("Do NOT use test metrics to select features.")


if __name__ == "__main__":
    main()
