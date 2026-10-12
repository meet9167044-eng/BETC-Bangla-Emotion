"""
scripts/experiments/phase1_debug_expressive.py
===============================================
BETC Improvement Plan — Phase 1: Debug Expressive/Punctuation Features

Performs comprehensive diagnostic checks:
  1. Check for NaNs, Infs, dtypes across splits.
  2. Statistical distribution profiling (min, max, mean, std, percentiles, sparsity).
  3. Row alignment verification with TF-IDF matrices and labels.
  4. Sparse matrix concatenation validation.
  5. Scaler experiment:
     - Raw (current E3): Unscaled counts
     - MaxAbsScaler: Scaled to [0, 1] preserving sparsity
     - Robust / Log1p Scaled: log(1 + count) to compress outliers
     - Selective Expressive: Only ratios and boolean presence (no raw char/word counts)
  6. Re-evaluate on validation set against E0 baseline (0.4625).
"""

import os
import sys
import io
import json
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain

# UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

from src.models.betc import TARGET_COLS, compute_label_order
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds
from src.features.tfidf import BETCFeatureExtractor
from src.features.expressive_features import ExpressiveFeatureExtractor, FEATURE_NAMES

DATA_DIR = os.path.join(WORKSPACE, "Data", "processed")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase1_debug")
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("=" * 70)
    print("BETC Phase 1 — Comprehensive Debugging of Expressive Features")
    print("=" * 70)

    # 1. Load splits
    print("\n[Step 1] Loading preprocessed data splits...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train", "train_preprocessed.csv"), low_memory=False)
    df_val   = pd.read_csv(os.path.join(DATA_DIR, "validation", "validation_preprocessed.csv"), low_memory=False)
    print(f"  Train: {len(df_train):,} rows | Val: {len(df_val):,} rows")

    train_raw = df_train["text"].fillna("").astype(str)
    val_raw   = df_val["text"].fillna("").astype(str)
    train_proc = df_train["processed_text"].fillna("").astype(str)
    val_proc   = df_val["processed_text"].fillna("").astype(str)

    Y_train = df_train[TARGET_COLS].values.astype(int)
    Y_val   = df_val[TARGET_COLS].values.astype(int)

    # 2. Extract expressive features
    print("\n[Step 2] Extracting Expressive Features...")
    expr_extractor = ExpressiveFeatureExtractor()
    X_expr_train = expr_extractor.transform(train_raw)
    X_expr_val   = expr_extractor.transform(val_raw)

    dense_train = X_expr_train.toarray()
    dense_val   = X_expr_val.toarray()

    # 3. Check NaNs, Infs, types
    print("\n[Step 3] Data Integrity Check (NaN / Inf / Dtype):")
    has_nan_tr = np.isnan(dense_train).any()
    has_inf_tr = np.isinf(dense_train).any()
    has_nan_va = np.isnan(dense_val).any()
    has_inf_va = np.isinf(dense_val).any()
    print(f"  Train NaN: {has_nan_tr} | Train Inf: {has_inf_tr} | Dtype: {dense_train.dtype}")
    print(f"  Val   NaN: {has_nan_va} | Val   Inf: {has_inf_va} | Dtype: {dense_val.dtype}")

    # 4. Feature Distribution Profiling
    print("\n[Step 4] Feature Statistics & Distribution Profiling (Train):")
    stats = []
    print(f"{'Feature Name':<28} {'Min':>6} {'Max':>8} {'Mean':>8} {'Std':>8} {'% Zero':>8}")
    print("-" * 72)
    for i, name in enumerate(FEATURE_NAMES):
        col = dense_train[:, i]
        c_min = float(col.min())
        c_max = float(col.max())
        c_mean = float(col.mean())
        c_std = float(col.std())
        pct_zero = float((col == 0).mean() * 100)
        print(f"{name:<28} {c_min:>6.1f} {c_max:>8.1f} {c_mean:>8.2f} {c_std:>8.2f} {pct_zero:>7.1f}%")
        stats.append({
            "feature": name,
            "min": c_min, "max": c_max, "mean": round(c_mean, 4),
            "std": round(c_std, 4), "pct_zero": round(pct_zero, 2)
        })

    # 5. Row alignment verification
    print("\n[Step 5] Verifying Row Alignment:")
    print(f"  Train: df rows = {len(df_train)} | X_expr rows = {X_expr_train.shape[0]} | Y_train rows = {Y_train.shape[0]}")
    print(f"  Val:   df rows = {len(df_val)}   | X_expr rows = {X_expr_val.shape[0]}   | Y_val rows = {Y_val.shape[0]}")
    assert len(df_train) == X_expr_train.shape[0] == Y_train.shape[0], "Row mismatch in train!"
    assert len(df_val) == X_expr_val.shape[0] == Y_val.shape[0], "Row mismatch in val!"
    print("  -> Alignment is 100% verified.")

    # 6. Fit TF-IDF Baseline (E0)
    print("\n[Step 6] Extracting TF-IDF Features...")
    tfidf = BETCFeatureExtractor()
    X_tfidf_tr = tfidf.fit_transform(train_proc)
    X_tfidf_va = tfidf.transform(val_proc)
    print(f"  TF-IDF train shape: {X_tfidf_tr.shape} | max val = {X_tfidf_tr.max():.4f}")

    freq_order, freq_names, _ = compute_label_order(Y_train, TARGET_COLS)

    def evaluate_model(X_tr, X_va, desc):
        print(f"\n--- Testing: {desc} ---")
        t0 = time.time()
        base_clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="saga", random_state=42)
        chain = ClassifierChain(base_clf, order=freq_order, random_state=42)
        chain.fit(X_tr, Y_train)
        P_val = chain.predict_proba(X_va)
        thresholds = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
        Y_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
        m = compute_metrics(Y_val, Y_pred, TARGET_COLS)
        elapsed = time.time() - t0
        print(f"  Val Macro-F1: {m['macro_f1']:.4f} | Micro-F1: {m['micro_f1']:.4f} | Time: {elapsed:.1f}s")
        print("  Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in m["per_class"].items()))
        return m

    # 7. Evaluate Configurations:
    results = {}

    # Config 1: E0 Baseline (TF-IDF only)
    results["E0_Baseline"] = evaluate_model(X_tfidf_tr, X_tfidf_va, "E0 (TF-IDF Only Baseline)")

    # Config 2: Unscaled Expressive (Reproduce E3)
    X_e3_tr = sp.hstack([X_tfidf_tr, X_expr_train], format="csr")
    X_e3_va = sp.hstack([X_tfidf_va, X_expr_val], format="csr")
    results["E3_Unscaled"] = evaluate_model(X_e3_tr, X_e3_va, "E3 (TF-IDF + Unscaled Expressive)")

    # Config 3: MaxAbsScaler Expressive (Scales all expressive features to [0, 1] without destroying sparsity)
    scaler = MaxAbsScaler()
    X_scaled_tr = scaler.fit_transform(X_expr_train)
    X_scaled_va = scaler.transform(X_expr_val)
    X_e3_scaled_tr = sp.hstack([X_tfidf_tr, X_scaled_tr], format="csr")
    X_e3_scaled_va = sp.hstack([X_tfidf_va, X_scaled_va], format="csr")
    results["E3_MaxAbsScaled"] = evaluate_model(X_e3_scaled_tr, X_e3_scaled_va, "E3_MaxAbsScaled (TF-IDF + MaxAbsScaled Expressive)")

    # Config 4: Log1p Transformed Expressive
    log_dense_tr = np.log1p(dense_train)
    log_dense_va = np.log1p(dense_val)
    scaler_log = MaxAbsScaler()
    X_log_tr = scaler_log.fit_transform(sp.csr_matrix(log_dense_tr))
    X_log_va = scaler_log.transform(sp.csr_matrix(log_dense_va))
    X_e3_log_tr = sp.hstack([X_tfidf_tr, X_log_tr], format="csr")
    X_e3_log_va = sp.hstack([X_tfidf_va, X_log_va], format="csr")
    results["E3_Log1pScaled"] = evaluate_model(X_e3_log_tr, X_e3_log_va, "E3_Log1pScaled (TF-IDF + Log1p + MaxAbsScaled)")

    # Config 5: Selective Expressive (Punctuation indicators & ratios only, NO char_count or word_count)
    # Exclude: char_count (idx 6), word_count (idx 7), avg_word_len (idx 8), digit_count (idx 10)
    selective_indices = [0, 1, 2, 3, 4, 5, 9, 11, 12]
    sel_tr = MaxAbsScaler().fit_transform(sp.csr_matrix(dense_train[:, selective_indices]))
    sel_va = MaxAbsScaler().fit_transform(sp.csr_matrix(dense_val[:, selective_indices]))
    X_e3_sel_tr = sp.hstack([X_tfidf_tr, sel_tr], format="csr")
    X_e3_sel_va = sp.hstack([X_tfidf_va, sel_va], format="csr")
    results["E3_SelectiveScaled"] = evaluate_model(X_e3_sel_tr, X_e3_sel_va, "E3_Selective (Punctuation Ratios & Indicators Only)")

    # 8. Summary Comparison
    print("\n" + "=" * 70)
    print("PHASE 1 DEBUGGING SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Configuration':<25} {'Macro-F1':>10} {'Micro-F1':>10} {'Delta vs E0':>12}")
    print("-" * 70)
    e0_f1 = results["E0_Baseline"]["macro_f1"]
    for k, v in results.items():
        delta = v["macro_f1"] - e0_f1
        sign = "+" if delta >= 0 else ""
        print(f"{k:<25} {v['macro_f1']:>10.4f} {v['micro_f1']:>10.4f} {sign}{delta:>11.4f}")
    print("-" * 70)

    # Save findings
    out_file = os.path.join(RESULTS_DIR, "phase1_diagnostic_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feature_statistics": stats,
            "results": {k: {"macro_f1": v["macro_f1"], "micro_f1": v["micro_f1"], "per_class": v["per_class"]} for k, v in results.items()}
        }, f, indent=2)
    print(f"\nReport saved to: {out_file}")

if __name__ == "__main__":
    main()
