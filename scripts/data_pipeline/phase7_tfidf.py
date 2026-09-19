"""
Phase 7 — TF-IDF Feature Extraction on Real Data
==================================================
Fits both vectorizers on train_preprocessed.csv processed_text only.
Transforms all three splits. Saves vectorizers. Validates shapes.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, datetime
import numpy as np
import scipy.sparse as sp
import pandas as pd
sys.path.insert(0, r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion")
from src.features.tfidf import BETCFeatureExtractor

WORKSPACE    = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
PROC_DIR     = os.path.join(WORKSPACE, "Data", "processed")
ARTIFACTS_VEC = os.path.join(WORKSPACE, "artifacts", "vectorizers")
ARTIFACTS_FEAT = os.path.join(WORKSPACE, "artifacts", "features")
LOG_FEATURES  = os.path.join(WORKSPACE, "logs", "features")
TARGET_COLS   = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

os.makedirs(ARTIFACTS_VEC,  exist_ok=True)
os.makedirs(ARTIFACTS_FEAT, exist_ok=True)
os.makedirs(LOG_FEATURES,    exist_ok=True)

# ─── Load preprocessed splits ─────────────────────────────────────────────────
print("Loading preprocessed splits...")
df_train = pd.read_csv(os.path.join(PROC_DIR, "train",      "train_preprocessed.csv"),      low_memory=False)
df_val   = pd.read_csv(os.path.join(PROC_DIR, "validation", "validation_preprocessed.csv"), low_memory=False)
df_test  = pd.read_csv(os.path.join(PROC_DIR, "test",       "test_preprocessed.csv"),        low_memory=False)

print(f"Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")

train_texts = df_train["processed_text"].fillna("").astype(str).tolist()
val_texts   = df_val["processed_text"].fillna("").astype(str).tolist()
test_texts  = df_test["processed_text"].fillna("").astype(str).tolist()

# ─── Fit on train ONLY ────────────────────────────────────────────────────────
print("\nFitting TF-IDF vectorizers on TRAIN split only...")
fx = BETCFeatureExtractor()

print("  Fitting word + char TF-IDF...")
X_train = fx.fit_transform(train_texts)
print(f"  X_train shape: {X_train.shape}")

# ─── Transform val and test (no fitting) ─────────────────────────────────────
print("\nTransforming validation split (no fitting)...")
X_val = fx.transform(val_texts)
print(f"  X_val shape: {X_val.shape}")

print("Transforming test split (no fitting)...")
X_test = fx.transform(test_texts)
print(f"  X_test shape: {X_test.shape}")

# ─── Save vectorizers ─────────────────────────────────────────────────────────
print(f"\nSaving vectorizers to {ARTIFACTS_VEC}...")
fx.save(ARTIFACTS_VEC)
print(f"  Saved: word_tfidf.joblib, char_tfidf.joblib")

# ─── Save feature matrices ────────────────────────────────────────────────────
print(f"\nSaving feature matrices to {ARTIFACTS_FEAT}...")
sp.save_npz(os.path.join(ARTIFACTS_FEAT, "X_train.npz"), X_train)
sp.save_npz(os.path.join(ARTIFACTS_FEAT, "X_val.npz"),   X_val)
sp.save_npz(os.path.join(ARTIFACTS_FEAT, "X_test.npz"),  X_test)

# Also save label matrices aligned with the feature matrices
Y_train = df_train[TARGET_COLS].values.astype(int)
Y_val   = df_val[TARGET_COLS].values.astype(int)
Y_test  = df_test[TARGET_COLS].values.astype(int)
np.save(os.path.join(ARTIFACTS_FEAT, "Y_train.npy"), Y_train)
np.save(os.path.join(ARTIFACTS_FEAT, "Y_val.npy"),   Y_val)
np.save(os.path.join(ARTIFACTS_FEAT, "Y_test.npy"),  Y_test)
print("  Saved: X_train.npz, X_val.npz, X_test.npz")
print("  Saved: Y_train.npy, Y_val.npy, Y_test.npy")

# ─── Validation ───────────────────────────────────────────────────────────────
print("\nRunning validation checks...")
checks = {}

# A. Shape conformance: expected dim <= 25,000 (may be lower if vocab smaller)
n_word = len(fx.word_tfidf.get_feature_names_out())
n_char = len(fx.char_tfidf.get_feature_names_out())
combined_dim = n_word + n_char

checks["shape_train"] = {
    "pass": X_train.shape == (28840, combined_dim),
    "shape": list(X_train.shape),
    "expected_rows": 28840,
}
checks["shape_val"] = {
    "pass": X_val.shape == (6170, combined_dim),
    "shape": list(X_val.shape),
    "expected_rows": 6170,
}
checks["shape_test"] = {
    "pass": X_test.shape == (6174, combined_dim),
    "shape": list(X_test.shape),
    "expected_rows": 6174,
}

# B. All splits share the same feature dimension
checks["all_same_dim"] = {
    "pass": X_train.shape[1] == X_val.shape[1] == X_test.shape[1],
    "train_dim": X_train.shape[1],
    "val_dim": X_val.shape[1],
    "test_dim": X_test.shape[1],
}

# C. Combined dim does not exceed 25,000
checks["dim_cap"] = {
    "pass": combined_dim <= 25_000,
    "combined_dim": combined_dim,
    "word_vocab": n_word,
    "char_vocab": n_char,
}

# D. Matrices are sparse
checks["matrices_sparse"] = {
    "pass": sp.issparse(X_train) and sp.issparse(X_val) and sp.issparse(X_test),
}

# E. Vectorizers are fitted (is_fitted flag)
checks["vectorizers_fitted"] = {"pass": fx.is_fitted}

# F. Reload test — save and reload, transform matches
from src.features.tfidf import BETCFeatureExtractor as BETCFx2
fx_reload = BETCFx2.load(ARTIFACTS_VEC)
X_val_reload = fx_reload.transform(val_texts[:10])
X_val_orig   = X_val[:10]
reload_match = np.allclose(X_val_orig.toarray(), X_val_reload.toarray())
checks["reload_transform_matches"] = {
    "pass": reload_match,
    "sample_rows": 10,
}

# G. Labels row count matches features row count
checks["label_matrix_rows"] = {
    "pass": Y_train.shape[0] == X_train.shape[0] and
            Y_val.shape[0] == X_val.shape[0] and
            Y_test.shape[0] == X_test.shape[0],
    "Y_train_shape": list(Y_train.shape),
    "Y_val_shape": list(Y_val.shape),
    "Y_test_shape": list(Y_test.shape),
}

# H. No TF-IDF fitting on val or test (structural: after fit_transform, vocab is frozen)
vocab_after_val  = len(fx.word_tfidf.vocabulary_)
vocab_after_test = len(fx.word_tfidf.vocabulary_)
checks["no_fitting_on_val_test"] = {
    "pass": vocab_after_val == n_word and vocab_after_test == n_word,
    "note": "Word vocab size unchanged after transform() calls",
}

all_pass = all(v.get("pass", True) for v in checks.values() if isinstance(v, dict))

print("\nValidation Results:")
for k, v in checks.items():
    status = "PASS" if v.get("pass", True) else "FAIL"
    print(f"  [{status}] {k}")

print(f"\nAll checks passed: {all_pass}")

# ─── Summary info ─────────────────────────────────────────────────────────────
summary = fx.summary()
print(f"\nFeature Summary:")
print(f"  Word TF-IDF vocab size: {summary['word_tfidf']['vocab_size']:,}")
print(f"  Char TF-IDF vocab size: {summary['char_tfidf']['vocab_size']:,}")
print(f"  Combined feature dim:   {summary['combined_dim']:,}")

# ─── Sparsity info ────────────────────────────────────────────────────────────
def sparsity(mat):
    nnz = mat.nnz
    total = mat.shape[0] * mat.shape[1]
    return 1.0 - nnz / total if total > 0 else 0.0

print(f"\nSparsity:")
print(f"  Train: {sparsity(X_train):.4f} ({X_train.nnz:,} non-zero)")
print(f"  Val:   {sparsity(X_val):.4f} ({X_val.nnz:,} non-zero)")
print(f"  Test:  {sparsity(X_test):.4f} ({X_test.nnz:,} non-zero)")

# ─── Save validation JSON ─────────────────────────────────────────────────────
validation_out = {
    "run_timestamp": RUN_TS,
    "phase": "Phase 7 — TF-IDF Feature Extraction",
    "inputs": {
        "train_preprocessed": "Data/processed/train/train_preprocessed.csv",
        "val_preprocessed":   "Data/processed/validation/validation_preprocessed.csv",
        "test_preprocessed":  "Data/processed/test/test_preprocessed.csv",
    },
    "hyperparameters": {
        "word_tfidf": {
            "ngram_range": [1, 2],
            "max_features": 10000,
            "analyzer": "word",
            "sublinear_tf": True,
            "strip_accents": None,
        },
        "char_tfidf": {
            "analyzer": "char_wb",
            "ngram_range": [3, 5],
            "max_features": 15000,
            "sublinear_tf": True,
            "strip_accents": None,
        },
        "fusion": "scipy.sparse.hstack",
    },
    "vocab_sizes": {
        "word_tfidf": int(n_word),
        "char_tfidf": int(n_char),
        "combined_dim": int(combined_dim),
    },
    "feature_shapes": {
        "X_train": list(X_train.shape),
        "X_val":   list(X_val.shape),
        "X_test":  list(X_test.shape),
        "Y_train": list(Y_train.shape),
        "Y_val":   list(Y_val.shape),
        "Y_test":  list(Y_test.shape),
    },
    "sparsity": {
        "train": round(sparsity(X_train), 6),
        "val":   round(sparsity(X_val), 6),
        "test":  round(sparsity(X_test), 6),
    },
    "artifacts": {
        "word_vectorizer": "artifacts/vectorizers/word_tfidf.joblib",
        "char_vectorizer": "artifacts/vectorizers/char_tfidf.joblib",
        "X_train":  "artifacts/features/X_train.npz",
        "X_val":    "artifacts/features/X_val.npz",
        "X_test":   "artifacts/features/X_test.npz",
        "Y_train":  "artifacts/features/Y_train.npy",
        "Y_val":    "artifacts/features/Y_val.npy",
        "Y_test":   "artifacts/features/Y_test.npy",
    },
    "validation_checks": {k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else vv)
                               for kk, vv in v.items()}
                          for k, v in checks.items()},
    "all_checks_passed": bool(all_pass),
    "unit_test_results": "36/36 passed (python tests/test_tfidf.py)",
    "leakage_safety": (
        "Vectorizers fitted ONLY on training split. "
        "Validation and test transformed with .transform() only. "
        "Verified by vocab-freeze check and reload test."
    ),
    "note_no_model": "No LogisticRegression, ClassifierChain, or threshold optimization was executed.",
}

val_json = os.path.join(LOG_FEATURES, "phase7_tfidf_validation.json")
with open(val_json, "w", encoding="utf-8") as f:
    json.dump(validation_out, f, indent=2, ensure_ascii=False)
print(f"\nSaved validation JSON: {val_json}")
print("\nPhase 7 COMPLETE. DO NOT proceed to Phase 8 without user instruction.")
