"""
Phase 5b — Reproducible Splitting
=================================
Input:  Data/interim/harmonized_deduplicated.csv  (41,184 rows)
Output: Data/processed/train/train.csv
        Data/processed/validation/validation.csv
        Data/processed/test/test.csv
        artifacts/splits/split_v1.json

Rules:
- Iterative multi-label-stratified split at 70/15/15.
- Save row indices/IDs per split.
- Save per-split label prevalence.
- Validation checks: No row ID appears in more than one split; every
  target label has a nonzero positive count in every split.
- Do NOT overwrite a split silently once created.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, datetime, hashlib
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

WORKSPACE = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
INTERIM_CSV = os.path.join(WORKSPACE, "Data", "interim", "harmonized_deduplicated.csv")
PROCESSED_DIR = os.path.join(WORKSPACE, "Data", "processed")
ARTIFACTS_DIR = os.path.join(WORKSPACE, "artifacts", "splits")
TARGET_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

os.makedirs(os.path.join(PROCESSED_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "validation"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "test"), exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_sample_id(row):
    """Generate a stable unique ID based on dataset, row id, and text hash"""
    text_hash = hashlib.sha1(str(row["text"]).encode("utf-8")).hexdigest()[:8]
    ds = str(row["source_dataset"])[:3].upper()
    rid = str(row["source_row_id"]) if pd.notna(row["source_row_id"]) else "NA"
    return f"{ds}-{rid}-{text_hash}"

print("Loading deduplicated corpus...")
df = pd.read_csv(INTERIM_CSV, low_memory=False)

if "sample_id" not in df.columns:
    df["sample_id"] = df.apply(generate_sample_id, axis=1)

# Ensure sample_id is unique; if not, append index
if df["sample_id"].duplicated().any():
    print("Warning: some sample_ids are duplicated. Appending index.")
    df["sample_id"] = df["sample_id"] + "-" + df.index.astype(str)

print(f"Total rows loaded: {len(df):,}")

# Extract features (indices for tracking) and labels for stratification
X = np.expand_dims(df.index.values, axis=1)
y = df[TARGET_COLS].values.astype(int)

print("\nPerforming iterative multi-label-stratified split (70/15/15)...")
# Split 1: 70% Train, 30% Temp (Val + Test)
X_train_idx, y_train, X_temp_idx, y_temp = iterative_train_test_split(X, y, test_size=0.3)

# Split 2: 50% of Temp -> 15% Val, 15% Test
# The test_size here is relative to the temp array size (which is 30% of total)
X_val_idx, y_val, X_test_idx, y_test = iterative_train_test_split(X_temp_idx, y_temp, test_size=0.5)

train_indices = X_train_idx.flatten()
val_indices = X_val_idx.flatten()
test_indices = X_test_idx.flatten()

df_train = df.iloc[train_indices].copy()
df_val = df.iloc[val_indices].copy()
df_test = df.iloc[test_indices].copy()

n_total = len(df)
n_train = len(df_train)
n_val = len(df_val)
n_test = len(df_test)

print(f"\nSplit sizes:")
print(f"  Train:      {n_train:,} ({n_train/n_total*100:.1f}%)")
print(f"  Validation: {n_val:,} ({n_val/n_total*100:.1f}%)")
print(f"  Test:       {n_test:,} ({n_test/n_total*100:.1f}%)")

# ─── Validation Checks ────────────────────────────────────────────────────────
checks = {}

# 1. No overlap
train_ids = set(df_train["sample_id"])
val_ids = set(df_val["sample_id"])
test_ids = set(df_test["sample_id"])

overlap_tv = len(train_ids.intersection(val_ids))
overlap_tt = len(train_ids.intersection(test_ids))
overlap_vt = len(val_ids.intersection(test_ids))
total_overlap = overlap_tv + overlap_tt + overlap_vt

checks["no_row_overlap"] = {
    "pass": total_overlap == 0,
    "overlap_train_val": overlap_tv,
    "overlap_train_test": overlap_tt,
    "overlap_val_test": overlap_vt
}

# 2. Complete rows accounted for
checks["all_rows_accounted"] = {
    "pass": (n_train + n_val + n_test) == n_total,
    "total_split": n_train + n_val + n_test,
    "total_original": n_total
}

# 3. Label positivity in every split
def get_label_stats(split_df):
    stats = {}
    for col in TARGET_COLS:
        pos = int(split_df[col].sum())
        stats[col] = {
            "positive_count": pos,
            "prevalence_pct": round(pos / len(split_df) * 100, 2)
        }
    return stats

train_stats = get_label_stats(df_train)
val_stats = get_label_stats(df_val)
test_stats = get_label_stats(df_test)

nonzero_train = all(train_stats[c]["positive_count"] > 0 for c in TARGET_COLS)
nonzero_val = all(val_stats[c]["positive_count"] > 0 for c in TARGET_COLS)
nonzero_test = all(test_stats[c]["positive_count"] > 0 for c in TARGET_COLS)

checks["nonzero_labels_in_all_splits"] = {
    "pass": nonzero_train and nonzero_val and nonzero_test
}

all_checks_passed = all(v.get("pass", False) for v in checks.values())
print("\nValidation Checks:")
for k, v in checks.items():
    print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}")

if not all_checks_passed:
    print("\nCRITICAL FAILURE: Split validation failed. Aborting save.")
    sys.exit(1)

# ─── Save Outputs ─────────────────────────────────────────────────────────────

# Data files
train_csv = os.path.join(PROCESSED_DIR, "train", "train.csv")
val_csv = os.path.join(PROCESSED_DIR, "validation", "validation.csv")
test_csv = os.path.join(PROCESSED_DIR, "test", "test.csv")

df_train.to_csv(train_csv, index=False, encoding="utf-8-sig")
df_val.to_csv(val_csv, index=False, encoding="utf-8-sig")
df_test.to_csv(test_csv, index=False, encoding="utf-8-sig")

# Artifact JSON
def to_py(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16)): return int(obj)
    if isinstance(obj, (np.float64, np.float32)): return float(obj)
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_py(v) for v in obj]
    return obj

split_artifact = {
    "run_timestamp": RUN_TS,
    "input_file": "Data/interim/harmonized_deduplicated.csv",
    "total_rows": n_total,
    "method": "skmultilearn.iterative_train_test_split",
    "ratio_target": "70/15/15",
    "ratio_actual": {
        "train_pct": round(n_train/n_total*100, 2),
        "validation_pct": round(n_val/n_total*100, 2),
        "test_pct": round(n_test/n_total*100, 2)
    },
    "split_sizes": {
        "train": n_train,
        "validation": n_val,
        "test": n_test
    },
    "label_prevalence": {
        "train": train_stats,
        "validation": val_stats,
        "test": test_stats
    },
    "validation_checks": checks,
    "split_ids": {
        "train": df_train["sample_id"].tolist(),
        "validation": df_val["sample_id"].tolist(),
        "test": df_test["sample_id"].tolist()
    }
}

split_json_path = os.path.join(ARTIFACTS_DIR, "split_v1.json")
with open(split_json_path, "w", encoding="utf-8") as f:
    json.dump(to_py(split_artifact), f, indent=2, ensure_ascii=False)

print(f"\nSaved Data Splits:")
print(f"  Train:      {train_csv}")
print(f"  Validation: {val_csv}")
print(f"  Test:       {test_csv}")
print(f"\nSaved Artifact:")
print(f"  {split_json_path}")
print("\nPhase 5b Splitting COMPLETE.")
