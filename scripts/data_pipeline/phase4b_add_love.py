"""
Phase 4b — Add Love Emotion Label to Final Dataset
====================================================
Decision (2026-10-10, user):
  - Add 'love' as a 7th target emotion column (alongside anger, disgust, fear,
    joy, sadness, surprise).
  - Re-include previously excluded EmoNoBa love-only rows (LOVE-1 strategy
    reversed) — they are now valid data since love is a target emotion.
  - Love only exists in EmoNoBa. UBMEC and MONOVAB have no love annotations,
    so 'love' is set to NaN for those sources.

Input:
  - Data/interim/harmonized_deduplicated.csv   (existing corpus, 41,184 rows)
  - Data/raw/Emonoba/{Train,Val,Test}.csv       (raw EmoNoBa for love-only rows)

Output:
  - Data/interim/harmonized_deduplicated.csv   (overwritten, +love column +love-only rows)
  - Data/processed/train/train.csv             (re-split)
  - Data/processed/validation/validation.csv   (re-split)
  - Data/processed/test/test.csv               (re-split)
  - artifacts/splits/split_v2_love.json        (split metadata)

Rules:
  - love=1/0 for EmoNoBa rows (based on original_love column).
  - love=NaN for UBMEC and MONOVAB rows (no love annotation available).
  - Love-only rows: love=1, all other 6 target emotions = their original values
    (which are all 0 by definition of "love-only").
  - Multi-label stratified split using the 7 target labels.
  - Deduplication is re-applied (text-level) before splitting.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, hashlib, datetime
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

# Paths
WORKSPACE     = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
RAW_EMONOBA   = os.path.join(WORKSPACE, "Data", "raw", "Emonoba")
INTERIM       = os.path.join(WORKSPACE, "Data", "interim")
PROCESSED_DIR = os.path.join(WORKSPACE, "Data", "processed")
ARTIFACTS_DIR = os.path.join(WORKSPACE, "artifacts", "splits")

DEDUP_CSV  = os.path.join(INTERIM, "harmonized_deduplicated.csv")
TRAIN_CSV  = os.path.join(PROCESSED_DIR, "train",      "train.csv")
VAL_CSV    = os.path.join(PROCESSED_DIR, "validation", "validation.csv")
TEST_CSV   = os.path.join(PROCESSED_DIR, "test",       "test.csv")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TARGET_COLS_NEW = ["anger", "disgust", "fear", "joy", "love", "sadness", "surprise"]

RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("=" * 65)
print("Phase 4b - Adding Love Emotion Label to Final Dataset")
print("=" * 65)

# =============================================================================
# STEP 1: Load existing harmonized deduplicated corpus
# =============================================================================
print("\n[1/5] Loading existing harmonized corpus ...")
df = pd.read_csv(DEDUP_CSV, low_memory=False)
print(f"      Loaded: {len(df):,} rows, {len(df.columns)} columns")

# =============================================================================
# STEP 2: Add 'love' column to existing rows
# =============================================================================
print("\n[2/5] Adding 'love' column to existing rows ...")

def assign_love(row):
    if row["source_dataset"] == "EmoNoBa":
        ov = row.get("original_love", None)
        if pd.isna(ov):
            return 0
        return int(ov)
    return np.nan  # UBMEC / MONOVAB -- no love annotation

df["love"] = df.apply(assign_love, axis=1)

emonoba_mask = df["source_dataset"] == "EmoNoBa"
love_count_existing = int(df.loc[emonoba_mask, "love"].sum())
print(f"      EmoNoBa rows in existing corpus: {emonoba_mask.sum():,}")
print(f"      Of these, love=1 (co-occurring with other emotions): {love_count_existing:,}")
print(f"      Non-EmoNoBa rows with love=NaN: {(df['source_dataset'] != 'EmoNoBa').sum():,}")

# =============================================================================
# STEP 3: Extract and build love-only rows from raw EmoNoBa
# =============================================================================
print("\n[3/5] Re-including previously excluded love-only EmoNoBa rows ...")

love_only_rows = []
for split_name in ["Train", "Val", "Test"]:
    fpath = os.path.join(RAW_EMONOBA, f"{split_name}.csv")
    raw = pd.read_csv(fpath)
    target_compat = raw["Joy"] + raw["Surprise"] + raw["Anger"] + raw["Sadness"] + raw["Fear"]
    love_only_mask = (raw["Love"] == 1) & (target_compat == 0)
    love_only_split = raw[love_only_mask]
    print(f"      {split_name}: {len(love_only_split):,} love-only rows re-included")

    for _, row in love_only_split.iterrows():
        love_only_rows.append({
            "source_dataset": "EmoNoBa",
            "source_split":   split_name,
            "source_row_id":  int(row["ID"]),
            "text":           str(row["Data"]),
            "anger":   0,
            "disgust": 0,
            "fear":    0,
            "joy":     0,
            "love":    1,
            "sadness": 0,
            "surprise": 0,
            "emonoba_disgust_assumption": True,
            "f5_eligible":               True,
            "original_love":     1,
            "original_class":    None,
            "original_contempt": None,
            "original_enjoyment": None,
            "domain": str(row.get("Domain", "")),
            "topic":  str(row.get("Topic",  "")),
            "_was_love_only_excluded": True,
        })

df_love_only = pd.DataFrame(love_only_rows)
total_love_only_added = len(df_love_only)
print(f"\n      Total love-only rows to add: {total_love_only_added:,}")

# =============================================================================
# STEP 4: Combine and deduplicate
# =============================================================================
print("\n[4/5] Combining, deduplicating, and re-splitting ...")

df["_was_love_only_excluded"] = False

for col in df.columns:
    if col not in df_love_only.columns:
        df_love_only[col] = np.nan

df_combined = pd.concat([df, df_love_only], ignore_index=True)
print(f"\n      Combined before dedup: {len(df_combined):,} rows")

before_dedup = len(df_combined)
df_combined["_text_norm"] = df_combined["text"].str.strip().str.lower()
df_combined = df_combined.drop_duplicates(subset=["_text_norm"], keep="first")
df_combined = df_combined.drop(columns=["_text_norm"])
after_dedup = len(df_combined)
dedup_removed = before_dedup - after_dedup
print(f"      Rows removed by deduplication: {dedup_removed:,}")
print(f"      Final corpus size: {after_dedup:,} rows")

print(f"\n      Emotion label counts:")
for col in TARGET_COLS_NEW:
    print(f"        {col}: {int(df_combined[col].fillna(0).sum()):,}")

df_combined = df_combined.drop(columns=["_was_love_only_excluded"], errors="ignore")

# Enforce column order
meta_cols = [
    "source_dataset", "source_split", "source_row_id", "text",
    "anger", "disgust", "fear", "joy", "love", "sadness", "surprise",
    "emonoba_disgust_assumption", "f5_eligible",
    "original_love", "original_class", "original_contempt", "original_enjoyment",
    "domain", "topic"
]
extra_cols = [c for c in df_combined.columns if c not in meta_cols]
final_col_order = meta_cols + extra_cols
df_combined = df_combined[[c for c in final_col_order if c in df_combined.columns]]
df_combined = df_combined.reset_index(drop=True)

print(f"\n      Saving updated harmonized_deduplicated.csv ...")
df_combined.to_csv(DEDUP_CSV, index=False, encoding="utf-8-sig")
print(f"      Saved: {DEDUP_CSV}")

# =============================================================================
# STEP 5: Re-split with 7-label stratification
# =============================================================================
print("\n[5/5] Performing multi-label stratified split (70/15/15) on 7 labels ...")

# For stratification, treat love=NaN as 0
strat_df = df_combined.copy()
strat_df["love_strat"] = strat_df["love"].fillna(0).astype(int)

STRAT_COLS = ["anger", "disgust", "fear", "joy", "love_strat", "sadness", "surprise"]

def generate_sample_id(row):
    text_hash = hashlib.sha1(str(row["text"]).encode("utf-8")).hexdigest()[:8]
    ds = str(row["source_dataset"])[:3].upper()
    rid = str(row["source_row_id"]) if pd.notna(row["source_row_id"]) else "NA"
    return f"{ds}-{rid}-{text_hash}"

df_combined["sample_id"] = df_combined.apply(generate_sample_id, axis=1)
strat_df["sample_id"] = df_combined["sample_id"]

if df_combined["sample_id"].duplicated().any():
    print("      Warning: some sample_ids duplicated -- appending index")
    df_combined["sample_id"] = df_combined["sample_id"] + "-" + df_combined.index.astype(str)
    strat_df["sample_id"] = df_combined["sample_id"]

X = np.expand_dims(strat_df.index.values, axis=1)
y = strat_df[STRAT_COLS].values.astype(int)

X_train_idx, y_train, X_temp_idx, y_temp = iterative_train_test_split(X, y, test_size=0.3)
X_val_idx, y_val, X_test_idx, y_test     = iterative_train_test_split(X_temp_idx, y_temp, test_size=0.5)

train_indices = X_train_idx.flatten()
val_indices   = X_val_idx.flatten()
test_indices  = X_test_idx.flatten()

df_train = df_combined.iloc[train_indices].copy()
df_val   = df_combined.iloc[val_indices].copy()
df_test  = df_combined.iloc[test_indices].copy()

n_total = len(df_combined)
n_train = len(df_train)
n_val   = len(df_val)
n_test  = len(df_test)

print(f"\n      Split sizes:")
print(f"        Train:      {n_train:,} ({n_train/n_total*100:.1f}%)")
print(f"        Validation: {n_val:,} ({n_val/n_total*100:.1f}%)")
print(f"        Test:       {n_test:,} ({n_test/n_total*100:.1f}%)")

# Validation
train_ids = set(df_train["sample_id"])
val_ids   = set(df_val["sample_id"])
test_ids  = set(df_test["sample_id"])
overlaps  = len(train_ids & val_ids) + len(train_ids & test_ids) + len(val_ids & test_ids)
accounted = n_train + n_val + n_test

print(f"\n      Validation checks:")
print(f"        No row overlap: {'PASS' if overlaps == 0 else 'FAIL'} (overlaps={overlaps})")
print(f"        All rows accounted: {'PASS' if accounted == n_total else 'FAIL'} ({accounted}/{n_total})")

print(f"\n      Love distribution per split:")
for split_name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    love_pos = int(split_df["love"].fillna(0).sum())
    print(f"        {split_name}: love=1 count={love_pos}")

# Save processed splits
df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
df_val.to_csv(VAL_CSV,     index=False, encoding="utf-8-sig")
df_test.to_csv(TEST_CSV,   index=False, encoding="utf-8-sig")

print(f"\n      Saved splits:")
print(f"        {TRAIN_CSV}")
print(f"        {VAL_CSV}")
print(f"        {TEST_CSV}")

# Save artifact JSON
def to_py(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16)): return int(obj)
    if isinstance(obj, (np.float64, np.float32)):        return float(obj)
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):  return [to_py(v) for v in obj]
    return obj

def label_stats(split_df, label_cols):
    stats = {}
    for col in label_cols:
        pos = int(split_df[col].fillna(0).sum())
        stats[col] = {"positive_count": pos, "prevalence_pct": round(pos / len(split_df) * 100, 2)}
    return stats

artifact = {
    "run_timestamp": RUN_TS,
    "change": "Added love as 7th emotion target label; re-included LOVE-1 excluded rows",
    "input_file": "Data/interim/harmonized_deduplicated.csv",
    "total_rows_before": before_dedup - total_love_only_added,
    "love_only_rows_re_included": total_love_only_added,
    "dedup_removed": dedup_removed,
    "total_rows_after": after_dedup,
    "method": "skmultilearn.iterative_train_test_split",
    "ratio_target": "70/15/15",
    "ratio_actual": {
        "train_pct": round(n_train/n_total*100, 2),
        "validation_pct": round(n_val/n_total*100, 2),
        "test_pct": round(n_test/n_total*100, 2),
    },
    "split_sizes": {"train": n_train, "validation": n_val, "test": n_test},
    "label_prevalence": {
        "train":      to_py(label_stats(df_train, TARGET_COLS_NEW)),
        "validation": to_py(label_stats(df_val,   TARGET_COLS_NEW)),
        "test":       to_py(label_stats(df_test,  TARGET_COLS_NEW)),
    },
    "validation_checks": {
        "no_row_overlap": {"pass": overlaps == 0, "overlaps": overlaps},
        "all_rows_accounted": {"pass": accounted == n_total},
    },
    "notes": {
        "love_in_emonoba": "love=1/0 based on original EmoNoBa Love column",
        "love_in_ubmec_monovab": "love=NaN -- no love annotations exist in these datasets",
        "love_only_rows": "Previously excluded LOVE-1 rows re-included; all other 6 emotions remain 0",
        "disgust_assumption": "EmoNoBa disgust=0 assumption still applies (Strategy D)",
    }
}

artifact_path = os.path.join(ARTIFACTS_DIR, "split_v2_love.json")
with open(artifact_path, "w", encoding="utf-8") as f:
    json.dump(to_py(artifact), f, indent=2, ensure_ascii=False)

print(f"\n      Saved split artifact: {artifact_path}")

print("\n" + "=" * 65)
print("Phase 4b COMPLETE - Love emotion label added successfully!")
print("=" * 65)
print(f"\nFinal corpus: {after_dedup:,} rows  (was {before_dedup - total_love_only_added:,})")
print(f"Love-only rows re-included: {total_love_only_added:,}")
print(f"New emotion columns: {TARGET_COLS_NEW}")
