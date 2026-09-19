"""
Phase 4 — Dataset Harmonization
=================================
Approved decisions (2026-09-19, user):
  1. EmoNoBa Disgust  → Strategy D: assign disgust=0 as documented assumption; F5 mandatory
  2. EmoNoBa Love-only rows (2277) → LOVE-1: EXCLUDE
  3. MONOVAB Contempt-only (2128) → CONTEMPT-1: EXCLUDE
  4. MONOVAB enjoyment → joy mapping: FROZEN (semantic harmonization)
  5. UBMEC single-label → one-hot six binary: FROZEN
  6. Target taxonomy: anger, disgust, fear, joy, sadness, surprise: FROZEN

Raw data under Data/raw/ is NOT modified.
All output goes to Data/interim/ and logs/harmonization/.
"""

import os
import json
import csv
import re
import unicodedata
import datetime
import pandas as pd
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
WORKSPACE = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
RAW_DIR   = os.path.join(WORKSPACE, "Data", "raw")
INTERIM   = os.path.join(WORKSPACE, "Data", "interim")
LOG_DIR   = os.path.join(WORKSPACE, "logs", "harmonization")

os.makedirs(INTERIM,  exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ─── Target columns in fixed order ───────────────────────────────────────────
TARGET_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# Provenance & metadata columns for the harmonized corpus
META_COLS = [
    "source_dataset",          # EmoNoBa / UBMEC / MONOVAB
    "source_split",            # Train / Val / Test (EmoNoBa only, else NONE)
    "source_row_id",           # original identifier (EmoNoBa: ID column; UBMEC: row index; MONOVAB: Unnamed:0)
    "text",                    # the Bangla comment
    # Target label columns injected here
    "emonoba_disgust_assumption",   # True for all EmoNoBa included rows (disgust=0 is assumed)
    "f5_eligible",             # True for EmoNoBa rows — these must be identifiable for F5 sensitivity analysis
    "original_love",           # EmoNoBa: original Love value; others: NaN
    "original_class",          # UBMEC: original categorical class string; others: NaN
    "original_contempt",       # MONOVAB: original contempt value; others: NaN
    "original_enjoyment",      # MONOVAB: original enjoyment value (before →joy rename); others: NaN
    "domain",                  # EmoNoBa Domain column (Youtube/Facebook/Twitter); others: NaN
    "topic",                   # EmoNoBa Topic column; others: NaN
]

# ─── Manifest / log accumulators ─────────────────────────────────────────────
manifest_rows  = []   # every row in every source dataset, included or excluded
exclusion_rows = []   # excluded rows only
transformation_log = []  # one entry per dataset × transformation


def log_transform(source_dataset, transformation, native_label, target_label, n_affected, rule, notes):
    transformation_log.append({
        "source_dataset": source_dataset,
        "transformation": transformation,
        "native_label": native_label,
        "target_label": target_label,
        "n_affected_rows": n_affected,
        "harmonization_rule": rule,
        "notes": notes,
        "timestamp": RUN_TS
    })


# =============================================================================
# 1. EMONOBA
# =============================================================================
print("="*60)
print("Processing EmoNoBa …")
print("="*60)

emonoba_splits = {
    "Train": os.path.join(RAW_DIR, "Emonoba", "Train.csv"),
    "Val":   os.path.join(RAW_DIR, "Emonoba", "Val.csv"),
    "Test":  os.path.join(RAW_DIR, "Emonoba", "Test.csv"),
}

emonoba_included_frames = []

for split_name, fpath in emonoba_splits.items():
    df = pd.read_csv(fpath)
    print(f"  Loaded {split_name}: {len(df)} rows")

    for _, row in df.iterrows():
        src_id = int(row["ID"])
        text   = str(row["Data"])

        orig_love     = int(row["Love"])
        orig_joy      = int(row["Joy"])
        orig_surprise = int(row["Surprise"])
        orig_anger    = int(row["Anger"])
        orig_sadness  = int(row["Sadness"])
        orig_fear     = int(row["Fear"])

        # Detect Love-only: Love=1 AND all 5 target-compatible source labels = 0
        target_compat_sum = orig_joy + orig_surprise + orig_anger + orig_sadness + orig_fear
        is_love_only = (orig_love == 1) and (target_compat_sum == 0)

        if is_love_only:
            # LOVE-1: EXCLUDE
            reason = "LOVE-1: Love is the only active native label; row excluded after Love excluded from target taxonomy"
            manifest_rows.append({
                "source_dataset":  "EmoNoBa",
                "source_split":    split_name,
                "source_row_id":   src_id,
                "text_snippet":    text[:80],
                "included":        False,
                "exclusion_reason": reason,
                "harmonization_rule": "LOVE-1 (approved 2026-09-19)",
                "original_labels": f"Love={orig_love} Joy={orig_joy} Surprise={orig_surprise} Anger={orig_anger} Sadness={orig_sadness} Fear={orig_fear}",
                "harmonized_labels": "EXCLUDED"
            })
            exclusion_rows.append({
                "source_dataset": "EmoNoBa",
                "source_split":   split_name,
                "source_row_id":  src_id,
                "text_snippet":   text[:80],
                "exclusion_reason": reason,
                "harmonization_rule": "LOVE-1",
                "original_Love": orig_love,
                "original_Joy": orig_joy,
                "original_Surprise": orig_surprise,
                "original_Anger": orig_anger,
                "original_Sadness": orig_sadness,
                "original_Fear": orig_fear
            })
        else:
            # INCLUDED: apply harmonization
            # Direct mappings for 5 labels; disgust = 0 (Strategy D assumption)
            harmonized = {
                "anger":   orig_anger,
                "disgust": 0,       # Strategy D: annotation-coverage assumption; NOT original annotation
                "fear":    orig_fear,
                "joy":     orig_joy,
                "sadness": orig_sadness,
                "surprise": orig_surprise,
            }
            harmonized_str = " ".join(f"{k}={v}" for k, v in harmonized.items())

            manifest_rows.append({
                "source_dataset":  "EmoNoBa",
                "source_split":    split_name,
                "source_row_id":   src_id,
                "text_snippet":    text[:80],
                "included":        True,
                "exclusion_reason": "",
                "harmonization_rule": "DIRECT+DISGUST-D",
                "original_labels": f"Love={orig_love} Joy={orig_joy} Surprise={orig_surprise} Anger={orig_anger} Sadness={orig_sadness} Fear={orig_fear}",
                "harmonized_labels": harmonized_str
            })

            emonoba_included_frames.append({
                "source_dataset": "EmoNoBa",
                "source_split":   split_name,
                "source_row_id":  src_id,
                "text": text,
                # Target labels
                "anger":   harmonized["anger"],
                "disgust": harmonized["disgust"],
                "fear":    harmonized["fear"],
                "joy":     harmonized["joy"],
                "sadness": harmonized["sadness"],
                "surprise": harmonized["surprise"],
                # Provenance flags
                "emonoba_disgust_assumption": True,   # disgust=0 is an assumed value, NOT annotated
                "f5_eligible": True,                   # must be identifiable for F5 sensitivity analysis
                # Original values for full traceability
                "original_love":     orig_love,
                "original_class":    None,
                "original_contempt": None,
                "original_enjoyment": None,
                # Metadata from EmoNoBa
                "domain": str(row.get("Domain", "")),
                "topic":  str(row.get("Topic", "")),
            })

# Log EmoNoBa transformations
love_only_count = sum(1 for r in manifest_rows
                      if r["source_dataset"]=="EmoNoBa" and not r["included"])
included_emonoba = sum(1 for r in manifest_rows
                       if r["source_dataset"]=="EmoNoBa" and r["included"])

log_transform("EmoNoBa", "DIRECT column rename",
              "Joy/Anger/Sadness/Surprise/Fear", "joy/anger/sadness/surprise/fear",
              included_emonoba, "Phase 3 FROZEN direct mappings",
              "Capitalization-only difference; exact semantic equivalence")
log_transform("EmoNoBa", "STRATEGY-D: disgust=0 annotation-coverage assumption",
              "(absent — not annotated)", "disgust=0",
              included_emonoba, "DISGUST-D (approved 2026-09-19)",
              "EmoNoBa did not collect Disgust annotations. disgust=0 is an operational assumption, NOT an original annotation. F5 sensitivity analysis is mandatory. Rows flagged as f5_eligible=True and emonoba_disgust_assumption=True.")
log_transform("EmoNoBa", "LOVE-1 exclusion",
              "Love (only active label)", "EXCLUDED",
              love_only_count, "LOVE-1 (approved 2026-09-19)",
              "2277 rows where Love was the ONLY active native label. Dropped to avoid all-zero target vectors that would misrepresent unannotated examples as negatives.")
log_transform("EmoNoBa", "Love column dropped (non-target)",
              "Love", "DROPPED",
              included_emonoba, "DATASET_SPEC.md Sec 3 FROZEN",
              "Love is not in the BETC target taxonomy. Not mapped to any target label.")

print(f"  EmoNoBa included rows: {included_emonoba}")
print(f"  EmoNoBa excluded (Love-only): {love_only_count}")

# =============================================================================
# 2. UBMEC
# =============================================================================
print("\n" + "="*60)
print("Processing UBMEC …")
print("="*60)

ubmec_path = os.path.join(RAW_DIR, "UBMEC Corpus_Sakib(updated).xlsx")
ubmec_df   = pd.read_excel(ubmec_path, sheet_name="UBMEC")
print(f"  Loaded UBMEC: {len(ubmec_df)} rows")

ubmec_included_frames = []
valid_ubmec_classes = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}

for row_idx, row in ubmec_df.iterrows():
    text       = str(row["text"])
    orig_class = str(row["classes"]).strip().lower()

    if orig_class not in valid_ubmec_classes:
        # Unexpected class — log and skip (should not happen with this dataset)
        reason = f"UBMEC-UNEXPECTED-CLASS: class '{orig_class}' is not in the 6 target classes"
        manifest_rows.append({
            "source_dataset":  "UBMEC",
            "source_split":    "NONE",
            "source_row_id":   row_idx,
            "text_snippet":    text[:80],
            "included":        False,
            "exclusion_reason": reason,
            "harmonization_rule": "UBMEC-UNEXPECTED-CLASS",
            "original_labels": f"classes={orig_class}",
            "harmonized_labels": "EXCLUDED"
        })
        exclusion_rows.append({
            "source_dataset": "UBMEC",
            "source_split":   "NONE",
            "source_row_id":  row_idx,
            "text_snippet":   text[:80],
            "exclusion_reason": reason,
            "harmonization_rule": "UBMEC-UNEXPECTED-CLASS",
            "original_class": orig_class
        })
        continue

    # One-hot conversion: 1 for observed class, 0 for all others
    harmonized = {c: (1 if c == orig_class else 0) for c in TARGET_COLS}
    harmonized_str = " ".join(f"{k}={v}" for k, v in harmonized.items())

    manifest_rows.append({
        "source_dataset":  "UBMEC",
        "source_split":    "NONE",
        "source_row_id":   row_idx,
        "text_snippet":    text[:80],
        "included":        True,
        "exclusion_reason": "",
        "harmonization_rule": "UBMEC-ONE-HOT",
        "original_labels": f"classes={orig_class}",
        "harmonized_labels": harmonized_str
    })

    ubmec_included_frames.append({
        "source_dataset": "UBMEC",
        "source_split":   "NONE",
        "source_row_id":  row_idx,
        "text": text,
        "anger":   harmonized["anger"],
        "disgust": harmonized["disgust"],
        "fear":    harmonized["fear"],
        "joy":     harmonized["joy"],
        "sadness": harmonized["sadness"],
        "surprise": harmonized["surprise"],
        "emonoba_disgust_assumption": False,
        "f5_eligible": False,
        "original_love":     None,
        "original_class":    orig_class,   # preserve original categorical class
        "original_contempt": None,
        "original_enjoyment": None,
        "domain": None,
        "topic":  None,
    })

included_ubmec = len(ubmec_included_frames)
excluded_ubmec = len(ubmec_df) - included_ubmec
log_transform("UBMEC", "Single-label to one-hot six-binary conversion",
              "classes (categorical string)", "six binary target columns",
              included_ubmec, "UBMEC-ONE-HOT (FROZEN)",
              "UBMEC is single-label; observed class → 1, all other 5 → 0. This is a representation change, NOT a claim that UBMEC is multi-label. The 0s mean 'not the observed category in a mutually-exclusive 6-class scheme'.")

print(f"  UBMEC included rows: {included_ubmec}")
if excluded_ubmec > 0:
    print(f"  UBMEC excluded (unexpected class): {excluded_ubmec}")

# =============================================================================
# 3. MONOVAB
# =============================================================================
print("\n" + "="*60)
print("Processing MONOVAB …")
print("="*60)

monovab_path = os.path.join(RAW_DIR, "MONOVAB (1).csv")
monovab_df   = pd.read_csv(monovab_path)
print(f"  Loaded MONOVAB: {len(monovab_df)} rows")

monovab_included_frames = []
monovab_label_cols_native = ["anger", "contempt", "disgust", "enjoyment", "fear", "sadness", "surprise"]

for _, row in monovab_df.iterrows():
    src_id  = int(row["Unnamed: 0"])
    text    = str(row["comment"])

    orig_anger    = int(row["anger"])
    orig_contempt = int(row["contempt"])
    orig_disgust  = int(row["disgust"])
    orig_enjoyment= int(row["enjoyment"])
    orig_fear     = int(row["fear"])
    orig_sadness  = int(row["sadness"])
    orig_surprise = int(row["surprise"])

    # Detect Contempt-only: contempt=1 AND all other 6 native columns = 0
    other_sum = orig_anger + orig_disgust + orig_enjoyment + orig_fear + orig_sadness + orig_surprise
    is_contempt_only = (orig_contempt == 1) and (other_sum == 0)

    orig_label_str = (f"anger={orig_anger} contempt={orig_contempt} disgust={orig_disgust} "
                      f"enjoyment={orig_enjoyment} fear={orig_fear} sadness={orig_sadness} surprise={orig_surprise}")

    if is_contempt_only:
        # CONTEMPT-1: EXCLUDE
        reason = ("CONTEMPT-1: contempt is the only active native label; row excluded after contempt "
                  "excluded from target taxonomy. Retaining as all-zero would risk disgust=0 label "
                  "noise given semantic overlap between contempt and disgust.")
        manifest_rows.append({
            "source_dataset":  "MONOVAB",
            "source_split":    "NONE",
            "source_row_id":   src_id,
            "text_snippet":    text[:80],
            "included":        False,
            "exclusion_reason": reason,
            "harmonization_rule": "CONTEMPT-1 (approved 2026-09-19)",
            "original_labels": orig_label_str,
            "harmonized_labels": "EXCLUDED"
        })
        exclusion_rows.append({
            "source_dataset": "MONOVAB",
            "source_split":   "NONE",
            "source_row_id":  src_id,
            "text_snippet":   text[:80],
            "exclusion_reason": reason,
            "harmonization_rule": "CONTEMPT-1",
            "original_anger": orig_anger,
            "original_contempt": orig_contempt,
            "original_disgust": orig_disgust,
            "original_enjoyment": orig_enjoyment,
            "original_fear": orig_fear,
            "original_sadness": orig_sadness,
            "original_surprise": orig_surprise
        })
    else:
        # INCLUDED: apply harmonization
        # enjoyment → joy (FROZEN semantic mapping)
        harmonized = {
            "anger":   orig_anger,
            "disgust": orig_disgust,
            "fear":    orig_fear,
            "joy":     orig_enjoyment,   # enjoyment → joy: FROZEN mapping
            "sadness": orig_sadness,
            "surprise": orig_surprise,
        }
        harmonized_str = " ".join(f"{k}={v}" for k, v in harmonized.items())

        manifest_rows.append({
            "source_dataset":  "MONOVAB",
            "source_split":    "NONE",
            "source_row_id":   src_id,
            "text_snippet":    text[:80],
            "included":        True,
            "exclusion_reason": "",
            "harmonization_rule": "DIRECT+ENJOYMENT-JOY+CONTEMPT-DROP",
            "original_labels": orig_label_str,
            "harmonized_labels": harmonized_str
        })

        monovab_included_frames.append({
            "source_dataset": "MONOVAB",
            "source_split":   "NONE",
            "source_row_id":  src_id,
            "text": text,
            "anger":   harmonized["anger"],
            "disgust": harmonized["disgust"],
            "fear":    harmonized["fear"],
            "joy":     harmonized["joy"],
            "sadness": harmonized["sadness"],
            "surprise": harmonized["surprise"],
            "emonoba_disgust_assumption": False,
            "f5_eligible": False,
            "original_love":     None,
            "original_class":    None,
            "original_contempt": orig_contempt,    # preserve for traceability
            "original_enjoyment": orig_enjoyment,  # preserve for traceability (enjoyment→joy)
            "domain": None,
            "topic":  None,
        })

included_monovab = len(monovab_included_frames)
excluded_monovab = len(monovab_df) - included_monovab
log_transform("MONOVAB", "CONTEMPT-1 exclusion",
              "contempt (only active label)", "EXCLUDED",
              excluded_monovab, "CONTEMPT-1 (approved 2026-09-19)",
              "2128 rows where contempt was the ONLY active native label. Excluded to avoid all-zero target vectors and to prevent disgust=0 label noise given semantic contempt-disgust overlap.")
log_transform("MONOVAB", "Contempt column dropped (non-target)",
              "contempt", "DROPPED",
              included_monovab, "DATASET_SPEC.md Sec 3 FROZEN",
              "contempt is not in the BETC target taxonomy.")
log_transform("MONOVAB", "ENJOYMENT→JOY semantic harmonization mapping (FROZEN)",
              "enjoyment", "joy",
              sum(1 for r in monovab_included_frames if r["original_enjoyment"] == 1),
              "enjoyment→joy FROZEN (approved 2026-09-19)",
              "Semantic near-equivalence: enjoyment is MONOVAB's Ekman Joy/Happiness category. "
              "Mapping is a semantic harmonization, NOT a claim of theoretical label identity. "
              "original_enjoyment preserved in harmonized corpus for traceability.")
log_transform("MONOVAB", "DIRECT mappings",
              "anger/disgust/fear/sadness/surprise", "anger/disgust/fear/sadness/surprise",
              included_monovab, "Phase 3 FROZEN direct mappings",
              "Five MONOVAB labels are exact Ekman equivalents; column names retained as-is.")

print(f"  MONOVAB included rows: {included_monovab}")
print(f"  MONOVAB excluded (Contempt-only): {excluded_monovab}")

# =============================================================================
# 4. ASSEMBLE HARMONIZED CORPUS
# =============================================================================
print("\n" + "="*60)
print("Assembling harmonized corpus …")
print("="*60)

all_rows = emonoba_included_frames + ubmec_included_frames + monovab_included_frames
harmonized_df = pd.DataFrame(all_rows)

# Enforce column order
col_order = (
    ["source_dataset", "source_split", "source_row_id", "text"] +
    TARGET_COLS +
    ["emonoba_disgust_assumption", "f5_eligible",
     "original_love", "original_class", "original_contempt", "original_enjoyment",
     "domain", "topic"]
)
harmonized_df = harmonized_df[col_order]

print(f"  Total harmonized rows: {len(harmonized_df)}")
print(f"  Columns: {list(harmonized_df.columns)}")

# =============================================================================
# 5. VALIDATION CHECKS
# =============================================================================
print("\n" + "="*60)
print("Running validation checks …")
print("="*60)

checks = {}

# Check 1: Target label columns contain only 0/1
for col in TARGET_COLS:
    unique_vals = sorted(harmonized_df[col].unique().tolist())
    ok = all(v in [0, 1] for v in unique_vals)
    checks[f"target_col_{col}_binary_only"] = {"pass": ok, "unique_values": unique_vals}
    if ok:
        print(f"  ✅ {col}: only 0/1 values")
    else:
        print(f"  ❌ {col}: unexpected values {unique_vals}")

# Check 2: No null text values in included rows
null_text = harmonized_df["text"].isna().sum()
checks["no_null_text"] = {"pass": null_text == 0, "null_count": int(null_text)}
print(f"  {'✅' if null_text==0 else '❌'} null texts: {null_text}")

# Check 3: source_dataset only contains expected values
source_vals = set(harmonized_df["source_dataset"].unique().tolist())
expected_sources = {"EmoNoBa", "UBMEC", "MONOVAB"}
checks["source_dataset_valid"] = {"pass": source_vals == expected_sources, "found": list(source_vals)}
print(f"  {'✅' if source_vals==expected_sources else '❌'} source datasets: {source_vals}")

# Check 4: All EmoNoBa rows have emonoba_disgust_assumption=True
e_rows = harmonized_df[harmonized_df["source_dataset"] == "EmoNoBa"]
all_flagged = e_rows["emonoba_disgust_assumption"].all()
checks["emonoba_disgust_assumption_flag"] = {"pass": bool(all_flagged), "count_true": int(e_rows["emonoba_disgust_assumption"].sum())}
print(f"  {'✅' if all_flagged else '❌'} All EmoNoBa rows have emonoba_disgust_assumption=True: {e_rows['emonoba_disgust_assumption'].sum()}/{len(e_rows)}")

# Check 5: No UBMEC or MONOVAB rows have emonoba_disgust_assumption=True
non_e = harmonized_df[harmonized_df["source_dataset"] != "EmoNoBa"]
false_flags = non_e["emonoba_disgust_assumption"].sum()
checks["non_emonoba_disgust_assumption_false"] = {"pass": bool(false_flags == 0), "false_positives": int(false_flags)}
print(f"  {'✅' if false_flags==0 else '❌'} Non-EmoNoBa rows with emonoba_disgust_assumption=True: {false_flags}")

# Check 6: All-zero rows (may be legitimate but worth counting)
all_zero_mask = (harmonized_df[TARGET_COLS].sum(axis=1) == 0)
all_zero_count = int(all_zero_mask.sum())
checks["all_zero_rows"] = {"count": all_zero_count, "pct": round(all_zero_count/len(harmonized_df)*100, 2)}
print(f"  ℹ️  All-zero target vector rows: {all_zero_count} ({checks['all_zero_rows']['pct']}%)")
# Show breakdown
if all_zero_count > 0:
    az_by_source = harmonized_df[all_zero_mask]["source_dataset"].value_counts().to_dict()
    checks["all_zero_by_source"] = az_by_source
    print(f"     by source: {az_by_source}")

# Check 7: UBMEC rows should be exactly one-hot (sum of target labels = 1 always)
ubmec_rows = harmonized_df[harmonized_df["source_dataset"] == "UBMEC"]
ubmec_label_sums = ubmec_rows[TARGET_COLS].sum(axis=1)
ubmec_not_one = (ubmec_label_sums != 1).sum()
checks["ubmec_one_hot_valid"] = {"pass": bool(ubmec_not_one == 0), "rows_not_exactly_one": int(ubmec_not_one)}
print(f"  {'✅' if ubmec_not_one==0 else '❌'} All UBMEC rows are one-hot (sum=1): {ubmec_not_one} violations")

# Check 8: Row counts per source
rc = harmonized_df["source_dataset"].value_counts().to_dict()
checks["row_counts_by_source"] = rc
print(f"  ℹ️  Row counts: {rc}")

# Check 9: No Love-only rows remain
emonoba_rows_included = harmonized_df[harmonized_df["source_dataset"] == "EmoNoBa"]
# A Love-only row would have original_love=1 AND all 6 target labels=0
if "original_love" in harmonized_df.columns:
    potential_love_only = (
        (emonoba_rows_included["original_love"] == 1) &
        (emonoba_rows_included[TARGET_COLS].sum(axis=1) == 0)
    ).sum()
    checks["no_love_only_in_included"] = {"pass": bool(potential_love_only == 0), "count": int(potential_love_only)}
    print(f"  {'✅' if potential_love_only==0 else '❌'} Love-only rows in included corpus: {potential_love_only}")

# =============================================================================
# 6. LABEL STATISTICS
# =============================================================================
print("\n" + "="*60)
print("Computing label statistics …")
print("="*60)

label_stats = {}
for col in TARGET_COLS:
    total_pos = int(harmonized_df[col].sum())
    total_rows = len(harmonized_df)
    prevalence = round(total_pos / total_rows * 100, 2)
    by_source = {}
    for src in ["EmoNoBa", "UBMEC", "MONOVAB"]:
        src_df = harmonized_df[harmonized_df["source_dataset"] == src]
        by_source[src] = int(src_df[col].sum())
    label_stats[col] = {
        "total_positive": total_pos,
        "total_rows": total_rows,
        "prevalence_pct": prevalence,
        "positive_by_source": by_source
    }
    if col == "disgust":
        # Separate: how many of these are actual annotations vs assumptions?
        e_disgust_pos = by_source["EmoNoBa"]  # should be 0 under Strategy D
        true_disgust_annotations = by_source["UBMEC"] + by_source["MONOVAB"]
        label_stats[col]["note"] = (
            f"EmoNoBa disgust positive = {e_disgust_pos} (expected 0 under Strategy D). "
            f"True annotated Disgust positives from UBMEC+MONOVAB = {true_disgust_annotations}. "
            f"All EmoNoBa rows have disgust=0 by assumption (f5_eligible=True)."
        )
    print(f"  {col}: {total_pos} positive ({prevalence}%) | by source: {by_source}")

# Cardinality distribution
label_sums = harmonized_df[TARGET_COLS].sum(axis=1)
cardinality_dist = label_sums.value_counts(dropna=False).sort_index().to_dict()
cardinality_dist = {int(k): int(v) for k, v in cardinality_dist.items()}
mean_labels = round(float(label_sums.mean()), 4)
print(f"\n  Active labels per example: {cardinality_dist}")
print(f"  Mean active labels per example: {mean_labels}")

checks["cardinality_distribution"] = cardinality_dist
checks["mean_labels_per_example"] = mean_labels
checks["label_statistics"] = label_stats

# =============================================================================
# 7. SAVE ALL OUTPUTS
# =============================================================================
print("\n" + "="*60)
print("Saving outputs …")
print("="*60)

# 7a. Harmonized corpus (CSV + Parquet)
csv_out     = os.path.join(INTERIM, "harmonized_pre_dedup.csv")
parquet_out = os.path.join(INTERIM, "harmonized_pre_dedup.parquet")
harmonized_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
harmonized_df.to_parquet(parquet_out, index=False)
print(f"  Saved harmonized corpus: {csv_out}")
print(f"  Saved harmonized corpus: {parquet_out}")

# 7b. Exclusion log CSV
excl_out = os.path.join(LOG_DIR, "exclusion_log.csv")
if exclusion_rows:
    # Different sources have different column sets — normalise
    all_excl_cols = set()
    for r in exclusion_rows:
        all_excl_cols.update(r.keys())
    all_excl_cols = sorted(all_excl_cols)
    with open(excl_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=all_excl_cols, extrasaction="ignore")
        w.writeheader()
        for r in exclusion_rows:
            w.writerow({k: r.get(k, "") for k in all_excl_cols})
print(f"  Saved exclusion log: {excl_out}  ({len(exclusion_rows)} rows)")

# 7c. Transformation log
tf_out = os.path.join(LOG_DIR, "transformation_log.csv")
with open(tf_out, "w", newline="", encoding="utf-8-sig") as f:
    if transformation_log:
        w = csv.DictWriter(f, fieldnames=list(transformation_log[0].keys()))
        w.writeheader()
        w.writerows(transformation_log)
print(f"  Saved transformation log: {tf_out}  ({len(transformation_log)} entries)")

# 7d. Manifest (may be large; save as CSV)
manifest_out = os.path.join(LOG_DIR, "harmonization_manifest.csv")
if manifest_rows:
    manifest_cols = ["source_dataset","source_split","source_row_id","included",
                     "exclusion_reason","harmonization_rule","original_labels",
                     "harmonized_labels","text_snippet"]
    with open(manifest_out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=manifest_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(manifest_rows)
print(f"  Saved manifest: {manifest_out}  ({len(manifest_rows)} rows)")

# 7e. Validation + stats JSON
stats_out = os.path.join(LOG_DIR, "harmonization_validation.json")
final_stats = {
    "run_timestamp": RUN_TS,
    "approved_decisions": {
        "emonoba_disgust": "STRATEGY-D: disgust=0 operational assumption; F5 mandatory",
        "emonoba_love_only": "LOVE-1: 2277 rows excluded",
        "monovab_contempt_only": "CONTEMPT-1: 2128 rows excluded",
        "monovab_enjoyment_joy": "FROZEN: enjoyment→joy semantic harmonization",
        "ubmec_conversion": "FROZEN: single-label→one-hot six-binary",
        "target_taxonomy": "anger, disgust, fear, joy, sadness, surprise"
    },
    "row_counts": {
        "emonoba_raw": 22739,
        "emonoba_excluded": love_only_count,
        "emonoba_included": included_emonoba,
        "ubmec_raw": len(ubmec_df),
        "ubmec_excluded": excluded_ubmec,
        "ubmec_included": included_ubmec,
        "monovab_raw": len(monovab_df),
        "monovab_excluded": excluded_monovab,
        "monovab_included": included_monovab,
        "total_raw": 22739 + len(ubmec_df) + len(monovab_df),
        "total_excluded": love_only_count + excluded_ubmec + excluded_monovab,
        "total_harmonized_pre_dedup": len(harmonized_df)
    },
    "exclusion_summary": {
        "emonoba_love_only_rule": "LOVE-1",
        "emonoba_love_only_count": love_only_count,
        "monovab_contempt_only_rule": "CONTEMPT-1",
        "monovab_contempt_only_count": excluded_monovab,
        "ubmec_unexpected_class_count": excluded_ubmec
    },
    "validation_checks": checks,
    "label_statistics": label_stats,
    "cardinality_distribution": cardinality_dist,
    "mean_labels_per_example": mean_labels,
    "note_deduplication": "NOT YET APPLIED. This corpus is pre-deduplication. Deduplication is Phase 5.",
    "note_f5": "EmoNoBa rows are flagged f5_eligible=True and emonoba_disgust_assumption=True so they can be identified and excluded from Disgust-specific scoring in Experiment F5.",
}
with open(stats_out, "w", encoding="utf-8") as f:
    json.dump(final_stats, f, indent=2, ensure_ascii=False)
print(f"  Saved validation stats: {stats_out}")

print("\n" + "="*60)
print("Phase 4 harmonization COMPLETE.")
print("="*60)
print(f"\nFinal harmonized corpus (pre-dedup): {len(harmonized_df):,} rows")
print(f"Columns: {list(harmonized_df.columns)}")
print("\nDo NOT proceed to deduplication without explicit user approval (Phase 5).")
