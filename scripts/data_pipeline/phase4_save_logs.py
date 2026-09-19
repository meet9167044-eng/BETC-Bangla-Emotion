"""
Phase 4 Part 2 — Save logs, manifest, and validation JSON
(The harmonized CSV is already saved. This script generates the log artifacts.)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, csv, datetime
import pandas as pd

WORKSPACE = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
INTERIM   = os.path.join(WORKSPACE, "Data", "interim")
LOG_DIR   = os.path.join(WORKSPACE, "logs", "harmonization")
RAW_DIR   = os.path.join(WORKSPACE, "Data", "raw")
os.makedirs(LOG_DIR, exist_ok=True)

RUN_TS = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
TARGET_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

# ─── Re-read harmonized corpus ─────────────────────────────────────────────
csv_out = os.path.join(INTERIM, "harmonized_pre_dedup.csv")
harmonized_df = pd.read_csv(csv_out, dtype={
    "original_love": "Int64", "original_contempt": "Int64",
    "original_enjoyment": "Int64"
}, low_memory=False)
print(f"Loaded harmonized corpus: {len(harmonized_df)} rows")

# ─── Re-load raw datasets to rebuild manifests / logs ──────────────────────
print("Rebuilding manifests from raw data ...")

manifest_rows  = []
exclusion_rows = []
transformation_log = []

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

# ─── EmoNoBa ───────────────────────────────────────────────────────────────
emonoba_splits = {
    "Train": os.path.join(RAW_DIR, "Emonoba", "Train.csv"),
    "Val":   os.path.join(RAW_DIR, "Emonoba", "Val.csv"),
    "Test":  os.path.join(RAW_DIR, "Emonoba", "Test.csv"),
}
love_only_count = 0
included_emonoba = 0

for split_name, fpath in emonoba_splits.items():
    df = pd.read_csv(fpath)
    for _, row in df.iterrows():
        src_id = int(row["ID"])
        text   = str(row["Data"])
        orig_love     = int(row["Love"])
        orig_joy      = int(row["Joy"])
        orig_surprise = int(row["Surprise"])
        orig_anger    = int(row["Anger"])
        orig_sadness  = int(row["Sadness"])
        orig_fear     = int(row["Fear"])
        target_compat_sum = orig_joy + orig_surprise + orig_anger + orig_sadness + orig_fear
        is_love_only = (orig_love == 1) and (target_compat_sum == 0)
        orig_label_str = f"Love={orig_love} Joy={orig_joy} Surprise={orig_surprise} Anger={orig_anger} Sadness={orig_sadness} Fear={orig_fear}"
        if is_love_only:
            love_only_count += 1
            reason = "LOVE-1: Love is the only active native label; excluded after Love excluded from target taxonomy"
            manifest_rows.append({
                "source_dataset":  "EmoNoBa", "source_split": split_name,
                "source_row_id":   src_id,
                "text_snippet":    text[:80],
                "included":        False,
                "exclusion_reason": reason,
                "harmonization_rule": "LOVE-1",
                "original_labels": orig_label_str,
                "harmonized_labels": "EXCLUDED"
            })
            exclusion_rows.append({
                "source_dataset": "EmoNoBa", "source_split": split_name,
                "source_row_id":  src_id, "text_snippet": text[:80],
                "exclusion_reason": reason, "harmonization_rule": "LOVE-1",
                "original_Love": orig_love, "original_Joy": orig_joy,
                "original_Surprise": orig_surprise, "original_Anger": orig_anger,
                "original_Sadness": orig_sadness, "original_Fear": orig_fear
            })
        else:
            included_emonoba += 1
            harmonized = {"anger": orig_anger, "disgust": 0, "fear": orig_fear,
                          "joy": orig_joy, "sadness": orig_sadness, "surprise": orig_surprise}
            manifest_rows.append({
                "source_dataset": "EmoNoBa", "source_split": split_name,
                "source_row_id":  src_id, "text_snippet": text[:80],
                "included":       True, "exclusion_reason": "",
                "harmonization_rule": "DIRECT+DISGUST-D",
                "original_labels": orig_label_str,
                "harmonized_labels": " ".join(f"{k}={v}" for k,v in harmonized.items())
            })

log_transform("EmoNoBa", "DIRECT column rename",
              "Joy/Anger/Sadness/Surprise/Fear", "joy/anger/sadness/surprise/fear",
              included_emonoba, "Phase 3 FROZEN direct mappings",
              "Capitalization-only difference; exact semantic equivalence.")
log_transform("EmoNoBa", "STRATEGY-D: disgust=0 annotation-coverage assumption",
              "(absent — not annotated)", "disgust=0", included_emonoba,
              "DISGUST-D (approved 2026-09-19)",
              "EmoNoBa did not collect Disgust annotations. disgust=0 is an operational assumption, NOT an original annotation. F5 sensitivity analysis is mandatory. Rows flagged as f5_eligible=True.")
log_transform("EmoNoBa", "LOVE-1 exclusion", "Love (only active label)", "EXCLUDED",
              love_only_count, "LOVE-1 (approved 2026-09-19)",
              "2277 rows excluded. Love-only rows would create all-zero target vectors, misrepresenting unannotated status as confirmed negatives.")
log_transform("EmoNoBa", "Love column dropped (non-target)", "Love", "DROPPED",
              included_emonoba, "DATASET_SPEC.md Sec 3 FROZEN",
              "Love is not in the BETC target taxonomy. Not mapped to any target label. Never mapped to Disgust.")

# ─── UBMEC ─────────────────────────────────────────────────────────────────
ubmec_df = pd.read_excel(os.path.join(RAW_DIR, "UBMEC Corpus_Sakib(updated).xlsx"), sheet_name="UBMEC")
valid_ubmec_classes = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
included_ubmec = 0
excluded_ubmec_unexpected = 0

for row_idx, row in ubmec_df.iterrows():
    text       = str(row["text"])
    orig_class = str(row["classes"]).strip().lower()
    orig_label_str = f"classes={orig_class}"
    if orig_class not in valid_ubmec_classes:
        excluded_ubmec_unexpected += 1
        reason = f"UBMEC-UNEXPECTED-CLASS: '{orig_class}' is not in the 6 target classes"
        manifest_rows.append({
            "source_dataset": "UBMEC", "source_split": "NONE",
            "source_row_id": row_idx, "text_snippet": text[:80],
            "included": False, "exclusion_reason": reason,
            "harmonization_rule": "UBMEC-UNEXPECTED-CLASS",
            "original_labels": orig_label_str, "harmonized_labels": "EXCLUDED"
        })
        exclusion_rows.append({
            "source_dataset": "UBMEC", "source_split": "NONE",
            "source_row_id": row_idx, "text_snippet": text[:80],
            "exclusion_reason": reason, "harmonization_rule": "UBMEC-UNEXPECTED-CLASS",
            "original_class": orig_class
        })
    else:
        included_ubmec += 1
        harmonized = {c: (1 if c == orig_class else 0) for c in TARGET_COLS}
        manifest_rows.append({
            "source_dataset": "UBMEC", "source_split": "NONE",
            "source_row_id": row_idx, "text_snippet": text[:80],
            "included": True, "exclusion_reason": "",
            "harmonization_rule": "UBMEC-ONE-HOT",
            "original_labels": orig_label_str,
            "harmonized_labels": " ".join(f"{k}={v}" for k,v in harmonized.items())
        })

log_transform("UBMEC", "Single-label to one-hot six-binary conversion",
              "classes (categorical string)", "six binary target columns",
              included_ubmec, "UBMEC-ONE-HOT (FROZEN)",
              "UBMEC is single-label; observed class -> 1, all other 5 -> 0. This is a representation change, NOT a claim that UBMEC is multi-label. The 0s mean 'not the observed category in a mutually-exclusive 6-class scheme'. Original categorical class preserved in original_class column.")

# ─── MONOVAB ───────────────────────────────────────────────────────────────
monovab_df = pd.read_csv(os.path.join(RAW_DIR, "MONOVAB (1).csv"))
contempt_only_count = 0
included_monovab = 0
enjoyment_mapped = 0

for _, row in monovab_df.iterrows():
    src_id       = int(row["Unnamed: 0"])
    text         = str(row["comment"])
    orig_anger   = int(row["anger"])
    orig_contempt= int(row["contempt"])
    orig_disgust = int(row["disgust"])
    orig_enjoy   = int(row["enjoyment"])
    orig_fear    = int(row["fear"])
    orig_sadness = int(row["sadness"])
    orig_surprise= int(row["surprise"])
    other_sum    = orig_anger + orig_disgust + orig_enjoy + orig_fear + orig_sadness + orig_surprise
    is_contempt_only = (orig_contempt == 1) and (other_sum == 0)
    orig_label_str = (f"anger={orig_anger} contempt={orig_contempt} disgust={orig_disgust} "
                      f"enjoyment={orig_enjoy} fear={orig_fear} sadness={orig_sadness} surprise={orig_surprise}")
    if is_contempt_only:
        contempt_only_count += 1
        reason = ("CONTEMPT-1: contempt is the only active native label; excluded. Retaining as all-zero would "
                  "risk disgust=0 label noise given semantic contempt-disgust overlap.")
        manifest_rows.append({
            "source_dataset": "MONOVAB", "source_split": "NONE",
            "source_row_id": src_id, "text_snippet": text[:80],
            "included": False, "exclusion_reason": reason,
            "harmonization_rule": "CONTEMPT-1 (approved 2026-09-19)",
            "original_labels": orig_label_str, "harmonized_labels": "EXCLUDED"
        })
        exclusion_rows.append({
            "source_dataset": "MONOVAB", "source_split": "NONE",
            "source_row_id": src_id, "text_snippet": text[:80],
            "exclusion_reason": reason, "harmonization_rule": "CONTEMPT-1",
            "original_anger": orig_anger, "original_contempt": orig_contempt,
            "original_disgust": orig_disgust, "original_enjoyment": orig_enjoy,
            "original_fear": orig_fear, "original_sadness": orig_sadness,
            "original_surprise": orig_surprise
        })
    else:
        included_monovab += 1
        if orig_enjoy == 1:
            enjoyment_mapped += 1
        harmonized = {
            "anger": orig_anger, "disgust": orig_disgust, "fear": orig_fear,
            "joy": orig_enjoy,    # enjoyment -> joy FROZEN mapping
            "sadness": orig_sadness, "surprise": orig_surprise,
        }
        manifest_rows.append({
            "source_dataset": "MONOVAB", "source_split": "NONE",
            "source_row_id": src_id, "text_snippet": text[:80],
            "included": True, "exclusion_reason": "",
            "harmonization_rule": "DIRECT+ENJOYMENT-JOY+CONTEMPT-DROP",
            "original_labels": orig_label_str,
            "harmonized_labels": " ".join(f"{k}={v}" for k,v in harmonized.items())
        })

log_transform("MONOVAB", "CONTEMPT-1 exclusion", "contempt (only active label)", "EXCLUDED",
              contempt_only_count, "CONTEMPT-1 (approved 2026-09-19)",
              "2128 rows excluded. Retaining as all-zero would create disgust=0 label noise given semantic contempt-disgust overlap.")
log_transform("MONOVAB", "Contempt column dropped (non-target)", "contempt", "DROPPED",
              included_monovab, "DATASET_SPEC.md Sec 3 FROZEN",
              "contempt is not in the BETC target taxonomy.")
log_transform("MONOVAB", "ENJOYMENT->JOY semantic harmonization (FROZEN)",
              "enjoyment", "joy", enjoyment_mapped, "enjoyment->joy FROZEN (approved 2026-09-19)",
              "Semantic near-equivalence: enjoyment is MONOVAB's Ekman Joy/Happiness category. "
              "Mapping is a semantic harmonization, NOT a claim of theoretical label identity. "
              "original_enjoyment preserved in harmonized corpus for traceability.")
log_transform("MONOVAB", "DIRECT mappings",
              "anger/disgust/fear/sadness/surprise", "anger/disgust/fear/sadness/surprise",
              included_monovab, "Phase 3 FROZEN direct mappings", "Five MONOVAB labels are exact Ekman equivalents.")

# ─── Save exclusion log ─────────────────────────────────────────────────────
excl_out = os.path.join(LOG_DIR, "exclusion_log.csv")
all_excl_cols = sorted({k for r in exclusion_rows for k in r.keys()})
with open(excl_out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=all_excl_cols, extrasaction="ignore")
    w.writeheader()
    for r in exclusion_rows:
        w.writerow({k: r.get(k, "") for k in all_excl_cols})
print(f"Saved exclusion_log.csv: {len(exclusion_rows)} rows")

# ─── Save transformation log ─────────────────────────────────────────────────
tf_out = os.path.join(LOG_DIR, "transformation_log.csv")
with open(tf_out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(transformation_log[0].keys()))
    w.writeheader()
    w.writerows(transformation_log)
print(f"Saved transformation_log.csv: {len(transformation_log)} entries")

# ─── Save manifest ───────────────────────────────────────────────────────────
manifest_out = os.path.join(LOG_DIR, "harmonization_manifest.csv")
manifest_cols = ["source_dataset","source_split","source_row_id","included",
                 "exclusion_reason","harmonization_rule","original_labels",
                 "harmonized_labels","text_snippet"]
with open(manifest_out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=manifest_cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(manifest_rows)
print(f"Saved harmonization_manifest.csv: {len(manifest_rows)} rows")

# ─── Compute final stats and validation from harmonized_df ───────────────────
checks = {}
for col in TARGET_COLS:
    unique_vals = sorted(harmonized_df[col].dropna().unique().tolist())
    ok = all(v in [0, 1] for v in unique_vals)
    checks[f"target_col_{col}_binary_only"] = {"pass": ok, "unique_values": unique_vals}

null_text = int(harmonized_df["text"].isna().sum())
checks["no_null_text"] = {"pass": null_text == 0, "null_count": null_text}
source_vals = list(harmonized_df["source_dataset"].unique())
checks["source_dataset_valid"] = {"pass": set(source_vals) == {"EmoNoBa","UBMEC","MONOVAB"}, "found": source_vals}

e_rows = harmonized_df[harmonized_df["source_dataset"] == "EmoNoBa"]
checks["emonoba_disgust_assumption_flag"] = {
    "pass": bool(e_rows["emonoba_disgust_assumption"].all()),
    "count_true": int(e_rows["emonoba_disgust_assumption"].sum()),
    "total_emonoba": len(e_rows)
}
non_e = harmonized_df[harmonized_df["source_dataset"] != "EmoNoBa"]
false_flags = int(non_e["emonoba_disgust_assumption"].sum())
checks["non_emonoba_disgust_assumption_false"] = {"pass": false_flags == 0, "false_positives": false_flags}

all_zero_mask = (harmonized_df[TARGET_COLS].sum(axis=1) == 0)
all_zero_count = int(all_zero_mask.sum())
checks["all_zero_rows"] = {"count": all_zero_count, "pct": round(all_zero_count/len(harmonized_df)*100,2)}
if all_zero_count > 0:
    checks["all_zero_by_source"] = harmonized_df[all_zero_mask]["source_dataset"].value_counts().to_dict()

ubmec_rows = harmonized_df[harmonized_df["source_dataset"] == "UBMEC"]
ubmec_not_one = int((ubmec_rows[TARGET_COLS].sum(axis=1) != 1).sum())
checks["ubmec_one_hot_valid"] = {"pass": ubmec_not_one == 0, "rows_not_exactly_one": ubmec_not_one}

rc = harmonized_df["source_dataset"].value_counts().to_dict()
rc = {k: int(v) for k,v in rc.items()}
checks["row_counts_by_source"] = rc

label_stats = {}
for col in TARGET_COLS:
    total_pos = int(harmonized_df[col].sum())
    by_source = {src: int(harmonized_df[harmonized_df["source_dataset"]==src][col].sum())
                 for src in ["EmoNoBa","UBMEC","MONOVAB"]}
    prevalence = round(total_pos/len(harmonized_df)*100, 2)
    label_stats[col] = {"total_positive": total_pos, "total_rows": len(harmonized_df),
                        "prevalence_pct": prevalence, "positive_by_source": by_source}
    if col == "disgust":
        true_disgust_ann = by_source["UBMEC"] + by_source["MONOVAB"]
        label_stats[col]["note"] = (f"EmoNoBa disgust positive = {by_source['EmoNoBa']} (expected 0 under Strategy D). "
                                    f"True annotated Disgust positives UBMEC+MONOVAB = {true_disgust_ann}. "
                                    f"All EmoNoBa rows have disgust=0 by assumption (f5_eligible=True).")

label_sums = harmonized_df[TARGET_COLS].sum(axis=1)
cardinality_dist = {int(k): int(v) for k,v in label_sums.value_counts().sort_index().items()}
mean_labels = round(float(label_sums.mean()), 4)

# Check no Love-only rows slipped through
emonoba_inc = harmonized_df[harmonized_df["source_dataset"]=="EmoNoBa"]
potential_loveonly = int(
    ((emonoba_inc["original_love"]==1) & (emonoba_inc[TARGET_COLS].sum(axis=1)==0)).sum()
)
checks["no_love_only_in_included"] = {"pass": potential_loveonly==0, "count": potential_loveonly}

final_stats = {
    "run_timestamp": RUN_TS,
    "approved_decisions": {
        "emonoba_disgust": "STRATEGY-D: disgust=0 operational assumption; F5 mandatory",
        "emonoba_love_only": "LOVE-1: 2277 rows excluded",
        "monovab_contempt_only": "CONTEMPT-1: 2128 rows excluded",
        "monovab_enjoyment_joy": "FROZEN: enjoyment->joy semantic harmonization",
        "ubmec_conversion": "FROZEN: single-label->one-hot six-binary",
        "target_taxonomy": ["anger","disgust","fear","joy","sadness","surprise"]
    },
    "row_counts": {
        "emonoba_raw": 22739,
        "emonoba_excluded_love_only": love_only_count,
        "emonoba_included": included_emonoba,
        "ubmec_raw": len(ubmec_df),
        "ubmec_excluded_unexpected_class": excluded_ubmec_unexpected,
        "ubmec_included": included_ubmec,
        "monovab_raw": len(monovab_df),
        "monovab_excluded_contempt_only": contempt_only_count,
        "monovab_included": included_monovab,
        "total_raw": 22739 + len(ubmec_df) + len(monovab_df),
        "total_excluded": love_only_count + excluded_ubmec_unexpected + contempt_only_count,
        "total_harmonized_pre_dedup": len(harmonized_df)
    },
    "exclusion_reasons_summary": {
        "LOVE-1 (EmoNoBa Love-only)": love_only_count,
        "CONTEMPT-1 (MONOVAB Contempt-only)": contempt_only_count,
        "UBMEC-UNEXPECTED-CLASS": excluded_ubmec_unexpected
    },
    "validation_checks": checks,
    "label_statistics": label_stats,
    "cardinality_distribution": cardinality_dist,
    "mean_labels_per_example": mean_labels,
    "enjoyment_to_joy_mapped_rows": enjoyment_mapped,
    "f5_eligible_rows": int(harmonized_df["f5_eligible"].sum()),
    "emonoba_disgust_assumption_rows": int(harmonized_df["emonoba_disgust_assumption"].sum()),
    "note_deduplication": "NOT YET APPLIED. This is the pre-deduplication harmonized corpus. Deduplication is Phase 5.",
    "note_f5": "EmoNoBa rows flagged f5_eligible=True and emonoba_disgust_assumption=True. These rows must be identifiable and excludable for Experiment F5 sensitivity analysis on Disgust performance.",
    "note_reproducibility": "Raw data in Data/raw/ is unchanged. To reproduce this corpus: re-run phase4_harmonize.py with the same approved decisions. The harmonization_manifest.csv records every row decision."
}

stats_out = os.path.join(LOG_DIR, "harmonization_validation.json")
with open(stats_out, "w", encoding="utf-8") as f:
    json.dump(final_stats, f, indent=2, ensure_ascii=False)
print(f"Saved harmonization_validation.json")

# ─── Print summary ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 4 COMPLETE — Summary")
print("="*60)
print(f"\nHarmonized corpus (pre-dedup): {len(harmonized_df):,} rows")
print(f"\nRow counts per source:")
for src, cnt in rc.items():
    print(f"  {src}: {cnt:,}")
print(f"\nExcluded rows:")
print(f"  EmoNoBa Love-only  (LOVE-1):     {love_only_count:,}")
print(f"  MONOVAB Contempt-only (CONTEMPT-1): {contempt_only_count:,}")
print(f"  UBMEC unexpected class:            {excluded_ubmec_unexpected}")
print(f"  TOTAL excluded: {love_only_count+contempt_only_count+excluded_ubmec_unexpected:,}")
print(f"\nLabel statistics (positive examples):")
for col in TARGET_COLS:
    s = label_stats[col]
    print(f"  {col:8s}: {s['total_positive']:6,} ({s['prevalence_pct']:5.2f}%) | "
          f"EmoNoBa={s['positive_by_source']['EmoNoBa']} "
          f"UBMEC={s['positive_by_source']['UBMEC']} "
          f"MONOVAB={s['positive_by_source']['MONOVAB']}")
print(f"\nCardinality distribution (active labels per example):")
for k,v in cardinality_dist.items():
    print(f"  {k} active label(s): {v:,} rows")
print(f"  Mean: {mean_labels}")
print(f"\nAll-zero rows: {all_zero_count}")
print(f"UBMEC one-hot violations: {ubmec_not_one}")
print(f"F5-eligible (EmoNoBa) rows: {final_stats['f5_eligible_rows']:,}")
print(f"enjoyment->joy mapped rows: {enjoyment_mapped:,}")
print(f"\nAll validation checks passed: {all(v.get('pass', True) for v in checks.values() if 'pass' in v)}")
print(f"\nDo NOT proceed to deduplication without explicit user approval (Phase 5).")
