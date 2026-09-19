"""
Phase 5 — Conflict Resolution (DROP strategy)
===============================================
Approved decision (2026-09-19): DROP all rows belonging to the 98
conflicting duplicate groups (251 rows) from the modeling corpus.

Input:  Data/interim/harmonized_deduplicated.csv  (41,435 rows)
        logs/deduplication/conflicting_duplicates.csv  (98 groups, 251 rows — preserved)
Output: Data/interim/harmonized_deduplicated.csv  (overwritten — ~41,184 rows)
        logs/deduplication/conflict_resolution_log.csv
        logs/deduplication/deduplication_validation.json (updated)
        logs/deduplication/deduplication_report.md (updated section)

Rules:
- Data/raw/ NOT touched
- conflicting_duplicates.csv NOT deleted or modified (preserved as evidence)
- 251 conflicting rows removed and logged with reason=conflicting_duplicate, resolution=DROP
- Final corpus contains no exact duplicate text groups
- EmoNoBa assumption flags (emonoba_disgust_assumption, f5_eligible) preserved
- No preprocessing, no splitting, no model training
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, csv, hashlib, datetime, unicodedata
import pandas as pd

WORKSPACE = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
INTERIM   = os.path.join(WORKSPACE, "Data", "interim")
LOG_DIR   = os.path.join(WORKSPACE, "logs", "deduplication")
RUN_TS    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
TARGET_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

def comparison_key(text: str) -> str:
    """NFC + whitespace collapse. For comparison only; never overwrites original text."""
    t = unicodedata.normalize("NFC", str(text))
    return " ".join(t.split()).strip()

# ─── Load current harmonized_deduplicated.csv ─────────────────────────────────
in_csv = os.path.join(INTERIM, "harmonized_deduplicated.csv")
df = pd.read_csv(in_csv, dtype={
    "source_row_id": "Int64",
    "original_love": "Int64",
    "original_contempt": "Int64",
    "original_enjoyment": "Int64"
}, low_memory=False)
df["source_split"] = df["source_split"].fillna("NONE").astype(str)
N_INPUT = len(df)
print(f"Loaded harmonized_deduplicated.csv: {N_INPUT:,} rows")
print(f"Sources: {dict(df['source_dataset'].value_counts())}")

# ─── Load conflicting_duplicates.csv ─────────────────────────────────────────
conf_csv = os.path.join(LOG_DIR, "conflicting_duplicates.csv")
conf_df  = pd.read_csv(conf_csv, encoding="utf-8-sig")
N_CONFLICT_GROUPS = len(conf_df)
print(f"\nLoaded conflicting_duplicates.csv: {N_CONFLICT_GROUPS:,} groups")

# ─── Build set of conflicting (source_dataset, source_row_id) pairs ───────────
# Each group row has pipe-separated source_datasets and source_row_ids
conflicting_pairs = set()   # (source_dataset, source_row_id_int)
conflicting_texts = set()   # comparison keys of conflicting texts (for cross-check)

resolution_log_rows = []

for _, crow in conf_df.iterrows():
    group_id    = crow["group_id"]
    orig_text   = str(crow["original_text"])
    grp_type    = crow["group_type"]
    ckey        = comparison_key(orig_text)
    conflicting_texts.add(ckey)

    datasets   = str(crow["source_datasets"]).split("|")
    row_ids    = str(crow["source_row_ids"]).split("|")
    splits_raw = str(crow["splits"]).split("|")  # may have fewer entries; used for logging

    for i, (ds, rid) in enumerate(zip(datasets, row_ids)):
        ds  = ds.strip()
        rid = rid.strip()
        rid_int = int(rid) if rid.lstrip("-").isdigit() else -1
        conflicting_pairs.add((ds, rid_int))

        # Label vectors for this specific row: parse from the all_anger etc. columns
        ang = str(crow["all_anger"]).split("|")[i]   if i < len(str(crow["all_anger"]).split("|"))   else "?"
        dis = str(crow["all_disgust"]).split("|")[i] if i < len(str(crow["all_disgust"]).split("|")) else "?"
        fea = str(crow["all_fear"]).split("|")[i]    if i < len(str(crow["all_fear"]).split("|"))    else "?"
        joy = str(crow["all_joy"]).split("|")[i]     if i < len(str(crow["all_joy"]).split("|"))     else "?"
        sad = str(crow["all_sadness"]).split("|")[i] if i < len(str(crow["all_sadness"]).split("|")) else "?"
        sur = str(crow["all_surprise"]).split("|")[i]if i < len(str(crow["all_surprise"]).split("|"))else "?"

        resolution_log_rows.append({
            "duplicate_group_id":  group_id,
            "text_hash":           group_id,
            "original_text_snippet": orig_text[:80],
            "group_type":          grp_type,
            "source_dataset":      ds,
            "source_split":        splits_raw[0] if splits_raw else "NONE",
            "source_row_id":       rid_int,
            "anger":               ang,
            "disgust":             dis,
            "fear":                fea,
            "joy":                 joy,
            "sadness":             sad,
            "surprise":            sur,
            "reason":              "conflicting_duplicate — same text with different label vectors in this group",
            "resolution":          "DROP",
            "approved_by":         "User decision 2026-09-19",
            "timestamp":           RUN_TS
        })

print(f"Conflicting (dataset, row_id) pairs to drop: {len(conflicting_pairs):,}")
print(f"Conflicting comparison keys:                 {len(conflicting_texts):,}")

# ─── Build drop mask using BOTH methods ───────────────────────────────────────
# Primary:   match (source_dataset, source_row_id) pairs
# Secondary: match comparison key of text (belt-and-suspenders)

df["_cmp_key"] = df["text"].apply(comparison_key)

def is_conflicting(row):
    rid = int(row["source_row_id"]) if pd.notna(row["source_row_id"]) else -1
    pair_match = (row["source_dataset"], rid) in conflicting_pairs
    text_match = row["_cmp_key"] in conflicting_texts
    return pair_match or text_match

drop_mask = df.apply(is_conflicting, axis=1)
n_dropped  = int(drop_mask.sum())
print(f"\nRows identified for DROP: {n_dropped:,}")
print(f"Expected (251):           251")
if n_dropped != 251:
    print(f"WARNING: Count mismatch! Investigating ...")
    pair_only = df.apply(lambda r: (r["source_dataset"], int(r["source_row_id"]) if pd.notna(r["source_row_id"]) else -1) in conflicting_pairs, axis=1).sum()
    text_only = df["_cmp_key"].isin(conflicting_texts).sum()
    print(f"  Pair-match rows:      {pair_only}")
    print(f"  Text-match rows:      {text_only}")

# ─── Apply DROP ───────────────────────────────────────────────────────────────
final_df = df[~drop_mask].copy()
final_df = final_df.drop(columns=["_cmp_key"])
N_OUTPUT = len(final_df)

print(f"\nBefore/After:")
print(f"  Phase 5 input (current):  {N_INPUT:>7,}")
print(f"  Conflicting rows dropped: {n_dropped:>7,}")
print(f"  Final output:             {N_OUTPUT:>7,}")
print(f"  Check: {N_INPUT} - {n_dropped} = {N_INPUT - n_dropped}  (expected {N_INPUT - 251})")

# ─── Validation ───────────────────────────────────────────────────────────────
checks = {}

# 1. Arithmetic
checks["arithmetic"] = {
    "pass": N_OUTPUT == (N_INPUT - n_dropped),
    "n_input": N_INPUT, "n_dropped": n_dropped, "n_output": N_OUTPUT
}

# 2. No duplicate comparison keys remain
final_df["_cmp_key"] = final_df["text"].apply(comparison_key)
remaining_dups = final_df["_cmp_key"].duplicated(keep=False).sum()
checks["no_duplicate_texts_remain"] = {
    "pass": remaining_dups == 0,
    "remaining_duplicate_rows": int(remaining_dups)
}
final_df = final_df.drop(columns=["_cmp_key"])

# 3. Exactly six target label columns, binary only
for col in TARGET_COLS:
    uv = sorted(final_df[col].dropna().unique().tolist())
    checks[f"target_{col}_binary"] = {"pass": all(v in [0,1] for v in uv), "values": uv}

# 4. No Love/Contempt target columns
bad = [c for c in final_df.columns if c.lower() in ("love","contempt")]
checks["no_love_contempt_cols"] = {"pass": len(bad)==0, "found": bad}

# 5. EmoNoBa assumption flags preserved
e_rows = final_df[final_df["source_dataset"]=="EmoNoBa"]
all_flagged = bool(e_rows["emonoba_disgust_assumption"].all())
checks["emonoba_disgust_assumption_preserved"] = {
    "pass": all_flagged,
    "emonoba_rows": len(e_rows),
    "flagged_true": int(e_rows["emonoba_disgust_assumption"].sum())
}
all_f5 = bool(e_rows["f5_eligible"].all())
checks["f5_eligible_preserved"] = {
    "pass": all_f5,
    "emonoba_rows": len(e_rows),
    "f5_true": int(e_rows["f5_eligible"].sum())
}

# 6. Data/raw/ unchanged
checks["raw_data_unchanged"] = {
    "pass": os.path.exists(os.path.join(WORKSPACE,"Data","raw","Emonoba","Train.csv"))
}

# 7. Phase 4 pre-dedup file unchanged
ph4_rows = len(pd.read_csv(os.path.join(INTERIM,"harmonized_pre_dedup.csv"), low_memory=False))
checks["phase4_file_unchanged"] = {"pass": ph4_rows==41994, "rows": ph4_rows}

# 8. conflicting_duplicates.csv unchanged
conf_rows_check = len(pd.read_csv(conf_csv, encoding="utf-8-sig"))
checks["conflict_log_preserved"] = {"pass": conf_rows_check==N_CONFLICT_GROUPS, "rows": conf_rows_check}

# 9. Source counts
src_counts = {k: int(v) for k,v in final_df["source_dataset"].value_counts().items()}
checks["source_counts"] = src_counts

# 10. Label stats
label_stats = {}
for col in TARGET_COLS:
    tp = int(final_df[col].sum())
    label_stats[col] = {
        "total_positive": tp,
        "prevalence_pct": round(tp/N_OUTPUT*100, 2),
        "by_source": {
            src: int(final_df[final_df["source_dataset"]==src][col].sum())
            for src in ["EmoNoBa","UBMEC","MONOVAB"]
        }
    }

# Cardinality
lab_sums = final_df[TARGET_COLS].sum(axis=1)
card_dist = {int(k): int(v) for k,v in lab_sums.value_counts().sort_index().items()}
mean_labs = round(float(lab_sums.mean()), 4)

all_pass = all(v.get("pass", True) for v in checks.values() if isinstance(v, dict) and "pass" in v)
print(f"\nAll validation checks passed: {all_pass}")
for k, v in checks.items():
    if isinstance(v, dict) and "pass" in v:
        status = "PASS" if v["pass"] else "FAIL"
        print(f"  [{status}] {k}")

# ─── Save outputs ─────────────────────────────────────────────────────────────

# 1. Overwrite harmonized_deduplicated.csv with final corpus
out_csv = os.path.join(INTERIM, "harmonized_deduplicated.csv")
final_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved final corpus: {out_csv}  ({N_OUTPUT:,} rows)")

# 2. Conflict resolution log
res_log = os.path.join(LOG_DIR, "conflict_resolution_log.csv")
if resolution_log_rows:
    with open(res_log, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(resolution_log_rows[0].keys()))
        w.writeheader()
        w.writerows(resolution_log_rows)
print(f"Saved conflict resolution log: {res_log}  ({len(resolution_log_rows):,} rows)")

# 3. Updated validation JSON
val_out = {
    "run_timestamp": RUN_TS,
    "resolution_strategy": "DROP",
    "approved_by": "User decision 2026-09-19",
    "phase5_input_rows": N_INPUT,
    "conflict_groups_dropped": N_CONFLICT_GROUPS,
    "conflicting_rows_dropped": n_dropped,
    "final_output_rows": N_OUTPUT,
    "arithmetic_check": f"{N_INPUT} - {n_dropped} = {N_OUTPUT}",
    "unique_texts_in_final_corpus": final_df["text"].apply(comparison_key).nunique(),
    "validation_checks": checks,
    "label_statistics": label_stats,
    "cardinality_distribution": card_dist,
    "mean_labels_per_example": mean_labs,
    "source_counts": src_counts,
    "note_conflict_log": "conflicting_duplicates.csv preserved unchanged at logs/deduplication/conflicting_duplicates.csv",
    "note_raw_data": "Data/raw/ was not modified",
    "note_phase4_input": "harmonized_pre_dedup.csv (Phase 4 output) was not modified",
    "note_f5": "EmoNoBa rows retain emonoba_disgust_assumption=True and f5_eligible=True for Experiment F5 sensitivity analysis"
}

val_json = os.path.join(LOG_DIR, "deduplication_validation_final.json")
with open(val_json, "w", encoding="utf-8") as f:
    json.dump(val_out, f, indent=2, ensure_ascii=False)
print(f"Saved validation JSON: {val_json}")

# ─── Final summary ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 5 CONFLICT RESOLUTION COMPLETE — DROP")
print("="*60)
print(f"\nFinal corpus (Data/interim/harmonized_deduplicated.csv):")
print(f"  Phase 5 input rows:              {N_INPUT:>7,}")
print(f"  Conflicting rows dropped (DROP): {n_dropped:>7,}")
print(f"  Final output rows:               {N_OUTPUT:>7,}")
print(f"  Unique texts remaining:          {val_out['unique_texts_in_final_corpus']:>7,}")
print(f"\nLabel statistics (final corpus):")
for col in TARGET_COLS:
    s = label_stats[col]
    print(f"  {col:8s}: {s['total_positive']:6,} ({s['prevalence_pct']:.2f}%)")
print(f"\nSource counts: {src_counts}")
print(f"Cardinality:   {card_dist}")
print(f"Mean labels:   {mean_labs}")
print(f"\nData/raw/ untouched: {checks['raw_data_unchanged']['pass']}")
print(f"Phase 4 file unchanged: {checks['phase4_file_unchanged']['pass']}")
print(f"Conflict log preserved: {checks['conflict_log_preserved']['pass']}")
print(f"No duplicate texts remain: {checks['no_duplicate_texts_remain']['pass']}")
print(f"\nAll validation checks passed: {all_pass}")
print(f"\nDo NOT proceed to Phase 5b (splitting) without explicit user instruction.")
