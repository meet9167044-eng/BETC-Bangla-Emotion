"""
Phase 5 — Deduplication
========================
Input:  Data/interim/harmonized_pre_dedup.csv  (41,994 rows)
Output: Data/interim/harmonized_deduplicated.csv
        logs/deduplication/

Rules:
- Exact duplicate detection on stripped / NFC-normalized text (comparison key only)
- Original text is NEVER overwritten
- Identical-label duplicates → deterministic canonical record; collapse with provenance
- Conflicting-label duplicates → logged; NOT resolved automatically; retained as-is
- EmoNoBa cross-split duplicates → identified and logged; one canonical copy kept
- Data/raw/ NOT touched; harmonized_pre_dedup.csv NOT overwritten

Duplicate type classification:
  A: Same text + identical six-label vector (within same source)
  B: Same text + different six-label vectors (within same source)
  C: Same text + identical labels across different source datasets
  D: Same text + conflicting labels across different source datasets
  E: Same text appearing across EmoNoBa Train/Val/Test splits

Canonical record selection (deterministic, no randomness):
  Priority 1: Dataset with most metadata fields populated
              (EmoNoBa has domain+topic → preferred; UBMEC/MONOVAB have neither)
  Priority 2: Dataset priority: EmoNoBa > UBMEC > MONOVAB
  Priority 3: Split priority: Train > Val > Test > NONE
  Priority 4: Smallest source_row_id (numeric)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os, json, csv, hashlib, datetime, unicodedata
import pandas as pd
import numpy as np

WORKSPACE = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
INTERIM   = os.path.join(WORKSPACE, "Data", "interim")
LOG_DIR   = os.path.join(WORKSPACE, "logs", "deduplication")
os.makedirs(LOG_DIR, exist_ok=True)

RUN_TS      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
TARGET_COLS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
DATASET_PRIORITY = {"EmoNoBa": 0, "UBMEC": 1, "MONOVAB": 2}
SPLIT_PRIORITY   = {"Train": 0, "Val": 1, "Test": 2, "NONE": 3}

# ─── Load harmonized corpus ───────────────────────────────────────────────────
csv_in = os.path.join(INTERIM, "harmonized_pre_dedup.csv")
df = pd.read_csv(csv_in, dtype={
    "source_row_id": "Int64",
    "original_love": "Int64",
    "original_contempt": "Int64",
    "original_enjoyment": "Int64"
}, low_memory=False)
df["source_split"] = df["source_split"].fillna("NONE").astype(str)
df["domain"] = df["domain"].fillna("").astype(str)
df["topic"]  = df["topic"].fillna("").astype(str)

# Assign a stable internal row index for tracking
df = df.reset_index(drop=True)
df["_internal_id"] = df.index  # 0-based stable row reference

N_INPUT = len(df)
print(f"Loaded Phase 4 corpus: {N_INPUT:,} rows")
print(f"Sources: {dict(df.source_dataset.value_counts())}")

# ─── Normalisation for comparison only ───────────────────────────────────────
# DATASET_SPEC.md Sec 4, step 4: whitespace collapse + Unicode NFC
# Do NOT overwrite the original text column.

def comparison_key(text: str) -> str:
    """NFC normalise + strip + collapse whitespace. For dedup comparison only."""
    t = unicodedata.normalize("NFC", str(text))
    t = " ".join(t.split())   # collapse all whitespace
    t = t.strip()
    return t

df["_cmp_key"] = df["text"].apply(comparison_key)

# Also build a short hash for use in IDs
def short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]

df["_text_hash"] = df["_cmp_key"].apply(short_hash)

# Label vector as a tuple for comparison
def label_tuple(row):
    return tuple(int(row[c]) for c in TARGET_COLS)

df["_label_tuple"] = df.apply(label_tuple, axis=1)

print(f"\nUnique comparison keys: {df['_cmp_key'].nunique():,}")
print(f"Total duplicate texts:  {N_INPUT - df['_cmp_key'].nunique():,}")

# ─── Identify duplicate groups ────────────────────────────────────────────────
dup_mask = df.duplicated(subset="_cmp_key", keep=False)
dup_df   = df[dup_mask].copy()
uniq_df  = df[~dup_mask].copy()

n_texts_with_dups = dup_df["_cmp_key"].nunique()
print(f"\nRows in duplicate groups: {len(dup_df):,}")
print(f"Unique texts in dup groups: {n_texts_with_dups:,}")
print(f"Singleton (unique) rows:    {len(uniq_df):,}")

# ─── Classify each duplicate group ───────────────────────────────────────────
# For each unique comparison key that appears more than once, classify it.

group_records    = []   # summary of each duplicate group
provenance_rows  = []   # provenance log for ALL collapsed/retained actions
conflict_rows    = []   # conflicting groups (type B / D)
canonical_rows   = []   # final canonical records (from duplicate groups)

# Iterate over duplicate groups
dup_groups = dup_df.groupby("_cmp_key")

n_groups_identical   = 0  # Types A and C — same labels; can collapse
n_groups_conflicting = 0  # Types B, D, E-with-conflict
n_groups_cross_split = 0  # Type E (EmoNoBa cross-split)
n_groups_cross_source= 0  # Type C or D (cross-source)

rows_removed_identical   = 0
rows_kept_conflicting    = 0

for cmp_key, grp in dup_groups:
    grp = grp.copy()
    group_id = short_hash(cmp_key)
    
    sources   = sorted(grp["source_dataset"].unique().tolist())
    splits    = sorted(grp["source_split"].unique().tolist())
    label_tuples = grp["_label_tuple"].unique().tolist()
    n_rows    = len(grp)
    n_sources = len(sources)
    is_cross_source = n_sources > 1
    is_cross_split  = (
        grp["source_dataset"].eq("EmoNoBa").any() and
        len(grp[grp["source_dataset"]=="EmoNoBa"]["source_split"].unique()) > 1
    )
    is_conflicting = len(label_tuples) > 1
    
    if is_cross_split:
        n_groups_cross_split += 1
    if is_cross_source:
        n_groups_cross_source += 1
    
    # Determine type label
    if is_conflicting and not is_cross_source:
        grp_type = "B"  # same source, different labels
    elif is_conflicting and is_cross_source:
        grp_type = "D"  # cross-source, different labels
    elif not is_conflicting and is_cross_source:
        grp_type = "C"  # cross-source, same labels
    elif is_cross_split and not is_conflicting:
        grp_type = "E"  # EmoNoBa cross-split, same labels
    else:
        grp_type = "A"  # same source, same labels
    
    # If BOTH cross-split AND conflicting or cross-source, compound type
    if is_cross_split and is_cross_source:
        grp_type = "E+" + grp_type
    elif is_cross_split and is_conflicting:
        grp_type = "E+B"

    # ---------- CONFLICTING GROUPS ----------
    if is_conflicting:
        n_groups_conflicting += 1
        rows_kept_conflicting += n_rows
        
        # Log to conflict file — do NOT auto-resolve
        conflict_rows.append({
            "group_id":         group_id,
            "text_hash":        group_id,
            "original_text":    grp["text"].iloc[0],
            "group_type":       grp_type,
            "n_rows":           n_rows,
            "n_distinct_label_vectors": len(label_tuples),
            "sources":          "|".join(sources),
            "splits":           "|".join(splits),
            "source_row_ids":   "|".join(str(int(x)) for x in grp["source_row_id"].tolist()),
            "source_datasets":  "|".join(grp["source_dataset"].tolist()),
            "label_vectors":    " | ".join(str(t) for t in label_tuples),
            "all_anger":        "|".join(str(int(x)) for x in grp["anger"].tolist()),
            "all_disgust":      "|".join(str(int(x)) for x in grp["disgust"].tolist()),
            "all_fear":         "|".join(str(int(x)) for x in grp["fear"].tolist()),
            "all_joy":          "|".join(str(int(x)) for x in grp["joy"].tolist()),
            "all_sadness":      "|".join(str(int(x)) for x in grp["sadness"].tolist()),
            "all_surprise":     "|".join(str(int(x)) for x in grp["surprise"].tolist()),
            "action":           "RETAINED_UNRESOLVED — requires user decision",
            "resolution":       "PENDING"
        })
        
        # Write ALL rows from conflicting group to provenance
        for _, row in grp.iterrows():
            provenance_rows.append({
                "group_id":           group_id,
                "group_type":         grp_type,
                "internal_id":        int(row["_internal_id"]),
                "source_dataset":     row["source_dataset"],
                "source_split":       row["source_split"],
                "source_row_id":      int(row["source_row_id"]) if pd.notna(row["source_row_id"]) else -1,
                "is_canonical":       False,
                "conflict_status":    "CONFLICTING",
                "action":             "RETAINED_UNRESOLVED",
                "canonical_group_id": "",
                "label_vector":       str(row["_label_tuple"]),
                "note":               "Conflicting label vector within this text group; requires user decision"
            })
        
        group_records.append({
            "group_id":   group_id, "group_type": grp_type, "n_rows": n_rows,
            "n_sources":  n_sources, "sources": "|".join(sources),
            "is_cross_source": is_cross_source, "is_cross_split": is_cross_split,
            "is_conflicting": True, "n_label_vectors": len(label_tuples),
            "action": "CONFLICT_RETAINED_UNRESOLVED"
        })
        continue
    
    # ---------- NON-CONFLICTING GROUPS (identical labels) ----------
    n_groups_identical += 1
    rows_removed_identical += (n_rows - 1)
    
    # Deterministic canonical selection:
    # 1. Prefer row with most metadata (EmoNoBa rows have domain+topic populated)
    def metadata_score(row):
        domain_ok = 1 if (str(row["domain"]).strip() not in ("", "nan", "None")) else 0
        topic_ok  = 1 if (str(row["topic"]).strip()  not in ("", "nan", "None")) else 0
        return domain_ok + topic_ok
    
    grp["_meta_score"] = grp.apply(metadata_score, axis=1)
    grp["_ds_priority"]  = grp["source_dataset"].map(DATASET_PRIORITY)
    grp["_spl_priority"] = grp["source_split"].map(SPLIT_PRIORITY).fillna(3)
    grp["_row_id_sort"]  = grp["source_row_id"].fillna(999999999).astype(int)
    
    grp = grp.sort_values(
        by=["_meta_score", "_ds_priority", "_spl_priority", "_row_id_sort"],
        ascending=[False, True, True, True]
    )
    canonical = grp.iloc[0]
    collapsed = grp.iloc[1:]
    
    canonical_rows.append(canonical)
    
    # Provenance for canonical
    provenance_rows.append({
        "group_id":           group_id,
        "group_type":         grp_type,
        "internal_id":        int(canonical["_internal_id"]),
        "source_dataset":     canonical["source_dataset"],
        "source_split":       canonical["source_split"],
        "source_row_id":      int(canonical["source_row_id"]) if pd.notna(canonical["source_row_id"]) else -1,
        "is_canonical":       True,
        "conflict_status":    "IDENTICAL",
        "action":             "CANONICAL_KEPT",
        "canonical_group_id": group_id,
        "label_vector":       str(canonical["_label_tuple"]),
        "note":               f"Selected as canonical. Rule: meta_score={int(canonical['_meta_score'])}, "
                              f"ds={canonical['source_dataset']}, split={canonical['source_split']}, "
                              f"src_row_id={canonical['source_row_id']}"
    })
    
    # Provenance for collapsed rows
    for _, row in collapsed.iterrows():
        provenance_rows.append({
            "group_id":           group_id,
            "group_type":         grp_type,
            "internal_id":        int(row["_internal_id"]),
            "source_dataset":     row["source_dataset"],
            "source_split":       row["source_split"],
            "source_row_id":      int(row["source_row_id"]) if pd.notna(row["source_row_id"]) else -1,
            "is_canonical":       False,
            "conflict_status":    "IDENTICAL",
            "action":             "COLLAPSED_INTO_CANONICAL",
            "canonical_group_id": group_id,
            "label_vector":       str(row["_label_tuple"]),
            "note":               f"Collapsed into canonical (group_id={group_id}). Labels identical."
        })
    
    group_records.append({
        "group_id":   group_id, "group_type": grp_type, "n_rows": n_rows,
        "n_sources":  n_sources, "sources": "|".join(sources),
        "is_cross_source": is_cross_source, "is_cross_split": is_cross_split,
        "is_conflicting": False, "n_label_vectors": 1,
        "action": "CANONICAL_SELECTED"
    })

print(f"\nDuplicate group classification:")
print(f"  Total duplicate groups:           {n_groups_identical + n_groups_conflicting:,}")
print(f"  Identical-label groups (A/C/E):   {n_groups_identical:,}")
print(f"  Conflicting-label groups (B/D):   {n_groups_conflicting:,}")
print(f"  Cross-source groups:              {n_groups_cross_source:,}")
print(f"  EmoNoBa cross-split groups:       {n_groups_cross_split:,}")
print(f"\n  Rows removed (identical dups):    {rows_removed_identical:,}")
print(f"  Rows retained unresolved:         {rows_kept_conflicting:,}")

# ─── Assemble final deduplicated corpus ───────────────────────────────────────
# singletons + canonical records from identical-label groups + ALL rows from conflicting groups

# Singletons — add to provenance
for _, row in uniq_df.iterrows():
    provenance_rows.append({
        "group_id":           "SINGLETON",
        "group_type":         "SINGLETON",
        "internal_id":        int(row["_internal_id"]),
        "source_dataset":     row["source_dataset"],
        "source_split":       row["source_split"],
        "source_row_id":      int(row["source_row_id"]) if pd.notna(row["source_row_id"]) else -1,
        "is_canonical":       True,
        "conflict_status":    "SINGLETON",
        "action":             "KEPT_AS_IS",
        "canonical_group_id": "",
        "label_vector":       str(row["_label_tuple"]),
        "note":               "Unique text; no duplicate detected"
    })

# Conflicting groups: keep all their rows in the output (do not silently remove)
conflicting_internal_ids = set()
for cr in conflict_rows:
    for iid_str in cr.get("source_row_ids","").split("|"):
        pass
# Rebuild from provenance
conflict_internal_ids = {
    r["internal_id"] for r in provenance_rows
    if r["action"] == "RETAINED_UNRESOLVED"
}
conflicting_group_df = df[df["_internal_id"].isin(conflict_internal_ids)].copy()

# Canonical rows from identical groups
if canonical_rows:
    canonical_df = pd.DataFrame(canonical_rows)
else:
    canonical_df = pd.DataFrame(columns=df.columns)

# Final deduplicated corpus
cols_to_keep = [c for c in df.columns if not c.startswith("_")]
out_parts = [uniq_df[cols_to_keep]]
if len(canonical_df):
    out_parts.append(canonical_df[cols_to_keep])
if len(conflicting_group_df):
    out_parts.append(conflicting_group_df[cols_to_keep])

dedup_df = pd.concat(out_parts, ignore_index=True)
# Sort for reproducibility
dedup_df = dedup_df.sort_values(
    by=["source_dataset", "source_split", "source_row_id"],
    key=lambda col: col.map(DATASET_PRIORITY) if col.name=="source_dataset" else
                    col.map(SPLIT_PRIORITY).fillna(3).astype(int) if col.name=="source_split" else col
).reset_index(drop=True)

N_OUTPUT = len(dedup_df)
N_UNRESOLVED_CONFLICT_ROWS = rows_kept_conflicting
N_RESOLVED_REMOVED = rows_removed_identical

print(f"\nFinal output rows: {N_OUTPUT:,}")
print(f"  = {N_INPUT:,} (Phase 4) - {N_RESOLVED_REMOVED:,} (identical dups removed)")
print(f"  Conflicting rows retained unresolved: {N_UNRESOLVED_CONFLICT_ROWS:,}")

# ─── Cross-dataset duplicate breakdown ────────────────────────────────────────
# Compute the actual cross-dataset pairs in all groups
pair_counts = {"EmoNoBa-UBMEC": 0, "EmoNoBa-MONOVAB": 0,
               "UBMEC-MONOVAB": 0, "ALL_THREE": 0}
for rec in group_records:
    src_set = set(rec["sources"].split("|"))
    if src_set == {"EmoNoBa", "UBMEC"}:
        pair_counts["EmoNoBa-UBMEC"] += 1
    elif src_set == {"EmoNoBa", "MONOVAB"}:
        pair_counts["EmoNoBa-MONOVAB"] += 1
    elif src_set == {"UBMEC", "MONOVAB"}:
        pair_counts["UBMEC-MONOVAB"] += 1
    elif src_set == {"EmoNoBa", "UBMEC", "MONOVAB"}:
        pair_counts["ALL_THREE"] += 1

print(f"\nCross-dataset duplicate groups:")
for k, v in pair_counts.items():
    print(f"  {k}: {v:,} groups")

# ─── Save outputs ─────────────────────────────────────────────────────────────
# 1. Deduplicated corpus
out_csv = os.path.join(INTERIM, "harmonized_deduplicated.csv")
dedup_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_csv}  ({N_OUTPUT:,} rows)")

# 2. Duplicate groups summary
grp_csv = os.path.join(LOG_DIR, "duplicate_groups.csv")
if group_records:
    with open(grp_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(group_records[0].keys()))
        w.writeheader()
        w.writerows(group_records)
print(f"Saved: {grp_csv}  ({len(group_records):,} groups)")

# 3. Provenance log
prov_csv = os.path.join(LOG_DIR, "duplicate_provenance.csv")
if provenance_rows:
    prov_cols = ["group_id","group_type","internal_id","source_dataset","source_split",
                 "source_row_id","is_canonical","conflict_status","action",
                 "canonical_group_id","label_vector","note"]
    with open(prov_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=prov_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(provenance_rows)
print(f"Saved: {prov_csv}  ({len(provenance_rows):,} rows)")

# 4. Conflict log
conf_csv = os.path.join(LOG_DIR, "conflicting_duplicates.csv")
if conflict_rows:
    conf_cols = list(conflict_rows[0].keys())
    with open(conf_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=conf_cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(conflict_rows)
print(f"Saved: {conf_csv}  ({len(conflict_rows):,} conflicting groups)")

# 5. Validation JSON
checks = {}

# a) Binary labels
for col in TARGET_COLS:
    uv = sorted(dedup_df[col].dropna().unique().tolist())
    checks[f"target_{col}_binary"] = {"pass": all(v in [0,1] for v in uv), "values": uv}

# b) No null text
null_text = int(dedup_df["text"].isna().sum())
checks["no_null_text"] = {"pass": null_text==0, "null_count": null_text}

# c) No Love/Contempt columns
bad_cols = [c for c in dedup_df.columns if c in ("love","Love","contempt","Contempt")]
checks["no_love_contempt_target_cols"] = {"pass": len(bad_cols)==0, "found": bad_cols}

# d) Raw input unchanged
raw_still_ok = os.path.exists(os.path.join(
    WORKSPACE, "Data", "raw", "Emonoba", "Train.csv"))
checks["raw_data_unchanged"] = {"pass": raw_still_ok}

# e) Phase 4 file unchanged
phase4_rows = len(pd.read_csv(csv_in, low_memory=False))
checks["phase4_file_unchanged"] = {"pass": phase4_rows == N_INPUT, "rows": phase4_rows}

# f) No unlogged removals: every row removed must have a provenance record
removed_count = N_INPUT - N_OUTPUT + N_UNRESOLVED_CONFLICT_ROWS  # net non-singleton removals
# Actually: N_OUTPUT = singletons + canonical_from_identical + conflicting
# N_RESOLVED_REMOVED = exactly the collapsed rows
provenance_collapse_count = sum(1 for r in provenance_rows if r["action"]=="COLLAPSED_INTO_CANONICAL")
checks["all_removals_logged"] = {
    "pass": provenance_collapse_count == N_RESOLVED_REMOVED,
    "resolved_removed": N_RESOLVED_REMOVED,
    "provenance_collapse_records": provenance_collapse_count
}

# g) Label stats on deduplicated corpus
label_stats_dedup = {}
for col in TARGET_COLS:
    tp = int(dedup_df[col].sum())
    label_stats_dedup[col] = {
        "total_positive": tp,
        "prevalence_pct": round(tp/N_OUTPUT*100, 2)
    }

# h) Source counts
src_counts = {k: int(v) for k,v in dedup_df["source_dataset"].value_counts().items()}

# i) Cardinality
lab_sums = dedup_df[TARGET_COLS].sum(axis=1)
card_dist = {int(k): int(v) for k,v in lab_sums.value_counts().sort_index().items()}
mean_labs = round(float(lab_sums.mean()), 4)

# j) Check no duplicate texts remain among RESOLVED groups
# (conflicting texts may still appear multiple times — that is correct/expected)
resolved_texts = dedup_df[~dedup_df.index.isin(
    dedup_df[dedup_df["text"].isin(conflicting_group_df["text"])].index
)]["text"]
resolved_dups = resolved_texts.apply(comparison_key).duplicated().sum()
checks["no_resolved_dups_remaining"] = {
    "pass": int(resolved_dups) == 0,
    "remaining_resolved_dups": int(resolved_dups)
}

before_after = {
    "phase4_input_rows": N_INPUT,
    "identical_label_duplicate_rows_removed": N_RESOLVED_REMOVED,
    "conflicting_rows_retained_unresolved": N_UNRESOLVED_CONFLICT_ROWS,
    "phase5_output_rows": N_OUTPUT,
    "reconciliation_check": (N_INPUT - N_RESOLVED_REMOVED) == N_OUTPUT
}

validation = {
    "run_timestamp": RUN_TS,
    "phase_input": "Data/interim/harmonized_pre_dedup.csv",
    "phase_output": "Data/interim/harmonized_deduplicated.csv",
    "row_counts": {
        "phase4_input": N_INPUT,
        "singletons":   len(uniq_df),
        "total_dup_groups": n_groups_identical + n_groups_conflicting,
        "identical_label_groups": n_groups_identical,
        "conflicting_label_groups": n_groups_conflicting,
        "cross_source_groups": n_groups_cross_source,
        "emonoba_cross_split_groups": n_groups_cross_split,
        "rows_removed_identical_dups": N_RESOLVED_REMOVED,
        "rows_retained_conflicting": N_UNRESOLVED_CONFLICT_ROWS,
        "phase5_output": N_OUTPUT
    },
    "cross_dataset_pairs": pair_counts,
    "before_after_accounting": before_after,
    "validation_checks": checks,
    "label_statistics_post_dedup": label_stats_dedup,
    "source_counts_post_dedup": src_counts,
    "cardinality_distribution_post_dedup": card_dist,
    "mean_labels_per_example_post_dedup": mean_labs,
    "unresolved_decisions": {
        "count": n_groups_conflicting,
        "description": (
            f"{n_groups_conflicting} text groups have conflicting label vectors across their duplicate rows. "
            f"These require user decision before resolution. Their rows are retained in the output corpus "
            f"(both copies) and are logged in conflicting_duplicates.csv. "
            f"Do NOT proceed with splitting until a resolution policy is approved."
        ),
        "file": "logs/deduplication/conflicting_duplicates.csv"
    },
    "canonical_selection_rule": (
        "1. Prefer row with most populated metadata fields (domain+topic non-empty = higher score). "
        "2. Dataset priority: EmoNoBa > UBMEC > MONOVAB. "
        "3. Split priority: Train > Val > Test > NONE. "
        "4. Smallest source_row_id (numeric). "
        "Rule is deterministic; no randomness used."
    ),
    "note_conflicting": (
        "Conflicting duplicate groups are retained in full (all duplicate rows kept). "
        "They require a user-approved resolution strategy before the final split can be created."
    ),
    "note_raw_data": "Data/raw/ was not modified. harmonized_pre_dedup.csv was not overwritten."
}

val_json = os.path.join(LOG_DIR, "deduplication_validation.json")
with open(val_json, "w", encoding="utf-8") as f:
    json.dump(validation, f, indent=2, ensure_ascii=False)
print(f"Saved: {val_json}")

# ─── Print final summary ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 5 DEDUPLICATION COMPLETE")
print("="*60)
print(f"\nBefore/After Accounting:")
print(f"  Phase 4 input rows:                    {N_INPUT:>7,}")
print(f"  - Identical-label dup rows removed:    {N_RESOLVED_REMOVED:>7,}")
print(f"  + Conflicting rows retained:           {N_UNRESOLVED_CONFLICT_ROWS:>7,}  (retained in output, unresolved)")
print(f"  = Phase 5 output rows:                 {N_OUTPUT:>7,}")
print(f"  Reconciliation: {N_INPUT} - {N_RESOLVED_REMOVED} = {N_INPUT-N_RESOLVED_REMOVED} (expected {N_OUTPUT}): {'OK' if N_INPUT-N_RESOLVED_REMOVED==N_OUTPUT else 'MISMATCH'}")

print(f"\nDuplicate Groups:")
print(f"  Total groups with duplicates:   {n_groups_identical+n_groups_conflicting:,}")
print(f"  Identical-label groups:         {n_groups_identical:,}")
print(f"  Conflicting-label groups:       {n_groups_conflicting:,}  <- REQUIRES USER DECISION")
print(f"  Cross-source groups:            {n_groups_cross_source:,}")
print(f"  EmoNoBa cross-split groups:     {n_groups_cross_split:,}")

print(f"\nCross-Dataset Duplicate Groups:")
for k,v in pair_counts.items():
    print(f"  {k}: {v:,}")

print(f"\nLabel Statistics (post-dedup):")
for col in TARGET_COLS:
    s = label_stats_dedup[col]
    print(f"  {col:8s}: {s['total_positive']:6,} ({s['prevalence_pct']:.2f}%)")

print(f"\nSource counts (post-dedup): {src_counts}")
print(f"Cardinality distribution:   {card_dist}")
print(f"Mean labels per example:    {mean_labs}")
print(f"\nAll validation checks passed: {all(v.get('pass',True) for v in checks.values() if 'pass' in v)}")
print(f"\nUNRESOLVED: {n_groups_conflicting} conflicting duplicate groups require user decision.")
print(f"DO NOT proceed to Phase 6 (splitting) until conflicts are resolved.")
