import os
import sys
import json
import re
import unicodedata
import pandas as pd
import numpy as np

WORKSPACE_DIR = r"c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion"
RAW_DIR = os.path.join(WORKSPACE_DIR, "Data", "raw")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "results", "dataset_audit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex patterns
BANGLA_CHAR = re.compile(r'[\u0980-\u09FF]')
LATIN_CHAR = re.compile(r'[a-zA-Z]')
DIGIT_CHAR = re.compile(r'[0-9\u09E6-\u09EF]')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')

def check_file_encoding(filepath):
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    detected = {}
    with open(filepath, 'rb') as f:
        data = f.read()
    for enc in encodings:
        try:
            data.decode(enc)
            detected[enc] = True
        except UnicodeDecodeError:
            detected[enc] = False
    return detected, len(data)

def analyze_text_column(series):
    total = len(series)
    null_count = int(series.isna().sum())
    non_null = series.dropna().astype(str)
    
    empty_exact = int((non_null == '').sum())
    whitespace_only = int(((non_null != '') & (non_null.str.strip() == '')).sum())
    valid_text_mask = (non_null.str.strip() != '')
    valid_texts = non_null[valid_text_mask]
    
    lengths_char = valid_texts.apply(len)
    lengths_word = valid_texts.apply(lambda x: len(x.strip().split()))
    
    has_bangla = valid_texts.apply(lambda x: bool(BANGLA_CHAR.search(x)))
    has_latin = valid_texts.apply(lambda x: bool(LATIN_CHAR.search(x)))
    only_latin = valid_texts.apply(lambda x: bool(LATIN_CHAR.search(x)) and not bool(BANGLA_CHAR.search(x)))
    pure_bangla_no_latin = valid_texts.apply(lambda x: bool(BANGLA_CHAR.search(x)) and not bool(LATIN_CHAR.search(x)))
    mixed_bangla_latin = valid_texts.apply(lambda x: bool(BANGLA_CHAR.search(x)) and bool(LATIN_CHAR.search(x)))
    no_alpha = valid_texts.apply(lambda x: not bool(BANGLA_CHAR.search(x)) and not bool(LATIN_CHAR.search(x)))
    
    has_url = valid_texts.apply(lambda x: bool(URL_PATTERN.search(x)))
    has_mention = valid_texts.apply(lambda x: bool(MENTION_PATTERN.search(x)))
    
    unique_raw = int(series.nunique(dropna=False))
    unique_non_null = int(non_null.nunique())
    unique_stripped = int(valid_texts.str.strip().nunique())
    
    # Normalized duplicate check (NFC + whitespace collapse)
    def simple_norm(t):
        return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', t.strip()))
    
    norm_texts = valid_texts.apply(simple_norm)
    unique_norm = int(norm_texts.nunique())
    
    return {
        "total_rows": total,
        "null_count": null_count,
        "null_percentage": round(null_count / total * 100, 4) if total > 0 else 0,
        "empty_exact": empty_exact,
        "whitespace_only": whitespace_only,
        "total_unusable_texts": null_count + empty_exact + whitespace_only,
        "usable_texts_count": len(valid_texts),
        "unique_raw_texts": unique_raw,
        "unique_non_null_texts": unique_non_null,
        "unique_stripped_texts": unique_stripped,
        "unique_normalized_texts": unique_norm,
        "duplicate_raw_texts": total - unique_raw,
        "duplicate_normalized_texts": len(valid_texts) - unique_norm,
        "char_length_stats": {
            "min": int(lengths_char.min()) if len(lengths_char) > 0 else 0,
            "max": int(lengths_char.max()) if len(lengths_char) > 0 else 0,
            "mean": round(float(lengths_char.mean()), 2) if len(lengths_char) > 0 else 0,
            "median": float(lengths_char.median()) if len(lengths_char) > 0 else 0,
            "std": round(float(lengths_char.std()), 2) if len(lengths_char) > 0 else 0,
            "q25": float(lengths_char.quantile(0.25)) if len(lengths_char) > 0 else 0,
            "q75": float(lengths_char.quantile(0.75)) if len(lengths_char) > 0 else 0,
        },
        "word_count_stats": {
            "min": int(lengths_word.min()) if len(lengths_word) > 0 else 0,
            "max": int(lengths_word.max()) if len(lengths_word) > 0 else 0,
            "mean": round(float(lengths_word.mean()), 2) if len(lengths_word) > 0 else 0,
            "median": float(lengths_word.median()) if len(lengths_word) > 0 else 0,
            "std": round(float(lengths_word.std()), 2) if len(lengths_word) > 0 else 0,
            "q25": float(lengths_word.quantile(0.25)) if len(lengths_word) > 0 else 0,
            "q75": float(lengths_word.quantile(0.75)) if len(lengths_word) > 0 else 0,
        },
        "language_and_content_breakdown": {
            "pure_bangla_no_latin": int(pure_bangla_no_latin.sum()),
            "mixed_bangla_and_latin": int(mixed_bangla_latin.sum()),
            "latin_only_no_bangla": int(only_latin.sum()),
            "no_letters_only_symbols_digits": int(no_alpha.sum()),
            "contains_bangla_any": int(has_bangla.sum()),
            "contains_latin_any": int(has_latin.sum()),
            "contains_url": int(has_url.sum()),
            "contains_mention": int(has_mention.sum())
        }
    }

print("Running deep audit on EmoNoBa...")
emonoba_dir = os.path.join(RAW_DIR, "Emonoba")
emonoba_files = ["Train.csv", "Val.csv", "Test.csv"]
emonoba_splits_audit = {}
emonoba_dfs = {}

for fname in emonoba_files:
    fpath = os.path.join(emonoba_dir, fname)
    enc_status, fsize = check_file_encoding(fpath)
    df = pd.read_csv(fpath)
    emonoba_dfs[fname] = df
    
    label_cols = ['Love', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear']
    
    # Analyze label values and distribution
    label_dist = {}
    for col in label_cols:
        val_counts = df[col].value_counts(dropna=False).to_dict()
        # Convert keys to string for JSON serialization
        label_dist[col] = {str(k): int(v) for k, v in val_counts.items()}
        
    # Multi-label cardinality
    # Assuming 1 is positive, check if all values are 0 or 1
    # Check numeric conversion
    labels_matrix = df[label_cols]
    labels_active_per_row = (labels_matrix == 1).sum(axis=1)
    cardinality_counts = labels_active_per_row.value_counts(dropna=False).sort_index().to_dict()
    
    emonoba_splits_audit[fname] = {
        "file_name": fname,
        "file_size_bytes": fsize,
        "encoding_test": enc_status,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "text_column": "Data",
        "label_columns": label_cols,
        "metadata_columns": [c for c in df.columns if c not in label_cols and c != 'Data'],
        "missing_values_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "text_audit": analyze_text_column(df['Data']),
        "label_distribution": label_dist,
        "active_labels_per_example_distribution": {str(k): int(v) for k, v in cardinality_counts.items()},
        "mean_labels_per_example": round(float(labels_active_per_row.mean()), 4),
        "is_single_or_multi_label": "multi-label",
        "metadata_unique_counts": {c: int(df[c].nunique(dropna=False)) for c in ['Topic', 'Domain', 'is_admin'] if c in df.columns}
    }

# Full combined EmoNoBa audit
emonoba_full = pd.concat(emonoba_dfs.values(), ignore_index=True)
label_cols = ['Love', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear']
label_dist_full = {}
for col in label_cols:
    val_counts = emonoba_full[col].value_counts(dropna=False).to_dict()
    label_dist_full[col] = {str(k): int(v) for k, v in val_counts.items()}

labels_matrix_full = emonoba_full[label_cols]
cardinality_full = (labels_matrix_full == 1).sum(axis=1).value_counts(dropna=False).sort_index().to_dict()

emonoba_total_audit = {
    "dataset": "EmoNoBa",
    "total_rows": len(emonoba_full),
    "total_columns": len(emonoba_full.columns),
    "column_names": list(emonoba_full.columns),
    "text_column": "Data",
    "label_columns": label_cols,
    "metadata_columns": [c for c in emonoba_full.columns if c not in label_cols and c != 'Data'],
    "missing_values_per_column": {c: int(emonoba_full[c].isna().sum()) for c in emonoba_full.columns},
    "exact_duplicate_rows": int(emonoba_full.duplicated().sum()),
    "text_audit": analyze_text_column(emonoba_full['Data']),
    "label_distribution": label_dist_full,
    "active_labels_per_example_distribution": {str(k): int(v) for k, v in cardinality_full.items()},
    "mean_labels_per_example": round(float((labels_matrix_full == 1).sum(axis=1).mean()), 4),
    "is_single_or_multi_label": "multi-label",
    "domains": emonoba_full['Domain'].value_counts(dropna=False).to_dict() if 'Domain' in emonoba_full.columns else {},
    "topics": emonoba_full['Topic'].value_counts(dropna=False).head(20).to_dict() if 'Topic' in emonoba_full.columns else {},
    "is_admin": emonoba_full['is_admin'].value_counts(dropna=False).to_dict() if 'is_admin' in emonoba_full.columns else {},
    "splits": emonoba_splits_audit
}

with open(os.path.join(OUTPUT_DIR, "emonoba_audit.json"), "w", encoding="utf-8") as f:
    json.dump(emonoba_total_audit, f, indent=2, ensure_ascii=False)

print("EmoNoBa audit saved.")

# 2. UBMEC AUDIT
print("Running deep audit on UBMEC...")
ubmec_path = os.path.join(RAW_DIR, "UBMEC Corpus_Sakib(updated).xlsx")
ubmec_size = os.path.getsize(ubmec_path)
ubmec_df = pd.read_excel(ubmec_path, sheet_name="UBMEC")

ubmec_class_counts = ubmec_df['classes'].value_counts(dropna=False).to_dict()
# Check for any whitespace/case variation in classes
ubmec_classes_cleaned = ubmec_df['classes'].astype(str).str.strip().str.lower()
ubmec_class_cleaned_counts = ubmec_classes_cleaned.value_counts(dropna=False).to_dict()

ubmec_audit = {
    "dataset": "UBMEC",
    "file_name": "UBMEC Corpus_Sakib(updated).xlsx",
    "file_format": "Excel (.xlsx)",
    "file_size_bytes": ubmec_size,
    "sheet_name": "UBMEC",
    "rows": len(ubmec_df),
    "columns": len(ubmec_df.columns),
    "column_names": list(ubmec_df.columns),
    "dtypes": {c: str(t) for c, t in ubmec_df.dtypes.items()},
    "text_column": "text",
    "label_columns": ["classes"],
    "is_single_or_multi_label": "single-label",
    "missing_values_per_column": {c: int(ubmec_df[c].isna().sum()) for c in ubmec_df.columns},
    "exact_duplicate_rows": int(ubmec_df.duplicated().sum()),
    "text_audit": analyze_text_column(ubmec_df['text']),
    "raw_label_distribution": {str(k): int(v) for k, v in ubmec_class_counts.items()},
    "cleaned_label_distribution": {str(k): int(v) for k, v in ubmec_class_cleaned_counts.items()},
    "mean_labels_per_example": 1.0 if ubmec_df['classes'].isna().sum() == 0 else round(float(ubmec_df['classes'].notna().mean()), 4)
}

with open(os.path.join(OUTPUT_DIR, "ubmec_audit.json"), "w", encoding="utf-8") as f:
    json.dump(ubmec_audit, f, indent=2, ensure_ascii=False)

print("UBMEC audit saved.")

# 3. MONOVAB AUDIT
print("Running deep audit on MONOVAB...")
monovab_path = os.path.join(RAW_DIR, "MONOVAB (1).csv")
monovab_enc_status, monovab_size = check_file_encoding(monovab_path)
monovab_df = pd.read_csv(monovab_path)

monovab_label_cols = ['anger', 'contempt', 'disgust', 'enjoyment', 'fear', 'sadness', 'surprise']
monovab_label_dist = {}
for col in monovab_label_cols:
    vc = monovab_df[col].value_counts(dropna=False).to_dict()
    monovab_label_dist[col] = {str(k): int(v) for k, v in vc.items()}

# Cardinality
monovab_active = (monovab_df[monovab_label_cols] == 1).sum(axis=1)
monovab_cardinality = monovab_active.value_counts(dropna=False).sort_index().to_dict()

# Cardinality for target 6 (excluding contempt)
monovab_target_cols = ['anger', 'disgust', 'enjoyment', 'fear', 'sadness', 'surprise']
monovab_target_active = (monovab_df[monovab_target_cols] == 1).sum(axis=1)
monovab_target_cardinality = monovab_target_active.value_counts(dropna=False).sort_index().to_dict()

monovab_audit = {
    "dataset": "MONOVAB",
    "file_name": "MONOVAB (1).csv",
    "file_format": "CSV (.csv)",
    "file_size_bytes": monovab_size,
    "encoding_test": monovab_enc_status,
    "rows": len(monovab_df),
    "columns": len(monovab_df.columns),
    "column_names": list(monovab_df.columns),
    "dtypes": {c: str(t) for c, t in monovab_df.dtypes.items()},
    "index_column": "Unnamed: 0",
    "text_column": "comment",
    "label_columns": monovab_label_cols,
    "is_single_or_multi_label": "multi-label",
    "missing_values_per_column": {c: int(monovab_df[c].isna().sum()) for c in monovab_df.columns},
    "exact_duplicate_rows": int(monovab_df.duplicated().sum()),
    "exact_duplicate_rows_excluding_index": int(monovab_df.drop(columns=['Unnamed: 0']).duplicated().sum()),
    "text_audit": analyze_text_column(monovab_df['comment']),
    "label_distribution": monovab_label_dist,
    "active_labels_per_example_all_7": {str(k): int(v) for k, v in monovab_cardinality.items()},
    "mean_labels_per_example_all_7": round(float(monovab_active.mean()), 4),
    "active_labels_per_example_target_6": {str(k): int(v) for k, v in monovab_target_cardinality.items()},
    "mean_labels_per_example_target_6": round(float(monovab_target_active.mean()), 4)
}

with open(os.path.join(OUTPUT_DIR, "monovab_audit.json"), "w", encoding="utf-8") as f:
    json.dump(monovab_audit, f, indent=2, ensure_ascii=False)

print("MONOVAB audit saved.")

# Cross-dataset text overlap analysis
print("Analyzing cross-dataset text overlaps...")
emonoba_texts_set = set(emonoba_full['Data'].dropna().astype(str).str.strip())
ubmec_texts_set = set(ubmec_df['text'].dropna().astype(str).str.strip())
monovab_texts_set = set(monovab_df['comment'].dropna().astype(str).str.strip())

overlap_emonoba_ubmec = len(emonoba_texts_set.intersection(ubmec_texts_set))
overlap_emonoba_monovab = len(emonoba_texts_set.intersection(monovab_texts_set))
overlap_ubmec_monovab = len(ubmec_texts_set.intersection(monovab_texts_set))
overlap_all_three = len(emonoba_texts_set.intersection(ubmec_texts_set).intersection(monovab_texts_set))
union_all = len(emonoba_texts_set.union(ubmec_texts_set).union(monovab_texts_set))
total_sum_texts = len(emonoba_texts_set) + len(ubmec_texts_set) + len(monovab_texts_set)

cross_overlap_summary = {
    "unique_texts_emonoba": len(emonoba_texts_set),
    "unique_texts_ubmec": len(ubmec_texts_set),
    "unique_texts_monovab": len(monovab_texts_set),
    "overlap_emonoba_ubmec": overlap_emonoba_ubmec,
    "overlap_emonoba_monovab": overlap_emonoba_monovab,
    "overlap_ubmec_monovab": overlap_ubmec_monovab,
    "overlap_all_three": overlap_all_three,
    "union_unique_texts": union_all,
    "total_unique_across_datasets_sum": total_sum_texts,
    "cross_dataset_duplicate_texts": total_sum_texts - union_all
}

with open(os.path.join(OUTPUT_DIR, "cross_dataset_overlap.json"), "w", encoding="utf-8") as f:
    json.dump(cross_overlap_summary, f, indent=2)

print("Cross-dataset overlap saved.")

# Generate Label Distribution CSVs
# 1. EmoNoBa
emonoba_dist_rows = []
for col in ['Love', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear']:
    train_pos = int((emonoba_dfs['Train.csv'][col] == 1).sum())
    val_pos = int((emonoba_dfs['Val.csv'][col] == 1).sum())
    test_pos = int((emonoba_dfs['Test.csv'][col] == 1).sum())
    tot_pos = int((emonoba_full[col] == 1).sum())
    tot_rows = len(emonoba_full)
    emonoba_dist_rows.append({
        "label": col,
        "train_positive": train_pos,
        "val_positive": val_pos,
        "test_positive": test_pos,
        "total_positive": tot_pos,
        "total_rows": tot_rows,
        "prevalence_percent": round(tot_pos / tot_rows * 100, 2)
    })
pd.DataFrame(emonoba_dist_rows).to_csv(os.path.join(OUTPUT_DIR, "emonoba_label_distribution.csv"), index=False)

# 2. UBMEC
ubmec_dist_rows = []
tot_ubmec = len(ubmec_df)
for cls_val, count in ubmec_df['classes'].value_counts(dropna=False).items():
    ubmec_dist_rows.append({
        "class_label": str(cls_val),
        "count": int(count),
        "total_rows": tot_ubmec,
        "prevalence_percent": round(int(count) / tot_ubmec * 100, 2)
    })
pd.DataFrame(ubmec_dist_rows).to_csv(os.path.join(OUTPUT_DIR, "ubmec_label_distribution.csv"), index=False)

# 3. MONOVAB
monovab_dist_rows = []
tot_monovab = len(monovab_df)
for col in monovab_label_cols:
    pos_count = int((monovab_df[col] == 1).sum())
    zero_count = int((monovab_df[col] == 0).sum())
    other_count = tot_monovab - pos_count - zero_count
    monovab_dist_rows.append({
        "label": col,
        "positive_count": pos_count,
        "zero_count": zero_count,
        "other_or_nan": other_count,
        "total_rows": tot_monovab,
        "prevalence_percent": round(pos_count / tot_monovab * 100, 2)
    })
pd.DataFrame(monovab_dist_rows).to_csv(os.path.join(OUTPUT_DIR, "monovab_label_distribution.csv"), index=False)

# Master summary JSON
master_summary = {
    "timestamp": "2026-09-19",
    "dataset_counts": {
        "emonoba": {
            "train_rows": len(emonoba_dfs['Train.csv']),
            "val_rows": len(emonoba_dfs['Val.csv']),
            "test_rows": len(emonoba_dfs['Test.csv']),
            "total_rows": len(emonoba_full),
            "columns": len(emonoba_full.columns),
            "text_column": "Data",
            "type": "multi-label"
        },
        "ubmec": {
            "total_rows": len(ubmec_df),
            "columns": len(ubmec_df.columns),
            "text_column": "text",
            "type": "single-label"
        },
        "monovab": {
            "total_rows": len(monovab_df),
            "columns": len(monovab_df.columns),
            "text_column": "comment",
            "type": "multi-label"
        },
        "grand_total_raw_rows": len(emonoba_full) + len(ubmec_df) + len(monovab_df)
    },
    "cross_dataset_overlap": cross_overlap_summary
}

with open(os.path.join(OUTPUT_DIR, "raw_dataset_summary.json"), "w", encoding="utf-8") as f:
    json.dump(master_summary, f, indent=2)

print("All audit files generated successfully!")
