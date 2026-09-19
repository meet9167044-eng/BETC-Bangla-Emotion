import os
import sys
import re
import json
import pandas as pd

raw_dir = r'c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion\Data\raw'
output_dir = r'c:\Users\MEET JAIN\OneDrive\Desktop\BETC-Bangla-Emotion\results\dataset_audit'

bangla = re.compile(r'[\u0980-\u09FF]')
latin = re.compile(r'[a-zA-Z]')

anomalies = {}

# 1. UBMEC
ubmec = pd.read_excel(os.path.join(raw_dir, 'UBMEC Corpus_Sakib(updated).xlsx'), sheet_name='UBMEC')
no_alpha_ubmec = ubmec[~ubmec['text'].astype(str).str.contains(bangla, regex=True) & ~ubmec['text'].astype(str).str.contains(latin, regex=True)]
ubmec_no_alpha = []
for idx, row in no_alpha_ubmec.iterrows():
    ubmec_no_alpha.append({"row_index": int(idx), "text": str(row["text"]), "class": str(row["classes"])})

# UBMEC mixed Latin
mixed_u = ubmec[ubmec['text'].astype(str).str.contains(bangla, regex=True) & ubmec['text'].astype(str).str.contains(latin, regex=True)]

# UBMEC duplicates with conflicting labels
grouped_u = ubmec.groupby('text')['classes'].nunique()
conflicting_u_keys = grouped_u[grouped_u > 1].index.tolist()
conflicting_u_samples = []
for t in conflicting_u_keys[:10]:
    rows = ubmec[ubmec['text'] == t]
    conflicting_u_samples.append({
        "text": str(t),
        "classes": rows['classes'].tolist(),
        "indices": rows.index.tolist()
    })

anomalies["ubmec"] = {
    "total_rows": len(ubmec),
    "no_alpha_count": len(no_alpha_ubmec),
    "no_alpha_examples": ubmec_no_alpha,
    "mixed_latin_bangla_count": len(mixed_u),
    "sample_mixed_latin": [str(x) for x in mixed_u['text'].head(5).tolist()],
    "duplicate_exact_rows": int(ubmec.duplicated().sum()),
    "duplicate_texts_count": len(ubmec) - int(ubmec['text'].nunique()),
    "conflicting_label_texts_count": len(conflicting_u_keys),
    "sample_conflicting_labels": conflicting_u_samples
}

# 2. MONOVAB
monovab = pd.read_csv(os.path.join(raw_dir, 'MONOVAB (1).csv'))
no_alpha_monovab = monovab[~monovab['comment'].astype(str).str.contains(bangla, regex=True) & ~monovab['comment'].astype(str).str.contains(latin, regex=True)]
monovab_no_alpha = []
for idx, row in no_alpha_monovab.iterrows():
    monovab_no_alpha.append({"row_index": int(idx), "comment": str(row["comment"])})

mixed_m = monovab[monovab['comment'].astype(str).str.contains(bangla, regex=True) & monovab['comment'].astype(str).str.contains(latin, regex=True)]

label_cols_m = ['anger', 'contempt', 'disgust', 'enjoyment', 'fear', 'sadness', 'surprise']
grouped_m = monovab.groupby('comment')[label_cols_m].nunique()
conflicting_m_keys = grouped_m[(grouped_m > 1).any(axis=1)].index.tolist()
conflicting_m_samples = []
for t in conflicting_m_keys[:10]:
    rows = monovab[monovab['comment'] == t]
    conflicting_m_samples.append({
        "text": str(t),
        "indices": rows.index.tolist(),
        "labels": rows[label_cols_m].to_dict(orient='records')
    })

anomalies["monovab"] = {
    "total_rows": len(monovab),
    "no_alpha_count": len(no_alpha_monovab),
    "no_alpha_examples": monovab_no_alpha,
    "mixed_latin_bangla_count": len(mixed_m),
    "sample_mixed_latin": [str(x) for x in mixed_m['comment'].head(5).tolist()],
    "duplicate_exact_rows_including_id": int(monovab.duplicated().sum()),
    "duplicate_exact_rows_excluding_id": int(monovab.drop(columns=['Unnamed: 0']).duplicated().sum()),
    "duplicate_texts_count": len(monovab) - int(monovab['comment'].nunique()),
    "conflicting_label_texts_count": len(conflicting_m_keys),
    "sample_conflicting_labels": conflicting_m_samples
}

# 3. EmoNoBa
emonoba_files = ['Train.csv', 'Val.csv', 'Test.csv']
emonoba_dfs = [pd.read_csv(os.path.join(raw_dir, 'Emonoba', f)) for f in emonoba_files]
emonoba = pd.concat(emonoba_dfs, ignore_index=True)

label_cols_e = ['Love', 'Joy', 'Surprise', 'Anger', 'Sadness', 'Fear']
grouped_e = emonoba.groupby('Data')[label_cols_e].nunique()
conflicting_e_keys = grouped_e[(grouped_e > 1).any(axis=1)].index.tolist()
conflicting_e_samples = []
for t in conflicting_e_keys[:10]:
    rows = emonoba[emonoba['Data'] == t]
    conflicting_e_samples.append({
        "text": str(t),
        "indices": rows.index.tolist(),
        "labels": rows[label_cols_e].to_dict(orient='records')
    })

# Check cross-split duplicates in EmoNoBa
train_texts = set(emonoba_dfs[0]['Data'])
val_texts = set(emonoba_dfs[1]['Data'])
test_texts = set(emonoba_dfs[2]['Data'])

train_val_overlap = len(train_texts.intersection(val_texts))
train_test_overlap = len(train_texts.intersection(test_texts))
val_test_overlap = len(val_texts.intersection(test_texts))

anomalies["emonoba"] = {
    "total_rows": len(emonoba),
    "duplicate_exact_rows": int(emonoba.duplicated().sum()),
    "duplicate_texts_count": len(emonoba) - int(emonoba['Data'].nunique()),
    "conflicting_label_texts_count": len(conflicting_e_keys),
    "sample_conflicting_labels": conflicting_e_samples,
    "cross_split_leakage_in_original_split": {
        "train_val_shared_texts": train_val_overlap,
        "train_test_shared_texts": train_test_overlap,
        "val_test_shared_texts": val_test_overlap
    }
}

with open(os.path.join(output_dir, "data_quality_anomalies.json"), "w", encoding="utf-8") as f:
    json.dump(anomalies, f, indent=2, ensure_ascii=False)

print("Saved data quality anomalies to data_quality_anomalies.json successfully!")
