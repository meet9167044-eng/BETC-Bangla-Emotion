"""
scripts/experiments/phase4_negation_study.py
============================================
BETC Improvement Plan — Phase 4: Improve Negation and Intensifier Handling

Objectives:
  1. Compile and document expanded Bangla negation and intensifier vocabulary.
  2. Implement Negation Scope Extractor:
     - Detects when negation occurs within a 3-token window of an emotion term
       (using the 6-emotion lexicon from Phase 3: Joy, Fear, Sadness, Disgust, Anger, Surprise).
     - Computes emotion-specific negation scope features (neg_joy, neg_fear, neg_sadness, etc.).
  3. Verify preservation during preprocessing and normalization.
  4. Inspect qualitative examples where negation alters emotion semantics.
  5. Run independent ablation comparisons on validation split:
     - E8 Baseline
     - E8 + Advanced Negation Scope Features (E8_neg_scope)
     - Full Enhanced Feature Suite (TF-IDF + Emoji + EmotionLexicon + Advanced Negation)
  6. Output coverage and ablation deliverables.
"""

import os
import sys
import io
import json
import time
from collections import Counter
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

from src.models.betc import TARGET_COLS, compute_label_order
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds
from src.features.feature_combination import BETCCombinedExtractor

DATA_DIR = os.path.join(WORKSPACE, "Data", "processed")
CONFIGS_DIR = os.path.join(WORKSPACE, "configs")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase4_negation")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Expanded Negation vocabulary including informal/social-media forms
EXPANDED_NEGATIONS = [
    "না", "নাই", "নেই", "নয়", "না-ই", "নয়তো", "নয়তঃ", "নহে", "নাহ", "নাহ্", "নাইরে",
    "কখনো না", "কখনও না", "কখনই না", "কোনো না", "কোনোভাবেই না", "মোটেই না", "মোটেও না",
    "একেবারে না", "অবশ্যই না", "আর না", "আর নয়", "কিছুতেই না", "কিছুতেই নয়", "একদমই না",
    "না তো", "না রে", "নয়কো", "ছাড়া", "বিনা", "ব্যতীত"
]

# Expanded Intensifiers
EXPANDED_INTENSIFIERS = [
    "খুব", "অনেক", "একদম", "অত্যন্ত", "অত্যধিক", "অসাধারণ", "অবিশ্বাস্য", "অতি", "সত্যিই",
    "সত্যি", "আসলেই", "আসলে", "সত্যিকার", "একেবারে", "মোটেই", "মোটেও", "বেশি", "বেশ",
    "বেশি বেশি", "খুবই", "খুবি", "ভীষণ", "ভীষণ ভাবে", "ভীষণভাবে", "অনেক বেশি", "বহুত",
    "বহুত বেশি", "পুরোপুরি", "সম্পূর্ণ", "সম্পূর্ণভাবে", "হাজার", "লক্ষ", "কোটি", "অত্যন্তই",
    "অনেকটা", "মারাত্মক", "মারাত্মক ভাবে", "মারাত্মকভাবে", "জঘন্য", "চরম", "প্রচুর", "দারুণভাবে",
    "বিশাল", "ভয়ানকভাবে", "মারাত্বক"
]

def load_emotion_lexicon():
    path = os.path.join(CONFIGS_DIR, "emotion_lexicon_bn.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def extract_advanced_negation_features(texts, neg_vocab, intens_vocab, emo_dict):
    """
    Extract 16 dense features:
      - neg_count, neg_present
      - intens_count, intens_present
      - neg_intensified_count
      - neg_emotion_scope_total: count of any emotion words negated within 3 tokens
      - 6 emotion-specific negation scope counts:
        * neg_joy (e.g. ভালো না, খুশি নই)
        * neg_fear (e.g. ভয় নেই, আতঙ্ক নাই)
        * neg_sadness (e.g. কষ্ট নেই, দুঃখ নাই)
        * neg_disgust (e.g. খারাপ না)
        * neg_anger (e.g. রাগ নেই)
        * neg_surprise (e.g. অবাক হওয়ার কিছু নেই)
      - 4 modifier ratios: neg_ratio, intens_ratio, scope_ratio, neg_intens_ratio
    """
    neg_set = set(neg_vocab)
    intens_set = set(intens_vocab)
    emo_sets = {k: set(v) for k, v in emo_dict.items()}
    all_emo_set = set().union(*emo_sets.values())

    rows = []
    for t in texts:
        tokens = t.split()
        n_tokens = max(len(tokens), 1)

        # 1. Counts
        neg_c = sum(1 for w in tokens if w in neg_set)
        intens_c = sum(1 for w in tokens if w in intens_set)

        # 2. Negation near intensifier (window 3)
        neg_intens_c = 0
        for i, tok in enumerate(tokens):
            if tok in neg_set:
                window = tokens[max(0, i-3): i] + tokens[i+1: i+4]
                if any(w in intens_set for w in window):
                    neg_intens_c += 1

        # 3. Negation of Emotion Words (Negation Scope)
        scope_counts = {c: 0 for c in TARGET_COLS}
        total_scope = 0
        for i, tok in enumerate(tokens):
            if tok in neg_set:
                window = tokens[max(0, i-3): i] + tokens[i+1: i+4]
                for emo, v_set in emo_sets.items():
                    matches = sum(1 for w in window if w in v_set)
                    if matches > 0:
                        scope_counts[emo] += matches
                        total_scope += matches

        row = [
            float(neg_c),
            1.0 if neg_c > 0 else 0.0,
            float(intens_c),
            1.0 if intens_c > 0 else 0.0,
            float(neg_intens_c),
            float(total_scope),
            float(scope_counts["anger"]),
            float(scope_counts["disgust"]),
            float(scope_counts["fear"]),
            float(scope_counts["joy"]),
            float(scope_counts["sadness"]),
            float(scope_counts["surprise"]),
            neg_c / n_tokens,
            intens_c / n_tokens,
            total_scope / n_tokens,
            (neg_c + intens_c) / n_tokens
        ]
        rows.append(row)
    return np.array(rows, dtype=np.float32)

def main():
    print("=" * 75)
    print("BETC Phase 4 — Bangla Negation & Intensifier Scope Investigation")
    print("=" * 75)

    # 1. Load data
    print("\n[Step 1] Loading preprocessed data splits...")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "train", "train_preprocessed.csv"), low_memory=False)
    df_val   = pd.read_csv(os.path.join(DATA_DIR, "validation", "validation_preprocessed.csv"), low_memory=False)
    print(f"  Train: {len(df_train):,} rows | Val: {len(df_val):,} rows")

    train_proc = df_train["processed_text"].fillna("").astype(str).tolist()
    val_proc   = df_val["processed_text"].fillna("").astype(str).tolist()
    train_raw  = df_train["text"].fillna("").astype(str).tolist()
    val_raw    = df_val["text"].fillna("").astype(str).tolist()

    Y_train = df_train[TARGET_COLS].values.astype(int)
    Y_val   = df_val[TARGET_COLS].values.astype(int)

    # 2. Vocabulary Documentation & Updating Configs
    print("\n[Step 2] Updating & Documenting Vocabulary in configs/...")
    with open(os.path.join(CONFIGS_DIR, "negations_bn.txt"), "w", encoding="utf-8") as f:
        f.write("# configs/negations_bn.txt\n# Expanded documented negation vocabulary for BETC Phase 4\n")
        for w in EXPANDED_NEGATIONS:
            f.write(f"{w}\n")

    with open(os.path.join(CONFIGS_DIR, "intensifiers_bn.txt"), "w", encoding="utf-8") as f:
        f.write("# configs/intensifiers_bn.txt\n# Expanded documented intensifier vocabulary for BETC Phase 4\n")
        for w in EXPANDED_INTENSIFIERS:
            f.write(f"{w}\n")
    print(f"  Negation vocabulary size: {len(EXPANDED_NEGATIONS)} entries")
    print(f"  Intensifier vocabulary size: {len(EXPANDED_INTENSIFIERS)} entries")

    # 3. Measure Negation & Scope Coverage
    print("\n[Step 3] Measuring Negation Scope & Valence-Flip Coverage on Training Data...")
    emo_dict = load_emotion_lexicon()
    neg_features_train = extract_advanced_negation_features(train_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict)

    n_with_neg = int(np.sum(neg_features_train[:, 1] > 0))
    n_with_intens = int(np.sum(neg_features_train[:, 3] > 0))
    n_with_scope = int(np.sum(neg_features_train[:, 5] > 0))

    print(f"  Texts containing Negation: {n_with_neg:,} / {len(df_train):,} ({(n_with_neg/len(df_train))*100:.2f}%)")
    print(f"  Texts containing Intensifier: {n_with_intens:,} / {len(df_train):,} ({(n_with_intens/len(df_train))*100:.2f}%)")
    print(f"  Texts with Negated Emotion Words: {n_with_scope:,} / {len(df_train):,} ({(n_with_scope/len(df_train))*100:.2f}%)")

    # Scope breakdown per emotion
    print("  Negation Scope Breakdown by Emotion Target:")
    scope_stats = {}
    for i, c in enumerate(TARGET_COLS):
        col_idx = 6 + i
        count_c = int(np.sum(neg_features_train[:, col_idx] > 0))
        print(f"    - Negated {c:<10}: {count_c:>5} texts")
        scope_stats[c] = count_c

    # 4. Qualitative Examples of Valence Alterations
    print("\n[Step 4] Qualitative Examples of Negation Valence Alterations:")
    examples = []
    scope_indices = np.where(neg_features_train[:, 5] > 0)[0]
    for idx in scope_indices[:5]:
        sample_text = train_proc[idx]
        emotions_present = [TARGET_COLS[j] for j in range(6) if Y_train[idx, j] == 1]
        print(f"  Text: {sample_text}")
        print(f"  True Emotions: {emotions_present}\n")
        examples.append({"text": sample_text, "labels": emotions_present})

    # 5. Build Features & Run Ablation Evaluation
    print("\n[Step 5] Building Feature Matrices for Validation Ablation...")
    
    # E8 Baseline Features (25,072 dims)
    ext_e8 = BETCCombinedExtractor(use_tfidf=True, use_emoji=True, use_sentiment=True, use_negation=True, use_expressive=False)
    X_e8_tr = ext_e8.fit_transform(train_raw, train_proc)
    X_e8_va = ext_e8.transform(val_raw, val_proc)

    # Advanced Negation Scope Features (16 dims)
    X_neg_scope_tr = sp.csr_matrix(neg_features_train)
    neg_features_val = extract_advanced_negation_features(val_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict)
    X_neg_scope_va = sp.csr_matrix(neg_features_val)

    # Configuration A: E8 + Negation Scope (25,072 + 16 = 25,088 dims)
    X_e8_scope_tr = sp.hstack([X_e8_tr, X_neg_scope_tr], format="csr")
    X_e8_scope_va = sp.hstack([X_e8_va, X_neg_scope_va], format="csr")

    # Configuration B: Full Enhanced Pipeline (TF-IDF + Emoji + EmotionLexicon + Advanced Negation Scope)
    # Load emotion lexicon features from Phase 3 (18 dims)
    from scripts.experiments.phase3_lexicon_study import extract_emotion_lexicon_features, EMOTION_LEXICON
    emo_vocab_set = {k: set(v) for k, v in EMOTION_LEXICON.items()}
    X_emo_tr = sp.csr_matrix(extract_emotion_lexicon_features(train_proc, emo_vocab_set))
    X_emo_va = sp.csr_matrix(extract_emotion_lexicon_features(val_proc, emo_vocab_set))

    ext_base_no_sent = BETCCombinedExtractor(use_tfidf=True, use_emoji=True, use_sentiment=False, use_negation=False, use_expressive=False)
    X_base_tr = ext_base_no_sent.fit_transform(train_raw, train_proc)
    X_base_va = ext_base_no_sent.transform(val_raw, val_proc)

    # Stack: Base (TF-IDF + Emoji) + EmotionLexicon (18) + Advanced Negation Scope (16)
    X_full_opt_tr = sp.hstack([X_base_tr, X_emo_tr, X_neg_scope_tr], format="csr")
    X_full_opt_va = sp.hstack([X_base_va, X_emo_va, X_neg_scope_va], format="csr")

    freq_order, freq_names, _ = compute_label_order(Y_train, TARGET_COLS)

    def eval_model(X_tr, X_va, name):
        print(f"\n--- Testing: {name} ---")
        t0 = time.time()
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="saga", random_state=42)
        chain = ClassifierChain(clf, order=freq_order, random_state=42)
        chain.fit(X_tr, Y_train)
        P_val = chain.predict_proba(X_va)
        thresholds = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
        Y_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
        m = compute_metrics(Y_val, Y_pred, TARGET_COLS)
        fit_t = time.time() - t0
        print(f"  Val Macro-F1: {m['macro_f1']:.4f} | Micro-F1: {m['micro_f1']:.4f} | Time: {fit_t:.1f}s")
        print("  Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in m["per_class"].items()))
        m["fit_time"] = fit_t
        m["dim"] = X_tr.shape[1]
        return m

    # 6. Run Ablation Evaluations
    print("\n[Step 6] Running Ablation Models on Validation Split...")
    ablation_results = {}
    ablation_results["E8_Baseline"] = eval_model(X_e8_tr, X_e8_va, "E8 Baseline (Emoji + Sent + Neg)")
    ablation_results["E8_Plus_NegScope"] = eval_model(X_e8_scope_tr, X_e8_scope_va, "E8 + Negation Scope Features")
    ablation_results["Full_Optimal_Pipeline"] = eval_model(X_full_opt_tr, X_full_opt_va, "Full Enhanced Pipeline (Emoji + EmotionLexicon + NegScope)")

    # 7. Summary Comparison Table
    print("\n" + "=" * 75)
    print("PHASE 4 ABLATION SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Configuration':<34} {'Macro-F1':>10} {'Micro-F1':>10} {'Δ vs E8':>10} {'Disgust':>9} {'Surprise':>9}")
    print("-" * 75)
    base_macro = ablation_results["E8_Baseline"]["macro_f1"]
    for k, v in ablation_results.items():
        delta = v["macro_f1"] - base_macro
        s = "+" if delta >= 0 else ""
        disg = v["per_class"]["disgust"]["f1"]
        surp = v["per_class"]["surprise"]["f1"]
        print(f"{k:<34} {v['macro_f1']:>10.4f} {v['micro_f1']:>10.4f} {s}{delta:>9.4f} {disg:>9.4f} {surp:>9.4f}")
    print("-" * 75)

    # 8. Save Comprehensive Deliverables
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "negation_vocabulary": {
            "total_negations": len(EXPANDED_NEGATIONS),
            "total_intensifiers": len(EXPANDED_INTENSIFIERS),
            "texts_with_negation": n_with_neg,
            "texts_with_intensifier": n_with_intens,
            "texts_with_emotion_scope": n_with_scope,
            "scope_breakdown_by_emotion": scope_stats
        },
        "qualitative_examples": examples,
        "ablation_results": ablation_results
    }

    out_json = os.path.join(RESULTS_DIR, "negation_study_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport successfully saved to: {out_json}")

if __name__ == "__main__":
    main()
