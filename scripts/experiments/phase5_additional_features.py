"""
scripts/experiments/phase5_additional_features.py
=================================================
BETC Improvement Plan — Phase 5: Add More Feature Groups

Systematically evaluates auxiliary feature groups one at a time:
  - 5.1: Character elongation and expressive informal spelling (raw text)
  - 5.2: Hashtag features (count, ratio, presence)
  - 5.3: Script proportions and code-mixing features (Bangla, Latin, Digits)
  - 5.4: Normalized text statistics (TTR / Lexical diversity, log lengths)
  - 5.5: Spelling variation & normalization divergence (raw vs normalized diffs)
  - 5.6: Lightweight morphological / verbal inflection markers
  - Combined Winner: Top performing feature subsets merged into the optimal pipeline

Protocol:
  - Evaluated on canonical validation split with ClassifierChain.
  - Threshold optimization on validation split.
"""

import os
import sys
import io
import re
import json
import time
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MaxAbsScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

from src.models.betc import TARGET_COLS, compute_label_order
from src.models.evaluate import compute_metrics, optimize_thresholds, apply_thresholds
from src.features.feature_combination import BETCCombinedExtractor
from scripts.experiments.phase3_lexicon_study import extract_emotion_lexicon_features, EMOTION_LEXICON
from scripts.experiments.phase4_negation_study import (
    EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, load_emotion_lexicon,
    extract_advanced_negation_features
)

DATA_DIR = os.path.join(WORKSPACE, "Data", "processed")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase5_features")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Feature Extractor Definitions ───────────────────────────────────────────

# Regex patterns
_RE_ELONGATED = re.compile(r"(.)\1{2,}", re.UNICODE)
_RE_HASHTAG   = re.compile(r"#\S+")
_BANGLA_RANGE = range(0x0980, 0x0A00)
_RE_LATIN     = re.compile(r"[a-zA-Z]")
_RE_DIGITS    = re.compile(r"[\d০-৯]")

def extract_group_5_1_elongation(raw_texts):
    """5.1: Character elongation (raw text) — 4 features"""
    feats = []
    for t in raw_texts:
        words = t.split()
        n_words = max(len(words), 1)
        elong_words = sum(1 for w in words if _RE_ELONGATED.search(w))
        total_elong_runs = len(_RE_ELONGATED.findall(t))
        feats.append([
            float(elong_words),
            1.0 if elong_words > 0 else 0.0,
            float(total_elong_runs),
            float(elong_words / n_words)
        ])
    return np.array(feats, dtype=np.float32)

def extract_group_5_2_hashtags(raw_texts):
    """5.2: Hashtags — 3 features"""
    feats = []
    for t in raw_texts:
        tags = _RE_HASHTAG.findall(t)
        n_tags = len(tags)
        total_tag_chars = sum(len(tag) for tag in tags)
        n_chars = max(len(t), 1)
        feats.append([
            float(n_tags),
            1.0 if n_tags > 0 else 0.0,
            float(total_tag_chars / n_chars)
        ])
    return np.array(feats, dtype=np.float32)

def extract_group_5_3_codemixing(raw_texts):
    """5.3: Script proportion and code-mixing — 4 features"""
    feats = []
    for t in raw_texts:
        n_chars = max(len(t), 1)
        n_bn = sum(1 for c in t if ord(c) in _BANGLA_RANGE)
        n_lat = len(_RE_LATIN.findall(t))
        n_dig = len(_RE_DIGITS.findall(t))
        is_mixed = 1.0 if (n_bn >= 3 and n_lat >= 3) else 0.0
        feats.append([
            float(n_bn / n_chars),
            float(n_lat / n_chars),
            float(n_dig / n_chars),
            is_mixed
        ])
    return np.array(feats, dtype=np.float32)

def extract_group_5_4_textstats(raw_texts):
    """5.4: Normalized text statistics (TTR, log lengths) — 4 features"""
    feats = []
    for t in raw_texts:
        words = t.split()
        n_words = len(words)
        n_chars = len(t)
        ttr = len(set(words)) / max(n_words, 1)
        avg_wlen = (n_chars / max(n_words, 1)) / 15.0  # normalized
        log_chars = np.log1p(n_chars) / 10.0          # range ~[0, 1]
        log_words = np.log1p(n_words) / 6.0           # range ~[0, 1]
        feats.append([float(ttr), float(avg_wlen), float(log_chars), float(log_words)])
    return np.array(feats, dtype=np.float32)

def extract_group_5_5_spelling_var(raw_texts, proc_texts):
    """5.5: Normalization divergence (informal variation) — 3 features"""
    feats = []
    for raw, proc in zip(raw_texts, proc_texts):
        raw_words = raw.split()
        proc_words = proc.split()
        n_words = max(len(raw_words), 1)
        diff_count = sum(1 for rw, pw in zip(raw_words, proc_words) if rw != pw)
        diff_count += abs(len(raw_words) - len(proc_words))
        diff_ratio = diff_count / n_words
        feats.append([
            float(diff_count),
            1.0 if diff_count > 0 else 0.0,
            float(diff_ratio)
        ])
    return np.array(feats, dtype=np.float32)

def extract_group_5_6_verbal_inflection(proc_texts):
    """5.6: Common Bangla verbal inflection markers — 4 features"""
    # Past (-লাম, -ছিল), Present continuous (-ছে, -ছেন), Future (-ব, -বে)
    _PAST_PAT = re.compile(r"\b\S+(লাম|ছিল|লেন)\b")
    _PRES_PAT = re.compile(r"\b\S+(ছে|ছেন|ছি)\b")
    _FUT_PAT  = re.compile(r"\b\S+(ব|বে|বেন)\b")
    feats = []
    for t in proc_texts:
        words = t.split()
        n_words = max(len(words), 1)
        past_c = len(_PAST_PAT.findall(t))
        pres_c = len(_PRES_PAT.findall(t))
        fut_c  = len(_FUT_PAT.findall(t))
        total_v = past_c + pres_c + fut_c
        feats.append([
            float(past_c / n_words),
            float(pres_c / n_words),
            float(fut_c / n_words),
            1.0 if total_v > 0 else 0.0
        ])
    return np.array(feats, dtype=np.float32)

def main():
    print("=" * 75)
    print("BETC Phase 5 — Evaluating Auxiliary Feature Groups (5.1 – 5.6)")
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

    freq_order, freq_names, _ = compute_label_order(Y_train, TARGET_COLS)

    # 2. Extract Phase 4 Champion Base (TF-IDF + Emoji + EmotionLexicon + NegScope)
    print("\n[Step 2] Extracting Phase 4 Optimal Foundation Features...")
    t0 = time.time()
    ext_base = BETCCombinedExtractor(use_tfidf=True, use_emoji=True, use_sentiment=False, use_negation=False, use_expressive=False)
    X_base_tr = ext_base.fit_transform(train_raw, train_proc)
    X_base_va = ext_base.transform(val_raw, val_proc)

    emo_vocab_set = {k: set(v) for k, v in EMOTION_LEXICON.items()}
    X_emo_tr = sp.csr_matrix(extract_emotion_lexicon_features(train_proc, emo_vocab_set))
    X_emo_va = sp.csr_matrix(extract_emotion_lexicon_features(val_proc, emo_vocab_set))

    emo_dict = load_emotion_lexicon()
    neg_tr = extract_advanced_negation_features(train_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict)
    neg_va = extract_advanced_negation_features(val_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict)
    X_neg_tr = sp.csr_matrix(neg_tr)
    X_neg_va = sp.csr_matrix(neg_va)

    # Champion Matrix (from Phase 4: Val Macro-F1 = 0.4902, Micro-F1 = 0.5820)
    X_champ_tr = sp.hstack([X_base_tr, X_emo_tr, X_neg_tr], format="csr")
    X_champ_va = sp.hstack([X_base_va, X_emo_va, X_neg_va], format="csr")
    print(f"  Champion Foundation Shape: {X_champ_tr.shape} in {time.time()-t0:.1f}s")

    # 3. Model Evaluation Helper
    def eval_model(X_tr, X_va, desc):
        print(f"\n--- Testing: {desc} ---")
        t_start = time.time()
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="saga", random_state=42)
        chain = ClassifierChain(clf, order=freq_order, random_state=42)
        chain.fit(X_tr, Y_train)
        P_val = chain.predict_proba(X_va)
        thresholds = optimize_thresholds(P_val, Y_val, label_names=TARGET_COLS)
        Y_pred = apply_thresholds(P_val, thresholds, TARGET_COLS)
        m = compute_metrics(Y_val, Y_pred, TARGET_COLS)
        fit_t = time.time() - t_start
        print(f"  Val Macro-F1: {m['macro_f1']:.4f} | Micro-F1: {m['micro_f1']:.4f} | Time: {fit_t:.1f}s")
        print("  Per-class: " + ", ".join(f"{k}={v['f1']:.3f}" for k, v in m["per_class"].items()))
        m["fit_time"] = fit_t
        m["dim"] = X_tr.shape[1]
        return m

    # 4. Step-by-Step Feature Group Ablations
    results = {}

    # Benchmark: Champion Base
    results["Baseline_Champion"] = eval_model(X_champ_tr, X_champ_va, "Champion Base (TF-IDF + Emoji + EmoLex + NegScope)")

    # 5.1 Elongation
    f51_tr = MaxAbsScaler().fit_transform(extract_group_5_1_elongation(train_raw))
    f51_va = MaxAbsScaler().fit_transform(extract_group_5_1_elongation(val_raw))
    X_51_tr = sp.hstack([X_champ_tr, sp.csr_matrix(f51_tr)], format="csr")
    X_51_va = sp.hstack([X_champ_va, sp.csr_matrix(f51_va)], format="csr")
    results["Group_5.1_Elongation"] = eval_model(X_51_tr, X_51_va, "Group 5.1: Character Elongation (+4 dims)")

    # 5.2 Hashtags
    f52_tr = MaxAbsScaler().fit_transform(extract_group_5_2_hashtags(raw_texts=train_raw))
    f52_va = MaxAbsScaler().fit_transform(extract_group_5_2_hashtags(raw_texts=val_raw))
    X_52_tr = sp.hstack([X_champ_tr, sp.csr_matrix(f52_tr)], format="csr")
    X_52_va = sp.hstack([X_champ_va, sp.csr_matrix(f52_va)], format="csr")
    results["Group_5.2_Hashtags"] = eval_model(X_52_tr, X_52_va, "Group 5.2: Hashtags (+3 dims)")

    # 5.3 Code-mixing & Script Proportions
    f53_tr = MaxAbsScaler().fit_transform(extract_group_5_3_codemixing(train_raw))
    f53_va = MaxAbsScaler().fit_transform(extract_group_5_3_codemixing(val_raw))
    X_53_tr = sp.hstack([X_champ_tr, sp.csr_matrix(f53_tr)], format="csr")
    X_53_va = sp.hstack([X_champ_va, sp.csr_matrix(f53_va)], format="csr")
    results["Group_5.3_CodeMixing"] = eval_model(X_53_tr, X_53_va, "Group 5.3: Script Proportion & Code-Mixing (+4 dims)")

    # 5.4 Text Statistics
    f54_tr = MaxAbsScaler().fit_transform(extract_group_5_4_textstats(train_raw))
    f54_va = MaxAbsScaler().fit_transform(extract_group_5_4_textstats(val_raw))
    X_54_tr = sp.hstack([X_champ_tr, sp.csr_matrix(f54_tr)], format="csr")
    X_54_va = sp.hstack([X_champ_va, sp.csr_matrix(f54_va)], format="csr")
    results["Group_5.4_TextStats"] = eval_model(X_54_tr, X_54_va, "Group 5.4: Normalized Text Stats / TTR (+4 dims)")

    # 5.5 Spelling Variation
    f55_tr = MaxAbsScaler().fit_transform(extract_group_5_5_spelling_var(train_raw, train_proc))
    f55_va = MaxAbsScaler().fit_transform(extract_group_5_5_spelling_var(val_raw, val_proc))
    X_55_tr = sp.hstack([X_champ_tr, sp.csr_matrix(f55_tr)], format="csr")
    X_55_va = sp.hstack([X_champ_va, sp.csr_matrix(f55_va)], format="csr")
    results["Group_5.5_SpellingVar"] = eval_model(X_55_tr, X_55_va, "Group 5.5: Spelling & Normalization Divergence (+3 dims)")

    # 5.6 Verbal Inflection
    f56_tr = MaxAbsScaler().fit_transform(extract_group_5_6_verbal_inflection(train_proc))
    f56_va = MaxAbsScaler().fit_transform(extract_group_5_6_verbal_inflection(val_proc))
    X_56_tr = sp.hstack([X_champ_tr, sp.csr_matrix(f56_tr)], format="csr")
    X_56_va = sp.hstack([X_champ_va, sp.csr_matrix(f56_va)], format="csr")
    results["Group_5.6_VerbalMarkers"] = eval_model(X_56_tr, X_56_va, "Group 5.6: Verbal Inflection Markers (+4 dims)")

    # Combined Winner: Merge positive contributors
    # Identifying groups with non-negative delta
    base_macro = results["Baseline_Champion"]["macro_f1"]
    winning_blocks_tr = [X_champ_tr]
    winning_blocks_va = [X_champ_va]
    winning_names = []

    group_map = [
        ("5.1 Elongation", f51_tr, f51_va, results["Group_5.1_Elongation"]["macro_f1"]),
        ("5.3 CodeMixing", f53_tr, f53_va, results["Group_5.3_CodeMixing"]["macro_f1"]),
        ("5.4 TextStats", f54_tr, f54_va, results["Group_5.4_TextStats"]["macro_f1"]),
        ("5.5 SpellingVar", f55_tr, f55_va, results["Group_5.5_SpellingVar"]["macro_f1"]),
    ]
    for g_name, tr_b, va_b, score in group_map:
        if score >= base_macro:
            winning_blocks_tr.append(sp.csr_matrix(tr_b))
            winning_blocks_va.append(sp.csr_matrix(va_b))
            winning_names.append(g_name)

    if len(winning_blocks_tr) > 1:
        X_opt_comb_tr = sp.hstack(winning_blocks_tr, format="csr")
        X_opt_comb_va = sp.hstack(winning_blocks_va, format="csr")
        desc_comb = f"Combined Winners ({', '.join(winning_names)})"
        results["Combined_Phase5_Winners"] = eval_model(X_opt_comb_tr, X_opt_comb_va, desc_comb)

    # 5. Summary Table
    print("\n" + "=" * 80)
    print("PHASE 5 AUXILIARY FEATURES ABLATION SUMMARY")
    print("=" * 80)
    print(f"{'Feature Group':<30} {'Macro-F1':>10} {'Micro-F1':>10} {'Δ vs Base':>11} {'Decision':>12}")
    print("-" * 80)
    for k, v in results.items():
        delta = v["macro_f1"] - base_macro
        s = "+" if delta >= 0 else ""
        dec = "Keep" if delta > 0 else ("Neutral" if abs(delta) < 0.0005 else "Drop")
        print(f"{k:<30} {v['macro_f1']:>10.4f} {v['micro_f1']:>10.4f} {s}{delta:>10.4f} {dec:>12}")
    print("-" * 80)

    # Save Results
    out_path = os.path.join(RESULTS_DIR, "phase5_ablation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_champion_macro_f1": base_macro,
            "results": {k: {"macro_f1": v["macro_f1"], "micro_f1": v["micro_f1"], "dim": v.get("dim"), "per_class": v["per_class"]} for k, v in results.items()}
        }, f, indent=2)
    print(f"\nReport saved to: {out_path}")

if __name__ == "__main__":
    main()
