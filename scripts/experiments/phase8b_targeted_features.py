"""
scripts/experiments/phase8b_targeted_features.py
==================================================
BETC Improvement Plan — Phase 8b: Targeted Feature Engineering
(Interrogatives & Discourse Contrast Connectives)

Objectives:
  Target the two most error-prone labels from Phase 6 error analysis:
    - Disgust (F1=0.345): 69% of FPs and 64% of FNs involve negation;
      disgust often implicit in contrastive sentences.
    - Surprise (F1=0.331): 50% of FPs are rhetorical questions; many FNs
      lack explicit surprise vocabulary.

  Feature Groups Tested (independently + combined):
    Group A — Bengali Interrogatives & Rhetorical Question Markers  → target Surprise
    Group B — Discourse Contrast / Adversative Connectives          → target Disgust
    Group C — Intensifier–Emotion Co-occurrence Bigrams             → emotion intensity
    Group D — Combined A + B + C

  Protocol:
    - Phase 5 Champion baseline: TF-IDF(25k)+Emoji+6-EmoLex+NegScope+Hashtags = 25,089 dims
    - ClassifierChain (LR, class_weight='balanced', C=1.0, saga, seed=42)
    - Phase 7 calibrated thresholds re-optimized per feature configuration (step=0.001)
    - Evaluation: Validation split only (canonical). Test set not touched.
    - Selection rule: accept only if Macro-F1 > 0.4913 AND no label drops > 0.01 vs S0.

Deliverables:
    - results/ablations/phase8b_targeted_features/phase8b_report.json
    - results/ablations/phase8b_targeted_features/phase8b_report.md
    - configs/phase8b_thresholds.json  (if a group beats S0)
"""

import os
import sys
import json
import time
import re
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, WORKSPACE)

from src.models.betc import TARGET_COLS, compute_label_order
from src.models.evaluate import compute_metrics, apply_thresholds
from src.features.feature_combination import BETCCombinedExtractor
from scripts.experiments.phase3_lexicon_study import extract_emotion_lexicon_features, EMOTION_LEXICON
from scripts.experiments.phase4_negation_study import (
    EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, load_emotion_lexicon,
    extract_advanced_negation_features,
)
from scripts.experiments.phase5_additional_features import extract_group_5_2_hashtags

DATA_DIR    = os.path.join(WORKSPACE, "Data", "processed")
CONFIGS_DIR = os.path.join(WORKSPACE, "configs")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "ablations", "phase8b_targeted_features")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── S0 Reference (Phase 7 champion) ─────────────────────────────────────────
S0_MACRO = 0.4913
S0_MICRO = 0.5825
S0_PER_CLASS = {
    "anger": 0.549, "disgust": 0.345, "fear": 0.393,
    "joy": 0.826,   "sadness": 0.503, "surprise": 0.331,
}
REGRESSION_TOLERANCE = 0.010   # >0.01 drop on any label = rejected


# ─── Group A: Bengali Interrogatives & Rhetorical Question Markers ────────────
# Targeting Surprise false negatives: many are rhetorical questions without
# explicit surprise vocabulary. Sources: common Bangla wh-question words
# documented in Bangla grammar (Bhuiyan 1982; Ray et al. 1966).

# WH-question words (transliteration in parentheses)
_BANGLA_WH_WORDS = frozenset([
    "কেন",      # keno   – why
    "কী",       # ki     – what (formal)
    "কি",       # ki     – what (colloquial / yes-no marker)
    "কীভাবে",  # kibhabe – how
    "কেমন",     # kemon  – how (state)
    "কখন",      # kokhon – when
    "কোথায়",   # kothay – where
    "কোথা",     # kotha  – where (archaic/informal)
    "কার",      # kar    – whose
    "কাকে",     # kake   – whom
    "কতটা",     # kotota – how much/how far
    "কত",       # koto   – how many/how much
    "সত্যি",    # shotti – really (rhetorical intensifier-question word)
    "সত্যিই",   # really indeed (emphasis rhetorical)
    "কোনো",     # kono   – any (negative rhetorical)
    "আসলে",     # ashle  – actually (discourse-level question marker)
])

# Punctuation patterns
_RE_QMARK       = re.compile(r"\?+")
_RE_EXCL        = re.compile(r"!+")
_RE_QEXCL       = re.compile(r"[?!]{2,}")   # mixed ?! sequences
_RE_ELLIPSIS    = re.compile(r"\.{3,}|…")    # ... or Unicode ellipsis
_RE_TRIPLE_QMRK = re.compile(r"\?{2,}")      # double/triple ?

def extract_group_A_interrogative(texts):
    """
    Group A: Bengali Interrogatives & Rhetorical Question Markers
    8 features:
      0: count of WH-question words
      1: binary – any WH-word present
      2: count of '?' characters (normalized by length)
      3: binary – any '?' present
      4: binary – any '!' present
      5: binary – mixed ?! / expressive punctuation clusters
      6: binary – triple/double '???' present
      7: binary – ellipsis '...' / '…' present
    """
    feats = []
    for t in texts:
        words  = t.split()
        n_char = max(len(t), 1)

        wh_count     = sum(1 for w in words if w in _BANGLA_WH_WORDS)
        has_wh       = 1.0 if wh_count > 0 else 0.0
        qmark_cnt    = len(_RE_QMARK.findall(t))
        has_qmark    = 1.0 if qmark_cnt > 0 else 0.0
        has_excl     = 1.0 if _RE_EXCL.search(t) else 0.0
        has_qexcl    = 1.0 if _RE_QEXCL.search(t) else 0.0
        has_triple_q = 1.0 if _RE_TRIPLE_QMRK.search(t) else 0.0
        has_ellipsis = 1.0 if _RE_ELLIPSIS.search(t) else 0.0

        feats.append([
            float(wh_count),
            has_wh,
            float(qmark_cnt) / n_char * 10,   # rate × 10 to normalise
            has_qmark,
            has_excl,
            has_qexcl,
            has_triple_q,
            has_ellipsis,
        ])
    return np.array(feats, dtype=np.float32)


# ─── Group B: Discourse Contrast / Adversative Connectives ───────────────────
# Targeting Disgust false positives/negatives.
# Contrast markers indicate the second clause revises the first clause's sentiment.
# E.g., "I was happy BUT the behavior was disgusting."
# Source: Bangla adversative conjunctions (Islam 2018 Bangla syntax; Chatterji 1926).

_CONTRAST_CONNECTIVES = frozenset([
    "কিন্তু",    # kintu   – but
    "তবে",       # tobe    – however / but
    "যদিও",      # jodio   – although / even though
    "তবুও",      # tobuo   – yet / still / nevertheless
    "বরং",       # borong  – rather / instead
    "অথচ",       # othoch  – yet / whereas (adversative)
    "বরঞ্চ",     # boroncch– rather (emphatic)
    "পরন্তু",    # porontu – but (literary)
    "তথাপি",     # tothapi – nevertheless (literary)
    "এরপরও",     # erporeo – even after that
    "তারপরও",    # tarporeo– after that still
    "সত্ত্বেও",  # shotteo – despite
    "সত্ত্বে",   # shotte  – despite (stem form)
])

# Emoji-level polarity shift detector (very lightweight)
# If negative sentiment score drops after a contrast marker, it signals disgust context.

def extract_group_B_contrast(raw_texts, processed_texts, emo_vocab_set):
    """
    Group B: Discourse Contrast / Adversative Connectives
    10 features:
      0: count of contrast connective tokens
      1: binary – any contrast connective present
      2: binary – negative emotion word in second clause (after contrast marker)
      3: binary – positive emotion word in first clause AND negative in second
      4: binary – disgust word in second clause after contrast
      5: binary – anger word in second clause after contrast
      6: binary – sadness word in second clause after contrast
      7: count of disgust-lexicon words in text (all positions)
      8: count of total negative-emotion words (disgust + anger + fear)
      9: ratio of contrast markers to total words
    """
    disgust_vocab  = emo_vocab_set.get("disgust",  set())
    anger_vocab    = emo_vocab_set.get("anger",    set())
    fear_vocab     = emo_vocab_set.get("fear",     set())
    joy_vocab      = emo_vocab_set.get("joy",      set())
    sadness_vocab  = emo_vocab_set.get("sadness",  set())
    negative_vocab = disgust_vocab | anger_vocab | fear_vocab | sadness_vocab
    positive_vocab = joy_vocab

    feats = []
    for raw_t, proc_t in zip(raw_texts, processed_texts):
        words      = proc_t.split()
        n_words    = max(len(words), 1)

        # Count contrast connectives
        cc_count   = sum(1 for w in words if w in _CONTRAST_CONNECTIVES)
        has_cc     = 1.0 if cc_count > 0 else 0.0

        # Second-clause analysis: find first contrast connective position
        neg_in_second    = 0.0
        pos_first_neg_sec = 0.0
        dis_in_second    = 0.0
        ang_in_second    = 0.0
        sad_in_second    = 0.0

        cc_idx = None
        for i, w in enumerate(words):
            if w in _CONTRAST_CONNECTIVES:
                cc_idx = i
                break

        if cc_idx is not None and cc_idx + 1 < len(words):
            first_clause  = set(words[:cc_idx])
            second_clause = set(words[cc_idx + 1:])
            if negative_vocab & second_clause:
                neg_in_second = 1.0
            if (positive_vocab & first_clause) and (negative_vocab & second_clause):
                pos_first_neg_sec = 1.0
            if disgust_vocab & second_clause:
                dis_in_second = 1.0
            if anger_vocab & second_clause:
                ang_in_second = 1.0
            if sadness_vocab & second_clause:
                sad_in_second = 1.0

        # Total disgust and negative emotion counts
        disgust_total  = float(sum(1 for w in words if w in disgust_vocab))
        neg_emo_total  = float(sum(1 for w in words if w in negative_vocab))
        cc_ratio       = float(cc_count) / n_words

        feats.append([
            float(cc_count),
            has_cc,
            neg_in_second,
            pos_first_neg_sec,
            dis_in_second,
            ang_in_second,
            sad_in_second,
            disgust_total,
            neg_emo_total,
            cc_ratio,
        ])
    return np.array(feats, dtype=np.float32)


# ─── Group C: Intensifier–Emotion Co-occurrence Bigrams ──────────────────────
# Targeting emotion intensity discrimination.
# Check if an intensifier occurs within a 2-token window BEFORE an emotion lexicon word.
# E.g., "ভীষণ ভয়" (very fear), "অনেক আনন্দ" (much joy).
# This is a linguistically grounded proximity feature.

_INTENSIFIER_SET = frozenset(EXPANDED_INTENSIFIERS)

def extract_group_C_intensifier_bigrams(processed_texts, emo_vocab_set):
    """
    Group C: Intensifier-Emotion Co-occurrence Bigrams
    For each of 6 emotions: count of times any intensifier appears within
    a 2-token window before an emotion lexicon word.
    12 features total:
      0-5: count of intensifier-emotion bigrams per emotion (anger, disgust, fear, joy, sadness, surprise)
      6-11: binary presence per emotion
    """
    emotion_order = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    feats = []
    for text in processed_texts:
        words = text.split()
        n = len(words)
        counts = {emo: 0 for emo in emotion_order}

        for i, w in enumerate(words):
            if w in _INTENSIFIER_SET:
                # Check the next 1–2 tokens
                for j in range(i + 1, min(i + 3, n)):
                    candidate = words[j]
                    for emo in emotion_order:
                        if candidate in emo_vocab_set.get(emo, set()):
                            counts[emo] += 1

        row = []
        for emo in emotion_order:
            row.append(float(counts[emo]))         # count
        for emo in emotion_order:
            row.append(1.0 if counts[emo] > 0 else 0.0)  # binary
        feats.append(row)
    return np.array(feats, dtype=np.float32)


# ─── Threshold sweep (inline, no external dep) ───────────────────────────────
def sweep_thresholds_local(P, Y, step=0.001, label_names=None):
    """Per-class threshold sweep on validation data to maximise F1."""
    if label_names is None:
        label_names = [f"label_{i}" for i in range(Y.shape[1])]
    thresholds = {}
    for i, name in enumerate(label_names):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.955, step):
            preds = (P[:, i] >= t).astype(int)
            f1 = f1_score(Y[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t  = round(float(t), 4)
        thresholds[name] = best_t
    return thresholds


# ─── Build Phase 5 Champion Base Feature Matrix ──────────────────────────────
def build_champion_base(train_raw, train_proc, val_raw, val_proc):
    """Returns (X_tr, X_va) for the Phase 5 champion feature set (25,089 dims)."""
    emo_vocab_set = {k: set(v) for k, v in EMOTION_LEXICON.items()}
    cache_tr = os.path.join(RESULTS_DIR, "X_base_tr.npz")
    cache_va = os.path.join(RESULTS_DIR, "X_base_va.npz")
    if os.path.exists(cache_tr) and os.path.exists(cache_va):
        print("  Found cached base matrices on disk. Loading directly...")
        return sp.load_npz(cache_tr), sp.load_npz(cache_va), emo_vocab_set

    emo_dict      = load_emotion_lexicon()

    ext_base = BETCCombinedExtractor(
        use_tfidf=True, use_emoji=True, use_sentiment=False,
        use_negation=False, use_expressive=False
    )
    X_base_tr = ext_base.fit_transform(train_raw, train_proc)
    X_base_va = ext_base.transform(val_raw, val_proc)

    X_emo_tr  = sp.csr_matrix(extract_emotion_lexicon_features(train_proc, emo_vocab_set))
    X_emo_va  = sp.csr_matrix(extract_emotion_lexicon_features(val_proc, emo_vocab_set))

    X_neg_tr  = sp.csr_matrix(extract_advanced_negation_features(
        train_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict))
    X_neg_va  = sp.csr_matrix(extract_advanced_negation_features(
        val_proc, EXPANDED_NEGATIONS, EXPANDED_INTENSIFIERS, emo_dict))

    scaler_hash = MaxAbsScaler()
    X_hash_tr = sp.csr_matrix(scaler_hash.fit_transform(extract_group_5_2_hashtags(train_raw)))
    X_hash_va = sp.csr_matrix(scaler_hash.transform(extract_group_5_2_hashtags(val_raw)))

    X_tr = sp.hstack([X_base_tr, X_emo_tr, X_neg_tr, X_hash_tr], format="csr")
    X_va = sp.hstack([X_base_va, X_emo_va, X_neg_va, X_hash_va], format="csr")
    return X_tr, X_va, emo_vocab_set


# ─── Fit & evaluate one configuration ─────────────────────────────────────────
def run_config(name, X_tr, Y_tr, X_va, Y_va, freq_order):
    """Fit ClassifierChain, optimize thresholds on val, return metrics dict."""
    t0  = time.time()
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000,
        solver="saga", random_state=42
    )
    chain = ClassifierChain(clf, order=freq_order, random_state=42)
    chain.fit(X_tr, Y_tr)
    P_va   = chain.predict_proba(X_va)
    taus   = sweep_thresholds_local(P_va, Y_va, step=0.001, label_names=TARGET_COLS)
    Y_pred = apply_thresholds(P_va, taus, TARGET_COLS)
    m      = compute_metrics(Y_va, Y_pred, TARGET_COLS)
    elapsed = time.time() - t0

    macro_f1 = m["macro_f1"]
    micro_f1 = m["micro_f1"]
    per_cls  = {k: round(v["f1"], 4) for k, v in m["per_class"].items()}

    print(f"\n  [{name}]")
    print(f"    Dims      : {X_tr.shape[1]:,}")
    print(f"    Macro-F1  : {macro_f1:.4f}  (Δ={macro_f1 - S0_MACRO:+.4f} vs S0)")
    print(f"    Micro-F1  : {micro_f1:.4f}  (Δ={micro_f1 - S0_MICRO:+.4f} vs S0)")
    print("    Per-class : " + " | ".join(f"{k}={v:.3f}" for k, v in per_cls.items()))
    print(f"    Thresholds: { {k: round(v,3) for k, v in taus.items()} }")
    print(f"    Time      : {elapsed:.0f}s")

    # Regression check
    regressions = {
        k: round(per_cls[k] - S0_PER_CLASS[k], 4)
        for k in TARGET_COLS
        if per_cls[k] - S0_PER_CLASS[k] < -REGRESSION_TOLERANCE
    }
    if regressions:
        print(f"    ⚠  Regression (>0.01): {regressions}")
    else:
        print("    ✅ No regression violations")

    return {
        "name":        name,
        "dims":        int(X_tr.shape[1]),
        "macro_f1":    round(macro_f1, 4),
        "micro_f1":    round(micro_f1, 4),
        "per_class":   per_cls,
        "thresholds":  {k: round(v, 4) for k, v in taus.items()},
        "delta_macro": round(macro_f1 - S0_MACRO, 4),
        "delta_micro": round(micro_f1 - S0_MICRO, 4),
        "regressions": regressions,
        "safe":        len(regressions) == 0 and macro_f1 > S0_MACRO,
        "time_s":      round(elapsed, 1),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("BETC Phase 8b — Targeted Feature Engineering")
    print("Interrogatives (A), Contrast Connectives (B), Intensifier Bigrams (C)")
    print("=" * 75)

    # 1. Load data
    print("\n[Step 1] Loading canonical data splits...")
    df_tr = pd.read_csv(os.path.join(DATA_DIR, "train",      "train_preprocessed.csv"),      low_memory=False)
    df_va = pd.read_csv(os.path.join(DATA_DIR, "validation", "validation_preprocessed.csv"), low_memory=False)
    df_tr = df_tr.reset_index(drop=True)
    df_va = df_va.reset_index(drop=True)

    train_raw  = df_tr["text"].fillna("").astype(str).tolist()
    train_proc = df_tr["processed_text"].fillna("").astype(str).tolist()
    val_raw    = df_va["text"].fillna("").astype(str).tolist()
    val_proc   = df_va["processed_text"].fillna("").astype(str).tolist()

    Y_tr = df_tr[TARGET_COLS].values.astype(int)
    Y_va = df_va[TARGET_COLS].values.astype(int)
    freq_order, _, _ = compute_label_order(Y_tr, TARGET_COLS)
    print(f"  Train: {len(df_tr):,} | Val: {len(df_va):,}")

    # 2. Build Phase 5 Champion base
    print("\n[Step 2] Building Phase 5 Champion base (25,089 dims)...")
    t0 = time.time()
    X_base_tr, X_base_va, emo_vocab_set = build_champion_base(
        train_raw, train_proc, val_raw, val_proc
    )
    print(f"  Base dims: {X_base_tr.shape[1]:,}  ({time.time()-t0:.1f}s)")

    # 3. Build candidate feature arrays (val+train, separately)
    print("\n[Step 3] Extracting candidate feature groups...")

    # Group A — Interrogatives (8 features)
    scaler_A = MaxAbsScaler()
    fA_tr = sp.csr_matrix(scaler_A.fit_transform(extract_group_A_interrogative(train_raw)))
    fA_va = sp.csr_matrix(scaler_A.transform(extract_group_A_interrogative(val_raw)))
    print(f"  Group A (Interrogative): {fA_tr.shape[1]} features")

    # Group B — Contrast Connectives (10 features)
    scaler_B = MaxAbsScaler()
    fB_tr = sp.csr_matrix(scaler_B.fit_transform(
        extract_group_B_contrast(train_raw, train_proc, emo_vocab_set)))
    fB_va = sp.csr_matrix(scaler_B.transform(
        extract_group_B_contrast(val_raw, val_proc, emo_vocab_set)))
    print(f"  Group B (Contrast):      {fB_tr.shape[1]} features")

    # Group C — Intensifier Bigrams (12 features)
    scaler_C = MaxAbsScaler()
    fC_tr = sp.csr_matrix(scaler_C.fit_transform(
        extract_group_C_intensifier_bigrams(train_proc, emo_vocab_set)))
    fC_va = sp.csr_matrix(scaler_C.transform(
        extract_group_C_intensifier_bigrams(val_proc, emo_vocab_set)))
    print(f"  Group C (Intens-Bigram): {fC_tr.shape[1]} features")

    # 4. Evaluate all configurations
    print("\n[Step 4] Evaluating configurations...")
    print(f"\n  S0 BASELINE (Phase 7 champion) — Macro-F1={S0_MACRO} | Micro-F1={S0_MICRO}")

    partial_path = os.path.join(RESULTS_DIR, "phase8b_partial.json")
    results = {}
    if os.path.exists(partial_path):
        try:
            with open(partial_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"  Loaded {len(results)} previously completed configuration(s) from {partial_path}")
        except Exception:
            results = {}

    configs = [
        ("S0_baseline", X_base_tr, X_base_va),
        ("A_interrogative", sp.hstack([X_base_tr, fA_tr], format="csr"), sp.hstack([X_base_va, fA_va], format="csr")),
        ("B_contrast", sp.hstack([X_base_tr, fB_tr], format="csr"), sp.hstack([X_base_va, fB_va], format="csr")),
        ("C_intensifier_bigrams", sp.hstack([X_base_tr, fC_tr], format="csr"), sp.hstack([X_base_va, fC_va], format="csr")),
        ("D_combined_ABC", sp.hstack([X_base_tr, fA_tr, fB_tr, fC_tr], format="csr"), sp.hstack([X_base_va, fA_va, fB_va, fC_va], format="csr")),
    ]

    for cfg_name, xtr, xva in configs:
        if cfg_name in results:
            print(f"\n  [{cfg_name}] — Already completed (Macro-F1={results[cfg_name]['macro_f1']:.4f}). Skipping re-run.")
            continue
        r = run_config(cfg_name, xtr, Y_tr, xva, Y_va, freq_order)
        results[cfg_name] = r
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 5. Selection
    print("\n[Step 5] Selecting best safe configuration...")
    safe_configs = {k: v for k, v in results.items() if v["safe"] and k != "S0_baseline"}
    if safe_configs:
        best_name = max(safe_configs, key=lambda k: safe_configs[k]["macro_f1"])
        best      = safe_configs[best_name]
        print(f"  ✅ Best safe config: [{best_name}]")
        print(f"     Macro-F1={best['macro_f1']:.4f} (Δ={best['delta_macro']:+.4f})")
        print(f"     Micro-F1={best['micro_f1']:.4f} (Δ={best['delta_micro']:+.4f})")
    else:
        best_name = "S0_baseline"
        best      = results["S0_baseline"]
        print("  ⚪ No new group beat S0 safely — S0 (champion) retained.")

    # unconstrained best
    unc_best = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"  Unconstrained best: [{unc_best}] Macro-F1={results[unc_best]['macro_f1']:.4f}")

    # 6. Save outputs
    print("\n[Step 6] Saving outputs...")

    # JSON report
    report = {
        "phase":       "Phase 8b",
        "date":        pd.Timestamp.now().strftime("%Y-%m-%d"),
        "s0_reference": {
            "macro_f1": S0_MACRO, "micro_f1": S0_MICRO, "per_class": S0_PER_CLASS,
        },
        "selected":       best_name,
        "results":        results,
    }
    json_path = os.path.join(RESULTS_DIR, "phase8b_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  JSON report: {json_path}")

    # Frozen thresholds
    if best_name != "S0_baseline":
        thresh_out = {
            "phase":      "Phase 8b",
            "selected":   best_name,
            "dims":       best["dims"],
            "macro_f1":   best["macro_f1"],
            "micro_f1":   best["micro_f1"],
            "thresholds": best["thresholds"],
            "note":       "Thresholds optimised on validation split ONLY. Test set not touched.",
        }
        thresh_path = os.path.join(CONFIGS_DIR, "phase8b_thresholds.json")
        with open(thresh_path, "w", encoding="utf-8") as f:
            json.dump(thresh_out, f, indent=2, ensure_ascii=False)
        print(f"  Frozen thresholds: {thresh_path}")

    # Markdown report
    md = [
        "# BETC — Phase 8b: Targeted Feature Engineering Report",
        "",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        f"**Baseline (S0):** Phase 5 Champion + Phase 7 τ* | Macro-F1=`{S0_MACRO}` | Micro-F1=`{S0_MICRO}`",
        "",
        "---",
        "",
        "## Strategy Summary",
        "",
        "| Config | Description | Dims | Val Macro-F1 | Δ Macro | Val Micro-F1 | Δ Micro | Safe? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    desc_map = {
        "S0_baseline":          "Phase 5 Champ + Phase 7 τ* (reference)",
        "A_interrogative":      "Base + Bengali interrogatives & punctuation (8 feats)",
        "B_contrast":           "Base + Discourse contrast connectives (10 feats)",
        "C_intensifier_bigrams":"Base + Intensifier-emotion co-occ bigrams (12 feats)",
        "D_combined_ABC":       "Base + Groups A + B + C (30 feats)",
    }
    for k, v in results.items():
        selected = " ✅" if k == best_name else ""
        safe_sym = "✅" if v["safe"] else ("⚪" if k == "S0_baseline" else "❌")
        md.append(
            f"| **{k}**{selected} | {desc_map.get(k, k)} | {v['dims']:,} | "
            f"`{v['macro_f1']:.4f}` | `{v['delta_macro']:+.4f}` | "
            f"`{v['micro_f1']:.4f}` | `{v['delta_micro']:+.4f}` | {safe_sym} |"
        )
    md += [
        "",
        "## Per-Label F1 Breakdown",
        "",
        "| Config | anger | disgust | fear | joy | sadness | surprise |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k, v in results.items():
        pc = v["per_class"]
        md.append(
            f"| **{k}** | {pc['anger']:.3f} | {pc['disgust']:.3f} | "
            f"{pc['fear']:.3f} | {pc['joy']:.3f} | "
            f"{pc['sadness']:.3f} | {pc['surprise']:.3f} |"
        )
    md += [
        "",
        "## Selection",
        "",
        f"**Selected:** `{best_name}`  ",
        f"**Val Macro-F1:** `{best['macro_f1']:.4f}` | **Val Micro-F1:** `{best['micro_f1']:.4f}`",
        "",
        "> Selection rule: Macro-F1 must exceed 0.4913 AND no label may drop >0.01 F1 below S0.",
        "",
        "## Deliverables",
        "- `scripts/experiments/phase8b_targeted_features.py`",
        "- `results/ablations/phase8b_targeted_features/phase8b_report.json`",
        "- `tests/test_phase8b_targeted_features.py`",
    ]
    md_path = os.path.join(RESULTS_DIR, "phase8b_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"  Markdown report: {md_path}")

    # 7. Summary
    print("\n" + "=" * 75)
    print("PHASE 8b COMPLETE")
    print("=" * 75)
    print(f"  Entry:    Macro-F1={S0_MACRO:.4f} | Micro-F1={S0_MICRO:.4f}")
    print(f"  Selected: [{best_name}]")
    print(f"            Macro-F1={best['macro_f1']:.4f} | Micro-F1={best['micro_f1']:.4f}")
    print(f"            Δ Macro={best['delta_macro']:+.4f} | Δ Micro={best['delta_micro']:+.4f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
