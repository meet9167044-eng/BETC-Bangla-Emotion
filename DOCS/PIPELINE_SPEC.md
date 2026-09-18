# PIPELINE_SPEC.md

This file preserves the complete BETC architecture as originally specified
in `BETC_Bangla_Emotion_Architecture_and_Implementation_Plan.docx`. It is
**unchanged** by the dataset revision in `DATASET_SPEC.md`; only the input
data source changed, not the pipeline itself. Tags follow the legend in
`PROJECT_SPEC.md` Section 4.

## 1. Core Hypothesis

**[FINALIZED]** Emotional evidence in short, noisy Bangla text is carried
jointly by (a) word-level lexical choice and (b) character-level surface
form (spelling variants, inflection, informal typing). The six target
emotions are not independent — predicting one provides useful conditioning
information for predicting the next.

## 2. Architecture Overview

```
COMBINED, HARMONIZED BANGLA EMOTION DATASET   (see DATASET_SPEC.md)
                    ↓
              PREPROCESSING
  ┌─────────────────────────────────────────┐
  │ Unicode normalization (Bangla-specific)  │
  │ Noise / URL / @mention removal           │
  │ Preserve negations (না, নাই, নেই, নয়)      │
  │ Preserve intensifiers (খুব, অনেক, একদম)     │
  └─────────────────────────────────────────┘
                    ↓
            FEATURE EXTRACTION
  ┌─────────────────────────────────────────┐
  │ Word TF-IDF   | n-grams=(1,2) | max=10,000 │
  │                      +                    │
  │ Character TF-IDF | n-grams=(3,5)          │
  │   analyzer=char_wb | max=15,000           │
  └─────────────────────────────────────────┘
                    ↓
  COMBINED FEATURE MATRIX (scipy sparse hstack, 25,000 dims)
                    ↓
        MULTI-LABEL MODEL: CLASSIFIER CHAIN
                    ↓
                BASE CLASSIFIER
     Logistic Regression | C=1.0 | class_weight=balanced
                    ↓
     PER-CLASS THRESHOLD OPTIMIZATION (validation set)
                    ↓
                TEST PREDICTION
                    ↓
   MACRO-F1 + MICRO-F1 + PER-CLASS F1
   (+ Hamming Loss, Jaccard, Subset Accuracy)
```

## 3. Component-by-Component Design

### 3.1 Preprocessing **[FINALIZED]**
- Normalize Unicode using a Bangla-specific normalizer (e.g.
  `bnunicodenormalizer`), not a generic NFC/NFKC pass — generic
  normalization mishandles Bangla conjuncts and reph.
- Strip URLs, @mentions, and HTML noise.
- Preserve a manually curated negation word list and intensifier word
  list — these must **never** be removed by any stopword-filtering step.
- Normalize whitespace and repeated punctuation.
- **[PROPOSAL]** The exact negation/intensifier word lists must be
  written as explicit, version-controlled files
  (`configs/negations_bn.txt`, `configs/intensifiers_bn.txt`), manually
  reviewed against real sample sentences before use — not generated
  ad hoc by an LLM without human review (see `RESEARCH_RULES.md`).

### 3.2 Word-Level TF-IDF **[INITIAL-HP]**
- `TfidfVectorizer(ngram_range=(1,2), max_features=10000)`.
- Fit **only** on the training split (see `EVALUATION_PROTOCOL.md`).

### 3.3 Character-Level TF-IDF **[INITIAL-HP]**
- `TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=15000)`.
- Fit **only** on the training split.

### 3.4 Feature Fusion **[FINALIZED]**
- Concatenate word and character TF-IDF matrices via
  `scipy.sparse.hstack` into one combined feature matrix (25,000
  dimensions at the initial hyperparameter values above).
- No dimensionality reduction (SVD/PCA) in the initial version.
  **[EXPERIMENTAL]** SVD/PCA is a candidate ablation only, not a default.

### 3.5 Classifier Chain and Label Ordering **[FINALIZED mechanism / EXPERIMENTAL ordering]**
- Use `sklearn.multioutput.ClassifierChain` over the target emotion
  labels in this fixed order: **anger, disgust, fear, joy, sadness,
  surprise**. Source-only labels are not model outputs.
- **[INITIAL-HP]** Default chain order: descending label frequency in
  the training set (most frequent emotion predicted first).
- **[EXPERIMENTAL]** Alternative orders (correlation-based, random
  averaged over multiple seeds) are ablation variables — see
  `EXPERIMENT_PLAN.md`. The order actually used for a given run must
  always be logged.

### 3.6 Base Classifier: Logistic Regression **[INITIAL-HP]**
- `LogisticRegression(C=1.0, class_weight='balanced')` for every link in
  the chain.

### 3.7 Per-Class Threshold Optimization **[FINALIZED mechanism]**
- After fitting on the training split, obtain `predict_proba` on the
  validation split only.
- For each target emotion independently, sweep thresholds
  **[INITIAL-HP]** from 0.05 to 0.95 in steps of 0.01, selecting the
  threshold that maximizes that emotion's F1 on the validation set.
- Store the resulting per-class thresholds as an artifact
  (`artifacts/thresholds.json`) — never recompute or "peek" at test-set
  probabilities when choosing thresholds.

### 3.8 Test Prediction and Evaluation **[FINALIZED]**
- Apply the fitted chain and the stored per-class thresholds to the
  untouched test split **exactly once**. See `EVALUATION_PROTOCOL.md`
  for the full leakage-prevention protocol.

## 4. Mathematical Formulation

**TF-IDF weight:**
```
w(t,d) = tf(t,d) · idf(t),   idf(t) = log[(1+n)/(1+df(t))] + 1
```

**Combined feature vector:**
```
x_d = [ x_word(d) ; x_char(d) ]  ∈  R^(10,000 + 15,000)
```

**Chain label order:**
```
π = (π₁, π₂, …, π_K),  fixed by descending training-set label frequency  [INITIAL-HP]
```

**Chained conditional probability:**
```
P(y_π(k)=1 | x) = σ( w_π(k)ᵀ [x ; y_π(1), …, y_π(k-1)] + b_π(k) )
```

**Per-classifier objective (L2-regularized logistic loss):**
```
L_k = −Σᵢ [ θᵢ · ( yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ) ) ]  +  (1/C) · ||w_k||²
```
where `θᵢ` is the per-example weight induced by `class_weight='balanced'`.

**Per-class threshold selection:**
```
τ_k = argmax_{τ∈[0,1]}  F1( y_k , 1[p_k ≥ τ] ; validation set )
```

**Final prediction:**
```
ŷ_k = 1[ p_k ≥ τ_k ]   for k = 1 … K
```

**Total training objective (sum of per-link losses):**
```
L_total = Σ_{k=1}^{K} L_k
```

## 5. Pseudocode

```
text = preprocess(raw_text)               # unicode norm, noise removal,
                                            # negation/intensifier preservation

X_word = word_tfidf.fit_transform(train_text)      # (1,2)-grams, 10k features
X_char = char_tfidf.fit_transform(train_text)      # (3,5)-grams char_wb, 15k features
X = hstack([X_word, X_char])                       # combined 25k-dim features

order = labels_sorted_by_frequency_desc(Y_train)
chain = ClassifierChain(
    LogisticRegression(C=1.0, class_weight='balanced'),
    order=order
).fit(X_train, Y_train)

P_val = chain.predict_proba(X_val)
FOR each emotion k in target label set:
    thresholds[k] = argmax_tau F1(Y_val[:,k], P_val[:,k] >= tau)

P_test = chain.predict_proba(X_test)
Y_pred = P_test >= thresholds                       # per-class thresholds applied

report(macro_f1, micro_f1, per_class_f1, hamming_loss, jaccard)
```

## 6. Recommended Initial Hyperparameters

**[INITIAL-HP]** — all values below are starting points, not final:

| Component | Initial value | Tune later |
|---|---|---|
| Word TF-IDF n-grams | (1,2) | (1,1), (1,3) |
| Word TF-IDF max_features | 10,000 | 5,000 – 20,000 |
| Char TF-IDF n-grams | (3,5) | (2,4), (3,6) |
| Char TF-IDF max_features | 15,000 | 10,000 – 25,000 |
| Char analyzer | char_wb | char |
| Logistic Regression C | 1.0 | 0.1, 1, 10, 100 |
| class_weight | balanced | balanced / none |
| Chain label order | descending frequency | correlation-based, random-averaged |
| Threshold search step | 0.01 | 0.05 (coarse pass first) |
| Train/Val/Test split | 70/15/15 | 80/10/10 |

## 7. What Is Intentionally Excluded From BETC **[FINALIZED]**

These are deliberate scope boundaries, not oversights:

- No BERT, RoBERTa, Bangla-BERT, IndicBERT, XLM-R, or any pretrained
  Transformer encoder.
- No pretrained word embeddings (Word2Vec, FastText, GloVe) trained on
  external corpora.
- No external LLM or embedding API calls.
- No deep sequence models (LSTM, BiLSTM, CNN encoders) inside BETC
  itself — these appear only as externally reported baseline numbers in
  `EXPERIMENT_PLAN.md`, never retrained here unless explicitly
  authorized in a future phase.
- No dimensionality reduction (SVD/PCA) by default in Version 1.

## 8. Complexity and Feasibility Notes **[LIT / PROPOSAL]**

For `n` training examples, combined feature dimension `d = 25,000`
(at initial hyperparameters), and `K` chain links, each Logistic
Regression fit is approximately `O(n · d)` per solver iteration, and the
full chain is approximately `O(K · n · d)`. TF-IDF vectorization is
approximately `O(n · average document length)`. This is expected to be
feasible on a standard CPU or a free-tier Colab instance without GPU
acceleration; the main practical cost is memory for the sparse
25,000-dimension matrices. This is a design expectation, not a measured
benchmark — no runtime numbers exist yet **[NO-RESULT]**.

## 9. Contribution Framing **[PROPOSAL]**

The intended framing for any report or paper: BETC combines word- and
character-level TF-IDF fusion, explicit label-dependency modeling via a
fixed-order Classifier Chain, and per-class threshold calibration. Each
individual component (TF-IDF, Classifier Chain, Logistic Regression) is
well-established; the contribution is the specific combination applied
to harmonized, combined Bangla emotion data, evaluated against both
classical and (externally reported) neural baselines. **No performance
claim beyond this framing may be made until experiments in
`EXPERIMENT_PLAN.md` produce results.**
