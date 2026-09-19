# Phase 3 — Dataset Label Harmonization Analysis
**Project:** BETC — Bangla Emotion TF-IDF Classifier Chain  
**Phase:** Phase 3 — Design/Analysis Only  
**Date:** 2026-09-19  
**Status:** DESIGN COMPLETE — AWAITING USER REVIEW AND DECISIONS BEFORE PHASE 4  
**Governing documents:** `DOCS/DATASET_SPEC.md`, `DOCS/IMPLEMENTATION_PLAN.md`, `DOCS/RESEARCH_RULES.md`  
**Input:** `results/dataset_audit/` (all Phase 1 artifacts)  
**Outputs (this phase):** `results/harmonization/` (all files in this folder)

> [!IMPORTANT]
> **No raw data has been modified in this phase.** No rows deleted. No labels changed. No datasets merged. No deduplication applied. No models trained. All analysis is design-only. Execution (Phase 4) requires explicit user approval.

---

## 1. Final Target Taxonomy

**[FROZEN]** The BETC system outputs exactly six emotions in this fixed order:

| # | Target Label | Ekman (1992) | Notes |
|---|---|---|---|
| 1 | `anger` | Anger | Present in all 3 datasets |
| 2 | `disgust` | Disgust | **Absent from EmoNoBa** — critical gap |
| 3 | `fear` | Fear | Present in all 3 datasets; severely underrepresented in MONOVAB |
| 4 | `joy` | Happiness/Joy | Present in EmoNoBa + UBMEC; via `enjoyment` mapping in MONOVAB |
| 5 | `sadness` | Sadness | Present in all 3 datasets |
| 6 | `surprise` | Surprise | Present in all 3 datasets |

Labels not in this taxonomy: `love` (EmoNoBa), `contempt` (MONOVAB). These are **excluded**. Neither maps to any target label.

---

## 2. Native Labels in Each Dataset

### 2.1 EmoNoBa (22,739 rows — Multi-label)
| Native Column | dtype | Count Positive | Prevalence | Target Status |
|---|---|---|---|---|
| `Love` | int64 | 4,588 | 20.18% | **EXCLUDED** — not in taxonomy |
| `Joy` | int64 | 10,112 | 44.47% | → `joy` (DIRECT) |
| `Surprise` | int64 | 1,086 | 4.78% | → `surprise` (DIRECT) |
| `Anger` | int64 | 4,478 | 19.69% | → `anger` (DIRECT) |
| `Sadness` | int64 | 5,681 | 24.98% | → `sadness` (DIRECT) |
| `Fear` | int64 | 401 | 1.76% | → `fear` (DIRECT) |
| *(Disgust)* | *(absent)* | *(none)* | 0% | **UNANNOTATED** — strategy required |

**Annotation system:** Junto Emotion Wheel variant. EmoNoBa replaced the Ekman `disgust` category with `love`, so **no EmoNoBa example was ever asked about disgust**.

### 2.2 UBMEC (13,436 rows — Single-label categorical)
| Native Class Value | Count | Prevalence | Target Status |
|---|---|---|---|
| `joy` | 3,467 | 25.80% | → `joy` (DIRECT) |
| `sadness` | 2,683 | 19.97% | → `sadness` (DIRECT) |
| `anger` | 2,480 | 18.46% | → `anger` (DIRECT) |
| `disgust` | 2,079 | 15.47% | → `disgust` (DIRECT) |
| `surprise` | 1,366 | 10.17% | → `surprise` (DIRECT) |
| `fear` | 1,361 | 10.13% | → `fear` (DIRECT) |

**Annotation system:** Ekman's six basic emotions — one and only one class per example. No multi-label rows exist in UBMEC.

### 2.3 MONOVAB (10,224 rows — Multi-label binary)
| Native Column | dtype | Count Positive | Prevalence | Target Status |
|---|---|---|---|---|
| `anger` | int64 | 4,499 | 44.00% | → `anger` (DIRECT) |
| `contempt` | int64 | 2,960 | 28.95% | **EXCLUDED** — not in taxonomy |
| `disgust` | int64 | 2,067 | 20.22% | → `disgust` (DIRECT) |
| `enjoyment` | int64 | 1,601 | 15.66% | → `joy` (MAPPED — PROPOSED) |
| `sadness` | int64 | 1,188 | 11.62% | → `sadness` (DIRECT) |
| `surprise` | int64 | 100 | 0.98% | → `surprise` (DIRECT) |
| `fear` | int64 | 56 | 0.55% | → `fear` (DIRECT) |

---

## 3. Complete Native-to-Target Label Mapping Table

| Dataset | Native Label | Target Label | Status | Mapping Type | Rationale | Notes |
|---|---|---|---|---|---|---|
| EmoNoBa | `Joy` | `joy` | **[FROZEN] DIRECT** | Exact taxonomy match | Semantically identical to Ekman Happiness/Joy | Highest prevalence (44.47%) |
| EmoNoBa | `Anger` | `anger` | **[FROZEN] DIRECT** | Exact taxonomy match | Ekman Anger | 19.69% |
| EmoNoBa | `Sadness` | `sadness` | **[FROZEN] DIRECT** | Exact taxonomy match | Ekman Sadness | 24.98% |
| EmoNoBa | `Surprise` | `surprise` | **[FROZEN] DIRECT** | Exact taxonomy match | Ekman Surprise | 4.78% |
| EmoNoBa | `Fear` | `fear` | **[FROZEN] DIRECT** | Exact taxonomy match | Ekman Fear | 1.76% — severe minority |
| EmoNoBa | `Love` | *NONE* | **[FROZEN] EXCLUDED** | Out-of-taxonomy | Junto Wheel emotion; no Ekman equivalent; NOT mapped to Disgust | 2,277 Love-only rows need strategy |
| EmoNoBa | *(Disgust)* | `disgust` | **[FROZEN: excluded from source] UNANNOTATED** | No native annotation | EmoNoBa never collected Disgust; strategy A/B/C/D required | 22,739 rows affected; **[REQUIRES USER DECISION]** |
| UBMEC | `joy` | `joy` | **[FROZEN] DIRECT** | Exact + one-hot conversion | Ekman Happiness/Joy | 25.80% |
| UBMEC | `sadness` | `sadness` | **[FROZEN] DIRECT** | Exact + one-hot conversion | Ekman Sadness | 19.97% |
| UBMEC | `anger` | `anger` | **[FROZEN] DIRECT** | Exact + one-hot conversion | Ekman Anger | 18.46% |
| UBMEC | `disgust` | `disgust` | **[FROZEN] DIRECT** | Exact + one-hot conversion | Ekman Disgust | 15.47% — key Disgust source |
| UBMEC | `surprise` | `surprise` | **[FROZEN] DIRECT** | Exact + one-hot conversion | Ekman Surprise | 10.17% |
| UBMEC | `fear` | `fear` | **[FROZEN] DIRECT** | Exact + one-hot conversion | Ekman Fear | 10.13% |
| MONOVAB | `anger` | `anger` | **[FROZEN] DIRECT** | Exact taxonomy match | Ekman Anger | 44.00% — dominant |
| MONOVAB | `disgust` | `disgust` | **[FROZEN] DIRECT** | Exact taxonomy match | Ekman Disgust | 20.22% |
| MONOVAB | `enjoyment` | `joy` | **[PROPOSED] MAPPED** | Semantic near-equivalence | Enjoyment ≈ Ekman Happiness/Joy; see §7 | **[REQUIRES USER DECISION]** to freeze |
| MONOVAB | `fear` | `fear` | **[FROZEN] DIRECT** | Exact taxonomy match | Ekman Fear | 0.55% — extreme minority |
| MONOVAB | `sadness` | `sadness` | **[FROZEN] DIRECT** | Exact taxonomy match | Ekman Sadness | 11.62% |
| MONOVAB | `surprise` | `surprise` | **[FROZEN] DIRECT** | Exact taxonomy match | Ekman Surprise | 0.98% — extreme minority |
| MONOVAB | `contempt` | *NONE* | **[FROZEN] EXCLUDED** | Out-of-taxonomy | Not in Ekman six; no safe mapping to any target | 2,128 contempt-only rows need strategy |

---

## 4. The EmoNoBa Disgust Problem

### 4.1 Nature of the Problem

EmoNoBa's annotation scheme is based on the **Junto Emotion Wheel**, which uses the categories: `Love`, `Joy`, `Surprise`, `Anger`, `Sadness`, `Fear`. This wheel **replaces Ekman's `Disgust` with `Love`**. As a result:

- **No EmoNoBa annotator was ever asked "Does this text express disgust?"**
- Every EmoNoBa row has an unobserved Disgust annotation — not a zero, not a one, but **unknown**.
- This is a **structural annotation gap**, not a question of whether disgust exists in the text.

This is critically different from an EmoNoBa row that has `Disgust=0`. A `Disgust=0` annotation would mean an annotator saw the text, considered whether it expressed disgust, and marked it absent. **That did not happen for any EmoNoBa row.**

### 4.2 Impact on Corpus

Without a strategy, 22,739 rows (49.0% of the raw combined corpus) have an unknown Disgust value. This is the single largest methodological challenge in the dataset harmonization.

### 4.3 Strategy Analysis

#### Strategy A — Treat Missing Disgust as 0

**What it does:** Assign `disgust=0` to all 22,739 EmoNoBa rows, creating a complete six-column matrix.

| Metric | Value |
|---|---|
| Affected rows | 22,739 |
| Rows usable for Disgust training | 46,399 |
| True positive Disgust examples | 4,146 (UBMEC: 2,079 + MONOVAB: 2,067) |
| False negative Disgust examples (estimate) | Unknown — some EmoNoBa examples may genuinely express disgust |
| Implementation complexity | LOW |
| Annotation accuracy risk | HIGH |

**Training impact:** The Disgust link in the ClassifierChain sees 22,739 training examples labelled as negative that were never actually annotated. If, say, 5% of EmoNoBa texts genuinely express disgust (a plausible but unverified estimate), approximately 1,137 examples introduce label noise specifically for the Disgust classifier.

**Validation impact:** Disgust threshold is optimized on a validation set that includes EmoNoBa rows with assumed `disgust=0`. The threshold may be miscalibrated if actual Disgust content exists in those rows.

**Macro-F1 impact:** Disgust F1 appears in the Macro-F1 average; if the Disgust classifier is trained on noisy labels, its F1 will be systematically underestimated (the model cannot learn disgust from examples where disgust is present but labelled 0). This lowers overall Macro-F1 in a non-transparent way.

**Logistic Regression + ClassifierChain impact:** The `class_weight='balanced'` setting compensates for class imbalance but not label noise. With 22,739 additional Disgust=0 examples from EmoNoBa, the effective negative weight for Disgust increases, requiring even more class-weight correction.

**DATASET_SPEC.md status:** This strategy is **documented as the operational rule** in DATASET_SPEC.md Sec 3, with the explicit proviso that it must be described as an assumption and that Experiment F5 must run the sensitivity analysis. It is not described as scientifically accurate, only as operationally required.

**Advantages:** Simple; produces complete six-column matrix; compatible with sklearn ClassifierChain without modification; aligns with DATASET_SPEC.md's documented rule.

**Disadvantages:** Systematic label noise for Disgust; violates RESEARCH_RULES.md rule 9 unless explicitly logged (which is already required by DATASET_SPEC.md); Disgust F1 may be misleading.

**Reproducibility:** High.

---

#### Strategy B — Label-Coverage Mask (Unknown/Unobserved)

**What it does:** Assigns a special "unknown" marker to Disgust for all 22,739 EmoNoBa rows. The Disgust loss term is masked to zero for EmoNoBa rows during training.

| Metric | Value |
|---|---|
| Affected rows | 22,739 (masked from Disgust training) |
| Rows contributing Disgust signal | 23,660 (UBMEC 13,436 + MONOVAB 10,224) |
| True positive Disgust examples | 4,146 |
| Implementation complexity | HIGH |
| Annotation accuracy risk | LOW |

**Training impact:** The Disgust binary cross-entropy loss is computed only over UBMEC and MONOVAB examples. EmoNoBa examples still contribute to Anger/Joy/Sadness/Surprise/Fear training signals normally. Requires a custom training wrapper for `ClassifierChain` since sklearn's implementation does not natively support per-sample label masking.

**Validation impact:** Disgust threshold optimization uses only UBMEC+MONOVAB validation examples. Effective validation set for Disgust is smaller, which may yield a noisier threshold estimate.

**Test impact:** EmoNoBa test examples are excluded from Disgust F1 computation (or flagged with a caveat that Disgust is extrapolated from a cross-domain predictor). The reported Disgust F1 is computed only over the subset where ground truth exists.

**Macro-F1 impact:** If Disgust is excluded from EmoNoBa evaluation rows, the Macro-F1 computation must be defined carefully — either averaged only over the labelled subset, or imputed differently. This creates a non-standard evaluation that must be documented.

**Logistic Regression + ClassifierChain impact:** Masked loss changes the training objective for the Disgust link. In the chain, later links (trained after Disgust in the chain order) receive chain-augmented features including the Disgust prediction; if Disgust predictions are made for EmoNoBa rows at inference time (using the cross-domain predictor), those predictions feed into subsequent chain links. The quality of those downstream predictions depends on how well the Disgust predictor generalises from the UBMEC/MONOVAB domain to EmoNoBa's domain.

**Advantages:** Scientifically honest; Disgust classifier trained only on clean annotated data; no artificial negatives introduced.

**Disadvantages:** Requires substantial custom code; smaller Disgust training set; Disgust predictor must generalise across domain boundaries (EmoNoBa is YouTube/Facebook/Twitter; UBMEC/MONOVAB are different social media contexts); Macro-F1 becomes non-standard.

**Reproducibility:** Moderate — mask must be saved as a versioned artifact.

---

#### Strategy C — Exclude EmoNoBa from Disgust Training Only; Retain for Other 5 Emotions

**What it does:** Identical to Strategy B in terms of Disgust, but explicitly clarifies that EmoNoBa rows are retained for all other five emotion training signals while being excluded from Disgust.

| Metric | Value |
|---|---|
| Affected rows (Disgust exclusion) | 22,739 |
| EmoNoBa rows still used for Anger/Joy/Sadness/Surprise/Fear | 22,739 |
| Rows contributing Disgust signal | 23,660 |
| Implementation complexity | HIGH |
| Annotation accuracy risk | LOW |

**Training impact:** EmoNoBa contributes full training signal for 5 of 6 target emotions. The Disgust link in the chain is trained on UBMEC+MONOVAB only. When the Disgust classifier makes predictions on EmoNoBa test rows (for the purpose of chain conditioning and final Disgust output), it extrapolates across domain.

**Advantages:** Most principled; preserves EmoNoBa's valid annotations completely; separates what is known from what is inferred.

**Disadvantages:** Same implementation complexity as B; extrapolation risk; Disgust classifier has a smaller effective training domain.

**Reproducibility:** Moderate.

---

#### Strategy D — DATASET_SPEC.md Operational Rule + Mandatory Sensitivity Analysis (Recommended)

**What it does:** Executes Strategy A (Disgust=0 for EmoNoBa) as the primary training configuration, but makes Experiment F5 (sensitivity analysis) mandatory and reports two sets of Disgust metrics: (1) using the full corpus including EmoNoBa under the assumption, and (2) excluding EmoNoBa rows from Disgust scoring.

| Metric | Value |
|---|---|
| Affected rows | 22,739 (assume 0) |
| Rows usable for Disgust training | 46,399 |
| True positive Disgust examples | 4,146 |
| Implementation complexity | LOW (primary) + MEDIUM (F5 sensitivity) |
| Annotation accuracy risk | MEDIUM (label noise present but quantified) |

**Key distinction from A:** Strategy D is Strategy A with a built-in scientific safeguard: the assumption is acknowledged, documented, and tested. The F5 experiment (already defined in EXPERIMENT_PLAN.md) provides the mechanism to quantify how much the assumption inflates or deflates Disgust F1.

**Advantages over A:** Same implementation simplicity; produces two defensible sets of results rather than one misleadingly simple result; directly aligned with what DATASET_SPEC.md already requires; aligns with RESEARCH_RULES.md rule 9 (assumption must be logged and described as such).

**Recommendation:** Strategy D is the proposed approach because it (1) satisfies DATASET_SPEC.md's operational rule, (2) does not require custom sklearn code, (3) produces the F5 experiment already defined in EXPERIMENT_PLAN.md, and (4) is transparent about the limitation.

> [!IMPORTANT]
> **[REQUIRES USER DECISION]:** Confirm which strategy (A, B, C, or D) should be used. The recommended strategy is D.

---

## 5. The EmoNoBa Love Problem

### 5.1 Nature of the Problem

EmoNoBa's `Love` column is a valid annotation in the EmoNoBa taxonomy but is not part of the BETC target taxonomy. When `Love` is excluded:

- 4,588 examples had at least one `Love=1` annotation.
- **2,277 examples had `Love=1` as their ONLY active label.** For these 2,277 examples, all five target-compatible labels (`Joy`, `Surprise`, `Anger`, `Sadness`, `Fear`) are already zero in the original data.
- After Love is dropped and before Disgust is handled, these 2,277 rows have **zero known positive annotations for any target emotion**.

This is an annotation coverage problem: the text was annotated, but exclusively for a non-target emotion.

### 5.2 Strategy Analysis

#### LOVE-1 — Exclude Love-Only Rows (2,277 rows)

| | Value |
|---|---|
| Rows removed | 2,277 |
| EmoNoBa rows after exclusion | 20,462 |
| Combined corpus before dedup | ~44,122 |
| Label noise introduced | None |
| Information lost | 2,277 Bangla social-media texts |

**Analysis:** The cleanest approach. These 2,277 texts were annotated only for an emotion outside the target taxonomy. Including them as if they have known target emotions would introduce ambiguity. Excluding them produces a corpus where every retained EmoNoBa row has at least one known positive or at least provides valid negative labels for the five non-Love targets.

**Recommendation:** LOVE-1 is the recommended strategy. It avoids introducing ambiguity while losing only ~10% of EmoNoBa rows.

#### LOVE-2 — Retain with Masked Target Labels

**Analysis:** The texts are retained in the corpus with all six target labels masked as unknown. They contribute to TF-IDF vocabulary but not to any label training signal. This is the most honest approach but adds implementation complexity and effectively excludes these examples from evaluation anyway. The practical difference from LOVE-1 is minimal in terms of model training, so the additional complexity of masked labels is unlikely to pay off.

#### LOVE-3 — Retain as All-Zero Target Vector

**Analysis:** Assigns `[0,0,0,0,0,0]` to all six target labels for Love-only rows. This implicitly claims that annotators verified these texts were negative for all six target emotions — which did not happen. The original EmoNoBa annotators labelled these texts as expressing Love; they did not assess whether the texts also express Anger, Disgust, Fear, Joy, Sadness, or Surprise. Including them as all-zero examples teaches the classifier that "Love-expressing text → predict nothing," which may not be accurate and violates RESEARCH_RULES.md rule 9.

> [!IMPORTANT]
> **[REQUIRES USER DECISION]:** Which Love-only row strategy (LOVE-1, LOVE-2, or LOVE-3) to use. The recommended strategy is LOVE-1 (exclude 2,277 rows).

---

## 6. The MONOVAB Contempt Problem

### 6.1 Nature of the Problem

MONOVAB includes a `contempt` column (2,960 positive occurrences). Contempt is not in the BETC target taxonomy and must be excluded. Of those 2,960 contempt-positive rows, **2,128 have contempt as their ONLY active label across all seven native MONOVAB columns**.

After dropping `contempt`:
- 2,128 rows have zero active labels in the remaining six MONOVAB columns (anger, disgust, enjoyment, fear, sadness, surprise).
- This is analogous to the EmoNoBa Love-only problem.

**A critical additional risk:** Contempt and Disgust are semantically related emotions. In some emotion theories (Ekman, 1992; Izard, 1977), contempt is treated as a component or a variant of disgust. Texts expressing contempt may genuinely express disgust but were labelled as contempt-only by annotators. If Contempt-only rows are retained as all-zero target vectors, this may specifically introduce `disgust=0` label noise — the opposite of what is wanted given the EmoNoBa Disgust gap.

### 6.2 Strategy Analysis

#### CONTEMPT-1 — Exclude Contempt-Only Rows (Recommended)

| | Value |
|---|---|
| Rows removed | 2,128 |
| MONOVAB rows after exclusion | 8,096 |
| Label noise introduced | None |
| Contamination risk for Disgust | None |

**Analysis:** The cleanest approach. Removes only rows where the target emotion status is completely unknown. The remaining 8,096 MONOVAB rows all have at least one direct target emotion annotation.

**Recommendation:** CONTEMPT-1 is the recommended strategy, and it is especially important to avoid CONTEMPT-3 (all-zero) given the semantic overlap between contempt and disgust.

#### CONTEMPT-2 — Retain with Masked Labels

**Analysis:** Same considerations as LOVE-2. Scientifically honest but adds implementation complexity for minimal benefit over CONTEMPT-1.

#### CONTEMPT-3 — Retain as All-Zero Target Vector

**Analysis:** The riskiest option specifically because of the contempt-disgust semantic relationship. A text expressing contempt may genuinely express disgust; labelling it `disgust=0` introduces label noise for the most critical target label gap (Disgust). This is not recommended.

> [!IMPORTANT]
> **[REQUIRES USER DECISION]:** Which Contempt-only row strategy (CONTEMPT-1, CONTEMPT-2, or CONTEMPT-3) to use. The recommended strategy is CONTEMPT-1 (exclude 2,128 rows). CONTEMPT-3 is specifically discouraged due to the contempt-disgust semantic overlap.

---

## 7. MONOVAB `enjoyment` → `joy` Mapping Analysis

### 7.1 The Mapping

MONOVAB uses the label `enjoyment` where the BETC taxonomy uses `joy`. This requires an explicit mapping decision.

### 7.2 Semantic Analysis

**Is enjoyment equivalent to joy?**

- In Ekman's framework, the positive basic emotion is called **"happiness"** (1992) or **"joy"** in popularisations.
- Enjoyment is one specific form of the broader happiness/joy family. Fredrickson (2001) lists enjoyment as one of ten positive emotions, which also includes joy, interest, contentment, pride, amusement, inspiration, awe, love, and gratitude.
- In everyday Bangla emotional expression, texts that express enjoyment (`আনন্দ`, `ভালো লাগা`) are the same class of positive affect that EmoNoBa and UBMEC label as `joy`.
- MONOVAB's use of `enjoyment` rather than `joy` appears to be a naming choice by the MONOVAB dataset authors for the same Ekman Happiness/Joy category, not a distinct new emotion.

**Is this mapping exact or approximate?**

- **Not exactly equivalent** in formal emotion theory (enjoyment is a subcategory of joy/happiness, not its synonym).
- **Practically equivalent** for the purpose of Bangla social-media emotion classification, because both labels refer to the same observable emotional expression class in Bangla text.
- **Well-motivated** but not verified against MONOVAB's original annotation guidelines.

**What uncertainty remains?**

- If MONOVAB annotators used `enjoyment` to capture only a narrow subset of what EmoNoBa's `joy` covers (e.g., entertainment-specific positive affect, not general happiness), then mapping enjoyment→joy would underestimate Joy in MONOVAB.
- Without access to the original MONOVAB annotation guidelines, this cannot be confirmed.

### 7.3 Recommendation

The `enjoyment → joy` mapping is **[PROPOSED]** and should be treated as a semantic near-equivalence rather than an exact identity until MONOVAB's annotation documentation confirms the intent. For practical purposes, the mapping is the only defensible option — there is no other MONOVAB label that corresponds to Joy, and discarding `enjoyment` entirely would eliminate the 1,601 MONOVAB Joy signal examples.

> [!IMPORTANT]
> **[REQUIRES USER DECISION]:** Confirm whether to freeze `enjoyment → joy` as **[FROZEN]** or keep as **[PROPOSED]** pending documentation review. If rejected, MONOVAB provides no Joy signal.

---

## 8. UBMEC Single-Label → Six-Binary-Label Conversion

### 8.1 What the Conversion Does

UBMEC provides one categorical class string per row (e.g., `"anger"`, `"joy"`). The BETC training matrix requires a six-column binary label vector. The conversion is:

```
UBMEC row: classes = "anger"
→ [anger=1, disgust=0, fear=0, joy=0, sadness=0, surprise=0]

UBMEC row: classes = "disgust"
→ [anger=0, disgust=1, fear=0, joy=0, sadness=0, surprise=0]
```

### 8.2 Why This Conversion Is Valid

UBMEC is a **six-class mutually exclusive classification dataset**. In this type of dataset:
- An annotator assigned each text to exactly one emotion category.
- The zero values in the converted vector do not mean "annotators confirmed absence of these emotions" in the multi-label sense; they mean "this text was not assigned to these categories in a single-label scheme."
- However, because the six UBMEC classes are the same six Ekman emotions as the BETC targets, the conversion is a valid representation change. The one-hot vector encodes the same information as the categorical label.
- This conversion produces a **one-hot** multi-label representation: exactly one column is 1, the rest are 0. This is a degenerate case of the multi-label format.

### 8.3 What Must Be Documented

Per DATASET_SPEC.md Sec 3 and RESEARCH_RULES.md rule 21:
- **This conversion must never be described as UBMEC being "originally multi-label."** UBMEC is single-label; the six-column binary format is a representation artifact, not a reflection of the original annotation type.
- The zero values in UBMEC's converted rows mean "the observed category is not X" in a mutually exclusive scheme — not "annotators confirmed X is absent."
- This is a different semantic from EmoNoBa's and MONOVAB's multi-label zeros, which (for the five annotated labels) do reflect annotator assessments of absence.

**[FROZEN]** This conversion is finalized and straightforward. No user decision required.

---

## 9. Conflicting Duplicate Annotation Analysis

### 9.1 Summary from Phase 1 Audit

| Dataset | Texts with Conflicting Annotations | Total Conflicting Rows |
|---|---|---|
| **UBMEC** | 58 unique texts | >200 rows (some appear 14 times) |
| **MONOVAB** | 33 unique texts | ~80 rows |
| **EmoNoBa** | 8 unique texts | ~20 rows |
| **Total** | ~99 unique texts | ~300 rows |

### 9.2 Conflict Categories

**Type A — Same text, different single class (UBMEC):**  
The most extreme case: a text like "অসাধারণ" (meaning "extraordinary" or "wonderful") appears 4 times — twice labelled `joy`, twice labelled `surprise`. This is genuine annotator disagreement on an ambiguous short text. Because UBMEC is single-label, these rows are truly contradictory.

**Type B — Same text, different multi-label sets (MONOVAB):**  
A text appears twice with different binary label combinations. Example: same text labelled `anger=1, disgust=1` in one row and `disgust=1, sadness=1` in another. This reflects annotation inconsistency in a multi-label context.

**Type C — Same text, different multi-label sets (EmoNoBa):**  
Similar to Type B; a text appears in two different splits or twice in the same split with different label assignments.

**Type D — Mass duplication (UBMEC):**  
One UBMEC text appears 14 times across the dataset with 3 different class labels (anger, disgust, sadness). This is likely a data ingestion error or erroneous deduplication failure in the original UBMEC construction.

### 9.3 Why Conflicting Annotations Matter

1. **They inflate duplicate counts:** The 364 exact duplicate UBMEC rows include many conflicting-label duplicates, contributing to the +364 row discrepancy between raw UBMEC (13,436) and literature-reported UBMEC (13,072).
2. **They introduce label noise:** If duplicates with conflicting labels are retained, the classifier receives contradictory training signals for the same text.
3. **They complicate deduplication:** Simply removing duplicate texts is insufficient if different instances have different labels. A resolution policy is needed.
4. **They affect evaluation:** If the same text appears in both training and test (cross-split leakage), the evaluation is compromised regardless of label conflicts.

### 9.4 Candidate Resolution Strategies for Phase 4

| Strategy | Description | Advantage | Risk |
|---|---|---|---|
| **Drop all duplicates for conflicting texts** | Remove every row where a text appears with different labels | No label noise | Loses data; may remove non-conflicting occurrences |
| **Label union (multi-label only)** | For MONOVAB/EmoNoBa: merge label vectors with logical OR | Maximally inclusive | Can over-assign emotions; union may not reflect any individual annotator's intent |
| **Keep first occurrence by row index** | Keep the earliest occurrence; drop subsequent duplicates | Simple; deterministic | Arbitrary; first occurrence is not necessarily more accurate |
| **Manual review** | Flag conflicting texts for human adjudication | Most accurate | Not scalable for 99 texts; requires expertise |

> [!NOTE]
> Resolution of conflicting duplicates belongs to **Phase 4**. No resolution is applied here. The strategies above require user selection before Phase 4 begins.

---

## 10. Cross-Dataset Duplicate Analysis

### 10.1 Phase 1 Findings

| Pair | Shared Texts |
|---|---|
| EmoNoBa ∩ UBMEC | 10 texts |
| EmoNoBa ∩ MONOVAB | 12 texts |
| UBMEC ∩ MONOVAB | **45 texts** |
| All three datasets | 2 texts |
| **Total cross-dataset duplicates** | **65 texts** |

### 10.2 Why Cross-Dataset Duplicates Matter

**Train-test leakage risk:** When the three datasets are merged and then split into train/validation/test, the same text could appear in both training and test from different source datasets. If text X appears in EmoNoBa (assigned to train) and also in UBMEC (assigned to test), the model has already seen that text during training. This would inflate test metrics.

**Conflicting cross-source labels:** A text appearing in both UBMEC (single-label: `joy`) and MONOVAB (multi-label: `anger=1, disgust=1`) would carry contradictory labels. Which annotation is correct? The answer depends on source priority.

**Why UBMEC-MONOVAB overlap is highest (45 texts):** Both UBMEC and MONOVAB draw from Bangla social-media sources including political Facebook comments. Viral or widely-shared comments naturally appear in both corpora.

### 10.3 Implications for Pipeline

1. **Deduplication must happen before splitting** (DATASET_SPEC.md Sec 4, step 5 and EVALUATION_PROTOCOL.md rule 5). A text removed from one source dataset's occurrence must not also appear in the test split from another source.
2. **Source priority must be documented** if conflicting cross-source labels exist for the same text. Options: keep annotation from the source with richer label scheme; keep both occurrences in training only; take label union.
3. **The literature-reported 665 cross-dataset duplicates** from the prior published study is a **[VERIFY] / [LIT]** number. Our audit found only **65 exact cross-dataset text duplicates** using stripped text matching. The discrepancy may be due to near-duplicate detection (the prior study may have used approximate string matching), or to differences in the version of each dataset used. This project will report its own measured number, not copy the 665 figure.

> [!WARNING]
> Cross-dataset deduplication must occur before the train/validation/test split is created. Performing it after splitting would risk having a duplicated text in both train and test from different source datasets.

---

## 11. Candidate Harmonization Strategy Combinations

The following combinations represent the practical strategy space. All numbers are pre-deduplication estimates.

| Combination | Disgust Strategy | Love-Only | Contempt-Only | EmoNoBa Rows | UBMEC Rows | MONOVAB Rows | Combined Before Dedup | Estimated Final |
|---|---|---|---|---|---|---|---|---|
| **Recommended (R)** | D (A + F5) | LOVE-1 (exclude 2277) | CONTEMPT-1 (exclude 2128) | 20,462 | 13,436 | 8,096 | **41,994** | ~41,400 |
| **Literature-approximate** | D (A + F5) | LOVE-3 (all-zero) | CONTEMPT-1 (exclude 2128) | 22,739 | 13,436 | 8,096 | **44,271** | ~43,673 |
| **Maximum retention** | D (A + F5) | LOVE-3 (all-zero) | CONTEMPT-3 (all-zero) | 22,739 | 13,436 | 10,224 | **46,399** | ~45,800 |
| **Minimal noise** | B (masked) | LOVE-1 (exclude) | CONTEMPT-1 (exclude) | 20,462 (5 labels) | 13,436 | 8,096 | **41,994** | ~41,400 |

### Notes on the Literature-Approximate Combination

The published prior study combining EmoNoBa+UBMEC+MONOVAB reported 43,676 unique entries. The combination of Strategy D + LOVE-3 + CONTEMPT-1 yields an estimate of ~43,673 — extremely close. This suggests the prior study likely:
- Retained Love-only rows as all-zero target vectors
- Excluded Contempt-only rows
- Used Strategy A for EmoNoBa Disgust

This is documented as contextual information only. Our project must measure its own numbers independently, per RESEARCH_RULES.md rule 23.

---

## 12. Estimated Usable Rows Under Each Strategy

| Strategy Combination | Raw Rows Before Dedup | Est. Within-Dataset Dups | Est. Cross-Dataset Dups | Est. Degenerate Rows | Est. Final Usable Rows |
|---|---|---|---|---|---|
| Recommended (R) | 41,994 | ~453 | ~55 | ~5 | **~41,481** |
| Literature-approximate | 44,271 | ~480 | ~60 | ~5 | **~43,726** |
| Maximum retention | 46,399 | ~533 | ~65 | ~5 | **~45,796** |

> [!NOTE]
> These are estimates derived from Phase 1 audit measurements. Actual deduplication counts will be measured in Phase 4 and reported in the harmonization log.

---

## 13. Label Coverage by Target Label (Under Recommended Strategy)

| Target Label | Positive Examples (estimated) | Source | Prevalence % (estimated) |
|---|---|---|---|
| `anger` | 4,478 (EmoNoBa) + 2,480 (UBMEC) + 4,499 (MONOVAB) | All 3 datasets | ~27.1% |
| `disgust` | 0 (EmoNoBa — assumed 0) + 2,079 (UBMEC) + 2,067 (MONOVAB) | UBMEC + MONOVAB | ~10.0% |
| `fear` | 401 (EmoNoBa) + 1,361 (UBMEC) + 56 (MONOVAB) | All 3 datasets | ~4.4% |
| `joy` | 10,112 (EmoNoBa) + 3,467 (UBMEC) + 1,601 (MONOVAB via enjoyment) | All 3 datasets | ~36.6% |
| `sadness` | 5,681 (EmoNoBa) + 2,683 (UBMEC) + 1,188 (MONOVAB) | All 3 datasets | ~23.3% |
| `surprise` | 1,086 (EmoNoBa) + 1,366 (UBMEC) + 100 (MONOVAB) | All 3 datasets | ~6.2% |

**Implication for class-weight balancing:** `fear` (4.4%) and `surprise` (6.2%) are significantly underrepresented. The `class_weight='balanced'` setting in Logistic Regression will partially compensate, but both classes may still be challenging for the Classifier Chain. `joy` (36.6%) and `anger` (27.1%) are the dominant classes.

---

## 14. Recommended Strategy (Summary)

Based on this analysis, the recommended harmonization configuration is:

| Decision | Recommendation | Status |
|---|---|---|
| EmoNoBa Disgust | **Strategy D** — assign `disgust=0` as documented operational assumption; run Experiment F5 as mandatory sensitivity analysis | [PROPOSED] → [REQUIRES USER DECISION] |
| EmoNoBa Love-only rows | **LOVE-1** — exclude 2,277 Love-only rows from corpus | [PROPOSED] → [REQUIRES USER DECISION] |
| MONOVAB Contempt-only rows | **CONTEMPT-1** — exclude 2,128 Contempt-only rows from corpus | [PROPOSED] → [REQUIRES USER DECISION] |
| MONOVAB enjoyment → joy | **Freeze the mapping** — enjoyment is the MONOVAB representation of Ekman Joy; semantically well-motivated | [PROPOSED] → [REQUIRES USER DECISION] |
| UBMEC single-label → six-binary | **One-hot conversion** — assign 1 to observed class, 0 to all others | **[FROZEN]** |
| Conflicting duplicate resolution | **Drop all rows for conflicting texts** (safest) — to be executed in Phase 4 | Phase 4 decision |
| Cross-dataset deduplication | **Exact text deduplication before splitting** — to be executed in Phase 4 | Phase 4 decision |

**Estimated final corpus under recommended strategy:** ~41,481 unique rows (after deduplication and exclusions).

---

## 15. Decisions Requiring Explicit User Approval

The following decisions **must be reviewed and confirmed before Phase 4 begins**:

| # | Issue | Options | Blocking? |
|---|---|---|---|
| 1 | **EmoNoBa Disgust strategy** | A / B / C / **D (recommended)** | ✅ Yes |
| 2 | **EmoNoBa Love-only rows (2277)** | LOVE-1 (exclude) / LOVE-2 (mask) / LOVE-3 (all-zero) | ✅ Yes |
| 3 | **MONOVAB Contempt-only rows (2128)** | **CONTEMPT-1 (exclude, recommended)** / CONTEMPT-2 (mask) / CONTEMPT-3 (all-zero — discouraged) | ✅ Yes |
| 4 | **MONOVAB enjoyment → joy mapping** | Freeze it / keep as proposed / reject | ✅ Yes |
| 5 | Conflicting duplicate resolution policy | Drop conflicting texts / label union / keep first | ⚠️ Phase 4 |
| 6 | Degenerate text removal (5 rows) | Drop and log (recommended) / retain | ⚠️ Phase 4 |

---

## Appendix A — Machine-Readable Artifact Index

| File | Contents |
|---|---|
| [label_mapping.csv](label_mapping.csv) | Complete native-to-target label mapping table with rationale |
| [label_coverage.csv](label_coverage.csv) | Per-target-label coverage and prevalence by source dataset |
| [harmonization_strategies.csv](harmonization_strategies.csv) | All candidate strategies with quantified impacts |
| [source_label_semantics.json](source_label_semantics.json) | Structured taxonomy semantics for all native labels |
| [harmonization_decisions.json](harmonization_decisions.json) | Machine-readable decision log with frozen/proposed/pending statuses |
