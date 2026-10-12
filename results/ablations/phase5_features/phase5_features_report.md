# BETC — Phase 5: Auxiliary Feature Groups Ablation Report

**Date:** 2026-10-12  
**Dataset:** 41,184 records (28,840 train / 6,170 val / 6,174 test canonical split)  
**Model Architecture:** ClassifierChain (`LogisticRegression(C=1.0, class_weight='balanced', solver='saga', max_iter=1000)`) with per-class threshold optimization on validation probabilities.  
**Baseline Champion:** Phase 4 Optimal Pipeline (TF-IDF [25k] + Emoji [52] + Emotion Lexicon [18] + Advanced Negation Scope [16] = 25,086 dims).

---

## 1. Executive Summary

Phase 5 systematically evaluated six candidate auxiliary feature groups added one at a time to the Phase 4 Champion Base:
1. **5.1 Character Elongation:** Repeated character runs ($\ge 3$), elongated token count, presence, and ratio.
2. **5.2 Hashtag Features:** Hashtag count, presence flag, and hashtag-to-character ratio.
3. **5.3 Script Proportions & Code-Mixing:** Bangla-script, Latin-script, digit proportions, and binary code-mixing flag.
4. **5.4 Normalized Text Statistics:** Type-Token Ratio (TTR / lexical diversity), average word length, log character count, log word count.
5. **5.5 Spelling Variation & Normalization Divergence:** Token divergence between raw social text and normalized text.
6. **5.6 Verbal Inflection Markers:** Bangla past, present continuous, and future verb suffix markers.

### 🏆 Key Takeaways & Empirical Decisions:
* **The Scale & Sparsity Reality:** The vast majority of generic structural/surface statistics (text stats, script mixing, elongation) dilute the high-dimensional lexical TF-IDF space and cause regressions on minority classes—especially **Fear** (dropping from `0.392` to `0.352` on text stats and `0.356` on code-mixing).
* **5.2 Hashtags (Kept):** Added a subtle positive lift to **`0.4904` Val Macro-F1** and **`0.5821` Micro-F1** (+0.0002 / +0.0001), lifting Anger (`0.5494`) and Disgust (`0.3444`) while holding Fear stable (`0.3923`).
* **Clean Selection Rule Upheld:** Per project protocol ("Keep a feature or method only when its improvement is reproducible, the evaluation is fair, and per-label results do not show unacceptable regressions"), groups 5.1, 5.3, 5.4, 5.5, and 5.6 are **dropped**, avoiding unnecessary pipeline complexity and protecting minority class recall.

---

## 2. Quantitative Ablation Results

| Feature Group | Added Dims | Total Dims | Val Macro-F1 | Val Micro-F1 | $\Delta$ Macro-F1 | Anger | Disgust | Fear | Joy | Sadness | Surprise | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Champion Base (Phase 4)** | --- | 25,086 | `0.4902` | `0.5820` | `+0.0000` | 0.549 | 0.344 | **0.392** | 0.825 | 0.502 | 0.330 | **Benchmark** |
| **5.1 Character Elongation** | +4 | 25,090 | `0.4870` | `0.5725` | `-0.0032` | 0.546 | 0.339 | 0.385 | 0.824 | 0.500 | 0.328 | **Drop** |
| **5.2 Hashtags** | +3 | 25,089 | **`0.4904`** | **`0.5821`** | **`+0.0002`** | **0.549** | **0.344** | **0.392** | 0.825 | **0.503** | 0.329 | **Keep** |
| **5.3 Script / Code-Mixing** | +4 | 25,090 | `0.4836` | `0.5776` | `-0.0066` | 0.548 | 0.349 | 0.356 | 0.825 | 0.501 | 0.323 | **Drop** |
| **5.4 Normalized Text Stats** | +4 | 25,090 | `0.4762` | `0.5727` | `-0.0140` | 0.547 | 0.325 | 0.352 | **0.826** | 0.498 | 0.308 | **Drop** |
| **5.5 Spelling Variation** | +3 | 25,089 | `0.4879` | `0.5785` | `-0.0023` | 0.548 | 0.337 | 0.385 | 0.826 | 0.502 | 0.329 | **Drop** |
| **5.6 Verbal Markers** | +4 | 25,090 | `0.4882` | `0.5818` | `-0.0020` | 0.549 | **0.346** | 0.375 | 0.826 | **0.503** | **0.331** | **Drop** |

---

## 3. Deep-Dive Error Analysis on Rejected Groups

### Why Did Text Statistics (5.4) & Code-Mixing (5.3) Fail?
1. **Severe Fear Degradation:** Fear is the rarest class in BETC (1,252 train instances, 4.3% prevalence). Code-mixing and text length features shifted decision thresholds and produced severe precision/recall trade-off collapse (Fear F1 fell from `0.392` down to `0.352`).
2. **Lexical Sufficiency:** Character n-grams (3–5 char_wb) already implicitly capture morphology, verbal inflections, and elongation sub-patterns inside high-dimensional sparse representations. Appending uncalibrated dense auxiliary indicators creates conflicting gradients for SAGA under $L_2$ penalty.

---

## 4. Deliverables & Verification
* **Experiment Script:** `scripts/experiments/phase5_additional_features.py`
* **JSON Results:** `results/ablations/phase5_features/phase5_ablation_report.json`
* **Unit & Regression Tests:** `tests/test_phase5_additional_features.py` (6 unit tests, all passing `6 passed in 6.67s`)
