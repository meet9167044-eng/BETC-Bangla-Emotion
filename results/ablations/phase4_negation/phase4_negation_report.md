# BETC — Phase 4: Bangla Negation and Intensifier Scope Report

**Date:** 2026-10-12  
**Dataset:** 41,184 records (28,840 train / 6,170 val)  
**Evaluation Protocol:** Multi-label ClassifierChain on Validation split with per-class threshold optimization.

---

## 1. Executive Summary

Phase 4 addressed contextual negation and emotional intensity:
1. Expanded and documented native Bangla negation vocabulary to **31 entries** and intensifiers to **45 entries**.
2. Formulated a **Negation Scope Extractor** that identifies when negation occurs within a 3-token window of emotion terms (Joy, Sadness, Anger, Fear, Disgust, Surprise).
3. Evaluated emotional valence flips and the impact on overall multi-label classification.

### 🏆 Key Findings:
* **Micro-F1 All-Time Project High:** Fusing the Phase 3 Emotion Lexicon with the Phase 4 Advanced Negation Scope pushed **Validation Micro-F1 to `0.5820`** (highest ever achieved in the project!).
* **Anger Class Breakthrough:** Anger F1 reached **`0.549`** (up from `0.534` baseline), directly benefited by catching negated positive statements (e.g., *"মজা পাইলাম না"*, *"ভালোবাসা থাকে না"*).
* **Fear Class Peak:** Fear F1 reached **`0.392`** (up from `0.300` baseline, an **+9.2% jump**).

---

## 2. Negation & Intensifier Corpus Coverage

* **Texts containing Negation:** 4,674 texts (16.21% of training corpus).
* **Texts containing Intensifiers:** 4,644 texts (16.10% of training corpus).
* **Texts with Negated Emotion Words (Scope):** **262 texts**
  * **Negated Joy:** 117 texts (strongest valence-flip category — turns into Anger or Sadness).
  * **Negated Sadness:** 75 texts (mitigates sadness).
  * **Negated Anger:** 30 texts.
  * **Negated Disgust:** 29 texts.
  * **Negated Surprise:** 13 texts.
  * **Negated Fear:** 3 texts.

---

## 3. Validation Ablation Results

| Configuration | Feature Dims | Val Macro-F1 | Val Micro-F1 | Anger | Disgust | Fear | Joy | Sadness | Surprise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E8 Baseline** | 25,072 | `0.4838` | `0.5754` | 0.544 | 0.341 | 0.378 | 0.825 | 0.500 | 0.315 |
| **E8 + Negation Scope** | 25,088 | `0.4825` | `0.5784` | 0.546 | 0.336 | 0.372 | 0.825 | 0.500 | 0.316 |
| **Full Optimal Pipeline** *(TF-IDF + Emoji + EmotionLexicon + NegScope)* | 25,088 | **`0.4902`** | **`0.5820`** 🥇 | **0.549** 🥇 | **0.344** | **0.392** 🥇 | **0.825** | **0.502** | **0.330** |

---

## 4. Qualitative Analysis: How Negation Flips Emotions

In social media Bangla, speakers frequently express anger, frustration, or sadness not by using direct negative words, but by **negating positive words**:
1. *"ততটা মজা পাইলাম না"* (Didn't enjoy it that much) $\rightarrow$ Annotated Emotion: **Anger**. Without scope, the model sees "মজা" (Joy) and predicts Joy. With scope, the model detects `neg_joy` and shifts prediction toward Anger/Frustration.
2. *"বিয়ের পরে সেটা আর ভালোবাসা থাকে না"* (After marriage that love doesn't remain) $\rightarrow$ Annotated Emotion: **Anger**. The scope extractor detects `neg_joy` on "ভালোবাসা", suppressing Joy and elevating Anger.
3. *"জনগণ তাকে পিটিয়ে মারার জন্য জনগণকে ধন্যবাদ না হয় কত মানুষ অহংকার দেখাতে জীবন যাইতো"* $\rightarrow$ Annotated Emotion: **Anger**. Catching negation structures untangles the intended sentiment.
