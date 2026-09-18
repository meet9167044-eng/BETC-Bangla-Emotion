# DATASET_SPEC.md

This file is authoritative on all dataset matters. It **supersedes** the
dataset section of the original architecture `.docx`, which assumed
EmoNoBa alone. See tags defined in `PROJECT_SPEC.md` Section 4.

## 1. Dataset Strategy History (for context, do not re-litigate)

1. **Original decision:** use EmoNoBa alone. Recorded in the architecture
   `.docx`.
2. **Revised decision (current, FINAL):** combine three existing Bangla
   emotion datasets — **EmoNoBa**, **UBMEC**, and **MONOVAB** — harmonized
   toward a shared six-emotion label set, because a single dataset alone
   does not have sufficient labeled volume and because prior published
   work has already validated this exact combination.

**[FINALIZED]** The dataset strategy is: combined, harmonized
EmoNoBa + UBMEC + MONOVAB. Do not silently revert to EmoNoBa-only.

## 2. Source Datasets — What Is Known From Literature (all tagged [VERIFY])

Every number below is a **[VERIFY]** claim taken from published
descriptions of these datasets. None of these numbers may be used in a
report, table, or config file until independently confirmed against the
actual downloaded raw files (see Section 6, Blocking Verification Items).

### 2.1 EmoNoBa
- **[VERIFY]** Reported size: approximately 22,700–22,740 manually
  annotated Bangla public comments (different sources report 22,698 and
  22,739; the discrepancy itself must be resolved against the actual
  file).
- **[VERIFY]** Reported domain coverage: 12 domains (e.g. Personal,
  Politics, Health, and others).
- **[VERIFY]** Reported label set: **love, joy, surprise, anger,
  sadness, fear** — six labels, based on the Junto Emotion Wheel. Note:
  this set **does not include "disgust"** and **does include "love,"**
  which is a mismatch against Ekman's six basic emotions used by the
  other two source datasets.
- **[VERIFY]** Reported structure: multi-label (a comment may have zero,
  one, or several active emotion labels).
- **[VERIFY]** Reported test-set size: approximately 2,272 samples in the
  dataset's own published split (this project will not necessarily reuse
  that split — see `EVALUATION_PROTOCOL.md`).

### 2.2 UBMEC (Unified Bangla Multi-class Emotion Corpus)
- **[VERIFY]** Reported composition: a combination of three previously
  published resources — **BNEmo**, **BEmoC**, and a set of Bangla
  YouTube comments.
- **[VERIFY]** Reported size: totalling approximately 13,072 Bangla
  comments after combination.
- **[VERIFY]** Reported label set: **anger, disgust, fear, joy, sadness,
  surprise** — Ekman's six basic emotions.
- **[VERIFY]** Label structure (single-label vs multi-label) must be
  confirmed directly from the file; sources are not fully consistent on
  this point across the three original sub-sources.

### 2.3 MONOVAB
- **[VERIFY]** Reported size: approximately 10,244 entries.
- **[VERIFY]** Reported sources: diverse social media platforms and
  online news portals, collected via web scraping.
- **[VERIFY]** Reported label set: Ekman's six basic emotions, explicitly
  multi-label.

### 2.4 Prior Published Precedent for Combining These Three
- **[VERIFY] / [LIT]** A published study reports combining EmoNoBa,
  UBMEC, and MONOVAB into an initial pool of 46,035 entries, removing
  665 duplicate texts, and arriving at a refined set of 43,676 unique
  entries aligned with Ekman's six basic emotions. This is **literature
  precedent that the combination approach is valid**, not a guarantee
  that our own merge will produce these exact numbers. Our own pipeline
  must recompute all of these figures independently once the raw files
  are in hand.

## 3. Target Label Set

**[FINALIZED]** The project will use exactly these six output labels, in this
order throughout the repository:

```text
anger, disgust, fear, joy, sadness, surprise
```

The following source-specific labels are **not target labels** and must not
become additional output classes:

- EmoNoBa: `love`
- MONOVAB: `contempt`
- MONOVAB: any source-specific synonym that is not one of the six targets

**[FINALIZED HARMONIZATION RULE]** `love` is not mapped to `disgust`. It is
dropped as a target label. Because EmoNoBa does not natively annotate
`disgust`, the BETC six-label matrix will represent EmoNoBa's `disgust` as
0 **only under an explicit harmonization assumption**: an EmoNoBa row is
being treated as negative for the six target emotions when the source has no
positive annotation for that target. This is the operational rule required
to create one complete 0/1 target matrix for the classical BETC pipeline.

This assumption must be reported explicitly in the harmonization report and
final paper/report, and a sensitivity analysis must compare the main result
against an evaluation that excludes EmoNoBa-derived rows from
`disgust`-specific analysis. The coding agent must never describe the
EmoNoBa `disgust=0` values as original human annotations.

UBMEC is a six-class single-label corpus. Its single categorical emotion is
converted to a six-column binary vector: the observed class receives 1 and
the other five target classes receive 0. This conversion is a representation
change, not a claim that UBMEC was originally multi-label.

MONOVAB is multi-label; its target emotions are mapped directly to the six
output columns, while `contempt` is discarded as a non-target label.

The final harmonization must therefore produce exactly these columns:

```text
text | anger | disgust | fear | joy | sadness | surprise
```

No seventh `love` or `contempt` output is permitted in BETC.

## 4. Harmonization Requirements

Before the three datasets can be merged, the following must happen, in
order:

1. **Schema audit** — for each source dataset, document: file format,
   column names, label encoding (0/1 columns vs. single categorical
   column vs. list column), text column name, encoding (UTF-8
   verification), and any metadata columns (domain, source platform,
   date).
2. **Label mapping table** — an explicit, version-controlled mapping
   from each source dataset's native label names to the target label
   set decided in Section 3. This mapping must be a checked-in config
   file (e.g. `configs/label_mapping.yaml`), not inline code.
3. **Missing-label semantics** — every row in the merged dataset must
   distinguish between "labeled negative" (annotators considered this
   emotion and marked it absent) and "not annotated for this label"
   (the source dataset never asked about this emotion at all). These
   must **never** be collapsed into the same "0" without explicit
   justification recorded in the harmonization log — see
   `RESEARCH_RULES.md`.
4. **Text normalization pass for comparison** — before deduplication,
   apply a comparison-only normalization (whitespace collapse, Unicode
   NFC) to detect near-duplicate rows across datasets. This is separate
   from the modeling preprocessing pipeline defined in `PIPELINE_SPEC.md`.
5. **Deduplication** — remove exact and near-duplicate texts across all
   three combined sources (not just within each source individually,
   since the same viral comment could appear in more than one source
   dataset). Log the number of duplicates removed and the method used
   (exact string match, then a documented near-duplicate threshold if
   used).
6. **Post-harmonization audit** — recompute and report (not assume):
   total row count, per-label positive count, per-label prevalence
   percentage, multi-label cardinality distribution (how many labels
   are active per row on average), and per-source contribution to the
   final merged set.

## 5. Domain and Distribution Considerations

- EmoNoBa spans multiple domains (politics, health, personal, etc.);
  UBMEC and MONOVAB have different, only partially overlapping domain
  coverage (Facebook political/social comments, YouTube comments, news
  portals).
- **[PROPOSAL]** A `source_dataset` column and, where available, a
  `domain` column must be retained through harmonization so that
  cross-domain and cross-source behavior can be audited later (see
  `EXPERIMENT_PLAN.md`, cross-domain experiment).
- Domain shift across the three sources is a known risk: merging without
  tracking provenance would make it impossible to later diagnose whether
  a performance drop is due to genuine model weakness or domain
  mismatch.

## 6. BLOCKING DATASET VERIFICATION ITEMS

The remaining factual items below must be resolved by directly inspecting
the actual downloaded dataset files before Phase 2
(`IMPLEMENTATION_PLAN.md`) is considered complete:

1. Exact, current, working download source/URL/repository for each of
   EmoNoBa, UBMEC, and MONOVAB, and their license/redistribution terms
   for combined/derived use.
2. Exact file format, column names, and label encoding for each dataset
   as actually downloaded (not as described in papers, which may
   describe an earlier version).
3. Exact row counts for each raw dataset as downloaded (literature
   numbers in Section 2 are [VERIFY] only).
4. Whether EmoNoBa's label set truly excludes "disgust" and includes
   "love" in the version actually downloaded (confirm against the file
   header, not just the paper text).
5. Whether UBMEC's and MONOVAB's downloaded files are single-label or
   multi-label in practice (column structure), since sources differ in
   how they describe this.
6. The exact downloaded EmoNoBa schema and whether any hidden/source
   annotation provides a defensible `disgust` field. The project decision is
   already fixed: `love` is excluded and the operational six-label matrix
   uses `disgust=0` for EmoNoBa rows under the documented harmonization
   assumption; this must not be mistaken for an original annotation.
7. Actual duplicate count across the three combined sources on this
   project's own data (the literature-reported 665 is [VERIFY]/[LIT]
   only and must not be copied into this project's reports as if it
   were our own measured number).
8. Actual per-label prevalence in the final harmonized dataset (needed
   to configure `class_weight='balanced'` expectations and to interpret
   Macro-F1 vs Micro-F1 gaps later).
9. Confirmation of language purity — that no Banglish/code-mixed rows
   remain in any of the three sources after intended exclusion (per
   `PROJECT_SPEC.md` scope), since some of these datasets have documented
   code-mixed variants (e.g. an EmoNoBa/MONOVAB Banglish companion
   dataset exists in the literature and must not be accidentally
   included).

## 7. Repository Locations

```
data/raw/emonoba/       # exact, untouched original EmoNoBa download
data/raw/ubmec/         # exact, untouched original UBMEC download
data/raw/monovab/       # exact, untouched original MONOVAB download
data/interim/           # harmonized, deduplicated, label-mapped, NOT split
data/processed/         # final train/validation/test splits (see EVALUATION_PROTOCOL.md)
configs/label_mapping.yaml   # source-label -> target-label mapping, checked in
logs/dataset_audit/     # audit reports generated at each harmonization step
```

Raw data directories are **read-only inputs**. Nothing in `data/raw/` may
ever be edited, overwritten, or have rows silently deleted — see
`RESEARCH_RULES.md`.
