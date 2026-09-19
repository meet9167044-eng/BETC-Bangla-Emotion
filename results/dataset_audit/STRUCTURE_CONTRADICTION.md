# Raw Dataset Audit — Blocked Before Content Inspection

**Status:** BLOCKED

## Reason for stopping

The requested audit is governed by `DOCS/DATASET_SPEC.md`, which defines the
raw input locations as:

```text
data/raw/emonoba/
data/raw/ubmec/
data/raw/monovab/
```

The actual repository layout found on 2026-09-19 is:

```text
data/raw/Emonoba/Test.csv                 (450,627 bytes)
data/raw/Emonoba/Train.csv                (3,649,059 bytes)
data/raw/Emonoba/Val.csv                  (418,988 bytes)
data/raw/UBMEC Corpus_Sakib(updated).xlsx (1,116,902 bytes)
data/raw/MONOVAB (1).csv                  (1,938,912 bytes)
```

Only EmoNoBa is inside a source-named directory (with a case difference).
UBMEC and MONOVAB are files directly under `data/raw/`, not under their
documented source directories. This contradicts the documented raw-data
structure.

The user explicitly instructed: *"If the actual dataset structure
contradicts the documentation, STOP and report the contradiction instead of
silently changing the methodology."* Therefore, no dataset file was opened
for schema, row, label, text, encoding, missing-value, or duplicate analysis.
No dataset data was altered.

## Decision required

Please confirm one of the following before the raw content audit continues:

1. The documented layout is authoritative, and the raw files should be placed
   in the three documented source directories by the project owner; or
2. The current layout is authoritative, and the project documentation should
   be explicitly revised to match it.

Until that decision is recorded, the requested dataset-wise statistics are
unresolved rather than assumed.
