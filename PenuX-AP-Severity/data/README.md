# Data Directory

## Directory Structure

```
data/
├── public_sanitized/   ← Add de-identified legally usable datasets here
└── mimic/
    └── sql/            ← SQL extraction scripts for MIMIC-IV (no patient data)
```

## Governance Rules

1. **No raw hospital data** may be committed to this repository.
2. **No PHI** (names, IDs, addresses, dates of birth, etc.) may be stored here.
3. **MIMIC-IV** data must not be stored here — use SQL scripts to query a local MIMIC instance.
4. **Public datasets** must be legally redistributable, fully de-identified, and documented in `docs/dataset_sources.md`.
5. All datasets in `public_sanitized/` must have been processed through the sanitization pipeline.

## Adding a Dataset

```bash
python scripts/sanitize_datasets.py --input data/raw --output data/public_sanitized
python scripts/summarize_datasets.py --data data/public_sanitized/<file> --target-column severe
```
