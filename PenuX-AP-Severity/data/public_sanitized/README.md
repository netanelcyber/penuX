# Public Sanitized Datasets

Place only **legally usable, fully de-identified** datasets here.

## Rules

- Never place raw hospital data here
- Never commit protected health information (PHI)
- Never place MIMIC-IV or other restricted-access data here
- Only datasets that are legally redistributable or privately used for research
- Document provenance and license in `docs/dataset_sources.md`
- Run sanitization first: `python scripts/sanitize_datasets.py --input data/raw --output data/public_sanitized`

## Current Contents

No dataset is bundled with this repository.
Add a legally usable, de-identified dataset here before running training.
