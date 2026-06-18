# Third-Party Notices

## Data

**No MIMIC-IV or PhysioNet patient-level data is bundled with this repository.**

MIMIC-IV is a restricted-access resource. To use it:
- Complete credentialing at https://physionet.org
- Complete required human-subjects / data-protection training (e.g. CITI)
- Sign and comply with the MIMIC-IV Data Use Agreement
- Do not redistribute or commit patient-level MIMIC data

## Public Datasets

If you add any public dataset to `data/public_sanitized/`, you are responsible for:
- Verifying that the dataset license permits your intended use
- Verifying that the dataset is fully de-identified
- Keeping a record of the dataset's provenance and license in `docs/dataset_sources.md`
- Not committing any dataset that contains protected health information (PHI)
- Not committing any dataset under a license that prohibits redistribution

## Software Dependencies

This project uses open-source libraries (pandas, scikit-learn, FastAPI, etc.)
under their respective licenses. See `requirements.txt` for the full dependency list.
Each dependency retains its own license terms.

## Compliance

Users are solely responsible for:
- Compliance with applicable data-protection laws (GDPR, HIPAA, etc.)
- Compliance with institutional data-use agreements
- Obtaining local Helsinki / IRB approval before using hospital data
- Not deploying this software in any clinical setting without proper validation
