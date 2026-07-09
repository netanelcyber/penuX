# PenuX-Psychosis-EarlyDetection

**Exploring early prediction of a first psychotic episode from routine laboratory tests, prior to clinical presentation**

> ⚠️ **RESEARCH USE ONLY — NOT A DIAGNOSTIC OR SCREENING TOOL.**
> This project must never be used to label, flag, or make any determination about a real
> person's mental health status. Predicting risk of a first psychotic episode carries serious
> potential for harm if misused: false positives can cause unwarranted alarm, stigma, and
> discriminatory treatment (insurance, employment, social); false negatives can create false
> reassurance. Nothing produced here is validated for clinical, insurance, educational, or any
> other real-world decision about an identifiable person. Local Helsinki/IRB approval, and very
> likely a formal ethics review specific to psychiatric research, would be required before any
> use beyond this exploratory, public-data research setting.

---

## Objective (as requested)

Explore whether routine laboratory tests (the kind collected in ordinary primary-care or
general hospital blood panels — not specialized psychiatric biomarkers, imaging, or genetic
testing) carry a detectable signal for a first psychotic episode **up to 24 months before**
clinical presentation, using a large comparative model-zoo methodology (in the spirit of the
sibling `PenuX-AP-Severity` project, ultimately scaling to a comparably large — here, ~12,000
configuration — model space).

## Status: dataset search, not yet started modeling

**No dataset has been added to this project yet, and none is bundled.** Before any model is
built, this project needs a real, ethically-sourced, legally usable dataset that actually
supports the stated question (routine labs, with a genuine ~24-month lead time before a
documented first psychotic episode). See `docs/dataset_landscape.md` for what was found so far
and why this is a substantially harder data-access problem than the public CSV datasets used in
`PenuX-AP-Severity`.

**This project will not proceed to model-building on synthetic, simulated, or placeholder data
presented as if it were real.** If a real dataset cannot be secured, the honest fallback (as
was done for the quasi-SOFA feature in `PenuX-AP-Severity`) is to describe and reason about
*methodology* only, not to fabricate results.

---

## Why this is harder than PenuX-AP-Severity's dataset situation

- Psychiatric prodrome/first-episode research datasets are overwhelmingly **access-controlled**
  (institutional data use agreements, ethics-committee-approved researcher accounts), unlike the
  MIT/Apache-licensed CSV files that existed for acute pancreatitis severity.
- A genuine "routine labs, N months before a first psychotic episode" design requires
  **prospective or registry-linked cohorts** (e.g., birth-cohort or population-registry studies
  with banked/linked lab results predating diagnosis) — these are precisely the kind of data
  that carries the highest re-identification and stigma risk, so open release is rare.
- The strongest published precedents found so far (see `docs/dataset_landscape.md`) either (a)
  use blood tests collected **at or near first-episode presentation**, not 24 months prior, or
  (b) use large-scale EHR/claims data with genuine pre-onset lead time, but that data is
  proprietary/access-controlled (e.g., insurance claims databases), not an open download.

## Repository Structure (scaffold; mirrors PenuX-AP-Severity)

```
PenuX-Psychosis-EarlyDetection/
├── src/penux_psychosis/   # Core Python package (empty scaffold)
├── data/public_sanitized/ # Add a legally usable, de-identified dataset here (none bundled)
├── scripts/                # CLI scripts (to be added once real data is identified)
├── docs/
│   └── dataset_landscape.md  # Honest account of the dataset search
├── notebooks/
└── tests/
```

## Ethical & Legal Notes

- No patient data of any kind — real or synthetic-presented-as-real — is included in this
  repository.
- Any future real dataset used here would require: documented ethics/IRB approval covering
  psychiatric risk prediction specifically, a signed data use agreement with the data
  custodian, and explicit consideration of stigma and discrimination risk in how results are
  reported.
- This software, if it is ever built, will not be validated for clinical use, and predicting a
  future psychiatric diagnosis is a fundamentally higher-stakes claim than predicting severity
  of an already-diagnosed physical illness (as in `PenuX-AP-Severity`) — extra caution in
  framing, reporting, and any public communication is required.

## Citation

```
Stern, N. (2026). PenuX-Psychosis-EarlyDetection [Software, in progress].
https://github.com/netanelcyber/penux
```
