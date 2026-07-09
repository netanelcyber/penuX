# Dataset landscape for early prediction of first psychotic episode from routine labs

**Status: no usable public dataset identified yet that matches the exact target design
(routine labs, ~24-month lead time before a first psychotic episode, freely downloadable).**
This document records what was actually found, so the search doesn't need to be repeated, and
so no one mistakes "we haven't found one" for "one doesn't exist anywhere" — it may exist behind
a restricted-access application process.

## What the target design actually requires

1. A cohort of individuals who later had a documented first psychotic episode (ICD/DSM-coded),
   plus a comparable control group.
2. Routine laboratory results (the kind from an ordinary blood panel — CBC, metabolic panel,
   lipids, inflammatory markers, liver/renal function — not specialized psychiatric biomarkers)
   collected **before** the psychotic episode, ideally with a documented time gap of up to ~24
   months.
3. Public or reasonably obtainable access (open license, or an application process realistic to
   complete).

## What was found (as of this search)

| Source | What it is | Lead time | Access |
|---|---|---|---|
| Frontiers in Medicine (2026), "A clinical prediction model for schizophrenia based on machine learning algorithms" | 180 first-episode schizophrenia patients + 214 controls; routine peripheral blood biochemical indicators (Arg, TP, ALP, HDL, UA, LDL identified as top predictors); Random Forest AUC 0.877 | At/near first-episode presentation, **not** a pre-onset lead time | Data availability not confirmed as open; would need to check the paper's Data Availability Statement directly or contact the authors |
| DETECT model, *Lancet Digital Health* | EHR-based model detecting psychosis risk in a large (60M+ record) claims/EHR dataset, reportedly with lead time approaching ~1 year before documented onset | Genuine pre-onset lead time (closest real precedent to the requested design) | Proprietary/institutional EHR or claims data — not an open download |
| NAPLS (North American Prodrome Longitudinal Study) | Multi-site prospective clinical-high-risk cohort with rich longitudinal clinical/biological data | Prospective, includes pre-conversion data | Restricted access; requires a data-access application through the consortium |
| PRONIA (EU, Personalised Prognostic Tools for Early Psychosis Management) | Multi-modal (imaging, clinical, some blood biomarkers) European cohort for psychosis-risk prognosis | Prospective clinical-high-risk design | Restricted access, EU consortium data-sharing agreement |
| EU-GEI | Gene-environment interaction study in first-episode psychosis across European sites | Case-control, not designed around a lab-based pre-onset lead time | Restricted access |
| UK Biobank | ~500,000-participant cohort with baseline blood biomarkers (CRP, lipids, glucose, liver/renal panels, etc.) and linked longitudinal health records (including psychiatric diagnoses via linked hospital/primary-care data) | Genuinely supports a pre-diagnosis design (bloods collected at recruitment, psychiatric diagnosis potentially years later) — the most realistic *route* to the requested design | Requires a formal application, an approved research proposal, and typically institutional affiliation; not a same-day download; several weeks to months in practice |
| Kaggle / GitHub open datasets | Searched for a ready-made CSV analogous to the `PenuX-AP-Severity` "Longshike" pancreatitis datasets | N/A | **None found** matching this design; psychiatric prodrome data is essentially never released this way, for stigma/re-identification reasons |

## Assessment

The honest conclusion: **there is currently no dataset available to this project that would
support building even one legitimate model for this exact question, let alone 12,000.** The
closest realistic path is a formal **UK Biobank** application (the only option in the table
above with both (a) genuine pre-onset routine lab data and (b) a defined, completable
access process), but that is a weeks-to-months process requiring a registered research
project and is not something that can be completed inside this coding session.

## Options going forward

1. **Apply for UK Biobank access** (outside this session — requires the corresponding
   researcher's institutional details and a registered project). Once/if access is granted,
   this project's model-zoo methodology (reused from `PenuX-AP-Severity`) could be pointed at
   the real extracted cohort.
2. **Contact the Frontiers (2026) or DETECT study authors** to ask about data sharing.
3. **Reframe the question**: build the model-zoo benchmarking methodology now against a
   simulated dataset that is explicitly and permanently labeled as synthetic (e.g., for
   validating the pipeline/code itself), never presented as, or confused with, a real clinical
   finding. This has engineering value (the code/pipeline can be built and tested) but zero
   clinical or scientific evidential value, and must be labeled as such everywhere it appears.
4. **Do nothing further until real data is available** — the most conservative, and arguably
   most responsible, option given the stigma and misuse risk specific to psychiatric prediction.

No option was chosen automatically; this is a decision for the corresponding researcher.
