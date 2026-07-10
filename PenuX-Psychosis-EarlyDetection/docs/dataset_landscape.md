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
| *Translational Psychiatry* (2015), "Development of a blood-based molecular biomarker test for identification of schizophrenia before disease onset" | 957 serum samples across discovery/validation/pre-onset cohorts; 26-analyte multiplex immunoassay panel (A2M, ApoA1, ApoH, eotaxin, factor VII, FSH, haptoglobin, IgA, IGFBP2, IL-10, IL-1ra, IL-13, IL-8, leptin, MIF, SGOT, SCF, testosterone, TSH, VCAM-1, vWF, etc.) | **Genuine pre-onset lead time up to 2 years** — the closest real match found for the lead-time requirement | No repository/supplementary-data link found after a dedicated search; likely never deposited (2015 publication, predates most journals' mandatory data-deposit policies). Also: most of the 26 analytes are specialized multiplex-immunoassay cytokines/growth factors, not tests available in an ordinary "routine" primary-care blood panel, so it only partially matches the "routine tests" requirement even if data were available. |

### A note on network access, not just dataset existence

A second, follow-up search round confirmed this is partly an **access** problem, not only
a "no such dataset" problem:
- **Kaggle and all NCBI domains (including GEO) are hard-blocked by this environment's
  network policy** — direct connection attempts return `403`/"policy denial" at the proxy
  level, confirmed via `curl` and via this session's own proxy status endpoint. This is
  categorical: even a perfect public dataset hosted there could not be downloaded directly
  from this environment.
- **Publisher sites** (Frontiers, Nature, PMC) consistently return `403` to automated fetch
  attempts (Cloudflare-style bot protection), independent of the proxy issue above. This
  means full-text Data Availability Statements could not be directly verified for any of
  the candidate papers above — only search-engine snippets were available.
- **GitHub is reachable** and was searched specifically for a routine-blood-test schizophrenia/
  psychosis dataset; none was found (existing public GitHub repos on this topic are EEG,
  connectome/MRI, or genomic-microarray based, not routine blood chemistry).

## Assessment

The honest conclusion: **there is currently no dataset available to this project that would
support building even one legitimate model for this exact question, let alone 12,000.** The
closest realistic path is a formal **UK Biobank** application (the only option in the table
above with both (a) genuine pre-onset routine lab data and (b) a defined, completable
access process), but that is a weeks-to-months process requiring a registered research
project and is not something that can be completed inside this coding session.

## Options going forward

1. **Apply for UK Biobank access** — see `docs/uk_biobank_access_plan.md` for the concrete,
   confirmed process and requirements. **Important: UK Biobank is not currently accepting new
   applications** (paused for a platform migration; they state new applications are expected
   to reopen "late 2026"), so this is not actionable today regardless of eligibility. Once/if
   access is granted, this project's model-zoo methodology (reused from `PenuX-AP-Severity`)
   could be pointed at the real extracted cohort.
2. **Contact the Frontiers (2026), Frontiers (2025 "routine blood tests"), 2015 Translational
   Psychiatry, or DETECT study authors** to ask about data sharing.
3. **Reframe the question**: build the model-zoo benchmarking methodology now against a
   simulated dataset that is explicitly and permanently labeled as synthetic (e.g., for
   validating the pipeline/code itself), never presented as, or confused with, a real clinical
   finding. This has engineering value (the code/pipeline can be built and tested) but zero
   clinical or scientific evidential value, and must be labeled as such everywhere it appears.
4. **Do nothing further until real data is available** — the most conservative, and arguably
   most responsible, option given the stigma and misuse risk specific to psychiatric prediction.

No option was chosen automatically; this is a decision for the corresponding researcher.
