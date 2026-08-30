# UK Biobank access — action plan for a future real-data attempt

This documents the actual, current (as of July 2026) UK Biobank access process,
for if/when the corresponding researcher decides to pursue real data for this
project. **This is not something that could be completed inside this coding
session** — it requires a real identity, a real institutional or personal
email used long-term, a CV, and a review process measured in weeks.

## Critical blocker right now

**UK Biobank is not accepting new applications.** Per their own site: applications
are paused while the UK Biobank Research Analysis Platform undergoes changes,
with new applications expected to reopen **late 2026**. There is nothing to
submit today regardless of eligibility.

## Why UK Biobank is the right target for this project specifically

Unlike every other option investigated (see `docs/dataset_landscape.md`), UK
Biobank is the one real-world resource that could genuinely support the
originally requested design:
- **Baseline blood biochemistry** was collected from ~500,000 participants at
  recruitment — the "routine labs" part of the design.
- **Linked longitudinal health records** (hospital episode statistics, primary
  care data, and death registry) mean psychiatric diagnoses recorded **after**
  the baseline blood draw are identifiable — the "N months before diagnosis"
  part of the design, for participants who were undiagnosed at recruitment and
  later received an ICD-coded psychosis/schizophrenia diagnosis.

## Eligibility (confirmed from UK Biobank's own pages)

- Access is described as available to **"all bona fide researchers... regardless
  of their location"** — institutional affiliation is not an absolute
  requirement, but the application is judged on merit, and:
- Each registering researcher must provide:
  - An up-to-date CV/resume in English (or a link to a professional profile)
  - **PubMed reference numbers of up to 5 peer-reviewed publications** — this is
    the item most likely to be a real obstacle for an independent researcher
    without a prior publication record. Worth having a plan for (e.g., naming
    the PenuX project's existing GitHub-hosted analyses is not a substitute
    for peer-reviewed publication; consider whether the PLOS ONE/JAMIA/JMLR/
    TMLR submissions prepared elsewhere in this project could eventually serve
    as that publication record).
  - A long-term contact email (personal email explicitly recommended if
    affiliation may change)
  - Certificate of completion for the **MRC Confidentiality and Data Protection
    in Health Research** training (must be within the last 12 months at time
    of application)
  - Disclosure of any complaints raised against the researcher in the last 3 years

## Steps (once applications reopen)

1. Register in UK Biobank's **Access Management System (AMS)** with the
   long-term email.
2. Complete the MRC Confidentiality and Data Protection quiz, save the certificate.
3. Submit the application: research summary, confirmation the research is
   health-related and in the public interest, and the requested data tier.
4. Add any collaborators; nominate an authorized signatory for the Material
   Transfer Agreement (MTA) — for an independent researcher with no
   institution, this signatory role needs particular attention (who signs on
   behalf of a one-person research project is worth clarifying directly with
   UK Biobank support before applying).
5. UK Biobank reviews (registration review is typically ~10 working days;
   full application review takes longer and was not precisely confirmed here).
6. On approval: pay the access fee (standard fee amount not confirmed in this
   research; a reduced £500+VAT tier exists for students and researchers at
   institutions in eligible lower-income countries — likely not applicable
   here) and sign the MTA.
7. Data access granted for a 3-year window.

## Realistic timeline

Even after applications reopen (late 2026 per UK Biobank's own stated target),
expect registration + application review + fee/MTA processing to take
**weeks to a few months** before actual data access — this is not a
same-session or same-week solution under any circumstances.

## Recommendation

Given the pause and the publication-record eligibility item, the most useful
immediate step is not to wait idly, but to treat the manuscript-submission
work already done for this project's sibling (`PenuX-AP-Severity`'s PLOS ONE/
JAMIA/JMLR/TMLR drafts) as potentially relevant: an accepted or even
submitted peer-reviewed publication strengthens a future UK Biobank
application's credibility, even though it isn't one of the 5 PubMed-indexed
publications UK Biobank explicitly asks for at registration time.
