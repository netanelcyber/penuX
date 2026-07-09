# Journal submission drafts

These are **draft** manuscripts derived from the project's extended analysis
(`docs/sap_severity_english_article.md`), condensed and restructured to fit
the submission conventions of three realistic target journals:

- `PLOS_ONE_manuscript_DRAFT.docx` — PLOS ONE (judges technical soundness,
  not novelty/impact — the best fit for this work).
- `Scientific_Reports_manuscript_DRAFT.docx` — Scientific Reports (Nature
  Portfolio; broader scope and much higher acceptance rate than the
  flagship *Nature* journal, which is not a realistic target for this study).
- `BMC_MedInformDecisMak_manuscript_DRAFT.docx` — BMC Medical Informatics
  and Decision Making (structured abstract, clinical-ML focus).
- `JAMIA_manuscript_DRAFT.docx` — Journal of the American Medical Informatics
  Association (Oxford University Press). $0 author fee if the optional
  open-access route is declined at acceptance (standard subscription-access
  publication) — confirmed against academic.oup.com/jamia at the time of
  writing; re-check current policy before submitting, since publisher terms
  change.

PLOS ONE, Scientific Reports, and BMC MIDM all charge an article-processing
charge (APC) on acceptance (roughly $1,750–$2,490). JAMIA is $0 only if the
optional OA route is declined. The two genuinely $0-fee-under-any-option
venues, prepared as full drafts here, are:

- `JMLR_manuscript_DRAFT.docx` / `.pdf` / `jmlr_latex/` — Journal of Machine
  Learning Research. Diamond open access, no APC, not anonymized (author
  block as normal). Submit via https://jmlr.csail.mit.edu/manudb/
  (register → "submit manuscript"); pick 3–5 suggested Action Editors from
  JMLR's board. **`jmlr_latex/` contains a real, verified-compiling LaTeX
  submission** (`JMLR_manuscript_DRAFT.tex` + the official `jmlr2e.sty`
  from JmlrOrg/jmlr-style-file, using the `preprint` package option since
  no volume/issue/paper-id has been assigned yet) — compiled locally with
  `pdflatex` with zero errors (only cosmetic overfull-hbox warnings from a
  couple of long unbreakable tokens); `JMLR_manuscript_DRAFT.pdf` at this
  level is that compiled output, not a re-rendered `.docx`.
- `TMLR_manuscript_DRAFT.docx` / `.pdf` / `tmlr_latex/` — Transactions on
  Machine Learning Research. Diamond open access, no APC, submitted and
  reviewed via OpenReview.net. **`tmlr_latex/` contains a real,
  verified-compiling LaTeX submission** using the official `tmlr.sty` /
  `tmlr.bst` (from JmlrOrg/tmlr-style-file). TMLR's package itself enforces
  double-blind anonymity: in default mode it hardcodes "Anonymous authors /
  Paper under double-blind review" in the compiled output regardless of the
  `\author{}` block's content, which is why the real author info is left in
  the `.tex` source (matching TMLR's own example template) — verified by
  rendering the compiled PDF and confirming no name/email/affiliation
  appears anywhere in the visible text. PDF metadata (Author/Title fields)
  is also explicitly blanked via `\hypersetup{pdfauthor={},pdftitle={}}`
  and was checked with `doc.metadata` to confirm no leak there either. The
  GitHub repository URL is withheld from the Reproducibility Statement for
  the same reason (it would reveal the corresponding author's GitHub
  handle). Switch to `\usepackage[accepted]{tmlr}` (with real repo links
  and an Acknowledgments/Funding/Author Contributions section — TMLR
  convention adds these only post-acceptance) only after a decision.

Both are framed slightly more toward an ML-methodology audience (emphasis
on benchmarking scale and rigor) than the clinical-informatics framing used
in the PLOS ONE/Scientific Reports/BMC MIDM/JAMIA drafts, though the
underlying results are identical.

## Before actually submitting

These are starting points, not submission-ready files. Before uploading to
any journal portal:

1. **Verify the Ethics Statement.** It describes the situation honestly
   (secondary analysis of third-party public datasets; original
   data-collection ethics oversight not independently re-verified by this
   project) but does not claim IRB approval that doesn't exist. Some
   editors may request additional documentation from the original dataset
   authors (Longshike et al.) before proceeding.
2. **Fill in real author metadata** — ORCID, final affiliation wording.
3. **Reformat to each journal's live author-guidelines page** (reference
   style, figure/table placement, word/reference limits change over time).
4. **Trim/reorder the 40-reference list** if the target journal has a hard
   reference cap, and double-check citation order once final.
5. Journal submission itself (account creation, portal upload, any
   publication fees) is a step only the corresponding author can take.
