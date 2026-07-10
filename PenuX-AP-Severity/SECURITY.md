# Security Policy

PenuX-AP-Severity is a **research prototype**, not a certified medical device
or a HIPAA/ISO 27799-certified system. See `docs/hipaa_iso27799_gap_analysis_he.md`
for a full, honest gap analysis of the current technical security posture
against the HIPAA Security Rule (45 CFR §164.312) and ISO/IEC 27799.

## Supported Versions

This is a single-branch research repository. Only the `main` branch (and
active feature branches under review) receive security fixes. There is no
long-term-support version.

## Reporting a Vulnerability

If you find a security issue (e.g. an authentication bypass, an injection
vector, a data-leakage path), please open a GitHub issue marked
`security` or contact the maintainer directly rather than filing a public
issue with exploit details. Please include:
- A description of the issue and its potential impact
- Steps to reproduce
- Affected file(s)/endpoint(s)

## Before Using This Software With Real Patient Data

Do not point `api/` at a real hospital EHR/HIS feed (HL7, FHIR, or
Camelion) without first:
1. Setting `PENUX_AP_API_KEY` (see `api/security.py`) and deploying behind
   TLS (this app does not terminate TLS itself — use a reverse proxy).
2. Reviewing and completing every item in
   `docs/hipaa_iso27799_gap_analysis_he.md`.
3. Obtaining local Helsinki/IRB approval and, where applicable, a signed
   Data Use Agreement (see `docs/helsinki_irb_notes.md`).
4. Routing audit logs (see `AuditLogMiddleware` in `api/security.py`) to a
   tamper-evident, access-controlled log store — not repo-local stdout.

This software provides no medical advice and is not validated for clinical
use, regardless of the security controls in place.
