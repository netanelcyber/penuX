# Hierarchical LDAP-Style Directory & Email Authentication Protocol

**עברית:** מסמך זה מתאר עץ מידע היררכי בסגנון LDAP (DIT) ופרוטוקול הזדהות מבוסס-מייל
לקליטת משתמשים חדשים לדומיין `penux.uk`. משתמש חדש מוכיח בעלות על כתובת מייל
(קוד חד-פעמי) לפני שנוצרת לו רשומה תחת `ou=people`.

**English:** This document specifies a hierarchical LDAP-style Directory
Information Tree (DIT) and an email-based verification protocol for
onboarding new users to the `penux.uk` domain. A new user must prove control
of an email address (one-time code) before an active directory entry is
created for them.

Implementation: [`scripts/ldap_email_auth.py`](../scripts/ldap_email_auth.py).
Tests: [`tests/test_ldap_email_auth.py`](../tests/test_ldap_email_auth.py).

## 1. Directory Information Tree (DIT)

```
dc=penux,dc=uk
├── ou=pending          # email given, awaiting OTP confirmation
├── ou=people           # verified, active accounts
│   └── uid=<local-part>,ou=people,dc=penux,dc=uk
└── ou=groups
    └── cn=members,ou=groups,dc=penux,dc=uk
```

The hierarchy is intentionally shallow (two levels of `ou` under the domain
component) — it maps 1:1 onto the states of the enrollment protocol below,
so an entry's DN tells you its status without reading attributes.

### 1.1 Object classes / attributes

| Entry              | objectClass                          | Key attributes                                                              |
|--------------------|---------------------------------------|-------------------------------------------------------------------------------|
| `dc=penux,dc=uk`   | `dcObject`, `organization`            | `dc`, `o`                                                                      |
| `ou=pending`/`people`/`groups` | `organizationalUnit`       | `ou`                                                                           |
| person             | `inetOrgPerson`, `penuxAccount`       | `uid`, `cn`, `mail`, `penuxStatus`, `userPassword`, `memberOf`                 |
| group              | `groupOfNames`                        | `cn`, `member`                                                                 |

`penuxAccount` is a project-local auxiliary object class (placeholder OID
arc `1.3.6.1.4.1.99999.1` — replace with a registered enterprise number
before pointing a real OpenLDAP server at this schema). It adds:

- `penuxStatus`: `pending` | `active` | `revoked`
- `penuxEnrolledAt` / `penuxVerifiedAt`: timestamps

`userPassword` is stored as `{PBKDF2-SHA256}<salt-hex>$<digest-hex>`
(200,000 iterations), never in plaintext — mirroring the `{SSHA}`/`{CRYPT}`
prefix convention LDAP servers use for the `userPassword` attribute.

## 2. Enrollment protocol (state machine)

```
            request_enrollment(email)
ANONYMOUS ───────────────────────────► PENDING
                                          │  OTP mailed to `email`
                                          │  (sha256 hash + 15 min TTL stored,
                                          │   plaintext code never persisted)
                       verify(email, otp) │
                       ┌──────────────────┴───────────────────┐
                       │ correct, unexpired, attempts < 5      │ wrong / expired /
                       ▼                                       │ too many attempts
                    ACTIVE                                     ▼
             (uid=...,ou=people)                            REJECTED
             entry created, added to                  (entry deleted; caller
             cn=members group, one-time                must request again)
             bootstrap password issued
```

Rules enforced by `HierarchicalDirectory`:

- **Rate limiting**: at most 5 `request_enrollment` calls per email per
  rolling hour, to blunt mail-bombing / enumeration.
- **One-time codes**: 6-digit numeric OTP, compared with
  `secrets.compare_digest` against a SHA-256 hash (plaintext is only ever
  held in memory long enough to email it).
- **TTL**: a code expires 15 minutes after issuance; an expired pending
  entry is deleted outright (no silent renewal).
- **Attempt limit**: 5 wrong guesses deletes the pending entry, forcing a
  fresh `request_enrollment`.
- **uid collision**: the local-part of the email is used as `uid`; on
  collision a numeric suffix (`alice`, `alice2`, ...) is appended.
- **Bootstrap credential**: on success a random password is generated,
  hashed with PBKDF2-HMAC-SHA256, and returned to the caller exactly once
  (it is never stored in plaintext or logged) — the caller is expected to
  hand it to the user out-of-band and prompt a change on first login.

## 3. Authentication (bind)

Post-provisioning authentication is a simple LDAP-style bind: the caller
supplies `uid` + password; `HierarchicalDirectory.bind()` looks up the
`ou=people` entry, requires `penuxStatus == "active"`, and compares the
password against the stored PBKDF2 hash using constant-time comparison.
A production deployment should layer account lockout / backoff on top of
this, the same as it would for any LDAP simple bind.

## 4. Interoperability

`HierarchicalDirectory.export_ldif()` renders the current tree as RFC 2849
LDIF, so the JSON-backed store used here (a lightweight stand-in with no
external dependency) can be imported into a real OpenLDAP / 389 Directory
Server instance once the domain moves off the demo store — consistent with
the self-hosted-mail direction already established by
`.github/workflows/imap_dns.yml` (`mail.penux.uk` A/MX records).

## 5. CLI usage

```sh
python3 scripts/ldap_email_auth.py enroll alice@example.com
python3 scripts/ldap_email_auth.py verify alice@example.com 123456
python3 scripts/ldap_email_auth.py bind alice <temp_password>
python3 scripts/ldap_email_auth.py export-ldif
```

Mail delivery uses the `PENUX_SMTP_HOST` environment variable (default
`mail.penux.uk`), trying STARTTLS on 587 then plaintext on 25 — the same
ports `imap_dns.yml` opens on the mail server's firewall.

## 6. Out of scope / non-goals

This implementation is a protocol reference and demo, matching this
repo's "research/demo" posture (see root `README.md`): it is **not** a
hardened, internet-facing auth service. Before using it beyond a personal
domain, add: TLS-only SMTP submission with authentication, IP-based rate
limiting, CAPTCHA on `request_enrollment`, structured audit logging, and a
real LDAP/RDBMS backend instead of a single JSON file.
