"""Hierarchical LDAP-style directory + email-verified enrollment protocol.

Models a minimal Directory Information Tree (DIT) for the penux.uk domain

    dc=penux,dc=uk
      ou=pending   -- email given, awaiting OTP confirmation
      ou=people    -- verified, active accounts
      ou=groups    -- role groups (cn=members, ...)

and the state machine used to admit a brand-new user by proving control of
an email address before an entry is created under ou=people:

    REQUEST(email) -> PENDING (OTP mailed) -> VERIFY(otp) -> ACTIVE (entry
    provisioned under ou=people, password issued) | EXPIRED | REVOKED

The directory itself is a JSON-backed stand-in for a real LDAP backend
(no python-ldap / OpenLDAP dependency is required to run or test this);
`export_ldif()` renders the current tree as RFC 2849 LDIF so it can be
loaded into a real LDAP server for production use. See
docs/ldap_hierarchical_auth.md for the full protocol write-up.
"""
import argparse
import hashlib
import json
import os
import re
import secrets
import smtplib
import sys
import time
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable, Optional

BASE_DN = "dc=penux,dc=uk"
PENDING_OU = f"ou=pending,{BASE_DN}"
PEOPLE_OU = f"ou=people,{BASE_DN}"
GROUPS_OU = f"ou=groups,{BASE_DN}"
DEFAULT_GROUP_DN = f"cn=members,{GROUPS_OU}"

OTP_TTL_SECONDS = 15 * 60
OTP_MAX_ATTEMPTS = 5
ENROLL_RATE_LIMIT = 5      # max enrollment requests per email
ENROLL_RATE_WINDOW = 3600  # ...within this many seconds

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _default_store_path() -> Path:
    # When frozen (e.g. the PyInstaller Windows .exe built by
    # .github/workflows/build_ldap_client_exe.yml), __file__ points into a
    # throwaway extraction directory, so fall back to a stable per-user
    # location instead.
    if getattr(sys, "frozen", False):
        base = Path(os.environ.get("APPDATA") or Path(sys.executable).resolve().parent)
        return base / "penux" / "ldap_directory.json"
    return Path(__file__).resolve().parent / "data" / "ldap_directory.json"


DEFAULT_STORE = _default_store_path()

Mailer = Callable[[str, str, str], None]


def _now() -> float:
    return time.time()


def _hash_secret(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{{PBKDF2-SHA256}}{salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        _scheme, rest = stored.split("}", 1)
        salt_hex, digest_hex = rest.split("$", 1)
    except ValueError:
        return False
    salt = bytes.fromhex(salt_hex)
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return secrets.compare_digest(candidate.hex(), digest_hex)


def smtp_mailer(to_addr: str, subject: str, body: str) -> None:
    """Deliver mail through the penux.uk mail server (falls back to STARTTLS:587)."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = "netanel@penux.uk"
    msg["To"] = to_addr

    host = os.environ.get("PENUX_SMTP_HOST", "mail.penux.uk")
    last_error: Optional[Exception] = None
    for port, use_starttls in ((587, True), (25, False)):
        try:
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.ehlo()
                if use_starttls:
                    s.starttls()
                s.sendmail(msg["From"], [to_addr], msg.as_string())
            return
        except (OSError, smtplib.SMTPException) as exc:
            last_error = exc
            continue
    raise EnrollmentError(f"could not deliver mail to {to_addr} via {host}: {last_error}")


class EnrollmentError(Exception):
    """Raised for any invalid transition in the enrollment state machine."""


class HierarchicalDirectory:
    """JSON-backed hierarchical directory plus the email-verification protocol."""

    def __init__(self, store_path: Path = DEFAULT_STORE, mailer: Mailer = smtp_mailer):
        self.store_path = Path(store_path)
        self.mailer = mailer
        self._data = self._load()

    # -- persistence ----------------------------------------------------
    def _load(self) -> dict:
        if self.store_path.exists():
            return json.loads(self.store_path.read_text())
        return {
            "pending": {},
            "people": {},
            "groups": {DEFAULT_GROUP_DN: {"cn": "members", "member": []}},
        }

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    # -- helpers ----------------------------------------------------------
    def _uid_for(self, email: str) -> str:
        local = re.sub(r"[^a-z0-9._-]", "", email.split("@", 1)[0].lower()) or "user"
        taken = set(self._data["people"].keys())
        if local not in taken:
            return local
        n = 2
        while f"{local}{n}" in taken:
            n += 1
        return f"{local}{n}"

    # -- protocol step 1: request enrollment -----------------------------
    def request_enrollment(self, email: str) -> str:
        email = email.strip().lower()
        if not EMAIL_RE.match(email):
            raise EnrollmentError(f"invalid email address: {email!r}")
        for person in self._data["people"].values():
            if person["mail"] == email and person["penuxStatus"] == "active":
                raise EnrollmentError(f"{email} already has an active account")

        existing = self._data["pending"].get(email, {})
        recent = [t for t in existing.get("attempts_log", []) if _now() - t < ENROLL_RATE_WINDOW]
        if len(recent) >= ENROLL_RATE_LIMIT:
            raise EnrollmentError("rate limit exceeded, try again later")
        recent.append(_now())

        otp = f"{secrets.randbelow(1_000_000):06d}"
        dn = f"mail={email},{PENDING_OU}"
        self._data["pending"][email] = {
            "dn": dn,
            "otp_hash": _hash_secret(otp),
            "created_at": _now(),
            "expires_at": _now() + OTP_TTL_SECONDS,
            "verify_attempts": 0,
            "attempts_log": recent,
        }
        self._save()

        body = (
            f"A penux.uk account was requested for {email}.\n\n"
            f"Your verification code is: {otp}\n"
            f"It expires in {OTP_TTL_SECONDS // 60} minutes and can only be used once.\n\n"
            "If you did not request this, ignore this email."
        )
        self.mailer(email, "penux.uk - verify your email", body)
        return dn

    # -- protocol step 2: verify + provision -----------------------------
    def verify(self, email: str, otp: str) -> dict:
        email = email.strip().lower()
        record = self._data["pending"].get(email)
        if record is None:
            raise EnrollmentError("no pending enrollment for this email")
        if _now() > record["expires_at"]:
            del self._data["pending"][email]
            self._save()
            raise EnrollmentError("verification code expired, request a new one")
        if record["verify_attempts"] >= OTP_MAX_ATTEMPTS:
            del self._data["pending"][email]
            self._save()
            raise EnrollmentError("too many failed attempts, request a new code")
        if not secrets.compare_digest(_hash_secret(otp), record["otp_hash"]):
            record["verify_attempts"] += 1
            self._save()
            raise EnrollmentError("incorrect verification code")

        uid = self._uid_for(email)
        temp_password = secrets.token_urlsafe(12)
        dn = f"uid={uid},{PEOPLE_OU}"
        self._data["people"][uid] = {
            "dn": dn,
            "mail": email,
            "cn": uid,
            "penuxStatus": "active",
            "enrolledAt": record["created_at"],
            "verifiedAt": _now(),
            "memberOf": [DEFAULT_GROUP_DN],
            "userPassword": _hash_password(temp_password),
        }
        self._data["groups"][DEFAULT_GROUP_DN]["member"].append(dn)
        del self._data["pending"][email]
        self._save()

        return {"dn": dn, "uid": uid, "temp_password": temp_password}

    # -- simple LDAP-style bind (post-provisioning auth check) -----------
    def bind(self, uid: str, password: str) -> bool:
        person = self._data["people"].get(uid)
        if not person or person.get("penuxStatus") != "active":
            return False
        return _verify_password(password, person["userPassword"])

    # -- export as LDIF for a real LDAP server ---------------------------
    def export_ldif(self) -> str:
        lines = [
            f"dn: {BASE_DN}",
            "objectClass: dcObject",
            "objectClass: organization",
            "dc: penux",
            "o: penux.uk",
            "",
        ]
        for ou, dn in (("pending", PENDING_OU), ("people", PEOPLE_OU), ("groups", GROUPS_OU)):
            lines += [f"dn: {dn}", "objectClass: organizationalUnit", f"ou: {ou}", ""]

        for uid, person in sorted(self._data["people"].items()):
            lines += [
                f"dn: {person['dn']}",
                "objectClass: inetOrgPerson",
                "objectClass: penuxAccount",
                f"uid: {uid}",
                f"cn: {person['cn']}",
                f"mail: {person['mail']}",
                f"penuxStatus: {person['penuxStatus']}",
                f"userPassword: {person['userPassword']}",
            ]
            lines += [f"memberOf: {group_dn}" for group_dn in person["memberOf"]]
            lines.append("")

        for group_dn, group in sorted(self._data["groups"].items()):
            lines += [f"dn: {group_dn}", "objectClass: groupOfNames", f"cn: {group['cn']}"]
            lines += [f"member: {member}" for member in group["member"]]
            lines.append("")

        return "\n".join(lines)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="path to the JSON-backed directory store")
    sub = parser.add_subparsers(dest="command", required=True)

    enroll = sub.add_parser("enroll", help="request enrollment for a new email address")
    enroll.add_argument("email")

    verify = sub.add_parser("verify", help="verify a pending enrollment with its OTP")
    verify.add_argument("email")
    verify.add_argument("otp")

    sub.add_parser("export-ldif", help="dump the directory tree as LDIF")

    bind = sub.add_parser("bind", help="test a simple LDAP-style bind")
    bind.add_argument("uid")
    bind.add_argument("password")

    return parser


def main(argv=None) -> int:
    args = _build_cli().parse_args(argv)
    directory = HierarchicalDirectory(store_path=Path(args.store))
    try:
        if args.command == "enroll":
            dn = directory.request_enrollment(args.email)
            print(f"OTP sent to {args.email}; pending entry: {dn}")
        elif args.command == "verify":
            result = directory.verify(args.email, args.otp)
            print(f"Provisioned {result['dn']}")
            print(f"Temporary password (shown once): {result['temp_password']}")
        elif args.command == "export-ldif":
            print(directory.export_ldif())
        elif args.command == "bind":
            ok = directory.bind(args.uid, args.password)
            print("BIND OK" if ok else "BIND FAILED")
            return 0 if ok else 1
    except EnrollmentError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
