import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ldap_email_auth import (
    DEFAULT_GROUP_DN,
    HierarchicalDirectory,
    EnrollmentError,
    OTP_MAX_ATTEMPTS,
    OTP_TTL_SECONDS,
)


class RecordingMailer:
    def __init__(self):
        self.sent = []

    def __call__(self, to_addr, subject, body):
        self.sent.append((to_addr, subject, body))


def _otp_from(mailer):
    body = mailer.sent[-1][2]
    for line in body.splitlines():
        if line.startswith("Your verification code is:"):
            return line.rsplit(":", 1)[1].strip()
    raise AssertionError("no OTP found in mailed body")


def _new_directory(tmp_path):
    mailer = RecordingMailer()
    directory = HierarchicalDirectory(store_path=tmp_path / "directory.json", mailer=mailer)
    return directory, mailer


def test_full_enroll_verify_bind_flow(tmp_path):
    directory, mailer = _new_directory(tmp_path)

    dn = directory.request_enrollment("Alice@Example.com")
    assert dn == "mail=alice@example.com,ou=pending,dc=penux,dc=uk"
    assert len(mailer.sent) == 1
    assert mailer.sent[0][0] == "alice@example.com"

    otp = _otp_from(mailer)
    result = directory.verify("alice@example.com", otp)
    assert result["uid"] == "alice"
    assert result["dn"] == "uid=alice,ou=people,dc=penux,dc=uk"

    assert directory.bind("alice", result["temp_password"]) is True
    assert directory.bind("alice", "wrong-password") is False

    ldif = directory.export_ldif()
    assert "uid: alice" in ldif
    assert f"member: {result['dn']}" in ldif
    assert DEFAULT_GROUP_DN in ldif


def test_invalid_email_rejected(tmp_path):
    directory, _mailer = _new_directory(tmp_path)
    try:
        directory.request_enrollment("not-an-email")
        assert False, "expected EnrollmentError"
    except EnrollmentError:
        pass


def test_wrong_otp_increments_attempts_and_locks_out(tmp_path):
    directory, mailer = _new_directory(tmp_path)
    directory.request_enrollment("bob@example.com")

    for _ in range(OTP_MAX_ATTEMPTS):
        try:
            directory.verify("bob@example.com", "000000")
        except EnrollmentError:
            pass

    otp = _otp_from(mailer)
    try:
        directory.verify("bob@example.com", otp)
        assert False, "expected lockout after too many failed attempts"
    except EnrollmentError as exc:
        assert "too many failed attempts" in str(exc)


def test_expired_otp_rejected(tmp_path):
    directory, mailer = _new_directory(tmp_path)
    directory.request_enrollment("carol@example.com")
    otp = _otp_from(mailer)

    directory._data["pending"]["carol@example.com"]["expires_at"] = time.time() - 1
    directory._save()

    try:
        directory.verify("carol@example.com", otp)
        assert False, "expected expiry error"
    except EnrollmentError as exc:
        assert "expired" in str(exc)


def test_rate_limit_on_repeated_requests(tmp_path):
    directory, _mailer = _new_directory(tmp_path)
    for _ in range(5):
        directory.request_enrollment("dana@example.com")
    try:
        directory.request_enrollment("dana@example.com")
        assert False, "expected rate limit error"
    except EnrollmentError as exc:
        assert "rate limit" in str(exc)


def test_duplicate_active_account_rejected(tmp_path):
    directory, mailer = _new_directory(tmp_path)
    directory.request_enrollment("erin@example.com")
    otp = _otp_from(mailer)
    directory.verify("erin@example.com", otp)

    try:
        directory.request_enrollment("erin@example.com")
        assert False, "expected already-active error"
    except EnrollmentError as exc:
        assert "already has an active account" in str(exc)


def test_uid_collision_gets_suffixed(tmp_path):
    directory, mailer = _new_directory(tmp_path)

    directory.request_enrollment("frank@example.com")
    directory.verify("frank@example.com", _otp_from(mailer))

    directory.request_enrollment("frank@other.org")
    result = directory.verify("frank@other.org", _otp_from(mailer))
    assert result["uid"] == "frank2"


def test_store_persists_across_instances(tmp_path):
    store_path = tmp_path / "directory.json"
    mailer = RecordingMailer()

    directory = HierarchicalDirectory(store_path=store_path, mailer=mailer)
    directory.request_enrollment("gina@example.com")
    otp = _otp_from(mailer)
    result = directory.verify("gina@example.com", otp)

    reloaded = HierarchicalDirectory(store_path=store_path, mailer=mailer)
    assert reloaded.bind("gina", result["temp_password"]) is True
