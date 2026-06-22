"""Configuration for the PenuX IMAP server."""
import os
import json
import hashlib
import secrets
from pathlib import Path


class Config:
    def __init__(self):
        self.host = os.environ.get("IMAP_HOST", "0.0.0.0")
        self.port_imap = int(os.environ.get("IMAP_PORT", "143"))
        self.port_imaps = int(os.environ.get("IMAPS_PORT", "993"))
        self.maildir_base = Path(os.environ.get("IMAP_MAILDIR", "/var/mail/penux"))
        self.tls_cert = os.environ.get("IMAP_TLS_CERT", "/etc/ssl/penux/fullchain.pem")
        self.tls_key = os.environ.get("IMAP_TLS_KEY", "/etc/ssl/penux/privkey.pem")
        self.server_name = os.environ.get("IMAP_SERVERNAME", "mail.penux.uk")
        self.users_file = Path(os.environ.get("IMAP_USERS_FILE", "/etc/penux-imap/users.json"))
        self.max_connections = int(os.environ.get("IMAP_MAX_CONNECTIONS", "100"))
        self.idle_timeout = int(os.environ.get("IMAP_IDLE_TIMEOUT", "1800"))

    def load_users(self) -> dict:
        if self.users_file.exists():
            with open(self.users_file) as f:
                return json.load(f)
        # Fallback: check env for a single admin user (dev/testing)
        user = os.environ.get("IMAP_USER")
        pw = os.environ.get("IMAP_PASS")
        if user and pw:
            return {user: _hash_password(pw)}
        return {}

    def check_password(self, username: str, password: str) -> bool:
        users = self.load_users()
        if username not in users:
            return False
        stored = users[username]
        return _verify_password(stored, password)

    def get_maildir(self, username: str) -> Path:
        # Sanitise username — only allow safe characters
        safe = "".join(c for c in username if c.isalnum() or c in "-_.")
        return self.maildir_base / safe

    def tls_available(self) -> bool:
        return Path(self.tls_cert).exists() and Path(self.tls_key).exists()


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    h = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return f"$sha256${salt}${h}"


def _verify_password(stored: str, password: str) -> bool:
    if stored.startswith("$sha256$"):
        parts = stored.split("$")
        # $sha256$<salt>$<hash>
        salt, hashed = parts[2], parts[3]
        check = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
        return secrets.compare_digest(check, hashed)
    # Plain-text comparison (dev only)
    return secrets.compare_digest(stored, password)
