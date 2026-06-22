#!/usr/bin/env python3
"""
User management CLI for the PenuX IMAP server.

Usage:
  python manage_users.py add <username> <password>
  python manage_users.py delete <username>
  python manage_users.py list
  python manage_users.py passwd <username> <new_password>
"""
import json
import sys
from pathlib import Path
from imap_server.config import Config, _hash_password


def load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved to {path}")


def main():
    config = Config()
    users_file = config.users_file

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    users = load(users_file)

    if cmd == "add":
        if len(sys.argv) < 4:
            print("Usage: manage_users.py add <username> <password>")
            sys.exit(1)
        username, password = sys.argv[2], sys.argv[3]
        if username in users:
            print(f"User '{username}' already exists. Use passwd to change password.")
            sys.exit(1)
        users[username] = _hash_password(password)
        save(users_file, users)
        # Create maildir
        maildir = config.get_maildir(username)
        from imap_server.mailstore import UserMaildir
        ud = UserMaildir(maildir)
        ud.ensure_default_folders()
        print(f"User '{username}' created with maildir at {maildir}")

    elif cmd == "passwd":
        if len(sys.argv) < 4:
            print("Usage: manage_users.py passwd <username> <new_password>")
            sys.exit(1)
        username, password = sys.argv[2], sys.argv[3]
        if username not in users:
            print(f"User '{username}' not found.")
            sys.exit(1)
        users[username] = _hash_password(password)
        save(users_file, users)
        print(f"Password updated for '{username}'")

    elif cmd == "delete":
        if len(sys.argv) < 3:
            print("Usage: manage_users.py delete <username>")
            sys.exit(1)
        username = sys.argv[2]
        if username not in users:
            print(f"User '{username}' not found.")
            sys.exit(1)
        del users[username]
        save(users_file, users)
        print(f"User '{username}' deleted (maildir preserved)")

    elif cmd == "list":
        if not users:
            print("No users configured.")
        for u in sorted(users):
            print(f"  {u}")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
