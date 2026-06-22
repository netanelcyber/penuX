"""
Maildir++ storage backend.

Layout:
  <maildir_base>/<username>/
      INBOX/new/  INBOX/cur/  INBOX/tmp/
      .Sent/new/  .Sent/cur/  .Sent/tmp/
      .Drafts/...
      .uidmap.json   (per-folder UID maps, keyed by folder)
"""
from __future__ import annotations
import json
import os
import re
import time
import socket
import logging
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Maildir flag character → IMAP flag name
_MAILDIR_TO_IMAP: dict[str, str] = {
    "D": r"\Draft",
    "F": r"\Flagged",
    "P": r"$Forwarded",
    "R": r"\Answered",
    "S": r"\Seen",
    "T": r"\Deleted",
}
_IMAP_TO_MAILDIR: dict[str, str] = {v: k for k, v in _MAILDIR_TO_IMAP.items()}

_VALID_FLAGS = set(_MAILDIR_TO_IMAP.values())

# Shared per-process PID+hostname component for unique names
_HOSTNAME = socket.gethostname().replace("/", "_").replace(":", "_")

# Monotonic counter for unique Maildir filenames (prevents collision within same second)
_seq_lock = threading.Lock()
_seq = 0


def _next_unique() -> str:
    global _seq
    with _seq_lock:
        _seq += 1
        return f"{int(time.time())}.{_seq}.{os.getpid()}.{_HOSTNAME}"


def _maildir_flags_to_imap(flags_str: str) -> list[str]:
    return [_MAILDIR_TO_IMAP[c] for c in flags_str if c in _MAILDIR_TO_IMAP]


def _imap_flags_to_maildir(flags: list[str]) -> str:
    chars = sorted(
        _IMAP_TO_MAILDIR[f] for f in flags if f in _IMAP_TO_MAILDIR
    )
    return "".join(chars)


def _parse_flags_from_filename(name: str) -> list[str]:
    """Parse Maildir flags from filename like 'uid:2,SFR'."""
    m = re.search(r":2,([A-Za-z]*)$", name)
    if m:
        return _maildir_flags_to_imap(m.group(1))
    return []


def _set_flags_in_filename(name: str, flags: list[str]) -> str:
    base = re.sub(r":2,[A-Za-z]*$", "", name)
    flag_str = _imap_flags_to_maildir(flags)
    return f"{base}:2,{flag_str}"


class Message:
    __slots__ = ("uid", "seq", "path", "size", "_flags", "_cached_bytes")

    def __init__(self, uid: int, seq: int, path: Path):
        self.uid = uid
        self.seq = seq
        self.path = path
        self.size = path.stat().st_size
        self._flags: Optional[list[str]] = None
        self._cached_bytes: Optional[bytes] = None

    @property
    def flags(self) -> list[str]:
        if self._flags is None:
            self._flags = _parse_flags_from_filename(self.path.name)
            if self.path.parent.name == "new":
                # Messages in new/ are always \Recent
                if r"\Recent" not in self._flags:
                    self._flags.append(r"\Recent")
        return self._flags

    def read(self) -> bytes:
        if self._cached_bytes is None:
            self._cached_bytes = self.path.read_bytes()
        return self._cached_bytes

    def internaldate(self) -> str:
        """Return IMAP INTERNALDATE string."""
        ts = self.path.stat().st_mtime
        import datetime
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return dt.strftime("%d-%b-%Y %H:%M:%S +0000")


class Mailbox:
    """Represents a single Maildir folder (INBOX, .Sent, etc.)."""

    def __init__(self, folder_path: Path, uid_map: dict, folder_key: str):
        self.path = folder_path
        self._uid_map = uid_map       # shared dict loaded from JSON
        self._folder_key = folder_key
        self._messages: Optional[list[Message]] = None

    def _ensure_dirs(self):
        for sub in ("new", "cur", "tmp"):
            (self.path / sub).mkdir(parents=True, exist_ok=True)

    def _folder_uid_data(self) -> dict:
        return self._uid_map.setdefault(
            self._folder_key,
            {"uidvalidity": int(time.time()), "uidnext": 1, "map": {}},
        )

    @property
    def uidvalidity(self) -> int:
        return self._folder_uid_data()["uidvalidity"]

    @property
    def uidnext(self) -> int:
        return self._folder_uid_data()["uidnext"]

    def _scan_files(self) -> list[Path]:
        """Return all message paths in cur/ and new/, sorted by mtime."""
        files: list[Path] = []
        for sub in ("cur", "new"):
            d = self.path / sub
            if d.exists():
                files.extend(d.iterdir())
        files.sort(key=lambda p: p.stat().st_mtime)
        return files

    def _assign_uids(self, files: list[Path]) -> None:
        """Ensure every file has a UID in the map; assign new ones if needed."""
        data = self._folder_uid_data()
        uid_map: dict = data["map"]
        # Remove stale entries
        existing_names = {f.name for f in files}
        stale = [k for k in list(uid_map.keys()) if k not in existing_names]
        for k in stale:
            del uid_map[k]
        # Assign new UIDs
        changed = bool(stale)
        for f in files:
            if f.name not in uid_map:
                uid_map[f.name] = data["uidnext"]
                data["uidnext"] += 1
                changed = True
        if changed:
            data["map"] = uid_map

    def load(self) -> list[Message]:
        self._ensure_dirs()
        files = self._scan_files()
        self._assign_uids(files)
        data = self._folder_uid_data()
        uid_map = data["map"]
        msgs = []
        for seq, f in enumerate(files, 1):
            uid = uid_map.get(f.name, 0)
            if uid:
                msgs.append(Message(uid=uid, seq=seq, path=f))
        self._messages = msgs
        return msgs

    @property
    def messages(self) -> list[Message]:
        if self._messages is None:
            self.load()
        return self._messages

    def exists(self) -> int:
        return len(self.messages)

    def recent(self) -> int:
        return sum(1 for m in self.messages if r"\Recent" in m.flags)

    def unseen_seq(self) -> Optional[int]:
        for m in self.messages:
            if r"\Seen" not in m.flags:
                return m.seq
        return None

    def get_by_uid(self, uid: int) -> Optional[Message]:
        for m in self.messages:
            if m.uid == uid:
                return m
        return None

    def get_by_seq(self, seq: int) -> Optional[Message]:
        msgs = self.messages
        if 1 <= seq <= len(msgs):
            return msgs[seq - 1]
        return None

    def update_flags(self, msg: Message, new_flags: list[str], uid_map_save_fn) -> Message:
        """Set flags on a message (rename file to encode flags in Maildir style)."""
        # Move to cur/ if in new/
        old_path = msg.path
        new_name = _set_flags_in_filename(
            re.sub(r":2,[A-Za-z]*$", "", old_path.name), new_flags
        )
        new_path = old_path.parent.parent / "cur" / new_name

        if old_path != new_path:
            old_path.rename(new_path)
            # Update UID map key
            data = self._folder_uid_data()
            uid = data["map"].pop(old_path.name, None)
            if uid is not None:
                data["map"][new_path.name] = uid
            uid_map_save_fn()
            msg.path = new_path
            msg._flags = None  # re-parse on next access

        return msg

    def append(self, raw: bytes, flags: list[str], uid_map_save_fn) -> int:
        """Write a new message and return its UID."""
        self._ensure_dirs()
        unique = _next_unique()
        flag_str = _imap_flags_to_maildir(flags)
        filename = f"{unique}:2,{flag_str}"
        dest = self.path / "cur" / filename
        dest.write_bytes(raw)

        data = self._folder_uid_data()
        uid = data["uidnext"]
        data["uidnext"] += 1
        data["map"][filename] = uid
        uid_map_save_fn()

        # Invalidate cache
        self._messages = None
        return uid

    def expunge(self, uid_map_save_fn) -> list[int]:
        """Delete messages flagged \\Deleted. Returns list of expunged seq nums."""
        msgs = self.load()
        expunged_seqs = []
        for msg in msgs:
            if r"\Deleted" in msg.flags:
                expunged_seqs.append(msg.seq)
                data = self._folder_uid_data()
                data["map"].pop(msg.path.name, None)
                try:
                    msg.path.unlink()
                except FileNotFoundError:
                    pass
        if expunged_seqs:
            uid_map_save_fn()
            self._messages = None
        return expunged_seqs


class UserMaildir:
    """Root Maildir++ directory for a single user."""

    SYSTEM_FOLDERS = ["INBOX", "Sent", "Drafts", "Trash", "Junk"]

    def __init__(self, base: Path):
        self.base = base
        self._uid_map_path = base / ".uidmap.json"
        self._uid_map: Optional[dict] = None

    def _load_uid_map(self) -> dict:
        if self._uid_map is None:
            if self._uid_map_path.exists():
                try:
                    self._uid_map = json.loads(self._uid_map_path.read_text())
                except json.JSONDecodeError:
                    self._uid_map = {}
            else:
                self._uid_map = {}
        return self._uid_map

    def _save_uid_map(self):
        self._uid_map_path.write_text(json.dumps(self._uid_map, indent=2))

    def _folder_path(self, folder: str) -> Path:
        if folder == "INBOX":
            return self.base / "INBOX"
        # Maildir++ convention: subfolders are dot-prefixed
        return self.base / f".{folder}"

    def _folder_key(self, folder: str) -> str:
        return folder

    def ensure_default_folders(self):
        """Create default folder structure if it doesn't exist."""
        for f in self.SYSTEM_FOLDERS:
            path = self._folder_path(f)
            for sub in ("new", "cur", "tmp"):
                (path / sub).mkdir(parents=True, exist_ok=True)

    def get_mailbox(self, folder: str) -> Optional[Mailbox]:
        self._load_uid_map()
        path = self._folder_path(folder)
        if not (path / "cur").exists():
            return None
        return Mailbox(path, self._uid_map, self._folder_key(folder))

    def create_folder(self, folder: str) -> bool:
        path = self._folder_path(folder)
        if (path / "cur").exists():
            return False  # already exists
        for sub in ("new", "cur", "tmp"):
            (path / sub).mkdir(parents=True, exist_ok=True)
        return True

    def delete_folder(self, folder: str) -> bool:
        if folder == "INBOX":
            return False
        path = self._folder_path(folder)
        if not (path / "cur").exists():
            return False
        import shutil
        shutil.rmtree(path)
        uid_map = self._load_uid_map()
        uid_map.pop(self._folder_key(folder), None)
        self._save_uid_map()
        return True

    def rename_folder(self, old: str, new: str) -> bool:
        if old == "INBOX":
            return False
        old_path = self._folder_path(old)
        new_path = self._folder_path(new)
        if not (old_path / "cur").exists():
            return False
        if (new_path / "cur").exists():
            return False
        old_path.rename(new_path)
        uid_map = self._load_uid_map()
        data = uid_map.pop(self._folder_key(old), None)
        if data:
            uid_map[self._folder_key(new)] = data
        self._save_uid_map()
        return True

    def list_folders(self) -> list[str]:
        folders = []
        if (self.base / "INBOX" / "cur").exists():
            folders.append("INBOX")
        for item in sorted(self.base.iterdir()):
            if item.is_dir() and item.name.startswith(".") and not item.name.startswith(".."):
                name = item.name[1:]  # strip leading dot
                if (item / "cur").exists() and name not in ("", ):
                    folders.append(name)
        return folders

    def save_mailbox(self):
        """Save UID map to disk (called after mutations)."""
        if self._uid_map is not None:
            self._save_uid_map()

    def get_mailbox_with_save(self, folder: str):
        """Get mailbox with a bound save callback."""
        mb = self.get_mailbox(folder)
        if mb is None:
            return None
        # Bind the save method to this instance
        mb._uid_map = self._load_uid_map()
        return mb, self.save_mailbox
