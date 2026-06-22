"""
IMAP4rev1 (RFC 3501) protocol implementation.

State machine: NOT_AUTH -> AUTHENTICATED -> SELECTED -> LOGOUT
"""
from __future__ import annotations
import asyncio
import email
import email.header
import email.message
import email.policy
import logging
import re
import time
from typing import Optional, Union

from .config import Config
from .mailstore import UserMaildir, Mailbox, Message, _VALID_FLAGS

log = logging.getLogger(__name__)

CRLF = b"\r\n"
CAPABILITIES = "IMAP4rev1 STARTTLS AUTH=PLAIN IDLE UIDPLUS LITERAL+"


# ---------------------------------------------------------------------------
# Lightweight IMAP command parser
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


def _decode_header_value(val: Optional[str]) -> str:
    if not val:
        return ""
    parts = []
    for raw, enc in email.header.decode_header(val):
        if isinstance(raw, bytes):
            charset = enc or "utf-8"
            try:
                parts.append(raw.decode(charset, errors="replace"))
            except (LookupError, UnicodeDecodeError):
                parts.append(raw.decode("utf-8", errors="replace"))
        else:
            parts.append(raw)
    return " ".join(parts)


def _imap_quote(s: str) -> str:
    """Return s as a quoted IMAP string."""
    if s is None:
        return "NIL"
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _nstring(s: Optional[str]) -> str:
    return "NIL" if s is None else _imap_quote(s)


def _address_list(header_val: Optional[str]) -> str:
    """Parse RFC 822 address list into IMAP address structure."""
    if not header_val:
        return "NIL"
    import email.utils
    pairs = email.utils.getaddresses([header_val])
    if not pairs:
        return "NIL"
    parts = []
    for name, addr in pairs:
        if "@" in addr:
            mailbox, host = addr.rsplit("@", 1)
        else:
            mailbox, host = addr, ""
        parts.append(
            f"({_nstring(name or None)} NIL {_nstring(mailbox or None)} {_nstring(host or None)})"
        )
    return "(" + " ".join(parts) + ")"


def _build_envelope(msg: email.message.Message) -> str:
    date = _nstring(_decode_header_value(msg.get("Date")))
    subject = _nstring(_decode_header_value(msg.get("Subject")))
    frm = _address_list(msg.get("From"))
    sender = _address_list(msg.get("Sender") or msg.get("From"))
    reply_to = _address_list(msg.get("Reply-To") or msg.get("From"))
    to = _address_list(msg.get("To"))
    cc = _address_list(msg.get("Cc"))
    bcc = _address_list(msg.get("Bcc"))
    in_reply_to = _nstring(msg.get("In-Reply-To"))
    message_id = _nstring(msg.get("Message-ID"))
    return (
        f"({date} {subject} {frm} {sender} {reply_to} "
        f"{to} {cc} {bcc} {in_reply_to} {message_id})"
    )


def _body_params(params: list[tuple[str, str]]) -> str:
    if not params:
        return "NIL"
    parts = []
    for k, v in params:
        parts.append(_imap_quote(k))
        parts.append(_imap_quote(v))
    return "(" + " ".join(parts) + ")"


def _build_bodystructure(msg: email.message.Message, extended: bool = False) -> str:
    ctype = msg.get_content_type()
    maintype, subtype = ctype.split("/", 1)
    params = msg.get_params() or []
    # Remove content-type itself
    params = [(k, v) for k, v in params if k.lower() != ctype.lower()]

    if msg.is_multipart():
        parts = [_build_bodystructure(p, extended) for p in msg.get_payload()]
        result = "(" + " ".join(parts) + f" {_imap_quote(subtype)}"
        if extended:
            result += " NIL NIL NIL NIL"
        result += ")"
        return result

    body = msg.get_payload(decode=True) or b""
    size = len(body)
    lines = body.count(b"\n")
    encoding = _nstring(msg.get("Content-Transfer-Encoding"))
    description = _nstring(msg.get("Content-Description"))
    content_id = _nstring(msg.get("Content-ID"))

    result = (
        f"({_imap_quote(maintype)} {_imap_quote(subtype)} "
        f"{_body_params(params)} "
        f"{content_id} {description} {encoding} {size}"
    )
    if maintype.lower() == "text":
        result += f" {lines}"
    if extended:
        md5 = "NIL"
        disposition = "NIL"
        language = "NIL"
        location = "NIL"
        result += f" {md5} {disposition} {language} {location}"
    result += ")"
    return result


def _get_body_section(msg: email.message.Message, section: str) -> Optional[bytes]:
    """Return the bytes for a given BODY[section] specifier."""
    section = section.strip()

    if section == "" or section == "TEXT":
        # Full message text (body without headers if TEXT)
        raw = msg.as_bytes()
        if section == "TEXT":
            # Body without headers: find \r\n\r\n or \n\n
            idx = raw.find(b"\r\n\r\n")
            if idx == -1:
                idx = raw.find(b"\n\n")
                if idx == -1:
                    return b""
                return raw[idx + 2:]
            return raw[idx + 4:]
        return raw

    if section == "HEADER":
        raw = msg.as_bytes()
        idx = raw.find(b"\r\n\r\n")
        if idx == -1:
            idx = raw.find(b"\n\n")
            if idx == -1:
                return raw
            return raw[: idx + 2]
        return raw[: idx + 4]

    if section.startswith("HEADER.FIELDS"):
        m = re.match(r"HEADER\.FIELDS(?:\.NOT)?\s+\(([^)]+)\)", section, re.I)
        negate = "NOT" in section.upper()
        if m:
            wanted = {h.upper() for h in m.group(1).split()}
            lines = []
            raw = msg.as_bytes()
            end = raw.find(b"\r\n\r\n")
            if end == -1:
                end = raw.find(b"\n\n")
            header_bytes = raw[: end + 4] if end != -1 else raw
            for line in header_bytes.split(b"\n"):
                stripped = line.rstrip(b"\r")
                colon = stripped.find(b":")
                if colon != -1:
                    field = stripped[:colon].decode("utf-8", errors="replace").upper()
                    match = field in wanted
                    if match != negate:
                        lines.append(line)
                elif stripped == b"":
                    lines.append(b"\r\n")
                    break
                elif stripped and (stripped[0:1] in (b" ", b"\t")):
                    # Continuation of previous header
                    if lines:
                        lines.append(line)
            return b"\n".join(lines)

    # Numeric part specifier like "1" or "1.2"
    parts = section.split(".")
    current = msg
    for part_str in parts:
        if part_str.isdigit():
            idx = int(part_str) - 1
            if current.is_multipart():
                payload = current.get_payload()
                if 0 <= idx < len(payload):
                    current = payload[idx]
                else:
                    return None
            else:
                if idx == 0:
                    pass  # stay at current
                else:
                    return None
        elif part_str.upper() == "TEXT":
            raw = current.as_bytes()
            end = raw.find(b"\r\n\r\n")
            if end == -1:
                end = raw.find(b"\n\n")
                if end == -1:
                    return b""
                return raw[end + 2:]
            return raw[end + 4:]
        elif part_str.upper() == "HEADER":
            raw = current.as_bytes()
            end = raw.find(b"\r\n\r\n")
            if end == -1:
                end = raw.find(b"\n\n")
                if end == -1:
                    return raw
                return raw[: end + 2]
            return raw[: end + 4]

    payload = current.get_payload(decode=True)
    if payload is None:
        return current.as_bytes()
    return payload


# ---------------------------------------------------------------------------
# Sequence / UID set parser
# ---------------------------------------------------------------------------

def _parse_seqset(seqset: str, max_seq: int) -> list[int]:
    """Expand an IMAP sequence set like '1,3:5,*' into a sorted list."""
    result = set()
    for part in seqset.split(","):
        part = part.strip()
        if ":" in part:
            lo, hi = part.split(":", 1)
            lo_n = max_seq if lo == "*" else int(lo)
            hi_n = max_seq if hi == "*" else int(hi)
            if lo_n > hi_n:
                lo_n, hi_n = hi_n, lo_n
            result.update(range(lo_n, hi_n + 1))
        else:
            n = max_seq if part == "*" else int(part)
            result.add(n)
    return sorted(result)


# ---------------------------------------------------------------------------
# Simple tokeniser for IMAP command lines
# ---------------------------------------------------------------------------

def _tokenise(line: str) -> list[str]:
    """Split an IMAP command line into tokens (handles quoted strings, parens)."""
    tokens: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
        elif c == '"':
            # Quoted string
            j = i + 1
            buf = []
            while j < n and line[j] != '"':
                if line[j] == '\\' and j + 1 < n:
                    buf.append(line[j + 1])
                    j += 2
                else:
                    buf.append(line[j])
                    j += 1
            tokens.append("".join(buf))
            i = j + 1
        elif c == '(':
            # Find matching paren
            depth = 0
            j = i
            while j < n:
                if line[j] == '(':
                    depth += 1
                elif line[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            tokens.append(line[i:j + 1])
            i = j + 1
        elif c == '[':
            # Section spec [HEADER.FIELDS (From To)]
            depth = 0
            j = i
            while j < n:
                if line[j] == '[':
                    depth += 1
                elif line[j] == ']':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            tokens.append(line[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < n and line[j] not in " \t":
                j += 1
            tokens.append(line[i:j])
            i = j
    return tokens


def _parse_flag_list(token: str) -> list[str]:
    """Parse '(\\Seen \\Draft)' into ['\\Seen', '\\Draft']."""
    inner = token.strip("()")
    return [f.strip() for f in inner.split() if f.strip()]


# ---------------------------------------------------------------------------
# IMAP session handler
# ---------------------------------------------------------------------------

class State:
    NOT_AUTH = "NOT_AUTH"
    AUTHENTICATED = "AUTHENTICATED"
    SELECTED = "SELECTED"
    LOGOUT = "LOGOUT"


class IMAPSession:
    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        config: Config,
        tls_ctx=None,
    ):
        self._reader = reader
        self._writer = writer
        self._config = config
        self._tls_ctx = tls_ctx
        self._state = State.NOT_AUTH
        self._username: Optional[str] = None
        self._userdir: Optional[UserMaildir] = None
        self._mailbox: Optional[Mailbox] = None
        self._mailbox_name: Optional[str] = None
        self._read_only = False
        self._idle = False
        self._peer = writer.get_extra_info("peername", ("?", 0))

    # -----------------------------------------------------------------------
    # I/O helpers
    # -----------------------------------------------------------------------

    async def _send(self, *lines: bytes):
        for line in lines:
            self._writer.write(line + CRLF)
        await self._writer.drain()

    async def _send_raw(self, data: bytes):
        self._writer.write(data)
        await self._writer.drain()

    async def _readline(self) -> Optional[bytes]:
        try:
            line = await asyncio.wait_for(
                self._reader.readline(), timeout=self._config.idle_timeout
            )
        except asyncio.TimeoutError:
            return None
        if not line:
            return None
        return line.rstrip(b"\r\n")

    async def _read_literal(self, size: int) -> bytes:
        data = b""
        remaining = size
        while remaining > 0:
            chunk = await self._reader.read(remaining)
            if not chunk:
                break
            data += chunk
            remaining -= len(chunk)
        return data

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    async def run(self):
        await self._send(
            f"* OK [{self._config.server_name}] PenuX IMAP4rev1 server ready".encode()
        )
        while self._state != State.LOGOUT:
            line = await self._readline()
            if line is None:
                break
            if not line:
                continue

            log.debug("%s -> %s", self._peer, line[:200])

            # Handle literal continuation
            full_line, literals = await self._read_command_with_literals(line)
            if full_line is None:
                continue

            await self._dispatch(full_line, literals)

        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:
            pass

    async def _read_command_with_literals(
        self, first_line: bytes
    ) -> tuple[Optional[str], dict[int, bytes]]:
        """Read a complete IMAP command, fetching any literal strings."""
        parts = [first_line.decode("utf-8", errors="replace")]
        literals: dict[int, bytes] = {}
        current = first_line

        while True:
            m = re.search(rb"\{(\d+)\+?\}\s*$", current)
            if not m:
                break
            size = int(m.group(1))
            # Non-synchronising literal ({N+}) — don't send continuation
            if not current.rstrip().endswith(b"+}"):
                await self._send_raw(b"+ Ready for literal data\r\n")
            raw = await self._read_literal(size)
            literal_idx = len(literals)
            literals[literal_idx] = raw
            parts.append(f"\x00LITERAL{literal_idx}\x00")
            # Read next line
            current = await self._readline()
            if current is None:
                return None, {}
            parts.append(current.decode("utf-8", errors="replace"))

        return "".join(parts), literals

    async def _dispatch(self, line: str, literals: dict[int, bytes]):
        # Expand literals back into the line for simple cases
        def expand(s: str) -> str:
            for idx, data in literals.items():
                s = s.replace(
                    f"\x00LITERAL{idx}\x00", data.decode("utf-8", errors="replace")
                )
            return s

        tokens = _tokenise(line)
        if len(tokens) < 2:
            await self._send(b"* BAD Empty command")
            return

        tag = tokens[0]
        command = tokens[1].upper()
        args = tokens[2:]

        # For APPEND, keep raw literal bytes
        raw_literals = literals

        try:
            if command == "CAPABILITY":
                await self._cmd_capability(tag)
            elif command == "NOOP":
                await self._cmd_noop(tag)
            elif command == "LOGOUT":
                await self._cmd_logout(tag)
            elif command == "STARTTLS":
                await self._cmd_starttls(tag)
            elif command == "AUTHENTICATE":
                method = args[0].upper() if args else ""
                await self._cmd_authenticate(tag, method, expand(args[1]) if len(args) > 1 else None)
            elif command == "LOGIN":
                if len(args) < 2:
                    await self._bad(tag, "LOGIN requires username and password")
                    return
                await self._cmd_login(tag, expand(args[0]), expand(args[1]))
            elif self._state == State.NOT_AUTH:
                await self._no(tag, "Please authenticate first")
            elif command == "SELECT":
                await self._cmd_select(tag, expand(args[0]) if args else "INBOX", read_only=False)
            elif command == "EXAMINE":
                await self._cmd_select(tag, expand(args[0]) if args else "INBOX", read_only=True)
            elif command == "CREATE":
                await self._cmd_create(tag, expand(args[0]) if args else "")
            elif command == "DELETE":
                await self._cmd_delete(tag, expand(args[0]) if args else "")
            elif command == "RENAME":
                if len(args) < 2:
                    await self._bad(tag, "RENAME requires old and new names")
                    return
                await self._cmd_rename(tag, expand(args[0]), expand(args[1]))
            elif command in ("SUBSCRIBE", "UNSUBSCRIBE"):
                await self._ok(tag, f"{command} completed")
            elif command == "LIST":
                ref = expand(args[0]) if len(args) > 0 else ""
                pattern = expand(args[1]) if len(args) > 1 else "*"
                await self._cmd_list(tag, ref, pattern)
            elif command == "LSUB":
                ref = expand(args[0]) if len(args) > 0 else ""
                pattern = expand(args[1]) if len(args) > 1 else "*"
                await self._cmd_list(tag, ref, pattern, lsub=True)
            elif command == "STATUS":
                if len(args) < 2:
                    await self._bad(tag, "STATUS requires mailbox and items")
                    return
                await self._cmd_status(tag, expand(args[0]), args[1])
            elif command == "APPEND":
                await self._cmd_append(tag, args, raw_literals)
            elif command == "CHECK":
                await self._ok(tag, "CHECK completed")
            elif command == "CLOSE":
                await self._cmd_close(tag)
            elif command == "EXPUNGE":
                await self._cmd_expunge(tag)
            elif command == "SEARCH":
                await self._cmd_search(tag, args, uid=False)
            elif command == "FETCH":
                if len(args) < 2:
                    await self._bad(tag, "FETCH requires sequence set and items")
                    return
                await self._cmd_fetch(tag, args[0], args[1], uid=False)
            elif command == "STORE":
                if len(args) < 3:
                    await self._bad(tag, "STORE requires seq, item, value")
                    return
                await self._cmd_store(tag, args[0], args[1], args[2], uid=False)
            elif command == "COPY":
                if len(args) < 2:
                    await self._bad(tag, "COPY requires sequence set and mailbox")
                    return
                await self._cmd_copy(tag, args[0], expand(args[1]), uid=False)
            elif command == "UID":
                if not args:
                    await self._bad(tag, "UID requires sub-command")
                    return
                sub = args[0].upper()
                sub_args = args[1:]
                if sub == "FETCH":
                    await self._cmd_fetch(tag, sub_args[0], sub_args[1] if len(sub_args) > 1 else "(FLAGS)", uid=True)
                elif sub == "STORE":
                    await self._cmd_store(tag, sub_args[0], sub_args[1], sub_args[2] if len(sub_args) > 2 else "()", uid=True)
                elif sub == "SEARCH":
                    await self._cmd_search(tag, sub_args, uid=True)
                elif sub == "COPY":
                    await self._cmd_copy(tag, sub_args[0], expand(sub_args[1]) if len(sub_args) > 1 else "", uid=True)
                else:
                    await self._bad(tag, f"Unknown UID sub-command: {sub}")
            elif command == "IDLE":
                await self._cmd_idle(tag)
            else:
                await self._bad(tag, f"Unknown command: {command}")
        except Exception as exc:
            log.exception("Error handling %s %s", command, args)
            await self._no(tag, f"Internal error: {exc}")

    # -----------------------------------------------------------------------
    # Tagged response helpers
    # -----------------------------------------------------------------------

    async def _ok(self, tag: str, msg: str, code: str = ""):
        code_str = f"[{code}] " if code else ""
        await self._send(f"{tag} OK {code_str}{msg}".encode())

    async def _no(self, tag: str, msg: str):
        await self._send(f"{tag} NO {msg}".encode())

    async def _bad(self, tag: str, msg: str):
        await self._send(f"{tag} BAD {msg}".encode())

    # -----------------------------------------------------------------------
    # State checks
    # -----------------------------------------------------------------------

    def _require_selected(self) -> bool:
        return self._state == State.SELECTED and self._mailbox is not None

    def _reload_mailbox(self):
        if self._mailbox is not None and self._mailbox_name:
            self._mailbox.load()

    # -----------------------------------------------------------------------
    # Commands — pre-auth
    # -----------------------------------------------------------------------

    async def _cmd_capability(self, tag: str):
        caps = CAPABILITIES
        if self._tls_ctx is None:
            caps = caps.replace(" STARTTLS", "")
        await self._send(f"* CAPABILITY {caps}".encode())
        await self._ok(tag, "CAPABILITY completed")

    async def _cmd_noop(self, tag: str):
        if self._require_selected():
            self._reload_mailbox()
            await self._send_mailbox_status()
        await self._ok(tag, "NOOP completed")

    async def _cmd_logout(self, tag: str):
        await self._send(b"* BYE PenuX IMAP server logging out")
        await self._ok(tag, "LOGOUT completed")
        self._state = State.LOGOUT

    async def _cmd_starttls(self, tag: str):
        if self._tls_ctx is None:
            await self._no(tag, "STARTTLS not available")
            return
        await self._ok(tag, "Begin TLS negotiation now")
        # Upgrade the connection
        loop = asyncio.get_event_loop()
        transport = self._writer.transport
        protocol = transport.get_protocol()
        new_transport = await loop.start_tls(transport, protocol, self._tls_ctx)
        self._writer._transport = new_transport  # type: ignore[attr-defined]

    async def _cmd_authenticate(self, tag: str, method: str, initial: Optional[str]):
        if method != "PLAIN":
            await self._no(tag, f"Unsupported authentication mechanism: {method}")
            return
        if initial is None:
            await self._send_raw(b"+ \r\n")
            line = await self._readline()
            if line is None:
                return
            initial = line.decode("utf-8", errors="replace")
        import base64
        try:
            decoded = base64.b64decode(initial).decode("utf-8")
            parts = decoded.split("\x00")
            # authzid\0authcid\0passwd
            if len(parts) == 3:
                username, password = parts[1], parts[2]
            else:
                username, password = parts[0], parts[-1]
        except Exception:
            await self._bad(tag, "Invalid AUTHENTICATE data")
            return
        await self._do_login(tag, username, password)

    async def _cmd_login(self, tag: str, username: str, password: str):
        await self._do_login(tag, username, password)

    async def _do_login(self, tag: str, username: str, password: str):
        if not self._config.check_password(username, password):
            await self._no(tag, "Invalid credentials")
            return
        self._username = username
        self._userdir = UserMaildir(self._config.get_maildir(username))
        self._userdir.ensure_default_folders()
        self._state = State.AUTHENTICATED
        log.info("Login: %s from %s", username, self._peer)
        await self._ok(tag, "LOGIN completed")

    # -----------------------------------------------------------------------
    # Commands — authenticated
    # -----------------------------------------------------------------------

    async def _cmd_select(self, tag: str, folder: str, read_only: bool):
        folder = folder.strip('"')
        mb = self._userdir.get_mailbox(folder)
        if mb is None:
            await self._no(tag, f"Mailbox '{folder}' does not exist")
            return
        self._mailbox = mb
        self._mailbox_name = folder
        self._read_only = read_only
        self._state = State.SELECTED
        msgs = mb.load()
        await self._send_mailbox_status(full=True)
        rw = "READ-ONLY" if read_only else "READ-WRITE"
        await self._ok(tag, "SELECT completed", code=rw)

    async def _send_mailbox_status(self, full: bool = False):
        mb = self._mailbox
        if mb is None:
            return
        if full:
            await self._send(
                f"* FLAGS (\\Answered \\Flagged \\Deleted \\Seen \\Draft)".encode(),
                f"* OK [PERMANENTFLAGS (\\Answered \\Flagged \\Deleted \\Seen \\Draft \\*)] Flags permitted".encode(),
                f"* {mb.exists()} EXISTS".encode(),
                f"* {mb.recent()} RECENT".encode(),
                f"* OK [UIDVALIDITY {mb.uidvalidity}] UIDs valid".encode(),
                f"* OK [UIDNEXT {mb.uidnext}] Predicted next UID".encode(),
            )
            unseen = mb.unseen_seq()
            if unseen:
                await self._send(f"* OK [UNSEEN {unseen}] First unseen".encode())
        else:
            await self._send(
                f"* {mb.exists()} EXISTS".encode(),
                f"* {mb.recent()} RECENT".encode(),
            )

    async def _cmd_create(self, tag: str, folder: str):
        folder = folder.strip('"')
        if not folder:
            await self._bad(tag, "CREATE requires a mailbox name")
            return
        ok = self._userdir.create_folder(folder)
        if ok:
            await self._ok(tag, "CREATE completed")
        else:
            await self._no(tag, "Mailbox already exists")

    async def _cmd_delete(self, tag: str, folder: str):
        folder = folder.strip('"')
        ok = self._userdir.delete_folder(folder)
        if ok:
            await self._ok(tag, "DELETE completed")
        else:
            await self._no(tag, "DELETE failed: cannot delete or mailbox not found")

    async def _cmd_rename(self, tag: str, old: str, new: str):
        old = old.strip('"')
        new = new.strip('"')
        ok = self._userdir.rename_folder(old, new)
        if ok:
            await self._ok(tag, "RENAME completed")
        else:
            await self._no(tag, "RENAME failed")

    async def _cmd_list(self, tag: str, ref: str, pattern: str, lsub: bool = False):
        cmd = "LSUB" if lsub else "LIST"
        # Convert glob pattern to regex
        escaped = re.escape(pattern).replace(r"\*", ".*").replace(r"\%", "[^/]*")
        try:
            pat = re.compile(f"^{escaped}$", re.IGNORECASE)
        except re.error:
            pat = re.compile(".*")

        folders = self._userdir.list_folders()
        for folder in folders:
            if pat.match(folder):
                await self._send(f'* {cmd} (\\HasNoChildren) "/" {_imap_quote(folder)}'.encode())
        # Special: list "" returns hierarchy delimiter
        if pattern == "":
            await self._send(f'* {cmd} (\\Noselect) "/" ""'.encode())
        await self._ok(tag, f"{cmd} completed")

    async def _cmd_status(self, tag: str, folder: str, items_token: str):
        folder = folder.strip('"')
        mb = self._userdir.get_mailbox(folder)
        if mb is None:
            await self._no(tag, f"Mailbox '{folder}' not found")
            return
        msgs = mb.load()
        items = items_token.strip("()").upper().split()
        parts = []
        for item in items:
            if item == "MESSAGES":
                parts.append(f"MESSAGES {mb.exists()}")
            elif item == "RECENT":
                parts.append(f"RECENT {mb.recent()}")
            elif item == "UIDNEXT":
                parts.append(f"UIDNEXT {mb.uidnext}")
            elif item == "UIDVALIDITY":
                parts.append(f"UIDVALIDITY {mb.uidvalidity}")
            elif item == "UNSEEN":
                count = sum(1 for m in msgs if r"\Seen" not in m.flags)
                parts.append(f"UNSEEN {count}")
        await self._send(f"* STATUS {_imap_quote(folder)} ({' '.join(parts)})".encode())
        await self._ok(tag, "STATUS completed")

    async def _cmd_append(self, tag: str, args: list[str], raw_literals: dict[int, bytes]):
        if not args:
            await self._bad(tag, "APPEND requires mailbox")
            return
        folder = args[0].strip('"')
        mb = self._userdir.get_mailbox(folder)
        if mb is None:
            await self._no(tag, f"[TRYCREATE] Mailbox '{folder}' not found")
            return

        # Find flags and date in remaining args
        flags: list[str] = []
        i = 1
        while i < len(args):
            tok = args[i]
            if tok.startswith("("):
                flags = _parse_flag_list(tok)
                i += 1
            elif i + 1 < len(args) and args[i + 1].startswith("\""):
                i += 2  # skip INTERNALDATE
            else:
                break

        # The literal data is in raw_literals[0]
        if 0 not in raw_literals:
            await self._bad(tag, "APPEND requires literal message data")
            return
        raw = raw_literals[0]

        uid = mb.append(raw, flags, self._userdir.save_mailbox)
        self._userdir.save_mailbox()
        uidvalidity = mb.uidvalidity
        await self._ok(tag, "APPEND completed", code=f"APPENDUID {uidvalidity} {uid}")

    async def _cmd_close(self, tag: str):
        if not self._require_selected():
            await self._bad(tag, "No mailbox selected")
            return
        if not self._read_only:
            self._mailbox.expunge(self._userdir.save_mailbox)
            self._userdir.save_mailbox()
        self._mailbox = None
        self._mailbox_name = None
        self._state = State.AUTHENTICATED
        await self._ok(tag, "CLOSE completed")

    async def _cmd_expunge(self, tag: str):
        if not self._require_selected():
            await self._bad(tag, "No mailbox selected")
            return
        if self._read_only:
            await self._no(tag, "Mailbox is read-only")
            return
        seqs = self._mailbox.expunge(self._userdir.save_mailbox)
        self._userdir.save_mailbox()
        # Report in reverse order (highest seq first) per RFC
        for seq in sorted(seqs, reverse=True):
            await self._send(f"* {seq} EXPUNGE".encode())
        await self._ok(tag, "EXPUNGE completed")

    async def _cmd_search(self, tag: str, args: list[str], uid: bool):
        if not self._require_selected():
            await self._bad(tag, "No mailbox selected")
            return
        msgs = self._mailbox.messages
        # Basic search criteria
        criteria = " ".join(args).upper()
        matched = []
        for msg in msgs:
            if self._matches_search(msg, criteria):
                matched.append(msg.uid if uid else msg.seq)
        result = " ".join(str(n) for n in matched)
        await self._send(f"* SEARCH {result}".encode())
        await self._ok(tag, "SEARCH completed")

    def _matches_search(self, msg: Message, criteria: str) -> bool:
        criteria = criteria.strip()
        if criteria in ("ALL", ""):
            return True
        if criteria == "UNSEEN":
            return r"\Seen" not in msg.flags
        if criteria == "SEEN":
            return r"\Seen" in msg.flags
        if criteria == "FLAGGED":
            return r"\Flagged" in msg.flags
        if criteria == "UNFLAGGED":
            return r"\Flagged" not in msg.flags
        if criteria == "DELETED":
            return r"\Deleted" in msg.flags
        if criteria == "UNDELETED":
            return r"\Deleted" not in msg.flags
        if criteria == "ANSWERED":
            return r"\Answered" in msg.flags
        if criteria == "UNANSWERED":
            return r"\Answered" not in msg.flags
        if criteria == "DRAFT":
            return r"\Draft" in msg.flags
        if criteria == "UNDRAFT":
            return r"\Draft" not in msg.flags
        if criteria == "RECENT":
            return r"\Recent" in msg.flags
        if criteria == "NEW":
            return r"\Recent" in msg.flags and r"\Seen" not in msg.flags
        if criteria == "OLD":
            return r"\Recent" not in msg.flags
        # TEXT / BODY / SUBJECT / FROM / TO searches
        for keyword in ("TEXT", "BODY", "SUBJECT", "FROM", "TO", "CC"):
            m = re.match(rf"^{keyword}\s+\"?([^\"]+)\"?$", criteria, re.I)
            if m:
                needle = m.group(1).lower()
                raw = msg.read().decode("utf-8", errors="replace").lower()
                return needle in raw
        return True  # Unknown criteria: match all

    async def _cmd_fetch(self, tag: str, seqset: str, items_token: str, uid: bool):
        if not self._require_selected():
            await self._bad(tag, "No mailbox selected")
            return
        msgs = self._mailbox.messages
        if not msgs:
            await self._ok(tag, "FETCH completed")
            return

        if uid:
            uids = _parse_seqset(seqset, max(m.uid for m in msgs))
            targets = [m for m in msgs if m.uid in uids]
        else:
            seqs = _parse_seqset(seqset, len(msgs))
            targets = [m for m in msgs if m.seq in seqs]

        # Normalise items: could be bare word or parenthesized list
        items_str = items_token.strip()
        if items_str.startswith("("):
            items_str = items_str[1:-1]
        # Expand macros
        if items_str.upper() == "ALL":
            items_str = "FLAGS INTERNALDATE RFC822.SIZE ENVELOPE"
        elif items_str.upper() == "FAST":
            items_str = "FLAGS INTERNALDATE RFC822.SIZE"
        elif items_str.upper() == "FULL":
            items_str = "FLAGS INTERNALDATE RFC822.SIZE ENVELOPE BODY"

        # Tokenise fetch items (they can include BODY[section]<partial>)
        fetch_items = self._parse_fetch_items(items_str)

        for msg in targets:
            response = await self._build_fetch_response(msg, fetch_items, uid)
            await self._send(response)

        await self._ok(tag, "FETCH completed")

    def _parse_fetch_items(self, items_str: str) -> list[str]:
        """Parse FETCH item list, keeping BODY[...] as single tokens."""
        items: list[str] = []
        i = 0
        s = items_str.strip()
        n = len(s)
        while i < n:
            if s[i] in " \t":
                i += 1
                continue
            # Check for BODY[...] or BODY.PEEK[...]
            if s[i:].upper().startswith("BODY") or s[i:].upper().startswith("RFC822"):
                j = i
                while j < n and s[j] not in " \t":
                    if s[j] == "[":
                        # consume until matching ]
                        j += 1
                        depth = 1
                        while j < n and depth > 0:
                            if s[j] == "[":
                                depth += 1
                            elif s[j] == "]":
                                depth -= 1
                            j += 1
                        # possible <partial>
                        if j < n and s[j] == "<":
                            j += 1
                            while j < n and s[j] != ">":
                                j += 1
                            j += 1
                    else:
                        j += 1
                items.append(s[i:j])
                i = j
            else:
                j = i
                while j < n and s[j] not in " \t":
                    j += 1
                items.append(s[i:j])
                i = j
        return items

    async def _build_fetch_response(
        self, msg: Message, items: list[str], include_uid: bool
    ) -> bytes:
        parts: list[str] = []
        raw_parts: list[tuple[int, bytes]] = []  # (position, literal bytes)

        if include_uid and "UID" not in [i.upper() for i in items]:
            items = list(items) + ["UID"]

        for item in items:
            item_upper = item.upper()

            if item_upper == "FLAGS":
                flags_str = " ".join(msg.flags)
                parts.append(f"FLAGS ({flags_str})")

            elif item_upper == "UID":
                parts.append(f"UID {msg.uid}")

            elif item_upper == "RFC822.SIZE":
                parts.append(f"RFC822.SIZE {msg.size}")

            elif item_upper == "INTERNALDATE":
                parts.append(f'INTERNALDATE "{msg.internaldate()}"')

            elif item_upper == "ENVELOPE":
                raw_msg = email.message_from_bytes(msg.read())
                parts.append(f"ENVELOPE {_build_envelope(raw_msg)}")

            elif item_upper in ("BODY", "BODYSTRUCTURE"):
                raw_msg = email.message_from_bytes(msg.read())
                extended = item_upper == "BODYSTRUCTURE"
                parts.append(f"{item_upper} {_build_bodystructure(raw_msg, extended)}")

            elif item_upper in ("RFC822", "RFC822.HEADER", "RFC822.TEXT"):
                raw_msg_bytes = msg.read()
                raw_email = email.message_from_bytes(raw_msg_bytes)
                if item_upper == "RFC822":
                    content = raw_msg_bytes
                elif item_upper == "RFC822.HEADER":
                    content = _get_body_section(raw_email, "HEADER") or b""
                else:
                    content = _get_body_section(raw_email, "TEXT") or b""
                pos = len(parts)
                parts.append(f"{item_upper} {{{len(content)}}}")
                raw_parts.append((pos, content))
                # Mark \Seen unless PEEK
                if not msg.path.parent.name == "new":
                    self._set_seen(msg)

            else:
                # BODY[section]<partial> or BODY.PEEK[section]<partial>
                m = re.match(
                    r"(BODY(?:\.PEEK)?)\[([^\]]*)\](?:<(\d+)(?:\.(\d+))?>)?",
                    item,
                    re.IGNORECASE,
                )
                if m:
                    fetch_type = m.group(1).upper()
                    section = m.group(2)
                    partial_start = int(m.group(3)) if m.group(3) else None
                    partial_len = int(m.group(4)) if m.group(4) else None

                    raw_msg_bytes = msg.read()
                    raw_email = email.message_from_bytes(raw_msg_bytes)

                    if section == "":
                        content = raw_msg_bytes
                    else:
                        content = _get_body_section(raw_email, section) or b""

                    if partial_start is not None:
                        content = content[partial_start:]
                        if partial_len is not None:
                            content = content[:partial_len]

                    # Build response item label
                    label = f"BODY[{section}]"
                    if partial_start is not None:
                        actual_start = partial_start
                        label += f"<{actual_start}>"

                    pos = len(parts)
                    parts.append(f"{label} {{{len(content)}}}")
                    raw_parts.append((pos, content))

                    # Set \Seen unless PEEK
                    if "PEEK" not in fetch_type:
                        self._set_seen(msg)
                else:
                    parts.append(f"{item} NIL")

        # Build response line
        prefix = f"* {msg.seq} FETCH (".encode()
        suffix = b")"
        body = b" ".join(
            (part.encode() if idx not in dict(raw_parts) else part.encode())
            for idx, part in enumerate(parts)
        )

        # Reconstruct with actual literal bytes
        if raw_parts:
            # Build piece by piece
            result = prefix
            for idx, part in enumerate(parts):
                if idx > 0:
                    result += b" "
                result += part.encode()
                literal = dict(raw_parts).get(idx)
                if literal is not None:
                    result += CRLF + literal
            result += suffix
            return result

        return prefix + " ".join(parts).encode() + suffix

    def _set_seen(self, msg: Message):
        if r"\Seen" not in msg.flags:
            new_flags = list(msg.flags) + [r"\Seen"]
            self._mailbox.update_flags(msg, new_flags, self._userdir.save_mailbox)
            self._userdir.save_mailbox()

    async def _cmd_store(self, tag: str, seqset: str, item: str, value: str, uid: bool):
        if not self._require_selected():
            await self._bad(tag, "No mailbox selected")
            return
        if self._read_only:
            await self._no(tag, "Mailbox is read-only")
            return
        msgs = self._mailbox.messages
        if not msgs:
            await self._ok(tag, "STORE completed")
            return

        if uid:
            uids = _parse_seqset(seqset, max(m.uid for m in msgs))
            targets = [m for m in msgs if m.uid in uids]
        else:
            seqs = _parse_seqset(seqset, len(msgs))
            targets = [m for m in msgs if m.seq in seqs]

        item_upper = item.upper()
        silent = item_upper.endswith(".SILENT")
        new_flags = _parse_flag_list(value)

        for msg in targets:
            old_flags = list(msg.flags)
            if item_upper.startswith("+FLAGS"):
                merged = list(set(old_flags) | set(new_flags))
            elif item_upper.startswith("-FLAGS"):
                merged = [f for f in old_flags if f not in new_flags]
            else:
                merged = new_flags

            self._mailbox.update_flags(msg, merged, self._userdir.save_mailbox)
            self._userdir.save_mailbox()

            if not silent:
                flags_str = " ".join(msg.flags)
                resp = f"* {msg.seq} FETCH (FLAGS ({flags_str})"
                if uid:
                    resp += f" UID {msg.uid}"
                resp += ")"
                await self._send(resp.encode())

        await self._ok(tag, "STORE completed")

    async def _cmd_copy(self, tag: str, seqset: str, dest_folder: str, uid: bool):
        if not self._require_selected():
            await self._bad(tag, "No mailbox selected")
            return
        dest_mb = self._userdir.get_mailbox(dest_folder)
        if dest_mb is None:
            await self._no(tag, f"[TRYCREATE] Mailbox '{dest_folder}' not found")
            return
        msgs = self._mailbox.messages
        if uid:
            uids = _parse_seqset(seqset, max(m.uid for m in msgs) if msgs else 1)
            targets = [m for m in msgs if m.uid in uids]
        else:
            seqs = _parse_seqset(seqset, len(msgs))
            targets = [m for m in msgs if m.seq in seqs]

        src_uids = []
        dst_uids = []
        for msg in targets:
            new_uid = dest_mb.append(msg.read(), msg.flags, self._userdir.save_mailbox)
            self._userdir.save_mailbox()
            src_uids.append(str(msg.uid))
            dst_uids.append(str(new_uid))

        code = f"COPYUID {self._mailbox.uidvalidity} {','.join(src_uids)} {','.join(dst_uids)}"
        await self._ok(tag, "COPY completed", code=code)

    async def _cmd_idle(self, tag: str):
        await self._send_raw(b"+ idling\r\n")
        self._idle = True
        try:
            # Poll mailbox for changes every 30s until DONE
            while True:
                try:
                    line = await asyncio.wait_for(self._reader.readline(), timeout=30)
                except asyncio.TimeoutError:
                    if self._mailbox:
                        self._reload_mailbox()
                        await self._send_mailbox_status()
                    continue
                if not line:
                    break
                if line.strip().upper() == b"DONE":
                    break
        finally:
            self._idle = False
        await self._ok(tag, "IDLE terminated")
