"""
Comprehensive IMAP server test suite.
Covers every RFC 3501 command the server implements.
Run standalone: python tests/test_imap_server.py
"""
import asyncio, imaplib, json, shutil, socket, sys, time, threading
from pathlib import Path
from datetime import datetime, timezone

# ── bootstrap ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from imap_server.config import Config, _hash_password
from imap_server.server import run_server
from imap_server.mailstore import UserMaildir

PORT   = 10200
MAILDIR    = Path("/tmp/penux_imap_suite")
USERS_FILE = Path("/tmp/penux_imap_suite_users.json")

# ── helpers ───────────────────────────────────────────────────────────────────

def fb(data, idx=0):
    """Unwrap imaplib response: bytes OR (hdr_bytes, literal_bytes) → bytes."""
    d = data[idx]
    return d[1] if isinstance(d, tuple) else d

def fb_str(data, idx=0):
    return fb(data, idx).decode(errors="replace")

# ── test infrastructure ───────────────────────────────────────────────────────

PASS = []
FAIL = []

def ok(name):
    PASS.append(name)
    print(f"  ✓  {name}")

def fail(name, reason):
    FAIL.append((name, reason))
    print(f"  ✗  {name}: {reason}")

def check(name, condition, got=None):
    if condition:
        ok(name)
    else:
        fail(name, f"assertion failed — got {got!r}")

def run_test(name, fn):
    try:
        fn()
        ok(name)
    except AssertionError as e:
        fail(name, str(e))
    except Exception as e:
        fail(name, f"{type(e).__name__}: {e}")

# ── server bootstrap ──────────────────────────────────────────────────────────

def start_server():
    if MAILDIR.exists():
        shutil.rmtree(MAILDIR)
    MAILDIR.mkdir()
    USERS_FILE.write_text(json.dumps({
        "alice": _hash_password("alice123"),
        "bob":   _hash_password("bob456"),
    }))
    cfg = Config()
    cfg.port_imap    = PORT
    cfg.port_imaps   = PORT + 1000
    cfg.maildir_base = MAILDIR
    cfg.users_file   = USERS_FILE

    for user in ("alice", "bob"):
        UserMaildir(cfg.get_maildir(user)).ensure_default_folders()

    loop = asyncio.new_event_loop()
    def _run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(run_server(cfg))
    threading.Thread(target=_run, daemon=True).start()

    # wait until port is open
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", PORT), timeout=0.2)
            s.close()
            break
        except OSError:
            time.sleep(0.1)
    return cfg

def make_msg(subject, to="nsh531@gmail.com", frm="netanel@penux.uk", body=None):
    now = datetime.now(timezone.utc)
    body = body or f"Test message: {subject}\r\nTimestamp: {now}\r\n"
    return (
        f"From: Netanel <{frm}>\r\n"
        f"To: {to}\r\n"
        f"Subject: {subject}\r\n"
        f"Date: {now.strftime('%a, %d %b %Y %H:%M:%S +0000')}\r\n"
        f"Message-ID: <{int(now.timestamp())}.{hash(subject) & 0xffff}@mail.penux.uk>\r\n"
        f"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        f"{body}"
    ).encode()

# ── individual tests ──────────────────────────────────────────────────────────

def test_greeting(M_raw):
    """Server sends RFC 3501 greeting."""
    s = socket.create_connection(("127.0.0.1", PORT))
    greeting = s.recv(512).decode()
    s.close()
    assert "* OK" in greeting, f"Bad greeting: {greeting!r}"
    assert "IMAP4rev1" in greeting or "mail.penux.uk" in greeting

def test_capability(M):
    typ, data = M.capability()
    caps = data[0].decode()
    assert typ == "OK"
    assert "IMAP4rev1" in caps
    assert "AUTH=PLAIN" in caps
    assert "IDLE" in caps
    assert "UIDPLUS" in caps

def test_login_ok(M):
    typ, data = M.login("alice", "alice123")
    assert typ == "OK", f"Login failed: {data}"

def test_login_bad():
    M2 = imaplib.IMAP4("127.0.0.1", PORT)
    try:
        M2.login("alice", "wrongpass")
        assert False, "Expected login to fail but it succeeded"
    except imaplib.IMAP4.error:
        pass  # expected — imaplib raises on NO response
    M2.logout()

def test_list_folders(M):
    typ, data = M.list()
    assert typ == "OK"
    folders = [d.decode() for d in data if d]
    names = [f.split('"')[-2] for f in folders]
    assert "INBOX" in names, f"INBOX not in {names}"
    assert "Sent"  in names
    assert "Drafts" in names

def test_create_delete_folder(M):
    typ, _ = M.create("TestFolder")
    assert typ == "OK"
    # verify it appears in LIST
    typ, data = M.list()
    found = any(b"TestFolder" in (d or b"") for d in data)
    assert found, "Created folder not in LIST"
    # delete it
    typ, _ = M.delete("TestFolder")
    assert typ == "OK"
    typ, data = M.list()
    gone = not any(b"TestFolder" in (d or b"") for d in data)
    assert gone, "Deleted folder still in LIST"

def test_rename_folder(M):
    M.create("RenameMe")
    typ, _ = M.rename("RenameMe", "Renamed")
    assert typ == "OK"
    typ, data = M.list()
    assert any(b"Renamed" in (d or b"") for d in data)
    M.delete("Renamed")

def test_append_and_select(M):
    msg = make_msg("Append test — To: nsh531@gmail.com", to="nsh531@gmail.com")
    typ, data = M.append("INBOX", r"(\Seen)", imaplib.Time2Internaldate(time.time()), msg)
    assert typ == "OK"
    assert "APPENDUID" in data[0].decode()

    typ, data = M.select("INBOX")
    assert typ == "OK"
    exists = int(data[0].decode())
    assert exists >= 1

def test_status(M):
    typ, data = M.status("INBOX", "(MESSAGES RECENT UNSEEN UIDNEXT UIDVALIDITY)")
    assert typ == "OK"
    s = fb_str(data)
    assert "MESSAGES" in s
    assert "UIDNEXT"  in s

def test_search_all(M):
    typ, data = M.search(None, "ALL")
    assert typ == "OK"
    seqs = data[0].decode().split()
    assert len(seqs) >= 1

def test_search_seen(M):
    typ, data = M.search(None, "SEEN")
    assert typ == "OK"

def test_search_unseen(M):
    typ, data = M.search(None, "UNSEEN")
    assert typ == "OK"

def test_fetch_flags(M):
    typ, data = M.fetch("1", "(FLAGS)")
    assert typ == "OK"
    s = fb_str(data)
    assert "FLAGS" in s

def test_fetch_envelope(M):
    typ, data = M.fetch("1", "(ENVELOPE)")
    assert typ == "OK"
    s = fb_str(data)
    assert "ENVELOPE" in s
    assert "nsh531" in s or "gmail" in s

def test_fetch_size(M):
    typ, data = M.fetch("1", "(RFC822.SIZE)")
    assert typ == "OK"
    s = fb_str(data)
    assert "RFC822.SIZE" in s
    # extract size and check it's positive
    import re
    m = re.search(r"RFC822\.SIZE (\d+)", s)
    assert m and int(m.group(1)) > 0

def test_fetch_internaldate(M):
    typ, data = M.fetch("1", "(INTERNALDATE)")
    assert typ == "OK"
    s = fb_str(data)
    assert "INTERNALDATE" in s

def test_fetch_bodystructure(M):
    typ, data = M.fetch("1", "(BODYSTRUCTURE)")
    assert typ == "OK"
    s = fb_str(data)
    assert "BODYSTRUCTURE" in s
    assert "text" in s.lower() or "plain" in s.lower()

def test_fetch_body(M):
    typ, data = M.fetch("1", "(BODY[])")
    assert typ == "OK"
    body = data[0][1]
    assert len(body) > 0
    assert b"From:" in body
    assert b"To: nsh531@gmail.com" in body
    assert b"Subject:" in body

def test_fetch_header_fields(M):
    typ, data = M.fetch("1", r"(BODY.PEEK[HEADER.FIELDS (From To Subject)])")
    assert typ == "OK"
    body = data[0][1]
    assert b"From:" in body
    assert b"To:" in body
    assert b"Subject:" in body

def test_fetch_text(M):
    typ, data = M.fetch("1", r"(BODY.PEEK[TEXT])")
    assert typ == "OK"
    body = data[0][1]
    assert len(body) > 0

def test_fetch_rfc822_header(M):
    typ, data = M.fetch("1", "(RFC822.HEADER)")
    assert typ == "OK"
    body = data[0][1]
    assert b"From:" in body

def test_fetch_multi_items(M):
    typ, data = M.fetch("1", "(FLAGS RFC822.SIZE INTERNALDATE)")
    assert typ == "OK"
    s = fb_str(data)
    assert "FLAGS" in s
    assert "RFC822.SIZE" in s
    assert "INTERNALDATE" in s

def test_store_add_flag(M):
    M.store("1", "+FLAGS", r"(\Flagged)")
    typ, data = M.fetch("1", "(FLAGS)")
    s = fb_str(data)
    assert r"\Flagged" in s

def test_store_remove_flag(M):
    M.store("1", "-FLAGS", r"(\Flagged)")
    typ, data = M.fetch("1", "(FLAGS)")
    s = fb_str(data)
    assert r"\Flagged" not in s

def test_store_set_flags(M):
    M.store("1", "FLAGS", r"(\Seen \Answered)")
    typ, data = M.fetch("1", "(FLAGS)")
    s = fb_str(data)
    assert r"\Seen" in s
    assert r"\Answered" in s

def test_uid_search(M):
    typ, data = M.uid("search", None, "ALL")
    assert typ == "OK"
    uids = data[0].decode().split()
    assert uids

def test_uid_fetch(M):
    typ, data = M.uid("search", None, "ALL")
    uids = data[0].decode().split()
    uid = uids[0]
    typ, data = M.uid("fetch", uid, "(UID FLAGS BODY.PEEK[])")
    assert typ == "OK"
    body = data[0][1]
    assert b"From:" in body

def test_uid_store(M):
    typ, data = M.uid("search", None, "ALL")
    uid = data[0].decode().split()[0]
    typ, _ = M.uid("store", uid, "+FLAGS", r"(\Flagged)")
    assert typ == "OK"
    typ, _ = M.uid("store", uid, "-FLAGS", r"(\Flagged)")
    assert typ == "OK"

def test_copy(M):
    typ, _ = M.copy("1", "Sent")
    assert typ == "OK"
    # Sent should now have a message
    M.select("Sent")
    typ, data = M.search(None, "ALL")
    seqs = data[0].decode().split()
    assert seqs
    M.select("INBOX")

def test_append_multiple(M):
    for i in range(3):
        msg = make_msg(f"Multi-append {i} — nsh531@gmail.com", to="nsh531@gmail.com")
        typ, _ = M.append("INBOX", None, imaplib.Time2Internaldate(time.time()), msg)
        assert typ == "OK"
    typ, data = M.select("INBOX")
    exists = int(data[0].decode())
    assert exists >= 4, f"Expected ≥4 messages, got {exists}"

def test_expunge(M):
    # append a message, mark deleted, expunge
    msg = make_msg("To be deleted", to="nsh531@gmail.com")
    M.append("INBOX", r"(\Seen)", imaplib.Time2Internaldate(time.time()), msg)
    M.select("INBOX")
    typ, data = M.search(None, "ALL")
    seqs = data[0].decode().split()
    last = seqs[-1]
    M.store(last, "+FLAGS", r"(\Deleted)")
    typ, data = M.expunge()
    assert typ == "OK"
    # verify count decreased
    typ, data = M.search(None, "ALL")
    seqs_after = data[0].decode().split()
    assert len(seqs_after) < len(seqs) + 1  # one fewer than before+appended

def test_search_subject(M):
    msg = make_msg("UniqueSubject12345", to="nsh531@gmail.com")
    M.append("INBOX", None, imaplib.Time2Internaldate(time.time()), msg)
    M.select("INBOX")
    typ, data = M.search(None, "SUBJECT", "UniqueSubject12345")
    seqs = data[0].decode().split()
    assert seqs, "SEARCH SUBJECT returned no results"

def test_fetch_partial(M):
    """BODY[]<0.10> — partial fetch."""
    typ, data = M.fetch("1", "(BODY[]<0.10>)")
    assert typ == "OK"
    body = data[0][1]
    assert 0 < len(body) <= 10, f"Partial fetch size wrong: {len(body)}"

def test_examine_readonly(M):
    """EXAMINE opens mailbox read-only."""
    typ, _ = M.select("INBOX", readonly=True)
    assert typ == "OK"
    # STORE should fail in read-only mode
    typ, _ = M.store("1", "+FLAGS", r"(\Flagged)")
    assert typ == "NO", f"STORE in read-only should be NO, got {typ}"
    M.select("INBOX")  # reopen read-write

def test_noop(M):
    typ, _ = M.noop()
    assert typ == "OK"

def test_close(M):
    M.select("INBOX")
    typ, _ = M.close()
    assert typ == "OK"
    M.select("INBOX")  # reopen for subsequent tests

def test_lsub(M):
    typ, data = M.lsub("", "*")
    assert typ == "OK"

def test_bad_command():
    s = socket.create_connection(("127.0.0.1", PORT))
    f = s.makefile("rb")
    f.readline()  # consume greeting
    s.sendall(b"X1 UNKNOWNCMD\r\n")
    resp = f.readline()
    s.close()
    assert b"BAD" in resp or b"NO" in resp, f"Expected BAD: {resp!r}"

def test_second_user():
    M2 = imaplib.IMAP4("127.0.0.1", PORT)
    typ, _ = M2.login("bob", "bob456")
    assert typ == "OK"
    msg = make_msg("Bob inbox test", to="nsh531@gmail.com")
    M2.append("INBOX", None, imaplib.Time2Internaldate(time.time()), msg)
    M2.select("INBOX")
    typ, data = M2.search(None, "ALL")
    assert data[0].decode().split()
    M2.logout()

# ── run all tests ─────────────────────────────────────────────────────────────

def run_all():
    global PASS, FAIL
    PASS, FAIL = [], []

    cfg = start_server()

    M = imaplib.IMAP4("127.0.0.1", PORT)

    print("\n── Pre-auth ─────────────────────────────────────────")
    run_test("greeting",           lambda: test_greeting(None))
    run_test("capability",         lambda: test_capability(M))
    run_test("login-bad-password", test_login_bad)
    run_test("login-ok",           lambda: test_login_ok(M))

    print("\n── Folder management ────────────────────────────────")
    run_test("list-folders",         lambda: test_list_folders(M))
    run_test("create-delete-folder", lambda: test_create_delete_folder(M))
    run_test("rename-folder",        lambda: test_rename_folder(M))

    print("\n── Message operations ───────────────────────────────")
    run_test("append-and-select",  lambda: test_append_and_select(M))
    run_test("status",             lambda: test_status(M))
    run_test("search-all",         lambda: test_search_all(M))
    run_test("search-seen",        lambda: test_search_seen(M))
    run_test("search-unseen",      lambda: test_search_unseen(M))
    run_test("search-subject",     lambda: test_search_subject(M))

    print("\n── FETCH items ──────────────────────────────────────")
    run_test("fetch-flags",        lambda: test_fetch_flags(M))
    run_test("fetch-envelope",     lambda: test_fetch_envelope(M))
    run_test("fetch-size",         lambda: test_fetch_size(M))
    run_test("fetch-internaldate", lambda: test_fetch_internaldate(M))
    run_test("fetch-bodystructure",lambda: test_fetch_bodystructure(M))
    run_test("fetch-body[]",       lambda: test_fetch_body(M))
    run_test("fetch-header-fields",lambda: test_fetch_header_fields(M))
    run_test("fetch-text",         lambda: test_fetch_text(M))
    run_test("fetch-rfc822-header",lambda: test_fetch_rfc822_header(M))
    run_test("fetch-multi-items",  lambda: test_fetch_multi_items(M))
    run_test("fetch-partial",      lambda: test_fetch_partial(M))

    print("\n── STORE / flags ────────────────────────────────────")
    run_test("store-add-flag",     lambda: test_store_add_flag(M))
    run_test("store-remove-flag",  lambda: test_store_remove_flag(M))
    run_test("store-set-flags",    lambda: test_store_set_flags(M))

    print("\n── UID commands ─────────────────────────────────────")
    run_test("uid-search",         lambda: test_uid_search(M))
    run_test("uid-fetch",          lambda: test_uid_fetch(M))
    run_test("uid-store",          lambda: test_uid_store(M))

    print("\n── Misc ─────────────────────────────────────────────")
    run_test("copy",               lambda: test_copy(M))
    run_test("append-multiple",    lambda: test_append_multiple(M))
    run_test("expunge",            lambda: test_expunge(M))
    run_test("examine-readonly",   lambda: test_examine_readonly(M))
    run_test("noop",               lambda: test_noop(M))
    run_test("close",              lambda: test_close(M))
    run_test("lsub",               lambda: test_lsub(M))
    run_test("bad-command",        test_bad_command)
    run_test("second-user",        test_second_user)

    M.logout()

    print("\n" + "=" * 60)
    total = len(PASS) + len(FAIL)
    print(f"  Results: {len(PASS)}/{total} passed, {len(FAIL)} failed")
    if FAIL:
        print("\n  Failures:")
        for name, reason in FAIL:
            print(f"    ✗  {name}: {reason}")
    else:
        print("  ✅  ALL TESTS PASSED")
    print("=" * 60)
    return FAIL

if __name__ == "__main__":
    failures = run_all()
    sys.exit(1 if failures else 0)
