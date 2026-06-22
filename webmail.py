#!/usr/bin/env python3
"""
PenuX Webmail — browser-based IMAP client for mail.penux.uk.
Pure Python stdlib — zero external dependencies.

Run:  python3 webmail.py
Env:  WEB_PORT=80  IMAP_HOST=127.0.0.1  IMAP_PORT=143
"""
from __future__ import annotations

import email as _email
import email.header
import email.policy
import email.utils
import html
import http.server
import imaplib
import os
import re
import secrets
import threading
import time
import urllib.parse
from typing import Optional

# ── config ────────────────────────────────────────────────────────────────────
IMAP_HOST   = os.environ.get("IMAP_HOST",       "127.0.0.1")
IMAP_PORT   = int(os.environ.get("IMAP_PORT",   "143"))
WEB_HOST    = os.environ.get("WEB_HOST",         "0.0.0.0")
WEB_PORT    = int(os.environ.get("WEB_PORT",    "80"))
SERVER_NAME = os.environ.get("IMAP_SERVERNAME",  "mail.penux.uk")
TLS_CERT    = os.environ.get("IMAP_TLS_CERT",   "/etc/ssl/penux/fullchain.pem")
TLS_KEY     = os.environ.get("IMAP_TLS_KEY",    "/etc/ssl/penux/privkey.pem")

# ── sessions ──────────────────────────────────────────────────────────────────
_sessions: dict[str, dict] = {}
_sess_lock = threading.Lock()

def _sess_new(user: str, pw: str) -> str:
    tok = secrets.token_hex(32)
    with _sess_lock:
        _sessions[tok] = {"user": user, "pw": pw, "exp": time.time() + 86400}
    return tok

def _sess_get(tok: str) -> Optional[dict]:
    with _sess_lock:
        s = _sessions.get(tok)
        if s:
            if s["exp"] > time.time():
                return s
            del _sessions[tok]
    return None

def _sess_del(tok: str):
    with _sess_lock:
        _sessions.pop(tok, None)

# ── IMAP helpers ──────────────────────────────────────────────────────────────
def _imap_open(user: str, pw: str) -> imaplib.IMAP4:
    M = imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
    M.login(user, pw)
    return M

def _decode_header(val: Optional[str]) -> str:
    if not val:
        return ""
    parts = []
    for raw, enc in _email.header.decode_header(val):
        if isinstance(raw, bytes):
            try:
                parts.append(raw.decode(enc or "utf-8", errors="replace"))
            except (LookupError, UnicodeDecodeError):
                parts.append(raw.decode("utf-8", errors="replace"))
        else:
            parts.append(raw)
    return " ".join(parts)

def _short_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        import datetime
        t = email.utils.parsedate_to_datetime(date_str)
        now = datetime.datetime.now(datetime.timezone.utc)
        if t.date() == now.date():
            return t.strftime("%H:%M")
        if t.year == now.year:
            return t.strftime("%b %d")
        return t.strftime("%Y-%m-%d")
    except Exception:
        return date_str[:10]

def _list_folders(M: imaplib.IMAP4) -> list[str]:
    typ, data = M.list()
    folders: list[str] = []
    for d in (data or []):
        if not d:
            continue
        line = d.decode(errors="replace")
        m = re.search(r'"([^"]+)"\s*$', line) or re.search(r'(\S+)\s*$', line)
        if m:
            name = m.group(1).strip('"')
            if name not in folders:
                folders.append(name)
    return folders

def _list_msgs(M: imaplib.IMAP4, folder: str, page: int = 1, per: int = 25) -> tuple[list[dict], int]:
    try:
        rv, _ = M.select(folder, readonly=True)
        if rv != "OK":
            return [], 0
    except Exception:
        return [], 0

    typ, data = M.uid("search", None, "ALL")
    if typ != "OK" or not data[0]:
        return [], 0

    all_uids = data[0].decode().split()
    total = len(all_uids)
    page_uids = list(reversed(all_uids))[(page - 1) * per: page * per]
    if not page_uids:
        return [], total

    msgs: list[dict] = []
    for uid in page_uids:
        try:
            typ2, fdata = M.uid("fetch", uid, r"(FLAGS BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
        except Exception:
            continue
        if typ2 != "OK":
            continue
        for item in fdata:
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            info = item[0].decode(errors="replace") if isinstance(item[0], bytes) else ""
            hdr_bytes = item[1] if isinstance(item[1], bytes) else b""
            flags_m = re.search(r"FLAGS \(([^)]*)\)", info)
            flags = flags_m.group(1) if flags_m else ""
            msg = _email.message_from_bytes(hdr_bytes)
            msgs.append({
                "uid": uid,
                "seen": r"\Seen" in flags,
                "flagged": r"\Flagged" in flags,
                "from": _decode_header(msg.get("From", "")),
                "subject": _decode_header(msg.get("Subject", "(no subject)")),
                "date": msg.get("Date", ""),
            })
    return msgs, total

def _get_msg(M: imaplib.IMAP4, folder: str, uid: str) -> Optional[dict]:
    try:
        M.select(folder)
    except Exception:
        return None
    typ, data = M.uid("fetch", uid, r"(FLAGS BODY[])")
    if typ != "OK":
        return None

    raw_bytes: Optional[bytes] = None
    for item in data:
        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], bytes):
            raw_bytes = item[1]
            break
    if not raw_bytes:
        return None

    msg = _email.message_from_bytes(raw_bytes, policy=_email.policy.compat32)
    M.uid("store", uid, "+FLAGS", r"(\Seen)")

    body_text = ""
    body_html = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain" and not body_text:
                payload = part.get_payload(decode=True)
                if payload:
                    body_text = payload.decode(errors="replace")
            elif ct == "text/html" and not body_html:
                payload = part.get_payload(decode=True)
                if payload:
                    body_html = payload.decode(errors="replace")
    else:
        ct = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        if payload:
            if ct == "text/html":
                body_html = payload.decode(errors="replace")
            else:
                body_text = payload.decode(errors="replace")

    return {
        "from":      _decode_header(msg.get("From", "")),
        "to":        _decode_header(msg.get("To", "")),
        "cc":        _decode_header(msg.get("Cc", "")),
        "subject":   _decode_header(msg.get("Subject", "(no subject)")),
        "date":      msg.get("Date", ""),
        "body_text": body_text,
        "body_html": body_html,
    }

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#f0f2f5;color:#202124;font-size:14px}
a{color:#1a73e8;text-decoration:none}
a:hover{text-decoration:underline}
/* layout */
.layout{display:flex;min-height:100vh}
.sidebar{width:200px;background:#fff;border-right:1px solid #e0e0e0;
         padding:0;flex-shrink:0;display:flex;flex-direction:column}
.sidebar-logo{padding:18px 20px 12px;font-size:17px;font-weight:700;
              color:#1a73e8;border-bottom:1px solid #f0f0f0;user-select:none}
.sidebar-logo span{font-weight:300;color:#666;font-size:13px;display:block;margin-top:2px}
.sidebar ul{list-style:none;padding:8px 0;flex:1}
.sidebar li a{display:flex;align-items:center;gap:10px;padding:9px 20px;
              color:#444;border-radius:0 24px 24px 0;margin-right:8px;
              transition:background .1s}
.sidebar li a:hover,.sidebar li a.active{background:#e8f0fe;color:#1a73e8;
                                          text-decoration:none;font-weight:500}
.sidebar li a .ico{font-size:15px;width:20px;text-align:center}
.sidebar-footer{padding:12px 20px;font-size:12px;color:#888;border-top:1px solid #f0f0f0}
/* main */
.main{flex:1;padding:20px 24px;min-width:0}
.topbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.topbar h1{font-size:18px;font-weight:500}
.topbar .right{display:flex;gap:12px;align-items:center;font-size:13px;color:#666}
/* card */
.card{background:#fff;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
/* message list */
.msg-list{list-style:none}
.msg-item{display:grid;grid-template-columns:180px 1fr 90px;gap:8px;
          padding:11px 16px;border-bottom:1px solid #f5f5f5;
          cursor:pointer;transition:background .1s;align-items:center}
.msg-item:hover{background:#f8f9fa}
.msg-item:last-child{border-bottom:none}
.msg-item.unread{font-weight:600}
.msg-item.unread .msg-subj{color:#202124}
.msg-from{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px}
.msg-subj{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#5f6368;font-size:13px}
.msg-item.unread .msg-subj{color:#202124}
.msg-date{text-align:right;color:#999;font-size:12px}
.msg-dot{width:8px;height:8px;border-radius:50%;background:#1a73e8;
         display:inline-block;margin-right:6px;visibility:hidden}
.msg-item.unread .msg-dot{visibility:visible}
/* pager */
.pager{display:flex;gap:8px;padding:10px 16px;justify-content:flex-end;
       border-top:1px solid #f0f0f0;font-size:13px;color:#666;align-items:center}
.pager a{color:#1a73e8}
/* message view */
.msg-hdr{padding:20px 24px;border-bottom:1px solid #e8eaed}
.msg-hdr .subject{font-size:20px;font-weight:500;margin-bottom:12px;line-height:1.3}
.msg-meta{font-size:13px;color:#5f6368;line-height:1.8}
.msg-meta b{color:#202124;font-weight:500}
.msg-body{padding:20px 24px;line-height:1.7;font-size:14px}
.msg-body-text{white-space:pre-wrap;word-wrap:break-word;font-family:inherit}
.msg-body iframe{width:100%;border:none;min-height:400px;display:block}
.actions{display:flex;gap:8px;margin-top:14px}
/* buttons */
.btn{display:inline-flex;align-items:center;gap:6px;padding:7px 16px;
     border-radius:4px;font-size:13px;font-weight:500;border:none;cursor:pointer;transition:background .15s}
.btn-back{background:#f1f3f4;color:#3c4043}
.btn-back:hover{background:#e8eaed;text-decoration:none}
.btn-del{background:#fce8e6;color:#c5221f}
.btn-del:hover{background:#f5c2be}
.btn-flag{background:#fef3e2;color:#b06000}
/* empty state */
.empty{padding:40px;text-align:center;color:#999}
/* login */
.login-bg{display:flex;align-items:center;justify-content:center;min-height:100vh;
          background:linear-gradient(135deg,#e8f0fe 0%,#f0f7ff 100%)}
.login-card{background:#fff;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.12);
            padding:40px;width:380px}
.login-logo{font-size:28px;font-weight:700;color:#1a73e8;margin-bottom:4px}
.login-logo span{font-weight:300;color:#999;font-size:14px}
.login-sub{color:#888;font-size:13px;margin-bottom:28px}
.form-group{margin-bottom:16px}
.form-group label{display:block;margin-bottom:6px;font-weight:500;color:#3c4043;font-size:13px}
.form-group input{width:100%;padding:10px 12px;border:1.5px solid #dfe1e5;border-radius:6px;
                  font-size:14px;outline:none;transition:border-color .15s}
.form-group input:focus{border-color:#1a73e8;box-shadow:0 0 0 3px rgba(26,115,232,.15)}
.btn-primary{width:100%;padding:11px;background:#1a73e8;color:#fff;border-radius:6px;
             font-size:15px;font-weight:500;border:none;cursor:pointer;transition:background .15s}
.btn-primary:hover{background:#1765cc}
.error-msg{color:#c5221f;font-size:13px;margin-bottom:14px;background:#fce8e6;
           padding:9px 12px;border-radius:6px;border-left:3px solid #c5221f}
@media(max-width:640px){
  .sidebar{width:160px}
  .msg-item{grid-template-columns:120px 1fr 70px}
}
"""

# ── HTML helpers ──────────────────────────────────────────────────────────────
def _page(body: str, title: str = "PenuX Mail") -> str:
    return (
        f'<!DOCTYPE html><html lang="en"><head>'
        f'<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)}</title>'
        f'<style>{_CSS}</style></head><body>{body}</body></html>'
    )

def _sidebar(user: str, folders: list[str], active: str) -> str:
    _ICONS = {
        "INBOX": "📥", "Sent": "📤", "Drafts": "📝",
        "Trash": "🗑️", "Junk": "🚫",
    }
    items = ""
    for f in folders:
        icon = _ICONS.get(f, "📁")
        cls = "active" if f == active else ""
        href = f"/inbox?folder={urllib.parse.quote(f)}"
        items += (
            f'<li><a href="{href}" class="{cls}">'
            f'<span class="ico">{icon}</span>{html.escape(f)}</a></li>'
        )
    return (
        f'<div class="sidebar">'
        f'<div class="sidebar-logo">📬 PenuX<span>{html.escape(SERVER_NAME)}</span></div>'
        f'<ul>{items}</ul>'
        f'<div class="sidebar-footer">{html.escape(user)}<br>'
        f'<a href="/logout">Sign out</a></div>'
        f'</div>'
    )

def _login_page(error: str = "") -> str:
    err = f'<div class="error-msg">{html.escape(error)}</div>' if error else ""
    return _page(
        f'<div class="login-bg"><div class="login-card">'
        f'<div class="login-logo">📬 PenuX<span> Mail</span></div>'
        f'<div class="login-sub">{html.escape(SERVER_NAME)}</div>'
        f'{err}'
        f'<form method="post" action="/login">'
        f'<div class="form-group"><label>Username</label>'
        f'<input name="username" type="text" autofocus autocomplete="username" spellcheck="false"></div>'
        f'<div class="form-group"><label>Password</label>'
        f'<input name="password" type="password" autocomplete="current-password"></div>'
        f'<button type="submit" class="btn-primary">Sign in</button>'
        f'</form></div></div>',
        "Sign in — PenuX Mail",
    )

def _inbox_page(
    user: str, folders: list[str], msgs: list[dict],
    folder: str, page: int, total: int, per: int = 25,
) -> str:
    rows = ""
    for m in msgs:
        uid  = html.escape(m["uid"])
        frm  = html.escape(m["from"][:50] or "(unknown)")
        subj = html.escape(m["subject"][:100]) or "(no subject)"
        date = html.escape(_short_date(m["date"]))
        cls  = "" if m["seen"] else " unread"
        url  = f'/message?folder={urllib.parse.quote(folder)}&uid={uid}'
        rows += (
            f'<li class="msg-item{cls}" onclick="location.href=\'{url}\'">'
            f'<span class="msg-from"><span class="msg-dot"></span>{frm}</span>'
            f'<span class="msg-subj">{subj}</span>'
            f'<span class="msg-date">{date}</span>'
            f'</li>'
        )

    pages = (total + per - 1) // per
    pager = ""
    if pages > 1:
        qf = urllib.parse.quote(folder)
        pager = '<div class="pager">'
        if page > 1:
            pager += f'<a href="/inbox?folder={qf}&page={page-1}">← Prev</a>'
        pager += f'<span>Page {page} of {pages}</span>'
        if page < pages:
            pager += f'<a href="/inbox?folder={qf}&page={page+1}">Next →</a>'
        pager += '</div>'

    count_label = f"{total} message{'s' if total != 1 else ''}"
    empty_row = '<li class="empty">No messages in this folder</li>'
    body = (
        f'<div class="layout">{_sidebar(user, folders, folder)}'
        f'<div class="main">'
        f'<div class="topbar"><h1>{html.escape(folder)}</h1>'
        f'<div class="right">{count_label}'
        f' · <a href="/inbox?folder={urllib.parse.quote(folder)}">Refresh</a></div></div>'
        f'<div class="card"><ul class="msg-list">'
        f'{rows or empty_row}'
        f'</ul>{pager}</div>'
        f'</div></div>'
    )
    return _page(body, f"{folder} — PenuX Mail")

def _message_page(user: str, folders: list[str], msg: dict, folder: str, uid: str) -> str:
    qf = urllib.parse.quote(folder)

    if msg["body_html"]:
        import base64
        b64 = base64.b64encode(msg["body_html"].encode()).decode()
        body_content = (
            f'<iframe src="data:text/html;base64,{b64}" '
            f'sandbox="allow-same-origin" '
            f'onload="this.style.height=this.contentDocument.documentElement.scrollHeight+\'px\'"></iframe>'
        )
    else:
        body_content = (
            f'<div class="msg-body-text">{html.escape(msg["body_text"] or "(empty)")}</div>'
        )

    cc_row = (
        f'<br><b>Cc:</b> {html.escape(msg["cc"])}'
        if msg["cc"] else ""
    )

    body = (
        f'<div class="layout">{_sidebar(user, folders, folder)}'
        f'<div class="main">'
        f'<div class="topbar">'
        f'<a href="/inbox?folder={qf}" class="btn btn-back">← Back</a>'
        f'<div class="right"><a href="/logout">Sign out</a></div></div>'
        f'<div class="card">'
        f'<div class="msg-hdr">'
        f'<div class="subject">{html.escape(msg["subject"])}</div>'
        f'<div class="msg-meta">'
        f'<b>From:</b> {html.escape(msg["from"])}<br>'
        f'<b>To:</b> {html.escape(msg["to"])}'
        f'{cc_row}<br>'
        f'<b>Date:</b> {html.escape(msg["date"])}'
        f'</div>'
        f'<div class="actions">'
        f'<form method="post" action="/delete" onsubmit="return confirm(\'Delete this message?\')">'
        f'<input type="hidden" name="folder" value="{html.escape(folder)}">'
        f'<input type="hidden" name="uid" value="{html.escape(uid)}">'
        f'<button type="submit" class="btn btn-del">🗑 Delete</button>'
        f'</form>'
        f'</div>'
        f'</div>'
        f'<div class="msg-body">{body_content}</div>'
        f'</div>'
        f'</div></div>'
    )
    return _page(body, f"{msg['subject']} — PenuX Mail")

# ── HTTP request handler ──────────────────────────────────────────────────────
class _Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def _cookie(self, name: str) -> Optional[str]:
        for part in self.headers.get("Cookie", "").split(";"):
            part = part.strip()
            if part.startswith(f"{name}="):
                return part[len(name) + 1:]
        return None

    def _send(self, status: int, body: str, ctype: str = "text/html; charset=utf-8",
              extra_headers: dict | None = None):
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(data))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, url: str, extra: dict | None = None):
        self.send_response(302)
        self.send_header("Location", url)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _body(self) -> dict[str, str]:
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n).decode(errors="replace")
        return {k: urllib.parse.unquote_plus(v)
                for part in raw.split("&") if "=" in part
                for k, v in [part.split("=", 1)]}

    def _session(self) -> Optional[dict]:
        tok = self._cookie("session")
        return _sess_get(tok) if tok else None

    # ── GET ───────────────────────────────────────────────────────────────────
    def do_GET(self):
        p = urllib.parse.urlparse(self.path)
        path = p.path.rstrip("/") or "/"
        qs = dict(urllib.parse.parse_qsl(p.query))

        if path in ("/", ""):
            self._redirect("/inbox" if self._session() else "/login")

        elif path == "/login":
            if self._session():
                self._redirect("/inbox")
            else:
                self._send(200, _login_page())

        elif path == "/logout":
            tok = self._cookie("session")
            if tok:
                _sess_del(tok)
            self._redirect("/login", {"Set-Cookie": "session=; Path=/; HttpOnly; Max-Age=0"})

        elif path == "/inbox":
            sess = self._session()
            if not sess:
                self._redirect("/login")
                return
            folder = qs.get("folder", "INBOX")
            page   = max(1, int(qs.get("page", "1")))
            try:
                M = _imap_open(sess["user"], sess["pw"])
                folders = _list_folders(M)
                msgs, total = _list_msgs(M, folder, page)
                M.logout()
            except Exception as e:
                self._send(200, _login_page(f"Session error — please log in again ({e})"))
                return
            self._send(200, _inbox_page(sess["user"], folders, msgs, folder, page, total))

        elif path == "/message":
            sess = self._session()
            if not sess:
                self._redirect("/login")
                return
            folder = qs.get("folder", "INBOX")
            uid    = qs.get("uid", "")
            if not uid:
                self._redirect("/inbox")
                return
            try:
                M = _imap_open(sess["user"], sess["pw"])
                folders = _list_folders(M)
                msg = _get_msg(M, folder, uid)
                M.logout()
            except Exception as e:
                self._send(200, _login_page(f"Error loading message: {e}"))
                return
            if msg is None:
                self._redirect(f"/inbox?folder={urllib.parse.quote(folder)}")
                return
            self._send(200, _message_page(sess["user"], folders, msg, folder, uid))

        else:
            self._send(404, "<h1>404 Not Found</h1>")

    # ── POST ──────────────────────────────────────────────────────────────────
    def do_POST(self):
        p = urllib.parse.urlparse(self.path)
        path = p.path.rstrip("/")

        if path == "/login":
            body = self._body()
            user = body.get("username", "").strip()
            pw   = body.get("password", "")
            if not user or not pw:
                self._send(200, _login_page("Username and password are required"))
                return
            try:
                M = _imap_open(user, pw)
                M.logout()
            except imaplib.IMAP4.error:
                self._send(200, _login_page("Incorrect username or password"))
                return
            except Exception as e:
                self._send(200, _login_page(f"Cannot connect to mail server: {e}"))
                return
            tok = _sess_new(user, pw)
            self._redirect("/inbox", {"Set-Cookie": f"session={tok}; Path=/; HttpOnly; SameSite=Lax"})

        elif path == "/delete":
            sess = self._session()
            if not sess:
                self._redirect("/login")
                return
            body   = self._body()
            folder = body.get("folder", "INBOX")
            uid    = body.get("uid", "")
            if uid:
                try:
                    M = _imap_open(sess["user"], sess["pw"])
                    M.select(folder)
                    M.uid("store", uid, "+FLAGS", r"(\Deleted)")
                    M.expunge()
                    M.logout()
                except Exception:
                    pass
            self._redirect(f"/inbox?folder={urllib.parse.quote(folder)}")

        else:
            self._send(404, "<h1>404 Not Found</h1>")


# ── server bootstrap ──────────────────────────────────────────────────────────
def run():
    import ssl, os.path

    server = http.server.ThreadingHTTPServer((WEB_HOST, WEB_PORT), _Handler)

    use_tls = os.path.exists(TLS_CERT) and os.path.exists(TLS_KEY)
    if use_tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(TLS_CERT, TLS_KEY)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        proto = "https"
    else:
        proto = "http"

    print(f"PenuX Webmail  →  {proto}://{WEB_HOST}:{WEB_PORT}")
    print(f"IMAP backend   →  {IMAP_HOST}:{IMAP_PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
