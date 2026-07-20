#!/usr/bin/env python3
"""
Set up Cloudflare Email Routing for penux.uk so that mail to
netanel@penux.uk is forwarded to nsh531@gmail.com.

What it does (all via the Cloudflare API):
  1. Resolves the zone id for penux.uk (and the account id).
  2. Enables Email Routing on the zone.
  3. Adds the required MX + SPF DNS records automatically.
  4. Registers nsh531@gmail.com as a destination address
     (Cloudflare sends a one-time verification email — you must click it).
  5. Creates a routing rule: netanel@penux.uk  ->  nsh531@gmail.com.
  6. (Optional) enables catch-all so *@penux.uk also forwards.

Usage:
    export CF_API_TOKEN="...."          # token with Zone:Read, DNS:Edit,
                                        #   Email Routing:Edit, Account:Read
    python3 setup_cloudflare_email_routing.py

    # optional overrides
    export CF_ZONE="penux.uk"
    export FORWARD_FROM="netanel@penux.uk"
    export FORWARD_TO="nsh531@gmail.com"
    export ENABLE_CATCHALL="1"          # forward ALL @penux.uk too

The token is read from the environment only — nothing is written to disk
or committed.
"""

import os
import sys
import json
import urllib.request
import urllib.error

API = "https://api.cloudflare.com/client/v4"

TOKEN        = os.environ.get("CF_API_TOKEN", "").strip()
ZONE_NAME    = os.environ.get("CF_ZONE", "penux.uk").strip()
FORWARD_FROM = os.environ.get("FORWARD_FROM", "netanel@penux.uk").strip()
FORWARD_TO   = os.environ.get("FORWARD_TO", "nsh531@gmail.com").strip()
ENABLE_CATCHALL = os.environ.get("ENABLE_CATCHALL", "0").strip() in ("1", "true", "yes")

if not TOKEN:
    sys.exit("❌ CF_API_TOKEN is not set. Export it and re-run.")


def req(method, path, body=None):
    url = path if path.startswith("http") else API + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Authorization", f"Bearer {TOKEN}")
    r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        payload = e.read().decode()
        try:
            payload = json.dumps(json.loads(payload), indent=2)
        except Exception:
            pass
        print(f"⚠️  {method} {path} -> HTTP {e.code}\n{payload}")
        try:
            return json.loads(payload)
        except Exception:
            return {"success": False, "errors": [{"message": payload}]}


def ok(resp):
    return isinstance(resp, dict) and resp.get("success")


# ── 1. Resolve zone + account ──────────────────────────────────────────────
print(f"🔎 Looking up zone '{ZONE_NAME}'...")
z = req("GET", f"/zones?name={ZONE_NAME}")
if not ok(z) or not z["result"]:
    sys.exit(f"❌ Could not find zone '{ZONE_NAME}'. Check token permissions / zone name.")
zone = z["result"][0]
zone_id = zone["id"]
account_id = zone["account"]["id"]
print(f"   zone_id={zone_id}  account_id={account_id}")

# ── 2. Enable Email Routing ────────────────────────────────────────────────
print("📧 Enabling Email Routing...")
en = req("POST", f"/zones/{zone_id}/email/routing/enable", {})
if ok(en):
    print("   ✅ Email Routing enabled (or already on).")
else:
    # 'already enabled' comes back as an error; keep going.
    print("   ℹ️  Continuing (may already be enabled).")

# ── 3. Add required MX + SPF DNS records ───────────────────────────────────
print("🧭 Adding required MX + SPF DNS records...")
dns = req("POST", f"/zones/{zone_id}/email/routing/dns", {"name": ZONE_NAME})
if ok(dns):
    print("   ✅ DNS records created.")
    for rec in dns.get("result", []) or []:
        print(f"      {rec.get('type','?'):4} {rec.get('name','')} -> {rec.get('content','')}")
else:
    print("   ℹ️  DNS records may already exist; continuing.")

# ── 4. Register destination address ────────────────────────────────────────
print(f"📮 Registering destination address {FORWARD_TO}...")
dest = req("POST", f"/accounts/{account_id}/email/routing/addresses",
           {"email": FORWARD_TO})
if ok(dest):
    print("   ✅ Destination added — CHECK nsh531@gmail.com FOR A VERIFICATION EMAIL and click it.")
else:
    print("   ℹ️  Destination may already be registered.")

# ── 5. Routing rule: netanel@penux.uk -> nsh531@gmail.com ──────────────────
print(f"🔀 Creating rule: {FORWARD_FROM} -> {FORWARD_TO}...")
rule = req("POST", f"/zones/{zone_id}/email/routing/rules", {
    "name": f"Forward {FORWARD_FROM} to {FORWARD_TO}",
    "enabled": True,
    "matchers": [{"type": "literal", "field": "to", "value": FORWARD_FROM}],
    "actions":  [{"type": "forward", "value": [FORWARD_TO]}],
})
if ok(rule):
    print("   ✅ Forwarding rule created.")
else:
    print("   ⚠️  Rule may already exist (see message above).")

# ── 6. Optional catch-all ──────────────────────────────────────────────────
if ENABLE_CATCHALL:
    print(f"🪣 Enabling catch-all: *@{ZONE_NAME} -> {FORWARD_TO}...")
    ca = req("PUT", f"/zones/{zone_id}/email/routing/rules/catch_all", {
        "name": "Catch-all",
        "enabled": True,
        "matchers": [{"type": "all"}],
        "actions":  [{"type": "forward", "value": [FORWARD_TO]}],
    })
    print("   ✅ Catch-all enabled." if ok(ca) else "   ⚠️  Catch-all not set (see above).")

print("\n" + "=" * 64)
print("✅ DONE.")
print(f"   Mail to {FORWARD_FROM} will forward to {FORWARD_TO}")
print("   once you VERIFY the destination address (click the email link).")
print("   DNS propagation for the new MX records can take a few minutes.")
print("=" * 64)
