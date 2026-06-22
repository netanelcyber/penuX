#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PenuX — Oracle Cloud one-command setup
#
# Run this in OCI Cloud Shell (already authenticated, no config needed).
# It generates an API key, uploads it to OCI, sets all GitHub Secrets,
# and triggers the oracle_provision.yml workflow automatically.
#
# USAGE (in OCI Cloud Shell):
#   curl -fsSL https://raw.githubusercontent.com/netanelcyber/penuX/claude/imap-server-penux-1op6fl/deploy/oci_setup.sh \
#     | GH_PAT=ghp_xxxx CF_TOKEN=xxxx IMAP_PASS=xxxx bash
#
# Required env vars:
#   GH_PAT     — GitHub Fine-grained PAT with Actions + Secrets write permission
#   CF_TOKEN   — Cloudflare API token (Zone:Edit on penux.uk)
#   IMAP_PASS  — Desired password for the netanel mailbox
#
# Optional:
#   IMAP_USER  — Mail username (default: netanel)
#   GH_PAT_DEPLOY — Second PAT to register SSH deploy key (can be same as GH_PAT)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

GH_PAT="${GH_PAT:-}"
CF_TOKEN="${CF_TOKEN:-}"
IMAP_PASS="${IMAP_PASS:-}"
IMAP_USER="${IMAP_USER:-netanel}"
GH_PAT_DEPLOY="${GH_PAT_DEPLOY:-$GH_PAT}"
GH_REPO="netanelcyber/penuX"
GH_BRANCH="claude/imap-server-penux-1op6fl"
KEY_FILE="${HOME}/.oci/github-actions-penux"

# ── Validate inputs ────────────────────────────────────────────────────────
missing=()
[ -z "$GH_PAT"    ] && missing+=("GH_PAT")
[ -z "$CF_TOKEN"  ] && missing+=("CF_TOKEN")
[ -z "$IMAP_PASS" ] && missing+=("IMAP_PASS")
if [ ${#missing[@]} -gt 0 ]; then
  echo "ERROR: Missing required env vars: ${missing[*]}"
  echo "Usage: GH_PAT=xxx CF_TOKEN=xxx IMAP_PASS=xxx bash deploy/oci_setup.sh"
  exit 1
fi

echo ""
echo "┌─────────────────────────────────────────────────┐"
echo "│  PenuX — Oracle Cloud Automated Setup           │"
echo "└─────────────────────────────────────────────────┘"
echo ""

# ── OCI CLI ───────────────────────────────────────────────────────────────
if ! command -v oci &>/dev/null; then
  echo "Installing OCI CLI..."
  pip3 install --quiet oci-cli
fi

# ── Read OCI config (pre-populated in Cloud Shell) ────────────────────────
OCI_CONFIG="${HOME}/.oci/config"
if [ ! -f "$OCI_CONFIG" ]; then
  echo "OCI config not found. Running interactive setup..."
  oci setup config
fi

REGION=$(      grep -m1 '^region'  "$OCI_CONFIG" | cut -d= -f2 | tr -d ' \n\r')
USER_OCID=$(   grep -m1 '^user'    "$OCI_CONFIG" | cut -d= -f2 | tr -d ' \n\r')
TENANCY_OCID=$(grep -m1 '^tenancy' "$OCI_CONFIG" | cut -d= -f2 | tr -d ' \n\r')

if [ -z "$USER_OCID" ] || [ -z "$TENANCY_OCID" ]; then
  echo "ERROR: Cannot read user/tenancy from $OCI_CONFIG"
  echo "Run 'oci setup config' first."
  exit 1
fi

echo "  Region:  ${REGION}"
echo "  User:    ${USER_OCID}"
echo "  Tenancy: ${TENANCY_OCID}"

# ── Generate dedicated API keypair for GitHub Actions ─────────────────────
echo ""
echo "Generating API key for GitHub Actions..."
if [ ! -f "${KEY_FILE}" ]; then
  openssl genrsa -out "${KEY_FILE}" 2048 2>/dev/null
  chmod 600 "${KEY_FILE}"
  openssl rsa -pubout -in "${KEY_FILE}" -out "${KEY_FILE}.pub" 2>/dev/null
  echo "  Created: ${KEY_FILE}"
else
  echo "  Existing key found: ${KEY_FILE}"
fi

# ── Upload public key to OCI (idempotent) ─────────────────────────────────
echo ""
echo "Uploading API key to Oracle Cloud..."
UPLOAD_OUT=$(oci iam user api-key upload \
  --user-id "${USER_OCID}" \
  --key-file "${KEY_FILE}.pub" 2>&1) && UPLOAD_OK=true || UPLOAD_OK=false

if $UPLOAD_OK && echo "$UPLOAD_OUT" | grep -q '"fingerprint"'; then
  FINGERPRINT=$(echo "$UPLOAD_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['fingerprint'])")
  echo "  Uploaded. Fingerprint: ${FINGERPRINT}"
else
  # Key already uploaded — derive fingerprint locally
  FINGERPRINT=$(openssl rsa -pubout -in "${KEY_FILE}" -outform DER 2>/dev/null \
    | openssl dgst -md5 -c 2>/dev/null | awk '{print $2}')
  echo "  Key already exists.  Fingerprint: ${FINGERPRINT}"
fi

# ── Push all secrets to GitHub ────────────────────────────────────────────
echo ""
echo "Setting GitHub Secrets..."
pip3 install --quiet PyNaCl requests 2>/dev/null

GH_PAT="$GH_PAT" \
GH_REPO="$GH_REPO" \
USER_OCID="$USER_OCID" \
TENANCY_OCID="$TENANCY_OCID" \
FINGERPRINT="$FINGERPRINT" \
REGION="$REGION" \
KEY_FILE="$KEY_FILE" \
CF_TOKEN="$CF_TOKEN" \
python3 << 'PYEOF'
import base64, os
import requests
from nacl import encoding, public

token        = os.environ["GH_PAT"]
repo         = os.environ["GH_REPO"]
user_ocid    = os.environ["USER_OCID"]
tenancy_ocid = os.environ["TENANCY_OCID"]
fingerprint  = os.environ["FINGERPRINT"]
region       = os.environ["REGION"]
cf_token     = os.environ["CF_TOKEN"]

with open(os.environ["KEY_FILE"]) as f:
    private_key = f.read()

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

r = requests.get(
    f"https://api.github.com/repos/{repo}/actions/secrets/public-key",
    headers=headers, timeout=15)
r.raise_for_status()
pk_data = r.json()
key_id  = pk_data["key_id"]
pub_key = public.PublicKey(pk_data["key"].encode(), encoding.Base64Encoder)

def encrypt(val):
    sealed = public.SealedBox(pub_key).encrypt(val.encode())
    return base64.b64encode(sealed).decode()

secrets = {
    "OCI_USER_OCID":    user_ocid,
    "OCI_TENANCY_OCID": tenancy_ocid,
    "OCI_FINGERPRINT":  fingerprint,
    "OCI_PRIVATE_KEY":  private_key,
    "OCI_REGION":       region,
    "CF_TOKEN":         cf_token,
}

ok = True
for name, value in secrets.items():
    payload = {"encrypted_value": encrypt(value), "key_id": key_id}
    r = requests.put(
        f"https://api.github.com/repos/{repo}/actions/secrets/{name}",
        headers=headers, json=payload, timeout=15)
    if r.status_code in (201, 204):
        print(f"  ✓  {name}")
    else:
        print(f"  ✗  {name}  ({r.status_code}: {r.text[:80]})")
        ok = False

if not ok:
    import sys; sys.exit(1)
PYEOF

# ── Trigger oracle_provision.yml ──────────────────────────────────────────
echo ""
echo "Triggering oracle_provision.yml workflow..."

DISPATCH_BODY=$(python3 -c "
import json, os
print(json.dumps({
    'ref': '${GH_BRANCH}',
    'inputs': {
        'imap_pass': os.environ['IMAP_PASS'],
        'imap_user': os.environ['IMAP_USER'],
        'gh_pat':    os.environ['GH_PAT_DEPLOY'],
    }
}))")

HTTP=$(curl -sS -o /tmp/dispatch_resp.json -w "%{http_code}" \
  -X POST \
  -H "Authorization: Bearer ${GH_PAT}" \
  -H "Accept: application/vnd.github+json" \
  -H "Content-Type: application/json" \
  "https://api.github.com/repos/${GH_REPO}/actions/workflows/oracle_provision.yml/dispatches" \
  -d "$DISPATCH_BODY")

if [ "$HTTP" = "204" ]; then
  echo "  ✓ Workflow triggered!"
else
  echo "  ✗ Trigger failed (HTTP ${HTTP}):"
  cat /tmp/dispatch_resp.json
  echo ""
  echo "  Trigger it manually: Actions → Oracle Cloud — Provision Always Free VM → Run workflow"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Done! The oracle_provision.yml workflow is now running."
echo ""
echo "  Watch progress:"
echo "  https://github.com/${GH_REPO}/actions/workflows/oracle_provision.yml"
echo ""
echo "  When complete (~5 min):"
echo "    Webmail: http://mail.penux.uk"
echo "    IMAP:    mail.penux.uk:143  (user: ${IMAP_USER})"
echo "════════════════════════════════════════════════════════"
