#!/usr/bin/env bash
# Fully automated Launchpad PPA setup + GitHub Actions trigger
# Usage: GITHUB_TOKEN=ghp_... bash setup-launchpad-ppa.sh
set -euo pipefail

LAUNCHPAD_USER="${LAUNCHPAD_USER:-netanelcyber}"
PPA_NAME="${PPA_NAME:-kvmweb}"
GITHUB_REPO="${GITHUB_REPO:-netanelcyber/penuX}"
GITHUB_BRANCH="${GITHUB_BRANCH:-claude/beautiful-fermi-vkcmzu}"
GPG_KEY_ID="${GPG_KEY_ID:-32F8DBA242A47ED3}"
SSH_KEY_FILE="${SSH_KEY_FILE:-/tmp/launchpad_ppa_ed25519}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
step() { echo -e "\n${GREEN}==>${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
die()  { echo -e "${RED}[x]${NC} $*" >&2; exit 1; }

[[ -z "${GITHUB_TOKEN:-}" ]] && die "Set GITHUB_TOKEN env var first."

# ── 1. Generate SSH key ───────────────────────────────────────────────────────
step "Generating SSH key for Launchpad uploads"
if [[ ! -f "$SSH_KEY_FILE" ]]; then
  ssh-keygen -t ed25519 -f "$SSH_KEY_FILE" -N "" -C "kvmweb-launchpad-ci"
  echo "Generated: $SSH_KEY_FILE"
else
  warn "SSH key already exists at $SSH_KEY_FILE, reusing."
fi
SSH_PUB="$(cat "${SSH_KEY_FILE}.pub")"
SSH_PRIV="$(cat "$SSH_KEY_FILE")"

# ── 2. Export GPG public key ──────────────────────────────────────────────────
step "Exporting GPG public key $GPG_KEY_ID"
GPG_FINGERPRINT=$(gpg --with-colons --fingerprint "$GPG_KEY_ID" 2>/dev/null \
  | grep '^fpr' | head -1 | cut -d: -f10)
GPG_PUBKEY=$(gpg --export --armor "$GPG_KEY_ID" 2>/dev/null)
[[ -z "$GPG_PUBKEY" ]] && die "GPG key $GPG_KEY_ID not found in keyring."
echo "Fingerprint: $GPG_FINGERPRINT"

# ── 3. Upload GPG key to keyserver (required by Launchpad) ───────────────────
step "Uploading GPG key to Ubuntu keyserver"
gpg --keyserver keyserver.ubuntu.com --send-keys "$GPG_KEY_ID" 2>&1 || \
  warn "Keyserver upload failed (may already be there, continuing)"

# ── 4. Launchpad API: create PPA + upload SSH key + register GPG key ─────────
step "Automating Launchpad via API"
python3 << PYEOF
import os, sys, time, textwrap

try:
    from launchpadlib.launchpad import Launchpad
    from launchpadlib.credentials import RequestTokenAuthorizationEngine, CredentialStore
except ImportError:
    sys.exit("launchpadlib not installed: pip3 install launchpadlib")

LAUNCHPAD_USER = "${LAUNCHPAD_USER}"
PPA_NAME        = "${PPA_NAME}"
SSH_PUB         = """${SSH_PUB}"""
GPG_FINGERPRINT = "${GPG_FINGERPRINT}"

print("  Authenticating to Launchpad (browser OAuth flow)...")
print("  A URL will open — log in and click 'Allow'.")

lp = Launchpad.login_with(
    "kvmweb-ppa-setup",
    "production",
    version="devel",
    credential_save_failed=lambda: None,
)

me = lp.people[LAUNCHPAD_USER]
print(f"  Logged in as: {me.name}")

# ── Create PPA if it doesn't exist ───────────────────────────────────────────
existing_ppas = [p.name for p in me.ppas]
if PPA_NAME in existing_ppas:
    print(f"  PPA '{PPA_NAME}' already exists.")
    ppa = me.getPPAByName(name=PPA_NAME)
else:
    print(f"  Creating PPA '{PPA_NAME}'...")
    ppa = me.createPPA(
        name=PPA_NAME,
        displayname="kvmweb - KVM Web Interface",
        description="Lightweight KVM/QEMU web management interface with PAM auth and HTTPS.",
        private=False,
    )
    print(f"  PPA created: {ppa.web_link}")

# ── Upload SSH key ────────────────────────────────────────────────────────────
existing_ssh = [k.keytext.strip() for k in me.sshkeys]
if SSH_PUB.strip() in existing_ssh:
    print("  SSH key already registered on Launchpad.")
else:
    print("  Uploading SSH public key to Launchpad...")
    me.addSSHKey(key=SSH_PUB.strip())
    print("  SSH key uploaded.")

# ── Register GPG key ──────────────────────────────────────────────────────────
# Launchpad requires the key to be on the keyserver first, then you import by fingerprint.
existing_gpg = [k.fingerprint for k in me.gpg_keys]
if GPG_FINGERPRINT in existing_gpg:
    print(f"  GPG key {GPG_FINGERPRINT} already registered on Launchpad.")
else:
    print(f"  Requesting GPG key import for fingerprint {GPG_FINGERPRINT}...")
    print("  Launchpad will send a verification email — check your inbox and click the link.")
    try:
        me.validateGPG(fingerprint=GPG_FINGERPRINT)
        print("  GPG import request sent. Check your email to confirm.")
    except Exception as e:
        print(f"  GPG import note: {e}")
        print("  You may need to confirm manually at:")
        print(f"  https://launchpad.net/~{LAUNCHPAD_USER}/+editpgpkeys")

print()
print(f"  PPA URL: ppa:{LAUNCHPAD_USER}/{PPA_NAME}")
print(f"  Install users use: sudo add-apt-repository ppa:{LAUNCHPAD_USER}/{PPA_NAME}")
PYEOF

# ── 5. Upload SSH private key to GitHub Actions secrets ──────────────────────
step "Uploading LAUNCHPAD_SSH_PRIVATE_KEY to GitHub Actions"
python3 << PYEOF
import os, sys, json, base64, urllib.request, urllib.error

REPO  = "${GITHUB_REPO}"
TOKEN = os.environ.get("GITHUB_TOKEN") or sys.exit("No GITHUB_TOKEN")
KEY   = """${SSH_PRIV}"""

def gh(method, path, body=None):
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

# Get repo public key for secret encryption
status, pk = gh("GET", f"/repos/{REPO}/actions/secrets/public-key")
if status != 200:
    sys.exit(f"Failed to get repo public key: {status} {pk}")

key_id  = pk["key_id"]
key_b64 = pk["key"]

try:
    from nacl.public import PublicKey, SealedBox
    import nacl.encoding
    pub  = PublicKey(key_b64, encoder=nacl.encoding.Base64Encoder)
    box  = SealedBox(pub)
    enc  = base64.b64encode(box.encrypt(KEY.encode())).decode()
except ImportError:
    sys.exit("pynacl not installed: pip3 install pynacl")

status, resp = gh("PUT", f"/repos/{REPO}/actions/secrets/LAUNCHPAD_SSH_PRIVATE_KEY", {
    "encrypted_value": enc,
    "key_id": key_id,
})
if status in (201, 204):
    print("  Secret LAUNCHPAD_SSH_PRIVATE_KEY uploaded successfully.")
else:
    sys.exit(f"Failed to upload secret: {status} {resp}")
PYEOF

# ── 6. Trigger the GitHub Actions workflow ───────────────────────────────────
step "Triggering launchpad-ppa.yml workflow"
python3 << PYEOF
import os, sys, json, urllib.request, urllib.error

REPO   = "${GITHUB_REPO}"
TOKEN  = os.environ.get("GITHUB_TOKEN") or sys.exit("No GITHUB_TOKEN")
BRANCH = "${GITHUB_BRANCH}"
LPAD_USER = "${LAUNCHPAD_USER}"
PPA_NAME  = "${PPA_NAME}"

def gh(method, path, body=None):
    url = f"https://api.github.com{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req) as r:
            txt = r.read()
            return r.status, (json.loads(txt) if txt else {})
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

status, resp = gh("POST", f"/repos/{REPO}/actions/workflows/launchpad-ppa.yml/dispatches", {
    "ref": BRANCH,
    "inputs": {
        "ppa_name": PPA_NAME,
        "launchpad_user": LPAD_USER,
    }
})
if status == 204:
    print("  Workflow triggered!")
    print(f"  Watch it at: https://github.com/{REPO}/actions/workflows/launchpad-ppa.yml")
else:
    print(f"  Trigger response: {status} {resp}")
    sys.exit("Workflow trigger failed.")
PYEOF

step "Done!"
echo ""
echo "  Next step: check your email from Launchpad and confirm the GPG key."
echo "  Once confirmed, the workflow run will upload your package to:"
echo "  ppa:${LAUNCHPAD_USER}/${PPA_NAME}"
echo ""
echo "  Users will install with:"
echo "    sudo add-apt-repository ppa:${LAUNCHPAD_USER}/${PPA_NAME}"
echo "    sudo apt install kvmweb"
