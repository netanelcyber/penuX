#!/usr/bin/env bash
# Install PenuX IMAP server on a bare Ubuntu/Debian server
# Usage: sudo bash deploy/install.sh [--imap-user netanel] [--imap-pass secret]
set -euo pipefail

IMAP_USER="${IMAP_USER:-netanel}"
IMAP_PASS="${IMAP_PASS:-}"
INSTALL_DIR="/opt/penux-imap"
MAIL_DIR="/var/mail/penux"
CONFIG_DIR="/etc/penux-imap"

echo "=== PenuX IMAP Server installer ==="

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --imap-user) IMAP_USER="$2"; shift 2;;
    --imap-pass) IMAP_PASS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Dependencies
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip rsync

# System user
if ! id penux-imap &>/dev/null; then
  useradd --system --no-create-home --shell /usr/sbin/nologin penux-imap
fi

# Directories
mkdir -p "${INSTALL_DIR}" "${MAIL_DIR}" "${CONFIG_DIR}" /etc/ssl/penux
chown penux-imap:penux-imap "${MAIL_DIR}" "${CONFIG_DIR}"

# Copy code
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -a "${SCRIPT_DIR}/imap_server/" "${INSTALL_DIR}/imap_server/"
cp "${SCRIPT_DIR}/run_imap.py" "${SCRIPT_DIR}/manage_users.py" "${INSTALL_DIR}/"

# Virtualenv
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --quiet --upgrade pip

# Systemd service
cp "${SCRIPT_DIR}/deploy/systemd/penux-imap.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable penux-imap

# Create initial user if password provided
if [[ -n "${IMAP_PASS}" ]]; then
  IMAP_USERS_FILE="${CONFIG_DIR}/users.json" \
  IMAP_MAILDIR="${MAIL_DIR}" \
    "${INSTALL_DIR}/venv/bin/python" "${INSTALL_DIR}/manage_users.py" \
    add "${IMAP_USER}" "${IMAP_PASS}" || true
  chown penux-imap:penux-imap "${CONFIG_DIR}/users.json" 2>/dev/null || true
fi

echo ""
echo "Installation complete."
echo ""
echo "Next steps:"
echo "  1. Get a TLS cert:  certbot certonly --dns-cloudflare -d mail.penux.uk"
echo "     Copy certs to:   /etc/ssl/penux/fullchain.pem  and  /etc/ssl/penux/privkey.pem"
echo "  2. Start the server: systemctl start penux-imap"
echo "  3. Check logs:       journalctl -u penux-imap -f"
echo "  4. Add users:        IMAP_USERS_FILE=${CONFIG_DIR}/users.json \\"
echo "                         python ${INSTALL_DIR}/manage_users.py add <user> <pass>"
