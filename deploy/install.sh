#!/usr/bin/env bash
# Install PenuX IMAP + Webmail + DDNS on a bare Ubuntu/Debian server.
# Usage: sudo CF_TOKEN=<token> bash deploy/install.sh [--imap-user netanel] [--imap-pass secret]
set -euo pipefail

IMAP_USER="${IMAP_USER:-netanel}"
IMAP_PASS="${IMAP_PASS:-}"
CF_TOKEN="${CF_TOKEN:-}"
INSTALL_DIR="/opt/penux-imap"
MAIL_DIR="/var/mail/penux"
CONFIG_DIR="/etc/penux-imap"

echo "=== PenuX installer (IMAP + Webmail + DDNS) ==="

while [[ $# -gt 0 ]]; do
  case "$1" in
    --imap-user) IMAP_USER="$2"; shift 2;;
    --imap-pass) IMAP_PASS="$2"; shift 2;;
    --cf-token)  CF_TOKEN="$2";  shift 2;;
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
mkdir -p "${INSTALL_DIR}" "${MAIL_DIR}" "${CONFIG_DIR}" /etc/ssl/penux \
         /var/cache/penux-ddns
chown penux-imap:penux-imap "${MAIL_DIR}" "${CONFIG_DIR}" /var/cache/penux-ddns

# Write env file (CF_TOKEN + server name)
ENV_FILE="${CONFIG_DIR}/imap.env"
if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" << EOF
IMAP_SERVERNAME=mail.penux.uk
IMAP_PORT=143
IMAPS_PORT=993
IMAP_MAILDIR=${MAIL_DIR}
IMAP_TLS_CERT=/etc/ssl/penux/fullchain.pem
IMAP_TLS_KEY=/etc/ssl/penux/privkey.pem
IMAP_USERS_FILE=${CONFIG_DIR}/users.json
EOF
  chown root:penux-imap "${ENV_FILE}"
  chmod 640 "${ENV_FILE}"
fi

# Write CF_TOKEN to env file (append or replace)
if [[ -n "${CF_TOKEN}" ]]; then
  if grep -q "^CF_TOKEN=" "${ENV_FILE}" 2>/dev/null; then
    sed -i "s|^CF_TOKEN=.*|CF_TOKEN=${CF_TOKEN}|" "${ENV_FILE}"
  else
    echo "CF_TOKEN=${CF_TOKEN}" >> "${ENV_FILE}"
  fi
  echo "CF_TOKEN written to ${ENV_FILE}"
fi

# Copy code
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -a "${SCRIPT_DIR}/imap_server/" "${INSTALL_DIR}/imap_server/"
cp "${SCRIPT_DIR}/run_imap.py"   "${SCRIPT_DIR}/manage_users.py" \
   "${SCRIPT_DIR}/webmail.py"    "${SCRIPT_DIR}/run_webmail.py" \
   "${SCRIPT_DIR}/ddns.py"       "${INSTALL_DIR}/"

# Virtualenv
python3 -m venv "${INSTALL_DIR}/venv"
"${INSTALL_DIR}/venv/bin/pip" install --quiet --upgrade pip

# Systemd services
cp "${SCRIPT_DIR}/deploy/systemd/penux-imap.service"    /etc/systemd/system/
cp "${SCRIPT_DIR}/deploy/systemd/penux-webmail.service" /etc/systemd/system/
cp "${SCRIPT_DIR}/deploy/systemd/penux-ddns.service"    /etc/systemd/system/
cp "${SCRIPT_DIR}/deploy/systemd/penux-ddns.timer"      /etc/systemd/system/
systemctl daemon-reload
systemctl enable penux-imap penux-webmail penux-ddns.timer

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
echo ""
echo "=== Installation complete ==="
echo ""
echo "Start all services:"
echo "  systemctl start penux-imap penux-webmail penux-ddns.timer"
echo ""
echo "DDNS will update mail.penux.uk → this server's IP every 5 min."
echo "Run it once right now:"
echo "  systemctl start penux-ddns"
echo ""
echo "Open firewall:"
echo "  ufw allow 22 143 993 80 443 && ufw --force enable"
echo ""
echo "Get TLS cert (after DNS is pointing here):"
echo "  apt install certbot python3-certbot-dns-cloudflare"
echo "  certbot certonly --dns-cloudflare --dns-cloudflare-credentials /etc/cloudflare.ini \\"
echo "    -d mail.penux.uk"
echo "  cp /etc/letsencrypt/live/mail.penux.uk/fullchain.pem /etc/ssl/penux/"
echo "  cp /etc/letsencrypt/live/mail.penux.uk/privkey.pem   /etc/ssl/penux/"
echo "  systemctl restart penux-imap penux-webmail"
echo ""
echo "Check logs:"
echo "  journalctl -u penux-imap -f"
echo "  journalctl -u penux-webmail -f"
echo "  journalctl -u penux-ddns -f"
