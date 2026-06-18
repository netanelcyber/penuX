#!/usr/bin/env bash
# install-kvmweb.sh — Install kvmweb KVM web management interface
# Supports Ubuntu 22.04/24.04, Debian 11/12
set -euo pipefail

INSTALL_DIR=/opt/kvmweb
CONF_DIR=/etc/kvmweb
SSL_DIR=/etc/kvmweb/ssl
SERVICE_FILE=/etc/systemd/system/kvmweb.service
PORT=8080

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[kvmweb]${NC} $*"; }
warn()    { echo -e "${YELLOW}[warn]${NC}  $*"; }
err_exit(){ echo -e "${RED}[error]${NC} $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || err_exit "Run as root: sudo $0"

# ── 1. System packages ────────────────────────────────────────────────────────
info "Installing system packages…"
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    python3-libvirt \
    python3-pam \
    libvirt-daemon-system libvirt-clients \
    qemu-kvm qemu-utils \
    virtinst \
    openssl \
    curl

# ── 2. Python deps (venv) ─────────────────────────────────────────────────────
info "Creating Python virtual environment at ${INSTALL_DIR}…"
mkdir -p "$INSTALL_DIR"
python3 -m venv "$INSTALL_DIR/venv" --system-site-packages
# system-site-packages gives us python3-libvirt and python3-pam without root pip

"$INSTALL_DIR/venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/venv/bin/pip" install --quiet flask bcrypt

# ── 3. Copy application ───────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SRC="$SCRIPT_DIR/kvmweb"

[[ -d "$APP_SRC" ]] || err_exit "kvmweb/ directory not found next to this script (expected: $APP_SRC)"

info "Copying application files to ${INSTALL_DIR}…"
cp -r "$APP_SRC"/. "$INSTALL_DIR/"

# ── 4. Config directory & users file ─────────────────────────────────────────
info "Setting up config directory at ${CONF_DIR}…"
mkdir -p "$CONF_DIR"

if [[ ! -f "$CONF_DIR/users" ]]; then
    touch "$CONF_DIR/users"
    chmod 600 "$CONF_DIR/users"
    info "Created empty $CONF_DIR/users (PAM auth used by default; add bcrypt lines here as fallback)"
fi

# ── 5. TLS certificate ────────────────────────────────────────────────────────
info "Generating self-signed TLS certificate…"
mkdir -p "$SSL_DIR"
if [[ ! -f "$SSL_DIR/cert.pem" ]]; then
    HOSTNAME=$(hostname -f 2>/dev/null || hostname)
    openssl req -x509 -nodes -newkey rsa:2048 -days 3650 \
        -keyout "$SSL_DIR/key.pem" \
        -out    "$SSL_DIR/cert.pem" \
        -subj "/CN=${HOSTNAME}/O=kvmweb/C=US" \
        -addext "subjectAltName=DNS:${HOSTNAME},IP:127.0.0.1" \
        2>/dev/null
    chmod 600 "$SSL_DIR/key.pem"
    info "Certificate: $SSL_DIR/cert.pem"
else
    info "TLS cert already exists, skipping generation."
fi

# ── 6. systemd service ────────────────────────────────────────────────────────
info "Installing systemd service…"
cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=kvmweb - KVM Web Management Interface
After=network.target libvirtd.service
Wants=libvirtd.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/venv/bin/python3 ${INSTALL_DIR}/app.py \\
    --host 0.0.0.0 \\
    --port ${PORT} \\
    --cert ${SSL_DIR}/cert.pem \\
    --key  ${SSL_DIR}/key.pem
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=kvmweb
Environment=KVMWEB_SESSION_MINS=60

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable kvmweb
systemctl restart kvmweb

# ── 7. libvirt group ─────────────────────────────────────────────────────────
info "Ensuring libvirtd is running…"
systemctl enable --now libvirtd 2>/dev/null || true

# ── 8. Firewall hint ─────────────────────────────────────────────────────────
if command -v ufw &>/dev/null && ufw status | grep -q "Status: active"; then
    warn "ufw is active. Allow port ${PORT} with: ufw allow ${PORT}/tcp"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
IP=$(hostname -I | awk '{print $1}')
echo
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN} kvmweb installed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo
echo "  URL:   https://${IP}:${PORT}"
echo "  Auth:  Linux system credentials (PAM)"
echo "  Logs:  journalctl -u kvmweb -f"
echo "  Stop:  systemctl stop kvmweb"
echo
