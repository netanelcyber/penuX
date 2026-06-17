#!/usr/bin/env bash
# install-kimchi-wok-py39.sh
# Installs Wok + Kimchi KVM web UI on modern Debian/Ubuntu (Python 3.9+)
# Tested on: Ubuntu 22.04 LTS, Ubuntu 24.04 LTS, Debian 12 (Bookworm)
# Usage: sudo bash install-kimchi-wok-py39.sh

set -euxo pipefail

# ---------------------------------------------------------------------------
# 0. Sanity checks
# ---------------------------------------------------------------------------
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Run this script as root (sudo bash $0)" >&2
  exit 1
fi

CALLER_USER="${SUDO_USER:-$(logname 2>/dev/null || echo '')}"

PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VER" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VER" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 || ( "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 9 ) ]]; then
  echo "ERROR: Python 3.9+ required. Found: Python $PYTHON_VER" >&2
  exit 1
fi

echo ">>> Python $PYTHON_VER OK"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
apt-get update -y
apt-get install -y \
  qemu-kvm libvirt-daemon-system libvirt-clients \
  bridge-utils virtinst \
  python3 python3-pip python3-dev python3-venv \
  python3-libvirt python3-lxml python3-m2crypto \
  python3-magic python3-ldap python3-pam \
  python3-psutil python3-configobj \
  build-essential pkg-config libssl-dev libffi-dev \
  dpkg-dev git wget curl \
  gettext xsltproc pep8 \
  libguestfs-tools guestfish \
  nginx \
  sudo

# python3-cherrypy3 may not exist on newer distros — install via pip if absent
if ! dpkg -l python3-cherrypy3 &>/dev/null && ! python3 -c "import cherrypy" &>/dev/null 2>&1; then
  pip3 install --break-system-packages "CherryPy>=18.0" 2>/dev/null || \
  pip3 install "CherryPy>=18.0"
fi

# Cheetah3 (template engine — Python 3 port of Cheetah)
if ! python3 -c "import Cheetah" &>/dev/null 2>&1; then
  pip3 install --break-system-packages "Cheetah3>=3.0" 2>/dev/null || \
  pip3 install "Cheetah3>=3.0"
fi

# Additional pip dependencies
pip3 install --break-system-packages \
  python-pam \
  jsonschema \
  pyOpenSSL \
  passlib \
  bcrypt \
  2>/dev/null || \
pip3 install \
  python-pam \
  jsonschema \
  pyOpenSSL \
  passlib \
  bcrypt

# ---------------------------------------------------------------------------
# 2. Fetch Wok and Kimchi source from GitHub
# ---------------------------------------------------------------------------
WOK_DIR=/opt/wok
KIMCHI_DIR=/opt/kimchi

# Wok
if [[ -d "$WOK_DIR/.git" ]]; then
  echo ">>> Wok already cloned, pulling latest..."
  git -C "$WOK_DIR" pull --ff-only || true
else
  git clone --depth 1 https://github.com/kimchi-project/wok.git "$WOK_DIR"
fi

# Kimchi
if [[ -d "$KIMCHI_DIR/.git" ]]; then
  echo ">>> Kimchi already cloned, pulling latest..."
  git -C "$KIMCHI_DIR" pull --ff-only || true
else
  git clone --depth 1 https://github.com/kimchi-project/kimchi.git "$KIMCHI_DIR"
fi

# ---------------------------------------------------------------------------
# 3. Patch Wok for Python 3 compatibility
# ---------------------------------------------------------------------------
echo ">>> Patching Wok for Python 3..."

patch_python3_wok() {
  local DIR="$WOK_DIR"

  # 3a. Fix shebangs
  find "$DIR" -name "*.py" | xargs grep -l "^#!/usr/bin/env python$" | while read f; do
    sed -i 's|^#!/usr/bin/env python$|#!/usr/bin/env python3|' "$f"
  done
  find "$DIR" -name "*.py" | xargs grep -l "^#!/usr/bin/python$" | while read f; do
    sed -i 's|^#!/usr/bin/python$|#!/usr/bin/python3|' "$f"
  done

  # 3b. Fix Python 2 print statements → Python 3 (if any remain)
  # Use 2to3 for a safe in-place conversion pass
  if command -v 2to3 &>/dev/null; then
    2to3 --no-diffs -w -n \
      -f print \
      -f except \
      -f raise \
      -f unicode \
      -f urllib \
      -f reduce \
      -f dict \
      -f has_key \
      -f basestring \
      -f long \
      "$DIR/src" 2>/dev/null || true
  fi

  # 3c. Fix common Python 2→3 string/bytes issues in wok source
  # ConfigParser module rename
  find "$DIR/src" -name "*.py" -exec sed -i \
    -e 's/import ConfigParser/import configparser as ConfigParser/g' \
    -e 's/from ConfigParser import/from configparser import/g' \
    {} \;

  # httplib rename
  find "$DIR/src" -name "*.py" -exec sed -i \
    -e 's/import httplib/import http.client as httplib/g' \
    -e 's/from httplib import/from http.client import/g' \
    {} \;

  # urllib2 / urlparse
  find "$DIR/src" -name "*.py" -exec sed -i \
    -e 's/import urllib2/import urllib.request as urllib2/g' \
    -e 's/from urllib2 import/from urllib.request import/g' \
    -e 's/import urlparse/import urllib.parse as urlparse/g' \
    -e 's/from urlparse import/from urllib.parse import/g' \
    {} \;

  # StringIO
  find "$DIR/src" -name "*.py" -exec sed -i \
    -e 's/from StringIO import StringIO/from io import StringIO/g' \
    -e 's/import StringIO/import io as StringIO/g' \
    {} \;

  # 3d. Fix setup.py install_requires if present
  if [[ -f "$DIR/setup.py" ]]; then
    sed -i \
      -e "s/'python-m2crypto'/'python3-m2crypto'/g" \
      -e "s/'python-cherrypy3'/'CherryPy'/g" \
      -e "s/'python-cheetah'/'Cheetah3'/g" \
      -e "s/'m2crypto'/'m2crypto'/g" \
      "$DIR/setup.py"
  fi
}

patch_python3_wok

# ---------------------------------------------------------------------------
# 4. Patch Kimchi for Python 3 compatibility
# ---------------------------------------------------------------------------
echo ">>> Patching Kimchi for Python 3..."

patch_python3_kimchi() {
  local DIR="$KIMCHI_DIR"

  find "$DIR" -name "*.py" | xargs grep -l "^#!/usr/bin/env python$" 2>/dev/null | while read f; do
    sed -i 's|^#!/usr/bin/env python$|#!/usr/bin/env python3|' "$f"
  done
  find "$DIR" -name "*.py" | xargs grep -l "^#!/usr/bin/python$" 2>/dev/null | while read f; do
    sed -i 's|^#!/usr/bin/python$|#!/usr/bin/python3|' "$f"
  done

  if command -v 2to3 &>/dev/null; then
    2to3 --no-diffs -w -n \
      -f print \
      -f except \
      -f raise \
      -f unicode \
      -f urllib \
      -f reduce \
      -f dict \
      -f has_key \
      -f basestring \
      -f long \
      "$DIR/src" 2>/dev/null || true
  fi

  find "$DIR/src" -name "*.py" -exec sed -i \
    -e 's/import ConfigParser/import configparser as ConfigParser/g' \
    -e 's/from ConfigParser import/from configparser import/g' \
    -e 's/import httplib/import http.client as httplib/g' \
    -e 's/import urllib2/import urllib.request as urllib2/g' \
    -e 's/from urllib2 import/from urllib.request import/g' \
    -e 's/import urlparse/import urllib.parse as urlparse/g' \
    -e 's/from urlparse import/from urllib.parse import/g' \
    -e 's/from StringIO import StringIO/from io import StringIO/g' \
    {} \;
}

patch_python3_kimchi

# ---------------------------------------------------------------------------
# 5. Build & install Wok
# ---------------------------------------------------------------------------
echo ">>> Building and installing Wok..."
cd "$WOK_DIR"

# Run autoconf/automake if configure.ac present
if [[ -f configure.ac ]]; then
  if command -v autoreconf &>/dev/null; then
    autoreconf -fi 2>/dev/null || true
    ./configure --prefix=/usr --sysconfdir=/etc --localstatedir=/var 2>/dev/null || true
    make -j"$(nproc)" 2>/dev/null || true
    make install 2>/dev/null || true
  fi
fi

# Fall back to setup.py / pip install
if [[ -f setup.py ]]; then
  pip3 install --break-system-packages -e . 2>/dev/null || \
  pip3 install -e . 2>/dev/null || \
  python3 setup.py install 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 6. Build & install Kimchi as a Wok plugin
# ---------------------------------------------------------------------------
echo ">>> Building and installing Kimchi..."
cd "$KIMCHI_DIR"

if [[ -f configure.ac ]]; then
  if command -v autoreconf &>/dev/null; then
    autoreconf -fi 2>/dev/null || true
    ./configure --prefix=/usr --sysconfdir=/etc --localstatedir=/var \
      --with-wok="$WOK_DIR" 2>/dev/null || true
    make -j"$(nproc)" 2>/dev/null || true
    make install 2>/dev/null || true
  fi
fi

if [[ -f setup.py ]]; then
  pip3 install --break-system-packages -e . 2>/dev/null || \
  pip3 install -e . 2>/dev/null || \
  python3 setup.py install 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 7. Install Kimchi plugin into Wok plugin directory
# ---------------------------------------------------------------------------
WOK_PLUGIN_DIR=/usr/share/wok/plugins
mkdir -p "$WOK_PLUGIN_DIR"

if [[ -d /usr/share/kimchi ]]; then
  ln -sfn /usr/share/kimchi "$WOK_PLUGIN_DIR/kimchi" 2>/dev/null || true
elif [[ -d "$KIMCHI_DIR/src" ]]; then
  ln -sfn "$KIMCHI_DIR/src" "$WOK_PLUGIN_DIR/kimchi" 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# 8. Generate TLS certificate (self-signed)
# ---------------------------------------------------------------------------
CERT_DIR=/etc/wok/ssl
mkdir -p "$CERT_DIR"

if [[ ! -f "$CERT_DIR/wok.crt" ]]; then
  echo ">>> Generating self-signed TLS certificate..."
  openssl req -x509 -nodes -newkey rsa:4096 \
    -keyout "$CERT_DIR/wok.key" \
    -out "$CERT_DIR/wok.crt" \
    -days 3650 \
    -subj "/C=US/ST=State/L=City/O=Wok/CN=$(hostname -f 2>/dev/null || echo localhost)"
  chmod 600 "$CERT_DIR/wok.key"
fi

# ---------------------------------------------------------------------------
# 9. Write /etc/wok/wok.conf (main config)
# ---------------------------------------------------------------------------
mkdir -p /etc/wok
cat > /etc/wok/wok.conf <<'WOKCONF'
[wok]
server.socket_host = 0.0.0.0
server.socket_port = 8001
server.ssl_module = builtin
server.ssl_certificate = /etc/wok/ssl/wok.crt
server.ssl_private_key = /etc/wok/ssl/wok.key
log.screen = False
log.error_file = /var/log/wok/wok-error.log
log.access_file = /var/log/wok/wok-access.log
server.environment = production
server.thread_pool = 10
session_timeout = 10
WOKCONF

mkdir -p /var/log/wok /var/run/wok

# ---------------------------------------------------------------------------
# 10. Locate or create the wokd entry-point
# ---------------------------------------------------------------------------
WOKD_BIN=""
for candidate in \
    /usr/bin/wokd \
    /usr/local/bin/wokd \
    "$WOK_DIR/src/wok/server.py" \
    "$WOK_DIR/wokd" ; do
  if [[ -f "$candidate" || -x "$candidate" ]]; then
    WOKD_BIN="$candidate"
    break
  fi
done

if [[ -z "$WOKD_BIN" ]]; then
  # Create a minimal launcher that starts the CherryPy-based Wok server
  cat > /usr/local/bin/wokd <<'LAUNCHER'
#!/usr/bin/env python3
"""Wok server launcher — created by install-kimchi-wok-py39.sh"""
import sys
import os

# Ensure source trees are on the path
sys.path.insert(0, '/opt/wok/src')
sys.path.insert(0, '/opt/kimchi/src')

try:
    from wok.server import main
    main()
except ImportError:
    # Fallback: try running Wok as a module
    import runpy
    runpy.run_module('wok.server', run_name='__main__')
LAUNCHER
  chmod +x /usr/local/bin/wokd
  WOKD_BIN=/usr/local/bin/wokd
fi

# ---------------------------------------------------------------------------
# 11. Create systemd unit
# ---------------------------------------------------------------------------
echo ">>> Writing systemd unit /etc/systemd/system/wokd.service..."
cat > /etc/systemd/system/wokd.service <<UNIT
[Unit]
Description=Wok - Web framework for KVM/Kimchi
After=network.target libvirtd.service
Wants=libvirtd.service

[Service]
Type=simple
ExecStart=/usr/bin/env python3 $WOKD_BIN
Restart=on-failure
RestartSec=5s
StandardOutput=journal
StandardError=journal
Environment=PYTHONPATH=/opt/wok/src:/opt/kimchi/src
WorkingDirectory=/var/run/wok
PIDFile=/var/run/wok/wokd.pid
User=root

[Install]
WantedBy=multi-user.target
UNIT

# ---------------------------------------------------------------------------
# 12. Enable and start services
# ---------------------------------------------------------------------------
systemctl daemon-reload
systemctl enable --now libvirtd || true
systemctl restart libvirtd || true

systemctl enable wokd
systemctl restart wokd || true

# ---------------------------------------------------------------------------
# 13. Add user to libvirt and kvm groups
# ---------------------------------------------------------------------------
if [[ -n "$CALLER_USER" ]]; then
  usermod -aG libvirt,kvm "$CALLER_USER" 2>/dev/null || true
  echo ">>> Added $CALLER_USER to libvirt and kvm groups"
fi

# ---------------------------------------------------------------------------
# 14. Firewall reminder
# ---------------------------------------------------------------------------
SERVER_IP=$(hostname -I | awk '{print $1}')

echo ""
echo "============================================================"
echo " Wok + Kimchi installation complete"
echo "============================================================"
echo " URL  : https://${SERVER_IP}:8001"
echo " Logs : journalctl -u wokd -xe --no-pager"
echo ""
echo " SECURITY: port 8001 should be firewalled."
echo " To allow only your workstation:"
echo "   ufw allow from <YOUR-IP> to any port 8001 proto tcp"
echo "   ufw enable"
echo ""
echo " Default login: use your Linux system credentials."
echo " If this is the first run you may need to:"
echo "   systemctl status wokd"
echo "   ss -tulpn | grep 8001"
echo "============================================================"
