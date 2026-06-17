#!/usr/bin/env bash
# uninstall-kimchi-wok.sh
# Completely removes the Wok + Kimchi installation created by
# install-kimchi-wok-py39.sh
# Usage: sudo bash uninstall-kimchi-wok.sh

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: Run as root (sudo bash $0)" >&2
  exit 1
fi

echo ">>> Stopping wokd..."
systemctl stop wokd 2>/dev/null || true
systemctl disable wokd 2>/dev/null || true

echo ">>> Removing systemd unit..."
rm -f /etc/systemd/system/wokd.service
systemctl daemon-reload

echo ">>> Removing source trees..."
rm -rf /opt/wok /opt/kimchi

echo ">>> Removing configuration and logs..."
rm -rf /etc/wok /var/log/wok /var/run/wok

echo ">>> Removing launcher..."
rm -f /usr/local/bin/wokd

echo ">>> Removing plugin directory..."
rm -rf /usr/share/wok

echo ""
echo "Wok + Kimchi have been removed."
echo ""
echo "To also remove system packages installed by the script:"
echo "  apt remove --purge qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virtinst"
echo "  apt autoremove"
echo ""
echo "To remove pip packages:"
echo "  pip3 uninstall -y CherryPy Cheetah3 python-pam jsonschema pyOpenSSL passlib bcrypt"
