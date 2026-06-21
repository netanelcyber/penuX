#!/bin/bash
# Post-create setup for GitHub Codespaces
set -e

echo "=== PenuX LDAP Codespaces Setup ==="

# Install cloudflared
if ! command -v cloudflared &>/dev/null; then
  echo "Installing cloudflared..."
  curl -fsSL --output /tmp/cloudflared.deb \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i /tmp/cloudflared.deb
  rm /tmp/cloudflared.deb
fi
echo "cloudflared: $(cloudflared --version)"

# Make scripts executable
chmod +x cloudflare-tunnel-setup.sh 2>/dev/null || true
chmod +x codespaces-tunnel-setup.sh 2>/dev/null || true
chmod +x docker-hub-push.sh 2>/dev/null || true

echo ""
echo "=== Ready! ==="
echo ""
echo "Run the tunnel setup:"
echo "  ./codespaces-tunnel-setup.sh netanelcyber"
echo ""
echo "Required Codespaces secrets (Settings -> Secrets -> Codespaces):"
echo "  TUNNEL_TOKEN         - from dash.cloudflare.com -> Zero Trust -> Tunnels"
echo "  CLOUDFLARE_API_TOKEN - from dash.cloudflare.com -> My Profile -> API Tokens"
echo "  DOCKERHUB_USERNAME   - your Docker Hub username"
