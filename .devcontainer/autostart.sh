#!/bin/bash
# Auto-start the full PenuX stack + tunnel whenever the Codespace starts.
# Runs unattended. Logs to /tmp/penux-autostart.log
set -e

cd "$(dirname "$0")/.." || exit 1

echo "=== PenuX auto-start $(date) ==="

# Wait for Docker daemon to be ready (docker-in-docker can lag on boot)
for i in $(seq 1 30); do
  if docker info >/dev/null 2>&1; then break; fi
  echo "waiting for docker daemon ($i/30)..."
  sleep 2
done

# DOCKER_USER comes from remoteEnv; default to netanelcyber
USER_ARG="${DOCKER_USER:-netanelcyber}"

# NGROK_AUTHTOKEN / NGROK_DOMAIN are injected automatically if set as
# Codespaces secrets. The script picks the right method based on their presence.
exec ./free-tunnel-setup.sh "$USER_ARG"
