#!/bin/bash
# Cloudflare Tunnel setup for GitHub Codespaces (no static IP, no browser auth)
# Usage: ./codespaces-tunnel-setup.sh [DOCKER_USER]
#
# Required Codespaces secrets:
#   TUNNEL_TOKEN         - from Cloudflare dashboard -> Zero Trust -> Tunnels -> your tunnel -> Configure -> Install connector
#   CLOUDFLARE_API_TOKEN - from Cloudflare dashboard -> My Profile -> API Tokens (Zone:DNS:Edit, Account:Cloudflare Tunnel:Edit)
#
# Optional:
#   DOCKERHUB_USERNAME   - defaults to first argument or "netanelcyber"

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✅ $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; exit 1; }
hdr()  { echo -e "\n${BLUE}=== $1 ===${NC}\n"; }

DOCKER_USER="${1:-${DOCKERHUB_USERNAME:-netanelcyber}}"

hdr "PenuX LDAP - Codespaces Tunnel Setup"
echo "Docker user: $DOCKER_USER"
echo "Running in: ${CODESPACES:+GitHub Codespaces}${CODESPACES:-local}"
echo ""

# ── 1. Check secrets ────────────────────────────────────────────────────────
hdr "Step 1: Checking secrets"

if [ -z "$TUNNEL_TOKEN" ] && [ -z "$CLOUDFLARE_API_TOKEN" ]; then
  fail "No Cloudflare credentials found.

Set one of these as a Codespaces secret:
  TUNNEL_TOKEN         → Cloudflare dashboard → Zero Trust → Tunnels → <your tunnel> → Configure → Install connector → copy the token
  CLOUDFLARE_API_TOKEN → Cloudflare dashboard → My Profile → API Tokens

Then rebuild the Codespace (Ctrl+Shift+P → 'Rebuild Container') or export manually:
  export TUNNEL_TOKEN=eyJhIjo...
"
fi

[ -n "$TUNNEL_TOKEN" ] && ok "TUNNEL_TOKEN found (preferred method)" || warn "TUNNEL_TOKEN not set, will try API token method"
[ -n "$CLOUDFLARE_API_TOKEN" ] && ok "CLOUDFLARE_API_TOKEN found"

# ── 2. Check tools ──────────────────────────────────────────────────────────
hdr "Step 2: Checking tools"

command -v docker &>/dev/null || fail "Docker not found. Is docker-in-docker feature enabled in devcontainer.json?"
ok "Docker: $(docker --version)"

if ! docker compose version &>/dev/null && ! command -v docker-compose &>/dev/null; then
  fail "docker compose / docker-compose not found"
fi
ok "Docker Compose available"

if ! command -v cloudflared &>/dev/null; then
  info "Installing cloudflared..."
  curl -fsSL --output /tmp/cloudflared.deb \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
  sudo dpkg -i /tmp/cloudflared.deb
  rm /tmp/cloudflared.deb
fi
ok "cloudflared: $(cloudflared --version)"

# ── 3. Pull images ───────────────────────────────────────────────────────────
hdr "Step 3: Pull Docker Hub images"

info "Pulling $DOCKER_USER/penux-ldap-api:latest ..."
docker pull "$DOCKER_USER/penux-ldap-api:latest"
ok "API image pulled"

info "Pulling $DOCKER_USER/penux-ldap-web:latest ..."
docker pull "$DOCKER_USER/penux-ldap-web:latest"
ok "Web image pulled"

# ── 4. Start services ────────────────────────────────────────────────────────
hdr "Step 4: Start Docker services"

COMPOSE_FILE="docker-compose-hub.yml"
[ -f "$COMPOSE_FILE" ] || fail "$COMPOSE_FILE not found"

# Use 'docker compose' (v2) or fall back to 'docker-compose'
COMPOSE_CMD="docker compose"
docker compose version &>/dev/null || COMPOSE_CMD="docker-compose"

info "Starting services..."
$COMPOSE_CMD -f "$COMPOSE_FILE" up -d

sleep 5
$COMPOSE_CMD -f "$COMPOSE_FILE" ps
ok "Services started"

# ── 5. Start Cloudflare Tunnel ───────────────────────────────────────────────
hdr "Step 5: Start Cloudflare Tunnel"

LOGFILE="/tmp/cloudflared.log"

if [ -n "$TUNNEL_TOKEN" ]; then
  # Modern approach: single token embeds all credentials, no files needed
  info "Starting tunnel with TUNNEL_TOKEN..."
  pkill cloudflared 2>/dev/null || true
  nohup cloudflared tunnel run --token "$TUNNEL_TOKEN" >"$LOGFILE" 2>&1 &
  CF_PID=$!
  echo $CF_PID > /tmp/cloudflared.pid
  ok "Tunnel started (PID $CF_PID)"

elif [ -n "$CLOUDFLARE_API_TOKEN" ]; then
  # Legacy API token approach: requires cert + credentials file
  info "Authenticating with API token..."
  export CLOUDFLARE_API_TOKEN

  # Write cert using API token (non-interactive)
  mkdir -p ~/.cloudflared
  cloudflared tunnel login --api-token "$CLOUDFLARE_API_TOKEN" 2>/dev/null || \
    warn "Login warning — tunnel may already exist"

  TUNNEL_NAME="penux-ldap"
  cloudflared tunnel create "$TUNNEL_NAME" 2>/dev/null || warn "Tunnel already exists"

  CREDS=$(ls ~/.cloudflared/*.json 2>/dev/null | grep -v cert | head -1)
  mkdir -p ~/.cloudflared

  cat > ~/.cloudflared/config.yml << EOF
tunnel: $TUNNEL_NAME
credentials-file: $CREDS

ingress:
  - hostname: ldap.penux.uk
    service: http://localhost:3001
  - hostname: penux.uk
    service: http://localhost:3001
  - hostname: api.ldap.penux.uk
    service: http://localhost:3000
  - service: http_status:404
EOF

  info "Creating DNS routes..."
  cloudflared tunnel route dns "$TUNNEL_NAME" ldap.penux.uk 2>/dev/null || true
  cloudflared tunnel route dns "$TUNNEL_NAME" penux.uk 2>/dev/null || true
  cloudflared tunnel route dns "$TUNNEL_NAME" api.ldap.penux.uk 2>/dev/null || true
  ok "DNS routes configured"

  pkill cloudflared 2>/dev/null || true
  nohup cloudflared tunnel run "$TUNNEL_NAME" >"$LOGFILE" 2>&1 &
  CF_PID=$!
  echo $CF_PID > /tmp/cloudflared.pid
  ok "Tunnel started (PID $CF_PID)"
fi

# ── 6. Verify ────────────────────────────────────────────────────────────────
hdr "Step 6: Verify tunnel"

info "Waiting 10 seconds for tunnel to connect..."
sleep 10

if grep -q "Registered tunnel connection" "$LOGFILE" 2>/dev/null || \
   grep -q "connection registered" "$LOGFILE" 2>/dev/null || \
   grep -q "INF" "$LOGFILE" 2>/dev/null; then
  ok "Tunnel is connected!"
else
  warn "Tunnel may still be starting. Check logs:"
  tail -20 "$LOGFILE" 2>/dev/null || true
fi

# ── Summary ──────────────────────────────────────────────────────────────────
hdr "Done!"

echo ""
echo "  Live URLs:"
echo "    https://ldap.penux.uk     (Web UI)"
echo "    https://penux.uk          (Root)"
echo "    https://api.ldap.penux.uk (API)"
echo ""
echo "  Tunnel logs:    tail -f $LOGFILE"
echo "  Stop tunnel:    kill \$(cat /tmp/cloudflared.pid)"
echo "  Service logs:   $COMPOSE_CMD -f $COMPOSE_FILE logs -f"
echo "  Health check:   curl https://api.ldap.penux.uk/api/health"
echo ""
