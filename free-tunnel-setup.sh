#!/bin/bash
# Free tunnel setup — no Cloudflare, no credit card required.
#
# Method A (RECOMMENDED — permanent URL):
#   ngrok free tier: 1 free static domain, email signup only
#   Sign up at https://ngrok.com → copy your authtoken → set NGROK_AUTHTOKEN
#   Get your free static domain at https://dashboard.ngrok.com/domains
#   Then: NGROK_AUTHTOKEN=xxx NGROK_DOMAIN=your-name.ngrok-free.app ./free-tunnel-setup.sh netanelcyber
#
# Method B (zero signup — temporary URL changes each session):
#   No account needed. Just run: ./free-tunnel-setup.sh netanelcyber
#   Uses ssh tunnel via localhost.run

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

hdr "PenuX LDAP — Free Tunnel Setup (no Cloudflare)"
echo "Docker user : $DOCKER_USER"
echo "ngrok token : ${NGROK_AUTHTOKEN:+(set)}"
echo "ngrok domain: ${NGROK_DOMAIN:-(not set, will use localhost.run)}"
echo ""

# ── 1. Check Docker ──────────────────────────────────────────────────────────
hdr "Step 1: Docker"

command -v docker &>/dev/null || fail "Docker not found"
ok "Docker: $(docker --version)"

COMPOSE="docker compose"
docker compose version &>/dev/null 2>&1 || COMPOSE="docker-compose"
command -v docker-compose &>/dev/null 2>&1 || {
  $COMPOSE version &>/dev/null || fail "docker compose / docker-compose not found"
}
ok "Docker Compose available"

# ── 2. Pull images ───────────────────────────────────────────────────────────
hdr "Step 2: Pull Docker Hub images"

info "Pulling $DOCKER_USER/penux-ldap-api:latest ..."
docker pull "$DOCKER_USER/penux-ldap-api:latest"
ok "API image pulled"

info "Pulling $DOCKER_USER/penux-ldap-web:latest ..."
docker pull "$DOCKER_USER/penux-ldap-web:latest"
ok "Web image pulled"

# ── 3. Start services ────────────────────────────────────────────────────────
hdr "Step 3: Start Docker services"

COMPOSE_FILE="docker-compose-hub.yml"
[ -f "$COMPOSE_FILE" ] || fail "$COMPOSE_FILE not found"

info "Starting services..."
$COMPOSE -f "$COMPOSE_FILE" up -d

info "Waiting up to 90s for services to become healthy..."
TIMEOUT=90
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
  API_HEALTH=$(docker inspect --format='{{.State.Health.Status}}' penux-api 2>/dev/null || echo "starting")
  if [ "$API_HEALTH" = "healthy" ]; then
    break
  fi
  echo -n "."
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done
echo ""

if [ "$API_HEALTH" != "healthy" ]; then
  warn "API not yet healthy (status: $API_HEALTH). Continuing anyway..."
  warn "Check logs: $COMPOSE -f $COMPOSE_FILE logs api"
else
  ok "All services healthy"
fi

$COMPOSE -f "$COMPOSE_FILE" ps

# ── 4. Tunnel ────────────────────────────────────────────────────────────────
hdr "Step 4: Start tunnel"

WEB_LOG="/tmp/tunnel-web.log"
API_LOG="/tmp/tunnel-api.log"

if [ -n "$NGROK_AUTHTOKEN" ]; then
  # ── Method A: ngrok (permanent static domain) ──────────────────────────────
  info "Using ngrok (permanent URL)..."

  if ! command -v ngrok &>/dev/null; then
    info "Installing ngrok..."
    if command -v apt-get &>/dev/null; then
      curl -fsSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | \
        sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
      echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | \
        sudo tee /etc/apt/sources.list.d/ngrok.list
      sudo apt-get update -q && sudo apt-get install -y ngrok
    else
      # Alpine / other
      curl -fsSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | \
        sudo tar -xz -C /usr/local/bin
    fi
  fi
  ok "ngrok: $(ngrok --version)"

  ngrok config add-authtoken "$NGROK_AUTHTOKEN" >/dev/null
  ok "ngrok authtoken configured"

  pkill ngrok 2>/dev/null || true

  if [ -n "$NGROK_DOMAIN" ]; then
    info "Starting Web UI tunnel on $NGROK_DOMAIN ..."
    nohup ngrok http 3001 --domain="$NGROK_DOMAIN" >"$WEB_LOG" 2>&1 &
    WEB_PID=$!
    sleep 4

    info "Starting API tunnel on api.$NGROK_DOMAIN (if available) ..."
    nohup ngrok http 3000 >"$API_LOG" 2>&1 &
    API_PID=$!
    sleep 4

    echo ""
    ok "Tunnels started with static domain!"
    echo ""
    echo "  Web UI: https://$NGROK_DOMAIN"
    echo "  API:    check ngrok dashboard → https://dashboard.ngrok.com"
    echo ""
    echo "  To point ldap.penux.uk here:"
    echo "    In your DNS panel add CNAME:"
    echo "    ldap   CNAME   $NGROK_DOMAIN"
    echo "    api.ldap CNAME <api-ngrok-url>"
    echo ""
  else
    info "No NGROK_DOMAIN set — starting with random URL..."
    nohup ngrok http 3001 >"$WEB_LOG" 2>&1 &
    WEB_PID=$!
    sleep 5

    WEB_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | \
      python3 -c "import sys,json; t=json.load(sys.stdin)['tunnels']; print([x['public_url'] for x in t if 'https' in x['public_url']][0])" 2>/dev/null || echo "check dashboard")

    echo ""
    ok "Tunnel started!"
    echo "  Web UI: $WEB_URL"
    echo "  Get your free static domain at: https://dashboard.ngrok.com/domains"
    echo "  Then rerun with: NGROK_AUTHTOKEN=$NGROK_AUTHTOKEN NGROK_DOMAIN=<your-domain> $0 $DOCKER_USER"
    echo ""
  fi

else
  # ── Method B: localhost.run (zero signup, SSH) ─────────────────────────────
  info "No NGROK_AUTHTOKEN — using localhost.run (zero signup, SSH-based)"
  info "Starting Web UI tunnel on port 3001..."

  pkill -f "nokey@localhost.run" 2>/dev/null || true
  nohup ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R 80:localhost:3001 nokey@localhost.run >"$WEB_LOG" 2>&1 &
  WEB_PID=$!
  echo $WEB_PID > /tmp/tunnel-web.pid

  info "Starting API tunnel on port 3000..."
  nohup ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -R 80:localhost:3000 nokey@localhost.run >"$API_LOG" 2>&1 &
  API_PID=$!
  echo $API_PID > /tmp/tunnel-api.pid

  sleep 6

  WEB_URL=$(grep -oP 'https://[a-z0-9.-]+\.localhost\.run' "$WEB_LOG" 2>/dev/null | head -1 || echo "")
  API_URL=$(grep -oP 'https://[a-z0-9.-]+\.localhost\.run' "$API_LOG" 2>/dev/null | head -1 || echo "")

  echo ""
  ok "Tunnels started via localhost.run!"
  echo ""
  if [ -n "$WEB_URL" ]; then
    echo "  Web UI: $WEB_URL"
  else
    echo "  Web UI: check $WEB_LOG"
  fi
  if [ -n "$API_URL" ]; then
    echo "  API:    $API_URL"
  else
    echo "  API:    check $API_LOG"
  fi
  echo ""
  echo "  NOTE: These URLs change every session."
  echo "  For a permanent free URL use ngrok:"
  echo "    1. Sign up (email only): https://ngrok.com"
  echo "    2. Get token: https://dashboard.ngrok.com/get-started/your-authtoken"
  echo "    3. Get free static domain: https://dashboard.ngrok.com/domains"
  echo "    4. Rerun:"
  echo "       NGROK_AUTHTOKEN=<token> NGROK_DOMAIN=<your-domain>.ngrok-free.app \\"
  echo "       $0 $DOCKER_USER"
  echo ""
fi

# ── 5. Commands ──────────────────────────────────────────────────────────────
hdr "Done!"
echo ""
echo "  Service logs : $COMPOSE -f $COMPOSE_FILE logs -f"
echo "  Stop all     : $COMPOSE -f $COMPOSE_FILE down"
echo "  Tunnel logs  : tail -f $WEB_LOG"
echo "  API health   : curl http://localhost:3000/api/health"
echo ""
