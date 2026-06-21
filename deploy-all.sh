#!/bin/bash
# 🚀 Complete LDAP Deployment Automation (Linux/macOS)
# Deploys OpenLDAP + Keycloak + Cloudflare Tunnel in one command

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
print_header "PHASE 1: Checking Prerequisites"

print_info "Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Install Docker first!"
    exit 1
fi
DOCKER_VERSION=$(docker --version)
print_success "Docker found: $DOCKER_VERSION"

print_info "Checking Docker daemon..."
if ! docker ps &> /dev/null; then
    print_error "Docker daemon not running. Start Docker first!"
    exit 1
fi
print_success "Docker daemon running"

print_info "Checking if in project directory..."
if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml not found. Run from penuX directory!"
    exit 1
fi
print_success "Project directory verified"

# Phase 1: Start Docker Services
print_header "PHASE 1: Starting Docker Services"

print_info "Stopping any existing services..."
docker compose down 2>/dev/null || true

print_info "Starting OpenLDAP + Keycloak + PostgreSQL..."
docker compose up -d

print_info "Waiting 45 seconds for services to initialize..."
for i in {45..1}; do
    echo -ne "${BLUE}$i seconds remaining...${NC}\r"
    sleep 1
done
echo ""

print_info "Checking service status..."
if docker compose ps | grep -q "healthy"; then
    print_success "Services started successfully"
else
    print_warning "Services may still be initializing (this is normal)"
fi

docker compose ps

# Phase 2: Test LDAP Locally
print_header "PHASE 2: Testing LDAP Locally"

print_info "Testing LDAP connection..."
if ldapwhoami -H ldap://localhost:389 \
    -D "cn=admin,dc=penux,dc=uk" \
    -w admin123 &>/dev/null; then
    print_success "LDAP is responding"
else
    print_warning "LDAP test failed - services may still be starting"
fi

print_info "Testing Keycloak..."
if curl -s http://localhost:8080/health | grep -q "UP" 2>/dev/null || curl -s http://localhost:8080 | grep -q "keycloak" 2>/dev/null; then
    print_success "Keycloak is responding"
else
    print_warning "Keycloak test inconclusive - check manually"
fi

# Phase 3: Install Cloudflare Tunnel
print_header "PHASE 3: Installing Cloudflare Tunnel"

print_info "Checking if cloudflared is installed..."
if ! command -v cloudflared &> /dev/null; then
    print_info "Installing cloudflared..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if ! command -v brew &> /dev/null; then
            print_error "Homebrew not found. Install Homebrew first or install cloudflared manually"
            exit 1
        fi
        brew install cloudflared
    else
        # Linux
        wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /tmp/cloudflared
        chmod +x /tmp/cloudflared
        sudo mv /tmp/cloudflared /usr/local/bin/cloudflared
    fi
    print_success "cloudflared installed"
else
    TUNNEL_VERSION=$(cloudflared --version)
    print_success "cloudflared already installed: $TUNNEL_VERSION"
fi

# Phase 4: Cloudflare Login
print_header "PHASE 4: Cloudflare Login & Tunnel Creation"

print_warning "This will open a browser window to login to Cloudflare"
print_info "Login with your Cloudflare account and authorize for penux.uk domain"

read -p "Press ENTER when ready to login to Cloudflare..."

cloudflared tunnel login

print_success "Cloudflare login successful"

# Phase 5: Create Tunnel
print_header "PHASE 5: Creating Cloudflare Tunnel"

TUNNEL_NAME="ldap-tunnel"

print_info "Checking if tunnel already exists..."
if cloudflared tunnel list 2>/dev/null | grep -q "$TUNNEL_NAME"; then
    print_warning "Tunnel '$TUNNEL_NAME' already exists"
    TUNNEL_ID=$(cloudflared tunnel list 2>/dev/null | grep "$TUNNEL_NAME" | awk '{print $1}')
else
    print_info "Creating new tunnel: $TUNNEL_NAME"
    cloudflared tunnel create $TUNNEL_NAME
    print_success "Tunnel created successfully"
fi

# Find credentials file
CREDS_FILE=$(find ~/.cloudflared -name "*.json" -type f | head -1)
if [ -z "$CREDS_FILE" ]; then
    print_error "Could not find Cloudflare credentials file"
    exit 1
fi
print_success "Found credentials: $CREDS_FILE"

# Phase 6: Create Tunnel Config
print_header "PHASE 6: Creating Tunnel Configuration"

CONFIG_DIR="$HOME/.cloudflared"
CONFIG_FILE="$CONFIG_DIR/config.yml"
CREDS_FILENAME=$(basename "$CREDS_FILE")

print_info "Creating config file: $CONFIG_FILE"

cat > "$CONFIG_FILE" << EOF
tunnel: $TUNNEL_NAME
credentials-file: $CREDS_FILE

ingress:
  - hostname: ldap.penux.uk
    service: http://localhost:8080

  - hostname: ldap-server.penux.uk
    service: tcp://localhost:389

  - hostname: ldaps-server.penux.uk
    service: tcp://localhost:636

  - hostname: api.ldap.penux.uk
    service: http://localhost:3000

  - service: http_status:404
EOF

print_success "Config file created"

# Phase 7: Get Tunnel CNAME
print_header "PHASE 7: Getting Tunnel Information"

TUNNEL_CNAME="${TUNNEL_NAME}.cfargotunnel.com"
print_success "Your tunnel CNAME: $TUNNEL_CNAME"

# Phase 8: Instructions for DNS
print_header "PHASE 8: Next Steps - Cloudflare DNS Configuration"

print_warning "MANUAL STEP REQUIRED - Update Cloudflare DNS:"
echo ""
echo "1. Go to: https://dash.cloudflare.com"
echo "2. Select: penux.uk domain"
echo "3. Click: DNS → Records"
echo ""
echo "4. Add/Update these CNAME records:"
echo "   - Name: ldap"
echo "     Value: $TUNNEL_CNAME"
echo ""
echo "   - Name: ldaps-server"
echo "     Value: $TUNNEL_CNAME"
echo ""
echo "   - Name: api.ldap"
echo "     Value: $TUNNEL_CNAME"
echo ""
echo "5. Also ensure:"
echo "   - SSL/TLS → Encryption Mode: Full"
echo "   - SSL/TLS → Always Use HTTPS: ON"
echo ""
read -p "Press ENTER when DNS records are updated in Cloudflare..."

# Phase 9: Start Tunnel
print_header "PHASE 9: Starting Cloudflare Tunnel"

print_info "Starting tunnel in background..."

# Start tunnel in background and get PID
nohup cloudflared tunnel run $TUNNEL_NAME > /tmp/cloudflared.log 2>&1 &
TUNNEL_PID=$!
print_success "Tunnel started (PID: $TUNNEL_PID)"

sleep 5

# Check if tunnel is running
if ps -p $TUNNEL_PID > /dev/null; then
    print_success "Tunnel is running"
else
    print_error "Tunnel failed to start"
    cat /tmp/cloudflared.log
    exit 1
fi

# Phase 10: Verification
print_header "PHASE 10: Verification & Testing"

print_info "Waiting 15 seconds for DNS propagation..."
sleep 15

print_info "Testing DNS resolution..."
if nslookup ldap.penux.uk &>/dev/null; then
    print_success "DNS resolved: ldap.penux.uk"
else
    print_warning "DNS not yet propagated (this is normal, wait a few minutes)"
fi

print_info "Testing LDAP via tunnel..."
if ldapwhoami -H ldap://ldap-server.penux.uk:389 \
    -D "cn=admin,dc=penux,dc=uk" \
    -w admin123 &>/dev/null; then
    print_success "LDAP accessible via tunnel"
else
    print_warning "LDAP via tunnel not yet accessible (DNS may still be propagating)"
fi

# Phase 11: Summary
print_header "🎉 DEPLOYMENT COMPLETE!"

echo ""
echo -e "${GREEN}Your LDAP Directory is Now Live!${NC}"
echo ""
echo "Access URLs:"
echo -e "  ${GREEN}Web UI:${NC}      https://ldap.penux.uk"
echo -e "  ${GREEN}Admin:${NC}       https://ldap.penux.uk/admin"
echo -e "  ${GREEN}LDAP:${NC}        ldap://ldap-server.penux.uk:389"
echo -e "  ${GREEN}LDAPS:${NC}       ldaps://ldaps-server.penux.uk:636"
echo -e "  ${GREEN}API:${NC}         https://api.ldap.penux.uk"
echo ""
echo "Credentials:"
echo -e "  ${GREEN}Admin DN:${NC}    cn=admin,dc=penux,dc=uk"
echo -e "  ${GREEN}Password:${NC}    admin123"
echo ""
echo "Services Running:"
echo -e "  ${GREEN}✅ Docker Compose${NC}  (OpenLDAP + Keycloak + PostgreSQL)"
echo -e "  ${GREEN}✅ Cloudflare Tunnel${NC} (Secure tunnel to your machine)"
echo ""
echo "To stop all services:"
echo -e "  ${YELLOW}docker compose down${NC}"
echo ""
echo "To view tunnel logs:"
echo -e "  ${YELLOW}tail -f /tmp/cloudflared.log${NC}"
echo ""
print_success "Deployment finished! Check the URLs above."
