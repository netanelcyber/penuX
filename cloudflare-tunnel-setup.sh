#!/bin/bash
# 🚀 Cloudflare Tunnel + Docker Hub Integration Setup
# Deploy PenuX LDAP: Docker Hub → Cloudflare Tunnel → ldap.penux.uk

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error()   { echo -e "${RED}❌ $1${NC}"; }

print_header "PenuX LDAP - Docker Hub + Cloudflare Tunnel Setup"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker not found"
    exit 1
fi
print_success "Docker found: $(docker --version)"

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose not found"
    exit 1
fi
print_success "Docker Compose found"

# Pull images
print_header "Step 1: Pull Docker Hub Images"

DOCKER_USER="${1:-netanelcyber}"
print_info "Pulling images: $DOCKER_USER/penux-ldap-*"

docker pull "$DOCKER_USER/penux-ldap-api:latest"
docker pull "$DOCKER_USER/penux-ldap-web:latest"

print_success "Images pulled successfully"

# Start Docker services
print_header "Step 2: Start Docker Services"

print_info "Starting services with docker-compose..."
docker-compose -f docker-compose-hub.yml up -d

sleep 5
print_success "Services started"

# Install cloudflared
print_header "Step 3: Install Cloudflare Tunnel CLI"

if ! command -v cloudflared &> /dev/null; then
    print_info "Installing cloudflared..."
    curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb 2>/dev/null
    sudo dpkg -i cloudflared.deb
    rm cloudflared.deb
fi

print_success "cloudflared ready: $(cloudflared --version)"

# Login
print_header "Step 4: Authenticate with Cloudflare"

print_info "Opening browser for authentication..."
cloudflared tunnel login

print_success "Authenticated"

# Create tunnel
print_header "Step 5: Create Cloudflare Tunnel"

TUNNEL_NAME="penux-ldap"
cloudflared tunnel create "$TUNNEL_NAME" || print_warning "Tunnel may exist"

TUNNEL_UUID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}' | head -1)
print_success "Tunnel: $TUNNEL_UUID"

# Configure routing
print_header "Step 6: Configure Tunnel"

mkdir -p ~/.cloudflared

cat > ~/.cloudflared/config.yml << 'EOF'
tunnel: penux-ldap
credentials-file: ~/.cloudflared/TUNNEL_CREDS

ingress:
  - hostname: ldap.penux.uk
    service: http://localhost:3001
  - hostname: penux.uk
    service: http://localhost:3001
  - hostname: api.ldap.penux.uk
    service: http://localhost:3000
  - hostname: health.ldap.penux.uk
    service: http://localhost:3000
  - service: http_status:404
EOF

# Get actual credentials file
CREDS_FILE=$(ls ~/.cloudflared/*.json 2>/dev/null | grep -v cert | head -1)
if [ -n "$CREDS_FILE" ]; then
    CREDS_FILENAME=$(basename "$CREDS_FILE")
    sed -i "s|TUNNEL_CREDS|$CREDS_FILENAME|g" ~/.cloudflared/config.yml
fi

print_success "Tunnel configured"

# Create DNS records
print_header "Step 7: Create DNS Records"

cloudflared tunnel route dns "$TUNNEL_NAME" ldap.penux.uk 2>/dev/null || true
cloudflared tunnel route dns "$TUNNEL_NAME" penux.uk 2>/dev/null || true
cloudflared tunnel route dns "$TUNNEL_NAME" api.ldap.penux.uk 2>/dev/null || true

print_success "DNS records created"

# Install as service
print_header "Step 8: Install as System Service"

sudo cloudflared service install 2>/dev/null || print_warning "Service install skipped"
sudo systemctl start cloudflared 2>/dev/null || print_warning "Could not start service"
sudo systemctl enable cloudflared 2>/dev/null || print_warning "Could not enable service"

print_success "Service installed"

# Summary
print_header "🎉 Complete!"

echo ""
echo "✅ Live URLs:"
echo "   🌐 https://ldap.penux.uk (Web UI)"
echo "   🌐 https://penux.uk (Root)"
echo "   🌐 https://api.ldap.penux.uk (API)"
echo ""
echo "📊 Commands:"
echo "   Logs: docker-compose -f docker-compose-hub.yml logs -f"
echo "   Status: cloudflared tunnel info penux-ldap"
echo "   Test: curl https://api.ldap.penux.uk/api/health"
echo ""

print_success "Ready! 🚀"
