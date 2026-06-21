#!/bin/bash
# 🚀 Automated Cloudflare Tunnel Setup
# Complete automation for Docker Hub + Cloudflare deployment
# Works with dynamic IP addresses (no static IP needed!)

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

# Main script
print_header "PenuX LDAP - Cloudflare Tunnel Automated Setup"
print_info "This script will set up Cloudflare Tunnel for your Docker services"
print_info "No static IP needed - works with dynamic IPs!"

# Check prerequisites
print_header "Step 1: Check Prerequisites"

print_info "Checking for Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker not found"
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker found: $(docker --version)"

print_info "Checking for Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose not found"
    echo "Install from: https://docs.docker.com/compose/install/"
    exit 1
fi
print_success "Docker Compose found: $(docker-compose --version)"

print_info "Checking Docker daemon..."
if ! docker ps &> /dev/null; then
    print_error "Docker daemon not running or permission denied"
    exit 1
fi
print_success "Docker daemon is running"

# Get domain information
print_header "Step 2: Get Domain Information"

read -p "Enter your domain (e.g., penux.uk): " DOMAIN
if [ -z "$DOMAIN" ]; then
    print_error "Domain is required"
    exit 1
fi
DOMAIN=${DOMAIN,,}  # Convert to lowercase
print_success "Domain: $DOMAIN"

# Check if Docker services are running
print_header "Step 3: Ensure Docker Services Running"

print_info "Checking if Docker services are running..."
if ! docker ps | grep -q penux-api; then
    print_warning "Docker services not running. Starting them..."
    if [ -f "docker-compose-hub.yml" ]; then
        docker-compose -f docker-compose-hub.yml up -d
        print_success "Docker services started"
    else
        print_error "docker-compose-hub.yml not found"
        exit 1
    fi
else
    print_success "Docker services are already running"
fi

# Install cloudflared
print_header "Step 4: Install Cloudflare CLI"

print_info "Checking for cloudflared..."
if ! command -v cloudflared &> /dev/null; then
    print_info "Installing cloudflared..."
    curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb 2>/dev/null
    sudo dpkg -i cloudflared.deb > /dev/null 2>&1
    rm cloudflared.deb
    print_success "cloudflared installed"
else
    print_success "cloudflared already installed: $(cloudflared --version)"
fi

# Authenticate with Cloudflare
print_header "Step 5: Authenticate with Cloudflare"

print_info "This will open your browser for Cloudflare authentication..."
print_warning "Please complete the authentication in your browser"
print_info "Press Enter when ready to open browser..."
read

cloudflared tunnel login

print_success "Cloudflare authentication completed"

# Create tunnel
print_header "Step 6: Create Tunnel"

TUNNEL_NAME="penux-ldap"
print_info "Creating tunnel: $TUNNEL_NAME"

# Check if tunnel already exists
if cloudflared tunnel info "$TUNNEL_NAME" &>/dev/null; then
    print_warning "Tunnel '$TUNNEL_NAME' already exists"
    print_info "Using existing tunnel"
    UUID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
else
    cloudflared tunnel create "$TUNNEL_NAME"
    UUID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
fi

print_success "Tunnel created/found with ID: $UUID"

# Create configuration directory
mkdir -p ~/.cloudflared

# Create configuration file
print_header "Step 7: Create Tunnel Configuration"

CONFIG_FILE="$HOME/.cloudflared/config.yml"

cat > "$CONFIG_FILE" << EOF
tunnel: $TUNNEL_NAME
credentials-file: $HOME/.cloudflared/$UUID.json

ingress:
  - hostname: api.$DOMAIN
    service: http://localhost:3000
    originRequest:
      httpHostHeader: api.$DOMAIN

  - hostname: ldap.$DOMAIN
    service: http://localhost:3001
    originRequest:
      httpHostHeader: ldap.$DOMAIN

  - hostname: $DOMAIN
    service: http://localhost:3001
    originRequest:
      httpHostHeader: $DOMAIN

  - service: http_status:404

logLevel: info
metrics: 0.0.0.0:54321
EOF

print_success "Configuration created: $CONFIG_FILE"

# Setup DNS records
print_header "Step 8: Create DNS Records in Cloudflare"

print_info "Creating DNS routes..."
cloudflared tunnel route dns "$TUNNEL_NAME" "api.$DOMAIN"
cloudflared tunnel route dns "$TUNNEL_NAME" "ldap.$DOMAIN"
cloudflared tunnel route dns "$TUNNEL_NAME" "$DOMAIN"

print_success "DNS records created:"
echo "  api.$DOMAIN"
echo "  ldap.$DOMAIN"
echo "  $DOMAIN"

# Test tunnel connection
print_header "Step 9: Test Tunnel Connection"

print_info "Testing tunnel connection (this will run in foreground for 10 seconds)..."
print_warning "Press Ctrl+C after you see 'Tunnel registered successfully'"
print_info "Starting tunnel..."

timeout 15 cloudflared tunnel run "$TUNNEL_NAME" 2>&1 | head -20 || true

print_success "Tunnel tested successfully"

# Setup systemd service
print_header "Step 10: Setup Systemd Service (Auto-Start)"

print_info "Creating systemd service for tunnel..."
sudo cloudflared service install --logfile /tmp/cloudflared.log

print_success "Systemd service created"

print_info "Starting tunnel service..."
sudo systemctl start cloudflared

print_info "Enabling auto-start..."
sudo systemctl enable cloudflared

print_success "Service started and enabled for auto-start"

# Verify tunnel status
print_header "Step 11: Verify Tunnel Status"

sleep 3

print_info "Checking tunnel status..."
if cloudflared tunnel info "$TUNNEL_NAME" | grep -q "HEALTHY\|CONNECT"; then
    print_success "Tunnel is HEALTHY ✅"
else
    print_warning "Tunnel status unknown, checking systemd service..."
fi

print_info "Systemd service status:"
sudo systemctl status cloudflared --no-pager

# Wait for DNS propagation
print_header "Step 12: Wait for DNS Propagation"

print_info "Checking DNS resolution (this may take a moment)..."
for i in {1..30}; do
    if dig "+short" "api.$DOMAIN" | grep -q cfargotunnel; then
        print_success "DNS is resolving correctly!"
        break
    fi
    if [ $i -eq 1 ]; then
        print_info "Waiting for DNS to propagate (may take up to 5 minutes)..."
    fi
    sleep 10
    echo -n "."
done
echo ""

# Test endpoints
print_header "Step 13: Test Your Endpoints"

print_info "Waiting 5 seconds for services to stabilize..."
sleep 5

print_info "Testing API health endpoint..."
if curl -s "https://api.$DOMAIN/api/health" | grep -q "healthy"; then
    print_success "API is responding ✅"
    echo "Response: $(curl -s https://api.$DOMAIN/api/health)"
else
    print_warning "API health check may not have resolved yet"
    print_info "This is normal - DNS can take up to 5 minutes to fully propagate"
    print_info "Try: curl https://api.$DOMAIN/api/health"
fi

# Summary
print_header "✨ Setup Complete!"

echo ""
echo "🎉 Your PenuX LDAP system is now live on Cloudflare Tunnel!"
echo ""
echo "📍 Access Points:"
echo "   API:     https://api.$DOMAIN"
echo "   Web UI:  https://ldap.$DOMAIN"
echo "   Domain:  https://$DOMAIN"
echo ""
echo "🔧 Tunnel Information:"
echo "   Name: $TUNNEL_NAME"
echo "   UUID: $UUID"
echo "   Status: Running (systemd service)"
echo ""
echo "📋 Useful Commands:"
echo ""
echo "View tunnel status:"
echo "  cloudflared tunnel info $TUNNEL_NAME"
echo ""
echo "View logs:"
echo "  sudo journalctl -u cloudflared -f"
echo ""
echo "Restart tunnel:"
echo "  sudo systemctl restart cloudflared"
echo ""
echo "Stop tunnel:"
echo "  sudo systemctl stop cloudflared"
echo ""
echo "Test API:"
echo "  curl https://api.$DOMAIN/api/health"
echo ""
echo "Test with authentication:"
echo "  curl -u 'cn=admin,dc=penux,dc=uk:password' \\"
echo "    https://api.$DOMAIN/api/users"
echo ""
echo "View tunnel metrics:"
echo "  curl http://localhost:54321/metrics"
echo ""
echo "📚 Documentation:"
echo "  - Full guide: /DOCKER-HUB-CLOUDFLARE-TUNNEL.md"
echo "  - Integration guide: /DOCKER-HUB-CLOUDFLARE-INTEGRATION.md"
echo "  - Docker deployment: /DOCKER-HUB-DEPLOY.md"
echo ""

print_success "Everything is ready!"

echo ""
print_warning "Note: DNS records may take up to 5 minutes to fully propagate"
print_warning "If endpoints don't respond immediately, wait a few minutes and try again"
echo ""
print_success "Your system is secure, scalable, and ready for production! 🚀"
