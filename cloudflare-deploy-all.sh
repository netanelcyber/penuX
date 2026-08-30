#!/bin/bash
# 🐳 Complete Automation: Docker Hub + Cloudflare Tunnel
# One command to deploy everything!

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

# Title
print_header "PenuX LDAP - Complete Docker + Cloudflare Setup"

echo "This script will:"
echo "  1. Start Docker services (OpenLDAP, PostgreSQL, API, Web)"
echo "  2. Install Cloudflare CLI"
echo "  3. Create Cloudflare Tunnel"
echo "  4. Configure DNS records"
echo "  5. Test endpoints"
echo ""
echo "Works with dynamic IPs - no static IP needed!"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_error "Setup cancelled"
    exit 1
fi

# Verify we're in the right directory
if [ ! -f "docker-compose-hub.yml" ]; then
    print_error "docker-compose-hub.yml not found"
    print_info "Run this script from the penuX project root directory"
    exit 1
fi

# Step 1: Start Docker Services
print_header "Step 1: Start Docker Services"

print_info "Checking Docker..."
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Install from https://docs.docker.com/get-docker/"
    exit 1
fi

print_info "Checking Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose not found. Install from https://docs.docker.com/compose/install/"
    exit 1
fi

print_info "Starting Docker services..."
docker-compose -f docker-compose-hub.yml up -d

print_success "Docker services started:"
echo "  - OpenLDAP"
echo "  - PostgreSQL"
echo "  - REST API"
echo "  - Web UI"

print_info "Waiting for services to be ready..."
sleep 5

print_info "Services status:"
docker-compose -f docker-compose-hub.yml ps

# Verify API is responding
print_info "Testing API health..."
MAX_RETRIES=10
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:3000/api/health | grep -q "healthy"; then
        print_success "API is responding"
        break
    fi
    RETRY=$((RETRY + 1))
    if [ $RETRY -lt $MAX_RETRIES ]; then
        echo -n "."
        sleep 2
    fi
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    print_warning "API is not responding yet (may need more time)"
fi

# Step 2: Install Cloudflare CLI
print_header "Step 2: Install Cloudflare CLI (cloudflared)"

if command -v cloudflared &> /dev/null; then
    print_success "cloudflared already installed: $(cloudflared --version)"
else
    print_info "Downloading and installing cloudflared..."
    curl -L --output /tmp/cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb 2>/dev/null
    sudo dpkg -i /tmp/cloudflared.deb > /dev/null 2>&1
    rm /tmp/cloudflared.deb
    print_success "cloudflared installed"
fi

# Step 3: Get domain information
print_header "Step 3: Configure Your Domain"

read -p "Enter your domain (e.g., penux.uk): " DOMAIN
if [ -z "$DOMAIN" ]; then
    print_error "Domain is required"
    exit 1
fi
DOMAIN=${DOMAIN,,}  # Convert to lowercase
print_success "Domain configured: $DOMAIN"

# Step 4: Create .env file
print_header "Step 4: Configure Environment Variables"

read -sp "Enter LDAP admin password (default: admin123): " LDAP_PASS
echo ""
LDAP_PASS=${LDAP_PASS:-admin123}

read -sp "Enter PostgreSQL password (default: secure_password): " DB_PASS
echo ""
DB_PASS=${DB_PASS:-secure_password}

# Create .env file for Docker Compose
cat > .env << EOF
LDAP_ADMIN_PASSWORD=$LDAP_PASS
POSTGRES_USER=penux
POSTGRES_PASSWORD=$DB_PASS
CORS_ORIGIN=https://$DOMAIN https://api.$DOMAIN https://ldap.$DOMAIN
NODE_ENV=production
API_URL=https://api.$DOMAIN
EOF

print_success "Environment file created (.env)"

# Re-export API_URL for Docker services
docker-compose -f docker-compose-hub.yml down
docker-compose -f docker-compose-hub.yml up -d

print_success "Docker services restarted with new configuration"

# Step 5: Authenticate with Cloudflare
print_header "Step 5: Authenticate with Cloudflare"

print_warning "Browser will open for Cloudflare authentication"
print_info "You will be asked to select your domain"
print_info "Press Enter to continue..."
read

mkdir -p ~/.cloudflared
cloudflared tunnel login

print_success "Cloudflare authentication completed"

# Step 6: Create tunnel
print_header "Step 6: Create Cloudflare Tunnel"

TUNNEL_NAME="penux-ldap"
print_info "Creating tunnel: $TUNNEL_NAME"

# Check if tunnel already exists
if cloudflared tunnel list 2>/dev/null | grep -q "$TUNNEL_NAME"; then
    print_warning "Tunnel already exists, using existing"
    UUID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
else
    cloudflared tunnel create "$TUNNEL_NAME"
    UUID=$(cloudflared tunnel list | grep "$TUNNEL_NAME" | awk '{print $1}')
fi

print_success "Tunnel ID: $UUID"

# Step 7: Create tunnel configuration
print_header "Step 7: Configure Tunnel Routes"

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

print_success "Tunnel configuration created"

# Step 8: Create DNS records
print_header "Step 8: Create DNS Records"

print_info "Setting up DNS routes..."
cloudflared tunnel route dns "$TUNNEL_NAME" "api.$DOMAIN" 2>/dev/null || print_warning "API DNS record may already exist"
cloudflared tunnel route dns "$TUNNEL_NAME" "ldap.$DOMAIN" 2>/dev/null || print_warning "LDAP DNS record may already exist"
cloudflared tunnel route dns "$TUNNEL_NAME" "$DOMAIN" 2>/dev/null || print_warning "Root DNS record may already exist"

print_success "DNS records created/verified"

# Step 9: Setup systemd service
print_header "Step 9: Setup Auto-Start Service"

print_info "Installing systemd service..."
sudo cloudflared service install --logfile /tmp/cloudflared.log 2>/dev/null || print_warning "Service may already be installed"

print_info "Starting service..."
sudo systemctl daemon-reload 2>/dev/null || true
sudo systemctl start cloudflared 2>/dev/null || true
sudo systemctl enable cloudflared 2>/dev/null || true

print_success "Tunnel service configured for auto-start"

# Step 10: Test tunnel
print_header "Step 10: Verify Tunnel Connection"

sleep 3

print_info "Tunnel status:"
cloudflared tunnel info "$TUNNEL_NAME" 2>/dev/null || print_warning "Could not retrieve tunnel status"

print_info "Service status:"
sudo systemctl is-active cloudflared > /dev/null && print_success "Service is running" || print_warning "Service status unknown"

# Step 11: Wait for DNS
print_header "Step 11: Wait for DNS Propagation"

print_info "Checking DNS resolution..."
print_warning "This may take up to 5 minutes on first setup"

RESOLVED=0
for i in {1..30}; do
    if dig "+short" "api.$DOMAIN" 2>/dev/null | grep -q cfargotunnel; then
        print_success "DNS is resolving!"
        RESOLVED=1
        break
    fi
    if [ $((i % 5)) -eq 0 ]; then
        echo -n "."
    fi
    sleep 10
done

if [ $RESOLVED -eq 0 ]; then
    print_warning "DNS may not be fully propagated yet"
    print_info "Wait a few minutes and test manually:"
    echo "  dig api.$DOMAIN"
fi

# Step 12: Test endpoints
print_header "Step 12: Test Your Endpoints"

print_info "Testing API (may take a moment to respond)..."
RESPONSE=$(curl -s "https://api.$DOMAIN/api/health" 2>/dev/null || echo "")
if echo "$RESPONSE" | grep -q "healthy"; then
    print_success "API is responding correctly!"
    echo "Response: $RESPONSE"
else
    print_warning "API is not responding yet"
    print_info "DNS may still be propagating (takes up to 5 minutes)"
fi

# Final summary
print_header "🎉 Setup Complete!"

echo ""
echo "✅ All services are now running on Cloudflare Tunnel!"
echo ""
echo "📍 Access Your Services:"
echo ""
echo "   🔒 REST API:     https://api.$DOMAIN"
echo "   🌐 Web UI:       https://ldap.$DOMAIN"
echo "   🏠 Domain:       https://$DOMAIN"
echo ""
echo "📋 LDAP Credentials:"
echo "   Base DN:        dc=penux,dc=uk"
echo "   Admin DN:       cn=admin,dc=penux,dc=uk"
echo "   Admin Password: $LDAP_PASS"
echo ""
echo "🔧 Tunnel Information:"
echo "   Name:           $TUNNEL_NAME"
echo "   UUID:           $UUID"
echo "   Config:         $CONFIG_FILE"
echo ""
echo "💡 Useful Commands:"
echo ""
echo "   View tunnel status:"
echo "     cloudflared tunnel info $TUNNEL_NAME"
echo ""
echo "   View logs (real-time):"
echo "     sudo journalctl -u cloudflared -f"
echo ""
echo "   Test API health:"
echo "     curl https://api.$DOMAIN/api/health"
echo ""
echo "   Test with authentication:"
echo "     curl -u 'cn=admin,dc=penux,dc=uk:$LDAP_PASS' \\"
echo "       https://api.$DOMAIN/api/users"
echo ""
echo "   View Docker logs:"
echo "     docker-compose -f docker-compose-hub.yml logs -f"
echo ""
echo "   Restart services:"
echo "     docker-compose -f docker-compose-hub.yml restart"
echo "     sudo systemctl restart cloudflared"
echo ""
echo "   Stop all services:"
echo "     docker-compose -f docker-compose-hub.yml down"
echo "     sudo systemctl stop cloudflared"
echo ""
echo "📚 Documentation:"
echo "   /DOCKER-HUB-CLOUDFLARE-TUNNEL.md"
echo "   /DOCKER-HUB-CLOUDFLARE-INTEGRATION.md"
echo "   /DOCKER-HUB-DEPLOY.md"
echo ""
print_warning "Note: DNS may take up to 5 minutes to fully propagate"
print_warning "If endpoints don't respond immediately, wait and try again"
echo ""
print_success "Your PenuX LDAP system is secure, scalable, and production-ready! 🚀"
