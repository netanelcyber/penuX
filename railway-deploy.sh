#!/bin/bash
# 🚂 Railway Deployment Script
# Deploy PenuX LDAP to Railway in one command

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
print_header "PenuX LDAP - Railway Deployment"

# Check prerequisites
print_info "Checking prerequisites..."

if ! command -v railway &> /dev/null; then
    print_error "Railway CLI not found"
    echo "Install it with: npm install -g @railway/cli"
    exit 1
fi

print_success "Railway CLI found: $(railway --version)"

# Login
print_header "Step 1: Login to Railway"
print_info "Opening browser for authentication..."
railway login || {
    print_error "Login failed"
    exit 1
}
print_success "Logged in to Railway"

# Create/select project
print_header "Step 2: Create or Select Project"
print_info "Creating new Railway project..."
railway init || {
    print_warning "Using existing project"
}
print_success "Project ready"

# Show project info
print_header "Step 3: Project Information"
railway status || true

# Set environment variables
print_header "Step 4: Configure Environment Variables"

print_info "Setting LDAP admin password..."
read -p "Enter LDAP admin password (default: admin123): " LDAP_PASS
LDAP_PASS=${LDAP_PASS:-admin123}
railway variables set LDAP_ADMIN_PASSWORD="$LDAP_PASS"
print_success "LDAP password set"

print_info "Setting Keycloak admin password..."
read -p "Enter Keycloak admin password (default: admin123): " KC_PASS
KC_PASS=${KC_PASS:-admin123}
railway variables set KEYCLOAK_ADMIN_PASSWORD="$KC_PASS"
print_success "Keycloak password set"

# Mandatory variables
railway variables set LDAP_ORGANISATION="PenuX"
railway variables set LDAP_DOMAIN="penux.uk"
railway variables set LDAP_BASE_DN="dc=penux,dc=uk"
railway variables set LDAP_ADMIN_DN="cn=admin,dc=penux,dc=uk"
railway variables set NODE_ENV="production"
railway variables set CORS_ORIGIN="*"

print_success "Environment variables configured"

# Show variables
print_header "Step 5: Review Configuration"
railway variables

# Deploy
print_header "Step 6: Deploy to Railway"
print_info "Starting deployment..."
railway up --detach

print_success "Deployment initiated!"

# Wait for services
print_info "Waiting for services to start (this may take a few minutes)..."
sleep 10

# Show status
print_header "Step 7: Deployment Status"
railway status

# Get URLs
print_header "Step 8: Your Railway URLs"
echo ""
echo "Railway will generate URLs like:"
echo "  Web UI:      https://web-xxxxx.railway.app"
echo "  API:         https://api-xxxxx.railway.app"
echo "  Keycloak:    https://keycloak-xxxxx.railway.app"
echo ""
print_info "View your actual URLs:"
echo "  1. Go to: https://railway.app/dashboard"
echo "  2. Click your project"
echo "  3. View each service's deployment URL"
echo ""

# Next steps
print_header "Step 9: Next Steps"
echo ""
echo "✅ Services deploying (5-10 minutes)"
echo ""
echo "After deployment:"
echo "  1. Add custom domains in Railway dashboard"
echo "  2. Update DNS records at your registrar"
echo "  3. Test: curl https://api-xxxxx.railway.app/api/health"
echo "  4. Read: /DEPLOYMENT-ALTERNATIVES.md for more details"
echo ""
echo "View logs:"
echo "  railway logs -f"
echo ""
echo "View status:"
echo "  railway status"
echo ""
echo "Open dashboard:"
echo "  railway open"
echo ""

print_success "Deployment started! Check Railway dashboard for progress."
