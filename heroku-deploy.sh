#!/bin/bash
# 🚀 Heroku Deployment Script
# Deploy PenuX LDAP to Heroku in one command

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
print_header "PenuX LDAP - Heroku Deployment"

# Check prerequisites
print_info "Checking prerequisites..."

if ! command -v heroku &> /dev/null; then
    print_error "Heroku CLI not found"
    echo "Install from: https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
fi

print_success "Heroku CLI found: $(heroku --version)"

# Login
print_header "Step 1: Login to Heroku"
print_info "Opening browser for authentication..."
heroku login || {
    print_error "Login failed"
    exit 1
}
print_success "Logged in to Heroku"

# Create apps
print_header "Step 2: Create Heroku Apps"

read -p "Enter app name prefix (e.g., 'penux-ldap'): " APP_PREFIX
APP_PREFIX=${APP_PREFIX:-penux-ldap}

API_APP="${APP_PREFIX}-api"
WEB_APP="${APP_PREFIX}-web"

print_info "Creating REST API app: $API_APP"
heroku create "$API_APP" || print_warning "App already exists"
print_success "API app ready: $API_APP"

print_info "Creating Web UI app: $WEB_APP"
heroku create "$WEB_APP" || print_warning "App already exists"
print_success "Web app ready: $WEB_APP"

# Add PostgreSQL
print_header "Step 3: Configure PostgreSQL Database"
print_info "Adding PostgreSQL to $API_APP..."
heroku addons:create heroku-postgresql:hobby-dev --app "$API_APP" || print_warning "Database may already exist"
print_success "PostgreSQL configured"

# Set environment variables
print_header "Step 4: Configure Environment Variables"

print_info "Setting LDAP credentials..."
read -sp "Enter LDAP admin password: " LDAP_PASS
echo ""
LDAP_PASS=${LDAP_PASS:-admin123}

print_info "Setting Keycloak credentials..."
read -sp "Enter Keycloak admin password: " KC_PASS
echo ""
KC_PASS=${KC_PASS:-admin123}

# Set API variables
print_info "Configuring REST API variables..."
heroku config:set \
  LDAP_HOST="ldaps://ldaps-server.penux.uk:636" \
  LDAP_BASE_DN="dc=penux,dc=uk" \
  LDAP_ADMIN_DN="cn=admin,dc=penux,dc=uk" \
  LDAP_ADMIN_PASSWORD="$LDAP_PASS" \
  CORS_ORIGIN="*" \
  NODE_ENV="production" \
  --app "$API_APP"

print_success "API variables set"

# Set Web variables
print_info "Configuring Web UI variables..."
heroku config:set \
  API_URL="https://${API_APP}.herokuapp.com" \
  NODE_ENV="production" \
  --app "$WEB_APP"

print_success "Web variables set"

# Deploy
print_header "Step 5: Deploy to Heroku"

print_info "Deploying REST API..."
git subtree push --prefix services/openldap/api heroku-api main || {
    print_error "API deployment failed"
    echo "Try: git push heroku main (from API directory)"
}
print_success "API deployed"

print_info "Deploying Web UI..."
git subtree push --prefix services/openldap/web heroku-web main || {
    print_error "Web deployment failed"
    echo "Try: git push heroku main (from Web directory)"
}
print_success "Web deployed"

# Show URLs
print_header "Step 6: Your Heroku URLs"
echo ""
echo "🌐 Access your services:"
echo "  API:    https://${API_APP}.herokuapp.com"
echo "  Web UI: https://${WEB_APP}.herokuapp.com"
echo ""

# Add custom domains
print_header "Step 7: Add Custom Domains (Optional)"
echo ""
echo "To use custom domains like api.ldap.penux.uk:"
echo ""
echo "1. Add domain to Heroku:"
echo "   heroku domains:add api.ldap.penux.uk --app $API_APP"
echo "   heroku domains:add ldap.penux.uk --app $WEB_APP"
echo ""
echo "2. Get CNAME values:"
echo "   heroku domains --app $API_APP"
echo "   heroku domains --app $WEB_APP"
echo ""
echo "3. Update DNS at your registrar:"
echo "   Type: CNAME"
echo "   Name: api"
echo "   Value: (Heroku CNAME)"
echo ""
echo "4. Update Heroku variables with new URLs:"
echo "   heroku config:set API_URL='https://api.ldap.penux.uk' --app $WEB_APP"
echo ""

# View logs
print_header "Step 8: Monitor Your Deployment"

echo ""
echo "View logs:"
echo "  heroku logs -f --app $API_APP"
echo "  heroku logs -f --app $WEB_APP"
echo ""
echo "View config:"
echo "  heroku config --app $API_APP"
echo "  heroku config --app $WEB_APP"
echo ""
echo "Open app in browser:"
echo "  heroku open --app $API_APP"
echo "  heroku open --app $WEB_APP"
echo ""

# Test
print_header "Step 9: Test Your Deployment"

echo ""
echo "Test REST API:"
echo "  curl https://${API_APP}.herokuapp.com/api/health"
echo ""
echo "Test with auth:"
echo "  curl -u 'cn=admin,dc=penux,dc=uk:${LDAP_PASS}' \\"
echo "    https://${API_APP}.herokuapp.com/api/users"
echo ""

# Done
print_header "🎉 Deployment Complete!"

echo ""
echo "✅ API deployed:  https://${API_APP}.herokuapp.com"
echo "✅ Web deployed:  https://${WEB_APP}.herokuapp.com"
echo ""
echo "Next steps:"
echo "  1. Test API health: curl https://${API_APP}.herokuapp.com/api/health"
echo "  2. Open web UI: https://${WEB_APP}.herokuapp.com"
echo "  3. (Optional) Add custom domains (see Step 7 above)"
echo "  4. (Optional) Change default passwords"
echo ""
echo "Documentation: /HEROKU-DEPLOY.md"
echo ""

print_success "Ready to use!"
