#!/bin/bash
# 🐳 Docker Hub Deployment Script
# Deploy PenuX LDAP using Docker and Docker Hub images

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
print_header "PenuX LDAP - Docker Hub Deployment"

# Check prerequisites
print_info "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker not found"
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose not found"
    echo "Install from: https://docs.docker.com/compose/install/"
    exit 1
fi

print_success "Docker found: $(docker --version)"
print_success "Docker Compose found: $(docker-compose --version)"

# Check Docker daemon
print_info "Checking Docker daemon..."
if ! docker ps &> /dev/null; then
    print_error "Docker daemon not running or permission denied"
    echo "Make sure Docker is running and you have permission to access it"
    echo "Try: sudo usermod -aG docker \$USER"
    exit 1
fi

print_success "Docker daemon is running"

# Get configuration
print_header "Step 1: Configure Environment"

print_info "Setting LDAP admin password..."
read -sp "Enter LDAP admin password (default: admin123): " LDAP_PASS
echo ""
LDAP_PASS=${LDAP_PASS:-admin123}

print_info "Setting PostgreSQL password..."
read -sp "Enter PostgreSQL password (default: secure_password): " DB_PASS
echo ""
DB_PASS=${DB_PASS:-secure_password}

print_info "Setting CORS origin..."
read -p "Enter CORS origin (default: *): " CORS
CORS=${CORS:-*}

print_info "Setting Node environment..."
read -p "Enter Node environment (default: production): " NODE_ENV
NODE_ENV=${NODE_ENV:-production}

# Create .env file
print_header "Step 2: Create Configuration File"

cat > .env << EOF
LDAP_ADMIN_PASSWORD=$LDAP_PASS
POSTGRES_USER=penux
POSTGRES_PASSWORD=$DB_PASS
CORS_ORIGIN=$CORS
NODE_ENV=$NODE_ENV
API_URL=http://localhost:3000
EOF

print_success "Created .env file"

# Start containers
print_header "Step 3: Start Containers"

print_info "Pulling images..."
docker-compose -f docker-compose-hub.yml pull || print_warning "Some images could not be pulled"

print_info "Starting services..."
docker-compose -f docker-compose-hub.yml up -d

print_success "Services started!"

# Wait for services
print_header "Step 4: Wait for Services to Be Ready"

print_info "Waiting for OpenLDAP to be ready..."
sleep 5

print_info "Waiting for PostgreSQL to be ready..."
sleep 5

print_info "Waiting for API to be ready..."
sleep 10

# Check health
print_header "Step 5: Verify Services"

print_info "Checking OpenLDAP..."
if docker ps | grep -q penux-openldap; then
    print_success "OpenLDAP is running"
else
    print_error "OpenLDAP failed to start"
fi

print_info "Checking PostgreSQL..."
if docker ps | grep -q penux-postgres; then
    print_success "PostgreSQL is running"
else
    print_error "PostgreSQL failed to start"
fi

print_info "Checking API..."
if docker ps | grep -q penux-api; then
    print_success "API is running"
else
    print_error "API failed to start"
fi

print_info "Checking Web UI..."
if docker ps | grep -q penux-web; then
    print_success "Web UI is running"
else
    print_error "Web UI failed to start"
fi

# Test API
print_header "Step 6: Test API Health"

echo ""
print_info "Testing API health check..."
if curl -s http://localhost:3000/api/health | grep -q "healthy"; then
    print_success "API is responding correctly"
else
    print_warning "API health check returned unexpected response"
fi

echo ""

# Show URLs
print_header "Step 7: Access Your Services"

echo ""
echo "🌐 Access your PenuX LDAP services:"
echo ""
echo "  REST API:  http://localhost:3000"
echo "  Web UI:    http://localhost:3001"
echo ""
echo "📝 LDAP Server:"
echo "  Host: localhost"
echo "  Port: 389 (LDAP) or 636 (LDAPS)"
echo "  Base DN: dc=penux,dc=uk"
echo "  Admin DN: cn=admin,dc=penux,dc=uk"
echo "  Password: (you entered above)"
echo ""

# Show commands
print_header "Step 8: Useful Commands"

echo ""
echo "View logs:"
echo "  docker-compose -f docker-compose-hub.yml logs -f"
echo ""
echo "View specific service logs:"
echo "  docker-compose -f docker-compose-hub.yml logs -f api"
echo ""
echo "Stop services:"
echo "  docker-compose -f docker-compose-hub.yml stop"
echo ""
echo "Start services:"
echo "  docker-compose -f docker-compose-hub.yml start"
echo ""
echo "Restart services:"
echo "  docker-compose -f docker-compose-hub.yml restart"
echo ""
echo "View running containers:"
echo "  docker ps"
echo ""
echo "Remove all (careful!):"
echo "  docker-compose -f docker-compose-hub.yml down"
echo ""

# Test options
print_header "Step 9: Test Your Deployment"

echo ""
echo "Test API health:"
echo "  curl http://localhost:3000/api/health"
echo ""
echo "Test with authentication:"
echo "  curl -u 'cn=admin,dc=penux,dc=uk:${LDAP_PASS}' \\"
echo "    http://localhost:3000/api/users"
echo ""
echo "Open Web UI in browser:"
echo "  http://localhost:3001"
echo ""

# Final message
print_header "🎉 Deployment Complete!"

echo ""
echo "✅ All services are running!"
echo ""
echo "Next steps:"
echo "  1. Test the API: curl http://localhost:3000/api/health"
echo "  2. Open Web UI: http://localhost:3001"
echo "  3. View logs: docker-compose -f docker-compose-hub.yml logs -f"
echo "  4. Create LDAP users via Web UI or API"
echo "  5. For production: set up Nginx reverse proxy & HTTPS"
echo ""
echo "Documentation: /DOCKER-HUB-DEPLOY.md"
echo ""

print_success "Ready to use!"
