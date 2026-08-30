#!/bin/bash
# 🐳 Build & Push Images to Docker Hub
# Automates building and pushing PenuX LDAP images to Docker Hub

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

print_header "PenuX LDAP - Build & Push to Docker Hub"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker not found. Install from https://docs.docker.com/get-docker/"
    exit 1
fi
print_success "Docker found: $(docker --version)"

if ! docker ps &> /dev/null; then
    print_error "Docker daemon not running or permission denied"
    exit 1
fi
print_success "Docker daemon is running"

# Get Docker Hub username
print_header "Step 1: Docker Hub Account"

read -p "Enter your Docker Hub username (default: netanelcyber): " DOCKER_USER
DOCKER_USER=${DOCKER_USER:-netanelcyber}
print_success "Docker Hub username: $DOCKER_USER"

read -p "Enter image tag (default: latest): " TAG
TAG=${TAG:-latest}
print_success "Image tag: $TAG"

API_IMAGE="$DOCKER_USER/penux-ldap-api:$TAG"
WEB_IMAGE="$DOCKER_USER/penux-ldap-web:$TAG"

# Login to Docker Hub
print_header "Step 2: Login to Docker Hub"
print_info "Logging in to Docker Hub..."
docker login -u "$DOCKER_USER" || {
    print_error "Docker Hub login failed"
    exit 1
}
print_success "Logged in to Docker Hub"

# Build API image
print_header "Step 3: Build REST API Image"
print_info "Building $API_IMAGE..."
docker build -f Dockerfile.api -t "$API_IMAGE" .
print_success "API image built: $API_IMAGE"

# Build Web image
print_header "Step 4: Build Web UI Image"
print_info "Building $WEB_IMAGE..."
docker build -f Dockerfile.web -t "$WEB_IMAGE" .
print_success "Web image built: $WEB_IMAGE"

# Optional: Test images locally
print_header "Step 5: Test Images (Optional)"
read -p "Test images locally before pushing? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Starting API container for test..."
    docker run -d --name penux-api-test -p 3000:3000 \
      -e LDAP_HOST="ldap://localhost:389" \
      -e LDAP_BASE_DN="dc=penux,dc=uk" \
      -e LDAP_ADMIN_DN="cn=admin,dc=penux,dc=uk" \
      -e LDAP_ADMIN_PASSWORD="admin123" \
      "$API_IMAGE" || print_warning "Container may need LDAP backend"

    sleep 3
    print_info "Testing health endpoint..."
    curl -s http://localhost:3000/api/health || print_warning "Health check needs LDAP backend"
    echo ""

    print_info "Cleaning up test container..."
    docker stop penux-api-test &>/dev/null || true
    docker rm penux-api-test &>/dev/null || true
    print_success "Test cleanup done"
fi

# Push images
print_header "Step 6: Push Images to Docker Hub"

print_info "Pushing API image..."
docker push "$API_IMAGE"
print_success "API image pushed"

print_info "Pushing Web image..."
docker push "$WEB_IMAGE"
print_success "Web image pushed"

# Also tag and push 'latest' if a different tag was used
if [ "$TAG" != "latest" ]; then
    print_info "Also tagging as latest..."
    docker tag "$API_IMAGE" "$DOCKER_USER/penux-ldap-api:latest"
    docker tag "$WEB_IMAGE" "$DOCKER_USER/penux-ldap-web:latest"
    docker push "$DOCKER_USER/penux-ldap-api:latest"
    docker push "$DOCKER_USER/penux-ldap-web:latest"
    print_success "Latest tags pushed"
fi

# Summary
print_header "🎉 Images Published to Docker Hub!"

echo ""
echo "📦 Your images are now live:"
echo ""
echo "   API: https://hub.docker.com/r/$DOCKER_USER/penux-ldap-api"
echo "   Web: https://hub.docker.com/r/$DOCKER_USER/penux-ldap-web"
echo ""
echo "📥 Pull commands:"
echo "   docker pull $API_IMAGE"
echo "   docker pull $WEB_IMAGE"
echo ""
echo "🚀 Deploy with docker-compose:"
echo "   docker-compose -f docker-compose-hub.yml up -d"
echo ""
echo "💡 Note: Make sure docker-compose-hub.yml uses:"
echo "   image: $DOCKER_USER/penux-ldap-api:$TAG"
echo "   image: $DOCKER_USER/penux-ldap-web:$TAG"
echo ""

print_success "Done! Your images are ready to deploy anywhere."
