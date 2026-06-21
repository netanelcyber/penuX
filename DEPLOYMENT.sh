#!/bin/bash

# PenuX LDAP Complete Deployment Script
# Run this on your Windows (WSL2), Linux, or macOS machine
# This script will:
# 1. Start OpenLDAP Docker containers
# 2. Prepare GitHub Pages deployment
# 3. Create API deployment files
# 4. Display next steps

set -e

echo "=========================================="
echo "PenuX LDAP Deployment Script"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check Docker
echo -e "${BLUE}Checking Docker installation...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not installed${NC}"
    echo "  Download: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}✗ Docker Compose not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker found: $(docker --version)${NC}"

# Navigate to OpenLDAP directory
cd "$(dirname "$0")/services/openldap"

# Create .env file
echo ""
echo -e "${BLUE}Setting up configuration...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file${NC}"
else
    echo -e "${YELLOW}⚠ .env already exists, using existing configuration${NC}"
fi

# Create certs directory
mkdir -p certs
echo -e "${GREEN}✓ Created certs directory${NC}"

# Start Docker containers
echo ""
echo -e "${BLUE}Starting OpenLDAP services...${NC}"
docker-compose up -d 2>/dev/null || docker compose up -d

sleep 3

# Check if containers are running
echo ""
echo -e "${BLUE}Verifying containers...${NC}"
docker-compose ps 2>/dev/null || docker compose ps

echo ""
echo -e "${GREEN}✓ OpenLDAP services started!${NC}"

# Display access information
echo ""
echo "=========================================="
echo -e "${GREEN}OpenLDAP is now running!${NC}"
echo "=========================================="
echo ""
echo -e "${BLUE}Local Access:${NC}"
echo "  Web Admin:      http://localhost"
echo "  Web Admin HTTPS: https://localhost:6443"
echo "  LDAP Server:     ldap://localhost:389"
echo "  LDAPS Server:    ldaps://localhost:636"
echo ""
echo -e "${BLUE}Default Credentials:${NC}"
echo "  Admin DN:       cn=admin,dc=penux,dc=uk"
echo "  Admin Password: admin123 (change in .env!)"
echo ""
echo -e "${BLUE}Sample Users:${NC}"
echo "  - admin@penux.uk (Administrator)"
echo "  - john@penux.uk (Developer)"
echo "  - jane@penux.uk (DevOps Engineer)"
echo ""
echo -e "${YELLOW}⚠ Wait 30-40 seconds for full initialization...${NC}"
echo ""

# Display next steps
echo "=========================================="
echo -e "${BLUE}NEXT STEPS:${NC}"
echo "=========================================="
echo ""
echo -e "${GREEN}1. Access Web Interface${NC}"
echo "   Open: http://localhost"
echo "   Login with: cn=admin,dc=penux,dc=uk / admin123"
echo ""
echo -e "${GREEN}2. Manage Services${NC}"
echo "   View status:  ./manage.sh status"
echo "   View logs:    ./manage.sh logs"
echo "   Test LDAP:    ./manage.sh test"
echo ""
echo -e "${GREEN}3. Backup Database${NC}"
echo "   On Windows (PowerShell):"
echo "     .\backup.ps1 backup"
echo "   On Linux/macOS:"
echo "     Make executable: chmod +x backup.sh"
echo "     Run: ./backup.sh (if exists)"
echo ""
echo -e "${GREEN}4. Deploy to GitHub Pages${NC}"
echo "   Follow: WEB_DEPLOYMENT.md"
echo "   Steps:"
echo "     a) Create yourusername.github.io repo"
echo "     b) Copy web/index.html to repo"
echo "     c) Deploy API to Vercel/Netlify"
echo "     d) Connect web UI to API"
echo ""
echo -e "${GREEN}5. Configure Custom Domain${NC}"
echo "   Add DNS CNAME: ldap.penux.uk → yourusername.github.io"
echo "   Follow: GITHUB_PAGES_SETUP.md"
echo ""

echo "=========================================="
echo -e "${GREEN}Deployment Complete! 🎉${NC}"
echo "=========================================="
