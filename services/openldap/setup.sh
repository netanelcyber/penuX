#!/bin/bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "PenuX OpenLDAP Setup Script"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file from template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ".env file created. Please review and update passwords in .env"
else
    echo ".env file already exists."
fi

# Create certs directory
mkdir -p certs

echo ""
echo "Starting OpenLDAP services..."
docker-compose up -d

echo ""
echo "=========================================="
echo "OpenLDAP Services Started!"
echo "=========================================="
echo ""
echo "Access Information:"
echo "  Web Admin Interface: http://localhost"
echo "  HTTPS Admin Interface: https://localhost:6443"
echo "  LDAP Server: ldap://localhost:389"
echo "  LDAPS Server: ldaps://localhost:636"
echo ""
echo "Default Credentials:"
echo "  Admin DN: cn=admin,dc=penux,dc=uk"
echo "  Admin Password: (see .env file)"
echo ""
echo "Sample Users (all with password from bootstrap.ldif):"
echo "  - admin@penux.uk (Administrator)"
echo "  - john@penux.uk (Developer)"
echo "  - jane@penux.uk (DevOps Engineer)"
echo ""
echo "Next steps:"
echo "  1. Wait 30-40 seconds for services to fully initialize"
echo "  2. Open http://localhost in your browser"
echo "  3. Login with cn=admin,dc=penux,dc=uk"
echo "  4. Manage users and groups via the web interface"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f openldap"
echo "  docker-compose logs -f phpldapadmin"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To remove all data and start fresh:"
echo "  docker-compose down -v"
echo ""
