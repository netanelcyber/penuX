# PenuX OpenLDAP Setup Script for Windows PowerShell
# Requires: Docker Desktop for Windows (with WSL2 backend recommended)

param(
    [switch]$Help
)

if ($Help) {
    Write-Host @"
PenuX OpenLDAP Setup Script for Windows

Usage: .\setup.ps1

Prerequisites:
  - Docker Desktop for Windows (https://www.docker.com/products/docker-desktop)
  - PowerShell 5.0 or higher (or Windows PowerShell)
  - WSL2 backend enabled in Docker Desktop (recommended)

This script will:
  1. Check Docker and Docker Compose installation
  2. Create .env file from template (if not exists)
  3. Create certs directory
  4. Start OpenLDAP and phpLDAPadmin services
  5. Display access information

Run from the services\openldap directory.
"@
    exit 0
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "PenuX OpenLDAP Setup Script for Windows" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
Write-Host "Checking Docker installation..." -ForegroundColor Yellow
try {
    $dockerVersion = docker --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker found: $dockerVersion" -ForegroundColor Green
    } else {
        throw "Docker not found"
    }
} catch {
    Write-Host "✗ Docker is not installed or not in PATH" -ForegroundColor Red
    Write-Host "  Download Docker Desktop: https://www.docker.com/products/docker-desktop" -ForegroundColor Yellow
    exit 1
}

# Check if Docker Compose is installed
Write-Host "Checking Docker Compose installation..." -ForegroundColor Yellow
try {
    $composeVersion = docker-compose --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker Compose found: $composeVersion" -ForegroundColor Green
    } else {
        # Try with docker compose (newer versions)
        $composeVersion = docker compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Docker Compose found (via docker compose): $composeVersion" -ForegroundColor Green
        } else {
            throw "Docker Compose not found"
        }
    }
} catch {
    Write-Host "✗ Docker Compose is not installed" -ForegroundColor Red
    Write-Host "  It should be included with Docker Desktop" -ForegroundColor Yellow
    exit 1
}

# Create .env file if not exists
Write-Host ""
Write-Host "Configuring environment..." -ForegroundColor Yellow
if (!(Test-Path ".env")) {
    Write-Host "Creating .env file from template..."
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env file created. Please review and update passwords in .env" -ForegroundColor Green
} else {
    Write-Host "✓ .env file already exists" -ForegroundColor Green
}

# Create certs directory
if (!(Test-Path "certs")) {
    New-Item -ItemType Directory -Path "certs" -Force | Out-Null
    Write-Host "✓ Created certs directory" -ForegroundColor Green
}

# Start services
Write-Host ""
Write-Host "Starting OpenLDAP services..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "OpenLDAP Services Started!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""

    Write-Host "Access Information:" -ForegroundColor Cyan
    Write-Host "  Web Admin Interface: http://localhost" -ForegroundColor White
    Write-Host "  HTTPS Admin Interface: https://localhost:6443" -ForegroundColor White
    Write-Host "  LDAP Server: ldap://localhost:389" -ForegroundColor White
    Write-Host "  LDAPS Server: ldaps://localhost:636" -ForegroundColor White
    Write-Host ""

    Write-Host "Default Credentials:" -ForegroundColor Cyan
    Write-Host "  Admin DN: cn=admin,dc=penux,dc=uk" -ForegroundColor White
    Write-Host "  Admin Password: (see .env file)" -ForegroundColor White
    Write-Host ""

    Write-Host "Sample Users (all with password from bootstrap.ldif):" -ForegroundColor Cyan
    Write-Host "  - admin@penux.uk (Administrator)" -ForegroundColor White
    Write-Host "  - john@penux.uk (Developer)" -ForegroundColor White
    Write-Host "  - jane@penux.uk (DevOps Engineer)" -ForegroundColor White
    Write-Host ""

    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Wait 30-40 seconds for services to fully initialize" -ForegroundColor White
    Write-Host "  2. Open http://localhost in your browser" -ForegroundColor White
    Write-Host "  3. Login with cn=admin,dc=penux,dc=uk" -ForegroundColor White
    Write-Host "  4. Manage users and groups via the web interface" -ForegroundColor White
    Write-Host ""

    Write-Host "Service Management:" -ForegroundColor Cyan
    Write-Host "  View logs: docker-compose logs -f openldap" -ForegroundColor White
    Write-Host "  View admin logs: docker-compose logs -f phpldapadmin" -ForegroundColor White
    Write-Host "  Stop services: docker-compose down" -ForegroundColor White
    Write-Host "  Use manage.ps1 for additional operations" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "✗ Failed to start services" -ForegroundColor Red
    Write-Host "  Run: docker-compose logs" -ForegroundColor Yellow
    exit 1
}
