# PenuX OpenLDAP Setup Script for Windows PowerShell
# Requires: Docker Desktop for Windows (with WSL2 backend recommended)

param(
    [switch]$Help,
    [switch]$Force,
    [switch]$Silent
)

# Import utilities
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptPath\utils.ps1"

if ($Help) {
    Write-Host @"
PenuX OpenLDAP Setup Script for Windows

Usage: .\setup.ps1 [options]

Options:
  -Help      Show this help message
  -Force     Skip existing .env check (create new one)
  -Silent    Minimal output

Prerequisites:
  - Docker Desktop for Windows (https://www.docker.com/products/docker-desktop)
  - PowerShell 5.0 or higher
  - WSL2 backend enabled in Docker Desktop (recommended)

This script will:
  1. Check Docker and Docker Compose installation
  2. Check Docker daemon is running
  3. Create .env file from template (if not exists)
  4. Verify necessary files exist
  5. Create certs directory
  6. Start OpenLDAP and phpLDAPadmin services
  7. Wait for services to become healthy
  8. Display access information

Run from the services\openldap directory.

Examples:
  .\setup.ps1              # Standard setup
  .\setup.ps1 -Force       # Create new .env file
  .\setup.ps1 -Silent      # Minimal output
"@
    exit 0
}

if (-not $Silent) {
    Write-Header "PenuX OpenLDAP Setup for Windows"
}

# ==================== Pre-flight Checks ====================

if (-not $Silent) { Write-Info "Running pre-flight checks..." }

# Check execution policy
if (-not (Test-ExecutionPolicy)) {
    if (-not $Silent) {
        Write-Warning "PowerShell execution policy needs adjustment"
        Write-Info "Run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    }
}

# Check Docker installation
if (-not (Test-DockerInstalled)) {
    Write-Error "Docker is not installed or not in PATH"
    Write-Info "Download Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
}

if (-not $Silent) {
    Write-Success "Docker installed: $(Get-DockerVersion)"
}

# Check Docker daemon is running
if (-not (Test-DockerRunning)) {
    Write-Error "Docker daemon is not running"
    Write-Info "Please start Docker Desktop and try again"
    exit 1
}

if (-not $Silent) { Write-Success "Docker daemon is running" }

# Check Docker Compose
if (-not (Test-DockerComposeInstalled)) {
    Write-Error "Docker Compose is not installed"
    Write-Info "It should be included with Docker Desktop. Try reinstalling Docker."
    exit 1
}

if (-not $Silent) {
    Write-Success "Docker Compose installed: $(Get-DockerComposeVersion)"
}

# ==================== Configuration ====================

if (-not $Silent) { Write-Info "Setting up configuration..." }

# Handle .env file
if ($Force -or !(Test-Path ".env")) {
    if (-not (Create-EnvFile)) {
        Write-Error "Failed to create .env file from template"
        exit 1
    }
} else {
    if (-not $Silent) { Write-Success ".env file already exists" }
}

# Create certs directory
Create-DirectoryIfNotExists "certs" | Out-Null

# Verify required files
$requiredFiles = @(
    "docker-compose.yml",
    "bootstrap.ldif",
    ".env.example"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Error "Required file not found: $file"
        exit 1
    }
}

if (-not $Silent) { Write-Success "All required files verified" }

# ==================== Start Services ====================

if (-not $Silent) {
    Write-Header "Starting Services"
    Write-Info "Launching Docker containers..."
}

docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to start services"
    Write-Info "Run 'docker-compose logs' for details"
    exit 1
}

if (-not $Silent) { Write-Success "Containers started" }

# ==================== Wait for Services ====================

if (-not $Silent) { Write-Info "Waiting for services to initialize (up to 60 seconds)..." }

$openldapReady = Wait-ForService "openldap" 60 3
$adminReady = Wait-ForService "phpldapadmin" 60 3

if (-not $openldapReady -or -not $adminReady) {
    Write-Warning "Services may not be fully ready yet"
    Write-Info "Check status with: docker-compose ps"
}

# ==================== Display Results ====================

if (-not $Silent) {
    Write-Header "Setup Complete!"

    Write-Host "Access Information:" -ForegroundColor Cyan
    Write-Host "  Web Admin (HTTP):  http://localhost" -ForegroundColor White
    Write-Host "  Web Admin (HTTPS): https://localhost:6443" -ForegroundColor White
    Write-Host "  LDAP Server:       ldap://localhost:389" -ForegroundColor White
    Write-Host "  LDAPS Server:      ldaps://localhost:636" -ForegroundColor White
    Write-Host ""

    Write-Host "Default Credentials:" -ForegroundColor Cyan
    Write-Host "  Admin DN:       cn=admin,dc=penux,dc=uk" -ForegroundColor White
    Write-Host "  Admin Password: (see .env file)" -ForegroundColor White
    Write-Host ""

    Write-Host "Sample Users:" -ForegroundColor Cyan
    Write-Host "  admin@penux.uk  (Administrator)" -ForegroundColor White
    Write-Host "  john@penux.uk   (Developer)" -ForegroundColor White
    Write-Host "  jane@penux.uk   (DevOps Engineer)" -ForegroundColor White
    Write-Host ""

    Write-Host "Next Steps:" -ForegroundColor Cyan
    Write-Host "  1. Wait 30-40 seconds for full initialization" -ForegroundColor White
    Write-Host "  2. Open http://localhost in your browser" -ForegroundColor White
    Write-Host "  3. Login with: cn=admin,dc=penux,dc=uk" -ForegroundColor White
    Write-Host "  4. Manage users and groups" -ForegroundColor White
    Write-Host ""

    Write-Host "Management Commands:" -ForegroundColor Cyan
    Write-Host "  .\manage.ps1 status      # Show service status" -ForegroundColor White
    Write-Host "  .\manage.ps1 logs        # View logs" -ForegroundColor White
    Write-Host "  .\manage.ps1 test        # Test LDAP connection" -ForegroundColor White
    Write-Host "  .\manage.ps1 help        # Show all commands" -ForegroundColor White
    Write-Host ""
}

exit 0
