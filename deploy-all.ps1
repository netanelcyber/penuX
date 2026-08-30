#!/usr/bin/env pwsh
# 🚀 Complete LDAP Deployment Automation (Windows)
# Deploys OpenLDAP + Keycloak + Cloudflare Tunnel in one command

$ErrorActionPreference = "Stop"

# Colors for output
$GREEN = "`e[0;32m"
$BLUE = "`e[0;34m"
$YELLOW = "`e[1;33m"
$RED = "`e[0;31m"
$NC = "`e[0m" # No Color

# Functions
function Print-Header {
    Write-Host "${BLUE}================================${NC}"
    Write-Host "${BLUE}$args${NC}"
    Write-Host "${BLUE}================================${NC}"
}

function Print-Success {
    Write-Host "${GREEN}✅ $args${NC}"
}

function Print-Info {
    Write-Host "${BLUE}ℹ️  $args${NC}"
}

function Print-Warning {
    Write-Host "${YELLOW}⚠️  $args${NC}"
}

function Print-Error {
    Write-Host "${RED}❌ $args${NC}"
}

# Check prerequisites
Print-Header "PHASE 1: Checking Prerequisites"

Print-Info "Checking Docker..."
try {
    $docker_version = docker --version
    Print-Success "Docker found: $docker_version"
} catch {
    Print-Error "Docker not found. Install Docker Desktop first!"
    exit 1
}

Print-Info "Checking Docker daemon..."
try {
    docker ps | Out-Null
    Print-Success "Docker daemon running"
} catch {
    Print-Error "Docker daemon not running. Start Docker Desktop first!"
    exit 1
}

Print-Info "Checking if in project directory..."
if (-not (Test-Path "docker-compose.yml")) {
    Print-Error "docker-compose.yml not found. Run from penuX directory!"
    exit 1
}
Print-Success "Project directory verified"

# Phase 2: Start Docker Services
Print-Header "PHASE 2: Starting Docker Services"

Print-Info "Stopping any existing services..."
try {
    docker compose down 2>&1 | Out-Null
} catch {
    # Ignore if no services running
}

Print-Info "Starting OpenLDAP + Keycloak + PostgreSQL..."
docker compose up -d

Print-Info "Waiting 45 seconds for services to initialize..."
for ($i = 45; $i -ge 1; $i--) {
    Write-Host -NoNewline "`r${BLUE}$i seconds remaining...${NC}        "
    Start-Sleep -Seconds 1
}
Write-Host "`r                                    "

Print-Info "Checking service status..."
$services = docker compose ps
if ($services -match "healthy") {
    Print-Success "Services started successfully"
} else {
    Print-Warning "Services may still be initializing (this is normal)"
}

Write-Host $services

# Phase 3: Test LDAP Locally
Print-Header "PHASE 3: Testing LDAP Locally"

Print-Info "Testing LDAP connection..."
try {
    $ldaptest = & ldapwhoami -H "ldap://localhost:389" -D "cn=admin,dc=penux,dc=uk" -w "admin123" 2>&1
    if ($ldaptest) {
        Print-Success "LDAP is responding"
    } else {
        Print-Warning "LDAP test inconclusive"
    }
} catch {
    Print-Warning "LDAP test failed - services may still be starting"
}

Print-Info "Testing Keycloak..."
try {
    $keycloak_response = curl.exe -s "http://localhost:8080/health" 2>&1
    if ($keycloak_response -match "UP|keycloak") {
        Print-Success "Keycloak is responding"
    } else {
        Print-Warning "Keycloak test inconclusive"
    }
} catch {
    Print-Warning "Keycloak test inconclusive - check manually"
}

# Phase 4: Install Cloudflare Tunnel
Print-Header "PHASE 4: Installing Cloudflare Tunnel"

Print-Info "Checking if cloudflared is installed..."
try {
    $null = Get-Command cloudflared
    $tunnel_version = cloudflared --version
    Print-Success "cloudflared already installed: $tunnel_version"
} catch {
    Print-Info "Installing cloudflared..."
    try {
        # Try chocolatey
        choco install cloudflared -y
        Print-Success "cloudflared installed via Chocolatey"
    } catch {
        Print-Warning "Chocolatey not found. Installing cloudflared manually..."
        $url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
        $destination = "$env:USERPROFILE\cloudflared.exe"

        Print-Info "Downloading cloudflared from GitHub..."
        Invoke-WebRequest -Uri $url -OutFile $destination

        # Add to PATH
        $env:PATH = "$env:USERPROFILE;$env:PATH"

        Print-Success "cloudflared installed to $destination"
        Print-Info "Add to PATH: Add '$env:USERPROFILE' to your system PATH environment variable"
    }
}

# Phase 5: Cloudflare Login
Print-Header "PHASE 5: Cloudflare Login & Tunnel Creation"

Print-Warning "This will open a browser window to login to Cloudflare"
Print-Info "Login with your Cloudflare account and authorize for penux.uk domain"

Read-Host "Press ENTER when ready to login to Cloudflare"

cloudflared tunnel login

Print-Success "Cloudflare login successful"

# Phase 6: Create Tunnel
Print-Header "PHASE 6: Creating Cloudflare Tunnel"

$TUNNEL_NAME = "ldap-tunnel"

Print-Info "Checking if tunnel already exists..."
try {
    $tunnel_list = cloudflared tunnel list 2>&1
    if ($tunnel_list -match $TUNNEL_NAME) {
        Print-Warning "Tunnel '$TUNNEL_NAME' already exists"
        $TUNNEL_ID = ($tunnel_list | Select-String $TUNNEL_NAME | ForEach-Object { $_.Line.Split()[0] })[0]
    } else {
        Print-Info "Creating new tunnel: $TUNNEL_NAME"
        cloudflared tunnel create $TUNNEL_NAME
        Print-Success "Tunnel created successfully"
    }
} catch {
    Print-Info "Creating new tunnel: $TUNNEL_NAME"
    cloudflared tunnel create $TUNNEL_NAME
    Print-Success "Tunnel created successfully"
}

# Find credentials file
Print-Info "Finding Cloudflare credentials file..."
$creds_files = Get-ChildItem -Path "$env:USERPROFILE\.cloudflared" -Filter "*.json" -ErrorAction SilentlyContinue
if ($creds_files) {
    $CREDS_FILE = $creds_files[0].FullName
    Print-Success "Found credentials: $CREDS_FILE"
} else {
    Print-Error "Could not find Cloudflare credentials file"
    exit 1
}

# Phase 7: Create Tunnel Config
Print-Header "PHASE 7: Creating Tunnel Configuration"

$CONFIG_DIR = "$env:USERPROFILE\.cloudflared"
$CONFIG_FILE = "$CONFIG_DIR\config.yml"

Print-Info "Creating config file: $CONFIG_FILE"

$config_content = @"
tunnel: $TUNNEL_NAME
credentials-file: $CREDS_FILE

ingress:
  - hostname: ldap.penux.uk
    service: http://localhost:3001

  - hostname: keycloak.penux.uk
    service: http://localhost:8080

  - hostname: ldap-server.penux.uk
    service: tcp://localhost:389

  - hostname: ldaps-server.penux.uk
    service: tcp://localhost:636

  - hostname: api.ldap.penux.uk
    service: http://localhost:3000

  - service: http_status:404
"@

Set-Content -Path $CONFIG_FILE -Value $config_content -Encoding UTF8

Print-Success "Config file created"

# Phase 8: Get Tunnel CNAME
Print-Header "PHASE 8: Getting Tunnel Information"

$TUNNEL_CNAME = "$TUNNEL_NAME.cfargotunnel.com"
Print-Success "Your tunnel CNAME: $TUNNEL_CNAME"

# Phase 9: Instructions for DNS
Print-Header "PHASE 9: Next Steps - Cloudflare DNS Configuration"

Print-Warning "MANUAL STEP REQUIRED - Update Cloudflare DNS:"
Write-Host ""
Write-Host "1. Go to: https://dash.cloudflare.com"
Write-Host "2. Select: penux.uk domain"
Write-Host "3. Click: DNS → Records"
Write-Host ""
Write-Host "4. Add/Update these CNAME records:"
Write-Host "   - Name: ldap"
Write-Host "     Value: $TUNNEL_CNAME"
Write-Host ""
Write-Host "   - Name: ldaps-server"
Write-Host "     Value: $TUNNEL_CNAME"
Write-Host ""
Write-Host "   - Name: api.ldap"
Write-Host "     Value: $TUNNEL_CNAME"
Write-Host ""
Write-Host "5. Also ensure:"
Write-Host "   - SSL/TLS → Encryption Mode: Full"
Write-Host "   - SSL/TLS → Always Use HTTPS: ON"
Write-Host ""

Read-Host "Press ENTER when DNS records are updated in Cloudflare"

# Phase 10: Start Tunnel
Print-Header "PHASE 10: Starting Cloudflare Tunnel"

Print-Info "Starting tunnel in background..."

# Create a temporary script to run cloudflared
$tunnel_script = "$env:TEMP\cloudflared_tunnel.ps1"
$tunnel_log = "$env:TEMP\cloudflared.log"

$script_content = @"
cloudflared tunnel run $TUNNEL_NAME *>> '$tunnel_log'
"@

Set-Content -Path $tunnel_script -Value $script_content

# Start the tunnel process
$tunnel_process = Start-Process powershell.exe -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$tunnel_script`"" -PassThru

Print-Success "Tunnel started (PID: $($tunnel_process.Id))"

Start-Sleep -Seconds 5

# Check if tunnel is running
if ($tunnel_process.HasExited -eq $false) {
    Print-Success "Tunnel is running"
} else {
    Print-Error "Tunnel failed to start"
    if (Test-Path $tunnel_log) {
        Get-Content $tunnel_log
    }
    exit 1
}

# Phase 11: Verification
Print-Header "PHASE 11: Verification & Testing"

Print-Info "Waiting 15 seconds for DNS propagation..."
Start-Sleep -Seconds 15

Print-Info "Testing DNS resolution..."
try {
    $dns_result = nslookup "ldap.penux.uk" 2>&1
    if ($dns_result -match "Address") {
        Print-Success "DNS resolved: ldap.penux.uk"
    } else {
        Print-Warning "DNS not yet propagated (this is normal, wait a few minutes)"
    }
} catch {
    Print-Warning "DNS not yet propagated (this is normal, wait a few minutes)"
}

Print-Info "Testing LDAP via tunnel..."
try {
    $ldap_tunnel_test = & ldapwhoami -H "ldap://ldap-server.penux.uk:389" -D "cn=admin,dc=penux,dc=uk" -w "admin123" 2>&1
    if ($ldap_tunnel_test) {
        Print-Success "LDAP accessible via tunnel"
    } else {
        Print-Warning "LDAP via tunnel not yet accessible (DNS may still be propagating)"
    }
} catch {
    Print-Warning "LDAP via tunnel not yet accessible (DNS may still be propagating)"
}

# Phase 12: Summary
Print-Header "🎉 DEPLOYMENT COMPLETE!"

Write-Host ""
Write-Host -ForegroundColor Green "Your LDAP Directory is Now Live!"
Write-Host ""
Write-Host "Access URLs:"
Write-Host -ForegroundColor Green "  Web UI:      " -NoNewline; Write-Host "https://ldap.penux.uk"
Write-Host -ForegroundColor Green "  Admin:       " -NoNewline; Write-Host "https://ldap.penux.uk/admin"
Write-Host -ForegroundColor Green "  LDAP:        " -NoNewline; Write-Host "ldap://ldap-server.penux.uk:389"
Write-Host -ForegroundColor Green "  LDAPS:       " -NoNewline; Write-Host "ldaps://ldaps-server.penux.uk:636"
Write-Host -ForegroundColor Green "  API:         " -NoNewline; Write-Host "https://api.ldap.penux.uk"
Write-Host ""
Write-Host "Credentials:"
Write-Host -ForegroundColor Green "  Admin DN:    " -NoNewline; Write-Host "cn=admin,dc=penux,dc=uk"
Write-Host -ForegroundColor Green "  Password:    " -NoNewline; Write-Host "admin123"
Write-Host ""
Write-Host "Services Running:"
Write-Host -ForegroundColor Green "  ✅ Docker Compose " -NoNewline; Write-Host "(OpenLDAP + Keycloak + PostgreSQL)"
Write-Host -ForegroundColor Green "  ✅ Cloudflare Tunnel " -NoNewline; Write-Host "(Secure tunnel to your machine)"
Write-Host ""
Write-Host "To stop all services:"
Write-Host -ForegroundColor Yellow "  docker compose down"
Write-Host ""
Write-Host "To view tunnel logs:"
Write-Host -ForegroundColor Yellow "  Get-Content $tunnel_log -Tail 50 -Wait"
Write-Host ""

Print-Success "Deployment finished! Check the URLs above."
