@echo off
REM PenuX LDAP Complete Deployment Script for Windows
REM Run this from the penuX directory
REM Requires: Docker Desktop for Windows

setlocal enabledelayedexpansion

echo.
echo ==========================================
echo PenuX LDAP Deployment Script (Windows)
echo ==========================================
echo.

REM Check Docker
echo Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not installed
    echo Download: https://www.docker.com/products/docker-desktop
    exit /b 1
)

echo [OK] Docker found
docker --version

REM Navigate to OpenLDAP directory
cd /d "%~dp0services\openldap"
if errorlevel 1 (
    echo [ERROR] Could not navigate to OpenLDAP directory
    exit /b 1
)

REM Create .env file
echo.
echo Setting up configuration...
if not exist .env (
    copy .env.example .env
    echo [OK] Created .env file
) else (
    echo [WARNING] .env already exists
)

REM Create certs directory
if not exist certs mkdir certs
echo [OK] Created certs directory

REM Start Docker containers
echo.
echo Starting OpenLDAP services...
docker-compose up -d
if errorlevel 1 (
    REM Try newer docker compose syntax
    docker compose up -d
    if errorlevel 1 (
        echo [ERROR] Failed to start services
        exit /b 1
    )
)

timeout /t 3 /nobreak

REM Check if containers are running
echo.
echo Verifying containers...
docker-compose ps
if errorlevel 1 docker compose ps

REM Display information
echo.
echo ==========================================
echo [SUCCESS] OpenLDAP Services Started!
echo ==========================================
echo.
echo Local Access:
echo   Web Admin:         http://localhost
echo   Web Admin HTTPS:   https://localhost:6443
echo   LDAP Server:       ldap://localhost:389
echo   LDAPS Server:      ldaps://localhost:636
echo.
echo Default Credentials:
echo   Admin DN:          cn=admin,dc=penux,dc=uk
echo   Admin Password:    admin123 (change in .env!)
echo.
echo Sample Users:
echo   - admin@penux.uk (Administrator)
echo   - john@penux.uk (Developer)
echo   - jane@penux.uk (DevOps Engineer)
echo.
echo [WARNING] Wait 30-40 seconds for full initialization...
echo.

REM Display next steps
echo ==========================================
echo NEXT STEPS:
echo ==========================================
echo.
echo 1. Access Web Interface
echo    Open: http://localhost
echo    Login with: cn=admin,dc=penux,dc=uk / admin123
echo.
echo 2. Manage Services (PowerShell)
echo    View status:  .\manage.ps1 status
echo    View logs:    .\manage.ps1 logs
echo    Test LDAP:    .\manage.ps1 test
echo    Add user:     .\manage.ps1 add-user
echo.
echo 3. Backup Database
echo    .\backup.ps1 backup
echo    .\backup.ps1 backup -Compress
echo    .\backup.ps1 schedule      (Daily backups)
echo.
echo 4. Deploy to GitHub Pages
echo    Follow: WEB_DEPLOYMENT.md
echo    a) Create yourusername.github.io repo
echo    b) Copy web\index.html to repo
echo    c) Deploy API to Vercel/Netlify
echo    d) Connect web UI to API
echo.
echo 5. Configure Custom Domain
echo    Add DNS CNAME: ldap.penux.uk ^> yourusername.github.io
echo    Follow: GITHUB_PAGES_SETUP.md
echo.
echo ==========================================
echo [SUCCESS] Deployment Complete!
echo ==========================================
echo.

pause
