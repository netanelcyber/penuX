# PenuX OpenLDAP Management Script for Windows PowerShell

param(
    [Parameter(Position = 0)]
    [string]$Command = "help",
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

function Show-Usage {
    Write-Host @"
Usage: .\manage.ps1 [command]

Commands:
  start         Start OpenLDAP services
  stop          Stop OpenLDAP services
  restart       Restart OpenLDAP services
  status        Show service status
  logs          View service logs (Ctrl+C to exit)
  logs-ldap     View OpenLDAP logs only
  logs-admin    View phpLDAPadmin logs only
  shell-ldap    Open PowerShell in OpenLDAP container
  shell-admin   Open PowerShell in phpLDAPadmin container
  clean         Stop and remove all containers and volumes (WARNING: data loss!)
  test          Test LDAP connectivity
  add-user      Add a new user to LDAP (interactive)
  list-users    List all users in LDAP
  help          Show this help message

Examples:
  .\manage.ps1 start
  .\manage.ps1 logs
  .\manage.ps1 test
  .\manage.ps1 add-user

"@
}

function Start-Services {
    Write-Host "Starting OpenLDAP services..." -ForegroundColor Blue
    docker-compose up -d
    Write-Host "Services started!" -ForegroundColor Green
    Start-Sleep -Seconds 2
    docker-compose ps
}

function Stop-Services {
    Write-Host "Stopping OpenLDAP services..." -ForegroundColor Blue
    docker-compose down
    Write-Host "Services stopped!" -ForegroundColor Green
}

function Restart-Services {
    Write-Host "Restarting OpenLDAP services..." -ForegroundColor Blue
    docker-compose restart
    Write-Host "Services restarted!" -ForegroundColor Green
}

function Show-Status {
    Write-Host "OpenLDAP Service Status:" -ForegroundColor Cyan
    docker-compose ps
}

function Show-Logs {
    docker-compose logs -f
}

function Show-Logs-LDAP {
    docker-compose logs -f openldap
}

function Show-Logs-Admin {
    docker-compose logs -f phpldapadmin
}

function Test-LDAP {
    Write-Host "Testing LDAP connectivity..." -ForegroundColor Blue

    # Load .env file
    if (Test-Path ".env") {
        $env_content = Get-Content ".env"
        foreach ($line in $env_content) {
            if ($line -match "^LDAP_ADMIN_PASSWORD=(.*)$") {
                $adminPassword = $matches[1]
            }
        }
    } else {
        $adminPassword = "admin123"
    }

    try {
        docker run --rm --network openldap_penux_network `
            osixia/openldap:1.5.0 `
            ldapwhoami -H ldap://openldap:389 -D "cn=admin,dc=penux,dc=uk" -w "$adminPassword" 2>$null

        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ LDAP connection successful!" -ForegroundColor Green
        } else {
            Write-Host "✗ LDAP connection failed!" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ LDAP connection failed: $_" -ForegroundColor Red
    }
}

function Add-User {
    Write-Host "Add New LDAP User" -ForegroundColor Cyan
    Write-Host ""

    $uid = Read-Host "Enter username (uid)"
    $cn = Read-Host "Enter full name (cn)"
    $email = Read-Host "Enter email"
    $password = Read-Host "Enter password" -AsSecureString
    $plaintextPassword = [System.Net.NetworkCredential]::new('', $password).Password

    $ldifContent = @"
dn: cn=$cn,ou=users,dc=penux,dc=uk
objectClass: inetOrgPerson
objectClass: organizationalPerson
objectClass: person
cn: $cn
sn: User
givenName: $uid
uid: $uid
mail: $email
userPassword: $plaintextPassword
accountStatus: active
description: Added via script
"@

    $ldifPath = "$env:TEMP\user_$uid.ldif"
    Set-Content -Path $ldifPath -Value $ldifContent

    # Load .env file
    if (Test-Path ".env") {
        $env_content = Get-Content ".env"
        foreach ($line in $env_content) {
            if ($line -match "^LDAP_ADMIN_PASSWORD=(.*)$") {
                $adminPassword = $matches[1]
            }
        }
    } else {
        $adminPassword = "admin123"
    }

    try {
        docker cp $ldifPath penux-openldap:/tmp/user.ldif
        docker exec penux-openldap ldapadd -x -D "cn=admin,dc=penux,dc=uk" -w "$adminPassword" -f /tmp/user.ldif

        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ User $uid added successfully!" -ForegroundColor Green
        } else {
            Write-Host "✗ Failed to add user!" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ Error: $_" -ForegroundColor Red
    } finally {
        Remove-Item -Path $ldifPath -ErrorAction SilentlyContinue
    }
}

function List-Users {
    Write-Host "Listing all users in LDAP:" -ForegroundColor Cyan

    # Load .env file
    if (Test-Path ".env") {
        $env_content = Get-Content ".env"
        foreach ($line in $env_content) {
            if ($line -match "^LDAP_ADMIN_PASSWORD=(.*)$") {
                $adminPassword = $matches[1]
            }
        }
    } else {
        $adminPassword = "admin123"
    }

    docker exec penux-openldap ldapsearch -x -D "cn=admin,dc=penux,dc=uk" -w "$adminPassword" `
        -b "ou=users,dc=penux,dc=uk" "(objectClass=inetOrgPerson)" uid mail cn
}

function Shell-LDAP {
    docker exec -it penux-openldap cmd
}

function Shell-Admin {
    docker exec -it penux-phpldapadmin cmd
}

function Clean-Services {
    $confirm = Read-Host "WARNING: This will delete all data! Continue? (yes/no)"
    if ($confirm -eq "yes") {
        docker-compose down -v
        Write-Host "Cleaned!" -ForegroundColor Green
    } else {
        Write-Host "Cancelled" -ForegroundColor Yellow
    }
}

# Main script logic
switch ($Command.ToLower()) {
    "start" { Start-Services }
    "stop" { Stop-Services }
    "restart" { Restart-Services }
    "status" { Show-Status }
    "logs" { Show-Logs }
    "logs-ldap" { Show-Logs-LDAP }
    "logs-admin" { Show-Logs-Admin }
    "test" { Test-LDAP }
    "add-user" { Add-User }
    "list-users" { List-Users }
    "shell-ldap" { Shell-LDAP }
    "shell-admin" { Shell-Admin }
    "clean" { Clean-Services }
    "help" { Show-Usage }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Show-Usage
        exit 1
    }
}
