# PenuX OpenLDAP Utility Functions for PowerShell
# Shared functions used by setup.ps1 and manage.ps1

# Colors
$script:ColorGreen = 'Green'
$script:ColorRed = 'Red'
$script:ColorYellow = 'Yellow'
$script:ColorBlue = 'Blue'
$script:ColorCyan = 'Cyan'
$script:ColorWhite = 'White'

# ==================== Output Functions ====================

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor $script:ColorGreen
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $script:ColorRed
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor $script:ColorYellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor $script:ColorCyan
}

function Write-Header {
    param([string]$Message)
    Write-Host ""
    Write-Host "===========================================" -ForegroundColor $script:ColorCyan
    Write-Host $Message -ForegroundColor $script:ColorCyan
    Write-Host "===========================================" -ForegroundColor $script:ColorCyan
    Write-Host ""
}

# ==================== Docker Functions ====================

function Test-DockerInstalled {
    try {
        $output = docker --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

function Test-DockerRunning {
    try {
        docker info >$null 2>&1
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Test-DockerComposeInstalled {
    try {
        docker-compose --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        # Try newer format
        docker compose version 2>$null
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Get-DockerVersion {
    try {
        return docker --version 2>$null
    }
    catch {
        return "Unknown"
    }
}

function Get-DockerComposeVersion {
    try {
        return docker-compose --version 2>$null
    }
    catch {
        try {
            return docker compose version 2>$null
        }
        catch {
            return "Unknown"
        }
    }
}

# ==================== Environment Functions ====================

function Get-EnvVariable {
    param(
        [string]$VarName,
        [string]$DefaultValue = ""
    )

    if (Test-Path ".env") {
        $content = Get-Content ".env"
        foreach ($line in $content) {
            if ($line -match "^$VarName=(.*)$") {
                return $matches[1].Trim('"')
            }
        }
    }

    return $DefaultValue
}

function Load-EnvFile {
    if (Test-Path ".env") {
        $env_content = Get-Content ".env"
        foreach ($line in $env_content) {
            if ($line -match "^([^=]+)=(.*)$") {
                $varName = $matches[1]
                $varValue = $matches[2].Trim('"')
                Set-Item -Path "env:\$varName" -Value $varValue -ErrorAction SilentlyContinue
            }
        }
        return $true
    }
    return $false
}

function Create-EnvFile {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Success ".env file created from template"
        return $true
    }
    Write-Error ".env.example not found"
    return $false
}

# ==================== Service Functions ====================

function Test-ServicePort {
    param(
        [int]$Port
    )

    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        $listener.Stop()
        return $false  # Port is available
    }
    catch {
        return $true   # Port is in use
    }
}

function Get-ServiceStatus {
    try {
        docker-compose ps --format "json" | ConvertFrom-Json
        return $true
    }
    catch {
        return $false
    }
}

function Wait-ForService {
    param(
        [string]$ServiceName,
        [int]$TimeoutSeconds = 60,
        [int]$IntervalSeconds = 5
    )

    $elapsed = 0

    while ($elapsed -lt $TimeoutSeconds) {
        try {
            $status = docker-compose ps $ServiceName --format "json" | ConvertFrom-Json
            if ($status.State -like "*running*" -and $status.State -like "*healthy*") {
                Write-Success "$ServiceName is healthy"
                return $true
            }
            elseif ($status.State -like "*running*") {
                Write-Info "$ServiceName is running (initializing...)"
            }
        }
        catch {
            # Service not ready yet
        }

        Start-Sleep -Seconds $IntervalSeconds
        $elapsed += $IntervalSeconds
        Write-Host "." -NoNewline -ForegroundColor $script:ColorYellow
    }

    Write-Host ""
    Write-Error "$ServiceName failed to reach healthy state within $TimeoutSeconds seconds"
    return $false
}

# ==================== LDAP Functions ====================

function Test-LDAPConnection {
    param(
        [string]$AdminPassword = "admin123"
    )

    try {
        docker run --rm --network openldap_penux_network `
            osixia/openldap:1.5.0 `
            ldapwhoami -H ldap://openldap:389 -D "cn=admin,dc=penux,dc=uk" -w "$AdminPassword" 2>$null

        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        return $false
    }
    catch {
        return $false
    }
}

function Search-LDAPUsers {
    param(
        [string]$AdminPassword = "admin123"
    )

    try {
        docker exec penux-openldap ldapsearch -x -D "cn=admin,dc=penux,dc=uk" -w "$AdminPassword" `
            -b "ou=users,dc=penux,dc=uk" "(objectClass=inetOrgPerson)" uid mail cn
        return $true
    }
    catch {
        return $false
    }
}

function Search-LDAPGroups {
    param(
        [string]$AdminPassword = "admin123"
    )

    try {
        docker exec penux-openldap ldapsearch -x -D "cn=admin,dc=penux,dc=uk" -w "$AdminPassword" `
            -b "ou=groups,dc=penux,dc=uk" "(objectClass=groupOfNames)" cn description
        return $true
    }
    catch {
        return $false
    }
}

# ==================== File Functions ====================

function Create-DirectoryIfNotExists {
    param([string]$Path)

    if (!(Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Success "Created directory: $Path"
        return $true
    }
    return $false
}

function Ensure-Executable {
    param([string]$FilePath)

    if (Test-Path $FilePath) {
        Write-Success "Found: $FilePath"
        return $true
    }
    Write-Error "Not found: $FilePath"
    return $false
}

# ==================== Menu Functions ====================

function Show-Menu {
    param(
        [string]$Title,
        [string[]]$Options
    )

    Write-Host ""
    Write-Host $Title -ForegroundColor $script:ColorCyan
    Write-Host ""

    for ($i = 0; $i -lt $Options.Length; $i++) {
        Write-Host "$($i + 1). $($Options[$i])" -ForegroundColor $script:ColorWhite
    }
    Write-Host ""

    $selection = Read-Host "Select option (1-$($Options.Length))"

    if ([int]::TryParse($selection, [ref]$null) -and $selection -ge 1 -and $selection -le $Options.Length) {
        return [int]$selection - 1
    }

    return -1
}

# ==================== Backup/Restore Functions ====================

function Backup-LDAPDatabase {
    param(
        [string]$OutputFile = "ldap_backup.ldif",
        [string]$AdminPassword = "admin123"
    )

    try {
        Write-Info "Backing up LDAP database to $OutputFile..."

        $backup = docker exec penux-openldap slapcat 2>$null

        if ($LASTEXITCODE -eq 0) {
            $backup | Out-File -Encoding UTF8 -FilePath $OutputFile
            Write-Success "Backup created: $OutputFile"
            return $true
        }
        else {
            Write-Error "Backup failed"
            return $false
        }
    }
    catch {
        Write-Error "Backup error: $_"
        return $false
    }
}

function Restore-LDAPDatabase {
    param(
        [string]$BackupFile,
        [string]$AdminPassword = "admin123"
    )

    if (!(Test-Path $BackupFile)) {
        Write-Error "Backup file not found: $BackupFile"
        return $false
    }

    try {
        Write-Warning "Restoring from $BackupFile..."
        $confirm = Read-Host "This will overwrite existing data. Continue? (yes/no)"

        if ($confirm -eq "yes") {
            Get-Content $BackupFile | docker exec -i penux-openldap ldapadd -x `
                -D "cn=admin,dc=penux,dc=uk" -w "$AdminPassword"

            if ($LASTEXITCODE -eq 0) {
                Write-Success "Restore completed successfully"
                return $true
            }
            else {
                Write-Error "Restore failed"
                return $false
            }
        }
        else {
            Write-Info "Restore cancelled"
            return $false
        }
    }
    catch {
        Write-Error "Restore error: $_"
        return $false
    }
}

# ==================== Validation Functions ====================

function Test-ExecutionPolicy {
    $policy = Get-ExecutionPolicy

    if ($policy -eq "Restricted" -or $policy -eq "AllSigned") {
        Write-Warning "PowerShell execution policy is restrictive: $policy"
        Write-Info "Run this to fix:"
        Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor $script:ColorYellow
        return $false
    }
    return $true
}

function Test-AdminPrivileges {
    $isAdmin = ([System.Security.Principal.WindowsIdentity]::GetCurrent().Groups -contains [System.Security.Principal.SecurityIdentifier]'S-1-5-32-544')
    return $isAdmin
}

# Export all functions
Export-ModuleMember -Function *
