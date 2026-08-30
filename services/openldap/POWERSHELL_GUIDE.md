# PenuX OpenLDAP - PowerShell Scripts Guide

Complete guide to PowerShell scripts for managing OpenLDAP on Windows.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| **setup.ps1** | Initial Docker setup and service launch |
| **manage.ps1** | Service management and user operations |
| **backup.ps1** | Database backup and restore operations |
| **utils.ps1** | Shared utility functions (imported by other scripts) |

## Getting Started

### Prerequisites

- Windows 10 (2004+) or Windows 11
- PowerShell 5.0+ (built-in on Windows 10/11)
- Docker Desktop for Windows installed
- Administrator access for some operations

### First Run

```powershell
# Navigate to OpenLDAP directory
cd C:\path\to\penuX\services\openldap

# Run setup (creates .env and starts services)
.\setup.ps1

# For help on any script
.\setup.ps1 -Help
.\manage.ps1 help
.\backup.ps1 help
```

## setup.ps1 - Initial Setup

Complete initialization of OpenLDAP on Docker.

### Basic Usage

```powershell
# Standard setup
.\setup.ps1

# Create new .env file (skip existing)
.\setup.ps1 -Force

# Minimal output
.\setup.ps1 -Silent

# Show help
.\setup.ps1 -Help
```

### What It Does

1. **Pre-flight Checks**
   - Verifies Docker Desktop installation
   - Checks Docker daemon is running
   - Validates PowerShell execution policy
   - Confirms Docker Compose availability

2. **Configuration**
   - Creates `.env` from template if missing
   - Creates `certs/` directory for certificates
   - Verifies all required files exist

3. **Service Launch**
   - Starts OpenLDAP container
   - Starts phpLDAPadmin web interface
   - Waits for services to become healthy (up to 60 seconds)

4. **Information Display**
   - Shows access URLs
   - Lists default credentials
   - Displays sample users
   - Provides next steps

### Features

- **Docker Verification**: Checks both installation and running daemon
- **Automatic Initialization**: Creates required files if missing
- **Service Health Monitoring**: Waits for containers to be ready
- **Detailed Feedback**: Color-coded output with icons
- **Error Handling**: Clear error messages with solutions
- **Silent Mode**: Minimal output for scripting

## manage.ps1 - Service Management

Day-to-day management of OpenLDAP services.

### Service Commands

```powershell
# Start services
.\manage.ps1 start

# Stop services
.\manage.ps1 stop

# Restart services
.\manage.ps1 restart

# Show current status
.\manage.ps1 status

# View all logs
.\manage.ps1 logs

# View OpenLDAP logs only
.\manage.ps1 logs-ldap

# View phpLDAPadmin logs only
.\manage.ps1 logs-admin
```

### Connectivity Commands

```powershell
# Test LDAP connection
.\manage.ps1 test

# List all users
.\manage.ps1 list-users

# List all groups
.\manage.ps1 list-groups
```

### User Management

```powershell
# Add new user (interactive)
.\manage.ps1 add-user

# Will prompt for:
#  - Username (uid)
#  - Full name (cn)
#  - Email address
#  - Password

# Remove user (interactive)
.\manage.ps1 remove-user

# Change user password (interactive)
.\manage.ps1 change-password

# Set account status (active/inactive)
.\manage.ps1 set-account-status
```

### Container Access

```powershell
# Open shell in OpenLDAP container
.\manage.ps1 shell-ldap

# Open shell in phpLDAPadmin container
.\manage.ps1 shell-admin
```

### System Commands

```powershell
# Delete all containers and data (WARNING!)
.\manage.ps1 clean

# Show all commands
.\manage.ps1 help
```

## backup.ps1 - Backup and Restore

Backup and restore LDAP database.

### Backup Operations

```powershell
# Create uncompressed backup
.\backup.ps1 backup

# Create backup with specific filename
.\backup.ps1 backup -BackupFile "backup_important.ldif"

# Create compressed backup (.tar.gz)
.\backup.ps1 backup -Compress

# Non-interactive backup (for scripts)
.\backup.ps1 backup -Auto
```

### Backup Features

- **Automatic Timestamping**: Files named `ldap_backup_YYYY-MM-DD_HHMMSS.ldif`
- **Compression Support**: Optional gzip compression for storage
- **Size Verification**: Checks backup isn't empty
- **Full Export**: Complete LDAP directory structure

### Restore Operations

```powershell
# Restore from backup file
.\backup.ps1 restore "ldap_backup_2026-06-20_140000.ldif"

# Restore without confirmation prompt
.\backup.ps1 restore "ldap_backup_2026-06-20_140000.ldif" -Auto

# Restore from compressed backup
.\backup.ps1 restore "ldap_backup_2026-06-20_140000.ldif.gz"
```

### Restore Features

- **Compression Support**: Automatically detects and decompresses `.gz` files
- **Safety Prompt**: Requires confirmation before overwriting data
- **Error Recovery**: Clear error messages if restore fails
- **Cleanup**: Removes temporary files after completion

### Scheduled Backups

```powershell
# Set up automatic daily backups via Windows Task Scheduler
.\backup.ps1 schedule

# Will prompt for backup time (default: 02:00 AM)

# Manage scheduled task:
Enable-ScheduledTask -TaskName "PenuX-LDAP-DailyBackup"
Disable-ScheduledTask -TaskName "PenuX-LDAP-DailyBackup"
Start-ScheduledTask -TaskName "PenuX-LDAP-DailyBackup"
Unregister-ScheduledTask -TaskName "PenuX-LDAP-DailyBackup"
```

### Scheduled Backup Features

- **Daily Automation**: Runs at specified time each day
- **Timestamped Files**: Each backup has unique filename
- **Network Detection**: Only runs if network is available
- **Recovery**: Automatically retries if system is asleep
- **Task Management**: Easy enable/disable/delete

## utils.ps1 - Utility Functions

Shared functions used by other scripts. Imported automatically.

### Color Output Functions

```powershell
Write-Success "Operation completed"     # Green ✓
Write-Error "Something went wrong"      # Red ✗
Write-Warning "Be careful with this"    # Yellow ⚠
Write-Info "Here's some information"    # Cyan ℹ
Write-Header "Section Title"            # Cyan header with separators
```

### Docker Functions

```powershell
Test-DockerInstalled          # Check if Docker is installed
Test-DockerRunning           # Check if Docker daemon is running
Test-DockerComposeInstalled  # Check if Docker Compose exists
Get-DockerVersion            # Get Docker version string
Get-DockerComposeVersion     # Get Docker Compose version string
```

### Environment Functions

```powershell
Get-EnvVariable "VAR_NAME" "default"  # Get from .env file
Load-EnvFile                          # Load all .env variables
Create-EnvFile                        # Create .env from template
```

### Service Functions

```powershell
Test-ServicePort 389                  # Check if port is in use
Get-ServiceStatus                     # Get Docker Compose status
Wait-ForService "openldap" 60 3       # Wait for service (timeout, interval)
```

### LDAP Functions

```powershell
Test-LDAPConnection "password"        # Test LDAP connectivity
Search-LDAPUsers "password"           # List all users
Search-LDAPGroups "password"          # List all groups
```

### File Functions

```powershell
Create-DirectoryIfNotExists "path"    # Create dir if needed
Ensure-Executable "file.ps1"          # Verify file exists
```

### Validation Functions

```powershell
Test-ExecutionPolicy                  # Check PS execution policy
Test-AdminPrivileges                  # Check if running as admin
```

### Backup Functions

```powershell
Backup-LDAPDatabase "file.ldif"       # Create backup
Restore-LDAPDatabase "file.ldif"      # Restore backup
```

## Common Workflows

### Initial Setup

```powershell
# 1. Run setup
.\setup.ps1

# 2. Wait for initialization (30-40 seconds)
Start-Sleep -Seconds 40

# 3. Check status
.\manage.ps1 status

# 4. Test connection
.\manage.ps1 test

# 5. Access web UI
Start-Process "http://localhost"
```

### Daily Operations

```powershell
# Check status
.\manage.ps1 status

# View logs
.\manage.ps1 logs

# Add new user
.\manage.ps1 add-user

# List users
.\manage.ps1 list-users

# Backup before changes
.\backup.ps1 backup

# Restore if needed
.\backup.ps1 restore "ldap_backup_*.ldif"
```

### Backup Strategy

```powershell
# Set up automatic daily backups
.\backup.ps1 schedule

# Manual backups when needed
.\backup.ps1 backup -Compress

# List recent backups
Get-ChildItem ldap_backup*.ldif* | Sort-Object LastWriteTime -Descending

# Restore from recent backup
$latest = (Get-ChildItem ldap_backup*.ldif* | Sort-Object LastWriteTime -Descending)[0]
.\backup.ps1 restore $latest.Name
```

### Troubleshooting

```powershell
# Check all services
.\manage.ps1 status
docker-compose ps

# View OpenLDAP logs
.\manage.ps1 logs-ldap

# View admin interface logs
.\manage.ps1 logs-admin

# Test LDAP connection
.\manage.ps1 test

# Open container shell for debugging
.\manage.ps1 shell-ldap

# Check Docker daemon
docker info
```

## Advanced Usage

### Custom Functions

Create a custom script file `my-commands.ps1`:

```powershell
# Import utilities
. ".\utils.ps1"

# Use utility functions
if (Test-DockerRunning) {
    Write-Success "Docker is ready"
}

# Custom logic
$version = Get-DockerVersion
Write-Info "Using: $version"
```

### Automation

Combine with Windows Task Scheduler:

```powershell
# Create backup task
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-File C:\path\to\backup.ps1 backup -Auto"
$trigger = New-ScheduledTaskTrigger -Daily -At 02:00
Register-ScheduledTask -TaskName "LDAP-Backup" -Action $action -Trigger $trigger
```

### Monitoring Script

Create `monitor.ps1`:

```powershell
. ".\utils.ps1"

while ($true) {
    Clear-Host
    Write-Header "LDAP Status Monitor"

    Write-Info "Services:"
    docker-compose ps

    Write-Info ""
    Write-Info "LDAP Connection:"
    if (Test-LDAPConnection) {
        Write-Success "Connected"
    }
    else {
        Write-Error "Not responding"
    }

    Write-Info ""
    Write-Info "Updating in 30 seconds... (Ctrl+C to stop)"
    Start-Sleep -Seconds 30
}
```

Run with: `.\monitor.ps1`

### Batch Operations

```powershell
# Add multiple users from CSV
$users = Import-Csv "users.csv" # Requires: uid, cn, email, password columns

foreach ($user in $users) {
    Write-Info "Adding user: $($user.uid)"
    # Call add-user function from manage.ps1
}
```

## Error Messages and Solutions

| Error | Solution |
|-------|----------|
| "Docker is not installed" | Install Docker Desktop from docker.com |
| "Docker daemon is not running" | Start Docker Desktop application |
| "execution policy" | Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| "port already in use" | Edit docker-compose.yml and change port mappings |
| "LDAP connection refused" | Wait 30-40s for initialization, check `.\manage.ps1 logs` |
| "Cannot find module utils.ps1" | Run scripts from services\openldap directory |

## Tips and Tricks

### Create Shortcuts

Create a shortcut to open PowerShell in the correct directory:

```
Target: powershell.exe -NoExit -Command "cd C:\path\to\services\openldap"
Start in: C:\path\to\services\openldap
```

### Quick Commands

Add to PowerShell profile (`$PROFILE`):

```powershell
function Start-LDAP { Push-Location C:\path\to\services\openldap; .\manage.ps1 start }
function Stop-LDAP { Push-Location C:\path\to\services\openldap; .\manage.ps1 stop }
function LDAP-Status { Push-Location C:\path\to\services\openldap; .\manage.ps1 status }
function LDAP-Logs { Push-Location C:\path\to\services\openldap; .\manage.ps1 logs }
```

### Colored Output

All scripts use color-coded output:
- 🟢 **Green** - Success
- 🔴 **Red** - Errors
- 🟡 **Yellow** - Warnings
- 🔵 **Cyan** - Information

## File Structure

```
services/openldap/
├── setup.ps1              # Initial setup
├── manage.ps1             # Service management
├── backup.ps1             # Backup/restore
├── utils.ps1              # Shared functions
├── docker-compose.yml
├── bootstrap.ldif
├── .env.example
└── README.md
```

## Performance Optimization

### Allocate More Resources

Edit Docker Desktop settings:
- CPUs: Increase to 4+
- Memory: Increase to 4GB+
- Disk: Increase to 20GB+

Check in PowerShell:

```powershell
docker system df        # Docker disk usage
docker stats           # Live resource usage
```

## Security Considerations

### Password Management

```powershell
# Use secure strings for passwords
$password = Read-Host "Enter password" -AsSecureString

# Never log passwords
Write-Info "User created"  # Good
Write-Info "Password: $pwd"  # Bad
```

### Backup Security

```powershell
# Encrypt backups
$backup = Get-Content "ldap_backup.ldif" -Raw
$bytes = [System.Text.Encoding]::UTF8.GetBytes($backup)
# Encrypt using Windows Data Protection API (DPAPI)
```

## Troubleshooting Scripts

### Enable Debug Output

```powershell
# Run with debug output
$DebugPreference = "Continue"
.\manage.ps1 status
```

### Script Errors

```powershell
# Show full error details
$ErrorActionPreference = "Continue"
.\setup.ps1
```

## Support and Resources

- **PowerShell Docs**: https://docs.microsoft.com/powershell/
- **Docker Desktop**: https://www.docker.com/products/docker-desktop
- **OpenLDAP**: https://www.openldap.org/
- **Windows Task Scheduler**: Built-in Windows utility

## Next Steps

1. Run `.\setup.ps1` to initialize
2. Use `.\manage.ps1 help` for daily operations
3. Set up backups with `.\backup.ps1 schedule`
4. Access web interface at http://localhost
5. Read [README.md](README.md) for more information
