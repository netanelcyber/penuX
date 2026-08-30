# PenuX OpenLDAP Backup and Restore Script for PowerShell

param(
    [Parameter(Position = 0)]
    [ValidateSet('backup', 'restore', 'schedule', 'help')]
    [string]$Command = "help",

    [string]$BackupFile = "ldap_backup.ldif",
    [switch]$Auto,
    [switch]$Compress
)

# Import utilities
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
. "$scriptPath\utils.ps1"

function Show-Help {
    Write-Host @"
PenuX OpenLDAP Backup and Restore Script

Usage: .\backup.ps1 [command] [options]

Commands:
  backup [options]    Create backup of LDAP database
  restore [file]      Restore from backup file
  schedule            Set up automatic daily backups
  help                Show this help message

Backup Options:
  -BackupFile FILE    Output file name (default: ldap_backup.ldif)
  -Compress          Create compressed backup (.tar.gz)
  -Auto              Non-interactive mode

Examples:
  .\backup.ps1 backup                              # Backup to ldap_backup.ldif
  .\backup.ps1 backup -BackupFile backup_2026.ldif
  .\backup.ps1 backup -Compress                    # Create compressed backup
  .\backup.ps1 restore ldap_backup.ldif            # Restore from file
  .\backup.ps1 schedule                            # Set up daily backups

Features:
  - Full LDAP database export/import
  - Optional gzip compression
  - Automatic timestamped backups
  - Scheduled daily backups (Windows Task Scheduler)
  - Restore with confirmation prompt
  - Backup verification

"@
}

function Backup-LDAP {
    param(
        [string]$OutputFile = "ldap_backup.ldif",
        [bool]$Compress = $false
    )

    Write-Header "LDAP Database Backup"

    # Load admin password from .env
    $adminPassword = Get-EnvVariable "LDAP_ADMIN_PASSWORD" "admin123"

    # Generate timestamped filename
    $timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
    $baseFile = [System.IO.Path]::GetFileNameWithoutExtension($OutputFile)
    $backupFile = "$baseFile`_$timestamp.ldif"

    Write-Info "Backing up LDAP database..."
    Write-Info "Output: $backupFile"

    try {
        # Execute slapcat in container
        $backup = docker exec penux-openldap slapcat 2>$null

        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to export LDAP data"
            Write-Info "Verify OpenLDAP container is running: docker-compose ps"
            return $false
        }

        # Write to file
        $backup | Out-File -Encoding UTF8 -FilePath $backupFile -NoNewline

        # Check file size
        $fileSize = (Get-Item $backupFile).Length
        if ($fileSize -lt 100) {
            Write-Error "Backup file too small (possibly empty): $fileSize bytes"
            Remove-Item $backupFile -ErrorAction SilentlyContinue
            return $false
        }

        Write-Success "Backup created: $backupFile ($fileSize bytes)"

        # Optional compression
        if ($Compress) {
            Write-Info "Compressing backup..."
            # Note: PowerShell 5+ has built-in compression
            $compressedFile = "$backupFile.gz"

            try {
                $sourceFile = Get-Item $backupFile
                $source = [System.IO.File]::ReadAllBytes($sourceFile.FullName)
                $destination = [System.IO.File]::Create($compressedFile)
                $gzipStream = New-Object System.IO.Compression.GZipStream $destination, ([System.IO.Compression.CompressionMode]::Compress)
                $gzipStream.Write($source, 0, $source.Length)
                $gzipStream.Dispose()
                $destination.Dispose()

                $compressedSize = (Get-Item $compressedFile).Length
                $ratio = [math]::Round(($compressedSize / $fileSize) * 100, 2)

                Write-Success "Compressed: $compressedFile ($compressedSize bytes, $ratio% of original)"
                Remove-Item $backupFile -ErrorAction SilentlyContinue
            }
            catch {
                Write-Warning "Compression failed: $_"
                Write-Info "Keeping uncompressed backup: $backupFile"
            }
        }

        return $true
    }
    catch {
        Write-Error "Backup error: $_"
        return $false
    }
}

function Restore-LDAP {
    param(
        [string]$BackupFile,
        [bool]$Auto = $false
    )

    Write-Header "LDAP Database Restore"

    if (!(Test-Path $BackupFile)) {
        Write-Error "Backup file not found: $BackupFile"
        return $false
    }

    $fileSize = (Get-Item $BackupFile).Length
    Write-Info "Restore file: $BackupFile ($fileSize bytes)"

    # Handle compressed files
    $restoreFile = $BackupFile
    if ($BackupFile -like "*.gz") {
        Write-Info "Decompressing backup..."
        $decompressFile = $BackupFile -replace '\.gz$', '.ldif.tmp'

        try {
            $source = [System.IO.File]::OpenRead($BackupFile)
            $destination = [System.IO.File]::Create($decompressFile)
            $gzipStream = New-Object System.IO.Compression.GZipStream $source, ([System.IO.Compression.CompressionMode]::Decompress)
            $gzipStream.CopyTo($destination)
            $gzipStream.Dispose()
            $source.Dispose()
            $destination.Dispose()

            $restoreFile = $decompressFile
            Write-Success "Decompressed to temporary file"
        }
        catch {
            Write-Error "Decompression failed: $_"
            return $false
        }
    }

    # Confirmation
    if (!$Auto) {
        Write-Warning "⚠️ This will OVERWRITE all existing LDAP data!"
        Write-Host ""
        $confirm = Read-Host "Type 'yes' to continue with restore"

        if ($confirm -ne "yes") {
            Write-Info "Restore cancelled"
            if ($restoreFile -ne $BackupFile) {
                Remove-Item $restoreFile -ErrorAction SilentlyContinue
            }
            return $false
        }
    }

    Write-Info "Starting restore..."

    try {
        # Load admin password from .env
        $adminPassword = Get-EnvVariable "LDAP_ADMIN_PASSWORD" "admin123"

        # Read and import
        $content = Get-Content $restoreFile -Raw
        $content | docker exec -i penux-openldap ldapadd -x `
            -D "cn=admin,dc=penux,dc=uk" -w "$adminPassword" 2>&1

        if ($LASTEXITCODE -eq 0) {
            Write-Success "Restore completed successfully"

            # Cleanup
            if ($restoreFile -ne $BackupFile) {
                Remove-Item $restoreFile -ErrorAction SilentlyContinue
            }

            return $true
        }
        else {
            Write-Error "Restore failed with exit code: $LASTEXITCODE"
            return $false
        }
    }
    catch {
        Write-Error "Restore error: $_"
        return $false
    }
}

function Schedule-DailyBackup {
    Write-Header "Schedule Daily LDAP Backups"

    Write-Info "This will create a Windows Task Scheduler task for daily backups"
    Write-Host ""

    $taskName = "PenuX-LDAP-DailyBackup"
    $time = Read-Host "Backup time (HH:MM, default 02:00)"
    if ([string]::IsNullOrWhiteSpace($time)) {
        $time = "02:00"
    }

    $scriptFullPath = (Get-Item $MyInvocation.MyCommand.Path).FullName

    Write-Info "Creating scheduled task..."
    Write-Info "Task Name: $taskName"
    Write-Info "Time: $time daily"
    Write-Info "Script: $scriptFullPath"

    # Create task action
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptFullPath`" backup -Auto"

    # Create task trigger (daily)
    $trigger = New-ScheduledTaskTrigger -Daily -At $time

    # Create task settings
    $settings = New-ScheduledTaskSettingsSet -RunOnlyIfNetworkAvailable -StartWhenAvailable

    # Register task
    try {
        Register-ScheduledTask -TaskName $taskName `
            -Action $action `
            -Trigger $trigger `
            -Settings $settings `
            -Description "Automatic LDAP database backup for PenuX" `
            -Force | Out-Null

        Write-Success "Scheduled task created: $taskName"
        Write-Info "Backups will run daily at $time"
        Write-Info "Backup files will be saved to: $(Get-Location)\ldap_backup_*.ldif"

        Write-Host ""
        Write-Host "Management Commands:" -ForegroundColor Cyan
        Write-Host "  Enable task:   Enable-ScheduledTask -TaskName '$taskName'" -ForegroundColor White
        Write-Host "  Disable task:  Disable-ScheduledTask -TaskName '$taskName'" -ForegroundColor White
        Write-Host "  Run now:       Start-ScheduledTask -TaskName '$taskName'" -ForegroundColor White
        Write-Host "  Delete task:   Unregister-ScheduledTask -TaskName '$taskName'" -ForegroundColor White
        Write-Host "  View history:  Get-ScheduledTaskInfo -TaskName '$taskName'" -ForegroundColor White
    }
    catch {
        Write-Error "Failed to create scheduled task: $_"
        Write-Info "You may need to run PowerShell as Administrator"
        return $false
    }

    return $true
}

# Main command handling
switch ($Command.ToLower()) {
    "backup" {
        $result = Backup-LDAP -OutputFile $BackupFile -Compress $Compress
        exit ($result ? 0 : 1)
    }

    "restore" {
        if ($BackupFile -eq "ldap_backup.ldif") {
            Write-Error "Backup file not specified"
            Write-Info "Usage: .\backup.ps1 restore <filename>"
            exit 1
        }
        $result = Restore-LDAP -BackupFile $BackupFile -Auto $Auto
        exit ($result ? 0 : 1)
    }

    "schedule" {
        $result = Schedule-DailyBackup
        exit ($result ? 0 : 1)
    }

    "help" {
        Show-Help
        exit 0
    }

    default {
        Write-Error "Unknown command: $Command"
        Show-Help
        exit 1
    }
}
