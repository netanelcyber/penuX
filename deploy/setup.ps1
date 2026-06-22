# PenuX Windows Setup — run once as Administrator
# Right-click → Run with PowerShell

$ErrorActionPreference = "Stop"
$INSTALL_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

Write-Host ""
Write-Host "================================================"
Write-Host "  PenuX Setup"
Write-Host "================================================"
Write-Host ""

Set-Location $INSTALL_DIR

# ── Python ────────────────────────────────────────────────────────────────
Write-Host "Checking Python..."
$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(9|10|11|12|13)") {
            $python = $cmd
            Write-Host "  Found: $ver"
            break
        }
    } catch {}
}

if (-not $python) {
    Write-Host "  Python not found. Installing via winget..."
    try {
        winget install Python.Python.3.11 -e --silent --accept-package-agreements --accept-source-agreements
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + `
                    [System.Environment]::GetEnvironmentVariable("PATH","User")
        $python = "python"
        Write-Host "  Python installed."
    } catch {
        Write-Host ""
        Write-Host "  Could not auto-install Python."
        Write-Host "  Please install manually: https://www.python.org/downloads/"
        Write-Host "  Then run this script again."
        Read-Host "  Press Enter to exit"
        exit 1
    }
}

# ── cloudflared ───────────────────────────────────────────────────────────
$cloudflared = "$INSTALL_DIR\cloudflared.exe"
if (-not (Test-Path $cloudflared)) {
    Write-Host "Downloading cloudflared..."
    Invoke-WebRequest -Uri $CLOUDFLARED_URL -OutFile $cloudflared -UseBasicParsing
    Write-Host "  Downloaded."
} else {
    Write-Host "cloudflared: already present."
}

# ── Directories ───────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path "$INSTALL_DIR\mail"  | Out-Null
New-Item -ItemType Directory -Force -Path "$INSTALL_DIR\logs"  | Out-Null

# ── Windows Firewall — allow localhost ports ─────────────────────────────
Write-Host "Configuring firewall (localhost only)..."
try {
    netsh advfirewall firewall add rule name="PenuX IMAP" dir=in action=allow protocol=TCP localport=143 remoteip=127.0.0.1 | Out-Null
    netsh advfirewall firewall add rule name="PenuX Webmail" dir=in action=allow protocol=TCP localport=8080 remoteip=127.0.0.1 | Out-Null
} catch {
    Write-Host "  (firewall rules skipped — run as Administrator to set them)"
}

# ── Auto-start at login ───────────────────────────────────────────────────
Write-Host "Adding PenuX to Windows Startup..."
$startupFolder = [Environment]::GetFolderPath("Startup")
$shortcutPath  = "$startupFolder\PenuX.lnk"

$shell    = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath       = "$INSTALL_DIR\start.bat"
$shortcut.WorkingDirectory = $INSTALL_DIR
$shortcut.WindowStyle      = 7  # minimised
$shortcut.Description      = "PenuX IMAP + Webmail"
$shortcut.Save()
Write-Host "  Shortcut added: $shortcutPath"

# ── Launch now ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "Setup complete! Starting PenuX now..."
Write-Host ""
Start-Process "$INSTALL_DIR\start.bat"

Write-Host "================================================"
Write-Host "  PenuX is running!"
Write-Host ""
Write-Host "  Webmail:  https://mail.penux.uk"
Write-Host "  (starts automatically every time you log in)"
Write-Host "================================================"
Write-Host ""
Read-Host "Press Enter to close"
