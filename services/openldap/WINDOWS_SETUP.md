# PenuX OpenLDAP - Windows Installation Guide

Complete setup instructions for running OpenLDAP on Windows using Docker Desktop.

## Prerequisites

### System Requirements

- **OS**: Windows 10 (2004 or later) or Windows 11
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Disk Space**: 3GB free for Docker and containers
- **Virtualization**: Enabled in BIOS/UEFI
- **Network**: Internet access for downloading Docker images

### Required Software

1. **Docker Desktop for Windows**
   - Download: https://www.docker.com/products/docker-desktop
   - Version: 4.0 or later
   - Install with default settings

2. **PowerShell 5.0 or higher**
   - Built-in on Windows 10/11
   - Check version: Open PowerShell and run `$PSVersionTable.PSVersion`

## Installation Steps

### Step 1: Install Docker Desktop

1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop
2. Run the installer (`Docker Desktop Installer.exe`)
3. Follow the installation wizard
4. **Important**: During installation, ensure "WSL 2 backend" is selected
5. Accept the license agreement and click "Install"
6. Restart your computer when prompted

### Step 2: Verify Docker Installation

Open **PowerShell** and run:

```powershell
docker --version
docker-compose --version
```

Both should return version numbers. If not, Docker is not properly installed.

### Step 3: Clone or Navigate to PenuX Repository

Open PowerShell and navigate to your PenuX repository:

```powershell
cd "C:\path\to\penuX"
cd services\openldap
```

### Step 4: Run Setup Script

In PowerShell, execute the setup script:

```powershell
.\setup.ps1
```

**Important**: If you get an execution policy error, run this first:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then run setup again:

```powershell
.\setup.ps1
```

### Step 5: Wait for Services to Initialize

The setup script will start both OpenLDAP and phpLDAPadmin containers. Wait 30-40 seconds for full initialization.

Check status with:

```powershell
.\manage.ps1 status
```

### Step 6: Access the Web Interface

Open your web browser and go to:

```
http://localhost
```

Login credentials:
- **Admin DN**: `cn=admin,dc=penux,dc=uk`
- **Password**: Check in `.env` file (default: `admin123`)

## Managing Services on Windows

Use the `manage.ps1` script for service management:

### Start Services

```powershell
.\manage.ps1 start
```

### Stop Services

```powershell
.\manage.ps1 stop
```

### Restart Services

```powershell
.\manage.ps1 restart
```

### View Service Status

```powershell
.\manage.ps1 status
```

### View Logs

```powershell
# All logs
.\manage.ps1 logs

# OpenLDAP only
.\manage.ps1 logs-ldap

# phpLDAPadmin only
.\manage.ps1 logs-admin
```

### Test LDAP Connection

```powershell
.\manage.ps1 test
```

### Add New User (Interactive)

```powershell
.\manage.ps1 add-user
```

### List All Users

```powershell
.\manage.ps1 list-users
```

### Access Container Shells

```powershell
# OpenLDAP container
.\manage.ps1 shell-ldap

# phpLDAPadmin container
.\manage.ps1 shell-admin
```

## Configuration on Windows

### Edit Environment Variables

The `.env` file controls configuration:

```powershell
# Open in Notepad
notepad .env
```

Key settings:
```
LDAP_ADMIN_PASSWORD=admin123
LDAP_DOMAIN=penux.uk
LDAP_BASE_DN=dc=penux,dc=uk
```

**Change these in production!**

## Windows-Specific Tips

### Port Forwarding

Docker Desktop automatically maps ports to localhost. Access services:

| Service | Windows URL |
|---------|------------|
| LDAP | `ldap://localhost:389` |
| LDAPS | `ldaps://localhost:636` |
| Web Admin | `http://localhost:80` |
| Web Admin HTTPS | `https://localhost:6443` |

### Firewall

If you can't connect:

1. Open **Windows Defender Firewall**
2. Click "Allow an app through firewall"
3. Find and enable "Docker Desktop"
4. Restart Docker Desktop

### Storage Location

Docker volumes are stored in:

```
C:\Users\<YourUsername>\AppData\Local\Docker\volumes\
```

### WSL2 Integration

If using WSL2 (recommended):

1. Install WSL2: https://aka.ms/wsl2-kernel
2. Docker Desktop will use WSL2 automatically
3. You can also use the WSL2 terminal to run commands:

```bash
# From WSL2 terminal
cd /mnt/c/path/to/penuX/services/openldap
./manage.sh start
```

## Accessing from WSL2

If you're using WSL2, you can also use the bash scripts:

```bash
# In WSL2 terminal
cd /mnt/c/path/to/penux/services/openldap
chmod +x setup.sh manage.sh
./setup.sh
./manage.sh help
```

The containers are accessible from both Windows PowerShell and WSL2 terminals.

## Integration with Windows Applications

### LDAP Connection from Windows Apps

Use these settings when connecting from Windows applications:

```
Server: localhost:389
Base DN: dc=penux,dc=uk
Username: cn=admin,dc=penux,dc=uk
Password: (from .env file)
Use TLS: false (unless TLS is configured)
```

### Using with Windows Scheduled Tasks

To start OpenLDAP on system startup, create a scheduled task:

1. Open **Task Scheduler**
2. Right-click "Task Scheduler Library" → "Create Task"
3. Name: "Start OpenLDAP"
4. Trigger: "At system startup"
5. Action: "Start a program"
   - Program: `powershell.exe`
   - Arguments: `-NoProfile -ExecutionPolicy Bypass -File "C:\path\to\services\openldap\setup.ps1"`
6. Click OK

## Troubleshooting on Windows

### Docker Desktop Won't Start

1. Check if Virtualization is enabled in BIOS
2. Restart Docker Desktop
3. Check Event Viewer for errors
4. Try: `net start docker-desktop` in Command Prompt (as Administrator)

### Port 80/443 Already in Use

If you get "port already in use" error:

```powershell
# Find what's using the port
netstat -ano | findstr ":80"
netstat -ano | findstr ":443"

# In docker-compose.yml, change ports:
# Change "80:80" to "8080:80"
# Change "443:443" to "8443:443"
```

Then access at: `http://localhost:8080`

### Can't Access Web Interface

1. Check Docker is running: `docker-compose status`
2. Check firewall is allowing Docker
3. Try different browser (Chrome, Firefox, Edge)
4. Wait longer for services to initialize (up to 60 seconds)
5. View logs: `.\manage.ps1 logs-admin`

### LDAP Connection Refused

```powershell
# Check if OpenLDAP container is running
docker ps | Select-String openldap

# Check logs
.\manage.ps1 logs-ldap

# Test connectivity
.\manage.ps1 test
```

### Can't Change Admin Password

1. Ensure .env file has correct password
2. Container might not be fully initialized (wait 30-40 seconds)
3. Verify container is healthy:
   ```powershell
   docker ps
   # STATUS should show "healthy" or "Up"
   ```

### PowerShell Execution Policy Error

If you get an execution policy error:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or run PowerShell as Administrator:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

## Cleanup on Windows

### Stop and Remove Services

```powershell
.\manage.ps1 clean
```

This removes all containers and volumes (data will be lost).

### Manual Cleanup

```powershell
# Stop services
docker-compose down -v

# Remove images
docker rmi osixia/openldap:1.5.0
docker rmi osixia/phpldapadmin:0.9.0

# Clean up volumes
docker volume rm openldap_ldap_database
docker volume rm openldap_ldap_config
```

## Command Reference

### Docker Compose Commands

```powershell
# Start services in background
docker-compose up -d

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f openldap

# View service status
docker-compose ps

# Restart services
docker-compose restart

# Execute command in container
docker exec penux-openldap [command]
```

### Common Docker Exec Commands

```powershell
# Search LDAP directory
docker exec penux-openldap ldapsearch -x -D "cn=admin,dc=penux,dc=uk" -w admin123 -b "dc=penux,dc=uk"

# List users
docker exec penux-openldap ldapsearch -x -D "cn=admin,dc=penux,dc=uk" -w admin123 -b "ou=users,dc=penux,dc=uk"

# Change password
docker exec penux-openldap ldappasswd -x -D "cn=admin,dc=penux,dc=uk" -w admin123 -s newpassword "cn=john,ou=users,dc=penux,dc=uk"
```

## Backup and Restore on Windows

### Backup LDAP Database

```powershell
# Export LDAP data
docker exec penux-openldap slapcat | Out-File -Encoding UTF8 ldap_backup.ldif
```

### Restore LDAP Database

```powershell
# Restore from LDIF file
Get-Content ldap_backup.ldif | docker exec -i penux-openldap ldapadd -x -D "cn=admin,dc=penux,dc=uk" -w admin123
```

## Performance Optimization on Windows

### Allocate More Resources

Docker Desktop settings:

1. Right-click Docker icon → Settings
2. Resources:
   - CPUs: Increase to 4 or more
   - Memory: Increase to 4GB or more
   - Disk: Increase to 20GB or more
3. Click "Apply & Restart"

### Enable WSL2 Backend

For better performance:

1. Docker Desktop Settings → General
2. Enable "Use the new Virtualization framework"
3. Check "Use WSL 2 based engine"
4. Click "Apply & Restart"

## Advanced: Running from Command Prompt

If you prefer Command Prompt (cmd.exe), use these commands:

```cmd
# Navigate to directory
cd C:\path\to\penuX\services\openldap

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# View status
docker-compose ps
```

However, PowerShell is recommended for better features and color output.

## Additional Resources

- **Docker Desktop Documentation**: https://docs.docker.com/desktop/install/windows-install/
- **WSL2 Setup**: https://aka.ms/wsl2-kernel
- **OpenLDAP Documentation**: https://www.openldap.org/
- **phpLDAPadmin**: https://phpldapadmin.sourceforge.io/

## Next Steps

1. ✅ Setup complete with Docker Desktop
2. ✅ Services running with PowerShell management
3. Access web interface at http://localhost
4. Configure users and groups
5. Integrate with your applications
6. Plan backup and security strategy
7. Review [README.md](README.md) for full documentation
