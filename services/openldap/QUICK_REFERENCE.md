# PenuX OpenLDAP - Quick Reference

## Windows (PowerShell)

```powershell
# Initial setup
.\setup.ps1

# Manage services
.\manage.ps1 start              # Start services
.\manage.ps1 stop               # Stop services
.\manage.ps1 restart            # Restart services
.\manage.ps1 status             # Show status
.\manage.ps1 logs               # View all logs
.\manage.ps1 logs-ldap          # View LDAP logs
.\manage.ps1 logs-admin         # View phpLDAPadmin logs
.\manage.ps1 test               # Test LDAP connection
.\manage.ps1 add-user           # Add new user
.\manage.ps1 list-users         # List all users
.\manage.ps1 shell-ldap         # Open LDAP container shell
.\manage.ps1 shell-admin        # Open Admin container shell
.\manage.ps1 clean              # Delete all data (WARNING!)
```

## Linux/macOS (Bash)

```bash
# Initial setup
chmod +x setup.sh manage.sh
./setup.sh

# Manage services
./manage.sh start              # Start services
./manage.sh stop               # Stop services
./manage.sh restart            # Restart services
./manage.sh status             # Show status
./manage.sh logs               # View all logs
./manage.sh logs-ldap          # View LDAP logs
./manage.sh logs-admin         # View phpLDAPadmin logs
./manage.sh test               # Test LDAP connection
./manage.sh add-user           # Add new user
./manage.sh list-users         # List all users
./manage.sh shell-ldap         # Open LDAP container shell
./manage.sh shell-admin        # Open Admin container shell
./manage.sh clean              # Delete all data (WARNING!)
```

## Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Web Admin (HTTP) | http://localhost | See .env |
| Web Admin (HTTPS) | https://localhost:6443 | See .env |
| LDAP | ldap://localhost:389 | cn=admin,dc=penux,dc=uk |
| LDAPS | ldaps://localhost:636 | cn=admin,dc=penux,dc=uk |

## Default Users

| User | Email | Password |
|------|-------|----------|
| admin | admin@penux.uk | admin123 |
| john | john@penux.uk | admin123 |
| jane | jane@penux.uk | admin123 |

⚠️ **Change passwords in production!**

## Configuration

```bash
# Edit environment
notepad .env            # Windows
nano .env               # Linux/macOS

# Key settings
LDAP_ADMIN_PASSWORD=admin123
LDAP_DOMAIN=penux.uk
LDAP_BASE_DN=dc=penux,dc=uk
```

## Direct Docker Commands

```powershell
# Windows (PowerShell)
docker-compose ps                                    # Status
docker-compose logs -f                               # All logs
docker-compose logs -f openldap                      # LDAP logs
docker exec penux-openldap ldapsearch -x \
  -D "cn=admin,dc=penux,dc=uk" -w admin123 \
  -b "ou=users,dc=penux,dc=uk"                      # Search users
```

```bash
# Linux/macOS
docker-compose ps
docker-compose logs -f
docker-compose logs -f openldap
docker exec penux-openldap ldapsearch -x \
  -D "cn=admin,dc=penux,dc=uk" -w admin123 \
  -b "ou=users,dc=penux,dc=uk"
```

## Troubleshooting

| Issue | Windows | Linux/macOS |
|-------|---------|------------|
| Docker not found | Install Docker Desktop | Install Docker Engine |
| Port in use | Change in docker-compose.yml | Change in docker-compose.yml |
| Services won't start | Check Docker Desktop is running | Check Docker daemon status |
| Can't access web UI | Check firewall allows Docker | Check firewall rules |
| Connection refused | Wait 30-40s for init | Wait 30-40s for init |
| View detailed help | `.\manage.ps1 help` | `./manage.sh help` |

## Common Tasks

### Add User (Interactive)
```powershell
# Windows
.\manage.ps1 add-user

# Linux/macOS
./manage.sh add-user
```

### Change Password
```powershell
# Windows
docker exec penux-openldap ldappasswd -x `
  -D "cn=admin,dc=penux,dc=uk" -w admin123 `
  -s newpassword "cn=john,ou=users,dc=penux,dc=uk"

# Linux/macOS
docker exec penux-openldap ldappasswd -x \
  -D "cn=admin,dc=penux,dc=uk" -w admin123 \
  -s newpassword "cn=john,ou=users,dc=penux,dc=uk"
```

### Backup Database
```powershell
# Windows
docker exec penux-openldap slapcat | Out-File -Encoding UTF8 backup.ldif

# Linux/macOS
docker exec penux-openldap slapcat > backup.ldif
```

### Stop Services
```powershell
# Windows
.\manage.ps1 stop
docker-compose down

# Linux/macOS
./manage.sh stop
docker-compose down
```

## Documentation

- [Full Documentation](README.md) - Complete setup and usage guide
- [Windows Setup Guide](WINDOWS_SETUP.md) - Detailed Windows instructions
- [OpenLDAP Docs](https://www.openldap.org/) - Official OpenLDAP documentation
- [phpLDAPadmin](https://phpldapadmin.sourceforge.io/) - Web admin tool

## Directory Structure

```
dc=penux,dc=uk
├── ou=users
│   ├── cn=admin          (Administrator)
│   ├── cn=john           (Developer)
│   └── cn=jane           (DevOps Engineer)
├── ou=groups
│   ├── cn=penux-admins
│   ├── cn=penux-users
│   └── cn=penux-developers
└── ou=applications
    └── cn=app-service
```

## Need Help?

1. Check logs: `.\manage.ps1 logs` (Windows) or `./manage.sh logs` (Linux/macOS)
2. Test connection: `.\manage.ps1 test` or `./manage.sh test`
3. See detailed help: `.\manage.ps1 help` or `./manage.sh help`
4. Read [README.md](README.md) or [WINDOWS_SETUP.md](WINDOWS_SETUP.md)
