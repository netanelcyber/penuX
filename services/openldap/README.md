# PenuX OpenLDAP Service

A complete **OpenLDAP + phpLDAPadmin** setup for the penux.uk domain, providing centralized user and group management with a web-based administrative interface.

## Features

- **OpenLDAP Server** - Lightweight, fast LDAP directory service
- **phpLDAPadmin** - Web-based LDAP management interface
- **Docker Compose** - Easy deployment and management
- **Bootstrap Data** - Pre-configured users, groups, and organizational units
- **Management Scripts** - CLI tools for user management and service operations
- **Health Checks** - Automated service health monitoring
- **TLS/SSL Ready** - Secure LDAP communications (configurable)

## Quick Start

### 1. Initialize and Start Services

```bash
cd services/openldap
chmod +x setup.sh manage.sh
./setup.sh
```

This will:
- Create `.env` file with default configuration
- Start OpenLDAP and phpLDAPadmin containers
- Initialize the LDAP directory with bootstrap data

### 2. Access the Web Interface

- **URL**: http://localhost
- **Admin DN**: `cn=admin,dc=penux,dc=uk`
- **Password**: (see `.env` file - default: `admin123`)

### 3. Manage Users and Groups

Use the management script:

```bash
./manage.sh help
```

## Configuration

### Environment Variables

Edit `.env` file to customize:

```bash
cp .env.example .env
```

Key variables:
- `LDAP_ADMIN_PASSWORD` - Admin password (change in production!)
- `LDAP_DOMAIN` - Your domain name
- `LDAP_BASE_DN` - Base Distinguished Name
- `LDAP_TLS_ENFORCE` - Enable TLS enforcement

### Port Mapping

- **LDAP**: `389` - Standard LDAP port
- **LDAPS**: `636` - Secure LDAP port
- **Web Admin**: `80` - HTTP access to phpLDAPadmin
- **Web Admin HTTPS**: `6443` - HTTPS access to phpLDAPadmin

## Default Users

Bootstrap creates sample users for testing:

| User | Email | Password | Role |
|------|-------|----------|------|
| admin | admin@penux.uk | admin123 | Administrator |
| john | john@penux.uk | admin123 | Developer |
| jane | jane@penux.uk | admin123 | DevOps Engineer |

**Note**: Change all passwords in production!

## Directory Structure

```
├── bootstrap.ldif          # Initial LDAP directory data
├── docker-compose.yml      # Docker Compose configuration
├── setup.sh               # Initial setup script
├── manage.sh              # Service management script
├── .env.example           # Environment template
└── README.md              # This file
```

## Usage

### Service Management

```bash
# Start services
./manage.sh start

# Stop services
./manage.sh stop

# Restart services
./manage.sh restart

# View status
./manage.sh status

# View logs
./manage.sh logs
./manage.sh logs-ldap
./manage.sh logs-admin

# Test LDAP connectivity
./manage.sh test
```

### User Management

```bash
# Add a new user (interactive)
./manage.sh add-user

# List all users
./manage.sh list-users

# Access LDAP container shell
./manage.sh shell-ldap

# Access phpLDAPadmin container shell
./manage.sh shell-admin
```

### Direct LDAP Commands

Execute LDAP commands directly in the container:

```bash
# Query users
docker exec penux-openldap ldapsearch -x \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 \
  -b "ou=users,dc=penux,dc=uk" \
  "(objectClass=inetOrgPerson)"

# Query groups
docker exec penux-openldap ldapsearch -x \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 \
  -b "ou=groups,dc=penux,dc=uk" \
  "(objectClass=groupOfNames)"

# Change user password
docker exec penux-openldap ldappasswd -x \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 \
  -s newpassword \
  "cn=john,ou=users,dc=penux,dc=uk"
```

## LDAP Directory Structure

```
dc=penux,dc=uk
├── ou=users
│   ├── cn=admin
│   ├── cn=john
│   └── cn=jane
├── ou=groups
│   ├── cn=penux-admins
│   ├── cn=penux-users
│   └── cn=penux-developers
└── ou=applications
    └── cn=app-service
```

## Integration with Applications

### LDAP Connection Parameters

```
Server: ldap://localhost:389
Base DN: dc=penux,dc=uk
Admin DN: cn=admin,dc=penux,dc=uk
Admin Password: (from .env)
```

### Node.js Example

```javascript
const ldap = require('ldapjs');

const client = ldap.createClient({
  url: 'ldap://localhost:389'
});

client.bind('cn=admin,dc=penux,dc=uk', 'admin123', (err) => {
  if (err) throw err;
  
  const opts = {
    filter: '(uid=john)',
    scope: 'sub'
  };
  
  client.search('ou=users,dc=penux,dc=uk', opts, (err, res) => {
    res.on('searchEntry', (entry) => {
      console.log(entry.object);
    });
  });
});
```

### Python Example

```python
import ldap

conn = ldap.initialize('ldap://localhost:389')
conn.simple_bind_s('cn=admin,dc=penux,dc=uk', 'admin123')

results = conn.search_s(
    'ou=users,dc=penux,dc=uk',
    ldap.SCOPE_SUBTREE,
    '(uid=john)'
)

for dn, attrs in results:
    print(f"DN: {dn}")
    print(f"Attributes: {attrs}")
```

## Backup and Restore

### Backup LDAP Data

```bash
# Export all data
docker exec penux-openldap slapcat > ldap_backup.ldif

# Backup volumes
docker run --rm \
  -v openldap_ldap_database:/volume \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/ldap_database_backup.tar.gz -C / volume
```

### Restore LDAP Data

```bash
# Restore from LDIF
docker exec -i penux-openldap ldapadd -x \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 < backup.ldif

# Restore volumes
docker run --rm \
  -v openldap_ldap_database:/volume \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/ldap_database_backup.tar.gz -C /
```

## Troubleshooting

### Services Won't Start

```bash
# Check logs
./manage.sh logs

# Verify Docker and Docker Compose are installed
docker --version
docker-compose --version

# Ensure ports are not in use
lsof -i :389
lsof -i :80
```

### Can't Connect to LDAP

```bash
# Test connectivity
./manage.sh test

# Check if OpenLDAP container is running
docker ps | grep openldap

# Verify network
docker network ls
docker network inspect openldap_penux_network
```

### Password Issues

```bash
# Reset admin password in container
docker exec penux-openldap ldappasswd -x \
  -D "cn=admin,dc=penux,dc=uk" \
  -w oldpassword \
  -s newpassword

# Or reset via LDIF
docker cp reset_password.ldif penux-openldap:/tmp/
docker exec penux-openldap ldapmodify -x \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 < /tmp/reset_password.ldif
```

### phpLDAPadmin Access Issues

- Clear browser cache
- Try incognito/private mode
- Check firewall rules
- Verify container is running: `docker ps | grep phpldapadmin`

## Security Considerations

### Production Deployment

1. **Change all default passwords immediately**
2. **Use strong passwords** (minimum 16 characters)
3. **Enable TLS/SSL**:
   ```bash
   # Generate certificates
   openssl req -new -x509 -days 365 -nodes \
     -out certs/server.crt -keyout certs/server.key
   
   # Set permissions
   chmod 600 certs/server.key
   ```

4. **Restrict network access**:
   ```bash
   # Only allow internal traffic
   - "389:389"  # Replace with specific IP
   - "636:636"
   ```

5. **Set resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 1G
       reservations:
         memory: 512M
   ```

6. **Use environment variables** for secrets, not hardcoded values

7. **Regular backups** of LDAP database

8. **Monitor logs** for suspicious activity

## Performance Optimization

### For Large Directories

1. **Increase container resources**:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 4G
   ```

2. **Configure LDAP caching** in docker-compose.yml

3. **Use database indexing** for frequently searched attributes

4. **Consider replication** for high availability

## Advanced Configuration

### Enable TLS

```yaml
# In docker-compose.yml
environment:
  LDAP_TLS_ENFORCE: "true"
  LDAP_TLS_VERIFY_CLIENT: "demand"
```

### Add Custom Schemas

Create `custom-schema.ldif`:

```ldif
dn: cn=custom,cn=schema,cn=config
objectClass: olcSchemaConfig
cn: custom
olcAttributeTypes: (
  1.3.6.1.4.1.9.9.999.1.1.1.1
  NAME 'customAttr'
  SYNTAX 1.3.6.1.4.1.1466.115121.1.15
  )
```

Load it:
```bash
docker exec penux-openldap ldapadd -Y EXTERNAL -H ldapi:// -f custom-schema.ldif
```

### Configure Replication

For high availability, see [OpenLDAP Replication Guide](https://www.openldap.org/doc/admin/replication.html)

## Support and Documentation

- **OpenLDAP**: https://www.openldap.org/
- **phpLDAPadmin**: https://phpldapadmin.sourceforge.io/
- **Docker Images**: 
  - https://hub.docker.com/r/osixia/openldap
  - https://hub.docker.com/r/osixia/phpldapadmin

## License

This configuration is part of the PenuX project.

## Maintenance

### Clean Up Old Data

```bash
./manage.sh clean  # WARNING: Deletes all data!
```

### View Service Health

```bash
docker-compose ps
docker stats
```

### Update Images

```bash
docker-compose pull
docker-compose up -d
```
