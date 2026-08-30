# 🚀 PenuX LDAP - Quick Reference Card

## Access Points

| Service | URL/Host | Port | Credentials |
|---------|----------|------|-------------|
| **Web UI** | localhost | 3001 | None |
| **Admin Dashboard** | localhost/enhanced.html | 3001 | None |
| **REST API** | localhost | 3000 | admin / admin123 |
| **Keycloak** | localhost | 8080 | admin / admin123 |
| **LDAP** | localhost | 389 | cn=admin,dc=penux,dc=uk / admin123 |
| **LDAPS** | localhost | 636 | cn=admin,dc=penux,dc=uk / admin123 |

---

## Common Commands

### LDAP Operations

```bash
# Test LDAP connection
ldapwhoami -H ldap://localhost:389 -D "cn=admin,dc=penux,dc=uk" -w admin123

# Test LDAPS (encrypted)
ldapwhoami -H ldaps://localhost:636 -D "cn=admin,dc=penux,dc=uk" -w admin123

# List all users
ldapsearch -H ldap://localhost:389 -D "cn=admin,dc=penux,dc=uk" -w admin123 \
  -b "ou=users,dc=penux,dc=uk" objectClass=inetOrgPerson

# List all groups
ldapsearch -H ldap://localhost:389 -D "cn=admin,dc=penux,dc=uk" -w admin123 \
  -b "ou=groups,dc=penux,dc=uk" objectClass=groupOfNames

# Search for user
ldapsearch -H ldap://localhost:389 -D "cn=admin,dc=penux,dc=uk" -w admin123 \
  -b "dc=penux,dc=uk" uid=jdoe

# Change password
ldappasswd -H ldap://localhost:389 -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 -s "NewPassword123!" "cn=admin,dc=penux,dc=uk"
```

### REST API

```bash
# Health check
curl http://localhost:3000/api/health

# List users (requires auth)
curl -u "cn=admin,dc=penux,dc=uk:admin123" http://localhost:3000/api/users

# Get specific user
curl -u "cn=admin,dc=penux,dc=uk:admin123" http://localhost:3000/api/users/jdoe

# List groups
curl -u "cn=admin,dc=penux,dc=uk:admin123" http://localhost:3000/api/groups

# Search
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  "http://localhost:3000/api/search?query=john"

# Verify credentials
curl -X POST http://localhost:3000/api/verify \
  -H "Content-Type: application/json" \
  -d '{"dn":"cn=admin,dc=penux,dc=uk","password":"admin123"}'

# Get statistics
curl -u "cn=admin,dc=penux,dc=uk:admin123" http://localhost:3000/api/stats
```

### Docker Operations

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart api

# Access container shell
docker-compose exec openldap bash

# View container status
docker-compose ps
```

---

## Configuration Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Local development setup |
| `docker-compose-public.yml` | Public internet deployment |
| `services/openldap/api/openapi.yaml` | REST API specification |
| `services/openldap/api/package.json` | API dependencies |
| `services/openldap/web/package.json` | Web UI dependencies |

---

## Key Credentials

```
Admin DN: cn=admin,dc=penux,dc=uk
Default Password: admin123
Base DN: dc=penux,dc=uk

⚠️ CHANGE IMMEDIATELY IN PRODUCTION
```

---

## Ports

| Service | Port | Type |
|---------|------|------|
| LDAP | 389 | TCP |
| LDAPS | 636 | TCP |
| Web UI | 3001 | HTTP |
| REST API | 3000 | HTTP |
| Keycloak | 8080 | HTTP |
| PostgreSQL | 5432 | TCP (internal) |

---

## Authentication Methods

### HTTP Basic Auth (for API)

```bash
# Method 1: Using -u flag
curl -u "cn=admin,dc=penux,dc=uk:admin123" http://localhost:3000/api/users

# Method 2: Using Authorization header
curl -H "Authorization: Basic Y24hYWRtaW4sZGM9cGVudXgsZGM9dWs6YWRtaW4xMjM=" \
  http://localhost:3000/api/users

# Generate Base64 (JavaScript)
btoa("cn=admin,dc=penux,dc=uk:admin123")

# Generate Base64 (Bash)
echo -n "cn=admin,dc=penux,dc=uk:admin123" | base64
```

### LDAP Bind

```bash
# LDAP bind
ldapwhoami -H ldap://localhost:389 -D "cn=admin,dc=penux,dc=uk" -w admin123

# LDAPS bind (encrypted)
ldapwhoami -H ldaps://localhost:636 -D "cn=admin,dc=penux,dc=uk" -w admin123
```

---

## Troubleshooting

### Service won't start

```bash
# Check logs
docker-compose logs openldap
docker-compose logs api
docker-compose logs web

# Check ports in use
lsof -i :389  # LDAP
lsof -i :3000 # API
lsof -i :3001 # Web UI

# Restart services
docker-compose down
docker-compose up -d
```

### Can't connect to LDAP

```bash
# Test basic connectivity
telnet localhost 389
nc -zv localhost 389

# Check if service is running
docker ps | grep openldap

# Verify credentials
ldapwhoami -H ldap://localhost:389 -D "cn=admin,dc=penux,dc=uk" -w admin123
```

### API returns 401

1. Check credentials are correct
2. Verify Base64 encoding:
   ```bash
   echo -n "cn=admin,dc=penux,dc=uk:admin123" | base64
   ```
3. Test direct LDAP connection first
4. Check API logs: `docker-compose logs api`

### Can't access web UI

1. Verify on port 3001: http://localhost:3001
2. Check service is running: `docker-compose ps`
3. Wait 30 seconds for startup
4. Clear browser cache: Ctrl+Shift+Delete

---

## Performance Tuning

```bash
# Increase LDAP query timeout
ldapwhoami -H ldap://localhost:389 -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 -W -l 10  # 10 second timeout

# Monitor LDAP performance
tail -f /var/log/syslog | grep slapd

# Check API response times
curl -w "Response time: %{time_total}s\n" -s http://localhost:3000/api/health
```

---

## Security Checklist

- [ ] Changed default admin password
- [ ] Enabled LDAPS (port 636)
- [ ] Configured firewall rules
- [ ] Set up HTTPS for web UI
- [ ] Enabled audit logging
- [ ] Set password policy
- [ ] Configured rate limiting
- [ ] Set up backups
- [ ] Reviewed ACLs
- [ ] Tested disaster recovery

---

## Documentation Map

| Need | Document |
|------|----------|
| **Deploy locally** | README-COMPLETE.md |
| **Use REST API** | API-USAGE-GUIDE.md |
| **Secure system** | SECURITY-GUIDE.md |
| **Deploy to cloud** | DEPLOYMENT-ALTERNATIVES.md |
| **Enable HTTPS** | HTTPS-FULL-SETUP.md |
| **Public access** | PUBLIC_ACCESS_SETUP.md |
| **Test endpoints** | Run `test-api.sh` |

---

## File Locations

```
Web UI:      services/openldap/web/
API:         services/openldap/api/
LDAP Config: docker-compose.yml
Tests:       services/openldap/api/test-api.sh
OpenAPI:     services/openldap/api/openapi.yaml
```

---

## Fast Tips

💡 **Enable HTTPS locally:**
```bash
# Use ngrok for HTTPS tunnel
ngrok http 3001
# Then access via https://xxx.ngrok.io
```

💡 **Test with Postman:**
1. Import: `services/openldap/api/openapi.yaml`
2. Set Auth to Basic Auth
3. Enter: `cn=admin,dc=penux,dc=uk` / `admin123`

💡 **Monitor in real-time:**
```bash
watch -n 1 'curl -s http://localhost:3000/api/stats | jq'
```

💡 **Backup LDAP data:**
```bash
docker-compose exec openldap slapcat > ldap-backup.ldif
```

💡 **Restore LDAP data:**
```bash
docker-compose exec -T openldap slapadd -l ldap-backup.ldif
```

---

## Get Help

- 📖 **Docs**: Check README-COMPLETE.md first
- 🔍 **API**: See API-USAGE-GUIDE.md
- 🔐 **Security**: Read SECURITY-GUIDE.md
- ☁️ **Deploy**: See DEPLOYMENT-ALTERNATIVES.md
- 🧪 **Test**: Run `bash services/openldap/api/test-api.sh`

---

**Everything you need in 1 page!** ✨

Last Updated: 2026-06-21
