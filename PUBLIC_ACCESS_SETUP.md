# 🌐 Public Access Setup for LDAP + Keycloak

Complete guide to make your LDAP directory accessible from the internet with proper authentication.

---

## 📋 Prerequisites

- ✅ **Public IP address**: `curl ifconfig.me`
- ✅ **Domain name**: `penux.uk`
- ✅ **Cloudflare account**: Already configured
- ✅ **Router access**: For port forwarding
- ✅ **Firewall access**: For rules (if applicable)

---

## 🔧 Step 1: Update Docker Compose for Public Access

### Use the Public Configuration

```bash
# Replace the old docker-compose.yml
cp docker-compose-public.yml docker-compose.yml

# Or manually update:
# - Port 389 → 0.0.0.0:389:389 (LDAP)
# - Port 636 → 0.0.0.0:636:636 (LDAPS)
# - Port 80 → 0.0.0.0:80:8080 (HTTP)
```

**Key Changes:**
```yaml
# LDAP - Now public
ports:
  - "0.0.0.0:389:389"    # Public LDAP
  - "0.0.0.0:636:636"    # Public LDAPS (Secure)

# Keycloak - Now public
ports:
  - "0.0.0.0:80:8080"    # Public HTTP
```

---

## 🛜 Step 2: Configure Router Port Forwarding

### For Most Home Routers

1. **Access your router**: Usually `192.168.1.1` or `192.168.0.1`
2. **Find Port Forwarding section**: Settings → Port Forwarding
3. **Add these forwarding rules:**

| External Port | Internal IP | Internal Port | Protocol | Purpose |
|--------------|-------------|--------------|----------|---------|
| 80 | 192.168.1.100 | 8080 | TCP | Keycloak HTTP |
| 389 | 192.168.1.100 | 389 | TCP | LDAP |
| 636 | 192.168.1.100 | 636 | TCP | LDAPS (Secure) |

**Note:** Replace `192.168.1.100` with your machine's local IP

### **Find Your Local IP**

```bash
# Windows
ipconfig | findstr IPv4

# Linux/macOS
ifconfig | grep inet

# Should show something like: 192.168.1.100
```

---

## 🔒 Step 3: Configure Firewall

### Windows Firewall

```powershell
# Allow LDAP (port 389)
New-NetFirewallRule -DisplayName "LDAP" -Direction Inbound -LocalPort 389 -Protocol TCP -Action Allow

# Allow LDAPS (port 636)
New-NetFirewallRule -DisplayName "LDAPS" -Direction Inbound -LocalPort 636 -Protocol TCP -Action Allow

# Allow HTTP (port 80)
New-NetFirewallRule -DisplayName "HTTP" -Direction Inbound -LocalPort 80 -Protocol TCP -Action Allow
```

### Linux UFW

```bash
sudo ufw allow 80/tcp
sudo ufw allow 389/tcp
sudo ufw allow 636/tcp
sudo ufw reload
```

### macOS

```bash
# Disable macOS firewall (if enabled)
sudo defaults write /Library/Preferences/com.apple.alf globalstate -int 0
```

---

## 🌐 Step 4: Configure Cloudflare DNS

### Update DNS Records

Go to **Cloudflare Dashboard → DNS → Records**:

| Type | Name | Target | Proxy |
|------|------|--------|-------|
| A | @ | YOUR_PUBLIC_IP | DNS only |
| A | www | YOUR_PUBLIC_IP | DNS only |
| A | ldap | YOUR_PUBLIC_IP | DNS only |
| A | api.ldap | YOUR_PUBLIC_IP | DNS only |

**Get your public IP:**
```bash
curl ifconfig.me
# Output: 203.0.113.42
```

### SSL/TLS Configuration

1. Click **SSL/TLS** → **Overview**
2. Set **Encryption Mode**: **Full** (or Full Strict)
3. Toggle **Always Use HTTPS**: **ON**
4. Set **Minimum TLS Version**: **TLS 1.2**

---

## 🚀 Step 5: Start Services

### Pull Latest Images

```bash
docker pull osixia/openldap:latest
docker pull keycloak/keycloak:latest
docker pull postgres:15
```

### Start Services

```bash
# Start with public docker-compose.yml
docker compose -f docker-compose-public.yml up -d

# Wait for services to initialize
sleep 45

# Check status
docker compose ps
```

### Verify Services Started

```bash
# Check LDAP
ldapwhoami -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123

# Check Keycloak
curl http://localhost:8080/health
```

---

## 🔐 Step 6: Create Firewall Rules (Optional but Recommended)

### Using iptables (Linux)

```bash
# Allow LDAP
sudo iptables -A INPUT -p tcp --dport 389 -j ACCEPT

# Allow LDAPS
sudo iptables -A INPUT -p tcp --dport 636 -j ACCEPT

# Allow HTTP
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT

# Save rules
sudo iptables-save
```

### Using Windows Defender

```powershell
# Check current rules
netsh advfirewall firewall show rule name=LDAP

# Monitor connections
netstat -an | findstr :389
```

---

## 📊 Access Your Services (Public)

### **Web UI (Keycloak)**

```
URL: https://ldap.penux.uk
Admin: https://ldap.penux.uk/admin

Credentials:
  Username: admin
  Password: admin123
```

### **LDAP Server**

```
Server: ldap://ldap.penux.uk:389
        ldaps://ldap.penux.uk:636 (Secure)

Admin DN: cn=admin,dc=penux,dc=uk
Password: admin123
Base DN: dc=penux,dc=uk
```

### **API Endpoint**

```
https://api.ldap.penux.uk/api/health
https://api.ldap.penux.uk/api/users
```

---

## 🧪 Testing Public Access

### Test from External Machine

```bash
# Test DNS resolution
nslookup ldap.penux.uk

# Test LDAP (remote)
ldapwhoami -H ldap://ldap.penux.uk:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123

# Test LDAPS (secure)
ldapwhoami -H ldaps://ldap.penux.uk:636 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123

# Test Keycloak
curl https://ldap.penux.uk/admin

# Test API
curl https://api.ldap.penux.uk/api/health
```

### Test from Browser

```
https://ldap.penux.uk
https://ldap.penux.uk/admin
```

---

## 🔑 User Management

### Create Users in Keycloak

1. Go to `https://ldap.penux.uk/admin`
2. Login: `admin / admin123`
3. Click **Users** → **Add user**
4. Fill in details
5. Set password in **Credentials** tab
6. Users will sync to LDAP

### Test User Login

```bash
# Test user credentials (replace with your user)
ldapwhoami -H ldap://ldap.penux.uk:389 \
  -D "uid=john,ou=people,dc=penux,dc=uk" \
  -w password123
```

---

## 🔐 Security Best Practices

### 1. Change Default Passwords

```bash
# LDAP admin password (in docker-compose.yml)
LDAP_ADMIN_PASSWORD: "your-secure-password"

# Keycloak admin password
KEYCLOAK_ADMIN_PASSWORD: "your-secure-password"

# Restart services after changing
docker compose down
docker compose -f docker-compose-public.yml up -d
```

### 2. Use Strong Passwords

```
Minimum 16 characters
Mix: UPPERCASE, lowercase, numbers, symbols
Example: Tr0pic@lThund3r!P3nux
```

### 3. Enable LDAPS (Secure LDAP)

Already configured in docker-compose-public.yml on port 636

### 4. Create Read-Only LDAP Account

Instead of exposing admin account:

```bash
ldapadd -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 << EOF
dn: cn=ldapapi,ou=applications,dc=penux,dc=uk
objectClass: inetOrgPerson
cn: ldapapi
sn: API Account
uid: ldapapi
userPassword: read-only-password
description: Read-only LDAP API account
EOF
```

### 5. Set Up Rate Limiting

In Cloudflare:
1. Click **Security** → **Rate limiting**
2. Create rule for LDAP endpoints
3. Set threshold: 100 requests per 10 minutes

### 6. Monitor Access Logs

```bash
# View Docker logs
docker compose logs -f keycloak

# Check LDAP access
docker compose logs -f openldap
```

---

## 📊 Complete Access URLs

| Service | Local | Public | Credentials |
|---------|-------|--------|-------------|
| **Keycloak Web** | http://localhost:8080 | https://ldap.penux.uk | admin/admin123 |
| **Keycloak Admin** | http://localhost:8080/admin | https://ldap.penux.uk/admin | admin/admin123 |
| **LDAP Server** | ldap://localhost:389 | ldap://ldap.penux.uk:389 | cn=admin,dc=penux,dc=uk / admin123 |
| **LDAPS Server** | ldaps://localhost:636 | ldaps://ldap.penux.uk:636 | cn=admin,dc=penux,dc=uk / admin123 |
| **API** | http://localhost:3000 | https://api.ldap.penux.uk | Basic Auth |

---

## ✅ Deployment Checklist

- [ ] Get public IP: `curl ifconfig.me`
- [ ] Update docker-compose-public.yml with local IP
- [ ] Configure router port forwarding (80, 389, 636)
- [ ] Configure firewall rules
- [ ] Update Cloudflare DNS records
- [ ] Start services: `docker compose -f docker-compose-public.yml up -d`
- [ ] Wait 45 seconds for initialization
- [ ] Test LDAP locally: `ldapwhoami -H ldap://localhost:389`
- [ ] Test Keycloak: `http://localhost:8080`
- [ ] Verify DNS: `nslookup ldap.penux.uk`
- [ ] Test from external machine (mobile phone, different network)
- [ ] Access https://ldap.penux.uk in browser
- [ ] Login: admin / admin123
- [ ] Change default passwords
- [ ] Create additional users
- [ ] Test user login from external network

---

## 🆘 Troubleshooting

### "Connection refused" from outside

1. Check port forwarding is configured correctly
2. Verify firewall rules are allowing the ports
3. Check router has public IP (not carrier NAT)
4. Test with: `telnet ldap.penux.uk 389`

### "Certificate error" on HTTPS

1. Wait 24 hours for Cloudflare certificate
2. Clear browser cache
3. Try incognito/private mode
4. Check Cloudflare SSL mode is "Full"

### LDAP not responding

1. Check services running: `docker compose ps`
2. Check logs: `docker compose logs -f openldap`
3. Verify firewall isn't blocking 389/636
4. Check router port forwarding

### Keycloak slow or not responding

1. Check PostgreSQL: `docker compose ps keycloak-db`
2. View logs: `docker compose logs -f keycloak`
3. Check disk space: `df -h`
4. Check memory: `docker stats`

---

## 🎯 Next Steps

1. **Update docker-compose.yml** with public ports
2. **Configure router** port forwarding
3. **Configure firewall** rules
4. **Update DNS** to point to your IP
5. **Start services**: `docker compose -f docker-compose-public.yml up -d`
6. **Test access** from external network
7. **Change default passwords**
8. **Create users** in Keycloak
9. **Share access** with team

---

## 📞 Support

**Your LDAP services are now publicly accessible!**

```
Web UI:    https://ldap.penux.uk
LDAP:      ldap://ldap.penux.uk:389
Secure:    ldaps://ldap.penux.uk:636
```

**Credentials:**
```
Admin DN:  cn=admin,dc=penux,dc=uk
Password:  admin123 (change this!)
```

**All systems ready for public deployment!** 🚀
