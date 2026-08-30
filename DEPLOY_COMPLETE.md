# 🚀 Complete LDAP Deployment Guide with DNS

Full step-by-step deployment of OpenLDAP + Keycloak + Cloudflare Tunnel with DNS configuration.

---

## 📋 Prerequisites Check

```bash
# Verify Docker installed
docker --version

# Verify Cloudflare account (should have penux.uk)
# https://dash.cloudflare.com

# Verify you have:
✅ Docker installed
✅ Cloudflare account
✅ penux.uk domain
✅ Admin access to Cloudflare
```

---

## 🎯 Deployment Timeline

```
Phase 1: Start Docker Services (5 minutes)
Phase 2: Configure Cloudflare DNS (5 minutes)
Phase 3: Setup Cloudflare Tunnel (5 minutes)
Phase 4: Verify Everything Works (5 minutes)
────────────────────────────────────────────
Total: 20 minutes to full deployment ✅
```

---

## 🚀 PHASE 1: Start Docker Services

### Step 1.1: Navigate to Project

```bash
cd /path/to/penuX
pwd
# Should show: /home/user/penuX
```

### Step 1.2: Start Services

```bash
# Start all services
docker compose up -d

# Wait for initialization
echo "Waiting 45 seconds for services to start..."
sleep 45

# Check status
docker compose ps

# Expected output:
# ldap-server      running ✓
# keycloak-db      running ✓
# keycloak         running ✓
```

### Step 1.3: Verify Services Are Healthy

```bash
# Test LDAP
ldapwhoami -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123

# Expected: dn:cn=admin,dc=penux,dc=uk

# Test Keycloak
curl http://localhost:8080/health

# Expected: {"status":"UP"}
```

**✅ Phase 1 Complete: Services Running**

---

## 🌐 PHASE 2: Configure Cloudflare DNS

### Step 2.1: Get Your Public IP

```bash
# Get your public IP address
curl ifconfig.me

# Example output: 203.0.113.42
# Save this IP: YOUR_PUBLIC_IP = 203.0.113.42
```

### Step 2.2: Go to Cloudflare Dashboard

```
1. Open: https://dash.cloudflare.com
2. Click on: penux.uk domain
3. Click on: DNS (left sidebar)
4. Click on: Records tab
```

### Step 2.3: Add/Update DNS Records

**Record 1: A record for root domain (@)**
```
Type:   A
Name:   @ (or penux.uk)
IPv4:   203.0.113.42          (YOUR_PUBLIC_IP)
TTL:    Auto
Proxy:  DNS only
Status: Click Add/Update
```

**Record 2: A record for www**
```
Type:   A
Name:   www
IPv4:   203.0.113.42          (YOUR_PUBLIC_IP)
TTL:    Auto
Proxy:  DNS only
Status: Click Add/Update
```

**Record 3: CNAME for ldap**
```
Type:   CNAME
Name:   ldap
Value:  (leave blank for now, update after tunnel setup)
TTL:    Auto
Proxy:  DNS only
Status: Save for now
```

**Record 4: CNAME for api.ldap**
```
Type:   CNAME
Name:   api.ldap
Value:  (leave blank for now)
TTL:    Auto
Proxy:  DNS only
Status: Save for now
```

### Step 2.4: Configure SSL/TLS

**In Cloudflare Dashboard:**

1. Click **SSL/TLS** (left sidebar)
2. Click **Overview**
3. Set **Encryption Mode**: **Full**
4. Wait for HTTPS certificate (auto, ~5 minutes)

**Check Certificate:**
1. Click **Edge Certificates**
2. Should show: **Active Certificate** ✓

**✅ Phase 2 Complete: DNS Configured**

---

## 🔐 PHASE 3: Setup Cloudflare Tunnel

### Step 3.1: Install Cloudflared

#### Windows (PowerShell)
```powershell
# Install via chocolatey
choco install cloudflared

# Or download manually from:
# https://github.com/cloudflare/cloudflared/releases
```

#### macOS
```bash
brew install cloudflared
```

#### Linux
```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
```

**Verify installation:**
```bash
cloudflared --version
```

### Step 3.2: Login to Cloudflare

```bash
# This opens a browser window
cloudflared tunnel login

# Steps:
# 1. Browser opens automatically
# 2. Select: penux.uk domain
# 3. Click: Authorize
# 4. Return to terminal
```

### Step 3.3: Create Tunnel

```bash
# Create tunnel named "ldap-tunnel"
cloudflared tunnel create ldap-tunnel

# Output will show:
# Tunnel ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
# Tunnel credentials saved to: /path/to/credentials.json
```

### Step 3.4: Create Configuration File

**On Windows:** Create `C:\Users\{username}\.cloudflared\config.yml`
**On macOS/Linux:** Create `~/.cloudflared/config.yml`

**Content:**
```yaml
tunnel: ldap-tunnel
credentials-file: /Users/YOUR_USERNAME/.cloudflared/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.json

ingress:
  # Keycloak Web UI
  - hostname: ldap.penux.uk
    service: http://localhost:8080

  # LDAP Server (TCP)
  - hostname: ldap-server.penux.uk
    service: tcp://localhost:389

  # LDAPS Secure (TCP)
  - hostname: ldaps-server.penux.uk
    service: tcp://localhost:636

  # REST API
  - hostname: api.ldap.penux.uk
    service: http://localhost:3000

  # Catch-all
  - service: http_status:404
```

**Note:** Replace `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.json` with your actual credentials filename

### Step 3.5: Start Tunnel

```bash
# In a new terminal/PowerShell window:
cloudflared tunnel run ldap-tunnel

# Output:
# Tunnel running at: https://ldap-tunnel.cfargotunnel.com
# 2024-06-21 12:00:00.000Z    info    Registered tunnel connection
```

**Keep this terminal running!**

### Step 3.6: Update DNS with Tunnel CNAME

**Back in Cloudflare Dashboard:**

1. Go to **DNS → Records**
2. Find or create CNAME record for **ldap**
   ```
   Type:  CNAME
   Name:  ldap
   Value: ldap-tunnel.cfargotunnel.com
   Proxy: DNS only
   ```
3. Find or create CNAME record for **ldaps-server**
   ```
   Type:  CNAME
   Name:  ldaps-server
   Value: ldap-tunnel.cfargotunnel.com
   Proxy: DNS only
   ```
4. Find or create CNAME record for **api.ldap**
   ```
   Type:  CNAME
   Name:  api.ldap
   Value: ldap-tunnel.cfargotunnel.com
   Proxy: DNS only
   ```

**✅ Phase 3 Complete: Tunnel Running**

---

## ✅ PHASE 4: Verify Everything Works

### Step 4.1: Test DNS Resolution

```bash
# Test that DNS is pointing to tunnel
nslookup ldap.penux.uk

# Expected: Points to Cloudflare nameservers

dig ldap.penux.uk CNAME

# Expected: ldap.penux.uk CNAME ldap-tunnel.cfargotunnel.com
```

### Step 4.2: Test Web UI Access

**In Browser:**
```
https://ldap.penux.uk
```

**Expected:**
- ✅ Page loads
- ✅ See Keycloak login screen
- ✅ 🔒 Secure HTTPS connection

**Login:**
```
Username: admin
Password: admin123
```

### Step 4.3: Test Admin Console

**In Browser:**
```
https://ldap.penux.uk/admin
```

**Expected:**
- ✅ Admin console loads
- ✅ Can see users & realms
- ✅ Can manage settings

### Step 4.4: Test LDAP Access

```bash
# Test LDAP via tunnel
ldapwhoami -H ldap://ldap-server.penux.uk:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123

# Expected: dn:cn=admin,dc=penux,dc=uk
```

### Step 4.5: Test LDAPS (Secure)

```bash
# Test LDAPS (with TLS)
ldapwhoami -H ldaps://ldaps-server.penux.uk:636 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123

# Expected: dn:cn=admin,dc=penux,dc=uk
```

### Step 4.6: Test API

```bash
# Test API endpoint
curl https://api.ldap.penux.uk/api/health

# Expected JSON response:
# {"status":"ok","service":"PenuX LDAP API",...}
```

**✅ Phase 4 Complete: Everything Verified**

---

## 📊 Complete DNS Configuration Summary

### Final DNS Records in Cloudflare

| Type | Name | Value | TTL | Proxy | Status |
|------|------|-------|-----|-------|--------|
| A | @ | 203.0.113.42 | Auto | DNS only | ✅ |
| A | www | 203.0.113.42 | Auto | DNS only | ✅ |
| CNAME | ldap | ldap-tunnel.cfargotunnel.com | Auto | DNS only | ✅ |
| CNAME | ldaps-server | ldap-tunnel.cfargotunnel.com | Auto | DNS only | ✅ |
| CNAME | api.ldap | ldap-tunnel.cfargotunnel.com | Auto | DNS only | ✅ |

### SSL/TLS Configuration

| Setting | Value | Status |
|---------|-------|--------|
| Encryption Mode | Full | ✅ |
| Always Use HTTPS | ON | ✅ |
| Minimum TLS | 1.2 | ✅ |
| Certificate | Active | ✅ |

---

## 🌐 Access URLs (After Deployment)

### Web UI
```
https://ldap.penux.uk
https://ldap.penux.uk/admin
```

### LDAP Server
```
ldap://ldap-server.penux.uk:389
ldaps://ldaps-server.penux.uk:636
```

### REST API
```
https://api.ldap.penux.uk/api/health
https://api.ldap.penux.uk/api/users
```

### Credentials
```
Admin DN:  cn=admin,dc=penux,dc=uk
Password:  admin123
Base DN:   dc=penux,dc=uk
```

---

## 🔑 User Management

### Create New User

1. Go to `https://ldap.penux.uk/admin`
2. Login: `admin / admin123`
3. Click **Users**
4. Click **Add user**
5. Fill in details:
   ```
   Username: john
   Email: john@penux.uk
   First Name: John
   Last Name: Doe
   ```
6. Click **Create**
7. Go to **Credentials** tab
8. Set password
9. Toggle **Temporary: OFF**
10. Save

### Test User Login

```bash
# Test newly created user
ldapwhoami -H ldap://ldap-server.penux.uk:389 \
  -D "uid=john,ou=people,dc=penux,dc=uk" \
  -w user-password
```

---

## 🔒 Security Checklist

- [ ] Changed admin password from default (admin123)
- [ ] Created read-only LDAP account for API
- [ ] Enabled HTTPS (done via Cloudflare)
- [ ] Set rate limiting in Cloudflare
- [ ] Enabled WAF rules (optional)
- [ ] Configured firewall rules
- [ ] Tested LDAPS connection
- [ ] Verified certificate is valid
- [ ] Disabled admin account after setup (optional)

---

## 📈 What's Running

### Docker Services
```
✅ OpenLDAP Server (389, 636)
✅ Keycloak Web UI (8080)
✅ PostgreSQL Database (5432)
```

### Cloudflare Tunnel
```
✅ ldap.penux.uk → localhost:8080
✅ ldap-server.penux.uk → localhost:389
✅ ldaps-server.penux.uk → localhost:636
✅ api.ldap.penux.uk → localhost:3000
```

### DNS Configuration
```
✅ A records for main domain
✅ CNAME records pointing to tunnel
✅ SSL/TLS certificates active
✅ HTTPS enforced
```

---

## ✨ Verification Checklist

- [ ] Docker services running: `docker compose ps`
- [ ] Cloudflared tunnel running: `cloudflared tunnel run`
- [ ] LDAP responds: `ldapwhoami -H ldap://localhost`
- [ ] Keycloak web accessible: `http://localhost:8080`
- [ ] DNS resolves: `nslookup ldap.penux.uk`
- [ ] HTTPS works: `curl https://ldap.penux.uk`
- [ ] LDAP tunnel works: `ldapwhoami -H ldap://ldap-server.penux.uk`
- [ ] Admin console loads: `https://ldap.penux.uk/admin`
- [ ] Can login: admin / admin123

---

## 🎯 Testing from Different Locations

### From Same Network
```bash
ldapwhoami -H ldap://ldap-server.penux.uk:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123
```

### From Mobile (Different Network)
```
Open: https://ldap.penux.uk
Expected: Works perfectly ✓
```

### From Another Computer
```bash
# On different machine
ldapsearch -H ldaps://ldaps-server.penux.uk:636 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 \
  -b "dc=penux,dc=uk" \
  "(uid=*)"
```

---

## 🚀 Quick Deployment Command Reference

```bash
# Step 1: Start services
docker compose up -d && sleep 45

# Step 2: Install cloudflared (one time only)
# Windows: choco install cloudflared
# macOS: brew install cloudflared

# Step 3: Login
cloudflared tunnel login

# Step 4: Create tunnel
cloudflared tunnel create ldap-tunnel

# Step 5: Create ~/.cloudflared/config.yml (see above)

# Step 6: Run tunnel (in new terminal)
cloudflared tunnel run ldap-tunnel

# Step 7: Update DNS in Cloudflare (CNAME records)

# Step 8: Test
curl https://ldap.penux.uk
```

---

## 📞 Support & Troubleshooting

### Services Won't Start
```bash
docker compose logs
docker compose ps
```

### Tunnel Won't Connect
```bash
# Verify credentials
cat ~/.cloudflared/xxxxxxx.json

# Check tunnel status
cloudflared tunnel info ldap-tunnel
```

### DNS Not Resolving
```bash
# Wait 15-60 minutes for propagation
nslookup ldap.penux.uk
dig ldap.penux.uk
```

### HTTPS Certificate Error
```bash
# Wait 24 hours for Cloudflare certificate
# Clear browser cache
# Try incognito/private mode
```

---

## 🎉 Success Indicators

✅ `docker compose ps` shows 3 running services  
✅ `cloudflared tunnel run` shows "running at"  
✅ `https://ldap.penux.uk` loads in browser  
✅ Admin login works: admin / admin123  
✅ LDAP command responds successfully  
✅ 🔒 Secure HTTPS connection shown  
✅ Access works from different networks  

---

## 🏆 Deployment Complete!

Your LDAP directory is now:

```
✅ Fully Deployed
✅ Publicly Accessible via HTTPS
✅ Protected by Cloudflare
✅ Accessible from Anywhere
✅ Using Tunneling (No Port Forwarding)
✅ Properly Configured DNS
✅ Active & Ready for Use
```

**Access at:** `https://ldap.penux.uk` 🚀

