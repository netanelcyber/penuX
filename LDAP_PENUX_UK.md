# ldap.penux.uk - Complete Setup & Access Guide

Your **enterprise LDAP directory** hosted at **ldap.penux.uk**

---

## 🎯 What is ldap.penux.uk?

```
ldap.penux.uk
    ↓
GitHub Pages Web Interface (Public)
    ↓
REST API Backend (Vercel)
    ↓
OpenLDAP Server (Your Local Machine)
    ↓
Enterprise Directory Database
    ├── Users (admin, john, jane, ...)
    ├── Groups (admins, users, developers, ...)
    └── Organizational Units
```

---

## 📋 Setup Steps for ldap.penux.uk

### Step 1: Run Automation Script

```bash
python3 automate.py
```

The script will prompt you for:
- GitHub username
- Domain: `penux.uk` (default)
- Local machine IP (from `ipconfig` or `ifconfig`)

### Step 2: GitHub Pages Setup (Automated)

The script creates:
```
Repository: yourusername.github.io
File: index.html (LDAP web interface)
URL: https://yourusername.github.io
```

Then configures custom domain:
```
GitHub Pages Custom Domain: ldap.penux.uk
HTTPS: Automatic
```

### Step 3: Vercel API Deployment (Automated)

The script deploys:
```
API: Node.js/Express
Endpoint: https://your-api.vercel.app
Custom Domain: api.ldap.penux.uk
Environment: LDAP_HOST, credentials configured
```

### Step 4: DNS Configuration (Manual - 5 minutes)

Add these DNS records at your registrar:

**Cloudflare (Recommended - FREE):**
1. Go to https://dash.cloudflare.com
2. Add site: penux.uk
3. Add these DNS records:

```
Type    Name         Value
A       @            your-machine-ip
A       www          your-machine-ip
CNAME   ldap         yourusername.github.io
CNAME   api.ldap     your-api.vercel.app
```

**Or your domain registrar (GoDaddy, Namecheap, etc.):**

Same records as above in their DNS management panel.

### Step 5: GitHub Custom Domain (Automated)

The script configures:
```
Repository: yourusername.github.io
Settings → Pages
Custom domain: ldap.penux.uk
HTTPS: Enforce
```

Wait ~24 hours for HTTPS certificate.

### Step 6: Vercel Custom Domain (Manual - 2 minutes)

1. Go to https://vercel.com/dashboard
2. Select your API project
3. Settings → Domains
4. Add: `api.ldap.penux.uk`
5. Verify DNS records

### Step 7: Connect Web UI to API (Manual - 1 minute)

1. Open: `https://ldap.penux.uk`
2. Click ⚙️ Settings (bottom-right)
3. API Endpoint: `https://api.ldap.penux.uk`
4. Admin DN: `cn=admin,dc=penux,dc=uk`
5. Password: `admin123`
6. Click Save

---

## 🌐 Access ldap.penux.uk

### After Full Setup (32 minutes)

```
https://ldap.penux.uk
```

### What You'll See

**Top Status Bar:**
- ✓ Connected (green dot)
- Users: 3
- Groups: 3
- Server: api.ldap.penux.uk

**Left Panel - Users:**
- admin@penux.uk (Administrator)
- john@penux.uk (Developer)
- jane@penux.uk (DevOps Engineer)

Search users in real-time.

**Right Panel - Groups:**
- penux-admins
- penux-users
- penux-developers

View group members.

**Bottom Panel - Details:**
- Click any user/group to see full details
- View email, title, department, status

---

## 🔐 Login Credentials

### Administrator

```
Admin DN:       cn=admin,dc=penux,dc=uk
Password:       admin123
Base DN:        dc=penux,dc=uk
Domain:         penux.uk
```

### Test Users

| Email | Role | Password |
|-------|------|----------|
| admin@penux.uk | Administrator | admin123 |
| john@penux.uk | Developer | admin123 |
| jane@penux.uk | DevOps Engineer | admin123 |

⚠️ **Change all passwords in production!**

---

## 🔗 All URLs for ldap.penux.uk

### Public URLs (Global Access)

```
Web Directory:     https://ldap.penux.uk
API Backend:       https://api.ldap.penux.uk
Main Domain:       https://penux.uk
```

### Local URLs (Private - Your Machine)

```
Local Web UI:      http://localhost
LDAP Server:       ldap://localhost:389
LDAPS Server:      ldaps://localhost:636
Admin Panel:       http://localhost (phpLDAPadmin)
```

### GitHub Pages

```
Repository:        https://github.com/yourusername/yourusername.github.io
Web UI:            https://yourusername.github.io
```

### Vercel API

```
API Endpoint:      https://your-api.vercel.app
Custom Domain:     https://api.ldap.penux.uk
```

---

## 📊 API Endpoints (Behind ldap.penux.uk)

All endpoints use the web UI, but you can also access directly:

```
GET  /api/health              Health check
GET  /api/users               Get all users
GET  /api/users/:uid          Get specific user
GET  /api/groups              Get all groups
GET  /api/groups/:cn          Get specific group
GET  /api/search?query=X      Search users/groups
GET  /api/stats               Statistics
POST /api/verify              Verify credentials
```

**Example - Get all users:**
```bash
curl -u cn=admin,dc=penux,dc=uk:admin123 \
  https://api.ldap.penux.uk/api/users
```

---

## 🛠️ Management Commands

After deployment, manage from your machine:

### Windows (PowerShell)

```powershell
cd services\openldap

# Service management
.\manage.ps1 status              # Check status
.\manage.ps1 test               # Test LDAP connection
.\manage.ps1 logs               # View logs

# User management
.\manage.ps1 add-user           # Add new user
.\manage.ps1 list-users         # List all users

# Backup
.\backup.ps1 backup             # Create backup
.\backup.ps1 schedule           # Daily backups
```

### Linux/macOS (Bash)

```bash
cd services/openldap

# Service management
./manage.sh status              # Check status
./manage.sh test                # Test LDAP connection
./manage.sh logs                # View logs

# User management
./manage.sh add-user            # Add new user
./manage.sh list-users          # List all users
```

---

## 🧪 Test Your ldap.penux.uk Setup

### 1. Test DNS Resolution

```bash
# Windows
nslookup ldap.penux.uk
nslookup api.ldap.penux.uk

# Linux/macOS
dig ldap.penux.uk
dig api.ldap.penux.uk
```

Expected: Should resolve to GitHub Pages and Vercel IPs

### 2. Test HTTPS Certificate

```bash
# Should show valid certificate
curl -I https://ldap.penux.uk
curl -I https://api.ldap.penux.uk
```

### 3. Test API Health

```bash
curl https://api.ldap.penux.uk/api/health
```

Expected response:
```json
{
  "status": "ok",
  "service": "PenuX LDAP API",
  "timestamp": "2026-06-21T..."
}
```

### 4. Test Web UI

1. Open `https://ldap.penux.uk`
2. Wait for page to load
3. Check "Connected" status
4. See user/group counts

### 5. Test LDAP Connection

```bash
# From your local machine
ldapwhoami -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" -w admin123
```

Expected: `dn:cn=admin,dc=penux,dc=uk`

---

## 🔍 Troubleshooting ldap.penux.uk

### DNS Not Working

**Problem:** `ldap.penux.uk` not resolving

**Solution:**
```bash
# Wait for propagation (24-48 hours)
nslookup ldap.penux.uk

# Or flush DNS cache
# Windows
ipconfig /flushdns

# macOS
sudo dscacheutil -flushcache

# Linux
sudo systemctl restart systemd-resolved
```

### HTTPS Certificate Not Working

**Problem:** Certificate error or "Not Secure"

**Solution:**
1. GitHub Pages generates certificate automatically
2. Wait ~24 hours for certificate issuance
3. Clear browser cache: `Ctrl+Shift+R`
4. Try incognito/private mode

### Web UI Can't Reach API

**Problem:** "Cannot reach API" or CORS error

**Solution:**
1. Verify API URL in settings: `https://api.ldap.penux.uk`
2. Check Vercel environment variables set
3. Verify `CORS_ORIGIN=https://ldap.penux.uk` in Vercel
4. Redeploy Vercel if changed: `vercel --prod`

### API Connection to LDAP Failed

**Problem:** API can't connect to local LDAP

**Solution:**
1. Check `LDAP_HOST` is correct (use IP, not localhost)
2. Verify local LDAP is running: `./manage.sh test`
3. Check firewall allows port 389
4. Verify `LDAP_ADMIN_PASSWORD` matches

### Slow Loading

**Problem:** ldap.penux.uk loads slowly

**Solution:**
1. Check internet connection
2. Check if API is responding: `https://api.ldap.penux.uk/api/health`
3. Check Vercel logs for errors
4. Verify LDAP is healthy locally

---

## 📈 Monitoring ldap.penux.uk

### Monitor API Health

```bash
# Check every minute
watch -n 60 'curl -s https://api.ldap.penux.uk/api/health | jq'
```

### Monitor LDAP Locally

```bash
# View real-time logs
docker logs -f penux-openldap

# Check container status
docker ps | grep penux
```

### Monitor GitHub Pages

1. Go to: `https://github.com/yourusername/yourusername.github.io`
2. Click "Insights" → "Traffic"
3. View page views and referrers

### Monitor Vercel

1. Go to: `https://vercel.com/dashboard`
2. Select your project
3. View "Analytics" → invocations and bandwidth

---

## 🔒 Security for ldap.penux.uk

### 1. Change Default Passwords

```powershell
# Windows
cd services\openldap
.\manage.ps1 set-account-status

# Linux/macOS
cd services/openldap
./manage.sh password-change
```

### 2. Use Strong Passwords

- Minimum 16 characters
- Mix: uppercase, lowercase, numbers, symbols
- Examples: `Tr0pic@lThund3r!P3nux` or `M00nL1ght$Cr0ss`

### 3. Create Read-Only API Account

Instead of using admin account:

```ldif
dn: cn=ldapi,ou=applications,dc=penux,dc=uk
objectClass: inetOrgPerson
cn: ldapi
sn: API Service Account
uid: ldapi
userPassword: secure-password-here
accountStatus: active
description: Read-only API account
```

Then update Vercel with new credentials.

### 4. Enable HTTPS Everywhere

✅ GitHub Pages: Auto HTTPS (enforced)
✅ Vercel: Auto HTTPS (enforced)
✅ LDAP: Use LDAPS (port 636)

### 5. Restrict CORS

In Vercel environment:
```
CORS_ORIGIN=https://ldap.penux.uk
# NOT: * (wildcard)
```

### 6. Regular Backups

```powershell
# Windows
.\backup.ps1 backup -Compress
.\backup.ps1 schedule        # Daily backups at 02:00
```

### 7. Monitor Access Logs

```bash
# Check for failed login attempts
docker logs penux-openldap | grep -i "failed\|error"
```

---

## 📚 Documentation for ldap.penux.uk

| Document | Purpose |
|----------|---------|
| PENUX_UK_SETUP.md | Complete penux.uk domain setup |
| LDAP_PENUX_UK.md | This file - Access & management |
| README.md | LDAP documentation |
| QUICK_REFERENCE.md | Command reference |
| DEPLOY_NOW.md | Deployment guide |
| automate.py | Full automation script |

---

## ✅ Deployment Checklist for ldap.penux.uk

```
Local OpenLDAP:
  □ Docker running
  □ http://localhost accessible
  □ Admin login works
  □ Test users visible

GitHub Pages:
  □ yourusername.github.io repo created
  □ Web interface deployed
  □ GitHub Pages enabled
  □ https://yourusername.github.io working

Vercel API:
  □ API deployed
  □ npm dependencies installed
  □ Environment variables set
  □ https://your-api.vercel.app responding

DNS Configuration:
  □ A records added for penux.uk
  □ CNAME for ldap → GitHub Pages
  □ CNAME for api.ldap → Vercel
  □ DNS propagated (test with nslookup)

GitHub Custom Domain:
  □ Custom domain set to ldap.penux.uk
  □ HTTPS enforced
  □ HTTPS certificate issued (~24 hours)
  □ https://ldap.penux.uk accessible

Vercel Custom Domain:
  □ Custom domain set to api.ldap.penux.uk
  □ HTTPS working
  □ https://api.ldap.penux.uk accessible

Final Testing:
  □ https://ldap.penux.uk loads
  □ Shows "Connected" status
  □ Can see users and groups
  □ API responds to requests
  □ Web UI configured with correct API
```

---

## 🚀 Quick Start for ldap.penux.uk

### In 32 Minutes

```bash
# 1. Run automation
python3 automate.py

# 2. Follow prompts
# - GitHub username
# - Domain: penux.uk
# - Local IP

# 3. Configure DNS (5 minutes)
# Add A and CNAME records at registrar

# 4. Wait for DNS propagation
# Usually 1-24 hours (can be instant)

# 5. Access
https://ldap.penux.uk
```

### Default Login

```
Admin DN: cn=admin,dc=penux,dc=uk
Password: admin123
```

### See Users

```
admin@penux.uk (Administrator)
john@penux.uk (Developer)
jane@penux.uk (DevOps Engineer)
```

---

## 📞 Support

**Having issues with ldap.penux.uk?**

1. Check troubleshooting section above
2. Check Vercel logs: https://vercel.com/dashboard
3. Check GitHub Pages: Settings → Pages
4. Check DNS: `nslookup ldap.penux.uk`
5. Read PENUX_UK_SETUP.md for detailed steps

---

## 🎉 You're Ready!

Your **ldap.penux.uk** enterprise LDAP directory is ready to deploy and access!

```bash
python3 automate.py
```

**Then access at:** `https://ldap.penux.uk` 🚀

---

## 📊 Summary: ldap.penux.uk

| Component | URL | Type | Status |
|-----------|-----|------|--------|
| Web Directory | https://ldap.penux.uk | GitHub Pages | Public |
| REST API | https://api.ldap.penux.uk | Vercel | Public |
| Local UI | http://localhost | Docker | Private |
| LDAP Server | ldap://localhost:389 | Docker | Private |
| Admin | cn=admin,dc=penux,dc=uk | LDAP DN | - |

**All components working together to provide your enterprise LDAP directory!** ✅
