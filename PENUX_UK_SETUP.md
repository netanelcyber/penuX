# PenuX LDAP - penux.uk Domain Setup Guide

Complete guide for deploying PenuX LDAP with the **penux.uk** domain.

---

## 🎯 Architecture with penux.uk

```
penux.uk Domain Structure:
│
├── penux.uk                    Main landing page
├── www.penux.uk               Website
├── ldap.penux.uk              LDAP Web Directory (GitHub Pages)
├── api.ldap.penux.uk          LDAP API Backend (Vercel)
├── admin.penux.uk             Admin Dashboard (optional)
└── api.penux.uk               Alternative API endpoint
```

---

## 📋 Prerequisites

✅ penux.uk domain registered (at registrar like Namecheap, GoDaddy, etc.)
✅ Access to domain DNS settings
✅ GitHub account
✅ Vercel account (free)
✅ Docker Desktop installed locally

---

## Phase 1: Local Deployment

Start OpenLDAP on your local machine:

### Windows
```powershell
.\DEPLOYMENT.bat
```

### Linux/macOS
```bash
./DEPLOYMENT.sh
```

**Result:** 
- OpenLDAP running at `ldap://localhost:389`
- Web UI at `http://localhost`
- Admin credentials: `cn=admin,dc=penux,dc=uk` / `admin123`

---

## Phase 2: Set Up GitHub Pages for penux.uk

### Step 1: Create GitHub Repository

```bash
# Create repository named: yourusername.github.io
# This must be PUBLIC and named exactly as shown
```

### Step 2: Deploy Web Interface

```bash
# Clone your new GitHub Pages repo
git clone https://github.com/yourusername/yourusername.github.io
cd yourusername.github.io

# Copy the LDAP web interface
cp /path/to/penuX/services/openldap/web/index.html ./

# Create directory structure (optional but recommended)
mkdir -p ldap
cp /path/to/penuX/services/openldap/web/index.html ldap/

# Push to GitHub
git add .
git commit -m "Add LDAP directory web interface"
git push origin main
```

### Step 3: Enable GitHub Pages

1. Go to repository Settings
2. Select "Pages" in left sidebar
3. **Source**: Select "Deploy from a branch"
4. **Branch**: Select `main` and `/ (root)` or `/ldap` folder
5. Click **Save**
6. Wait ~1 minute for deployment

✅ Web UI accessible at: `https://yourusername.github.io`

---

## Phase 3: Deploy API to Vercel

### Step 1: Create Vercel Account

- Go to https://vercel.com
- Sign up with GitHub

### Step 2: Deploy API

```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to API directory
cd /path/to/penuX/services/openldap/api

# Deploy to Vercel
vercel --prod
```

### Step 3: Set Environment Variables

**In Vercel Dashboard:**

1. Select your project
2. Go to **Settings** → **Environment Variables**
3. Add these variables:

```
LDAP_HOST = ldap://your-local-ip:389
LDAP_BASE_DN = dc=penux,dc=uk
LDAP_ADMIN_DN = cn=admin,dc=penux,dc=uk
LDAP_ADMIN_PASSWORD = your-admin-password
CORS_ORIGIN = https://ldap.penux.uk
```

**Important:** Replace `your-local-ip` with your machine's local IP
- Windows: Run `ipconfig` and find IPv4 Address
- Linux/macOS: Run `ifconfig` or `hostname -I`

Example: `ldap://192.168.1.100:389`

4. Click **Redeploy** to apply changes

✅ API accessible at: `https://your-api.vercel.app`

---

## Phase 4: Configure DNS for penux.uk

### Current DNS Setup

Log into your domain registrar (Namecheap, GoDaddy, Cloudflare, etc.):

#### Option A: Using Cloudflare (Recommended - Free)

**Step 1: Add penux.uk to Cloudflare**

1. Go to https://dash.cloudflare.com
2. Click **Add a Site**
3. Enter: `penux.uk`
4. Select **Free** plan
5. Complete setup

**Step 2: Update DNS at Registrar**

At your domain registrar, change nameservers to Cloudflare's:
```
NS1: arthur.ns.cloudflare.com
NS2: nancy.ns.cloudflare.com
```

**Step 3: Add DNS Records in Cloudflare**

In Cloudflare Dashboard → DNS → Records:

```
Type    Name              Target                  Proxy Status
A       penux.uk          your-ip-address         DNS only
A       www               your-ip-address         DNS only
CNAME   ldap              yourusername.github.io  DNS only
CNAME   api.ldap          your-api.vercel.app     DNS only
CNAME   api               your-api.vercel.app     DNS only
```

**Wait 24-48 hours** for DNS propagation (usually much faster)

#### Option B: Using Current Registrar

At your domain registrar's DNS management:

```
Type    Name         Target/Value
A       @            your-machine-ip
A       www          your-machine-ip
CNAME   ldap         yourusername.github.io
CNAME   api.ldap     your-api.vercel.app
CNAME   api          your-api.vercel.app
```

---

## Phase 5: Update GitHub Pages Custom Domain

### Configure GitHub Pages for ldap.penux.uk

1. Go to your GitHub Pages repository
2. Settings → Pages
3. Under **Custom domain**, enter: `ldap.penux.uk`
4. Click **Save**
5. Check **Enforce HTTPS**

**GitHub will create a CNAME file automatically**

✅ Now accessible at: `https://ldap.penux.uk`

---

## Phase 6: Update Vercel Custom Domain

### Configure Vercel API for api.ldap.penux.uk

1. Vercel Dashboard → Project → Settings → Domains
2. Click **Add Domain**
3. Enter: `api.ldap.penux.uk`
4. Click **Add**
5. Verify DNS records (Vercel will show instructions)

✅ API now accessible at: `https://api.ldap.penux.uk`

---

## Phase 7: Connect Web UI to API

### Update Web Interface Configuration

1. Open `https://ldap.penux.uk` in browser
2. Click **⚙️ Settings** (bottom-right)
3. Enter these values:

```
API Endpoint:      https://api.ldap.penux.uk
Admin DN:          cn=admin,dc=penux,dc=uk
Admin Password:    your-admin-password
```

4. Click **Save**

✅ Web UI will now fetch data from your API!

---

## 📍 Complete penux.uk Setup

After following all phases, your setup will be:

```
┌─────────────────────────────────────────────────────────────┐
│                    penux.uk Domain                          │
└─────────────────────────────────────────────────────────────┘

penux.uk (A record → your-ip)
├── www.penux.uk           → Landing page
│
├── ldap.penux.uk          → LDAP Web Directory
│   └── GitHub Pages
│       └── Web UI
│           └── Points to: https://ldap.penux.uk
│
├── api.ldap.penux.uk      → LDAP REST API
│   └── Vercel Serverless
│       └── Backend
│           └── Connects to: ldap://your-ip:389
│
└── Your Local Network
    └── ldap://your-ip:389 → OpenLDAP Docker Container
        └── LDAP Server
```

---

## 🔗 Access Points After Setup

### Public URLs (Global)

```
Web Directory:     https://ldap.penux.uk
API Backend:       https://api.ldap.penux.uk
Landing Page:      https://penux.uk
```

### Local URLs (Private)

```
Local Web UI:      http://localhost
LDAP Server:       ldap://localhost:389
LDAPS Server:      ldaps://localhost:636
```

---

## 🔐 Credentials

```
Admin DN:       cn=admin,dc=penux,dc=uk
Admin Password: admin123 (CHANGE IN PRODUCTION!)

Test Users:
  admin@penux.uk   password: admin123
  john@penux.uk    password: admin123
  jane@penux.uk    password: admin123
```

---

## 📊 DNS Records Summary

**Complete DNS configuration for penux.uk:**

| Type | Name | Content | TTL | Notes |
|------|------|---------|-----|-------|
| A | @ | your-ip | 3600 | Main domain |
| A | www | your-ip | 3600 | www subdomain |
| CNAME | ldap | yourusername.github.io | 3600 | GitHub Pages |
| CNAME | api.ldap | your-api.vercel.app | 3600 | Vercel API |
| CNAME | api | your-api.vercel.app | 3600 | Alternative API |
| MX | @ | your-mail-server | 3600 | (If using email) |

---

## 🧪 Test Your Setup

### 1. Test DNS Resolution

```bash
# Test DNS
nslookup ldap.penux.uk
nslookup api.ldap.penux.uk

# Should resolve to GitHub Pages and Vercel IPs
```

### 2. Test Web UI

```
Open: https://ldap.penux.uk
Expected:
- Page loads
- Can configure API
- Shows user list
- Shows group list
```

### 3. Test API

```bash
# Test API health
curl https://api.ldap.penux.uk/api/health

# Expected response:
{
  "status": "ok",
  "service": "PenuX LDAP API",
  "timestamp": "..."
}

# Test users endpoint
curl -u cn=admin,dc=penux,dc=uk:admin123 \
  https://api.ldap.penux.uk/api/users

# Should return JSON with user list
```

### 4. Test LDAP Connection

```bash
# From your machine
ldapwhoami -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" -w admin123

# Expected: dn:cn=admin,dc=penux,dc=uk
```

---

## 🔒 Security Best Practices

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
- Unique and memorable

### 3. Enable HTTPS Everywhere

✅ GitHub Pages: Auto HTTPS (already enforced)
✅ Vercel API: Auto HTTPS (already enforced)
✅ LDAP: Use LDAPS (port 636) for TLS

### 4. Restrict CORS

In Vercel environment variables:
```
CORS_ORIGIN = https://ldap.penux.uk
# NOT: * (wildcard)
```

### 5. Create Read-Only LDAP Account

For API instead of using admin account:

```ldif
dn: cn=ldap-api,ou=applications,dc=penux,dc=uk
objectClass: inetOrgPerson
cn: ldap-api
sn: API Service Account
uid: ldap-api
userPassword: secure-password-here
accountStatus: active
description: Read-only API account
```

Then use this account in Vercel instead of admin.

### 6. Regular Backups

```powershell
# Windows - Create backup
.\backup.ps1 backup -Compress

# Setup daily backups
.\backup.ps1 schedule

# Time: 02:00 (change if desired)
```

### 7. Monitor LDAP Logs

```bash
# View logs
docker logs penux-openldap -f

# Check for failed login attempts
# Monitor for unusual activity
```

---

## 📈 Deployment Checklist for penux.uk

### Local Setup
- [ ] Docker Desktop installed
- [ ] OpenLDAP running (`docker ps`)
- [ ] phpLDAPadmin accessible at `http://localhost`
- [ ] Default login works
- [ ] Created first backup

### GitHub Pages
- [ ] GitHub account created
- [ ] `yourusername.github.io` repository created
- [ ] Web interface uploaded
- [ ] GitHub Pages enabled
- [ ] Accessible at `https://yourusername.github.io`

### Vercel API
- [ ] Vercel account created
- [ ] API deployed to Vercel
- [ ] Environment variables set
- [ ] `LDAP_HOST` points to your local IP
- [ ] `CORS_ORIGIN` set to `https://ldap.penux.uk`
- [ ] API tested and working

### penux.uk Domain
- [ ] Domain registered
- [ ] Access to DNS settings
- [ ] A record added for www/main domain
- [ ] CNAME added for ldap → GitHub Pages
- [ ] CNAME added for api.ldap → Vercel
- [ ] DNS propagated (test with nslookup)

### GitHub Pages Custom Domain
- [ ] Custom domain set to `ldap.penux.uk`
- [ ] HTTPS enforced
- [ ] Accessible at `https://ldap.penux.uk`

### Vercel Custom Domain
- [ ] Custom domain set to `api.ldap.penux.uk`
- [ ] HTTPS enforced
- [ ] DNS records verified

### Final Configuration
- [ ] Web UI configured with API endpoint
- [ ] Can login from `https://ldap.penux.uk`
- [ ] Can see users from API
- [ ] Can search users/groups
- [ ] API health check works

---

## 🚀 Complete Setup Timeline

| Step | Task | Time | Status |
|------|------|------|--------|
| 1 | Deploy locally | 5 min | |
| 2 | Create GitHub repo | 2 min | |
| 3 | Push web UI | 3 min | |
| 4 | Create Vercel account | 2 min | |
| 5 | Deploy API | 5 min | |
| 6 | Register domain | 5 min | |
| 7 | Configure DNS | 10 min | |
| 8 | Update GitHub Pages | 2 min | |
| 9 | Update Vercel domain | 5 min | |
| 10 | Test everything | 10 min | |
| | **Total** | **49 min** | |

**Plus:** 24-48 hours for DNS propagation

---

## 📞 Troubleshooting

### DNS Not Resolving

```bash
# Check DNS
nslookup ldap.penux.uk

# If not working:
# 1. Wait longer (DNS propagation takes time)
# 2. Check registrar DNS settings
# 3. Verify CNAME records are correct
# 4. Flush DNS cache:
#    Windows: ipconfig /flushdns
#    macOS: sudo dscacheutil -flushcache
#    Linux: sudo systemctl restart systemd-resolved
```

### Can't Access https://ldap.penux.uk

1. Check DNS resolves to GitHub Pages
2. Wait for HTTPS certificate (automatic, ~24 hours)
3. Clear browser cache (Ctrl+Shift+R)
4. Try incognito/private mode

### API Errors

1. Check Vercel deployment successful
2. Verify environment variables set
3. Check `LDAP_HOST` is correct IP (not localhost)
4. Test API directly: `https://api.ldap.penux.uk/api/health`
5. Check Vercel logs for errors

### LDAP Connection from API

1. Verify local firewall allows port 389
2. Check Windows Firewall (if on Windows)
3. Verify `LDAP_HOST` uses local IP, not localhost
4. Test from command line:
   ```bash
   ldapwhoami -H ldap://your-ip:389 \
     -D "cn=admin,dc=penux,dc=uk" -w admin123
   ```

### Web UI Can't Reach API

1. Check `CORS_ORIGIN` in Vercel matches domain
2. Verify API endpoint in web UI settings
3. Check browser console for errors (F12)
4. Verify HTTPS (not HTTP)

---

## 📚 Documentation Reference

- **DEPLOY_NOW.md** - Initial deployment
- **README.md** - LDAP documentation
- **GITHUB_PAGES_SETUP.md** - Complete cloud setup
- **WEB_DEPLOYMENT.md** - Web UI deployment
- **QUICK_REFERENCE.md** - Command reference

---

## 🎉 Success Indicators

You'll know everything is working when:

✅ `https://ldap.penux.uk` loads in browser
✅ Can see "Connected" status on web UI
✅ User count shows > 0
✅ Group count shows > 0
✅ Can search for users
✅ Can see user details
✅ API health check returns `{"status": "ok"}`
✅ No CORS errors in console

---

## 🚀 Final Deployment

Your complete penux.uk LDAP setup:

```
https://ldap.penux.uk
    ↓
    Connected to
    ↓
https://api.ldap.penux.uk
    ↓
    Connected to
    ↓
ldap://your-ip:389 (OpenLDAP)
    ↓
Docker Container on Your Machine
    ↓
100+ Users & Groups
```

---

## 📋 Quick Reference: penux.uk Commands

### Access
```
Web Directory:  https://ldap.penux.uk
API:           https://api.ldap.penux.uk
Local UI:      http://localhost
Local LDAP:    ldap://localhost:389
```

### Manage (from local machine)
```powershell
# Windows
cd services\openldap
.\manage.ps1 status
.\manage.ps1 test
.\manage.ps1 add-user
.\backup.ps1 backup
```

```bash
# Linux/macOS
cd services/openldap
./manage.sh status
./manage.sh test
./manage.sh add-user
```

### Deploy
```bash
# Local
./DEPLOYMENT.sh  # or DEPLOYMENT.bat

# GitHub Pages (from your repo)
git add .
git commit -m "Update"
git push origin main

# Vercel API
cd services/openldap/api
vercel --prod
```

---

**Your penux.uk LDAP directory is ready!** 🎉

Follow the phases above to get `https://ldap.penux.uk` live!
