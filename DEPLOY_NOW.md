# 🚀 PenuX LDAP - Deploy Now!

Complete deployment instructions for the entire OpenLDAP solution.

---

## ⚡ Quick Deploy (Choose Your Platform)

### Windows (Docker Desktop)

```powershell
# 1. Run deployment script
.\DEPLOYMENT.bat

# 2. Wait 30-40 seconds
# 3. Open http://localhost
# 4. Login: cn=admin,dc=penux,dc=uk / admin123
```

### Linux/macOS

```bash
# 1. Run deployment script
chmod +x DEPLOYMENT.sh
./DEPLOYMENT.sh

# 2. Wait 30-40 seconds
# 3. Open http://localhost
# 4. Login: cn=admin,dc=penux,dc=uk / admin123
```

---

## 📦 What Gets Deployed

### Local Components (Runs on Your Machine)

```
Your Machine
├── OpenLDAP Server (Docker)
│   ├── LDAP protocol: ldap://localhost:389
│   ├── LDAPS protocol: ldaps://localhost:636
│   ├── Database: /var/lib/ldap (persistent)
│   └── Config: /etc/ldap/slapd.d (persistent)
│
├── phpLDAPadmin Web UI (Docker)
│   ├── HTTP: http://localhost
│   └── HTTPS: https://localhost:6443
│
└── Management Tools (PowerShell/Bash)
    ├── manage.sh/ps1 - Service management
    ├── backup.sh/ps1 - Backup & restore
    └── setup.sh/ps1 - Initial setup
```

### Cloud Components (Optional - Deploy Later)

```
GitHub Pages
└── https://yourusername.github.io (Web UI)

Vercel/Netlify
└── https://your-api.vercel.app (API Backend)

penux.uk Domain
├── ldap.penux.uk → GitHub Pages
└── api.ldap.penux.uk → Vercel
```

---

## 📋 Step-by-Step Deployment

### Phase 1: Start Local Services (5 minutes)

#### Windows

1. **Open PowerShell as Administrator**
   ```powershell
   # Navigate to penuX directory
   cd C:\path\to\penuX
   
   # Run deployment
   .\DEPLOYMENT.bat
   ```

2. **Wait for initialization**
   - Watch Docker Desktop
   - Wait 30-40 seconds
   - See "OpenLDAP is now running!"

3. **Verify services**
   ```powershell
   cd services\openldap
   .\manage.ps1 status
   .\manage.ps1 test
   ```

#### Linux/macOS

1. **Open terminal**
   ```bash
   cd /path/to/penuX
   chmod +x DEPLOYMENT.sh
   ./DEPLOYMENT.sh
   ```

2. **Wait for initialization**
   ```bash
   cd services/openldap
   ./manage.sh status
   ./manage.sh test
   ```

### Phase 2: Access Web UI (1 minute)

1. **Open browser**
   - Go to: `http://localhost`

2. **Login**
   - Admin DN: `cn=admin,dc=penux,dc=uk`
   - Password: `admin123`

3. **Explore**
   - Browse users
   - Browse groups
   - Search functionality
   - View details

### Phase 3: Configure & Backup (Optional - 10 minutes)

#### Change Admin Password

```powershell
# Windows (PowerShell)
cd services\openldap
.\manage.ps1 set-account-status

# Linux/macOS
cd services/openldap
./manage.sh password-change
```

#### Create First Backup

```powershell
# Windows
.\backup.ps1 backup -Compress

# Linux/macOS
./backup.sh
```

#### Set Up Daily Backups

```powershell
# Windows - Schedule automatic backups
.\backup.ps1 schedule
# Choose time (default 02:00)

# Linux/macOS - Cron job
crontab -e
# Add: 0 2 * * * /path/to/services/openldap/backup.sh
```

### Phase 4: Deploy to GitHub Pages (Optional - 15 minutes)

#### Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: **yourusername.github.io** (must be exact!)
3. Make it Public
4. Click "Create repository"

#### Deploy Web UI

```bash
# Clone your new repository
git clone https://github.com/yourusername/yourusername.github.io
cd yourusername.github.io

# Copy web interface
cp /path/to/penuX/services/openldap/web/index.html ./

# Push to GitHub
git add index.html
git commit -m "Add LDAP directory web interface"
git push origin main

# Enable GitHub Pages
# Go to Settings → Pages
# Source: main / (root)
# Wait ~1 minute
# Access at: https://yourusername.github.io
```

✅ Your LDAP directory is now public!

#### Deploy API Backend (Vercel)

1. **Create Vercel Account**
   - Go to https://vercel.com
   - Sign up with GitHub

2. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

3. **Deploy API**
   ```bash
   cd /path/to/penuX/services/openldap/api
   vercel --prod
   ```

4. **Set Environment Variables**
   - Vercel Dashboard → Project Settings → Environment Variables
   - Add:
     ```
     LDAP_HOST=ldap://your-ip:389
     LDAP_BASE_DN=dc=penux,dc=uk
     LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk
     LDAP_ADMIN_PASSWORD=your-password
     CORS_ORIGIN=https://yourusername.github.io
     ```
   - Redeploy

5. **Update Web UI**
   - Open `https://yourusername.github.io`
   - Click ⚙️ (Settings)
   - API Endpoint: `https://your-api.vercel.app`
   - Admin DN: `cn=admin,dc=penux,dc=uk`
   - Password: Your LDAP password
   - Save

✅ Global LDAP directory is now live!

### Phase 5: Custom Domain (Optional - 10 minutes)

#### Add DNS Records

In your DNS provider (Cloudflare, GoDaddy, etc.):

```
Type    Name         Value
CNAME   ldap         yourusername.github.io
CNAME   api.ldap     your-api.vercel.app
```

#### Configure GitHub Pages

1. Go to yourusername.github.io repo
2. Settings → Pages
3. Custom domain: `ldap.penux.uk`
4. Check "Enforce HTTPS"

#### Configure Vercel

1. Vercel Dashboard → Project → Settings → Domains
2. Add domain: `api.ldap.penux.uk`

✅ Accessible at: `https://ldap.penux.uk`

---

## ✅ Deployment Checklist

### Phase 1: Local Services
- [ ] Docker Desktop running
- [ ] Run DEPLOYMENT script
- [ ] Wait 30-40 seconds
- [ ] Services showing healthy
- [ ] Web UI accessible at `http://localhost`
- [ ] Can login with admin credentials

### Phase 2: Management
- [ ] Changed default password
- [ ] Created backup
- [ ] Set up daily backups (optional)
- [ ] Tested LDAP connection
- [ ] Added test users (optional)

### Phase 3: GitHub Pages (Optional)
- [ ] Created `yourusername.github.io` repo
- [ ] Copied web interface
- [ ] Enabled GitHub Pages
- [ ] Web UI accessible at `https://yourusername.github.io`

### Phase 4: API Backend (Optional)
- [ ] Deployed to Vercel/Netlify
- [ ] Set environment variables
- [ ] Redeploy after env changes
- [ ] Web UI configured with API endpoint
- [ ] Can see users from web interface

### Phase 5: Custom Domain (Optional)
- [ ] Added DNS CNAME records
- [ ] Configured GitHub Pages custom domain
- [ ] Configured Vercel custom domain
- [ ] HTTPS enforced
- [ ] Accessible at `ldap.penux.uk`

---

## 🔧 Management Commands

### Windows (PowerShell)

```powershell
cd services\openldap

# Service Management
.\manage.ps1 start              # Start services
.\manage.ps1 stop               # Stop services
.\manage.ps1 status             # Show status
.\manage.ps1 logs               # View logs
.\manage.ps1 test               # Test LDAP

# User Management
.\manage.ps1 add-user           # Add new user
.\manage.ps1 list-users         # List all users
.\manage.ps1 set-account-status # Change password

# Backup
.\backup.ps1 backup             # Create backup
.\backup.ps1 backup -Compress   # Backup with compression
.\backup.ps1 restore backup.ldif # Restore backup
.\backup.ps1 schedule           # Set up daily backups
```

### Linux/macOS (Bash)

```bash
cd services/openldap

# Service Management
./manage.sh start              # Start services
./manage.sh stop               # Stop services
./manage.sh status             # Show status
./manage.sh logs               # View logs
./manage.sh test               # Test LDAP

# User Management
./manage.sh add-user           # Add new user
./manage.sh list-users         # List all users

# Backup
./backup.sh                    # Create backup
```

---

## 🔐 Accessing Your LDAP

### Local (On Your Machine)

```
Web UI:        http://localhost
LDAP:          ldap://localhost:389
LDAPS:         ldaps://localhost:636
Admin DN:      cn=admin,dc=penux,dc=uk
```

### Remote (After GitHub Pages Deployment)

```
Web UI:        https://yourusername.github.io
API:           https://your-api.vercel.app
API (custom):  https://api.ldap.penux.uk
Admin DN:      cn=admin,dc=penux,dc=uk
```

### Default Test Users

```
User: admin@penux.uk        Password: admin123
User: john@penux.uk         Password: admin123
User: jane@penux.uk         Password: admin123
```

⚠️ **Change all passwords in production!**

---

## 📊 Deployment Summary

| Component | Local | Cloud | Status |
|-----------|-------|-------|--------|
| **OpenLDAP Server** | ✅ Docker | - | Ready |
| **phpLDAPadmin** | ✅ Docker | - | Ready |
| **Web UI** | ✅ Optional | ✅ GitHub Pages | Ready |
| **API Backend** | - | ✅ Vercel | Ready |
| **Management** | ✅ PS/Bash | - | Ready |
| **Backups** | ✅ PS/Bash | - | Ready |
| **Documentation** | ✅ Guides | - | 15,000+ words |

---

## 📚 Documentation

After deployment, refer to these guides:

- **README.md** - Complete LDAP setup & usage
- **WINDOWS_SETUP.md** - Windows-specific setup
- **POWERSHELL_GUIDE.md** - PowerShell scripts guide
- **WEB_DEPLOYMENT.md** - GitHub Pages deployment
- **GITHUB_PAGES_SETUP.md** - Complete cloud setup
- **QUICK_REFERENCE.md** - Quick command reference

---

## 🆘 Troubleshooting

### Services Won't Start

```bash
# Check Docker
docker --version
docker ps

# Check logs
docker logs penux-openldap
docker logs penux-phpldapadmin

# Restart
docker-compose down -v
docker-compose up -d
```

### Can't Access Web UI

1. Check `http://localhost` is accessible
2. Wait 40 seconds for initialization
3. Check Docker: `docker ps`
4. Check logs: `docker logs penux-phpldapadmin`

### LDAP Connection Errors

```powershell
# Test connection
.\manage.ps1 test

# Check LDAP container
docker logs penux-openldap
```

### GitHub Pages Issues

- Clear browser cache: Ctrl+Shift+R
- Wait for DNS propagation (up to 24 hours)
- Check GitHub Pages in repo Settings

### API Errors

- Verify environment variables set
- Check Vercel logs in dashboard
- Test API directly: `https://api.vercel.app/api/health`

---

## 🎯 Next Steps

1. **Run DEPLOYMENT script** (Your platform)
2. **Access http://localhost**
3. **Verify admin login**
4. **Create backup** (optional)
5. **Deploy to GitHub Pages** (optional)
6. **Deploy API** (optional)
7. **Set custom domain** (optional)
8. **Share your LDAP directory!**

---

## 🎉 Success!

You now have:

✅ **Enterprise LDAP Directory** - Running locally
✅ **Web Interface** - For browsing users/groups
✅ **REST API** - For integrations
✅ **Global Access** - Via GitHub Pages (optional)
✅ **Complete Documentation** - 15,000+ words
✅ **Management Tools** - PowerShell & Bash scripts
✅ **Backup/Restore** - Automated or manual
✅ **Zero Cost** - All free services

---

## 📞 Support

- **Local Issues**: Check container logs
  ```bash
  docker logs penux-openldap
  docker logs penux-phpldapadmin
  ```

- **GitHub Pages**: See WEB_DEPLOYMENT.md

- **API Issues**: See GITHUB_PAGES_SETUP.md

- **General Help**: See README.md or QUICK_REFERENCE.md

---

## 🚀 Deploy Now!

**Windows:**
```powershell
.\DEPLOYMENT.bat
```

**Linux/macOS:**
```bash
./DEPLOYMENT.sh
```

**Your LDAP directory will be live in 5 minutes!** 🎉
