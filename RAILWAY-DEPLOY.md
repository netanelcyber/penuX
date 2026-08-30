# 🚂 Deploy PenuX LDAP to Railway

Complete guide to deploy PenuX LDAP directory on Railway in minutes.

---

## ⚡ Quick Deploy (5 minutes)

### Option 1: Using Deploy Script (Recommended)

```bash
# 1. Make script executable
chmod +x railway-deploy.sh

# 2. Run deployment
./railway-deploy.sh

# 3. Follow prompts
# Enter passwords when asked
# Script handles everything else
```

### Option 2: Manual CLI Setup

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Set variables
railway variables set LDAP_ADMIN_PASSWORD="YourPassword"
railway variables set KEYCLOAK_ADMIN_PASSWORD="YourPassword"

# 5. Deploy
railway up

# 6. Check status
railway status
```

### Option 3: Web Dashboard (No CLI)

1. Go to https://railway.app/dashboard
2. Click **"Create New Project"**
3. Select **"Deploy from GitHub"**
4. Select **penuX** repository
5. Railway auto-detects services
6. Set environment variables
7. Click **"Deploy"**

---

## 📊 What Gets Deployed

| Service | Status | Access |
|---------|--------|--------|
| **OpenLDAP** | Auto-deployed | ldap/ldaps |
| **PostgreSQL** | Auto-created | Internal |
| **REST API** | Auto-deployed | https://api-xxxxx.railway.app |
| **Web UI** | Auto-deployed | https://web-xxxxx.railway.app |
| **Keycloak** | Auto-deployed | https://keycloak-xxxxx.railway.app |

---

## 🔧 Configuration

### Required Environment Variables

Set these in Railway dashboard or via CLI:

```bash
# LDAP
LDAP_ORGANISATION=PenuX
LDAP_DOMAIN=penux.uk
LDAP_BASE_DN=dc=penux,dc=uk
LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk
LDAP_ADMIN_PASSWORD=YourSecurePassword123!

# Keycloak
KEYCLOAK_ADMIN=admin
KEYCLOAK_ADMIN_PASSWORD=YourSecurePassword123!

# API
CORS_ORIGIN=*
NODE_ENV=production

# Database (auto-created by Railway PostgreSQL)
DATABASE_URL=(auto-generated)
```

### Set Variables via CLI

```bash
railway variables set LDAP_ADMIN_PASSWORD="YourPassword"
railway variables set KEYCLOAK_ADMIN_PASSWORD="YourPassword"
```

### Set Variables via Dashboard

1. Go to https://railway.app/dashboard
2. Click your project
3. Click service
4. Go to **"Variables"** tab
5. Add key-value pairs
6. Save

---

## ✅ After Deployment

### 1. Get Your URLs

Railway generates URLs automatically:

```bash
# Via CLI
railway status

# Or in dashboard:
# Project → Services → Click each service
```

You'll see:
```
Web:      https://web-xxxxx.railway.app
API:      https://api-xxxxx.railway.app
Keycloak: https://keycloak-xxxxx.railway.app
```

### 2. Test Services

```bash
# API Health Check
curl https://api-xxxxx.railway.app/api/health

# List users
curl -u "cn=admin,dc=penux,dc=uk:YourPassword" \
  https://api-xxxxx.railway.app/api/users

# View logs
railway logs -f
```

### 3. Add Custom Domains (Optional)

To use your own domain (ldap.penux.uk):

1. **In Railway dashboard:**
   - Click service
   - Go to **"Domains"**
   - Click **"Add Domain"**
   - Enter: `ldap.penux.uk`
   - Copy the **CNAME value**

2. **At your registrar (Namecheap, GoDaddy, etc.):**
   - Go to DNS settings
   - Add CNAME record:
     ```
     Type: CNAME
     Name: ldap
     Value: (Railway CNAME)
     TTL: 3600
     ```

3. **Wait 24-48 hours** for DNS propagation

4. **Access via custom domain:**
   ```
   https://ldap.penux.uk
   https://api.ldap.penux.uk
   https://keycloak.penux.uk
   ```

---

## 📈 Scaling & Monitoring

### View Metrics

```bash
# Via CLI
railway status

# Via dashboard:
# Service → Metrics tab
# Shows: CPU, Memory, Network
```

### Increase Resources

1. Go to service in dashboard
2. Click **"Settings"**
3. Change plan:
   - **Hobby** (free, limited)
   - **Pro** (pay as you go)
   - **Business** (enterprise)

4. Adjust CPU/Memory as needed

### Auto-Scale

```bash
# Scale API to 2 instances
railway scale api=2

# View scaled status
railway status
```

---

## 🐛 Troubleshooting

### Services Won't Start

**Check logs:**
```bash
railway logs -f
```

**Look for:**
- Missing environment variables
- Database connection errors
- Port conflicts

**Fix:**
```bash
# Set missing variables
railway variables set VAR_NAME="value"

# Redeploy
railway up
```

### LDAP Connection Failed

**Verify variable is set:**
```bash
railway variables | grep LDAP

# Should show:
# LDAP_ADMIN_PASSWORD=...
# LDAP_BASE_DN=...
# LDAP_ADMIN_DN=...
```

**Test connection via API:**
```bash
curl -u "cn=admin,dc=penux,dc=uk:YourPassword" \
  https://api-xxxxx.railway.app/api/users
```

### High Resource Usage

**Check metrics:**
```bash
railway status
# Look for high CPU/Memory
```

**Reduce load:**
- Scale down instances: `railway scale api=1`
- Upgrade plan for more resources
- Check API logs for inefficient queries

### DNS Not Resolving

**Verify CNAME:**
```bash
nslookup ldap.penux.uk
# Should resolve to Railway endpoint
```

**Fix:**
1. Check CNAME record is correct
2. Wait for DNS TTL (3600 seconds = 1 hour)
3. Clear browser cache
4. Try different DNS server: `nslookup -server 8.8.8.8 ldap.penux.uk`

---

## 📚 Useful Commands

```bash
# View status
railway status

# View logs (live)
railway logs -f

# View specific service logs
railway logs -f --service api

# View variables
railway variables

# Set variable
railway variables set VAR_NAME="value"

# Remove variable
railway variables delete VAR_NAME

# Scale service
railway scale api=2

# Open dashboard
railway open

# Link existing project
railway link <project-id>

# Disconnect project
railway unlink

# View documentation
railway help
```

---

## 💰 Costs

**Typical monthly costs on Railway:**

| Service | Size | Cost |
|---------|------|------|
| OpenLDAP | 2GB RAM | $20 |
| PostgreSQL | 5GB storage | $15 |
| REST API | 2GB RAM | $20 |
| Web UI | 1GB RAM | $10 |
| Keycloak | 2GB RAM | $20 |
| **Total** | | **~$85** |

- Free tier available for testing
- Pay-as-you-go for production
- No upfront costs

---

## 🔐 Security Tips

After deployment:

1. **Change default passwords:**
   ```bash
   # Change LDAP password
   railway variables set LDAP_ADMIN_PASSWORD="NewPassword123!"
   railway up
   ```

2. **Enable HTTPS only:**
   - Railway automatically provides HTTPS
   - Redirect HTTP to HTTPS in settings

3. **Set up rate limiting:**
   - API has built-in rate limiting (100 req/15 min)
   - Monitor in dashboard

4. **Enable backups:**
   - PostgreSQL: Automatic daily backups
   - Configure retention in Railway settings

5. **Monitor logs:**
   ```bash
   # Check for errors
   railway logs -f | grep -i error
   ```

---

## 📞 Support

- **Railway Docs:** https://docs.railway.app
- **Railway Status:** https://status.railway.app
- **penuX GitHub:** https://github.com/netanelcyber/penuX
- **API Docs:** See `/API-USAGE-GUIDE.md`
- **Security:** See `/SECURITY-GUIDE.md`

---

## ✨ What's Next?

After deployment:

1. ✅ Test all endpoints (curl/Postman)
2. ✅ Create LDAP users and groups
3. ✅ Configure Keycloak SSO
4. ✅ Add custom domains
5. ✅ Set up monitoring alerts
6. ✅ Configure backups
7. ✅ Document configuration
8. ✅ Train team on usage

---

## 🎯 Quick Checklist

- [ ] Railway CLI installed (`npm install -g @railway/cli`)
- [ ] Logged into Railway (`railway login`)
- [ ] Project created (`railway init`)
- [ ] Environment variables set
- [ ] Deployment started (`railway up`)
- [ ] All services running (green status)
- [ ] API responding to requests
- [ ] LDAP connection verified
- [ ] Custom domains added (optional)
- [ ] DNS records updated (optional)
- [ ] Default passwords changed
- [ ] Monitoring configured
- [ ] Backups enabled

---

**You're ready to deploy! 🚀**

Run: `./railway-deploy.sh` or follow manual steps above.

Check `/DEPLOYMENT-ALTERNATIVES.md` for other platforms.
