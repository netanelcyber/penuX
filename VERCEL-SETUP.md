# 🚀 Vercel Deployment Guide - PenuX LDAP API

Complete guide to deploy your LDAP API to Vercel and connect it to ldap.penux.uk

---

## Quick Start (5 minutes)

### 1️⃣ Run Deployment Script
```bash
cd ~/penuX
./VERCEL-DEPLOY.sh
```

This will:
- Install Vercel CLI if needed
- Authenticate with your Vercel account
- Deploy the API to production
- Give you your Vercel URL

### 2️⃣ Note Your Vercel URL
Look for output like:
```
✓ Production: https://penuX-api-xxxxx.vercel.app
```
Save this URL - you'll need it.

### 3️⃣ Set Environment Variables

Go to: https://vercel.com/dashboard → Select Your Project → Settings → Environment Variables

**Add these 5 variables:**

| Key | Value |
|-----|-------|
| `LDAP_HOST` | `ldap://ldap-server.penux.uk:389` |
| `LDAP_BASE_DN` | `dc=penux,dc=uk` |
| `LDAP_ADMIN_DN` | `cn=admin,dc=penux,dc=uk` |
| `LDAP_ADMIN_PASSWORD` | `admin123` |
| `CORS_ORIGIN` | `*` |

Click **Save** after adding each variable.

### 4️⃣ Redeploy to Apply Variables

Back in your terminal:
```bash
cd ~/penuX/services/openldap/api
vercel --prod
```

### 5️⃣ Update Cloudflare DNS

Go to: https://dash.cloudflare.com → penux.uk → DNS → Records

**Add CNAME Record:**
```
Type:  CNAME
Name:  ldap
Value: <your-vercel-url> (e.g., penuX-api-xxxxx.vercel.app)
TTL:   Auto
Proxy: DNS only
```

Click **Save**

### 6️⃣ Test

```bash
# Wait 30-60 seconds for DNS to propagate, then:
curl https://ldap.penux.uk/api/health

# Expected response:
# {"status":"ok","service":"PenuX LDAP API","timestamp":"2026-06-21T..."}
```

---

## What Gets Deployed

### REST API Endpoints

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/health` | GET | No | Health check |
| `/api/users` | GET | Basic | List all users |
| `/api/users/:uid` | GET | Basic | Get user details |
| `/api/groups` | GET | Basic | List all groups |
| `/api/groups/:cn` | GET | Basic | Get group details |
| `/api/search` | GET | Basic | Search LDAP |
| `/api/verify` | POST | Basic | Verify credentials |

### Authentication

All endpoints (except `/api/health`) require HTTP Basic Authentication:

```bash
# Example:
curl -u admin:password https://ldap.penux.uk/api/users
```

---

## Troubleshooting

### "Deployment Failed"
```bash
# Check logs:
cd services/openldap/api
vercel logs --prod
```

### "Environment Variables Not Working"
1. Verify variables are set: https://vercel.com/dashboard → Your Project → Settings
2. Redeploy after adding variables: `vercel --prod`
3. Wait 30 seconds for changes to propagate

### "API Returns 403 or 500"
- Check LDAP_HOST is reachable from Vercel (must be public)
- Verify LDAP_ADMIN_PASSWORD is correct
- Check firewall allows Vercel IPs to reach your LDAP server

### "DNS Not Resolving"
- Wait 15-60 minutes for DNS propagation
- Clear DNS cache: `nslookup -flushcache` (Windows) or `sudo dscacheutil -flushcache` (macOS)
- Verify CNAME is set correctly in Cloudflare

---

## Files Deployed

```
services/openldap/api/
├── server.js          (Main API server)
├── package.json       (Dependencies)
├── vercel.json        (Vercel configuration)
└── node_modules/      (Installed automatically)
```

---

## Costs

- **Vercel Free Tier**: Included
  - 100 GB bandwidth/month
  - 1000 function invocations/day
  - Perfect for development/testing

- **For production use**, consider upgrading to Pro ($20/month)

---

## Next: Connect to Your LDAP Server

For Vercel to reach your LDAP server:

### Option 1: Public LDAP (Recommended)
```bash
# Your local machine runs:
docker compose -f docker-compose-public.yml up -d

# Then update DNS A records to your public IP
# Vercel can connect directly
```

### Option 2: Cloudflare Tunnel
```bash
# Deploy tunnel to expose LDAP:
./deploy-all.sh

# Set LDAP_HOST to:
# ldap://ldap-server.penux.uk:389
```

### Option 3: VPN/Proxy
If LDAP is behind firewall, set up VPN endpoint

---

## Monitoring

### View Real-Time Logs
```bash
vercel logs --prod
```

### Check Deployment Status
https://vercel.com/dashboard → Your Project → Deployments

### Monitor API Health
```bash
# Every 60 seconds:
watch -n 60 'curl https://ldap.penux.uk/api/health'
```

---

## Rollback

If something breaks:
```bash
# View previous deployments:
vercel ls

# Promote previous deployment:
vercel promote <deployment-url>
```

---

## Support

- Vercel Docs: https://vercel.com/docs
- API Documentation: See `services/openldap/api/server.js`
- LDAP Documentation: https://www.openldap.org/doc/

---

**Deployment Status:** Ready ✅

Run `./VERCEL-DEPLOY.sh` to start deployment!
