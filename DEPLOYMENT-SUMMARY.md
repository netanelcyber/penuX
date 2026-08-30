# 📋 PenuX LDAP Deployment — Complete Summary

Your LDAP system is **fully automated**. This is the complete reference.

---

## 🚀 Current Status

✅ **LIVE NOW**
- Deployment running on GitHub Actions
- URL: https://github.com/netanelcyber/penuX/actions/workflows/auto-deploy-tunnel.yml
- Auto-relaunches every 6 hours
- Self-heals on failure

---

## 📍 Two Deployment Modes

### Mode 1: Zero-Signup (Default)
**Currently Active**

```
Uses: localhost.run SSH tunnel
URL: https://<random>.lhr.life (changes each run)
Setup: None required
Cost: Free
Perfect for: Testing, development
```

### Mode 2: Permanent URL (Optional)
**Upgrade anytime**

```
Uses: ngrok free tier
URL: https://penux-ldap.ngrok-free.app (permanent, you choose)
Setup: 5 minutes (email signup + 2 secrets)
Cost: Free (1 static domain included)
Perfect for: Production, sharing with others
```

---

## ⚡ Quick Start (Already Done)

Your repo has:

```
.github/workflows/auto-deploy-tunnel.yml     ✅ Main automation
.devcontainer/                                ✅ Codespaces config
docker-compose-hub.yml                        ✅ Full stack definition
free-tunnel-setup.sh                          ✅ Manual tunnel script
deployment-status.sh                          ✅ Status dashboard
DEPLOYMENT-AUTO.md                            ✅ Automation guide
NGROK-AUTO-SETUP.md                           ✅ Permanent URL setup
```

---

## 🎯 Your Next Steps

### Option A: Use as-is (Temporary URL)
- Deployment is running now
- URL changes every 6 hours
- Perfect for testing/demos

**Find the current live URL:**
```bash
# Quick check — the URL is always here:
cat LIVE_URL.txt

# Or check status:
./deployment-status.sh
```

### Option B: Add permanent URL (5 min setup)

**Follow:** `NGROK-AUTO-SETUP.md`

1. Sign up at ngrok.com (email only)
2. Get token + claim free domain
3. Add 2 secrets to GitHub
4. Trigger workflow

That's it. Your LDAP has a permanent URL forever.

---

## 🔄 How It Works

```
Every 6 hours (or on manual trigger):
    ↓
GitHub Actions ubuntu runner starts
    ↓
Pull Docker images from Hub
    ↓
Boot 4 containers: OpenLDAP, PostgreSQL, API, Web UI
    ↓
Open tunnel (localhost.run or ngrok)
    ↓
Keep running for ~350 minutes
    ↓
Every 60s: health-check API, auto-restart if needed
    ↓
Job expires at 6-hour mark
    ↓
Cron relaunches → repeat
```

---

## 📊 Architecture

```
┌─ GitHub Actions Runner (ubuntu-latest)
│
├─ Docker Compose Stack
│  ├─ osixia/openldap:1.5.0       (LDAP protocol, port 389)
│  ├─ postgres:15-alpine          (Database)
│  ├─ netanelcyber/penux-ldap-api:latest  (Node.js REST API, port 3000)
│  └─ netanelcyber/penux-ldap-web:latest  (Node.js Web UI, port 3001)
│
├─ Tunnel (localhost.run or ngrok)
│  └─ Exposes port 3001 → public HTTPS
│
└─ Health Loop
   └─ Restarts stack if API becomes unhealthy
```

---

## 🛠️ Commands Reference

### Status
```bash
./deployment-status.sh
```

### Manual Trigger
```bash
# Option 1: GitHub Actions UI
# https://github.com/netanelcyber/penuX/actions → Auto-Deploy LDAP via Free Tunnel → Run workflow

# Option 2: GitHub CLI
gh workflow run auto-deploy-tunnel.yml -r claude/laughing-cori-jlr7od
```

### Local Testing (no GitHub Actions)
```bash
./free-tunnel-setup.sh netanelcyber
```

### View Logs
```bash
# Latest run
open https://github.com/netanelcyber/penuX/actions/workflows/auto-deploy-tunnel.yml

# Or via CLI
gh run list --workflow=auto-deploy-tunnel.yml --limit=1
```

---

## 🌐 Access Your LDAP

### Get the Live URL
```bash
# The live URL is always here (auto-updated each deployment):
cat LIVE_URL.txt
```

### Web UI
```
https://<tunnel-url>/
```

### REST API
```bash
# Read current URL
URL=$(cat LIVE_URL.txt)

# Health check
curl $URL/api/health

# List users (with auth)
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  $URL/api/users

# Search LDAP
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  $URL/api/search?filter=*
```

### LDAP Protocol
- Port: 389 (not exposed via tunnel)
- DN: `cn=admin,dc=penux,dc=uk`
- Password: `admin123` (default, stored in `.env`)
- For LDAP client access, use the container IP or a separate tunnel

---

## 🔐 Security

### What's Protected
- ✅ LDAP credentials stored in GitHub Secrets
- ✅ Tunnel uses HTTPS only
- ✅ No open ports on your machine
- ✅ ngrok/localhost.run handle DDoS protection

### What's NOT Protected (by design)
- ⚠️ Default LDAP password (`admin123`)
- ⚠️ Demo database (no persistence after restart)

**For production:**
1. Change `LDAP_ADMIN_PASSWORD` secret
2. Use persistent database volume
3. Add authentication to API endpoints

---

## 📈 Costs

**$0/month** — Completely Free

- ✅ GitHub Actions: Free tier (included)
- ✅ ngrok: Free tier (1 static domain)
- ✅ localhost.run: Free (no signup)
- ✅ Docker Hub pulls: Free (public images)

---

## 🛑 Stopping the Deployment

### Pause (Temporary)
1. Go to **Settings** → **Actions** → Disable the workflow
2. Existing run finishes in ~6 hours

### Resume
1. Re-enable the workflow
2. Trigger manually or wait for next cron

### Delete (Permanent)
```bash
git rm .github/workflows/auto-deploy-tunnel.yml
git commit -m "Remove auto-deployment"
git push
```

---

## 🐛 Troubleshooting

### No public URL in the summary
- Check the workflow logs
- Look for "Start localhost.run tunnel" step
- If it failed, next run (6h or manual trigger) should work

### API not healthy
- Check docker-compose logs in the Actions run
- Verify environment variables in `.env`
- Check `Dockerfile.api` and services are starting

### Want to change LDAP password
```bash
# Update the secret:
# Settings → Secrets and variables → Actions → LDAP_ADMIN_PASSWORD

# Redeploy:
# Actions → Auto-Deploy LDAP via Free Tunnel → Run workflow
```

### Deployment taking too long
- First run pulls all images (1-2 min)
- Subsequent runs reuse cached images
- Health checks add ~2 min to startup

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `.github/workflows/auto-deploy-tunnel.yml` | Main GitHub Actions workflow |
| `.devcontainer/devcontainer.json` | Codespaces configuration |
| `.devcontainer/autostart.sh` | Auto-boot in Codespaces |
| `docker-compose-hub.yml` | Full stack definition (4 services) |
| `free-tunnel-setup.sh` | Manual deployment script |
| `ngrok-auto-setup.sh` | Auto-configure ngrok secrets |
| `deployment-status.sh` | CLI status dashboard |
| `DEPLOYMENT-AUTO.md` | Detailed automation guide |
| `NGROK-AUTO-SETUP.md` | Permanent URL setup guide |
| `DEPLOYMENT-SUMMARY.md` | This file |

---

## ✨ Key Features

✅ **Zero Secrets Required** — Works immediately  
✅ **Auto-Relaunching** — Every 6 hours via cron  
✅ **Self-Healing** — Restarts stack on failure  
✅ **Free** — No credit card, no costs  
✅ **Codespaces Support** — Auto-boots when you open in GitHub Codespaces  
✅ **Permanent URL** — Optional ngrok upgrade (email signup only)  
✅ **Auto-Published URL** — Live tunnel URL committed to `LIVE_URL.txt` each run  
✅ **GitHub Actions Only** — Leverages free ubuntu runners  

---

## 🎉 You're Done

Your LDAP deployment is:
- ✅ Running now
- ✅ Auto-relaunching every 6 hours
- ✅ Self-healing
- ✅ Completely free
- ✅ Zero maintenance

**Access it:**
1. Check the Actions run summary for the public URL
2. Or run `./deployment-status.sh`
3. Open `https://<url>` in your browser

**Upgrade to permanent URL (optional):**
- Follow `NGROK-AUTO-SETUP.md`
- Takes 5 minutes

---

**Questions?** Check the relevant `.md` file above or the GitHub Actions logs.

**Ready?** Your LDAP is live. 🚀
