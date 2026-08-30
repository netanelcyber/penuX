# 🚀 Automated LDAP Deployment

Your PenuX LDAP system is now **fully automated** — zero manual setup needed after initial deploy.

## Status

**Current Deployment:** Running on GitHub Actions free ubuntu runner  
**URL:** See https://github.com/netanelcyber/penuX/actions → **Auto-Deploy LDAP via Free Tunnel** → latest run → Summary

```bash
./deployment-status.sh
```

---

## How It Works

### 1. GitHub Actions Automation (`auto-deploy-tunnel.yml`)

**What it does:**
- Boots the entire LDAP stack (OpenLDAP + PostgreSQL + API + Web UI)
- Exposes it via a free tunnel (localhost.run by default)
- Keeps it alive for ~5h50m, then auto-relaunches every 6 hours
- Self-heals: restarts the stack if the API becomes unhealthy

**No secrets required to start.** Deployment goes green immediately.

**Triggers:**
- 🔄 **Schedule:** Every 6 hours (via cron)
- 📤 **Push:** Any change to `docker-compose-hub.yml` or the workflow itself
- 🎯 **Manual:** Click "Run workflow" in the GitHub Actions tab

### 2. GitHub Codespaces Auto-Start (`autostart.sh`)

When you open the repo in GitHub Codespaces:
- Waits for Docker daemon to be ready
- Automatically boots the stack + tunnel
- No manual `./free-tunnel-setup.sh` needed

**To use:**
1. Open repo in Codespaces on branch `claude/laughing-cori-jlr7od`
2. Stack + tunnel auto-start in the background
3. Get the public URL from the terminal output

---

## Tunnel Options

### Default: **localhost.run** (Zero Signup)
- ✅ Works immediately with zero secrets
- ✅ No credit card required
- ⚠️ URL changes each run
- Perfect for testing

### Upgrade to: **ngrok** (Permanent Static Domain)
- ✅ Permanent URL — never changes
- ✅ Email signup only (no credit card)
- ✅ Free tier: 1 static domain

**To enable:**
1. Sign up at https://ngrok.com (email only)
2. Get your token: https://dashboard.ngrok.com/get-started/your-authtoken
3. Claim a free domain: https://dashboard.ngrok.com/domains
4. Add repo secrets (Settings → Secrets and variables → Actions):
   - `NGROK_AUTHTOKEN` = your token
   - `NGROK_DOMAIN` = your domain (e.g., `penux-ldap.ngrok-free.app`)
5. Next auto-run (or trigger manually) upgrades to permanent URL

---

## Deployment Lifecycle

```
GitHub Actions runs every 6 hours
         ↓
Bootstrap docker, pull images (30s)
         ↓
Start 4 containers: openldap, postgres, api, web (60s)
         ↓
Open tunnel: localhost.run or ngrok (3s)
         ↓
Keep deployment alive for ~350 minutes
         ↓
Health-check every 60s, auto-restart if API drops
         ↓
Job expires after ~6 hours
         ↓
Cron triggers next run → cycle repeats
```

---

## Commands

### Check Deployment Status
```bash
./deployment-status.sh
```

### Manual Trigger
In GitHub Actions tab → **Auto-Deploy LDAP via Free Tunnel** → **Run workflow**

### View Logs
https://github.com/netanelcyber/penuX/actions → **Auto-Deploy LDAP via Free Tunnel**

### Local Testing (no automation)
```bash
./free-tunnel-setup.sh netanelcyber
```

---

## Self-Healing

The deployment automatically restarts the LDAP stack if the API becomes unhealthy:

```yaml
# Every 60 seconds:
- Check if API is healthy
- If unhealthy or down → docker compose up -d (restart)
- Continue running
```

No manual intervention needed — it fixes itself.

---

## Costs

✅ **Completely Free**
- GitHub Actions: Free tier (2000 minutes/month per account, more than enough)
- Tunnel: localhost.run (zero signup) or ngrok free tier (email only)
- Docker Hub: Free pulls (no login needed for public images)

---

## File Structure

```
.github/workflows/
  └── auto-deploy-tunnel.yml       # Main automation workflow

.devcontainer/
  ├── devcontainer.json             # Codespaces config
  ├── setup.sh                       # Post-create hook
  └── autostart.sh                   # Post-start hook (boots stack)

docker-compose-hub.yml              # Full stack definition
free-tunnel-setup.sh                # Manual tunnel script
deployment-status.sh                # Status dashboard
DEPLOYMENT-AUTO.md                  # This file
```

---

## Troubleshooting

### No Public URL in the run summary
- Check the "Start localhost.run tunnel" step in the Actions log
- If it failed, check for SSH errors (localhost.run might be having issues)
- Next run (in 6h or manual trigger) should work

### API not becoming healthy
- Check docker-compose-hub.yml for environment variable issues
- View the "api" container logs in the Actions run

### Need a permanent URL immediately?
- Add ngrok secrets (see "Upgrade to ngrok" above)
- Run workflow manually (no need to wait 6 hours)

---

## Next Steps

1. ✅ Deployment is running now
2. Find the public URL in the Actions run summary
3. Test your LDAP at `https://<url>/` (Web UI) and `https://<url>/api/health` (API)
4. (Optional) Add ngrok secrets for a permanent domain
5. Monitor via `./deployment-status.sh` or GitHub Actions tab

---

**Your LDAP is live. Zero maintenance. Auto-scaling by schedule. Self-healing on failure.**

🎉 Done.
