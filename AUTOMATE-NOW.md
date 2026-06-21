# ⚡ Automate Everything — Right Now

Choose your automation level. All **free, zero credit card**.

---

## Level 1️⃣: Zero Setup (You are here)

**What you have:** LDAP running on free GitHub Actions  
**What you need to do:** Nothing  
**URL:** Changes every 6 hours (test/demo)  
**Time:** 0 minutes

```bash
# Check it
./deployment-status.sh

# Access it
https://github.com/netanelcyber/penuX/actions/workflows/auto-deploy-tunnel.yml
→ Latest run → Summary → Copy the URL
```

---

## Level 2️⃣: Permanent URL (Recommended)

**What you get:** Static domain that never changes  
**What you need:** Email (for ngrok signup)  
**Time:** 5 minutes

### Ultra-Quick (1 Command)
```bash
./ngrok-full-auto.sh
```

This script:
1. Opens ngrok signup in your browser
2. Waits for you to claim a free domain
3. Asks for token + domain
4. Adds GitHub secrets automatically
5. Triggers deployment → LIVE

**Done.** Your LDAP is at `https://your-domain.ngrok-free.app`

### Manual (if you prefer)
Follow: `NGROK-AUTO-SETUP.md`

---

## Level 3️⃣: Custom Tunnel (Advanced)

Replace the tunnel method with your own:
- AWS ALB + Route 53
- Cloudflare Workers
- Custom reverse proxy
- Etc.

Edit: `.github/workflows/auto-deploy-tunnel.yml` → "Start tunnel" steps

---

## 🎯 Right Now: Pick One

### If you want it **NOW** with a **permanent URL**
```bash
# 1. Make sure GitHub CLI is installed
which gh || brew install gh

# 2. Login to GitHub
gh auth login

# 3. Run the auto-setup
./ngrok-full-auto.sh

# Done! Your LDAP is at https://your-domain.ngrok-free.app
```

### If you want to **test first**, upgrade later
```bash
# Current state: you're running on localhost.run (temporary URL)
# Later: run ./ngrok-full-auto.sh to upgrade to permanent

# Check current deployment
./deployment-status.sh

# Access the test URL from the Actions run summary
```

---

## 📊 Comparison

| Feature | Level 1 | Level 2 | Level 3 |
|---------|---------|---------|---------|
| Setup time | 0 min | 5 min | 30+ min |
| URL | Changes | Permanent | Custom |
| Cost | $0 | $0 | Varies |
| Maintenance | None | None | Some |
| Best for | Testing | Production | Enterprise |

---

## ✨ What You Get

✅ **LDAP running 24/7**  
✅ **Auto-relaunches every 6 hours**  
✅ **Self-heals on failure**  
✅ **Completely free**  
✅ **Zero manual work**  

---

## 🚀 Go Now

```bash
# Permanent URL? (Recommended)
./ngrok-full-auto.sh

# Or check current status
./deployment-status.sh
```

Your LDAP is already live. Pick permanent URL, run one command, done. ✨
