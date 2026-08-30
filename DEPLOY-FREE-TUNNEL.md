# 🚀 Deploy PenuX LDAP with Free Tunnel (No Credit Card!)

Quick deployment using Docker Hub images + free tunnel (no Cloudflare, no static IP needed).

---

## Option A: Zero Setup (Temporary URL)

**No account needed. URL changes each session.**

```bash
./free-tunnel-setup.sh netanelcyber
```

This:
1. Pulls Docker images from Docker Hub
2. Starts all services (OpenLDAP, PostgreSQL, API, Web UI)
3. Opens a **temporary free tunnel** via `localhost.run` (SSH-based, zero signup)

**Output:**
```
✅ Tunnels started via localhost.run!

  Web UI: https://xxxxx-xxxxx.localhost.run
  API:    https://yyyyy-yyyyy.localhost.run
```

---

## Option B: Permanent Free URL (Recommended)

**Sign up with email only — no credit card required.**

### 1. Create ngrok account

Go to **https://ngrok.com** → Sign up with email

### 2. Get your authtoken

1. Go to https://dashboard.ngrok.com/get-started/your-authtoken
2. Copy your token (starts with `eyJ...`)

### 3. Get free static domain

1. Go to https://dashboard.ngrok.com/domains
2. Claim your free domain (e.g., `my-penux.ngrok-free.app`)

### 4. Run setup

```bash
export NGROK_AUTHTOKEN=<paste-your-token-here>
export NGROK_DOMAIN=<your-domain>.ngrok-free.app

./free-tunnel-setup.sh netanelcyber
```

**Your services are now live at:**
```
https://<your-domain>.ngrok-free.app     (Web UI)
https://api.localhost.run or your-api    (API, see logs)
```

---

## Using in GitHub Codespaces

1. **Create Codespaces secret** (optional, for permanent URL):
   - Go to **Settings → Secrets → Codespaces**
   - Add `NGROK_AUTHTOKEN` and `NGROK_DOMAIN`

2. **Open in Codespaces:**
   ```bash
   # Code → Codespaces → New codespace on claude/laughing-cori-jlr7od
   ```

3. **Run setup:**
   ```bash
   ./free-tunnel-setup.sh netanelcyber
   ```

---

## Service URLs

Once running, access via your tunnel:

| Service | Port | URL |
|---------|------|-----|
| **Web UI** | 3001 | `https://<your-tunnel-url>` |
| **API** | 3000 | `https://api.<your-tunnel-url>` |
| **LDAP** | 389 | `ldap://<your-tunnel-url>:389` |

### Test the API

```bash
curl https://<your-tunnel-url>/api/health
# Should return: {"status":"healthy"}
```

---

## Troubleshooting

### Services won't start
```bash
docker compose -f docker-compose-hub.yml logs api
```
Check if containers are healthy:
```bash
docker compose -f docker-compose-hub.yml ps
```

### Tunnel not connecting
```bash
tail -f /tmp/tunnel-web.log
tail -f /tmp/tunnel-api.log
```

### Docker images not found
```bash
docker pull netanelcyber/penux-ldap-api:latest
docker pull netanelcyber/penux-ldap-web:latest
```

---

## Stop Services

```bash
# Stop all services
docker compose -f docker-compose-hub.yml down

# Stop tunnel
kill $(cat /tmp/tunnel-web.pid) 2>/dev/null || true
```

---

## Costs

✅ **100% FREE** (no paid tiers required)

- Docker Hub pulls: free
- ngrok: free tier has unlimited connections
- localhost.run: completely free, no limits
- No static IP needed
- No port forwarding

---

## Advanced: Custom Domain

To use your own domain (e.g., `ldap.mycompany.com`):

1. **With ngrok:**
   - Upgrade to paid ngrok, configure custom domain
   - Add CNAME: `ldap CNAME <your-domain>.ngrok-free.app`

2. **With localhost.run + external DNS:**
   - Not recommended for production (URL changes each session)
   - Use ngrok for stable domains

---

**Ready to deploy?** Run:
```bash
./free-tunnel-setup.sh netanelcyber
```

🎉 Your LDAP system is live!
