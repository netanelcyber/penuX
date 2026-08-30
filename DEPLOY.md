# 🚀 Deploy PenuX LDAP with Free Tunnel (No Credit Card)

**Two methods — pick one:**

## Method A: Permanent Free URL (Recommended)

**Setup (5 minutes):**

1. Sign up at **https://ngrok.com** (email only, free tier)
2. Get your **authtoken**: https://dashboard.ngrok.com/get-started/your-authtoken
3. Get a **free static domain**: https://dashboard.ngrok.com/domains (e.g., `my-ldap.ngrok-free.app`)

**Run:**
```bash
chmod +x free-tunnel-setup.sh
NGROK_AUTHTOKEN=your_token NGROK_DOMAIN=my-ldap.ngrok-free.app \
  ./free-tunnel-setup.sh netanelcyber
```

**Your URLs:**
- Web UI: `https://my-ldap.ngrok-free.app`
- API: Check ngrok dashboard for secondary tunnel URL

**To use your custom domain (`ldap.penux.uk`):**

Add CNAME record in your DNS:
```
ldap    CNAME    my-ldap.ngrok-free.app
api.ldap CNAME   <your-api-ngrok-url-from-dashboard>
penux   CNAME    my-ldap.ngrok-free.app
```

---

## Method B: Zero Signup (Temporary URL)

**Run:**
```bash
chmod +x free-tunnel-setup.sh
./free-tunnel-setup.sh netanelcyber
```

**Your temporary URLs appear in the output** (e.g., `https://abc123.localhost.run`)

⚠️ URLs change every time you restart. Use **Method A** for permanent URLs.

---

## Troubleshooting

**Services won't start:**
```bash
docker compose -f docker-compose-hub.yml logs api
docker compose -f docker-compose-hub.yml logs web
```

**Check tunnel is working:**
```bash
curl https://your-url/api/health   # Should return {"status":"healthy"}
```

**Stop everything:**
```bash
docker compose -f docker-compose-hub.yml down
```

---

## GitHub Actions Auto-Deploy

To auto-deploy on every push to `main`:

1. Add these secrets to GitHub (Settings → Secrets → Actions):
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: Docker Hub access token
   - `SERVER_HOST`: Your server IP/hostname
   - `SERVER_USER`: SSH username (e.g., ubuntu)
   - `SERVER_SSH_KEY`: Your SSH private key
   - `CLOUDFLARE_API_TOKEN`: (optional, for Cloudflare Tunnel)

2. Push to `main` branch
3. GitHub Actions auto-builds, pushes to Docker Hub, deploys to your server

---

## For GitHub Codespaces

1. Open in Codespaces: https://github.com/netanelcyber/penuX/codespaces/new
2. Add secrets (Settings → Secrets → Codespaces):
   - `NGROK_AUTHTOKEN`
   - `NGROK_DOMAIN`
   - `DOCKERHUB_USERNAME`
3. Once loaded, run in terminal:
   ```bash
   ./free-tunnel-setup.sh netanelcyber
   ```

Done! 🎉
