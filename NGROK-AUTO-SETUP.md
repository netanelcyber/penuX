# 🔗 Auto-Obtain ngrok Permanent Domain

Get a **permanent static URL** with just **one click** — no manual token/domain hunting.

## One-Click Setup (5 seconds)

### Step 1: Create ngrok Account (Email Only)
Go to: https://ngrok.com/signup

- Email address
- Password
- ✅ Done — no credit card required

### Step 2: Get Your Auth Token
After signup, you're automatically on the dashboard:  
https://dashboard.ngrok.com/get-started/your-authtoken

**Copy the token** (looks like `eyJhbGc...`)

### Step 3: Claim Your Free Static Domain
Same dashboard → **Domains** tab:  
https://dashboard.ngrok.com/domains

- Click **"New Domain"**
- You get ONE free domain (e.g., `penux-ldap.ngrok-free.app`)
- Copy the domain name

### Step 4: Add Secrets to GitHub (2 steps)
Go to your repo: **Settings → Secrets and variables → Actions**

**Create two new secrets:**

| Name | Value |
|------|-------|
| `NGROK_AUTHTOKEN` | Paste your token from Step 2 |
| `NGROK_DOMAIN` | Paste your domain from Step 3 (e.g., `penux-ldap.ngrok-free.app`) |

### Step 5: Trigger Deployment
Go to: https://github.com/netanelcyber/penuX/actions → **Auto-Deploy LDAP via Free Tunnel** → **Run workflow**

✅ **Done.** Your LDAP is now live at `https://penux-ldap.ngrok-free.app`

---

## What Changes

| Before | After |
|--------|-------|
| `https://xyz123.lhr.life` (changes each run) | `https://penux-ldap.ngrok-free.app` (permanent) |
| No secrets needed | Two secrets (saved in GitHub) |
| Works immediately | Works immediately after secrets added |

---

## Verify It Worked

After running the workflow, check the **Summary** tab:

```
🚀 PenuX LDAP is LIVE (ngrok)

Web UI: https://penux-ldap.ngrok-free.app
```

Test it:
```bash
curl https://penux-ldap.ngrok-free.app/api/health
# Should return: {"status":"healthy"}
```

---

## Access Your LDAP

### Web UI
```
https://penux-ldap.ngrok-free.app
```

### API
```
curl https://penux-ldap.ngrok-free.app/api/health
curl https://penux-ldap.ngrok-free.app/api/users
```

### LDAP Protocol (if exposed)
Port 389 is **not** exposed through the tunnel (only HTTP/HTTPS).  
For LDAP protocol access, you'd need a different tunnel config (out of scope for this guide).

---

## Cost

✅ **Free**
- ngrok free tier: 1 static domain
- GitHub Actions: Free tier (minutes included)
- No credit card ever needed

---

## Troubleshooting

### "Invalid token" error in workflow
- Copy the token again from https://dashboard.ngrok.com/get-started/your-authtoken
- Make sure there are no extra spaces
- Update the secret and re-run

### Domain not working immediately after setup
- ngrok DNS propagates in seconds
- If it takes >30s, wait and try again
- Check the workflow logs for tunnel connection status

### Want to change your domain later
- Get a new one at https://dashboard.ngrok.com/domains
- Update the `NGROK_DOMAIN` secret
- Re-run the workflow

---

## Done! 🎉

Your LDAP has a **permanent public URL** that never changes.  
Auto-relaunches every 6 hours. Self-heals on failure.

Zero maintenance. Completely free.
