# 🌐 Deploy PenuX LDAP to Heroku via Web Dashboard

Complete guide to deploy without using CLI—everything through the Heroku website.

---

## ⚡ Quick Deploy (10 minutes)

### Step 1: Create Heroku Account

1. Go to **https://dashboard.heroku.com**
2. Click **"Sign up"** (top right)
3. Fill in:
   - Email
   - Password
   - Company name (optional)
4. Click **"Create free account"**
5. Verify email (check inbox)

---

### Step 2: Create First App (REST API)

1. Go to **https://dashboard.heroku.com/apps**
2. Click **"New"** button (top right)
3. Click **"Create new app"**
4. Fill in:
   - **App name:** `penux-ldap-api` (must be unique)
   - **Region:** Choose your region (US or Europe)
5. Click **"Create app"**

---

### Step 3: Connect GitHub Repository

1. In your new app, go to **"Deploy"** tab
2. Under **"Deployment method"**, click **"Connect to GitHub"**
3. Click **"Connect to GitHub"** button
4. Authorize Heroku with GitHub
5. Search for: `penuX`
6. Click **"Connect"** next to the repository

**You should see:** "Connected to netanelcyber/penuX"

---

### Step 4: Configure for API Deployment

1. Still in **"Deploy"** tab, scroll down to **"Manual deploy"**
2. Under **"Choose a branch to deploy"**, select: `claude/laughing-cori-jlr7od`
3. Click **"Deploy Branch"**

**Wait for deployment to complete** (shows green checkmark)

---

### Step 5: Set Environment Variables

1. Go to **"Settings"** tab
2. Scroll to **"Config Vars"**
3. Click **"Reveal Config Vars"**
4. Add each variable by clicking **"Add"**:

| Key | Value |
|-----|-------|
| `LDAP_HOST` | `ldaps://ldaps-server.penux.uk:636` |
| `LDAP_BASE_DN` | `dc=penux,dc=uk` |
| `LDAP_ADMIN_DN` | `cn=admin,dc=penux,dc=uk` |
| `LDAP_ADMIN_PASSWORD` | `YourSecurePassword123!` |
| `CORS_ORIGIN` | `*` |
| `NODE_ENV` | `production` |

5. Click **"Add"** for each variable

---

### Step 6: Add PostgreSQL Database

1. In your app, go to **"Resources"** tab
2. Search for: `postgres`
3. Click **"Heroku Postgres"**
4. Select plan: **"Hobby Dev"** (free, good for testing)
5. Click **"Provision"**

**Wait for database to be created** (may take 1-2 minutes)

---

### Step 7: Create Second App (Web UI)

1. Go to **https://dashboard.heroku.com/apps**
2. Click **"New"** → **"Create new app"**
3. App name: `penux-ldap-web`
4. Click **"Create app"**
5. Repeat Steps 3-4 (connect GitHub, deploy branch)
6. Go to **"Settings"** → **"Config Vars"** → **"Reveal Config Vars"**

Add:

| Key | Value |
|-----|-------|
| `API_URL` | `https://penux-ldap-api.herokuapp.com` |
| `NODE_ENV` | `production` |

7. Click **"Add"** for each

---

## ✅ After Deployment

### View Your Apps

1. Go to **https://dashboard.heroku.com/apps**
2. You should see both apps:
   - ✅ `penux-ldap-api`
   - ✅ `penux-ldap-web`

### Get Your URLs

1. Click **`penux-ldap-api`**
2. Top right, click **"Open app"**
3. Your URL: `https://penux-ldap-api.herokuapp.com`

Repeat for Web UI to get: `https://penux-ldap-web.herokuapp.com`

---

### Test the API

1. Open browser
2. Go to: `https://penux-ldap-api.herokuapp.com/api/health`
3. You should see: `{"status":"healthy"}`

**To test with credentials:**

```bash
curl -u "cn=admin,dc=penux,dc=uk:YourPassword" \
  https://penux-ldap-api.herokuapp.com/api/users
```

---

### View Logs

1. In your app, go to **"More"** (top right) → **"View logs"**
2. Watch real-time logs as requests come in
3. Look for errors (red text)

---

## 🔧 Redeploy After Changes

If you make code changes:

1. Go to app **"Deploy"** tab
2. Scroll to **"Manual deploy"**
3. Click **"Deploy Branch"** again
4. Wait for green checkmark

---

## 🌐 Add Custom Domains (Optional)

### For API (api.ldap.penux.uk)

1. Go to `penux-ldap-api` app
2. Click **"Settings"**
3. Scroll to **"Domains"**
4. Click **"Add domain"**
5. Enter: `api.ldap.penux.uk`
6. Copy the **CNAME target** shown

**At your domain registrar (Namecheap, GoDaddy, etc.):**

1. Go to DNS settings
2. Add CNAME record:
   ```
   Type: CNAME
   Name: api.ldap
   Value: (paste Heroku CNAME)
   TTL: 3600
   ```
3. Save

**Wait 24-48 hours for DNS propagation**

Then update config vars:
1. Go to `penux-ldap-web` app → **"Settings"** → **"Config Vars"**
2. Change `API_URL` to: `https://api.ldap.penux.uk`

---

## 📊 Monitor Your Apps

### View Metrics

1. In your app, click **"Metrics"** (top)
2. See CPU, memory, and network usage
3. Check for errors or high resource use

### View Recent Activity

1. Go to **"Activity"** tab
2. See all deployments and changes
3. Click any deployment to see build logs

---

## 🐛 Troubleshooting

### App Won't Start (Red Status)

1. Click **"View logs"** (More menu)
2. Look for error messages (red text)
3. Common issues:
   - Missing environment variable
   - Database not connected
   - Port not correct

**Fix:** Add missing variables to **Settings** → **Config Vars**

### Getting 503 Error

1. App may be starting up (takes 30 seconds)
2. Wait and refresh
3. If persists, check logs

### Database Connection Failed

1. Go to **"Resources"** tab
2. Click **Heroku Postgres**
3. Check database status (should be green)
4. Click **"Settings"** to view connection details

---

## 💰 Costs

**Free tier:**
- ✅ 2 Hobby dynos (free)
- ✅ 1 Hobby PostgreSQL (free)
- **Total: $0/month**

**With paid resources:**
- Standard-1x dyno: $7/month each
- Standard PostgreSQL: $50/month
- **Typical production: $64/month**

---

## 🔐 Security

After deployment:

### Change Default Passwords

1. Go to API app → **"Settings"** → **"Config Vars"**
2. Click **"Edit"** on `LDAP_ADMIN_PASSWORD`
3. Enter new secure password
4. Click **"Save changes"**

### Enable Force HTTPS

1. Go to **"Settings"**
2. Scroll to **"SSL/TLS"**
3. Click **"Configure SSL"**
4. Select **"Automatic"**

### View Security Info

1. Go to **"Settings"**
2. See **"Certificates"** section

---

## 📈 Scale Up Later

### Change Dyno Type

1. Go to **"Dynos"** tab
2. Click **"Change dyno type"**
3. Select:
   - **Hobby** (free, sleeps)
   - **Standard-1x** ($7/month)
   - **Professional** ($50+/month)
4. Click **"Confirm"**

### Scale Database

1. Go to **"Resources"**
2. Click **Heroku Postgres**
3. Click **"Upgrade"**
4. Select new plan
5. Click **"Confirm"**

---

## 📞 Support

- **Heroku Status:** https://status.heroku.com
- **Heroku Docs:** https://devcenter.heroku.com
- **PenuX GitHub:** https://github.com/netanelcyber/penuX

---

## ✨ What's Next?

After deployment:

1. ✅ Test API: `https://penux-ldap-api.herokuapp.com/api/health`
2. ✅ Access Web UI: `https://penux-ldap-web.herokuapp.com`
3. ✅ Create LDAP users via API
4. ✅ Add custom domains (optional)
5. ✅ Monitor metrics and logs
6. ✅ Set up backups (optional)

---

## 🎯 Quick Checklist

- [ ] Created Heroku account
- [ ] Created `penux-ldap-api` app
- [ ] Connected GitHub repository
- [ ] Set environment variables for API
- [ ] Created PostgreSQL database
- [ ] Deployed API successfully
- [ ] Created `penux-ldap-web` app
- [ ] Connected GitHub to Web app
- [ ] Set API_URL environment variable for Web
- [ ] Deployed Web UI successfully
- [ ] Tested API health endpoint
- [ ] Viewed logs (no errors)
- [ ] Accessed Web UI in browser
- [ ] (Optional) Added custom domain
- [ ] (Optional) Changed default passwords

---

**You're ready! 🚀**

Your apps are now live:
- 🔗 API: `https://penux-ldap-api.herokuapp.com`
- 🌐 Web: `https://penux-ldap-web.herokuapp.com`

For detailed information, see `/HEROKU-DEPLOY.md`
