# PenuX LDAP Web Interface - Deployment Guide

Deploy the LDAP directory web interface to free hosting.

## Quick Start (5 minutes)

### 1. GitHub Pages Setup

```bash
# Create repository (skip if you already have one)
# https://github.com/new
# Name: yourusername.github.io (must be exact!)

# Clone repository
git clone https://github.com/yourusername/yourusername.github.io
cd yourusername.github.io

# Copy web interface
cp services/openldap/web/index.html ./

# Push to GitHub
git add index.html
git commit -m "Add LDAP directory web interface"
git push origin main

# Enable GitHub Pages
# Repository → Settings → Pages → Source: main / (root)
```

✅ Your web UI is now live at: **https://yourusername.github.io**

### 2. Deploy API Backend

Choose one (Vercel recommended):

#### Vercel (Easiest)

```bash
cd services/openldap/api

# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod

# Set environment variables in Vercel Dashboard:
LDAP_HOST=ldap://your-server:389
LDAP_BASE_DN=dc=penux,dc=uk
LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk
LDAP_ADMIN_PASSWORD=your-password
CORS_ORIGIN=https://yourusername.github.io
```

#### Netlify (Alternative)

```bash
cd services/openldap/api

# Deploy
npm install -g netlify-cli
netlify deploy --prod

# Add environment variables in Netlify dashboard
```

### 3. Configure Web UI

Open `https://yourusername.github.io`

1. Click **⚙️** (Settings) bottom-right
2. Enter API endpoint: `https://your-api.vercel.app` (from step 2)
3. Enter Admin DN: `cn=admin,dc=penux,dc=uk`
4. Enter Password: Your LDAP admin password
5. Click **Save**

✅ **Done!** Your LDAP directory is now accessible online.

---

## Detailed Setup

### Directory Structure

```
yourusername.github.io/
├── index.html                    # Main web interface
├── assets/                       # Optional: images, styles
│   ├── logo.png
│   └── styles.css
├── docs/
│   └── README.md
└── .github/
    └── workflows/
        └── deploy.yml           # Optional: auto-deploy
```

### Files Needed

#### Web UI (`services/openldap/web/index.html`)

Single-file HTML application with:
- ✅ User browsing and search
- ✅ Group management
- ✅ User details view
- ✅ Responsive design
- ✅ Settings panel
- ✅ Status indicators

No build process required - just copy and deploy!

#### API Backend (`services/openldap/api/`)

Node.js/Express server providing:
- ✅ User/group queries
- ✅ Search functionality
- ✅ Credential verification
- ✅ Rate limiting
- ✅ CORS support
- ✅ LDAP authentication

---

## Hosting Options

### Option 1: GitHub Pages (Recommended for Web UI)

**Best for:** Static web interface

Pros:
- ✅ Free unlimited hosting
- ✅ Automatic HTTPS
- ✅ GitHub integration
- ✅ Custom domain support
- ✅ 0 configuration

Cons:
- ⚠️ Static only (but our UI is static JavaScript)

**Setup:**
1. Create `yourusername.github.io` repository (public)
2. Copy `index.html` to root
3. Push to GitHub
4. Wait ~1 minute for deployment
5. Access at `https://yourusername.github.io`

**Custom Domain:**
1. Go to Settings → Pages
2. Add custom domain (e.g., `ldap.penux.uk`)
3. Update DNS CNAME record
4. Enable HTTPS enforcement

### Option 2: Vercel (Recommended for API)

**Best for:** API backend

Pros:
- ✅ Free serverless Node.js
- ✅ GitHub integration
- ✅ Automatic deployments
- ✅ Custom domains
- ✅ Environment variables
- ✅ 1000 requests/day free

Cons:
- ⚠️ Usage limits on free tier

**Setup:**
```bash
npm install -g vercel
cd api
vercel --prod
```

### Option 3: Netlify

**Best for:** Both web UI and API

Pros:
- ✅ Free functions (125k invocations/month)
- ✅ GitHub integration
- ✅ Form handling
- ✅ Redirects and rewrites

Cons:
- ⚠️ More limited than Vercel for APIs

**Setup:**
```bash
npm install -g netlify-cli
netlify deploy --prod
```

### Option 4: Railway

**Best for:** Full-featured backend

Pros:
- ✅ $5/month free credit
- ✅ GitHub integration
- ✅ Databases included
- ✅ Better for long-running services

Cons:
- ⚠️ Can't be truly "free" long-term
- ⚠️ Requires credit card

---

## Configuration Files

### Vercel (`vercel.json`)

```json
{
  "version": 2,
  "builds": [{"src": "server.js", "use": "@vercel/node"}],
  "routes": [{"src": "/(.*)", "dest": "server.js"}]
}
```

Already included in `services/openldap/api/vercel.json`

### Netlify (`netlify.toml`)

```toml
[build]
  command = "npm install"
  functions = "api"

[functions]
  node_bundler = "esbuild"

[env.production]
  LDAP_HOST = "ldap://your-server:389"
```

### GitHub Pages (`_config.yml`)

Optional - allows customization:

```yaml
title: PenuX LDAP Directory
description: Enterprise Directory Management
theme: jekyll-theme-minimal
```

---

## Environment Variables Setup

### For Vercel

1. Go to Vercel Dashboard
2. Select your project
3. Settings → Environment Variables
4. Add each variable:

```
LDAP_HOST = ldap://your-server:389
LDAP_BASE_DN = dc=penux,dc=uk
LDAP_ADMIN_DN = cn=admin,dc=penux,dc=uk
LDAP_ADMIN_PASSWORD = your-secure-password
CORS_ORIGIN = https://yourusername.github.io
```

5. Redeploy for changes to take effect

### For Netlify

1. Go to Site settings
2. Build & deploy → Environment
3. Add variables (same as above)
4. Trigger redeploy

### For Local Testing

Create `.env` file:

```
LDAP_HOST=ldap://localhost:389
LDAP_BASE_DN=dc=penux,dc=uk
LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk
LDAP_ADMIN_PASSWORD=admin123
CORS_ORIGIN=*
```

Then run:
```bash
npm start
```

---

## DNS & Custom Domain

### For Web UI (GitHub Pages)

```
Type: CNAME
Name: ldap
Value: yourusername.github.io
TTL: 3600
```

Then in GitHub Settings → Pages:
- Custom domain: `ldap.penux.uk`
- Check "Enforce HTTPS"

### For API (Vercel)

```
Type: CNAME
Name: api.ldap
Value: <your-deployment>.vercel.app
TTL: 3600
```

Then in Vercel:
- Project Settings → Domains
- Add custom domain: `api.ldap.penux.uk`

---

## Testing Deployment

### Test Web UI

```bash
# 1. Open in browser
https://yourusername.github.io

# 2. Should show
- "Connecting..." initially
- Connection status
- User/group counts
- Search boxes
```

### Test API

```bash
# Health check
curl https://your-api.vercel.app/api/health

# Response should be:
{
  "status": "ok",
  "service": "PenuX LDAP API",
  "timestamp": "..."
}

# Test users endpoint
curl -u cn=admin,dc=penux,dc=uk:password \
  https://your-api.vercel.app/api/users

# Basic auth header is "dn:password" base64 encoded
```

---

## Troubleshooting

### Site Not Updating

```bash
# Hard refresh
Ctrl+Shift+R  (Windows/Linux)
Cmd+Shift+R   (Mac)

# Or clear cache
Open DevTools (F12) → Network → Disable cache → Refresh
```

### API Errors

**"Cannot reach API"**
- Check API URL is correct
- Verify API is deployed
- Check CORS_ORIGIN matches your domain

**"Authentication failed"**
- Verify admin DN is correct
- Verify password is correct
- Check LDAP server is accessible

**"LDAP connection refused"**
- Is LDAP server running?
- Can API reach LDAP server?
- Check firewall rules

### CORS Issues

Error: `Access to XMLHttpRequest blocked by CORS policy`

Fix:
1. Verify CORS_ORIGIN in API matches your web UI domain
2. Redeploy API after changing environment variables
3. Check in API response headers:
   ```
   Access-Control-Allow-Origin: https://yourusername.github.io
   ```

---

## Performance & Limits

### GitHub Pages

- ✅ Unlimited bandwidth
- ✅ Unlimited storage
- ✅ Instant deploys
- ⚠️ 100MB per file limit
- ⚠️ 1GB per repository limit

### Vercel

- ✅ 100GB bandwidth/month free
- ✅ 1000 function invocations free
- ⚠️ Serverless functions have timeout
- ⚠️ Limited to 30 seconds per request

### Netlify

- ✅ 125,000 function invocations/month
- ✅ 100GB bandwidth/month free
- ⚠️ Function execution time limited

---

## Advanced: CI/CD Pipeline

### Auto-deploy on Push

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy-pages:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: .

  deploy-api:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy API to Vercel
        run: |
          npm install -g vercel
          vercel --prod --token ${{ secrets.VERCEL_TOKEN }}
```

---

## Monitoring & Analytics

### GitHub Pages Analytics

- Repository → Insights → Traffic
- See page views and referrers

### Vercel Analytics

- Dashboard → Analytics
- Monitor invocations, bandwidth, performance

### Netlify Analytics

- Analytics tab in Netlify dashboard
- See function invocations

---

## Security Checklist

- ✅ HTTPS enforced (automatic)
- ✅ Credentials never in code (env vars)
- ✅ CORS restricted (not `*`)
- ✅ Rate limiting enabled
- ✅ Read-only LDAP account (optional but recommended)
- ✅ Secrets not in git history
- ✅ `.env` file in `.gitignore`

---

## Support & Resources

- **GitHub Pages**: https://pages.github.com/
- **Vercel Docs**: https://vercel.com/docs
- **Netlify Docs**: https://docs.netlify.com/
- **LDAP Guide**: [README.md](README.md)
- **API Guide**: [GITHUB_PAGES_SETUP.md](GITHUB_PAGES_SETUP.md)

---

## Next Steps

1. ✅ Deploy web UI to GitHub Pages
2. ✅ Deploy API to Vercel/Netlify
3. ✅ Set environment variables
4. ✅ Test connection
5. ✅ Configure custom domain
6. ✅ Monitor performance
7. ✅ Set up backups

**Your LDAP directory is now globally accessible!** 🎉
