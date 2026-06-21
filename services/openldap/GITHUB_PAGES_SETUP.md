# PenuX LDAP - GitHub Pages & API Deployment Guide

Host your LDAP directory web interface on GitHub Pages for free, with backend API on free services.

## Architecture

```
┌─────────────────────────────────────┐
│   GitHub Pages (Free)               │
│   - Static HTML/CSS/JS              │
│   - LDAP Directory Web UI           │
│   - Domain: yourusername.github.io  │
└──────────────┬──────────────────────┘
               │ API calls
               ▼
┌─────────────────────────────────────┐
│   Backend API (Free Service)        │
│   - Vercel / Netlify / Railway      │
│   - Node.js + Express               │
│   - LDAP Client                     │
└──────────────┬──────────────────────┘
               │ LDAP queries
               ▼
┌─────────────────────────────────────┐
│   Your LDAP Server                  │
│   - Windows/Linux/Docker            │
│   - ldap://your-server:389          │
└─────────────────────────────────────┘
```

---

## Step 1: Set Up GitHub Pages

### 1.1 Create Repository

1. Go to https://github.com/new
2. Create repository: **yourusername.github.io**
   - Must be public
   - Must be named exactly `yourusername.github.io`

3. Clone repository:
```bash
git clone https://github.com/yourusername/yourusername.github.io
cd yourusername.github.io
```

### 1.2 Add Web Interface Files

Copy the web files to your repository:

```bash
# From penuX directory
cp services/openldap/web/index.html ./

# Or structure it:
mkdir -p docs
cp services/openldap/web/index.html docs/index.html
```

### 1.3 Enable GitHub Pages

1. Go to repository Settings
2. Navigate to "Pages" section
3. Select Source: "Deploy from a branch"
4. Branch: `main` (or `master`)
5. Folder: `/ (root)` or `/docs`
6. Click "Save"

Your site will be available at: `https://yourusername.github.io`

### 1.4 Push to GitHub

```bash
git add .
git commit -m "Add LDAP directory web interface"
git push origin main
```

✅ **Web UI is now live!** (but won't work until API is set up)

---

## Step 2: Deploy API Backend

Choose your preferred free service:

### Option A: Vercel (Recommended)

**Free tier includes:**
- Serverless Node.js functions
- Automatic deployments from GitHub
- Custom domains
- Up to 1000 requests/day free tier

#### Setup:

1. **Create Vercel Account**
   - Go to https://vercel.com
   - Sign up with GitHub

2. **Create API Project**
   ```bash
   # In api directory
   cd services/openldap/api
   npm install -g vercel
   vercel login
   vercel
   ```

3. **Configure Environment Variables**
   - In Vercel Dashboard → Settings → Environment Variables
   - Add these variables:

   ```
   LDAP_HOST=ldap://your-server:389
   LDAP_BASE_DN=dc=penux,dc=uk
   LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk
   LDAP_ADMIN_PASSWORD=your-password
   CORS_ORIGIN=https://yourusername.github.io
   ```

4. **Deploy**
   ```bash
   vercel --prod
   ```

5. **Get API URL**
   - Vercel will give you: `https://your-api.vercel.app`

### Option B: Netlify Functions

**Free tier includes:**
- 125,000 function invocations/month
- Automatic Git deployments
- Environment variables

#### Setup:

1. **Create Netlify Account**
   - Go to https://netlify.com
   - Sign up with GitHub

2. **Connect Repository**
   - Click "New site from Git"
   - Select GitHub & repository
   - Build command: `npm install && npm start`
   - Publish directory: `.`

3. **Add Functions**
   - Create `netlify/functions/api.js`
   - Deploy server code as Netlify function

4. **Environment Variables**
   - Site settings → Build & deploy → Environment
   - Add LDAP variables

### Option C: Railway

**Free tier includes:**
- $5/month free credit
- Node.js support
- GitHub integration

#### Setup:

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub"
4. Choose repository
5. Add environment variables
6. Deploy

---

## Step 3: Connect Web UI to API

### 3.1 Update API Endpoint

After deploying API, update the web UI:

**Option A: Hardcode in index.html**
```javascript
// In index.html, change:
let config = {
    apiEndpoint: 'https://your-api.vercel.app',  // Change this
    // ...
};
```

**Option B: Use Settings in Web UI**
1. Open https://yourusername.github.io
2. Click ⚙️ (Settings) in bottom right
3. Enter API endpoint: `https://your-api.vercel.app`
4. Enter Admin DN and Password
5. Click Save

---

## Step 4: Verify Setup

### 4.1 Test API

```bash
# Test health check
curl https://your-api.vercel.app/api/health

# Expected response:
{
  "status": "ok",
  "service": "PenuX LDAP API",
  "timestamp": "2026-06-21T..."
}
```

### 4.2 Test Web UI

1. Open https://yourusername.github.io
2. Configure API endpoint
3. Should see:
   - ✓ Connected status
   - User count
   - Group count
   - List of users and groups

---

## Custom Domain Setup

### Using GitHub Pages with Custom Domain

1. **Add DNS Records**
   - For `ldap.penux.uk`:
   
   **DNS Provider (e.g., Cloudflare):**
   ```
   Type: CNAME
   Name: ldap
   Value: yourusername.github.io
   TTL: 3600
   ```

2. **Configure in GitHub**
   - Repository Settings → Pages
   - Custom domain: `ldap.penux.uk`
   - Check "Enforce HTTPS"

3. **Access**
   - https://ldap.penux.uk

### Using Custom Domain for API

**Vercel:**
1. Vercel Dashboard → Project Settings → Domains
2. Add custom domain: `api.penux.uk`
3. Update DNS records as shown

**Update Web UI:**
```javascript
let config = {
    apiEndpoint: 'https://api.penux.uk',
    // ...
};
```

---

## Environment Variables Reference

### For API Server

| Variable | Example | Description |
|----------|---------|-------------|
| `LDAP_HOST` | `ldap://192.168.1.100:389` | LDAP server address |
| `LDAP_BASE_DN` | `dc=penux,dc=uk` | LDAP base DN |
| `LDAP_ADMIN_DN` | `cn=admin,dc=penux,dc=uk` | Admin DN for connections |
| `LDAP_ADMIN_PASSWORD` | `secure-password` | Admin password |
| `CORS_ORIGIN` | `https://yourusername.github.io` | Allowed origin for CORS |
| `PORT` | `3000` | Server port (optional) |

---

## API Endpoints Reference

### Public Endpoints

```
GET  /api/health              Health check
POST /api/verify              Verify credentials
```

### Authenticated Endpoints (require Basic Auth)

```
GET  /api/users               Get all users
GET  /api/users/:uid          Get specific user
GET  /api/groups              Get all groups
GET  /api/groups/:cn          Get specific group
GET  /api/search?query=X      Search users/groups
GET  /api/ous                 Get organization units
GET  /api/stats               Get statistics
```

### Authentication

All authenticated endpoints require `Authorization` header:

```
Authorization: Basic base64(dn:password)
```

Example:
```bash
curl -H "Authorization: Basic Y249YWRtaW4sZGM9cGVudXgsZGM9dWs6cGFzc3dvcmQ=" \
  https://your-api.vercel.app/api/users
```

---

## Security Best Practices

### 1. Never Commit Credentials

❌ **Bad:**
```javascript
LDAP_ADMIN_PASSWORD: "admin123"  // In code!
```

✅ **Good:**
```
// Use environment variables only
const password = process.env.LDAP_ADMIN_PASSWORD;
```

### 2. Use HTTPS Only

- GitHub Pages: Automatic HTTPS ✓
- Vercel/Netlify: Automatic HTTPS ✓
- Custom domains: Enable HTTPS enforcement

### 3. Restrict CORS

```javascript
CORS_ORIGIN: "https://yourusername.github.io"  // Not *
```

### 4. Rate Limiting

Already implemented in API:
- 100 requests per 15 minutes per IP
- Adjust in `server.js`:

```javascript
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100  // Change this
});
```

### 5. Use Separate LDAP Account

Create read-only LDAP user instead of admin:

```ldif
dn: cn=api-user,ou=applications,dc=penux,dc=uk
objectClass: inetOrgPerson
cn: api-user
sn: API User
uid: api-user
userPassword: {SSHA}...
accountStatus: active
```

---

## Troubleshooting

### API Not Accessible

```bash
# Check if API is running
curl https://your-api.vercel.app/api/health

# Check logs
# Vercel: Dashboard → Functions → Logs
# Netlify: Site → Functions → Logs
# Railway: Deployments → Logs
```

### CORS Errors

Error: `Access to XMLHttpRequest blocked by CORS policy`

**Fix:**
1. Check API `CORS_ORIGIN` environment variable
2. Ensure it matches your web UI domain
3. Restart/redeploy API

### Connection Refused

Error: `Connection refused to LDAP server`

**Fix:**
1. Verify LDAP server is accessible from API location
2. Check firewall rules
3. Verify `LDAP_HOST` is correct
4. Test locally first:
   ```bash
   ldapsearch -x -H ldap://your-server:389 ...
   ```

### Wrong Credentials

Error: `Invalid credentials`

**Fix:**
1. Verify `LDAP_ADMIN_DN` is correct
2. Verify `LDAP_ADMIN_PASSWORD` is correct
3. Test with `ldapwhoami`:
   ```bash
   ldapwhoami -H ldap://your-server:389 \
     -D "cn=admin,dc=penux,dc=uk" -w password
   ```

---

## Advanced: Custom Domain for Everything

### Full Setup with penux.uk

```
penux.uk
├── www.penux.uk              → Website/landing page
├── ldap.penux.uk             → LDAP directory (GitHub Pages)
├── api.ldap.penux.uk         → API backend (Vercel)
└── admin.penux.uk            → Server admin panel
```

### DNS Configuration (Cloudflare Example)

```
Type    Name              Content
A       penux.uk          <your-ip>
A       www               <your-ip>
CNAME   ldap              yourusername.github.io
CNAME   api.ldap          your-api.vercel.app
```

---

## Performance Optimization

### 1. Cache Users/Groups

Update web UI to cache data:

```javascript
// Cache for 5 minutes
const CACHE_DURATION = 5 * 60 * 1000;
let cache = {
  users: { data: null, timestamp: 0 },
  groups: { data: null, timestamp: 0 }
};

async function loadUsers() {
  const now = Date.now();
  if (cache.users.data && (now - cache.users.timestamp) < CACHE_DURATION) {
    return cache.users.data;
  }
  // Fetch from API...
}
```

### 2. Compress API Response

Already implemented via gzip in Vercel/Netlify.

### 3. Database Connection Pooling

For production, use connection pooling in API.

---

## Monitoring

### Vercel

- Dashboard → Analytics
- Monitor invocations, bandwidth, performance

### Netlify

- Site analytics → Functions
- Monitor invocation counts

### GitHub Pages

- Repository insights
- Track traffic

---

## Backup and Maintenance

### Regular Backups

```bash
# Backup LDAP database
.\backup.ps1 backup -Compress

# Store backups in GitHub (encrypted)
git add ldap_backup*.ldif.gz
git commit -m "Backup: $(date)"
git push
```

### Keep Dependencies Updated

```bash
# Check for updates
npm outdated

# Update packages
npm update

# Redeploy
vercel --prod
```

---

## Cost Summary (All Free!)

| Service | Cost | Included |
|---------|------|----------|
| GitHub Pages | Free | Unlimited bandwidth |
| Vercel API | Free | 1000 req/day |
| Netlify Functions | Free | 125k invocations/month |
| Railway | Free | $5/month credit |
| Domain (Cloudflare) | Free | CNAME records, DNS |
| **Total** | **Free** | ✅ |

---

## Next Steps

1. ✅ Set up GitHub Pages repository
2. ✅ Deploy API to Vercel/Netlify
3. ✅ Configure environment variables
4. ✅ Update API endpoint in web UI
5. ✅ Test web interface
6. ✅ Set up custom domain (optional)
7. ✅ Configure backups
8. ✅ Monitor performance

---

## Support Resources

- **GitHub Pages Docs**: https://pages.github.com/
- **Vercel Docs**: https://vercel.com/docs
- **Netlify Docs**: https://docs.netlify.com/
- **LDAP Documentation**: https://www.openldap.org/
- **Express.js Docs**: https://expressjs.com/

---

## Example: Complete GitHub Actions Deploy

Auto-deploy to GitHub Pages when you push:

`.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./web
```

---

## Success! 🎉

Your LDAP directory is now:
- ✅ Publicly accessible via GitHub Pages
- ✅ API running on free serverless platform
- ✅ Secure HTTPS by default
- ✅ Custom domain support
- ✅ Zero cost hosting

Access it at: **https://yourusername.github.io**
