# Cloudflare DNS Configuration for penux.uk

Complete step-by-step guide for configuring **penux.uk** on Cloudflare for the LDAP deployment.

---

## 🎯 Why Cloudflare?

✅ **Free forever** - No charges for DNS, SSL/TLS, DDoS protection  
✅ **Instant propagation** - Changes apply in seconds, not hours  
✅ **Free SSL/TLS** - Automatic HTTPS certificates  
✅ **Performance** - Global CDN edge locations  
✅ **Security** - Built-in DDoS protection, WAF  
✅ **Analytics** - Traffic insights and logs  
✅ **Easy management** - Intuitive dashboard UI  

---

## 📋 Prerequisites

- ✅ **penux.uk domain** registered with any registrar (Namecheap, GoDaddy, etc.)
- ✅ **Access to domain registrar** to change nameservers
- ✅ **Cloudflare account** (free) - https://dash.cloudflare.com
- ✅ **Your machine's public IP address** (find with `curl ifconfig.me`)
- ✅ **GitHub Pages domain** (e.g., `yourusername.github.io`)
- ✅ **Vercel API domain** (e.g., `your-api.vercel.app`)

---

## 🔧 Phase 1: Add Domain to Cloudflare

### Step 1: Create Cloudflare Account

1. Go to https://dash.cloudflare.com
2. Click **Sign up**
3. Enter email and create password
4. Verify email
5. You're now logged in

### Step 2: Add penux.uk to Cloudflare

1. In Cloudflare Dashboard, click **Add a Site**
2. Enter: `penux.uk` (just the domain, no www)
3. Click **Add Site**
4. Select the **Free** plan
5. Click **Create Account**

Cloudflare will scan your existing DNS records (if any).

### Step 3: Review Scanned Records

Cloudflare displays any existing DNS records:
- If you have existing A records for penux.uk, they'll be listed
- Keep or remove as needed
- Continue to next step

### Step 4: Update Nameservers at Registrar

Cloudflare will display **your new nameservers**:

```
Nameserver 1: [something].ns.cloudflare.com
Nameserver 2: [something].ns.cloudflare.com
```

Now update your registrar (GoDaddy, Namecheap, etc.):

#### For GoDaddy:
1. Go to https://godaddy.com (login)
2. Click **My Products** → find penux.uk
3. Click **DNS** (or **Manage**)
4. Replace existing nameservers with Cloudflare's
5. Save changes

#### For Namecheap:
1. Go to https://namecheap.com (login)
2. Click **Manage** next to penux.uk
3. Go to **Nameservers** tab
4. Select **Custom DNS**
5. Enter Cloudflare nameservers
6. Save changes

#### For Other Registrars:
- Find "Nameserver" or "DNS Settings"
- Replace with Cloudflare's nameservers
- Save and wait ~30 minutes for propagation

---

## 🔗 Phase 2: Add DNS Records in Cloudflare

### Step 1: Navigate to DNS Records

1. In Cloudflare Dashboard for penux.uk
2. Click **DNS** in left sidebar
3. Click **Records** tab
4. You'll see the "Add Record" button

### Step 2: Add A Record for Main Domain

**Create first A record:**

| Field | Value |
|-------|-------|
| Type | **A** |
| Name | **@** (or `penux.uk`) |
| IPv4 Address | **your-public-ip** |
| TTL | **Auto** |
| Proxy Status | **DNS only** |

Click **Save**

**Example:**
```
Type: A
Name: @ 
IPv4: 203.0.113.42
TTL: Auto
Proxy: DNS only
```

### Step 3: Add A Record for www

**Create second A record:**

| Field | Value |
|-------|-------|
| Type | **A** |
| Name | **www** |
| IPv4 Address | **your-public-ip** |
| TTL | **Auto** |
| Proxy Status | **DNS only** |

Click **Save**

### Step 4: Add CNAME for LDAP (GitHub Pages)

**Create CNAME for ldap subdomain:**

| Field | Value |
|-------|-------|
| Type | **CNAME** |
| Name | **ldap** |
| Target | **yourusername.github.io** |
| TTL | **Auto** |
| Proxy Status | **DNS only** |

⚠️ **Important:** Set to **DNS only** (not Proxied)  
Why: GitHub Pages requires DNS-only CNAME for custom domains

Click **Save**

**Example:**
```
Type: CNAME
Name: ldap
Target: john-doe.github.io
TTL: Auto
Proxy: DNS only
```

### Step 5: Add CNAME for API (Vercel)

**Create CNAME for api.ldap subdomain:**

| Field | Value |
|-------|-------|
| Type | **CNAME** |
| Name | **api.ldap** |
| Target | **your-api.vercel.app** |
| TTL | **Auto** |
| Proxy Status | **DNS only** |

⚠️ **Important:** Set to **DNS only** (not Proxied)  
Why: Vercel requires DNS-only CNAME for custom domains

Click **Save**

**Example:**
```
Type: CNAME
Name: api.ldap
Target: penux-api.vercel.app
TTL: Auto
Proxy: DNS only
```

### Step 6: (Optional) Add Alternative API CNAME

If you want `api.penux.uk` to also work:

| Field | Value |
|-------|-------|
| Type | **CNAME** |
| Name | **api** |
| Target | **your-api.vercel.app** |
| TTL | **Auto** |
| Proxy Status | **DNS only** |

Click **Save**

---

## 🔍 Phase 3: Verify DNS Records

### View All Records

In Cloudflare DNS page, you should see:

```
Type  Name           Target                    Proxy Status
A     penux.uk       203.0.113.42              DNS only
A     www            203.0.113.42              DNS only
CNAME ldap           yourusername.github.io    DNS only
CNAME api.ldap       your-api.vercel.app       DNS only
CNAME api            your-api.vercel.app       DNS only
```

### Test DNS Resolution

From your terminal:

```bash
# Windows PowerShell
nslookup ldap.penux.uk
nslookup api.ldap.penux.uk
nslookup penux.uk

# Linux/macOS
dig ldap.penux.uk
dig api.ldap.penux.uk
dig penux.uk
```

**Expected output:**
```
ldap.penux.uk points to yourusername.github.io
api.ldap.penux.uk points to your-api.vercel.app
penux.uk points to 203.0.113.42
```

---

## 🔒 Phase 4: Configure SSL/TLS (HTTPS)

### Step 1: Set SSL/TLS Mode

1. In Cloudflare Dashboard
2. Click **SSL/TLS** in left sidebar
3. Click **Overview**
4. **Encryption mode**: Select **Full** (Flexible is not secure)

### Step 2: Always Use HTTPS

1. Click **Edge Certificates** (under SSL/TLS)
2. Toggle **Always Use HTTPS**: **ON**
3. This forces all HTTP traffic to HTTPS

### Step 3: Minimum TLS Version

1. Still under **Edge Certificates**
2. **Minimum TLS Version**: Select **TLS 1.2**
3. This ensures secure connections only

### Step 4: Wait for Certificate

Cloudflare automatically provisions SSL/TLS certificates:
- Usually completes within **5-15 minutes**
- Check status under **Edge Certificates** → **Universal SSL**
- Should show: **Active Certificate** with green checkmark

---

## 🛡️ Phase 5: Security Settings

### Step 1: Enable Security Features

1. Click **Security** in left sidebar
2. Click **Settings**
3. Configure:

| Setting | Value | Reason |
|---------|-------|--------|
| **Challenge Passage** | 30 minutes | Cache challenge results |
| **Browser Integrity Check** | ON | Block suspicious browsers |
| **Hotlink Protection** | ON | Prevent direct image linking |

### Step 2: DDoS Protection

1. Click **DDoS Protection** (under Security)
2. **DDoS** level: Select **High**
3. Cloudflare blocks malicious traffic automatically

### Step 3: Bot Management (Optional)

1. Click **Bots** (under Security)
2. Review detected bot traffic
3. Free plan includes basic bot detection

---

## ⚡ Phase 6: Performance Optimization

### Step 1: Caching

1. Click **Caching** in left sidebar
2. **Cache Level**: Select **Aggressive**
3. **Browser Cache TTL**: Select **4 hours**

### Step 2: Web Optimization

1. Click **Speed** in left sidebar
2. Click **Optimization**
3. Enable all free options:
   - ✅ **Brotli compression**: ON
   - ✅ **Gzip compression**: ON
   - ✅ **Minify CSS**: ON
   - ✅ **Minify JavaScript**: ON
   - ✅ **Minify HTML**: ON

### Step 3: Polish

1. Under **Speed** → **Polish**
2. Set to **Lossless** (or **Lossy** if fine with slight quality loss)

---

## 📊 Phase 7: Monitoring & Analytics

### View Traffic

1. Click **Analytics** in left sidebar
2. View real-time metrics:
   - **Requests**: Total traffic
   - **Bandwidth**: Data transferred
   - **Threats Blocked**: DDoS/malicious traffic
   - **Top Countries**: Traffic by geography

### Monitor Status

1. Click **Status** (top of page)
2. See if any services have issues
3. Check **Performance** metrics

### Check DNS Logs

1. Click **DNS** → **Records**
2. All DNS lookups pass through Cloudflare
3. View in **DNS Query Logs** (Pro+ plan only, but basic info visible)

---

## 🧪 Phase 8: Testing

### Test 1: DNS Resolution

```bash
# Should show Cloudflare nameservers
nslookup -type=NS penux.uk

# Should return your IP
nslookup penux.uk

# Should return GitHub Pages IP
nslookup ldap.penux.uk

# Should return Vercel IP
nslookup api.ldap.penux.uk
```

### Test 2: HTTPS Certificate

```bash
# Check certificate
curl -I https://ldap.penux.uk
curl -I https://api.ldap.penux.uk

# Should show:
# HTTP/2 200 (or 301 redirect)
# Server: cloudflare (proxy) or GitHub/Vercel
```

### Test 3: HTTP Redirect

```bash
# Should redirect to HTTPS
curl -I http://penux.uk
curl -I http://ldap.penux.uk

# Should see 301/302 redirect to https://
```

### Test 4: Website Access

1. Open browser
2. Go to: `https://ldap.penux.uk`
3. Should load your LDAP web interface
4. Check for **secure connection** (🔒 icon)

---

## 🔄 Phase 9: GitHub Pages Integration

### Configure GitHub Pages Custom Domain

1. Go to your GitHub Pages repository
2. Settings → **Pages**
3. Under **Custom domain**:
   - Enter: `ldap.penux.uk`
   - Click **Save**
4. GitHub creates a **CNAME file** automatically
5. Check **Enforce HTTPS**

**Wait 5-10 minutes** for certificate issuance.

Verify by going to: `https://ldap.penux.uk`

---

## 🚀 Phase 10: Vercel API Integration

### Configure Vercel Custom Domain

1. Go to Vercel Dashboard
2. Select your API project
3. Click **Settings**
4. Go to **Domains**
5. Click **Add**:
   - Enter: `api.ldap.penux.uk`
   - Confirm DNS records match Cloudflare
6. Click **Add**

**Wait 5-10 minutes** for deployment.

Verify by going to: `https://api.ldap.penux.uk/api/health`

---

## 📋 Complete DNS Records Reference

After all steps, your Cloudflare DNS should show:

```
Type   Name       Target/Value               TTL    Proxy
────────────────────────────────────────────────────────
A      @          203.0.113.42               Auto   DNS only
A      www        203.0.113.42               Auto   DNS only
CNAME  ldap       yourusername.github.io    Auto   DNS only
CNAME  api.ldap   your-api.vercel.app       Auto   DNS only
CNAME  api        your-api.vercel.app       Auto   DNS only
```

**Important Settings:**
- All CNAMEs: **DNS only** (not Proxied)
- SSL/TLS: **Full**
- Always HTTPS: **ON**
- Security: **High**

---

## 🆘 Troubleshooting

### DNS Not Resolving After Nameserver Change

**Problem:** `nslookup penux.uk` still shows old nameservers

**Solution:**
1. Wait **15-60 minutes** after changing nameservers
2. Check registrar confirms nameserver change
3. Flush local DNS cache:
   ```powershell
   # Windows
   ipconfig /flushdns
   
   # macOS
   sudo dscacheutil -flushcache
   
   # Linux
   sudo systemctl restart systemd-resolved
   ```
4. Try different DNS servers:
   ```bash
   nslookup penux.uk 1.1.1.1  # Cloudflare DNS
   nslookup penux.uk 8.8.8.8  # Google DNS
   ```

### CNAME Record Errors

**Problem:** "Cannot add CNAME record - conflicts with existing record"

**Solution:**
1. Delete conflicting A/AAAA records if they point to the same name
2. Use **@** symbol only for root domain
3. For subdomains (ldap, api), use the subdomain name without **@**

### GitHub Pages Certificate Not Issuing

**Problem:** `https://ldap.penux.uk` shows untrusted certificate

**Solution:**
1. Verify CNAME record in Cloudflare matches GitHub Pages requirement
2. Wait **24 hours** for GitHub to issue certificate
3. Check GitHub Pages settings confirm custom domain is set
4. Try: `curl -I https://ldap.penux.uk` to see current certificate issuer

### Vercel API Not Accessible

**Problem:** `https://api.ldap.penux.uk` shows 404 or connection error

**Solution:**
1. Test Vercel default domain: `https://your-api.vercel.app`
2. If that works, verify DNS in Cloudflare shows correct Vercel nameservers
3. Wait 5-10 minutes after adding domain to Vercel
4. Redeploy in Vercel if needed: `vercel --prod`

### Proxy Status Shows "Proxied" Instead of "DNS only"

**Problem:** GitHub Pages/Vercel breaks when Cloudflare proxies CNAME

**Solution:**
1. Go to Cloudflare DNS Records
2. Find the problematic CNAME
3. Click the **cloud icon** to change from **Proxied** → **DNS only**
4. Save and test again

### Too Many Redirects

**Problem:** Browser shows "Too many redirects" accessing `https://ldap.penux.uk`

**Solution:**
1. Check **Always Use HTTPS** in SSL/TLS settings
2. Verify GitHub Pages doesn't also have redirects configured
3. Try in incognito/private mode to bypass cached redirects
4. Check browser console (F12) for redirect chain

---

## ✅ Success Checklist

- [ ] Cloudflare account created
- [ ] penux.uk added to Cloudflare
- [ ] Nameservers updated at registrar
- [ ] A records for penux.uk and www added
- [ ] CNAME record for ldap added (DNS only)
- [ ] CNAME record for api.ldap added (DNS only)
- [ ] All DNS records verified with nslookup/dig
- [ ] SSL/TLS set to Full
- [ ] Always Use HTTPS enabled
- [ ] Minimum TLS Version set to 1.2
- [ ] Security level set to High
- [ ] Performance optimizations enabled
- [ ] GitHub Pages custom domain configured
- [ ] Vercel custom domain configured
- [ ] Can access https://ldap.penux.uk (loads successfully)
- [ ] Can access https://api.ldap.penux.uk/api/health (returns JSON)
- [ ] Browser shows 🔒 secure connection
- [ ] No certificate warnings

---

## 📊 DNS Propagation Checklist

Track DNS propagation with online tools:

1. **DNSChecker**: https://dnschecker.org/
   - Check penux.uk, ldap.penux.uk, api.ldap.penux.uk
   - Should show Cloudflare IPs globally

2. **What's My DNS**: https://whatsmydns.net/
   - Detailed global DNS propagation map
   - Usually completes in 30 minutes

3. **Cloudflare DNS Checker**: https://dash.cloudflare.com/
   - Built-in status in Cloudflare Dashboard
   - Shows when nameservers active

---

## 🚀 Final Verification

```bash
# 1. Check Cloudflare nameservers active
nslookup -type=NS penux.uk

# 2. Verify all subdomains resolve
nslookup penux.uk
nslookup www.penux.uk
nslookup ldap.penux.uk
nslookup api.ldap.penux.uk

# 3. Test HTTPS certificates
curl -I https://ldap.penux.uk
curl -I https://api.ldap.penux.uk

# 4. Test API endpoint
curl https://api.ldap.penux.uk/api/health

# 5. Verify certificate chain
openssl s_client -connect ldap.penux.uk:443 -servername ldap.penux.uk
```

All commands should return successful responses ✅

---

## 📚 Related Documentation

- **PENUX_UK_SETUP.md** - Complete domain setup guide
- **LDAP_PENUX_UK.md** - LDAP access and management
- **DEPLOY_NOW.md** - Deployment overview
- **GitHub Pages** - https://pages.github.com
- **Vercel Domains** - https://vercel.com/docs/concepts/projects/domains

---

## 🎉 You're Ready!

Your **penux.uk** domain is now configured on Cloudflare with:

✅ Fast DNS resolution  
✅ Automatic HTTPS via SSL/TLS  
✅ Built-in security (DDoS protection)  
✅ Performance optimization  
✅ Complete analytics  
✅ Free forever!

**Next steps:**
1. Access your LDAP directory at: **https://ldap.penux.uk**
2. Access your API at: **https://api.ldap.penux.uk**
3. Enjoy your secure, fast, global LDAP infrastructure! 🚀

