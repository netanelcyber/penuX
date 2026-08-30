# 🐳 Docker Hub + Cloudflare DNS Integration

Complete guide to deploy PenuX LDAP on Docker Hub with Cloudflare DNS, security, and DDoS protection.

---

## ⚡ Quick Setup (20 minutes)

### Prerequisites

- ✅ Docker Hub images deployed (running locally or on server)
- ✅ Own domain (e.g., penux.uk)
- ✅ Cloudflare account (free tier available)

---

## 📊 Architecture

```
┌─────────────────────────────────────────┐
│         Cloudflare DNS + CDN            │
│  (DDoS Protection, Caching, SSL/TLS)    │
└──────────────┬──────────────────────────┘
               │
        ┌──────▼──────┐
        │  Your Domain │
        │  (penux.uk)  │
        └──────┬───────┘
               │
    ┌──────────┴──────────┐
    │                     │
┌───▼────┐          ┌────▼───┐
│   API  │          │   Web  │
│ :3000  │          │ :3001  │
└────────┘          └────────┘
   Docker Hub Services
```

---

## Step 1: Cloudflare Setup

### 1.1 Create Cloudflare Account

1. Go to **https://dash.cloudflare.com/sign-up**
2. Sign up with email
3. Verify email
4. Log in

### 1.2 Add Domain to Cloudflare

1. Click **"Add a site"** (top left)
2. Enter your domain: `penux.uk`
3. Click **"Add site"**
4. Select plan: **Free** (good for testing)
5. Click **"Continue"**

### 1.3 Update Nameservers

Cloudflare shows your new nameservers:
```
nsX.cloudflare.com
nsY.cloudflare.com
```

Go to your domain registrar (GoDaddy, Namecheap, etc.):

1. Login to registrar
2. Find DNS/Nameserver settings
3. Replace old nameservers with Cloudflare ones
4. Save
5. **Wait 5-30 minutes** for propagation

---

## Step 2: DNS Records

### 2.1 Create DNS Records in Cloudflare

1. In Cloudflare, go to **"DNS"** (top menu)
2. Click **"Add record"**

**Create Record 1 (API):**
- Type: `A`
- Name: `api`
- IPv4: Your server IP (or `127.0.0.1` for local)
- TTL: `Auto`
- Proxy: `Proxied` (orange cloud = Cloudflare protection)
- Click **"Save"**

**Create Record 2 (Web UI):**
- Type: `A`
- Name: `ldap`
- IPv4: Same as above
- TTL: `Auto`
- Proxy: `Proxied`
- Click **"Save"**

**Result:**
```
api.penux.uk  →  your-server-ip
ldap.penux.uk →  your-server-ip
```

---

## Step 3: Nginx Configuration

### 3.1 Update Nginx to Route by Domain

On your server, update `/etc/nginx/sites-available/penux`:

```nginx
upstream api {
    server localhost:3000;
}

upstream web {
    server localhost:3001;
}

server {
    listen 80;
    server_name api.penux.uk ldap.penux.uk penux.uk;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.penux.uk;

    ssl_certificate /etc/letsencrypt/live/api.penux.uk/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.penux.uk/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Trust Cloudflare
    set_real_ip_from 173.245.48.0/20;
    set_real_ip_from 103.21.244.0/22;
    set_real_ip_from 103.22.200.0/22;
    set_real_ip_from 103.31.4.0/22;
    set_real_ip_from 141.101.64.0/18;
    set_real_ip_from 108.162.192.0/18;
    set_real_ip_from 190.93.240.0/20;
    set_real_ip_from 188.114.96.0/20;
    set_real_ip_from 197.234.240.0/22;
    set_real_ip_from 198.41.128.0/17;
    set_real_ip_from 162.158.0.0/15;
    set_real_ip_from 104.16.0.0/13;
    set_real_ip_from 104.24.0.0/14;
    set_real_ip_from 172.64.0.0/13;
    set_real_ip_from 131.0.72.0/22;
    set_real_ip_from 2400:cb00::/32;
    set_real_ip_from 2606:4700::/32;
    set_real_ip_from 2803:f800::/32;
    set_real_ip_from 2405:b500::/32;
    set_real_ip_from 2405:8100::/32;
    set_real_ip_from 2a06:98c0::/29;
    set_real_ip_from 2c0f:f248::/32;
    real_ip_header CF-Connecting-IP;

    location / {
        proxy_pass http://api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 443 ssl http2;
    server_name ldap.penux.uk penux.uk;

    ssl_certificate /etc/letsencrypt/live/penux.uk/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/penux.uk/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Trust Cloudflare (same as above)
    set_real_ip_from 173.245.48.0/20;
    set_real_ip_from 103.21.244.0/22;
    set_real_ip_from 103.22.200.0/22;
    set_real_ip_from 103.31.4.0/22;
    set_real_ip_from 141.101.64.0/18;
    set_real_ip_from 108.162.192.0/18;
    set_real_ip_from 190.93.240.0/20;
    set_real_ip_from 188.114.96.0/20;
    set_real_ip_from 197.234.240.0/22;
    set_real_ip_from 198.41.128.0/17;
    set_real_ip_from 162.158.0.0/15;
    set_real_ip_from 104.16.0.0/13;
    set_real_ip_from 104.24.0.0/14;
    set_real_ip_from 172.64.0.0/13;
    set_real_ip_from 131.0.72.0/22;
    set_real_ip_from 2400:cb00::/32;
    set_real_ip_from 2606:4700::/32;
    set_real_ip_from 2803:f800::/32;
    set_real_ip_from 2405:b500::/32;
    set_real_ip_from 2405:8100::/32;
    set_real_ip_from 2a06:98c0::/29;
    set_real_ip_from 2c0f:f248::/32;
    real_ip_header CF-Connecting-IP;

    location / {
        proxy_pass http://web;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3.2 Enable and Restart Nginx

```bash
# Check config
sudo nginx -t

# Enable if new
sudo ln -s /etc/nginx/sites-available/penux /etc/nginx/sites-enabled/

# Restart
sudo systemctl restart nginx
```

---

## Step 4: SSL/TLS with Let's Encrypt

### 4.1 Install Certbot

```bash
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx -y
```

### 4.2 Get SSL Certificates

```bash
# For API domain
sudo certbot --nginx -d api.penux.uk

# For Web domain
sudo certbot --nginx -d ldap.penux.uk -d penux.uk
```

When prompted:
- Enter email
- Agree to terms
- Choose redirect: **2** (redirect HTTP to HTTPS)

### 4.3 Auto-Renewal

```bash
# Test renewal
sudo certbot renew --dry-run

# Auto-renewal is enabled by default
sudo systemctl status certbot.timer
```

---

## Step 5: Cloudflare SSL/TLS Settings

### 5.1 Configure Cloudflare SSL

1. In Cloudflare dashboard, go to **"SSL/TLS"** (left menu)
2. Click **"Overview"**
3. Select: **"Full (strict)"**
   - Uses your Let's Encrypt certificate
   - Validates it
   - Maximum security

### 5.2 Enable Additional Security

1. Go to **"Edge Certificates"**
2. Enable:
   - ✅ **Always Use HTTPS**
   - ✅ **Automatic HTTPS Rewrites**
   - ✅ **Minimum TLS Version: 1.2**

### 5.3 Enable Security Headers

1. Go to **"Rules"** → **"Transform Rules"** → **"Modify Response Header"**
2. Click **"Create rule"**
3. Add headers:

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: SAMEORIGIN
X-XSS-Protection: 1; mode=block
```

---

## Step 6: DDoS Protection & Security

### 6.1 Enable DDoS Protection

1. Go to **"Security"** → **"DDoS"**
2. Sensitivity: **Medium** (balance protection vs. false positives)
3. Save

### 6.2 Enable WAF (Web Application Firewall)

1. Go to **"Security"** → **"WAF"**
2. Enable: **"OWASP ModSecurity Core Rule Set"**
3. Sensitivity: **Medium**
4. Save

### 6.3 Rate Limiting

1. Go to **"Security"** → **"Rate Limiting"**
2. Click **"Create rate limiting rule"**

**For API:**
- URL: `api.penux.uk/api/*`
- Threshold: `100` requests per `15` minutes
- Action: `Block`

**For Web:**
- URL: `ldap.penux.uk/*`
- Threshold: `500` requests per `15` minutes
- Action: `Block`

### 6.4 Bot Management (Optional - Paid)

For free tier, use:
1. Go to **"Bots"**
2. Enable: **"Super Bot Fight Mode"** (free)

---

## Step 7: Caching & Performance

### 7.1 Enable Caching

1. Go to **"Caching"** → **"Configuration"**
2. Cache Level: **"Standard"** (cache static resources)
3. Browser Cache TTL: **1 day**

### 7.2 Minify

1. Go to **"Speed"** → **"Optimization"**
2. Enable:
   - ✅ **Minify CSS**
   - ✅ **Minify JavaScript**
   - ✅ **Minify HTML**

### 7.3 Compression

1. Still in **"Speed"** → **"Optimization"**
2. Enable: **"Brotli"** (better compression than gzip)

---

## Step 8: DNS Only (Alternative - No Proxying)

If you want DNS only (no Cloudflare proxy):

1. In Cloudflare DNS, for your records, click the **cloud icon**
2. Change from **Proxied** (orange) to **DNS Only** (gray)

**Use case:** 
- Keep full control of traffic
- Still use Cloudflare DNS for reliability
- No DDoS protection
- Faster (direct connection)

---

## Step 9: Cloudflare Tunnel (Optional - Advanced)

### No Port Forwarding Needed!

Cloudflare Tunnel lets you expose your Docker services without opening ports.

### 9.1 Install Cloudflare Tunnel

```bash
# On your server
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
```

### 9.2 Authenticate

```bash
cloudflared tunnel login
```

Opens browser → approve → returns credentials

### 9.3 Create Tunnel

```bash
cloudflared tunnel create penux-ldap
```

Returns: `UUID` for your tunnel

### 9.4 Configure Tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: penux-ldap
credentials-file: /home/user/.cloudflared/<UUID>.json

ingress:
  - hostname: api.penux.uk
    service: http://localhost:3000
  - hostname: ldap.penux.uk
    service: http://localhost:3001
  - hostname: penux.uk
    service: http://localhost:3001
  - service: http_status:404
```

### 9.5 Route Tunnel in Cloudflare

```bash
# Create CNAME records automatically
cloudflared tunnel route dns penux-ldap api.penux.uk
cloudflared tunnel route dns penux-ldap ldap.penux.uk
cloudflared tunnel route dns penux-ldap penux.uk
```

### 9.6 Run Tunnel

```bash
# Foreground (testing)
cloudflared tunnel run penux-ldap

# Background (production)
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
```

**Benefits:**
- ✅ No port forwarding needed
- ✅ No open ports = more secure
- ✅ Cloudflare handles all traffic
- ✅ Built-in DDoS protection
- ✅ Works from anywhere (even home Internet)

---

## Step 10: Monitoring & Logs

### 10.1 Cloudflare Analytics

1. Go to **"Analytics & Logs"** → **"Analytics"**
2. View:
   - Traffic patterns
   - Blocked requests
   - Cache performance
   - Security events

### 10.2 Docker Logs

```bash
# Real-time logs
docker-compose -f docker-compose-hub.yml logs -f

# Specific service
docker-compose -f docker-compose-hub.yml logs -f api

# JSON format
docker-compose -f docker-compose-hub.yml logs --format '{{json .}}'
```

### 10.3 Cloudflare Logs

```bash
# Download logs via API
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/logs/http_requests" \
  | jq '.result'
```

---

## 🎯 Quick Reference

### Access Your Services

```
API:     https://api.penux.uk
Web UI:  https://ldap.penux.uk
Domain:  https://penux.uk
```

### Test API

```bash
# Health check
curl https://api.penux.uk/api/health

# With auth
curl -u "cn=admin,dc=penux,dc=uk:password" \
  https://api.penux.uk/api/users
```

### Useful Commands

```bash
# Check DNS propagation
dig api.penux.uk
nslookup api.penux.uk

# Test SSL
openssl s_client -connect api.penux.uk:443

# Check Nginx
sudo nginx -t
sudo systemctl status nginx

# View Cloudflare Tunnel status
cloudflared tunnel info penux-ldap
```

---

## 🔐 Security Checklist

- [ ] Cloudflare account created
- [ ] Domain added to Cloudflare
- [ ] Nameservers updated
- [ ] DNS records created (api, ldap)
- [ ] SSL certificates generated
- [ ] Nginx configured
- [ ] SSL/TLS: "Full (strict)"
- [ ] HTTPS redirect enabled
- [ ] DDoS protection enabled
- [ ] WAF enabled
- [ ] Rate limiting configured
- [ ] Security headers set
- [ ] API tested via HTTPS
- [ ] Web UI tested via HTTPS
- [ ] Logs checked (Cloudflare + Docker)

---

## 🐛 Troubleshooting

### DNS Not Resolving

```bash
# Check nameservers
whois penux.uk | grep -i nameserver

# Should show Cloudflare nameservers
# May take 5-30 minutes to propagate
```

### SSL Certificate Error

```bash
# Check certificate validity
openssl s_client -connect api.penux.uk:443

# Renew manually
sudo certbot renew --force-renewal

# Check Nginx config
sudo nginx -t
```

### Cloudflare Showing Error

1. Go to **"DNS"**
2. Ensure A records point to correct IP
3. Check if server is online: `ping your-server-ip`
4. Check Nginx: `sudo systemctl status nginx`

### 502 Bad Gateway Error

- Docker service not running: `docker ps`
- Nginx not running: `sudo systemctl restart nginx`
- Port conflict: `sudo netstat -tlnp | grep 3000`

---

## 💰 Costs

**Free Tier (Good for Development):**
- ✅ Free domain DNS
- ✅ Free DDoS protection
- ✅ Free WAF (basic)
- ✅ Free SSL/TLS
- ✅ Free Cloudflare Tunnel
- **Total: $0/month**

**Pro Tier ($20/month):**
- Advanced WAF
- 200+ rule customization
- Priority support

---

## ✨ Next Steps

After setup:

1. ✅ Test all endpoints
2. ✅ Monitor Cloudflare analytics
3. ✅ Review security rules
4. ✅ Set up backups
5. ✅ Create monitoring alerts
6. ✅ Document configuration
7. ✅ Train team on usage

---

## 📚 Resources

- **Cloudflare Dashboard:** https://dash.cloudflare.com
- **Cloudflare Docs:** https://developers.cloudflare.com
- **Let's Encrypt:** https://letsencrypt.org
- **Nginx Docs:** https://nginx.org/en/docs/
- **Docker Hub:** https://hub.docker.com

---

**You're ready to deploy with Cloudflare! 🚀**

Your secure, protected, and performant PenuX LDAP system is now live on:

```
🔒 https://api.penux.uk
🌐 https://ldap.penux.uk
```

With:
- ✅ SSL/TLS encryption
- ✅ DDoS protection
- ✅ WAF security
- ✅ Rate limiting
- ✅ Global CDN
- ✅ Monitoring & analytics
