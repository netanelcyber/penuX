# 🚀 Docker Hub + Cloudflare Tunnel (No Static IP Needed!)

Complete guide for deploying PenuX LDAP when you **don't have a static IP**.

**Perfect for:**
- Home internet connections
- ISPs that change your IP
- Shared hosting
- ISPs blocking port 80/443

---

## ✨ Why Cloudflare Tunnel?

**Traditional Method (With Static IP):**
```
Internet → Your IP → Nginx → Docker
                ❌ Requires fixed IP
                ❌ Must open ports
                ❌ Less secure
```

**Cloudflare Tunnel (No Static IP Needed):**
```
Internet → Cloudflare → Tunnel → Your Local Docker
            ✅ Works with dynamic IP
            ✅ No port forwarding
            ✅ No open ports = more secure
            ✅ Cloudflare DDoS protection
```

---

## ⚡ Quick Setup (10 minutes)

### Step 1: Install Cloudflare Tunnel CLI

```bash
# Download
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb

# Install
sudo dpkg -i cloudflared.deb

# Verify
cloudflared --version
```

### Step 2: Authenticate with Cloudflare

```bash
cloudflared tunnel login
```

**This will:**
1. Open browser to Cloudflare login
2. Ask permission to create tunnel
3. Return credentials automatically

### Step 3: Create Tunnel

```bash
cloudflared tunnel create penux-ldap
```

**Output shows:**
```
Tunnel credentials have been saved to:
~/.cloudflared/UUID.json

Tunnel 'penux-ldap' created with ID: UUID
```

Save this UUID - you'll need it.

### Step 4: Ensure Docker Services Running

```bash
# Start your Docker services
docker-compose -f docker-compose-hub.yml up -d

# Verify
docker ps
```

Should show:
- penux-openldap
- penux-postgres
- penux-api
- penux-web

### Step 5: Configure Tunnel Routing

**Option A: Quick Setup (Automatic DNS)**

```bash
# Create tunnel routes
cloudflared tunnel route dns penux-ldap api.penux.uk
cloudflared tunnel route dns penux-ldap ldap.penux.uk
cloudflared tunnel route dns penux-ldap penux.uk
```

This automatically creates CNAME records in Cloudflare.

**Option B: Manual DNS Setup**

1. Go to Cloudflare dashboard → **DNS**
2. Add CNAME records:
   ```
   api    CNAME    penux-ldap.cfargotunnel.com
   ldap   CNAME    penux-ldap.cfargotunnel.com
   penux  CNAME    penux-ldap.cfargotunnel.com
   ```

### Step 6: Create Configuration File

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: penux-ldap
credentials-file: /home/user/.cloudflared/UUID.json

ingress:
  - hostname: api.penux.uk
    service: http://localhost:3000
    originRequest:
      httpHostHeader: api.penux.uk
  
  - hostname: ldap.penux.uk
    service: http://localhost:3001
    originRequest:
      httpHostHeader: ldap.penux.uk
  
  - hostname: penux.uk
    service: http://localhost:3001
    originRequest:
      httpHostHeader: penux.uk
  
  - service: http_status:404
```

**Replace `UUID`** with your actual UUID from Step 3.

### Step 7: Test Tunnel Locally

```bash
# Run tunnel (shows logs)
cloudflared tunnel run penux-ldap
```

Should show:
```
INF Tunnel credentials loaded from /home/user/.cloudflared/UUID.json
INF Registering tunnel connection from LOCATION
INF Tunnel registered successfully
INF Serving http requests from http://127.0.0.1:3000
```

Test in another terminal:

```bash
# Test API
curl https://api.penux.uk/api/health

# Should return: {"status":"healthy"}
```

---

## 📋 Run Tunnel in Background

### Method 1: Systemd Service (Recommended)

```bash
# Install as service
sudo cloudflared service install

# Start service
sudo systemctl start cloudflared

# Enable auto-start
sudo systemctl enable cloudflared

# View status
sudo systemctl status cloudflared

# View logs
sudo journalctl -u cloudflared -f
```

### Method 2: Docker Container

Create `docker-compose-tunnel.yml`:

```yaml
version: '3.8'

services:
  cloudflared:
    image: cloudflare/cloudflared:latest
    command: tunnel run penux-ldap
    volumes:
      - /home/user/.cloudflared:/root/.cloudflared
    restart: unless-stopped
    environment:
      TUNNEL_ORIGIN_CERT: /root/.cloudflared/cert.pem
```

Run:

```bash
docker-compose -f docker-compose-tunnel.yml up -d
```

### Method 3: Systemd User Service

```bash
# Create directory
mkdir -p ~/.config/systemd/user

# Create service file
cat > ~/.config/systemd/user/cloudflared.service << 'EOF'
[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/cloudflared tunnel run penux-ldap
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=default.target
EOF

# Enable and start
systemctl --user daemon-reload
systemctl --user enable cloudflared
systemctl --user start cloudflared

# View status
systemctl --user status cloudflared
```

---

## 🔧 Configuration Deep Dive

### Add Multiple Services

```yaml
ingress:
  # API with custom headers
  - hostname: api.penux.uk
    service: http://localhost:3000
    originRequest:
      httpHostHeader: api.penux.uk
      headers:
        add:
          X-Custom-Header: "value"
  
  # Web UI
  - hostname: ldap.penux.uk
    service: http://localhost:3001
  
  # Root domain
  - hostname: penux.uk
    service: http://localhost:3001
  
  # Health check page
  - hostname: health.penux.uk
    service: http://localhost:3000
  
  # 404 fallback
  - service: http_status:404
```

### Advanced Options

```yaml
tunnel: penux-ldap
credentials-file: /home/user/.cloudflared/UUID.json

ingress:
  - hostname: api.penux.uk
    service: http://localhost:3000
    originRequest:
      # HTTP settings
      httpHostHeader: api.penux.uk
      noTLSVerify: false
      
      # Connection settings
      connectTimeout: 30s
      tlsTimeout: 10s
      
      # Headers
      headers:
        add:
          X-Forwarded-Proto: https
      
      # TCP keep-alive
      keepAliveConnections: 100
      keepAliveTimeout: 15s

  - service: http_status:404

logLevel: info
metrics: 0.0.0.0:54321
```

---

## ✅ Verify Tunnel Status

### Check Tunnel Health

```bash
# From command line
cloudflared tunnel info penux-ldap

# From Cloudflare dashboard:
# Go to "Access" → "Tunnels" → "penux-ldap"
# Should show: "HEALTHY" ✅
```

### View Real-Time Logs

```bash
# If running as service
sudo journalctl -u cloudflared -f

# If running in Docker
docker-compose -f docker-compose-tunnel.yml logs -f cloudflared
```

### Test Endpoints

```bash
# API health
curl https://api.penux.uk/api/health

# API with auth
curl -u "cn=admin,dc=penux,dc=uk:password" \
  https://api.penux.uk/api/users

# Web UI
curl -I https://ldap.penux.uk

# Root domain
curl https://penux.uk
```

---

## 🌐 Cloudflare Dashboard Setup

### Verify DNS Records

1. Go to **https://dash.cloudflare.com**
2. Select your domain
3. Go to **"DNS"**
4. You should see CNAME records:
   ```
   api   CNAME   penux-ldap.cfargotunnel.com
   ldap  CNAME   penux-ldap.cfargotunnel.com
   penux CNAME   penux-ldap.cfargotunnel.com
   ```

### Check Tunnel Status

1. Go to **"Access"** → **"Tunnels"**
2. Click **"penux-ldap"**
3. Status should show:
   - ✅ **HEALTHY**
   - 🟢 Connected
   - Uptime showing

### View Traffic Analytics

1. Go to **"Analytics & Logs"** → **"Analytics"**
2. View:
   - Requests per hour
   - Blocked traffic
   - Security events
   - Bandwidth usage

---

## 🔐 Security Features (Built-in)

### Cloudflare DDoS Protection

✅ **Enabled by default**
- Mitigates UDP, TCP, HTTP attacks
- No configuration needed

### Cloudflare WAF (Web Application Firewall)

1. Go to **"Security"** → **"WAF"**
2. Enable: **"OWASP ModSecurity Core Rule Set"**
3. Sensitivity: **Medium**

### Rate Limiting

1. Go to **"Security"** → **"Rate limiting"**
2. Create rules:

```
For API:
URL Path: api.penux.uk/api/*
Threshold: 100 requests per 15 minutes
Action: Challenge
```

### Additional Headers

1. Go to **"Rules"** → **"Transform Rules"** → **"Modify Response Header"**
2. Add security headers:

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
```

---

## 🚨 Handle IP Changes Automatically

Cloudflare Tunnel automatically handles:
- ✅ ISP IP changes
- ✅ Routing changes
- ✅ Connection drops
- ✅ Network outages

**How it works:**
1. Tunnel maintains persistent connection to Cloudflare
2. If connection drops, reconnects automatically
3. DNS continues resolving (cached)
4. No manual intervention needed

---

## 📊 Monitoring & Alerts

### View Tunnel Logs

```bash
# Real-time logs (systemd service)
sudo journalctl -u cloudflared -f

# Persistent logs (last 100 lines)
sudo journalctl -u cloudflared -n 100

# Filter by level
sudo journalctl -u cloudflared -p err
sudo journalctl -u cloudflared -p warning
```

### Docker Logs

```bash
# Tunnel container logs
docker-compose -f docker-compose-tunnel.yml logs -f cloudflared

# All services
docker-compose -f docker-compose-hub.yml logs -f
```

### Cloudflare API Monitoring

```bash
# View Tunnel details via API
curl -X GET \
  "https://api.cloudflare.com/client/v4/accounts/ACCOUNT_ID/cfd_tunnel" \
  -H "X-Auth-Email: your@email.com" \
  -H "X-Auth-Key: API_KEY"
```

---

## 🐛 Troubleshooting

### Tunnel Not Connecting

```bash
# 1. Check credentials file exists
ls -la ~/.cloudflared/

# 2. Check tunnel status
cloudflared tunnel info penux-ldap

# 3. Check service status (if running as service)
sudo systemctl status cloudflared

# 4. View detailed logs
sudo journalctl -u cloudflared -f --all
```

### DNS Not Resolving

```bash
# Check CNAME records
dig api.penux.uk
nslookup api.penux.uk

# Should resolve to cfargotunnel.com
```

### Getting 502 Error

1. **Docker service not running:**
   ```bash
   docker ps
   docker-compose -f docker-compose-hub.yml up -d
   ```

2. **Tunnel not running:**
   ```bash
   sudo systemctl status cloudflared
   sudo systemctl restart cloudflared
   ```

3. **Check tunnel logs:**
   ```bash
   sudo journalctl -u cloudflared -f
   ```

4. **Verify port is correct:**
   ```bash
   docker port penux-api
   docker port penux-web
   ```

### High Latency

```bash
# Check tunnel metrics
curl http://localhost:54321/metrics

# Reduce origin timeout in config
connectTimeout: 10s  # reduce from 30s
```

---

## 📈 Scaling & Performance

### Add More Connections

```yaml
# More concurrent connections
originRequest:
  keepAliveConnections: 200  # increase from 100
```

### Monitor Bandwidth Usage

1. Cloudflare dashboard → **Analytics** → **Bandwidth**
2. Check usage patterns
3. Optimize if needed

---

## 💰 Costs

**Free Tier:**
- ✅ Unlimited Cloudflare Tunnel connections
- ✅ Unlimited bandwidth through tunnel
- ✅ Free DDoS protection
- ✅ Free SSL/TLS
- **Total: $0/month**

---

## 🎯 Complete Workflow

```bash
# 1. Install Cloudflare CLI
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb

# 2. Authenticate
cloudflared tunnel login

# 3. Create tunnel
cloudflared tunnel create penux-ldap

# 4. Start Docker services
docker-compose -f docker-compose-hub.yml up -d

# 5. Create config
nano ~/.cloudflared/config.yml
# Paste config from Step 6 above

# 6. Create DNS routes
cloudflared tunnel route dns penux-ldap api.penux.uk
cloudflared tunnel route dns penux-ldap ldap.penux.uk
cloudflared tunnel route dns penux-ldap penux.uk

# 7. Run tunnel in background
sudo cloudflared service install
sudo systemctl start cloudflared
sudo systemctl enable cloudflared

# 8. Verify
curl https://api.penux.uk/api/health
```

---

## ✨ Next Steps

After successful setup:

1. ✅ Test all endpoints
2. ✅ Monitor Cloudflare analytics
3. ✅ Enable WAF & rate limiting
4. ✅ Set up monitoring alerts
5. ✅ Test with dynamic IP changes
6. ✅ Document for team
7. ✅ Train team on usage

---

## 📚 Resources

- **Cloudflare Tunnel Docs:** https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/install-and-setup/tunnel-guide/
- **Cloudflare Dashboard:** https://dash.cloudflare.com
- **GitHub Releases:** https://github.com/cloudflare/cloudflared/releases

---

**Perfect Setup for Dynamic IPs! 🚀**

Your PenuX LDAP system is now:
- ✅ Accessible via Cloudflare (no static IP needed)
- ✅ Protected by Cloudflare DDoS
- ✅ No open ports
- ✅ Automatic IP change handling
- ✅ Secure end-to-end

**Access:**
```
🔒 https://api.penux.uk
🌐 https://ldap.penux.uk
```

**No static IP? No problem!** 🎉
