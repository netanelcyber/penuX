# 🐳 Deploy PenuX LDAP via Docker Hub

Complete guide to build, push, and deploy using Docker Hub.

---

## ⚡ Quick Deploy (15 minutes)

### Option 1: Run Pre-built Images (Recommended)

If images are already on Docker Hub, just pull and run:

```bash
# Pull images
docker pull netanelcyber/penux-ldap-api:latest
docker pull netanelcyber/penux-ldap-web:latest

# Run with docker-compose
docker-compose -f docker-compose-public.yml up -d
```

---

### Option 2: Build & Push Your Own Images

#### Step 1: Set Up Docker Hub Account

1. Go to **https://hub.docker.com**
2. Click **"Sign Up"**
3. Create account (username, email, password)
4. Verify email
5. Log in

---

#### Step 2: Create Repositories on Docker Hub

1. Click **"Repositories"** (top menu)
2. Click **"Create repository"**

**Create first repository:**
- Name: `penux-ldap-api`
- Description: "PenuX LDAP REST API"
- Visibility: **Public** (so others can pull)
- Click **"Create"**

**Create second repository:**
- Name: `penux-ldap-web`
- Description: "PenuX LDAP Web UI"
- Visibility: **Public**
- Click **"Create"**

---

#### Step 3: Build Docker Images Locally

On your machine with Docker installed:

```bash
# Navigate to project root
cd ~/penuX

# Build API image
docker build -f services/openldap/api/Dockerfile \
  -t YOUR_DOCKERHUB_USERNAME/penux-ldap-api:latest \
  services/openldap/api/

# Build Web image
docker build -f services/openldap/web/Dockerfile \
  -t YOUR_DOCKERHUB_USERNAME/penux-ldap-web:latest \
  services/openldap/web/
```

**Replace `YOUR_DOCKERHUB_USERNAME`** with your actual Docker Hub username.

Example:
```bash
docker build -f services/openldap/api/Dockerfile \
  -t netanelcyber/penux-ldap-api:latest \
  services/openldap/api/
```

---

#### Step 4: Test Images Locally

```bash
# Test API image
docker run -p 3000:3000 \
  -e LDAP_HOST="ldaps://ldaps-server.penux.uk:636" \
  -e LDAP_BASE_DN="dc=penux,dc=uk" \
  -e LDAP_ADMIN_DN="cn=admin,dc=penux,dc=uk" \
  -e LDAP_ADMIN_PASSWORD="admin123" \
  YOUR_DOCKERHUB_USERNAME/penux-ldap-api:latest

# In another terminal, test:
curl http://localhost:3000/api/health

# Should show: {"status":"healthy"}
```

Stop with: `Ctrl+C`

---

#### Step 5: Push to Docker Hub

```bash
# Login to Docker Hub (first time only)
docker login

# When prompted:
# Username: YOUR_DOCKERHUB_USERNAME
# Password: YOUR_PASSWORD

# Push API image
docker push YOUR_DOCKERHUB_USERNAME/penux-ldap-api:latest

# Push Web image
docker push YOUR_DOCKERHUB_USERNAME/penux-ldap-web:latest
```

**Verify on Docker Hub:**
1. Go to https://hub.docker.com/repositories
2. You should see both repositories
3. Click each to view details and tags

---

## 📦 Deployment Methods

### Method 1: Docker Compose (Local/Server)

Create `docker-compose-hub.yml`:

```yaml
version: '3.8'

services:
  openldap:
    image: osixia/openldap:1.5.0
    environment:
      LDAP_ORGANISATION: "PenuX"
      LDAP_DOMAIN: "penux.uk"
      LDAP_BASE_DN: "dc=penux,dc=uk"
      LDAP_ADMIN_DN: "cn=admin,dc=penux,dc=uk"
      LDAP_ADMIN_PASSWORD: "admin123"
    ports:
      - "389:389"
      - "636:636"
    volumes:
      - ldap_data:/var/lib/ldap
      - ldap_config:/etc/ldap/slapd.d

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: penux
      POSTGRES_USER: penux
      POSTGRES_PASSWORD: secure_password_here
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  api:
    image: YOUR_DOCKERHUB_USERNAME/penux-ldap-api:latest
    environment:
      LDAP_HOST: "ldap://openldap:389"
      LDAP_BASE_DN: "dc=penux,dc=uk"
      LDAP_ADMIN_DN: "cn=admin,dc=penux,dc=uk"
      LDAP_ADMIN_PASSWORD: "admin123"
      DATABASE_URL: "postgresql://penux:secure_password_here@postgres:5432/penux"
      CORS_ORIGIN: "*"
      NODE_ENV: "production"
    ports:
      - "3000:3000"
    depends_on:
      - openldap
      - postgres

  web:
    image: YOUR_DOCKERHUB_USERNAME/penux-ldap-web:latest
    environment:
      API_URL: "http://api:3000"
      NODE_ENV: "production"
    ports:
      - "3001:3001"
    depends_on:
      - api

volumes:
  ldap_data:
  ldap_config:
  postgres_data:
```

**Run:**

```bash
# Replace YOUR_DOCKERHUB_USERNAME with actual username
docker-compose -f docker-compose-hub.yml up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

### Method 2: Docker Run (Individual Containers)

```bash
# Create network
docker network create penux-net

# Run OpenLDAP
docker run -d \
  --name openldap \
  --network penux-net \
  -e LDAP_ORGANISATION="PenuX" \
  -e LDAP_DOMAIN="penux.uk" \
  -e LDAP_BASE_DN="dc=penux,dc=uk" \
  -e LDAP_ADMIN_DN="cn=admin,dc=penux,dc=uk" \
  -e LDAP_ADMIN_PASSWORD="admin123" \
  -p 389:389 \
  -p 636:636 \
  osixia/openldap:1.5.0

# Run PostgreSQL
docker run -d \
  --name postgres \
  --network penux-net \
  -e POSTGRES_DB="penux" \
  -e POSTGRES_USER="penux" \
  -e POSTGRES_PASSWORD="secure_password" \
  -p 5432:5432 \
  postgres:15-alpine

# Run API
docker run -d \
  --name penux-api \
  --network penux-net \
  -e LDAP_HOST="ldap://openldap:389" \
  -e LDAP_BASE_DN="dc=penux,dc=uk" \
  -e LDAP_ADMIN_DN="cn=admin,dc=penux,dc=uk" \
  -e LDAP_ADMIN_PASSWORD="admin123" \
  -e DATABASE_URL="postgresql://penux:secure_password@postgres:5432/penux" \
  -e CORS_ORIGIN="*" \
  -p 3000:3000 \
  YOUR_DOCKERHUB_USERNAME/penux-ldap-api:latest

# Run Web UI
docker run -d \
  --name penux-web \
  --network penux-net \
  -e API_URL="http://penux-api:3000" \
  -p 3001:3001 \
  YOUR_DOCKERHUB_USERNAME/penux-ldap-web:latest
```

---

### Method 3: On Remote Server (Production)

**On your server (AWS EC2, DigitalOcean, Linode, etc.):**

```bash
# SSH into server
ssh user@your-server.com

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (optional, for sudo-less docker)
sudo usermod -aG docker $USER
newgrp docker

# Create docker-compose.yml (paste content from Method 1 above)
nano docker-compose-hub.yml

# Replace YOUR_DOCKERHUB_USERNAME with actual username

# Start services
docker-compose -f docker-compose-hub.yml up -d

# View logs
docker-compose logs -f

# Test API
curl http://localhost:3000/api/health
```

---

## ✅ Verify Deployment

### Check Running Containers

```bash
docker ps
```

Output should show:
- `openldap`
- `postgres`
- `penux-api`
- `penux-web`

### Test API Health

```bash
curl http://localhost:3000/api/health
```

Should return: `{"status":"healthy"}`

### Test with Authentication

```bash
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  http://localhost:3000/api/users
```

### Access Web UI

Open browser: `http://localhost:3001`

---

## 🌐 Expose to Internet

### Using Nginx Reverse Proxy

```bash
# Install Nginx
sudo apt-get update
sudo apt-get install nginx -y

# Create Nginx config
sudo nano /etc/nginx/sites-available/penux

# Paste:
```

```nginx
upstream api {
    server localhost:3000;
}

upstream web {
    server localhost:3001;
}

server {
    listen 80;
    server_name api.ldap.penux.uk ldap.penux.uk;

    location / {
        if ($host = api.ldap.penux.uk) {
            proxy_pass http://api;
        }
        if ($host = ldap.penux.uk) {
            proxy_pass http://web;
        }
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
# Enable config
sudo ln -s /etc/nginx/sites-available/penux /etc/nginx/sites-enabled/

# Test
sudo nginx -t

# Restart Nginx
sudo systemctl restart nginx
```

---

### Add HTTPS with Let's Encrypt

```bash
# Install Certbot
sudo apt-get install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d api.ldap.penux.uk -d ldap.penux.uk

# Auto-renewal (enabled by default)
sudo systemctl enable certbot.timer
```

---

## 📊 Container Management

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api
```

### Update Image

```bash
# Pull latest
docker pull YOUR_DOCKERHUB_USERNAME/penux-ldap-api:latest

# Restart
docker-compose restart api
```

### Stop & Start

```bash
# Stop all
docker-compose stop

# Start all
docker-compose start

# Restart
docker-compose restart

# Remove (careful!)
docker-compose down
```

---

## 🔐 Security Best Practices

### 1. Use Environment Files

Create `.env` file (never commit):

```bash
LDAP_ADMIN_PASSWORD=YourSecurePassword123!
POSTGRES_PASSWORD=AnotherSecurePassword456!
CORS_ORIGIN=https://yourdomain.com
```

Use in compose:

```yaml
environment:
  LDAP_ADMIN_PASSWORD: ${LDAP_ADMIN_PASSWORD}
```

Run:

```bash
docker-compose --env-file .env up -d
```

### 2. Use Volumes for Persistence

```yaml
volumes:
  ldap_data:
  postgres_data:
```

Data persists even if containers stop.

### 3. Network Isolation

Use bridge network:

```yaml
networks:
  penux-net:
    driver: bridge
```

Containers only accessible internally unless ports exposed.

### 4. Resource Limits

```yaml
services:
  api:
    resources:
      limits:
        cpus: '1'
        memory: 512M
      reservations:
        cpus: '0.5'
        memory: 256M
```

### 5. Health Checks

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 📈 Scaling

### Horizontal Scaling (Multiple API Containers)

```yaml
services:
  api:
    image: YOUR_DOCKERHUB_USERNAME/penux-ldap-api:latest
    deploy:
      replicas: 3
```

Use with load balancer (HAProxy, Traefik, etc.)

### Vertical Scaling (More Resources)

```yaml
services:
  api:
    resources:
      limits:
        memory: 2G
        cpus: '2'
```

---

## 🐛 Troubleshooting

### Container Won't Start

```bash
docker-compose logs api
# Look for error messages
```

### Connection Refused

```bash
# Check if container is running
docker ps

# Check network
docker network inspect penux-net

# Verify ports
docker port penux-api
```

### Out of Memory

```bash
# Check resource usage
docker stats

# Increase memory limit in docker-compose.yml
# Restart: docker-compose up -d
```

### Database Connection Failed

```bash
# Check if postgres is running
docker ps | grep postgres

# Check connection string in API config
docker-compose logs api | grep DATABASE
```

---

## 🌍 Push Updates to Docker Hub

After code changes:

```bash
# Rebuild image
docker build -f services/openldap/api/Dockerfile \
  -t YOUR_DOCKERHUB_USERNAME/penux-ldap-api:v2.0 \
  services/openldap/api/

# Tag as latest
docker tag YOUR_DOCKERHUB_USERNAME/penux-ldap-api:v2.0 \
  YOUR_DOCKERHUB_USERNAME/penux-ldap-api:latest

# Push both tags
docker push YOUR_DOCKERHUB_USERNAME/penux-ldap-api:v2.0
docker push YOUR_DOCKERHUB_USERNAME/penux-ldap-api:latest

# Update docker-compose.yml with new tag
# Restart: docker-compose up -d
```

---

## 💰 Costs

**Self-Hosted (VPS):**
- DigitalOcean: $5-60/month
- Linode: $5-80/month
- AWS EC2: $5-100+/month
- **Docker Hub:** Free (public images)

**Managed (easier but pricier):**
- Docker Swarm: Included
- Kubernetes (managed): $50-500+/month

---

## 📚 Useful Commands

```bash
# Build image
docker build -t name:tag .

# Push to Hub
docker push name:tag

# Pull from Hub
docker pull name:tag

# Run container
docker run -d --name my-container image:tag

# View running containers
docker ps

# View all containers
docker ps -a

# View logs
docker logs -f container-name

# Execute command in container
docker exec -it container-name bash

# Stop container
docker stop container-name

# Remove container
docker rm container-name

# Remove image
docker rmi image:tag

# Docker Compose up
docker-compose up -d

# Docker Compose down
docker-compose down

# Docker Compose logs
docker-compose logs -f
```

---

## ✨ What's Next?

After deployment:

1. ✅ Test API: `curl http://localhost:3000/api/health`
2. ✅ Access Web UI: `http://localhost:3001`
3. ✅ Create LDAP users
4. ✅ Set up Nginx reverse proxy (for public access)
5. ✅ Add HTTPS with Let's Encrypt
6. ✅ Monitor container logs
7. ✅ Set up backups (docker volumes)
8. ✅ Scale for production

---

## 🎯 Quick Checklist

**Setup:**
- [ ] Docker Hub account created
- [ ] Repositories created (api, web)
- [ ] Local Docker installed
- [ ] Images built locally
- [ ] Images tested
- [ ] Images pushed to Docker Hub

**Deployment:**
- [ ] docker-compose.yml created
- [ ] Environment variables set
- [ ] Containers started
- [ ] API responds to health check
- [ ] Web UI accessible
- [ ] Logs checked (no errors)

**Production:**
- [ ] Server SSH access ready
- [ ] Docker installed on server
- [ ] Services running in background
- [ ] Nginx reverse proxy configured
- [ ] HTTPS enabled with Let's Encrypt
- [ ] Backups scheduled

---

**You're ready to deploy! 🚀**

Start with:
```bash
docker-compose -f docker-compose-hub.yml up -d
```

For detailed info: `/DEPLOYMENT-ALTERNATIVES.md`
