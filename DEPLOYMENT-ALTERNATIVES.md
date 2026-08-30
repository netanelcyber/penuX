# 🚀 PenuX LDAP - Alternative Deployment Options

Deploy PenuX LDAP on various cloud platforms and hosting providers

---

## Table of Contents

1. [Heroku Deployment](#heroku-deployment)
2. [Railway Deployment](#railway-deployment)
3. [Docker Hub & Container Registry](#docker-hub--container-registry)
4. [AWS Deployment](#aws-deployment)
5. [Google Cloud Deployment](#google-cloud-deployment)
6. [DigitalOcean Deployment](#digitalocean-deployment)
7. [Self-Hosted (VPS)](#self-hosted-vps)
8. [Kubernetes Deployment](#kubernetes-deployment)

---

## Heroku Deployment

### Prerequisites

```bash
npm install -g heroku
heroku login
```

### Step 1: Create Heroku Apps

```bash
# Create apps for each service
heroku create penux-ldap-api
heroku create penux-ldap-web
heroku create penux-keycloak

# Or use existing apps
heroku apps
```

### Step 2: Deploy API

```bash
cd services/openldap/api

# Create Heroku Procfile
cat > Procfile << EOF
web: node server.js
EOF

# Set environment variables
heroku config:set \
  LDAP_HOST=ldaps://ldaps-server.penux.uk:636 \
  LDAP_BASE_DN=dc=penux,dc=uk \
  LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk \
  LDAP_ADMIN_PASSWORD=your_secure_password \
  CORS_ORIGIN=https://ldap.penux.uk

# Deploy
git add Procfile
git commit -m "Add Heroku Procfile"
git push heroku main

# View logs
heroku logs --tail
```

### Step 3: Deploy Web UI

```bash
cd services/openldap/web

# Create Procfile
cat > Procfile << EOF
web: node server.js
EOF

# Set environment variables
heroku config:set \
  API_URL=https://penux-ldap-api.herokuapp.com \
  PORT=5000

# Deploy
git push heroku main
```

### Step 4: Configure Custom Domain

```bash
# Add custom domain to Heroku apps
heroku domains:add api.ldap.penux.uk --app penux-ldap-api
heroku domains:add ldap.penux.uk --app penux-ldap-web

# Add CNAME records to DNS:
# api.ldap.penux.uk -> penux-ldap-api.herokuapp.com
# ldap.penux.uk -> penux-ldap-web.herokuapp.com
```

### Step 5: Monitor & Scale

```bash
# View logs
heroku logs -t

# Scale dynos
heroku ps:scale web=2

# View metrics
heroku metrics

# Access app
heroku open
```

---

## Railway Deployment

### Prerequisites

```bash
npm install -g @railway/cli
railway login
```

### Step 1: Create Railway Project

```bash
railway init

# Select: Node.js
# Select: Yes, deploy from current directory
```

### Step 2: Configure Services

```bash
# Create railway.json in root
cat > railway.json << EOF
{
  "services": {
    "api": {
      "root": "services/openldap/api",
      "buildCommand": "npm install",
      "startCommand": "npm start"
    },
    "web": {
      "root": "services/openldap/web",
      "buildCommand": "npm install",
      "startCommand": "npm start"
    }
  }
}
EOF
```

### Step 3: Set Environment Variables

```bash
# Via CLI
railway variables add LDAP_HOST=ldaps://ldaps-server.penux.uk:636
railway variables add LDAP_BASE_DN=dc=penux,dc=uk
railway variables add LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk
railway variables add LDAP_ADMIN_PASSWORD=your_secure_password

# Or via Dashboard
# https://railway.app/dashboard
```

### Step 4: Deploy

```bash
railway up

# View deployment
railway status

# View logs
railway logs
```

### Step 5: Custom Domain

```bash
# Via Railway Dashboard
# Project Settings → Domains
# Add domain: api.ldap.penux.uk
# Add CNAME to DNS pointing to Railway domain
```

---

## Docker Hub & Container Registry

### Build and Push Docker Images

```bash
# Build OpenLDAP image
docker build -t myusername/penux-ldap:latest ./services/openldap/openldap

# Build API image
docker build -t myusername/penux-ldap-api:latest ./services/openldap/api

# Build Web UI image
docker build -t myusername/penux-ldap-web:latest ./services/openldap/web

# Push to Docker Hub
docker push myusername/penux-ldap:latest
docker push myusername/penux-ldap-api:latest
docker push myusername/penux-ldap-web:latest

# Verify images
docker run myusername/penux-ldap:latest
docker run myusername/penux-ldap-api:latest
docker run myusername/penux-ldap-web:latest
```

### Create Dockerfile for API

```dockerfile
# services/openldap/api/Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY server.js .

EXPOSE 3000

ENV NODE_ENV=production
ENV PORT=3000

CMD ["node", "server.js"]
```

### Create Dockerfile for Web UI

```dockerfile
# services/openldap/web/Dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3001

ENV NODE_ENV=production
ENV PORT=3001

CMD ["node", "server.js"]
```

### Push to GitHub Container Registry (GHCR)

```bash
# Login to GHCR
echo $PAT | docker login ghcr.io -u username --password-stdin

# Tag images
docker tag myusername/penux-ldap:latest ghcr.io/username/penux-ldap:latest
docker tag myusername/penux-ldap-api:latest ghcr.io/username/penux-ldap-api:latest

# Push
docker push ghcr.io/username/penux-ldap:latest
docker push ghcr.io/username/penux-ldap-api:latest
```

---

## AWS Deployment

### Option 1: ECS (Elastic Container Service)

```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name penux-ldap

# Create task definition
aws ecs register-task-definition \
  --family penux-ldap-api \
  --container-definitions file://task-definition.json

# Create service
aws ecs create-service \
  --cluster penux-ldap \
  --service-name penux-ldap-api \
  --task-definition penux-ldap-api \
  --desired-count 2
```

### Option 2: Elastic Beanstalk

```bash
# Initialize Elastic Beanstalk
eb init -p "Node.js 18 running on 64bit Amazon Linux 2" penux-ldap

# Create environment
eb create penux-ldap-prod

# Deploy
eb deploy

# View logs
eb logs
```

### Option 3: Lambda (Serverless)

```bash
# Install serverless framework
npm install -g serverless

# Create serverless service
serverless create --template aws-nodejs

# Configure serverless.yml
# Deploy
serverless deploy

# Invoke function
serverless invoke -f api
```

---

## Google Cloud Deployment

### Cloud Run (Serverless)

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/penux-ldap-api

# Deploy to Cloud Run
gcloud run deploy penux-ldap-api \
  --image gcr.io/PROJECT_ID/penux-ldap-api \
  --platform managed \
  --region us-central1 \
  --set-env-vars LDAP_HOST=ldaps://ldaps-server.penux.uk:636

# Get URL
gcloud run services describe penux-ldap-api --platform managed --region us-central1
```

### App Engine

```bash
# Create app.yaml
cat > services/openldap/api/app.yaml << EOF
runtime: nodejs18

env: standard

env_variables:
  LDAP_HOST: "ldaps://ldaps-server.penux.uk:636"
  LDAP_BASE_DN: "dc=penux,dc=uk"
  NODE_ENV: "production"
EOF

# Deploy
gcloud app deploy services/openldap/api/app.yaml

# View logs
gcloud app logs read
```

### Compute Engine (VPS)

```bash
# Create instance
gcloud compute instances create penux-ldap \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --machine-type=e2-medium \
  --zone=us-central1-a

# SSH into instance
gcloud compute ssh penux-ldap --zone us-central1-a

# Install and run
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
docker compose up -d
```

---

## DigitalOcean Deployment

### App Platform

```bash
# Create app from GitHub
# 1. Connect GitHub repository
# 2. Select branch to deploy
# 3. Configure environment variables:
#    - LDAP_HOST=ldaps://ldaps-server.penux.uk:636
#    - LDAP_BASE_DN=dc=penux,dc=uk
# 4. Click Deploy

# Via CLI
doctl apps create --spec app.yaml
```

### Droplet (VPS)

```bash
# Create droplet
doctl compute droplet create penux-ldap \
  --region nyc3 \
  --image ubuntu-22-04-x64 \
  --size s-1vcpu-1gb \
  --enable-monitoring

# SSH in
ssh root@<droplet_ip>

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Deploy services
docker compose up -d
```

### App Platform (YAML)

```yaml
# app.yaml
name: penux-ldap
services:
- name: api
  github:
    repo: netanelcyber/penuX
    branch: main
    deploy_on_push: true
  build_command: cd services/openldap/api && npm install
  run_command: cd services/openldap/api && npm start
  http_port: 3000
  envs:
  - key: LDAP_HOST
    value: ldaps://ldaps-server.penux.uk:636
  - key: LDAP_BASE_DN
    value: dc=penux,dc=uk
  - key: LDAP_ADMIN_DN
    value: cn=admin,dc=penux,dc=uk
  - key: LDAP_ADMIN_PASSWORD
    scope: RUN_AND_BUILD_TIME
    value: ${LDAP_ADMIN_PASSWORD}
```

---

## Self-Hosted (VPS)

### Ubuntu 22.04 Setup

```bash
#!/bin/bash
# setup.sh

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Nginx
sudo apt-get install -y nginx

# Install Let's Encrypt
sudo apt-get install -y certbot python3-certbot-nginx

# Clone repository
git clone https://github.com/netanelcyber/penuX.git
cd penuX

# Deploy
docker compose -f docker-compose-public.yml up -d

# Configure Nginx reverse proxy
sudo nano /etc/nginx/sites-available/penux.uk

# Enable site
sudo ln -s /etc/nginx/sites-available/penux.uk /etc/nginx/sites-enabled/

# Get SSL certificate
sudo certbot --nginx -d ldap.penux.uk -d api.ldap.penux.uk

# Restart Nginx
sudo systemctl restart nginx
```

### Nginx Reverse Proxy Configuration

```nginx
# /etc/nginx/sites-available/penux.uk

upstream ldap_web {
    server localhost:3001;
}

upstream ldap_api {
    server localhost:3000;
}

upstream keycloak {
    server localhost:8080;
}

server {
    listen 80;
    server_name ldap.penux.uk www.ldap.penux.uk;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ldap.penux.uk www.ldap.penux.uk;

    ssl_certificate /etc/letsencrypt/live/ldap.penux.uk/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ldap.penux.uk/privkey.pem;

    location / {
        proxy_pass http://ldap_web;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 443 ssl http2;
    server_name api.ldap.penux.uk;

    ssl_certificate /etc/letsencrypt/live/ldap.penux.uk/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ldap.penux.uk/privkey.pem;

    location / {
        proxy_pass http://ldap_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 443 ssl http2;
    server_name keycloak.penux.uk;

    ssl_certificate /etc/letsencrypt/live/ldap.penux.uk/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ldap.penux.uk/privkey.pem;

    location / {
        proxy_pass http://keycloak;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Kubernetes Deployment

### Create Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: penux-ldap
```

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ldap-config
  namespace: penux-ldap
data:
  LDAP_BASE_DN: "dc=penux,dc=uk"
  LDAP_ADMIN_DN: "cn=admin,dc=penux,dc=uk"
```

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ldap-secret
  namespace: penux-ldap
type: Opaque
stringData:
  LDAP_ADMIN_PASSWORD: "your_secure_password"
```

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ldap-api
  namespace: penux-ldap
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ldap-api
  template:
    metadata:
      labels:
        app: ldap-api
    spec:
      containers:
      - name: api
        image: myregistry.azurecr.io/penux-ldap-api:latest
        ports:
        - containerPort: 3000
        env:
        - name: LDAP_HOST
          value: "ldaps://ldap-service:636"
        - name: LDAP_BASE_DN
          valueFrom:
            configMapKeyRef:
              name: ldap-config
              key: LDAP_BASE_DN
        - name: LDAP_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: ldap-secret
              key: LDAP_ADMIN_PASSWORD
        livenessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 3000
          initialDelaySeconds: 5
          periodSeconds: 5
```

```yaml
# k8s/api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ldap-api-service
  namespace: penux-ldap
spec:
  selector:
    app: ldap-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 3000
  type: LoadBalancer
```

### Deploy to Kubernetes

```bash
# Create namespace and apply configs
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml

# Check deployment
kubectl get deployments -n penux-ldap
kubectl get services -n penux-ldap
kubectl get pods -n penux-ldap

# View logs
kubectl logs -n penux-ldap deployment/ldap-api

# Scale deployment
kubectl scale deployment ldap-api --replicas=3 -n penux-ldap

# Port forward for testing
kubectl port-forward -n penux-ldap svc/ldap-api-service 3000:80
```

---

## Comparison Matrix

| Platform | Cost | Ease | Scalability | Support |
|----------|------|------|-------------|---------|
| Heroku | $$$ | ⭐⭐⭐ | ⭐⭐ | Good |
| Railway | $$ | ⭐⭐⭐ | ⭐⭐⭐ | Good |
| AWS | $$ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Excellent |
| Google Cloud | $$ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Excellent |
| DigitalOcean | $ | ⭐⭐⭐ | ⭐⭐⭐ | Good |
| Self-Hosted | $ | ⭐ | ⭐⭐ | None |
| Kubernetes | $$ | ⭐ | ⭐⭐⭐⭐⭐ | Community |

---

## Choosing Your Platform

### Best for Beginners
- **Railway**: Simple, fast, great documentation
- **Heroku**: Familiar to many, straightforward deployment

### Best for Production
- **AWS**: Enterprise-grade, most flexible
- **Kubernetes**: Scalable, reliable, industry standard

### Best Budget
- **DigitalOcean Droplet**: Cheap VPS
- **Self-Hosted**: Minimum cost (just pay for server)

### Best for Teams
- **Kubernetes**: Easy management at scale
- **AWS/GCP**: Full enterprise features

---

**Last Updated**: 2026-06-21
