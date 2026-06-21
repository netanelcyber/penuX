# 🔐 PenuX LDAP Security Guide

Complete security configuration and hardening for LDAP deployment

---

## Table of Contents

1. [Authentication & Authorization](#authentication--authorization)
2. [Network Security](#network-security)
3. [LDAP Server Hardening](#ldap-server-hardening)
4. [API Security](#api-security)
5. [Password Policy](#password-policy)
6. [Audit & Monitoring](#audit--monitoring)
7. [Incident Response](#incident-response)
8. [Security Checklist](#security-checklist)

---

## Authentication & Authorization

### 1. Change Default Credentials

**CRITICAL**: Change default admin password immediately after deployment.

```bash
# Change OpenLDAP admin password
ldappasswd -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w "admin123" \
  -s "NewSecurePassword123!" \
  "cn=admin,dc=penux,dc=uk"

# Verify change
ldapwhoami -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w "NewSecurePassword123!"
```

### 2. Implement Password Policy

#### LDAP Password Requirements

Edit `/etc/ldap/slapd.conf` or use LDAP ACLs:

```ldif
# Add to OpenLDAP configuration
dn: cn=config
changetype: modify
add: olcPasswordHash
olcPasswordHash: {CRYPT}
-
add: olcPasswordModule
olcPasswordModule: {0}pw-sha2

# Password policy overlay
dn: cn=module{0},cn=config
changetype: add
objectClass: olcModuleList
cn: module{0}
olcModulePath: /usr/lib/ldap
olcModuleLoad: ppolicy.la

# Apply password policy
dn: cn=ppolicy,ou=policies,dc=penux,dc=uk
changetype: add
objectClass: device
objectClass: pwdPolicy
cn: ppolicy
pwdMaxAge: 7776000
pwdInHistory: 5
pwdMinLength: 12
pwdMaxFailure: 5
pwdLockout: TRUE
pwdLockoutDuration: 600
pwdMustChange: TRUE
pwdAllowUserChange: TRUE
```

#### Password Requirements

- **Minimum Length**: 12 characters
- **Complexity**: At least:
  - 1 uppercase letter (A-Z)
  - 1 lowercase letter (a-z)
  - 1 number (0-9)
  - 1 special character (!@#$%^&*)
- **History**: Cannot reuse last 5 passwords
- **Expiration**: 90 days
- **Account Lockout**: 5 failed attempts → 10 minute lockout

### 3. Implement MFA (Multi-Factor Authentication)

Use Keycloak for MFA enforcement:

1. Login to Keycloak admin console
2. Navigate to: Realm Settings → Security Defenses
3. Enable: Brute Force Detection, OTP Required Actions
4. Configure: TOTP (Google Authenticator), FIDO2

```bash
# Access Keycloak
https://keycloak.penux.uk/admin
Username: admin
Password: admin123
```

---

## Network Security

### 1. Firewall Rules

#### UFW (Ubuntu Firewall)

```bash
# Allow SSH (management)
sudo ufw allow 22/tcp

# Allow LDAP (internal only)
sudo ufw allow from 192.168.0.0/16 to any port 389

# Allow LDAPS (encrypted)
sudo ufw allow from any to any port 636

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Deny everything else
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Enable firewall
sudo ufw enable
```

#### iptables Rules

```bash
# Restrict LDAP to internal network
sudo iptables -A INPUT -p tcp --dport 389 -s 192.168.0.0/16 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 389 -j DROP

# Allow LDAPS globally
sudo iptables -A INPUT -p tcp --dport 636 -j ACCEPT

# Allow HTTPS globally
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT

# Save rules
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

### 2. Network Isolation

```yaml
# docker-compose.yml network isolation
networks:
  internal:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  openldap:
    networks:
      - internal
    expose:
      - 389
      - 636
    # Do NOT expose to 0.0.0.0 if internal-only

  keycloak:
    networks:
      - internal
    expose:
      - 8080
```

### 3. SSL/TLS Configuration

#### LDAPS (Encrypted LDAP)

```bash
# Generate self-signed certificate (if not already done)
openssl req -x509 -nodes -days 365 -newkey rsa:4096 \
  -keyout /etc/ldap/certs/ldap.key \
  -out /etc/ldap/certs/ldap.crt \
  -subj "/CN=penux.uk/O=PenuX/C=US"

# Set permissions
sudo chown -R openldap:openldap /etc/ldap/certs
sudo chmod 600 /etc/ldap/certs/ldap.key
sudo chmod 644 /etc/ldap/certs/ldap.crt

# Update slapd.conf
olcTLSCertificateFile: /etc/ldap/certs/ldap.crt
olcTLSCertificateKeyFile: /etc/ldap/certs/ldap.key
olcTLSCipherSuite: HIGH:!aNULL:!MD5
olcTLSProtocolMin: 3.3  # TLS 1.2 minimum
```

#### HTTPS for Web Services

```bash
# Use Cloudflare certificates or let's encrypt
# Via Cloudflare Tunnel (recommended)
# via Let's Encrypt (if self-hosted)

sudo certbot certonly --standalone \
  -d ldap.penux.uk \
  -d api.ldap.penux.uk \
  -d keycloak.penux.uk
```

---

## LDAP Server Hardening

### 1. Access Control Lists (ACLs)

```ldif
# File: /etc/ldap/acl.conf
# Restrict user access

# 1. Admin can do anything
dn.base="cn=admin,dc=penux,dc=uk" read,write,search,compare,auth

# 2. Users can read their own entries
dn.subtree="ou=users,dc=penux,dc=uk" 
  selfwrite,read,search,compare,auth

# 3. Groups readable by authenticated users
dn.subtree="ou=groups,dc=penux,dc=uk"
  read,search,compare

# 4. Deny anonymous access to sensitive attributes
dn.subtree="dc=penux,dc=uk"
  attrs=userPassword
  access to by self write by * none

# 5. Default deny
dn.subtree="dc=penux,dc=uk"
  access to by * none
```

### 2. Disable Anonymous Bind

Edit `/etc/ldap/slapd.conf`:

```conf
# Disable anonymous binds
disallow bind_anon

# Require authentication for most operations
require bind
```

### 3. Rate Limiting

```bash
# Use fail2ban to protect against brute force
sudo apt-get install fail2ban

# Create /etc/fail2ban/jail.local
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[slapd]
enabled = true
port = ldap,ldaps
logpath = /var/log/syslog
maxretry = 5
findtime = 600
bantime = 1800
```

### 4. Audit Logging

```bash
# Enable OpenLDAP audit logging
ldapmodify -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 << EOF
dn: cn=config
changetype: modify
add: olcLogLevel
olcLogLevel: stats
EOF

# Check logs
tail -f /var/log/syslog | grep slapd
```

---

## API Security

### 1. Authentication Methods

#### Basic Authentication (Default)

```bash
# Encode credentials
echo -n "cn=admin,dc=penux,dc=uk:password" | base64
# cnHhYWRtaW4sZGM9cGVudXgsZGM9dWs6cGFzc3dvcmQ=

# Use in requests
curl -H "Authorization: Basic cnHhYWRtaW4sZGM9cGVudXgsZGM9dWs6cGFzc3dvcmQ=" \
  https://api.ldap.penux.uk/api/users
```

#### API Key Authentication (Enhancement)

```javascript
// Add to server.js
const VALID_API_KEYS = {
  'key_production_abc123xyz': { name: 'Production', level: 'read' },
  'key_testing_def456uvw': { name: 'Testing', level: 'write' }
};

function authenticateApiKey(req, res, next) {
  const apiKey = req.headers['x-api-key'];
  
  if (!apiKey || !VALID_API_KEYS[apiKey]) {
    return res.status(401).json({
      success: false,
      error: 'Invalid or missing API key'
    });
  }
  
  req.apiKey = VALID_API_KEYS[apiKey];
  next();
}

// Use: app.use(authenticateApiKey);
```

### 2. CORS Security

```javascript
// Strict CORS configuration
const corsOptions = {
  origin: [
    'https://ldap.penux.uk',
    'https://api.ldap.penux.uk'
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  maxAge: 3600
};

app.use(cors(corsOptions));
```

### 3. Rate Limiting

```javascript
// Enhanced rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  keyGenerator: (req) => {
    return req.user?.dn || req.ip;
  },
  handler: (req, res) => {
    res.status(429).json({
      success: false,
      error: 'Too many requests. Please try again later.'
    });
  },
  skip: (req) => {
    // Skip rate limiting for health checks
    return req.path === '/api/health';
  }
});

app.use(limiter);
```

### 4. Input Validation

```javascript
// Validate and sanitize inputs
const validator = require('express-validator');

app.get('/api/users/:uid', [
  param('uid').isAlphanumeric().isLength({ min: 1, max: 255 })
], (req, res) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({ errors: errors.array() });
  }
  // Process request
});
```

### 5. LDAP Injection Prevention

Already implemented in server.js:

```javascript
function escapeFilter(str) {
  const metaChars = ['*', '(', ')', '\0'];
  let escaped = '';
  for (let i = 0; i < str.length; i++) {
    if (metaChars.includes(str[i])) {
      escaped += '\\' + str[i];
    } else {
      escaped += str[i];
    }
  }
  return escaped;
}
```

---

## Password Policy

### Implementation

```bash
#!/bin/bash
# enforce-password-policy.sh

# Requirements
MIN_LENGTH=12
REQUIRE_UPPER=true
REQUIRE_LOWER=true
REQUIRE_DIGIT=true
REQUIRE_SPECIAL=true
MAX_AGE_DAYS=90
PASSWORD_HISTORY=5

validate_password() {
    local password=$1
    
    # Check length
    if [ ${#password} -lt $MIN_LENGTH ]; then
        return 1
    fi
    
    # Check uppercase
    if $REQUIRE_UPPER && ! echo "$password" | grep -q '[A-Z]'; then
        return 1
    fi
    
    # Check lowercase
    if $REQUIRE_LOWER && ! echo "$password" | grep -q '[a-z]'; then
        return 1
    fi
    
    # Check digit
    if $REQUIRE_DIGIT && ! echo "$password" | grep -q '[0-9]'; then
        return 1
    fi
    
    # Check special
    if $REQUIRE_SPECIAL && ! echo "$password" | grep -q '[@#$%^&*]'; then
        return 1
    fi
    
    return 0
}

# Usage
validate_password "NewPass123!@" && echo "Valid" || echo "Invalid"
```

### User Self-Service Password Change

```bash
# Allow users to change their own password
ldappasswd -H ldap://localhost:389 \
  -D "uid=jdoe,ou=users,dc=penux,dc=uk" \
  -w "oldpassword" \
  -s "NewSecurePassword123!" \
  "uid=jdoe,ou=users,dc=penux,dc=uk"
```

---

## Audit & Monitoring

### 1. Access Logging

```bash
# Enable audit logging in slapd
echo "olcLogLevel: 128" >> /etc/ldap/slapd.conf

# Monitor in real-time
tail -f /var/log/syslog | grep slapd
```

### 2. Monitor Failed Logins

```bash
# Check for failed authentication attempts
grep "BIND" /var/log/syslog | grep -i "fail"

# Count attempts per user
grep "BIND" /var/log/syslog | awk '{print $NF}' | sort | uniq -c
```

### 3. Alerting

```bash
#!/bin/bash
# alert-security-events.sh

# Monitor for suspicious activity
while true; do
  FAILED_BINDS=$(grep -c "BIND.*failed" /var/log/syslog.1 2>/dev/null || echo 0)
  
  if [ $FAILED_BINDS -gt 10 ]; then
    echo "ALERT: Multiple failed bind attempts detected!"
    # Send email or webhook notification
  fi
  
  sleep 60
done
```

### 4. Prometheus Metrics

```javascript
// Add to server.js for monitoring
const prometheus = require('prom-client');

const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'endpoint', 'status']
});

const ldapConnectionErrors = new prometheus.Counter({
  name: 'ldap_connection_errors_total',
  help: 'Total number of LDAP connection errors',
  labelNames: ['error_type']
});

app.get('/metrics', (req, res) => {
  res.set('Content-Type', prometheus.register.contentType);
  res.end(prometheus.register.metrics());
});
```

---

## Incident Response

### 1. Account Lockout

```bash
# Unlock a user account
ldapmodify -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 << EOF
dn: uid=jdoe,ou=users,dc=penux,dc=uk
changetype: modify
delete: pwdAccountLockedTime
EOF
```

### 2. Password Reset

```bash
# Admin-initiated password reset
ldappasswd -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 \
  -s "TemporaryPassword123!" \
  "uid=jdoe,ou=users,dc=penux,dc=uk"

# Notify user to change password on next login
```

### 3. Revoke Access

```bash
# Disable user account
ldapmodify -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 << EOF
dn: uid=jdoe,ou=users,dc=penux,dc=uk
changetype: modify
add: accountStatus
accountStatus: disabled
EOF

# Remove from all groups
ldapmodify -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w admin123 << EOF
dn: cn=developers,ou=groups,dc=penux,dc=uk
changetype: modify
delete: member
member: uid=jdoe,ou=users,dc=penux,dc=uk
EOF
```

### 4. Change Management

Document all changes:

```bash
# Log all administrative actions
echo "$(date): Changed password for uid=jdoe" >> /var/log/ldap-admin.log
echo "$(date): Disabled account uid=jsmith" >> /var/log/ldap-admin.log
```

---

## Security Checklist

### Pre-Deployment

- [ ] Change all default credentials
- [ ] Generate SSL/TLS certificates
- [ ] Configure firewall rules
- [ ] Set up LDAP ACLs
- [ ] Enable audit logging
- [ ] Configure password policy
- [ ] Set up fail2ban/rate limiting
- [ ] Review CORS settings
- [ ] Configure API authentication
- [ ] Set up monitoring/alerting

### Post-Deployment

- [ ] Verify LDAPS is working
- [ ] Test firewall rules
- [ ] Verify password policy enforced
- [ ] Check audit logs are being generated
- [ ] Test account lockout mechanism
- [ ] Verify MFA on Keycloak (if enabled)
- [ ] Document all administrative passwords (encrypted)
- [ ] Test backup and recovery procedures
- [ ] Schedule security audits (quarterly)
- [ ] Plan incident response drills

### Ongoing

- [ ] Monitor failed login attempts (daily)
- [ ] Review audit logs (weekly)
- [ ] Test password policy (monthly)
- [ ] Update certificates before expiration
- [ ] Patch OS and dependencies (as needed)
- [ ] Review user access (quarterly)
- [ ] Conduct security training (annually)
- [ ] Perform penetration testing (annually)

---

## Resources

- [OWASP LDAP Injection](https://owasp.org/www-community/attacks/LDAP_Injection)
- [RFC 4876: LDAP Access Control Model](https://tools.ietf.org/html/rfc4876)
- [OpenLDAP Admin Guide](https://www.openldap.org/doc/admin/)
- [NIST Password Guidelines](https://pages.nist.gov/800-63-3/sp800-63b.html)

---

**Last Updated**: 2026-06-21
**Version**: 1.0.0
