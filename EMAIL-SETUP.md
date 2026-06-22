# 📧 Email Configuration

PenuX LDAP now includes **email-enabled users** and an **SMTP mail server** for testing and development.

---

## 🎯 What's Included

### 1. MailHog Mail Server
A lightweight, development-friendly mail server that:
- ✅ Accepts SMTP connections (no real email sending)
- ✅ Stores all emails in memory for inspection
- ✅ Provides a web UI to view captured emails
- ✅ Zero configuration required

### 2. Email-Enabled LDAP Users
Pre-populated with sample users that have email addresses:

| Name | Email | Role | Password |
|------|-------|------|----------|
| John Smith | john.smith@penux.uk | Engineering Manager | password123 |
| Alice Johnson | alice.johnson@penux.uk | Senior Developer | password123 |
| Bob Wilson | bob.wilson@penux.uk | QA Engineer | password123 |
| Admin | admin | Administrator | admin123 |

### 3. LDAP Email Schema
All users support standard email attributes:
- `mail` — Email address
- `uid` — Username
- `cn` — Common name
- `givenName` — First name
- `sn` — Last name
- `displayName` — Display name
- `telephoneNumber` — Phone (optional)

---

## 🔧 Using the Mail Server

### Access MailHog Web UI

**Local development:**
```bash
http://localhost:8025
```

**Via Docker Compose:**
MailHog runs automatically as the `mailhog` service.

### Send Test Email via SMTP

From any application, configure SMTP:
```
Host: mailhog
Port: 1025
Auth: None required
TLS: Not needed
```

Example using `curl`:
```bash
# Send email to an LDAP user
curl --url "smtp://mailhog:1025" \
  --mail-from "system@penux.uk" \
  --mail-rcpt "john.smith@penux.uk" \
  --data "From: System <system@penux.uk>
To: John Smith <john.smith@penux.uk>
Subject: Test Email

Hello John!"
```

---

## 🔍 Querying Email Attributes in LDAP

### Via ldapsearch

```bash
# Find users with email addresses
ldapsearch -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w "admin123" \
  -b "ou=users,dc=penux,dc=uk" \
  "objectClass=inetOrgPerson"
```

### Via REST API

```bash
# Search users by email
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  "https://tunnel-url/api/search?filter=mail=*"

# Get specific user
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  "https://tunnel-url/api/users?filter=uid=john.smith"
```

---

## ✉️ Using Email in Applications

### 1. Authenticate with LDAP User Email

```bash
# Login as alice.johnson using LDAP credentials
curl -u "cn=alice.johnson,ou=users,dc=penux,dc=uk:password123" \
  "https://tunnel-url/api/users"
```

### 2. Retrieve User's Email Address

```bash
# Query user and get email
ldapsearch -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w "admin123" \
  -b "cn=alice.johnson,ou=users,dc=penux,dc=uk" \
  "mail"

# Output: mail: alice.johnson@penux.uk
```

### 3. Send Emails to LDAP Users

```bash
# Get all users with email
ldapsearch -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w "admin123" \
  -b "ou=users,dc=penux,dc=uk" \
  "(objectClass=inetOrgPerson)" \
  mail uid

# Use results to send emails via MailHog SMTP
```

---

## 📝 Adding New Email Users

### Via LDAP (ldapadd)

```ldif
dn: cn=david.miller,ou=users,dc=penux,dc=uk
objectClass: inetOrgPerson
cn: david.miller
sn: Miller
givenName: David
mail: david.miller@penux.uk
uid: david.miller
userPassword: password123
displayName: David Miller
```

```bash
# Add the user
ldapadd -H ldap://localhost:389 \
  -D "cn=admin,dc=penux,dc=uk" \
  -w "admin123" \
  -f new-user.ldif
```

### Via REST API (if supported by your implementation)

```bash
curl -X POST "https://tunnel-url/api/users" \
  -u "cn=admin,dc=penux,dc=uk:admin123" \
  -H "Content-Type: application/json" \
  -d '{
    "uid": "david.miller",
    "cn": "David Miller",
    "mail": "david.miller@penux.uk",
    "password": "password123"
  }'
```

---

## 🔐 Security Notes

### Development Only
- MailHog stores emails **in-memory** (no persistence)
- Emails are **not encrypted** during transmission
- Use **only in development** environments

### For Production
1. Use a real SMTP server (SendGrid, AWS SES, etc.)
2. Enable SMTP authentication
3. Use TLS/SSL encryption
4. Hash LDAP user passwords (passwords currently in plaintext)
5. Audit email access logs

---

## 🚀 Example: Email Notifications

### Send notification when user logs in

1. **Update API** to trigger email on login:
```javascript
const nodemailer = require('nodemailer');

// Configure to use MailHog
const transporter = nodemailer.createTransport({
  host: 'mailhog',
  port: 1025,
  secure: false
});

// On successful login
await transporter.sendMail({
  from: 'auth@penux.uk',
  to: user.mail,
  subject: 'Login Notification',
  text: `Hello ${user.givenName}, you logged in at ${new Date().toISOString()}`
});
```

2. **Monitor emails** in MailHog UI: http://localhost:8025

---

## 📞 Support

For issues with:
- **LDAP users** — check OpenLDAP logs in Docker
- **Email service** — verify MailHog is running (`docker ps | grep mailhog`)
- **SMTP connection** — use MailHog web UI to debug

---

**Email integration is now ready. Test with sample users or add your own!**

🎉 All email sent to MailHog is captured and viewable in the MailHog web interface.
