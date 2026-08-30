# 📚 PenuX LDAP REST API - Complete Usage Guide

Comprehensive guide for using the PenuX LDAP REST API

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Examples](#requestresponse-examples)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [SDKs & Libraries](#sdks--libraries)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Get API URL

```
Production: https://api.ldap.penux.uk
Development: http://localhost:3000
```

### 2. Prepare Credentials

```bash
# Your LDAP DN
DN="cn=admin,dc=penux,dc=uk"
PASSWORD="admin123"

# Encode to Base64
ENCODED=$(echo -n "$DN:$PASSWORD" | base64)
echo $ENCODED
# Output: Y24=YWRtaW4sZGM9cGVudXgsZGM9dWs6YWRtaW4xMjM=
```

### 3. Test Connection

```bash
curl -H "Authorization: Basic $ENCODED" \
  https://api.ldap.penux.uk/api/health
```

---

## Authentication

### HTTP Basic Authentication

All endpoints (except `/api/health` and `/api/verify`) require HTTP Basic Authentication.

```bash
# Method 1: Using curl -u flag
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  https://api.ldap.penux.uk/api/users

# Method 2: Using Authorization header
curl -H "Authorization: Basic Y24hYWRtaW4sZGM9cGVudXgsZGM9dWs6YWRtaW4xMjM=" \
  https://api.ldap.penux.uk/api/users

# Method 3: Using JavaScript
fetch('https://api.ldap.penux.uk/api/users', {
  headers: {
    'Authorization': 'Basic ' + btoa('cn=admin,dc=penux,dc=uk:admin123')
  }
})
```

### Authorization Header Format

```
Authorization: Basic <base64(DN:password)>
```

### Generate Base64 Credentials

```bash
# Bash
echo -n "cn=admin,dc=penux,dc=uk:admin123" | base64

# Node.js
Buffer.from("cn=admin,dc=penux,dc=uk:admin123").toString('base64')

# Python
import base64
base64.b64encode(b"cn=admin,dc=penux,dc=uk:admin123").decode()

# JavaScript
btoa("cn=admin,dc=penux,dc=uk:admin123")
```

---

## API Endpoints

### System

#### Health Check

```
GET /api/health
```

Check if the API is running. No authentication required.

**Response:**
```json
{
  "status": "ok",
  "service": "PenuX LDAP API",
  "timestamp": "2026-06-21T12:34:56.789Z"
}
```

---

### Users

#### List All Users

```
GET /api/users
```

Retrieve all LDAP users. Requires authentication.

**Response:**
```json
{
  "success": true,
  "count": 5,
  "users": [
    {
      "dn": "uid=jdoe,ou=users,dc=penux,dc=uk",
      "uid": "jdoe",
      "cn": "John Doe",
      "givenName": "John",
      "sn": "Doe",
      "mail": "john@example.com",
      "title": "Software Engineer",
      "department": "Engineering",
      "accountStatus": "active"
    }
  ]
}
```

#### Get Specific User

```
GET /api/users/{uid}
```

Get information about a specific user by UID.

**Example:**
```bash
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  https://api.ldap.penux.uk/api/users/jdoe
```

**Response:**
```json
{
  "success": true,
  "user": {
    "dn": "uid=jdoe,ou=users,dc=penux,dc=uk",
    "uid": "jdoe",
    "cn": "John Doe",
    "givenName": "John",
    "sn": "Doe",
    "mail": "john@example.com",
    "title": "Software Engineer",
    "department": "Engineering",
    "phone": "+1-555-123-4567",
    "accountStatus": "active"
  }
}
```

---

### Groups

#### List All Groups

```
GET /api/groups
```

Retrieve all LDAP groups.

**Response:**
```json
{
  "success": true,
  "count": 3,
  "groups": [
    {
      "dn": "cn=developers,ou=groups,dc=penux,dc=uk",
      "cn": "developers",
      "description": "Development team",
      "members": [
        "uid=jdoe,ou=users,dc=penux,dc=uk",
        "uid=jsmith,ou=users,dc=penux,dc=uk"
      ]
    }
  ]
}
```

#### Get Specific Group

```
GET /api/groups/{cn}
```

Get information about a specific group.

**Example:**
```bash
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  https://api.ldap.penux.uk/api/groups/developers
```

**Response:**
```json
{
  "success": true,
  "group": {
    "dn": "cn=developers,ou=groups,dc=penux,dc=uk",
    "cn": "developers",
    "description": "Development team",
    "members": [
      "uid=jdoe,ou=users,dc=penux,dc=uk",
      "uid=jsmith,ou=users,dc=penux,dc=uk"
    ],
    "memberCount": 2
  }
}
```

---

### Search

#### Search Directory

```
GET /api/search?query={query}
```

Search for users and groups by various attributes.

**Parameters:**
- `query` (required): Search term (searches cn, uid, mail)

**Examples:**
```bash
# Search by name
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  "https://api.ldap.penux.uk/api/search?query=john"

# Search by email
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  "https://api.ldap.penux.uk/api/search?query=john@example.com"

# Search by UID
curl -u "cn=admin,dc=penux,dc=uk:admin123" \
  "https://api.ldap.penux.uk/api/search?query=jdoe"
```

**Response:**
```json
{
  "success": true,
  "count": 2,
  "results": [
    {
      "dn": "uid=jdoe,ou=users,dc=penux,dc=uk",
      "uid": "jdoe",
      "cn": "John Doe",
      "mail": "john@example.com"
    },
    {
      "dn": "cn=developers,ou=groups,dc=penux,dc=uk",
      "cn": "developers",
      "description": "Development team"
    }
  ]
}
```

---

### Authentication

#### Verify Credentials

```
POST /api/verify
```

Verify that a DN and password combination is valid.

**Request Body:**
```json
{
  "dn": "cn=admin,dc=penux,dc=uk",
  "password": "admin123"
}
```

**Example:**
```bash
curl -X POST https://api.ldap.penux.uk/api/verify \
  -H "Content-Type: application/json" \
  -d '{
    "dn": "cn=admin,dc=penux,dc=uk",
    "password": "admin123"
  }'
```

**Success Response (200):**
```json
{
  "success": true,
  "message": "Credentials verified"
}
```

**Failure Response (401):**
```json
{
  "success": false,
  "error": "Invalid credentials"
}
```

---

### Organization

#### List Organizational Units

```
GET /api/ous
```

Retrieve all organizational units (OUs) in the directory.

**Response:**
```json
{
  "success": true,
  "count": 2,
  "ous": [
    {
      "dn": "ou=users,dc=penux,dc=uk",
      "ou": "users",
      "description": "User accounts"
    },
    {
      "dn": "ou=groups,dc=penux,dc=uk",
      "ou": "groups",
      "description": "Security groups"
    }
  ]
}
```

---

### Statistics

#### Get Directory Statistics

```
GET /api/stats
```

Retrieve statistics about the LDAP directory.

**Response:**
```json
{
  "success": true,
  "stats": {
    "totalUsers": 42,
    "totalGroups": 8,
    "baseDN": "dc=penux,dc=uk",
    "timestamp": "2026-06-21T12:34:56.789Z"
  }
}
```

---

## Request/Response Examples

### JavaScript/Node.js

```javascript
const API_BASE = 'https://api.ldap.penux.uk';

async function getUsers() {
  const response = await fetch(`${API_BASE}/api/users`, {
    headers: {
      'Authorization': 'Basic ' + btoa('cn=admin,dc=penux,dc=uk:admin123')
    }
  });
  
  const data = await response.json();
  console.log(data.users);
}

async function getUser(uid) {
  const response = await fetch(`${API_BASE}/api/users/${uid}`, {
    headers: {
      'Authorization': 'Basic ' + btoa('cn=admin,dc=penux,dc=uk:admin123')
    }
  });
  
  const data = await response.json();
  if (data.success) {
    console.log(data.user);
  } else {
    console.error(data.error);
  }
}

async function verifyCredentials(dn, password) {
  const response = await fetch(`${API_BASE}/api/verify`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ dn, password })
  });
  
  const data = await response.json();
  return data.success;
}

// Usage
getUsers();
getUser('jdoe');
verifyCredentials('cn=admin,dc=penux,dc=uk', 'admin123');
```

### Python

```python
import requests
import base64
from typing import Dict, List

class LDAPAPI:
    def __init__(self, base_url: str, dn: str, password: str):
        self.base_url = base_url
        self.dn = dn
        self.password = password
        self.headers = {
            'Authorization': f'Basic {self._get_auth()}',
            'Content-Type': 'application/json'
        }
    
    def _get_auth(self) -> str:
        """Generate Basic auth header value"""
        credentials = f'{self.dn}:{self.password}'
        return base64.b64encode(credentials.encode()).decode()
    
    def get_users(self) -> List[Dict]:
        """Get all users"""
        response = requests.get(
            f'{self.base_url}/api/users',
            headers=self.headers
        )
        return response.json()['users']
    
    def get_user(self, uid: str) -> Dict:
        """Get specific user"""
        response = requests.get(
            f'{self.base_url}/api/users/{uid}',
            headers=self.headers
        )
        return response.json()['user']
    
    def get_groups(self) -> List[Dict]:
        """Get all groups"""
        response = requests.get(
            f'{self.base_url}/api/groups',
            headers=self.headers
        )
        return response.json()['groups']
    
    def search(self, query: str) -> List[Dict]:
        """Search directory"""
        response = requests.get(
            f'{self.base_url}/api/search',
            params={'query': query},
            headers=self.headers
        )
        return response.json()['results']
    
    def verify_credentials(self, dn: str, password: str) -> bool:
        """Verify user credentials"""
        response = requests.post(
            f'{self.base_url}/api/verify',
            json={'dn': dn, 'password': password},
            headers={'Content-Type': 'application/json'}
        )
        return response.json()['success']

# Usage
api = LDAPAPI(
    'https://api.ldap.penux.uk',
    'cn=admin,dc=penux,dc=uk',
    'admin123'
)

users = api.get_users()
admin = api.get_user('admin')
results = api.search('john')
```

### cURL Examples

```bash
#!/bin/bash

API_URL="https://api.ldap.penux.uk"
DN="cn=admin,dc=penux,dc=uk"
PASSWORD="admin123"

# Health check (no auth)
curl "$API_URL/api/health"

# List users
curl -u "$DN:$PASSWORD" "$API_URL/api/users"

# Get specific user
curl -u "$DN:$PASSWORD" "$API_URL/api/users/jdoe"

# Search
curl -u "$DN:$PASSWORD" "$API_URL/api/search?query=john"

# Verify credentials
curl -X POST "$API_URL/api/verify" \
  -H "Content-Type: application/json" \
  -d "{\"dn\":\"$DN\",\"password\":\"$PASSWORD\"}"

# Get statistics
curl -u "$DN:$PASSWORD" "$API_URL/api/stats"

# Format output with jq
curl -s -u "$DN:$PASSWORD" "$API_URL/api/users" | jq '.users[] | {uid, cn, mail}'
```

---

## Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

### Common Errors

#### 400 - Bad Request

Missing or invalid parameters.

```json
{
  "success": false,
  "error": "Query parameter required"
}
```

#### 401 - Unauthorized

Authentication failed or missing.

```json
{
  "success": false,
  "error": "Authentication required"
}
```

#### 404 - Not Found

Resource not found.

```json
{
  "success": false,
  "error": "User not found"
}
```

#### 429 - Too Many Requests

Rate limit exceeded.

```json
{
  "success": false,
  "error": "Too many requests from this IP, please try again later."
}
```

#### 500 - Internal Server Error

Server-side error.

```json
{
  "success": false,
  "error": "Internal server error"
}
```

### Error Handling Examples

```javascript
// JavaScript
async function handleErrors() {
  try {
    const response = await fetch('https://api.ldap.penux.uk/api/users', {
      headers: {
        'Authorization': 'Basic ' + btoa('cn=admin,dc=penux,dc=uk:admin123')
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    
    if (!data.success) {
      throw new Error(data.error);
    }
    
    console.log(data.users);
  } catch (error) {
    console.error('API Error:', error.message);
  }
}
```

```python
# Python
import requests

try:
    response = requests.get(
        'https://api.ldap.penux.uk/api/users',
        headers={
            'Authorization': f'Basic {base64_creds}',
        },
        timeout=5
    )
    
    response.raise_for_status()  # Raise for HTTP errors
    
    data = response.json()
    
    if not data.get('success'):
        raise Exception(data.get('error'))
    
    print(data['users'])
    
except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

---

## Rate Limiting

### Limits

- **100 requests per 15 minutes** per IP address
- `/api/health` endpoint is exempt from rate limiting

### Rate Limit Headers

Response includes headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1624276200
```

### Handling Rate Limits

```javascript
// Retry with exponential backoff
async function apiCallWithRetry(url, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    const response = await fetch(url, {
      headers: {
        'Authorization': 'Basic ' + btoa('cn=admin,dc=penux,dc=uk:admin123')
      }
    });
    
    if (response.status === 429) {
      const retryAfter = response.headers.get('Retry-After') || Math.pow(2, i);
      console.log(`Rate limited. Waiting ${retryAfter}s...`);
      await new Promise(r => setTimeout(r, retryAfter * 1000));
      continue;
    }
    
    return response.json();
  }
  
  throw new Error('Max retries exceeded');
}
```

---

## SDKs & Libraries

### JavaScript/Node.js

```bash
npm install ldapjs
npm install node-fetch  # If using Node < 18
```

### Python

```bash
pip install requests
pip install ldap3
```

### Go

```bash
go get ldap.v3
```

### Ruby

```bash
gem install net-ldap
```

---

## Troubleshooting

### Issue: 401 Unauthorized

**Symptoms:** Getting 401 errors even with credentials

**Solutions:**
1. Verify DN format: `uid=username,ou=users,dc=penux,dc=uk`
2. Check password is correct
3. Ensure Base64 encoding is correct:
   ```bash
   echo -n "dn:password" | base64
   ```
4. Verify user exists in LDAP directory

### Issue: 404 User Not Found

**Symptoms:** User exists but endpoint returns 404

**Solutions:**
1. Check exact UID (case-sensitive)
2. Verify user is in `ou=users` organizational unit
3. Test with search endpoint first:
   ```bash
   curl -u "cn=admin,dc=penux,dc=uk:admin123" \
     "https://api.ldap.penux.uk/api/search?query=jdoe"
   ```

### Issue: 429 Rate Limit

**Symptoms:** Getting rate limit errors

**Solutions:**
1. Implement backoff/retry logic
2. Batch requests when possible
3. Cache results locally
4. Contact admin for rate limit increase if needed

### Issue: Connection Timeout

**Symptoms:** Requests timing out

**Solutions:**
1. Check if API server is running: `https://api.ldap.penux.uk/api/health`
2. Verify LDAP server connectivity
3. Check firewall/network rules
4. Increase timeout: `curl --max-time 30`

### Debug Mode

Enable verbose logging:

```bash
# curl
curl -v -u "cn=admin,dc=penux,dc=uk:admin123" \
  https://api.ldap.penux.uk/api/users

# Node.js
DEBUG=* node app.js

# Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

**Last Updated**: 2026-06-21
**API Version**: 1.0.0
