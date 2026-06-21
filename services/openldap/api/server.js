// PenuX LDAP API Server
// Deploy to: Vercel, Netlify, Heroku, Railway, or any Node.js host
//
// Environment variables:
// LDAP_HOST=ldap://your-server:389
// LDAP_BASE_DN=dc=penux,dc=uk
// LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk
// LDAP_ADMIN_PASSWORD=password
// CORS_ORIGIN=*

const express = require('express');
const ldap = require('ldapjs');
const cors = require('cors');
const rateLimit = require('express-rate-limit');

const app = express();
const port = process.env.PORT || 3000;

// Configuration
const LDAP_HOST = process.env.LDAP_HOST || 'ldap://localhost:389';
const LDAP_BASE_DN = process.env.LDAP_BASE_DN || 'dc=penux,dc=uk';
const LDAP_ADMIN_DN = process.env.LDAP_ADMIN_DN || 'cn=admin,dc=penux,dc=uk';
const LDAP_ADMIN_PASSWORD = process.env.LDAP_ADMIN_PASSWORD || 'admin123';
const CORS_ORIGIN = process.env.CORS_ORIGIN || '*';

// Middleware
app.use(express.json());
app.use(cors({ origin: CORS_ORIGIN }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.'
});
app.use(limiter);

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'PenuX LDAP API',
    timestamp: new Date().toISOString()
  });
});

// Get all users
app.get('/api/users', authenticateRequest, async (req, res) => {
  try {
    const users = await searchLDAP(
      'ou=users,' + LDAP_BASE_DN,
      '(objectClass=inetOrgPerson)'
    );

    res.json({
      success: true,
      count: users.length,
      users: users.map(user => ({
        dn: user.dn,
        cn: user.cn,
        uid: user.uid,
        email: user.mail,
        mail: user.mail,
        title: user.title,
        department: user.department,
        accountStatus: user.accountStatus,
        givenName: user.givenName,
        sn: user.sn
      }))
    });
  } catch (error) {
    console.error('Error loading users:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get all groups
app.get('/api/groups', authenticateRequest, async (req, res) => {
  try {
    const groups = await searchLDAP(
      'ou=groups,' + LDAP_BASE_DN,
      '(objectClass=groupOfNames)'
    );

    res.json({
      success: true,
      count: groups.length,
      groups: groups.map(group => ({
        dn: group.dn,
        cn: group.cn,
        description: group.description,
        members: group.member || []
      }))
    });
  } catch (error) {
    console.error('Error loading groups:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get single user
app.get('/api/users/:uid', authenticateRequest, async (req, res) => {
  try {
    const users = await searchLDAP(
      'ou=users,' + LDAP_BASE_DN,
      `(uid=${escapeFilter(req.params.uid)})`
    );

    if (users.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'User not found'
      });
    }

    const user = users[0];
    res.json({
      success: true,
      user: {
        dn: user.dn,
        cn: user.cn,
        uid: user.uid,
        email: user.mail,
        mail: user.mail,
        title: user.title,
        department: user.department,
        accountStatus: user.accountStatus,
        givenName: user.givenName,
        sn: user.sn,
        phone: user.telephoneNumber
      }
    });
  } catch (error) {
    console.error('Error loading user:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get single group
app.get('/api/groups/:cn', authenticateRequest, async (req, res) => {
  try {
    const groups = await searchLDAP(
      'ou=groups,' + LDAP_BASE_DN,
      `(cn=${escapeFilter(req.params.cn)})`
    );

    if (groups.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Group not found'
      });
    }

    const group = groups[0];
    res.json({
      success: true,
      group: {
        dn: group.dn,
        cn: group.cn,
        description: group.description,
        members: group.member || [],
        memberCount: (group.member || []).length
      }
    });
  } catch (error) {
    console.error('Error loading group:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Search users
app.get('/api/search', authenticateRequest, async (req, res) => {
  const { query } = req.query;

  if (!query) {
    return res.status(400).json({
      success: false,
      error: 'Query parameter required'
    });
  }

  try {
    const results = await searchLDAP(
      LDAP_BASE_DN,
      `(|(cn=*${escapeFilter(query)}*)(uid=*${escapeFilter(query)}*)(mail=*${escapeFilter(query)}*))`
    );

    res.json({
      success: true,
      count: results.length,
      results
    });
  } catch (error) {
    console.error('Error searching LDAP:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Verify user credentials
app.post('/api/verify', async (req, res) => {
  const { dn, password } = req.body;

  if (!dn || !password) {
    return res.status(400).json({
      success: false,
      error: 'DN and password required'
    });
  }

  try {
    const client = ldap.createClient({ url: LDAP_HOST });

    await new Promise((resolve, reject) => {
      client.bind(dn, password, (err) => {
        client.unbind((unbindErr) => {
          if (err) reject(err);
          else if (unbindErr) reject(unbindErr);
          else resolve();
        });
      });
    });

    res.json({
      success: true,
      message: 'Credentials verified'
    });
  } catch (error) {
    res.status(401).json({
      success: false,
      error: 'Invalid credentials'
    });
  }
});

// List organization units
app.get('/api/ous', authenticateRequest, async (req, res) => {
  try {
    const ous = await searchLDAP(
      LDAP_BASE_DN,
      '(objectClass=organizationalUnit)'
    );

    res.json({
      success: true,
      count: ous.length,
      ous: ous.map(ou => ({
        dn: ou.dn,
        ou: ou.ou,
        description: ou.description
      }))
    });
  } catch (error) {
    console.error('Error loading OUs:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Statistics
app.get('/api/stats', authenticateRequest, async (req, res) => {
  try {
    const users = await searchLDAP('ou=users,' + LDAP_BASE_DN, '(objectClass=inetOrgPerson)');
    const groups = await searchLDAP('ou=groups,' + LDAP_BASE_DN, '(objectClass=groupOfNames)');

    res.json({
      success: true,
      stats: {
        totalUsers: users.length,
        totalGroups: groups.length,
        baseDN: LDAP_BASE_DN,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error loading stats:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Helper function: Search LDAP
async function searchLDAP(base, filter) {
  return new Promise((resolve, reject) => {
    const client = ldap.createClient({ url: LDAP_HOST });
    const results = [];

    client.bind(LDAP_ADMIN_DN, LDAP_ADMIN_PASSWORD, (err) => {
      if (err) {
        client.destroy();
        return reject(err);
      }

      const searchOptions = { filter, scope: 'sub' };

      client.search(base, searchOptions, (err, res) => {
        if (err) {
          client.destroy();
          return reject(err);
        }

        res.on('searchEntry', (entry) => {
          const attrs = {};
          entry.attributes.forEach(attr => {
            if (attr.vals.length === 1) {
              attrs[attr.type] = attr.vals[0];
            } else {
              attrs[attr.type] = attr.vals;
            }
          });
          results.push(attrs);
        });

        res.on('error', (err) => {
          client.destroy();
          reject(err);
        });

        res.on('end', (result) => {
          client.unbind(() => {
            if (result.status === 0) {
              resolve(results);
            } else {
              reject(new Error('LDAP search failed'));
            }
          });
        });
      });
    });
  });
}

// Helper function: Escape LDAP filter
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

// Authentication middleware
function authenticateRequest(req, res, next) {
  const auth = req.get('authorization');

  if (!auth) {
    return res.status(401).json({
      success: false,
      error: 'Authentication required'
    });
  }

  const parts = auth.split(' ');
  if (parts[0] !== 'Basic' || parts.length !== 2) {
    return res.status(401).json({
      success: false,
      error: 'Invalid authentication format'
    });
  }

  const credentials = Buffer.from(parts[1], 'base64').toString('utf8');
  const [dn, password] = credentials.split(':');

  req.ldap = { dn, password };
  next();
}

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    path: req.path
  });
});

// Error handler
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  res.status(500).json({
    success: false,
    error: 'Internal server error'
  });
});

// Start server
app.listen(port, () => {
  console.log(`PenuX LDAP API Server running on port ${port}`);
  console.log(`LDAP Host: ${LDAP_HOST}`);
  console.log(`Base DN: ${LDAP_BASE_DN}`);
});

module.exports = app;
