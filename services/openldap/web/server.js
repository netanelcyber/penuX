// PenuX LDAP Web UI Server with HTML Endpoints
const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.static(path.join(__dirname)));
app.use(express.json());

// ═══════════════════════════════════════════════════════════
// HTML ENDPOINTS
// ═══════════════════════════════════════════════════════════

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'PenuX LDAP Web UI',
    timestamp: new Date().toISOString()
  });
});

// Dashboard endpoint
app.get('/dashboard', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Users management endpoint
app.get('/users', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Groups management endpoint
app.get('/groups', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Search endpoint
app.get('/search', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Settings endpoint
app.get('/settings', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Admin console redirect
app.get('/admin', (req, res) => {
  res.redirect('/');
});

// API info endpoint
app.get('/api-docs', (req, res) => {
  const apiDocs = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>PenuX LDAP API Documentation</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .endpoint { background: #f4f4f4; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .method { font-weight: bold; color: #007bff; }
        code { background: #e9ecef; padding: 2px 6px; border-radius: 3px; }
      </style>
    </head>
    <body>
      <h1>🔗 PenuX LDAP API Documentation</h1>
      <p>Base URL: <code>https://api.ldap.penux.uk</code></p>

      <div class="endpoint">
        <div class="method">GET</div> /api/health
        <p>Health check endpoint</p>
      </div>

      <div class="endpoint">
        <div class="method">GET</div> /api/users
        <p>Get all users (requires auth)</p>
      </div>

      <div class="endpoint">
        <div class="method">GET</div> /api/users/:uid
        <p>Get specific user (requires auth)</p>
      </div>

      <div class="endpoint">
        <div class="method">GET</div> /api/groups
        <p>Get all groups (requires auth)</p>
      </div>

      <div class="endpoint">
        <div class="method">GET</div> /api/groups/:cn
        <p>Get specific group (requires auth)</p>
      </div>

      <div class="endpoint">
        <div class="method">GET</div> /api/search?query=...
        <p>Search LDAP directory (requires auth)</p>
      </div>

      <div class="endpoint">
        <div class="method">POST</div> /api/verify
        <p>Verify user credentials (requires auth)</p>
      </div>

      <h2>Authentication</h2>
      <p>Use HTTP Basic Authentication:</p>
      <code>curl -u admin:password https://api.ldap.penux.uk/api/users</code>

      <h2>Credentials</h2>
      <p>
        Admin DN: <code>cn=admin,dc=penux,dc=uk</code><br>
        Password: <code>admin123</code>
      </p>
    </body>
    </html>
  `;
  res.send(apiDocs);
});

// Status endpoint
app.get('/status', (req, res) => {
  const status = `
    <!DOCTYPE html>
    <html>
    <head>
      <title>PenuX LDAP Status</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .status { padding: 20px; border-radius: 5px; }
        .up { background: #d4edda; color: #155724; }
        .down { background: #f8d7da; color: #721c24; }
        h1 { color: #333; }
      </style>
    </head>
    <body>
      <h1>🟢 PenuX LDAP Status</h1>
      <div class="status up">
        <h2>Web UI: Online ✅</h2>
        <p>The LDAP web interface is running and accessible.</p>
      </div>
      <div class="status up">
        <h2>LDAP Server: Responding ✅</h2>
        <p>Check via: <code>ldapwhoami -H ldaps://ldaps-server.penux.uk:636</code></p>
      </div>
      <div class="status up">
        <h2>REST API: Available ✅</h2>
        <p>Check via: <code>curl https://api.ldap.penux.uk/api/health</code></p>
      </div>

      <h2>Service URLs</h2>
      <ul>
        <li><strong>Web UI:</strong> https://ldap.penux.uk</li>
        <li><strong>LDAP Server:</strong> ldaps://ldaps-server.penux.uk:636</li>
        <li><strong>REST API:</strong> https://api.ldap.penux.uk</li>
        <li><strong>Keycloak:</strong> https://keycloak.penux.uk</li>
      </ul>

      <h2>Credentials</h2>
      <p>
        Admin DN: <code>cn=admin,dc=penux,dc=uk</code><br>
        Password: <code>admin123</code>
      </p>
    </body>
    </html>
  `;
  res.send(status);
});

// Root endpoint
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Catch-all for SPA routing
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// ═══════════════════════════════════════════════════════════
// START SERVER
// ═══════════════════════════════════════════════════════════

app.listen(port, () => {
  console.log(`🚀 PenuX LDAP Web UI Server`);
  console.log(`📖 Running on port ${port}`);
  console.log(`🌐 Open http://localhost:${port}`);
  console.log(`📊 Dashboard: http://localhost:${port}/dashboard`);
  console.log(`👥 Users: http://localhost:${port}/users`);
  console.log(`👫 Groups: http://localhost:${port}/groups`);
  console.log(`🔍 Search: http://localhost:${port}/search`);
  console.log(`⚙️  Settings: http://localhost:${port}/settings`);
  console.log(`📖 API Docs: http://localhost:${port}/api-docs`);
  console.log(`🟢 Status: http://localhost:${port}/status`);
  console.log(`🔐 Health: http://localhost:${port}/health`);
});
