// Simple HTTP server to serve PenuX LDAP Web UI
const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.static(path.join(__dirname)));

// Serve index.html for root and SPA routes
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Catch-all for SPA routing
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'PenuX LDAP Web UI',
    timestamp: new Date().toISOString()
  });
});

app.listen(port, () => {
  console.log(`🚀 PenuX LDAP Web UI running on port ${port}`);
  console.log(`📖 Open http://localhost:${port}`);
});
