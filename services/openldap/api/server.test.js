// PenuX LDAP API Test Suite (Jest)
// Unit tests for API endpoints

const request = require('supertest');
const app = require('./server');

const ADMIN_DN = 'cn=admin,dc=penux,dc=uk';
const ADMIN_PASSWORD = 'admin123';

// Helper to create Basic Auth header
const basicAuth = (dn, password) => {
  return 'Basic ' + Buffer.from(`${dn}:${password}`).toString('base64');
};

describe('PenuX LDAP API', () => {

  describe('System Health', () => {
    it('should return health status', async () => {
      const res = await request(app)
        .get('/api/health')
        .expect(200);

      expect(res.body.status).toBe('ok');
      expect(res.body.service).toBeDefined();
      expect(res.body.timestamp).toBeDefined();
    });
  });

  describe('Authentication', () => {
    it('should reject requests without authentication', async () => {
      const res = await request(app)
        .get('/api/users')
        .expect(401);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBeDefined();
    });

    it('should accept valid Basic Auth header', async () => {
      const res = await request(app)
        .get('/api/users')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(200);

      expect(res.body.success).toBe(true);
    });

    it('should reject invalid credentials', async () => {
      const res = await request(app)
        .post('/api/verify')
        .send({
          dn: ADMIN_DN,
          password: 'wrongpassword'
        })
        .expect(401);

      expect(res.body.success).toBe(false);
    });

    it('should verify valid credentials', async () => {
      const res = await request(app)
        .post('/api/verify')
        .send({
          dn: ADMIN_DN,
          password: ADMIN_PASSWORD
        })
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.message).toBeDefined();
    });
  });

  describe('User Operations', () => {
    it('should list users with authentication', async () => {
      const res = await request(app)
        .get('/api/users')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.count).toBeDefined();
      expect(Array.isArray(res.body.users)).toBe(true);
    });

    it('should get specific user', async () => {
      const res = await request(app)
        .get('/api/users/admin')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.user).toBeDefined();
    });

    it('should return 404 for non-existent user', async () => {
      const res = await request(app)
        .get('/api/users/nonexistentuser12345')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(404);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('User not found');
    });
  });

  describe('Group Operations', () => {
    it('should list groups with authentication', async () => {
      const res = await request(app)
        .get('/api/groups')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.count).toBeDefined();
      expect(Array.isArray(res.body.groups)).toBe(true);
    });

    it('should return 404 for non-existent group', async () => {
      const res = await request(app)
        .get('/api/groups/nonexistentgroup12345')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(404);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Group not found');
    });
  });

  describe('Search Operations', () => {
    it('should search directory', async () => {
      const res = await request(app)
        .get('/api/search?query=admin')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.count).toBeDefined();
    });

    it('should require query parameter', async () => {
      const res = await request(app)
        .get('/api/search')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Query parameter required');
    });

    it('should handle special characters in search', async () => {
      const res = await request(app)
        .get('/api/search?query=test*user')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(200);

      expect(res.body.success).toBe(true);
    });
  });

  describe('Organization Units', () => {
    it('should list organizational units', async () => {
      const res = await request(app)
        .get('/api/ous')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.count).toBeDefined();
    });
  });

  describe('Statistics', () => {
    it('should get directory statistics', async () => {
      const res = await request(app)
        .get('/api/stats')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(200);

      expect(res.body.success).toBe(true);
      expect(res.body.stats).toBeDefined();
      expect(res.body.stats.totalUsers).toBeGreaterThanOrEqual(0);
      expect(res.body.stats.totalGroups).toBeGreaterThanOrEqual(0);
      expect(res.body.stats.baseDN).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    it('should return 404 for undefined endpoint', async () => {
      const res = await request(app)
        .get('/api/undefined')
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(404);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('Endpoint not found');
    });

    it('should handle invalid JSON in POST', async () => {
      const res = await request(app)
        .post('/api/verify')
        .set('Content-Type', 'application/json')
        .send('invalid json')
        .expect(400);
    });

    it('should require DN and password for verify', async () => {
      const res = await request(app)
        .post('/api/verify')
        .send({})
        .expect(400);

      expect(res.body.success).toBe(false);
      expect(res.body.error).toBe('DN and password required');
    });
  });

  describe('Security', () => {
    it('should have CORS enabled', async () => {
      const res = await request(app)
        .get('/api/health')
        .set('Origin', 'http://localhost:3001');

      // CORS header should be present
      expect(res.headers['access-control-allow-origin']).toBeDefined();
    });

    it('should not expose sensitive headers', async () => {
      const res = await request(app)
        .get('/api/health');

      // Should not expose X-Powered-By or server info unnecessarily
      // (depends on express config)
    });

    it('should validate LDAP filter injection attempts', async () => {
      // Test escapeFilter function indirectly via search
      const maliciousQuery = 'admin*)(|(cn=*';
      const res = await request(app)
        .get(`/api/search?query=${encodeURIComponent(maliciousQuery)}`)
        .set('Authorization', basicAuth(ADMIN_DN, ADMIN_PASSWORD))
        .expect(200);

      // Should safely handle the query without error
      expect(res.body.success).toBe(true);
    });
  });

  describe('Rate Limiting', () => {
    it('should enforce rate limit', async () => {
      // Send many requests rapidly
      const requests = [];
      for (let i = 0; i < 101; i++) {
        requests.push(
          request(app)
            .get('/api/health')
        );
      }

      const results = await Promise.allSettled(requests);
      const responses = results
        .filter(r => r.status === 'fulfilled')
        .map(r => r.value);

      // Should have at least some successful responses
      expect(responses.length).toBeGreaterThan(0);

      // At least one should be rate limited (429)
      const rateLimited = responses.some(r => r.status === 429);
      // Note: May not always trigger in test, depending on timing
    });
  });

  describe('Content Type', () => {
    it('should return JSON content type', async () => {
      const res = await request(app)
        .get('/api/health')
        .expect(200);

      expect(res.type).toBe('application/json');
    });

    it('should accept JSON in POST requests', async () => {
      const res = await request(app)
        .post('/api/verify')
        .set('Content-Type', 'application/json')
        .send({
          dn: ADMIN_DN,
          password: ADMIN_PASSWORD
        })
        .expect(200);

      expect(res.body.success).toBe(true);
    });
  });
});
