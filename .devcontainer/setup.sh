#!/bin/bash
# Post-create setup for GitHub Codespaces
set -e

echo "=== PenuX LDAP Codespaces Setup ==="

# Make scripts executable
chmod +x cloudflare-tunnel-setup.sh 2>/dev/null || true
chmod +x codespaces-tunnel-setup.sh 2>/dev/null || true
chmod +x free-tunnel-setup.sh 2>/dev/null || true
chmod +x docker-hub-push.sh 2>/dev/null || true

echo ""
echo "=== Ready! ==="
echo ""
echo "To deploy with a FREE tunnel (no credit card needed):"
echo ""
echo "  Option 1 (Recommended - permanent URL):"
echo "    1. Sign up at https://ngrok.com (email only)"
echo "    2. Get token: https://dashboard.ngrok.com/get-started/your-authtoken"
echo "    3. Get free static domain: https://dashboard.ngrok.com/domains"
echo "    4. Run: NGROK_AUTHTOKEN=xxx NGROK_DOMAIN=your-name.ngrok-free.app ./free-tunnel-setup.sh netanelcyber"
echo ""
echo "  Option 2 (Zero signup - temporary URL):"
echo "    ./free-tunnel-setup.sh netanelcyber"
echo "    (URL changes each session, no account needed)"
echo ""
echo "Commands:"
echo "  Status       : docker compose -f docker-compose-hub.yml ps"
echo "  Logs         : docker compose -f docker-compose-hub.yml logs -f api"
echo "  Stop         : docker compose -f docker-compose-hub.yml down"
echo "  Health check : curl http://localhost:3000/api/health"
