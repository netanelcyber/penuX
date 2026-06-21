#!/bin/bash
# Auto-configure ngrok secrets via GitHub API
# Usage: ./ngrok-auto-setup.sh <GITHUB_TOKEN>

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <GITHUB_TOKEN>"
  echo ""
  echo "Steps:"
  echo "1. Create a GitHub Personal Access Token (Settings → Developer settings → Personal access tokens → Tokens (classic))"
  echo "   Scopes needed: repo (write access to secrets)"
  echo "2. Get your ngrok token: https://dashboard.ngrok.com/get-started/your-authtoken"
  echo "3. Get your ngrok domain: https://dashboard.ngrok.com/domains"
  echo "4. Run: $0 <YOUR_GITHUB_TOKEN>"
  echo ""
  exit 1
fi

GITHUB_TOKEN="$1"
REPO="netanelcyber/penuX"

echo "🔑 ngrok Auto-Setup"
echo "===================="
echo ""

# Get ngrok token
read -sp "Enter your ngrok auth token: " NGROK_TOKEN
echo ""

# Get ngrok domain
read -p "Enter your ngrok domain (e.g., penux-ldap.ngrok-free.app): " NGROK_DOMAIN
echo ""

if [ -z "$NGROK_TOKEN" ] || [ -z "$NGROK_DOMAIN" ]; then
  echo "❌ Token and domain are required"
  exit 1
fi

echo "Adding secrets to GitHub..."

# Create base64 encoded secret for GitHub API
encode_secret() {
  echo -n "$1" | base64 -w0
}

NGROK_TOKEN_B64=$(encode_secret "$NGROK_TOKEN")
NGROK_DOMAIN_B64=$(encode_secret "$NGROK_DOMAIN")

# Add NGROK_AUTHTOKEN secret
echo -n "Adding NGROK_AUTHTOKEN... "
curl -s -X PUT \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$REPO/actions/secrets/NGROK_AUTHTOKEN" \
  -d "{\"encrypted_value\": \"$NGROK_TOKEN_B64\", \"key_id\": \"\"}" >/dev/null 2>&1 || {
  echo "❌ Failed"
  echo ""
  echo "Troubleshooting:"
  echo "- Check that your GitHub token has 'repo' scope"
  echo "- Check that you have push access to $REPO"
  exit 1
}
echo "✅"

# Add NGROK_DOMAIN secret
echo -n "Adding NGROK_DOMAIN... "
curl -s -X PUT \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$REPO/actions/secrets/NGROK_DOMAIN" \
  -d "{\"encrypted_value\": \"$NGROK_DOMAIN_B64\", \"key_id\": \"\"}" >/dev/null 2>&1 || {
  echo "❌ Failed"
  exit 1
}
echo "✅"

echo ""
echo "✅ Secrets configured!"
echo ""
echo "Next step: Trigger the workflow"
echo "  https://github.com/$REPO/actions/workflows/auto-deploy-tunnel.yml"
echo "  Click 'Run workflow' → 'Run workflow'"
echo ""
echo "Your LDAP will be live at: https://$NGROK_DOMAIN"
