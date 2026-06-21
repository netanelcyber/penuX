#!/bin/bash
# 🚀 Full ngrok auto-setup — from signup to live deployment in one go
#
# This script:
# 1. Opens ngrok signup in your browser
# 2. Waits for you to complete signup + claim domain
# 3. Prompts for token + domain
# 4. Adds GitHub secrets automatically
# 5. Triggers the workflow → LIVE with permanent URL
#
# Requirements:
#   - gh CLI (GitHub CLI) installed and authenticated
#   - Browser (auto-opens signup page)
#
# Usage: ./ngrok-full-auto.sh

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✅ $1${NC}"; }
info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; exit 1; }

REPO="netanelcyber/penuX"
BRANCH="claude/laughing-cori-jlr7od"

echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${BLUE}   ngrok Permanent Domain — Full Auto${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""

# Check dependencies
if ! command -v gh &>/dev/null; then
  fail "GitHub CLI (gh) not found. Install from: https://cli.github.com"
fi
ok "GitHub CLI found"

if ! gh auth status >/dev/null 2>&1; then
  fail "Not authenticated. Run: gh auth login"
fi
ok "GitHub authenticated"

# Step 1: Open ngrok signup
echo ""
info "Opening ngrok signup in your browser..."
echo "  → Please complete the signup (takes ~30 seconds)"
echo "  → After signup, you'll be on the dashboard"
echo "  → Go to 'Domains' tab and claim ONE free domain"
echo ""

if command -v open &>/dev/null; then
  open "https://ngrok.com/signup"
elif command -v xdg-open &>/dev/null; then
  xdg-open "https://ngrok.com/signup"
else
  echo "   Please manually visit: https://ngrok.com/signup"
fi

read -p "Press Enter once you've signed up and claimed your free domain..."
echo ""

# Step 2: Get ngrok credentials
info "Getting your ngrok credentials..."
echo ""
echo "Instructions:"
echo "  1. Go to: https://dashboard.ngrok.com/get-started/your-authtoken"
echo "  2. Copy your auth token (looks like: eyJhbGc...)"
echo "  3. Paste it below"
echo ""
read -sp "Paste your ngrok auth token: " NGROK_TOKEN
echo ""

if [ -z "$NGROK_TOKEN" ]; then
  fail "Token cannot be empty"
fi
ok "Token received"

echo ""
echo "Instructions:"
echo "  1. Go to: https://dashboard.ngrok.com/domains"
echo "  2. Copy your free domain (e.g., penux-ldap.ngrok-free.app)"
echo "  3. Paste it below"
echo ""
read -p "Paste your ngrok domain: " NGROK_DOMAIN

if [ -z "$NGROK_DOMAIN" ]; then
  fail "Domain cannot be empty"
fi
ok "Domain received: $NGROK_DOMAIN"

# Step 3: Verify repo and auth
echo ""
info "Verifying GitHub access..."
REPO_CHECK=$(gh repo view "$REPO" --json name 2>/dev/null | jq -r '.name' || echo "")
if [ "$REPO_CHECK" != "penuX" ]; then
  fail "Cannot access repo. Make sure you have push access to $REPO"
fi
ok "Repository access verified"

# Step 4: Add secrets
echo ""
info "Adding secrets to GitHub..."

echo -n "  NGROK_AUTHTOKEN... "
gh secret set NGROK_AUTHTOKEN --body "$NGROK_TOKEN" --repo "$REPO" 2>/dev/null || fail "Failed"
echo "✅"

echo -n "  NGROK_DOMAIN... "
gh secret set NGROK_DOMAIN --body "$NGROK_DOMAIN" --repo "$REPO" 2>/dev/null || fail "Failed"
echo "✅"

ok "Secrets configured in GitHub"

# Step 5: Trigger deployment
echo ""
info "Triggering deployment workflow..."
gh workflow run auto-deploy-tunnel.yml -r "$BRANCH" 2>/dev/null || fail "Failed to trigger workflow"
ok "Workflow triggered"

# Step 6: Wait for run to start and show status
echo ""
info "Waiting for deployment to start..."
sleep 5

# Get latest run
RUN_JSON=$(gh run list --workflow auto-deploy-tunnel.yml --branch "$BRANCH" --limit 1 --json databaseId,status,conclusion)
RUN_ID=$(echo "$RUN_JSON" | jq -r '.[0].databaseId')
RUN_STATUS=$(echo "$RUN_JSON" | jq -r '.[0].status')

if [ -z "$RUN_ID" ] || [ "$RUN_ID" = "null" ]; then
  fail "Could not get workflow run ID"
fi

ok "Deployment started: Run #$RUN_ID"

# Step 7: Show summary
echo ""
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo -e "${GREEN}✨ All Done! Your LDAP is Deploying${NC}"
echo -e "${BLUE}════════════════════════════════════════${NC}"
echo ""
echo "Your deployment details:"
echo "  Permanent URL: https://$NGROK_DOMAIN"
echo "  Workflow run:  https://github.com/$REPO/actions/runs/$RUN_ID"
echo "  Status:        $RUN_STATUS"
echo ""
echo "Next steps:"
echo "  1. Workflow will finish in ~2-3 minutes"
echo "  2. Check the run summary for confirmation"
echo "  3. Access your LDAP at: https://$NGROK_DOMAIN"
echo ""
echo "Commands:"
echo "  • View logs:   gh run view $RUN_ID --log"
echo "  • Test API:    curl https://$NGROK_DOMAIN/api/health"
echo "  • Monitor:     ./deployment-status.sh"
echo ""
