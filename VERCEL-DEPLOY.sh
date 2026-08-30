#!/bin/bash
# 🚀 Vercel Deployment Script for PenuX LDAP API

set -e

echo "🚀 PenuX LDAP API - Vercel Deployment"
echo "======================================"
echo ""

# Check if vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "✅ Vercel CLI found"
echo ""

# Navigate to API directory
cd "$(dirname "$0")/services/openldap/api"
echo "📁 Deploying from: $(pwd)"
echo ""

# Check if already logged in
echo "🔐 Authenticating with Vercel..."
vercel login || {
    echo "❌ Authentication failed"
    exit 1
}

echo ""
echo "🚀 Deploying to Vercel..."
vercel --prod --yes || {
    echo "❌ Deployment failed"
    exit 1
}

echo ""
echo "✅ Deployment successful!"
echo ""
echo "📝 Next steps:"
echo "1. Get your Vercel URL from the output above"
echo "2. Go to https://vercel.com/dashboard"
echo "3. Click your project"
echo "4. Go to Settings → Environment Variables"
echo "5. Add these variables:"
echo "   - LDAP_HOST=ldap://ldap-server.penux.uk:389"
echo "   - LDAP_BASE_DN=dc=penux,dc=uk"
echo "   - LDAP_ADMIN_DN=cn=admin,dc=penux,dc=uk"
echo "   - LDAP_ADMIN_PASSWORD=admin123"
echo "   - CORS_ORIGIN=*"
echo "6. Redeploy by running: vercel --prod"
echo ""
echo "7. Update Cloudflare DNS:"
echo "   - Add CNAME: ldap → <your-vercel-url>"
echo ""
echo "8. Test: curl https://ldap.penux.uk/api/health"
echo ""
