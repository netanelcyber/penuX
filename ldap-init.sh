#!/bin/bash
# Initialize LDAP with sample email users
# This script is called by docker-compose after openldap starts

set -e

LDAP_HOST="${LDAP_HOST:-openldap}"
LDAP_PORT="${LDAP_PORT:-389}"
LDAP_ADMIN_DN="${LDAP_ADMIN_DN:-cn=admin,dc=penux,dc=uk}"
LDAP_ADMIN_PASSWORD="${LDAP_ADMIN_PASSWORD:-admin123}"
LDIF_FILE="${1:-/ldap-init-users.ldif}"

echo "Waiting for LDAP to be ready..."
for i in {1..30}; do
  if ldapwhoami -H "ldap://$LDAP_HOST:$LDAP_PORT" -D "$LDAP_ADMIN_DN" -w "$LDAP_ADMIN_PASSWORD" 2>/dev/null; then
    echo "✅ LDAP is ready"
    break
  fi
  echo "Attempt $i/30 - waiting for LDAP..."
  sleep 2
done

echo "Initializing LDAP with email users from $LDIF_FILE..."
ldapadd -H "ldap://$LDAP_HOST:$LDAP_PORT" \
  -D "$LDAP_ADMIN_DN" \
  -w "$LDAP_ADMIN_PASSWORD" \
  -f "$LDIF_FILE" 2>&1 || {
  # If entries already exist, that's OK - script may have run before
  if grep -q "already exists" /dev/stderr 2>/dev/null; then
    echo "ℹ️ LDAP entries already initialized"
  else
    echo "❌ Failed to initialize LDAP"
    exit 1
  fi
}

echo "✅ LDAP initialization complete"
echo ""
echo "Sample email users created:"
echo "  - john.smith@penux.uk (Engineering Manager)"
echo "  - alice.johnson@penux.uk (Senior Developer)"
echo "  - bob.wilson@penux.uk (QA Engineer)"
echo "  - carol.davis@penux.uk (Product Manager)"
echo ""
echo "All users have password: 'password'"
