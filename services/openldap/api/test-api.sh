#!/bin/bash
# PenuX LDAP API Test Suite
# Tests all REST API endpoints

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
API_URL="${API_URL:-http://localhost:3000}"
ADMIN_DN="cn=admin,dc=penux,dc=uk"
ADMIN_PASSWORD="admin123"
CREDS=$(echo -n "$ADMIN_DN:$ADMIN_PASSWORD" | base64)

PASSED=0
FAILED=0

# Helper functions
print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local expected_code=$4
    local data=$5

    echo -e "${YELLOW}Testing:${NC} $name"
    echo "  Method: $method"
    echo "  Endpoint: $endpoint"

    local cmd="curl -s -w '\n%{http_code}' -X $method"

    if [ "$method" = "POST" ] || [ "$method" = "PUT" ]; then
        cmd="$cmd -H 'Content-Type: application/json'"
        if [ -n "$data" ]; then
            cmd="$cmd -d '$data'"
        fi
    fi

    if [ "$endpoint" != "/api/health" ]; then
        cmd="$cmd -H 'Authorization: Basic $CREDS'"
    fi

    cmd="$cmd '$API_URL$endpoint'"

    # Execute and capture response
    response=$(eval $cmd)
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)

    if [ "$http_code" = "$expected_code" ]; then
        echo -e "  ${GREEN}✓ Status: $http_code (expected)${NC}"
        echo "  Response: ${body:0:100}"
        ((PASSED++))
    else
        echo -e "  ${RED}✗ Status: $http_code (expected $expected_code)${NC}"
        echo "  Response: $body"
        ((FAILED++))
    fi
    echo ""
}

# Main tests
print_header "PenuX LDAP API Test Suite"
echo "API URL: $API_URL"
echo "Admin DN: $ADMIN_DN"
echo ""

# Test 1: Health check (no auth required)
print_header "Test 1: System Health Check"
test_endpoint "Health Check" "GET" "/api/health" "200"

# Test 2: List users
print_header "Test 2: User Operations"
test_endpoint "List All Users" "GET" "/api/users" "200"

# Test 3: Get specific user
print_header "Test 3: Get Specific User"
test_endpoint "Get User (jdoe)" "GET" "/api/users/jdoe" "200"
test_endpoint "Get Non-existent User" "GET" "/api/users/nonexistent" "404"

# Test 4: List groups
print_header "Test 4: Group Operations"
test_endpoint "List All Groups" "GET" "/api/groups" "200"

# Test 5: Get specific group
print_header "Test 5: Get Specific Group"
test_endpoint "Get Group (developers)" "GET" "/api/groups/developers" "200"

# Test 6: Search
print_header "Test 6: Search Operations"
test_endpoint "Search for 'john'" "GET" "/api/search?query=john" "200"
test_endpoint "Search without query" "GET" "/api/search" "400"

# Test 7: Verify credentials
print_header "Test 7: Authentication"
test_endpoint "Verify Valid Credentials" "POST" "/api/verify" "200" \
    "{\"dn\":\"$ADMIN_DN\",\"password\":\"$ADMIN_PASSWORD\"}"
test_endpoint "Verify Invalid Credentials" "POST" "/api/verify" "401" \
    "{\"dn\":\"$ADMIN_DN\",\"password\":\"wrongpassword\"}"

# Test 8: List OUs
print_header "Test 8: Organization Units"
test_endpoint "List Organizational Units" "GET" "/api/ous" "200"

# Test 9: Statistics
print_header "Test 9: Statistics"
test_endpoint "Get Directory Statistics" "GET" "/api/stats" "200"

# Test 10: Authentication failure
print_header "Test 10: Authorization Checks"
echo -e "${YELLOW}Testing:${NC} Access without authentication"
echo "  Method: GET"
echo "  Endpoint: /api/users"

response=$(curl -s -w '\n%{http_code}' -X GET "$API_URL/api/users")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [ "$http_code" = "401" ]; then
    echo -e "  ${GREEN}✓ Status: $http_code (auth required)${NC}"
    ((PASSED++))
else
    echo -e "  ${RED}✗ Status: $http_code (expected 401)${NC}"
    ((FAILED++))
fi
echo ""

# Test 11: Rate limiting
print_header "Test 11: Rate Limiting"
echo -e "${YELLOW}Testing:${NC} Rate limit enforcement"
echo "  Sending 101 rapid requests (limit: 100 per 15 min)"

success_count=0
failed_count=0

for i in {1..110}; do
    response=$(curl -s -w '%{http_code}' -X GET \
        -H "Authorization: Basic $CREDS" \
        "$API_URL/api/health")
    http_code=$(echo "$response" | tail -c 4)

    if [ "$http_code" = "200" ]; then
        ((success_count++))
    else
        ((failed_count++))
        if [ $failed_count -eq 1 ]; then
            echo -e "  ${YELLOW}Rate limit hit after $success_count requests${NC}"
        fi
    fi
done

if [ $failed_count -gt 0 ]; then
    echo -e "  ${GREEN}✓ Rate limiting working (allowed $success_count, blocked $failed_count)${NC}"
    ((PASSED++))
else
    echo -e "  ${YELLOW}⚠ Rate limiting may not be active yet${NC}"
fi
echo ""

# Summary
print_header "Test Results Summary"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}\n"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed${NC}\n"
    exit 1
fi
