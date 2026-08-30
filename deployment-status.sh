#!/bin/bash
# Real-time deployment status dashboard
# Shows: tunnel URL, services health, auto-restart schedule

set -e

REPO="netanelcyber/penuX"
WORKFLOW="auto-deploy-tunnel.yml"

echo "🚀 PenuX LDAP Deployment Status"
echo "=================================="
echo ""

# Get latest run
RUN_JSON=$(curl -s "https://api.github.com/repos/$REPO/actions/workflows/$WORKFLOW/runs?per_page=1")
RUN_ID=$(echo "$RUN_JSON" | jq -r '.workflow_runs[0].id')
RUN_STATUS=$(echo "$RUN_JSON" | jq -r '.workflow_runs[0].status')
RUN_CONCLUSION=$(echo "$RUN_JSON" | jq -r '.workflow_runs[0].conclusion')
RUN_CREATED=$(echo "$RUN_JSON" | jq -r '.workflow_runs[0].created_at')

echo "Latest Run: #${RUN_ID}"
echo "Status: $RUN_STATUS ${RUN_CONCLUSION:+/ $RUN_CONCLUSION}"
echo "Created: $RUN_CREATED"
echo ""

# Get job details
JOBS_JSON=$(curl -s "https://api.github.com/repos/$REPO/actions/runs/$RUN_ID/jobs")
TOTAL_JOBS=$(echo "$JOBS_JSON" | jq '.jobs | length')

echo "Job Steps:"
echo "$JOBS_JSON" | jq -r '.jobs[0].steps[] | "  \(.name): \(.status) \(if .conclusion then "/ \(.conclusion)" else "" end)"'
echo ""

# If deployment is live, try to get the public URL from logs
if echo "$JOBS_JSON" | jq -e '.jobs[0].steps[] | select(.name | contains("localhost.run") and .conclusion == "success")' >/dev/null 2>&1; then
  echo "✅ Deployment is LIVE"
  echo ""
  echo "Access your LDAP:"
  echo "  Web UI:  https://<check Actions run summary for URL>"
  echo "  API:     /api/* paths on the same host"
  echo ""
  echo "Cron Schedule: Every 6 hours (auto-relaunches)"
  echo "Self-healing: Restarts stack if API becomes unhealthy"
else
  echo "⏳ Deployment is starting or waiting for secrets..."
fi

echo ""
echo "Full details: https://github.com/$REPO/actions/runs/$RUN_ID"
