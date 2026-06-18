#!/bin/bash
# One-command SMTP email campaign launcher
# Sends international expert outreach emails via Gmail SMTP or any email provider

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}🌍 PenuX-AP-Severity: SMTP Email Campaign${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python3 not found. Installing...${NC}"
    exit 1
fi

# Get email credentials
echo -e "${YELLOW}Email Credentials (SMTP)${NC}\n"

read -p "📧 Email address (e.g., your.email@gmail.com): " email_addr
if [ -z "$email_addr" ]; then
    echo -e "${YELLOW}Error: Email required${NC}"
    exit 1
fi

read -sp "🔐 Password or App Password: " email_pass
echo ""

if [ -z "$email_pass" ]; then
    echo -e "${YELLOW}Error: Password required${NC}"
    exit 1
fi

# Optional: Custom SMTP settings
echo ""
read -p "🔌 SMTP Server (default: smtp.gmail.com): " smtp_server
smtp_server=${smtp_server:-smtp.gmail.com}

read -p "📍 SMTP Port (default: 587): " smtp_port
smtp_port=${smtp_port:-587}

# Dry-run option
echo ""
echo -e "${YELLOW}Options:${NC}"
echo "  1. Send emails (actual)"
echo "  2. Preview only (dry-run)"
read -p "Choose (1/2): " choice

DRY_RUN=""
if [ "$choice" == "2" ]; then
    DRY_RUN="--dry-run"
    echo -e "\n${YELLOW}🔍 DRY-RUN MODE (no emails will be sent)${NC}\n"
else
    echo -e "\n${YELLOW}📧 SENDING MODE (emails will be sent)${NC}\n"
fi

# Run Python script
python3 "$SCRIPT_DIR/send_smtp_international.py" \
    --email "$email_addr" \
    --password "$email_pass" \
    --smtp "$smtp_server" \
    --port "$smtp_port" \
    $DRY_RUN

echo -e "${GREEN}Done!${NC}"
