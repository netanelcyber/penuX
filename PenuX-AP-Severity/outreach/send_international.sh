#!/bin/bash
# Automated international expert outreach for PenuX-AP-Severity
# Sends research collaboration emails to 30+ leading gastroenterology experts worldwide
# OUTSIDE Israel focus

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
CREDS_FILE="$SCRIPT_DIR/gmail_credentials.json"
TOKEN_FILE="$SCRIPT_DIR/gmail_token.json"
RECIPIENTS_FILE="$SCRIPT_DIR/recipients_international.csv"
LOG_FILE="$SCRIPT_DIR/email_log_international.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}🌍 PenuX-AP-Severity: International Expert Outreach Campaign${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}\n"

# Step 1: Check Gmail credentials
setup_gmail_credentials() {
    echo -e "${YELLOW}[1/4] Checking Gmail credentials...${NC}"

    if [ -f "$CREDS_FILE" ]; then
        echo -e "${GREEN}✓ Gmail credentials found${NC}\n"
        return
    fi

    echo -e "${YELLOW}⚠️  Gmail OAuth credentials not found.${NC}\n"
    echo "Quick setup (1 min):"
    echo "  1. Go to: https://console.cloud.google.com/apis/credentials"
    echo "  2. Create 'OAuth 2.0 Desktop' credentials"
    echo "  3. Download JSON file"
    echo ""
    read -p "📋 Paste Gmail credentials JSON (then press CTRL+D): " -d '' creds_json || true

    if [ -z "$creds_json" ]; then
        echo -e "${RED}✗ No credentials provided${NC}"
        exit 1
    fi

    echo "$creds_json" > "$CREDS_FILE"
    echo -e "${GREEN}✓ Credentials saved${NC}\n"
}

# Step 2: Display recipient list
show_recipients() {
    echo -e "${YELLOW}[2/4] International expert recipients...${NC}\n"

    # Count by region
    europe=$(grep -c ",Europe," "$RECIPIENTS_FILE" || true)
    usa=$(grep -c ",USA," "$RECIPIENTS_FILE" || true)
    asia=$(grep -c ",Asia," "$RECIPIENTS_FILE" || true)
    other=$(grep -v "Europe\|USA\|Asia" "$RECIPIENTS_FILE" | grep -c "," || true)

    total=$((europe + usa + asia + other))

    echo "📊 Recipients by region:"
    echo "  🇪🇺 Europe: $europe experts"
    echo "  🇺🇸 USA: $usa experts"
    echo "  🌏 Asia: $asia experts"
    echo "  🌍 Other: $other experts"
    echo "  ────────────────────"
    echo "  📧 Total: $total experts\n"

    echo "🏥 Key institutions:"
    echo "  • Erasmus Medical Center (Netherlands)"
    echo "  • University of Michigan Medical Center (USA)"
    echo "  • Queen Mary University of London (UK)"
    echo "  • University of Padua (Italy)"
    echo "  • Karolinska Institute (Sweden)"
    echo "  • University of Tokyo (Japan)"
    echo "  • All India Institute (India)"
    echo ""
}

# Step 3: Authenticate
authenticate_gmail() {
    echo -e "${YELLOW}[3/4] Authenticating with Gmail...${NC}"

    if [ -f "$TOKEN_FILE" ]; then
        echo -e "${GREEN}✓ Gmail token valid${NC}\n"
        return
    fi

    echo -e "${YELLOW}First-time setup: will open browser for OAuth consent${NC}\n"
}

# Step 4: Send emails
send_emails() {
    echo -e "${YELLOW}[4/4] Sending research collaboration emails...${NC}\n"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ Python3 not found${NC}"
        exit 1
    fi

    # Install dependencies silently
    python3 -c "import google.auth" 2>/dev/null || {
        echo -e "${YELLOW}Installing dependencies...${NC}"
        pip install -q google-auth-oauthlib google-auth-httplib2 google-api-python-client
    }

    # Run Python email sender
    python3 << 'PYTHON_SCRIPT'
import sys
import os
import csv
import json
from pathlib import Path
from datetime import datetime
import base64
from email.mime.text import MIMEText
import time

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CREDS_FILE = os.path.join(SCRIPT_DIR, "gmail_credentials.json")
TOKEN_FILE = os.path.join(SCRIPT_DIR, "gmail_token.json")
RECIPIENTS_FILE = os.path.join(SCRIPT_DIR, "recipients_international.csv")
LOG_FILE = os.path.join(SCRIPT_DIR, "email_log_international.json")
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.send']

# Email template
EMAIL_TEMPLATE = """Dear {name},

I hope this message finds you well. I am reaching out from the PenuX research initiative to introduce an exciting machine learning project on early prediction of Severe Acute Pancreatitis (SAP).

PROJECT OVERVIEW
The PenuX-AP-Severity model uses routine admission laboratory data to identify high-risk pancreatitis patients within the first 4 hours of hospitalization. Our approach:

✓ Non-invasive: Uses only routine admission labs (WBC, CRP, creatinine, glucose, LDH, AST, etc.)
✓ Evidence-based: Implements 2012 Revised Atlanta Classification
✓ Validated: AUROC ~0.82 on retrospective cohorts
✓ Open-source: Full code and documentation on GitHub
✓ Ethical: IRB-approved, de-identified data only

WHY THIS MATTERS
Early identification of severe cases enables:
→ Risk stratification for ICU/high-dependency unit triage
→ Optimized fluid resuscitation and critical care protocols
→ Early intervention for organ failure prevention
→ Improved patient outcomes and resource allocation

YOUR EXPERTISE
Given your recognized leadership in {research_area}, we believe your insights would be invaluable for:

1. Clinical Validation
   Assess model performance on patient cohorts from your institution(s)

2. Workflow Integration
   Provide feedback on practical implementation in clinical settings

3. Joint Research
   Co-author validation studies and compare to existing severity scores (BISAP, APACHE II, Ranson)

4. Future Directions
   Explore prospective studies and clinical trial opportunities

RESEARCH ETHICS & TRANSPARENCY
• Institutional IRB approval obtained
• De-identified data only (GDPR / institutional privacy law compliant)
• No patient identifiers retained in analysis
• Open publication model (results shared transparently)
• GitHub repository: github.com/netanelcyber/penuX

NEXT STEP
I would be delighted to discuss collaboration opportunities in a brief call (20-30 min, virtual). Would you be interested?

I can provide:
→ Full technical documentation (model architecture, validation methodology)
→ Preliminary performance metrics vs. existing scores
→ Data sharing agreement templates
→ IRB/ethics compliance documentation

Thank you for considering this opportunity to advance SAP prediction research. Looking forward to connecting.

Best regards,
PenuX Research Team
Acute Pancreatitis Prediction Initiative

📧 GitHub: https://github.com/netanelcyber/penuX
📋 Documentation: github.com/netanelcyber/penuX/tree/main/PenuX-AP-Severity
🔬 Open Science • 🏥 Clinical Research • 🌍 International Collaboration
"""

def authenticate():
    """Authenticate with Gmail."""
    creds = None
    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, GMAIL_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())
    return creds

def send_email(service, to_email, subject, body):
    """Send email."""
    try:
        message = MIMEText(body)
        message['to'] = to_email
        message['subject'] = subject
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        result = service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
        return True, result.get('id')
    except Exception as e:
        return False, str(e)

def log_send(recipient, subject, success, msg_id=None):
    """Log email."""
    logs = []
    if Path(LOG_FILE).exists():
        with open(LOG_FILE, 'r') as f:
            logs = json.load(f)
    logs.append({
        "timestamp": datetime.utcnow().isoformat(),
        "expert_name": recipient.get('expert_name'),
        "institution": recipient.get('institution'),
        "country": recipient.get('country'),
        "region": recipient.get('region'),
        "email": recipient.get('email_address'),
        "subject": subject,
        "success": success,
        "message_id": msg_id,
    })
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

# Main
try:
    print("🔐 Authenticating with Gmail...")
    creds = authenticate()
    service = build('gmail', 'v1', credentials=creds)
    print("✅ Authenticated\n")

    recipients = []
    with open(RECIPIENTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        recipients = list(reader)

    print(f"📧 Sending to {len(recipients)} international experts...\n")

    sent = 0
    failed = 0

    for i, recipient in enumerate(recipients, 1):
        email = recipient['email_address']
        name = recipient['expert_name'].split()[0]
        country = recipient['country']
        research = recipient['research_focus']

        subject = "Research Collaboration: Early Prediction of Severe Acute Pancreatitis (PenuX-AP-Severity)"
        body = EMAIL_TEMPLATE.format(name=name, research_area=research.lower())

        print(f"[{i:2d}/{len(recipients)}] {name:15s} ({country:12s}) ... ", end="", flush=True)

        success, result = send_email(service, email, subject, body)
        if success:
            log_send(recipient, subject, True, result)
            print("✅")
            sent += 1
        else:
            log_send(recipient, subject, False)
            print(f"❌")
            failed += 1

        # Rate limit: 1 email per 2 seconds (Gmail friendly)
        if i < len(recipients):
            time.sleep(2)

    print(f"\n{'='*70}")
    print(f"✅ CAMPAIGN COMPLETE")
    print(f"{'='*70}")
    print(f"📊 Results: {sent} sent, {failed} failed")
    print(f"📋 Log file: {LOG_FILE}")
    print(f"{'='*70}\n")

except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT
}

# Main
main() {
    setup_gmail_credentials
    show_recipients
    authenticate_gmail

    # Confirmation
    echo -e "${CYAN}Ready to send research collaboration emails to ${CYAN}30+ international experts?${NC}"
    read -p "Continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled."
        exit 0
    fi
    echo ""

    send_emails

    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🌍 INTERNATIONAL OUTREACH CAMPAIGN LAUNCHED!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}\n"

    echo "📊 Campaign Summary:"
    echo "  • Recipients: 30+ international gastroenterology experts"
    echo "  • Regions: Europe, USA, Asia, Africa, Oceania"
    echo "  • Template: Research collaboration inquiry"
    echo "  • Log: $LOG_FILE"
    echo ""
    echo "⏱️  Expected Response Timeline:"
    echo "  • Week 1-2: Initial responses from interested experts"
    echo "  • Week 3-4: Follow-up conversations scheduled"
    echo "  • Month 2+: Formalized collaboration agreements"
    echo ""
    echo "📝 Next Steps:"
    echo "  1. Monitor email responses"
    echo "  2. Schedule 30-min calls with interested experts"
    echo "  3. Use Template 3 for follow-ups after 10 days if no response"
    echo "  4. Draft collaboration agreements / MOUs"
    echo ""
    echo "🔗 Resources:"
    echo "  • Full templates: $PROJECT_ROOT/docs/email_templates_research_collab.md"
    echo "  • Expert list: $RECIPIENTS_FILE"
    echo "  • Workflow guide: $SCRIPT_DIR/README.md"
    echo ""
}

main
