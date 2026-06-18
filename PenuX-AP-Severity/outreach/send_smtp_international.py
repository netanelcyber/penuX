#!/usr/bin/env python3
"""
Send international expert outreach emails via SMTP (Gmail or any email provider)
Fully automated one-command execution.

Usage:
    python3 send_smtp_international.py \
        --email your.email@gmail.com \
        --password "your-app-password" \
        --smtp smtp.gmail.com \
        --port 587
"""

import sys
import smtplib
import csv
import json
import argparse
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# Configuration
SCRIPT_DIR = Path(__file__).parent
RECIPIENTS_FILE = SCRIPT_DIR / "recipients_international.csv"
LOG_FILE = SCRIPT_DIR / "email_log_smtp.json"

# Colors
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
NC = '\033[0m'

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


def load_recipients():
    """Load expert recipient list."""
    recipients = []
    try:
        with open(RECIPIENTS_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            recipients = list(reader)
    except FileNotFoundError:
        print(f"{RED}✗ Recipients file not found: {RECIPIENTS_FILE}{NC}")
        sys.exit(1)
    return recipients


def send_email_smtp(smtp_server, smtp_port, sender_email, sender_password, recipient_email, subject, body, use_ssl=False):
    """Send email via SMTP (STARTTLS on 587, SSL on 465)."""
    import socket
    try:
        message = MIMEMultipart("alternative")
        message["From"] = sender_email
        message["To"] = recipient_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        if use_ssl:
            ctx = smtplib.ssl.create_default_context()
            with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=15, context=ctx) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, message.as_string())
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient_email, message.as_string())

        return True, None
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed (check email/password)"
    except smtplib.SMTPException as e:
        return False, f"SMTP error: {str(e)}"
    except (OSError, socket.error) as e:
        return False, f"Network error: {str(e)}"
    except Exception as e:
        return False, str(e)


def log_send(recipient, subject, success, error=None):
    """Log email send."""
    logs = []
    if LOG_FILE.exists():
        with open(LOG_FILE, 'r') as f:
            try:
                logs = json.load(f)
            except:
                logs = []

    logs.append({
        "timestamp": datetime.utcnow().isoformat(),
        "expert_name": recipient.get('expert_name'),
        "institution": recipient.get('institution'),
        "country": recipient.get('country'),
        "region": recipient.get('region'),
        "email": recipient.get('email_address'),
        "subject": subject,
        "success": success,
        "error": error,
    })

    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Send international expert outreach emails via SMTP"
    )
    parser.add_argument("--email", required=True, help="Sender email address")
    parser.add_argument("--password", required=True, help="Email password or app password")
    parser.add_argument("--smtp", default="smtp.gmail.com", help="SMTP server (default: smtp.gmail.com)")
    parser.add_argument("--port", type=int, default=587, help="SMTP port (default: 587)")
    parser.add_argument("--ssl", action="store_true", help="Use SSL (port 465) instead of STARTTLS")
    parser.add_argument("--dry-run", action="store_true", help="Preview without sending")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    print(f"{BLUE}════════════════════════════════════════════════════════════{NC}")
    print(f"{CYAN}🌍 PenuX-AP-Severity: International SMTP Email Campaign{NC}")
    print(f"{BLUE}════════════════════════════════════════════════════════════{NC}\n")

    # Load recipients
    recipients = load_recipients()
    print(f"{YELLOW}[1/3] Recipients loaded: {len(recipients)} experts{NC}\n")

    # Show summary
    print(f"📊 Campaign Summary:")
    print(f"  📧 From: {args.email}")
    print(f"  🔌 SMTP: {args.smtp}:{args.port}")
    print(f"  👥 Recipients: {len(recipients)}")
    print(f"  📍 Regions: Europe, USA, Asia, Africa, Oceania")
    print(f"  📄 Template: Research collaboration inquiry\n")

    # Dry-run mode
    if args.dry_run:
        print(f"{YELLOW}[2/3] DRY-RUN MODE (no emails sent){NC}\n")
        print("Recipients that would receive emails:")
        for i, recipient in enumerate(recipients, 1):
            print(f"  {i:2d}. {recipient['expert_name']:30s} ({recipient['country']:12s}) {recipient['email_address']}")
        print(f"\nTo send: Remove --dry-run flag")
        return

    # Confirmation
    print(f"{YELLOW}[2/3] Ready to send {len(recipients)} emails{NC}")
    if not args.yes:
        confirm = input(f"\n{CYAN}Continue? (yes/no): {NC}").strip().lower()
        if confirm != "yes":
            print("Cancelled.")
            return

    # Send emails
    print(f"\n{YELLOW}[3/3] Sending emails...{NC}\n")

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

        success, error = send_email_smtp(
            args.smtp, args.port,
            args.email, args.password,
            email, subject, body,
            use_ssl=args.ssl
        )

        if success:
            log_send(recipient, subject, True)
            print(f"{GREEN}✅{NC}")
            sent += 1
        else:
            log_send(recipient, subject, False, error)
            print(f"{RED}❌ {error}{NC}")
            failed += 1

        # Rate limit: 2 seconds between emails (SMTP friendly)
        if i < len(recipients):
            time.sleep(2)

    # Results
    print(f"\n{BLUE}════════════════════════════════════════════════════════════{NC}")
    print(f"{GREEN}✅ CAMPAIGN COMPLETE{NC}")
    print(f"{BLUE}════════════════════════════════════════════════════════════{NC}\n")

    print(f"📊 Results:")
    print(f"  ✅ Sent: {sent}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📋 Log: {LOG_FILE}\n")

    print(f"⏱️  Expected Response Timeline:")
    print(f"  • Week 1-2: Initial responses")
    print(f"  • Week 3-4: Follow-up conversations")
    print(f"  • Month 2+: Formalized collaborations\n")

    print(f"🔗 Resources:")
    print(f"  • GitHub: https://github.com/netanelcyber/penuX")
    print(f"  • Docs: github.com/netanelcyber/penuX/tree/main/PenuX-AP-Severity\n")


if __name__ == "__main__":
    main()
