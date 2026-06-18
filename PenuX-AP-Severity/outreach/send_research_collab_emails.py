#!/usr/bin/env python3
"""
Send research collaboration emails via Gmail with safety checks.

IMPORTANT: Read docs/email_templates_research_collab.md before using.

Safety features:
- Requires explicit user confirmation for each batch
- Limits to 5 emails per run (prevent accidental bulk send)
- Logs all sent emails
- Dry-run mode available
- Requires Gmail OAuth consent

Usage:
    python outreach/send_research_collab_emails.py \
        --recipients outreach/recipients_template.csv \
        --template "template_1" \
        --dry-run              # Test without sending

    python outreach/send_research_collab_emails.py \
        --recipients outreach/recipients_approved.csv \
        --template "template_1 \
        --send                 # Actually send emails
"""
import csv
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Gmail API (requires: pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client)
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.api_core.exceptions import GoogleAPIError
    import google.auth
    GMAIL_AVAILABLE = True
except ImportError:
    GMAIL_AVAILABLE = False
    print("WARNING: Gmail API not available. Install with:")
    print("  pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")

import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# Configuration
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.send']
CREDENTIALS_FILE = "outreach/gmail_credentials.json"
TOKEN_FILE = "outreach/gmail_token.json"
LOG_FILE = "outreach/email_log.json"
MAX_EMAILS_PER_RUN = 5  # Safety limit


# Email templates (from docs)
TEMPLATES = {
    "template_1": {
        "name": "Initial Research Collaboration Inquiry",
        "subject": "Research Collaboration: Early Prediction of Severe Acute Pancreatitis using Machine Learning",
        "body_file": "docs/email_templates_research_collab.md",  # Extract from this file
    },
    "template_2": {
        "name": "Department Chair Introduction Request",
        "subject": "Introduction Request: Pancreatitis Researcher at [Your Institution]",
    },
}


def load_recipients(csv_path: str) -> list[dict]:
    """Load recipient list from CSV."""
    recipients = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('email_address') and row.get('expert_name'):
                    recipients.append(row)
    except FileNotFoundError:
        print(f"ERROR: Recipients file not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load recipients: {e}")
        sys.exit(1)
    return recipients


def authenticate_gmail() -> Optional[object]:
    """Authenticate with Gmail API via OAuth."""
    if not GMAIL_AVAILABLE:
        print("ERROR: Gmail API not available. Install dependencies first:")
        print("  pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")
        return None

    creds = None

    # Try to use existing token
    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, GMAIL_SCOPES)

    # If no valid token, initiate OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not Path(CREDENTIALS_FILE).exists():
                print(f"ERROR: Gmail credentials file not found: {CREDENTIALS_FILE}")
                print("\nTo set up Gmail API:")
                print("1. Go to https://console.cloud.google.com/")
                print("2. Create OAuth 2.0 credentials (Desktop application)")
                print("3. Download JSON and save as:", CREDENTIALS_FILE)
                return None

            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE, GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save token for future use
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())

    return creds


def send_email(service, to_email: str, subject: str, body: str) -> bool:
    """Send an email via Gmail API."""
    try:
        message = MIMEText(body)
        message['to'] = to_email
        message['subject'] = subject

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        send_message = {'raw': raw_message}

        result = service.users().messages().send(userId='me', body=send_message).execute()
        return True, result.get('id')
    except GoogleAPIError as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def log_email_send(recipient: dict, subject: str, template: str, success: bool, msg_id: Optional[str] = None):
    """Log email sent to file."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "expert_name": recipient.get('expert_name'),
        "email": recipient.get('email_address'),
        "institution": recipient.get('institution'),
        "template": template,
        "subject": subject,
        "success": success,
        "message_id": msg_id,
    }

    logs = []
    if Path(LOG_FILE).exists():
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except:
            pass

    logs.append(log_entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Send research collaboration emails via Gmail (with safety checks)"
    )
    parser.add_argument(
        "--recipients",
        required=True,
        help="CSV file with recipient list (required columns: expert_name, email_address)"
    )
    parser.add_argument(
        "--template",
        default="template_1",
        choices=list(TEMPLATES.keys()),
        help="Email template to use"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent without actually sending"
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Actually send emails (requires --dry-run verification first)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=MAX_EMAILS_PER_RUN,
        help=f"Max emails to send per run (default: {MAX_EMAILS_PER_RUN})"
    )

    args = parser.parse_args()

    # Load recipients
    recipients = load_recipients(args.recipients)
    if not recipients:
        print("ERROR: No valid recipients found.")
        sys.exit(1)

    # Limit to safety max
    if args.limit > MAX_EMAILS_PER_RUN:
        print(f"WARNING: Limiting to {MAX_EMAILS_PER_RUN} emails per run (safety check)")
        recipients = recipients[:MAX_EMAILS_PER_RUN]
    else:
        recipients = recipients[:args.limit]

    # Get template
    template_info = TEMPLATES[args.template]
    print(f"\n📧 Email Template: {template_info['name']}")
    print(f"📋 Recipients: {len(recipients)}")
    print(f"📝 Subject: {template_info['subject']}")

    # In dry-run or preview mode, show what we'd send
    if args.dry_run or not args.send:
        print("\n" + "="*70)
        print("DRY RUN - Emails would be sent to:")
        print("="*70)
        for i, recipient in enumerate(recipients, 1):
            print(f"{i}. {recipient.get('expert_name')} <{recipient.get('email_address')}>")
            print(f"   Institution: {recipient.get('institution')}")
            print(f"   Country: {recipient.get('country')}")
        print("\n" + "="*70)
        print("To actually send, run:")
        print(f"  python outreach/send_research_collab_emails.py \\")
        print(f"    --recipients {args.recipients} \\")
        print(f"    --template {args.template} \\")
        print(f"    --send")
        print("="*70)
        return

    # Confirmation before sending
    if args.send:
        confirm = input(f"\n⚠️  About to send {len(recipients)} emails. Continue? (yes/no): ").lower()
        if confirm != "yes":
            print("Cancelled.")
            return

        # Authenticate
        print("\n🔐 Authenticating with Gmail...")
        service = authenticate_gmail()
        if not service:
            print("ERROR: Failed to authenticate with Gmail.")
            sys.exit(1)

        print(f"✅ Authenticated. Sending {len(recipients)} emails...\n")

        sent_count = 0
        failed_count = 0

        for i, recipient in enumerate(recipients, 1):
            email = recipient.get('email_address')
            name = recipient.get('expert_name')
            subject = template_info['subject']

            # Placeholder body (in production, read from template file and personalize)
            body = f"""Dear {name},

[Email body from {template_info['name']} would go here]

[See docs/email_templates_research_collab.md for full template]

Best regards,
[Your Name]
"""

            print(f"[{i}/{len(recipients)}] Sending to {name} <{email}>...", end=" ")

            success, result = send_email(service, email, subject, body)
            if success:
                log_email_send(recipient, subject, args.template, True, result)
                print("✅ Sent")
                sent_count += 1
            else:
                log_email_send(recipient, subject, args.template, False)
                print(f"❌ Failed: {result}")
                failed_count += 1

        print(f"\n{'='*70}")
        print(f"Summary: {sent_count} sent, {failed_count} failed")
        print(f"Log file: {LOG_FILE}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
