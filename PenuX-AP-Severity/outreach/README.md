# Research Collaboration Outreach

This directory contains tools and templates for outreach to gastroenterology and pancreatitis experts for research collaboration on the PenuX-AP-Severity project.

## Overview

The PenuX-AP-Severity research initiative seeks collaboration with leading experts in acute pancreatitis to:
- Validate the machine learning model on diverse patient cohorts
- Gather clinical workflow feedback
- Enable future prospective studies and clinical trials
- Co-author research publications

## Files

### Documentation

- **`docs/ehr_integration_guide.md`** — EHR system integration guide (HL7, FHIR, Camelion)
- **`docs/gastro_experts_israel.txt`** — List of Israeli gastroenterology experts (institutional affiliations)
- **`docs/email_templates_research_collab.md`** — Professional email templates (English) with best practices

### Tools

- **`send_research_collab_emails.py`** — Python script to send emails via Gmail API with safety features
- **`recipients_template.csv`** — Template for building approved recipient lists

### Logs

- **`email_log.json`** — Automatically created log of all sent emails (timestamp, recipient, template, status)

## Quick Start

### 1. Prepare Recipient List

Copy the template and add your approved recipients:

```bash
cp outreach/recipients_template.csv outreach/recipients_approved.csv
# Edit recipients_approved.csv with desired expert list
```

**CSV columns required:**
- `expert_name` — Full name (e.g., "Prof. Eran Goldin")
- `email_address` — Email address
- `institution` — Hospital/University name
- `department` — Department (optional but helpful)
- `country` — Country (optional)

### 2. Review Email Templates

Open `docs/email_templates_research_collab.md` and choose a template:
- **Template 1**: Initial research collaboration inquiry (recommended)
- **Template 2**: Department chair introduction request
- **Template 3**: Follow-up after positive response
- **Template 4**: IRB/data privacy assurance

### 3. Set Up Gmail API (One-time)

To send via Gmail OAuth:

**a. Create Gmail API credentials:**
1. Go to https://console.cloud.google.com/
2. Create a new project
3. Enable Gmail API
4. Create OAuth 2.0 credentials (Desktop application)
5. Download JSON file and save as `outreach/gmail_credentials.json`

**b. Install Python dependencies:**
```bash
pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 4. Run in Dry-Run Mode (Verify Before Sending)

Always test first to ensure emails look correct:

```bash
python outreach/send_research_collab_emails.py \
    --recipients outreach/recipients_approved.csv \
    --template template_1 \
    --dry-run
```

This shows which experts would receive emails **without sending**.

### 5. Send Emails

Once you've verified the dry-run output:

```bash
python outreach/send_research_collab_emails.py \
    --recipients outreach/recipients_approved.csv \
    --template template_1 \
    --send
```

The script will:
- Ask for final confirmation before sending
- Authenticate with Gmail
- Send up to 5 emails per run (safety limit)
- Log all sent emails to `email_log.json`

## Safety Features

### Built-in Safeguards

✅ **Dry-run mode**: Preview emails before sending  
✅ **Per-run limit**: Max 5 emails per execution (prevents accidental bulk send)  
✅ **Explicit confirmation**: Requires typing "yes" to actually send  
✅ **Email logging**: All sends logged with timestamp, recipient, status  
✅ **OAuth authentication**: Uses Gmail account (user must authenticate)  
✅ **No default sending**: Must explicitly use `--send` flag

### Recommended Workflow

1. Prepare CSV with approved recipients
2. Review email templates in `docs/email_templates_research_collab.md`
3. Run `--dry-run` to preview
4. Review dry-run output carefully
5. Run `--send` with confirmation
6. Check `email_log.json` to verify sends
7. Wait 10 days, then run with follow-up template if no response

## Email Best Practices

See `docs/email_templates_research_collab.md` for:
- Personalization tips
- Subject line recommendations
- Follow-up strategy
- Response rate expectations
- Collaboration timeline guidance

### Key Principles

- **Personalize**: Mention their specific research
- **Be clear**: Explain collaboration scope upfront
- **Be patient**: Academics typically respond in 2-4 weeks
- **Respect**: Use institutional channels when possible
- **Ethics**: Mention IRB approval and data privacy upfront

## Response Management

### If Expert Responds Positively

1. Review `docs/email_templates_research_collab.md` Template 3 (Follow-up)
2. Offer a 30-min virtual meeting
3. Share technical documentation
4. Draft collaboration agreement / MOU

### If Expert Doesn't Respond

1. Wait 10 days
2. Use similar template but note "following up on message from [date]"
3. If still no response after 21 days, move on
4. Consider asking them to refer a colleague

### If Expert Declines

- Thank them and ask if they know collaborators
- Do not follow up again
- Update `email_log.json` with status

## Tracking Outreach

Monitor your outreach progress:

```bash
# View sent emails
cat outreach/email_log.json | python -m json.tool

# Count sent vs. failed
grep '"success": true' outreach/email_log.json | wc -l
```

## Manual Email Sending

You can also send manually from `docs/email_templates_research_collab.md`:

1. Choose a template
2. Personalize (add expert's name, institution)
3. Send from your institutional email
4. Log in `email_log.json` manually

## Troubleshooting

### Gmail API Not Available
```
pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### OAuth Authentication Failed
- Check that `outreach/gmail_credentials.json` exists
- Ensure the OAuth app is configured for Desktop use
- Try deleting `outreach/gmail_token.json` and re-authenticating

### Recipients CSV Not Found
- Check CSV path is correct
- Ensure CSV has columns: `expert_name`, `email_address`
- Validate CSV format (no extra quotes or encoding issues)

### Emails Marked as Spam
- Ensure sending from institutional email (set in Gmail app)
- Include unsubscribe/contact info in signature
- Avoid "marketing" language or excessive formatting

## Important Notes

⚠️ **This is a research-only initiative**
- All communications should emphasize research purpose
- Mention IRB approval and data privacy upfront
- Do NOT claim clinical utility
- Be transparent about model maturity

⚠️ **Respect Expert Time**
- Academics receive many outreach emails
- Personalize every message
- Keep initial email concise
- Offer flexible collaboration terms

⚠️ **Data Privacy**
- Only share de-identified data summaries
- Use formal Data Use Agreements
- Mention GDPR/Privacy Law compliance
- Respect institutional policies

## Resources

- Full email templates: `docs/email_templates_research_collab.md`
- Israeli expert list: `docs/gastro_experts_israel.txt`
- Research GitHub: https://github.com/netanelcyber/penuX
- PenuX-AP-Severity docs: `../docs/`

## Support

For questions about:
- **Email templates**: See `docs/email_templates_research_collab.md`
- **Expert contacts**: See `docs/gastro_experts_israel.txt`
- **Technical setup**: See this README
- **Research collaboration**: Contact project maintainers

---

**Last Updated:** 2024-06-18  
**Version:** 1.0
