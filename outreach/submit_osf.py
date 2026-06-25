#!/usr/bin/env python3
"""
Automated OSF Preprints submission for PenuX Stage I Study Protocol.
Usage:
    OSF_TOKEN=<your_token> python3 submit_osf.py
Or set token in .env or pass as --token argument.

Get your token at: https://osf.io/settings/tokens/
  -> Create token -> check 'osf.full_write' scope -> copy
"""

import os
import sys
import json
import time
import argparse
import requests

# ── Metadata ──────────────────────────────────────────────────────────────────
TITLE = (
    "PenuX-SAP Stage I Observational Cohort Study Protocol: "
    "Prospective External Validation of a Machine-Learning-Based "
    "Severity Prediction System in Adults Admitted with Acute Pancreatitis"
)
ABSTRACT = (
    "Background: Severe acute pancreatitis (SAP) carries 20-40% mortality. "
    "Existing severity scores (BISAP, APACHE II, Ranson) require 24-48 hours "
    "of observation before reliable stratification, limiting early triage. "
    "PenuX is a Random Forest-based ensemble that predicts SAP severity at "
    "admission using 59 routine laboratory variables (AUC-ROC 0.877, "
    "sensitivity 96.8%, F1 0.917 in a retrospective cohort of 722 patients "
    "labelled by the 2012 Revised Atlanta Classification).\n\n"
    "Objectives: This Stage I protocol describes a prospective, observational, "
    "silent-mode external validation of PenuX in a real-world hospital "
    "population. The primary endpoint is AUC-ROC of PenuX SAP severity "
    "probability vs. RAC 2012 gold-standard label at 48 hours (H0: AUC <= 0.70, "
    "one-sided alpha = 0.05, power 80%). Secondary endpoints include "
    "calibration, lead-time advantage over BISAP, pancreatic sepsis sub-model "
    "performance, and subgroup heterogeneity.\n\n"
    "Design: Prospective observational cohort, n = 220, 12-month enrolment, "
    "silent-mode (clinicians blinded to PenuX outputs). IRB/Helsinki approval "
    "required. Data stored in REDCap. GDPR and Israeli Privacy Protection Law "
    "compliant. Outcomes will inform a Stage II prospective interventional pilot."
)
AUTHORS = [
    {"name": "Netanel Stern", "email": "nsh531@gmail.com"}
]
TAGS = [
    "acute pancreatitis", "severe acute pancreatitis", "machine learning",
    "random forest", "severity prediction", "SAP", "observational study",
    "external validation", "PenuX", "clinical AI", "gastroenterology",
    "Revised Atlanta Classification", "study protocol"
]
PROVIDER = "osf"           # OSF Preprints (free, indexed by Google Scholar)
LICENSE_NAME = "CC-By Attribution 4.0 International"
LICENSE_ID   = "563c1cf88c5e4a3877f9e96c"   # CC-BY 4.0 on OSF
SUBJECTS = [
    # OSF subject IDs for Medicine & Public Health > Gastroenterology
    # Using text-based subjects (OSF accepts free-text tags as well)
]
PDF_PATH = os.path.join(os.path.dirname(__file__), "PenuX_SAP_Stage1_Study_Protocol.pdf")

OSF_API = "https://api.osf.io/v2"


def get_token():
    token = os.environ.get("OSF_TOKEN") or ""
    if not token:
        # Try .env file in same dir
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            for line in open(env_path):
                if line.startswith("OSF_TOKEN="):
                    token = line.strip().split("=", 1)[1].strip().strip('"')
    if not token:
        print("ERROR: OSF_TOKEN not set.")
        print("  Get it at https://osf.io/settings/tokens/")
        print("  Then run: OSF_TOKEN=<token> python3 submit_osf.py")
        sys.exit(1)
    return token


def headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }


def create_osf_node(token):
    """Create a private OSF project to host the preprint file."""
    payload = {
        "data": {
            "type": "nodes",
            "attributes": {
                "title": TITLE,
                "category": "project",
                "public": False,
                "description": ABSTRACT[:300]
            }
        }
    }
    r = requests.post(f"{OSF_API}/nodes/", headers=headers(token), json=payload)
    r.raise_for_status()
    node_id = r.json()["data"]["id"]
    print(f"  Created OSF node: {node_id}")
    return node_id


def upload_pdf(token, node_id, pdf_path):
    """Upload PDF to the OSF node's default storage via WaterButler."""
    filename = os.path.basename(pdf_path)
    upload_url = (
        f"https://files.osf.io/v1/resources/{node_id}/providers/osfstorage/?name={filename}"
    )
    with open(pdf_path, "rb") as f:
        r = requests.put(
            upload_url,
            headers={"Authorization": f"Bearer {token}"},
            data=f
        )
    r.raise_for_status()
    data = r.json()["data"]
    file_id = data["id"]
    print(f"  Uploaded PDF: {filename} (file id: {file_id})")
    return file_id


def create_preprint(token, node_id, file_id):
    """Create the preprint record pointing to the uploaded file."""
    payload = {
        "data": {
            "type": "preprints",
            "attributes": {
                "title": TITLE,
                "description": ABSTRACT,
                "is_published": True,
                "subjects": [],
                "tags": TAGS,
            },
            "relationships": {
                "node": {
                    "data": {"type": "nodes", "id": node_id}
                },
                "primary_file": {
                    "data": {"type": "files", "id": file_id}
                },
                "provider": {
                    "data": {"type": "preprint-providers", "id": PROVIDER}
                },
                "license": {
                    "data": {"type": "licenses", "id": LICENSE_ID}
                }
            }
        }
    }
    r = requests.post(f"{OSF_API}/preprints/", headers=headers(token), json=payload)
    if not r.ok:
        print(f"  ERROR creating preprint: {r.status_code} {r.text[:400]}")
        r.raise_for_status()
    data = r.json()["data"]
    preprint_id = data["id"]
    url = data["links"].get("html") or f"https://osf.io/{preprint_id}"
    return preprint_id, url


def add_contributors(token, preprint_id):
    """Add author metadata to the preprint."""
    for author in AUTHORS:
        # Search for OSF user by email
        r = requests.get(
            f"{OSF_API}/users/?filter[username]={author['email']}",
            headers=headers(token)
        )
        if r.ok and r.json()["data"]:
            user_id = r.json()["data"][0]["id"]
            payload = {
                "data": {
                    "type": "contributors",
                    "attributes": {"bibliographic": True},
                    "relationships": {
                        "users": {"data": {"type": "users", "id": user_id}}
                    }
                }
            }
            requests.post(
                f"{OSF_API}/preprints/{preprint_id}/contributors/",
                headers=headers(token), json=payload
            )
            print(f"  Added contributor: {author['name']}")


def make_node_public(token, node_id):
    payload = {
        "data": {
            "id": node_id,
            "type": "nodes",
            "attributes": {"public": True}
        }
    }
    r = requests.patch(f"{OSF_API}/nodes/{node_id}/", headers=headers(token), json=payload)
    if r.ok:
        print(f"  Node set to public.")


def main():
    parser = argparse.ArgumentParser(description="Submit PenuX protocol to OSF Preprints")
    parser.add_argument("--token", help="OSF personal access token (or set OSF_TOKEN env var)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without submitting")
    args = parser.parse_args()

    if args.token:
        os.environ["OSF_TOKEN"] = args.token
    token = get_token()

    if not os.path.exists(PDF_PATH):
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN — would submit:")
        print(f"  Title: {TITLE[:80]}...")
        print(f"  PDF: {PDF_PATH}")
        print(f"  Tags: {', '.join(TAGS[:5])}...")
        print("  Token: ****" + token[-4:])
        return

    print("Submitting to OSF Preprints...")

    print("1/4 Creating OSF project node...")
    node_id = create_osf_node(token)

    print("2/4 Uploading PDF...")
    file_id = upload_pdf(token, node_id, PDF_PATH)

    print("3/4 Creating preprint...")
    preprint_id, url = create_preprint(token, node_id, file_id)

    print("4/4 Finalising (contributors + public)...")
    add_contributors(token, preprint_id)
    make_node_public(token, node_id)

    print()
    print("=" * 60)
    print(f"SUBMITTED SUCCESSFULLY")
    print(f"Preprint URL : {url}")
    print(f"Preprint ID  : {preprint_id}")
    print(f"Node ID      : {node_id}")
    print()
    print("Google Scholar typically indexes new OSF preprints within 1-7 days.")
    print("To verify: https://scholar.google.com -> search 'PenuX SAP Stage I'")
    print("=" * 60)

    # Save the result
    result_path = os.path.join(os.path.dirname(__file__), "osf_submission_result.json")
    with open(result_path, "w") as f:
        json.dump({
            "preprint_id": preprint_id,
            "url": url,
            "node_id": node_id,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }, f, indent=2)
    print(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
