#!/usr/bin/env python3
"""
Automated Cureus submission for PenuX manuscript.
Usage: python3 submit_cureus.py --email YOUR_EMAIL --password YOUR_PASSWORD
"""

import argparse
import time
import sys
from pathlib import Path

TITLE = (
    "PenuX: A Comparative Study of 11 Machine Learning and Deep Learning Models "
    "for Early Severity Prediction of Acute Pancreatitis Using Routine Admission "
    "Laboratory Values, with FHIR R4 Integration"
)

ABSTRACT = """Background
Severe Acute Pancreatitis (SAP) carries a mortality rate of 20–30% and requires early risk stratification. Classical scoring systems (Ranson, BISAP, APACHE II) require 24–48 hours of serial laboratory observation and lack electronic health record (EHR) integration. There is an unmet need for admission-time, data-driven severity prediction that integrates with modern health information standards.

Methods
We conducted a retrospective analysis of 722 acute pancreatitis (AP) admissions (585 severe / 137 mild; Atlanta 2012 classification) from a single Chinese tertiary institution. Eleven models were trained on 106 routine admission laboratory features using 5-fold stratified cross-validation: three classical machine learning models (Logistic Regression, Random Forest, Gradient Boosting), three multilayer perceptron (MLP) deep learning models, and five long short-term memory (LSTM)-based sequence models (Vanilla LSTM, Stacked LSTM, Bidirectional LSTM, LSTM+Attention, CNN-LSTM). Optimal decision thresholds were selected by maximum F1 score on out-of-fold predictions. A client-side FHIR R4 integration was implemented for automated risk scoring at point of care.

Results
Random Forest achieved the highest discrimination (AUC=0.877, F1=0.917, sensitivity=96.8%, specificity=38.7% at threshold 0.535), missing only 19 of 585 severe cases. Gradient Boosting was comparable (AUC=0.874, sensitivity=97.1%). MLP achieved AUC=0.836. LSTM-based models achieved AUC=0.675–0.772, with CNN-LSTM performing best among recurrent architectures (AUC=0.772, sensitivity=98.6%). Key predictive features across models were calcium, D-dimer, LDH, lactate, and hematocrit. A label inversion effect was identified: mild biliary AP cases showed higher WBC, CRP, and lipase than severe necrotising AP cases, explaining sub-chance performance of heuristic scoring models on this cohort.

Conclusions
Random Forest achieves SAP triage with AUC=0.877 from a single admission blood draw, eliminating the 24–48 hour observation window required by classical scoring systems. The open-source PenuX platform provides FHIR R4, HL7 v2.x, and Israeli HIS (Camelion) integration for automated deployment. External validation is required before clinical use."""

KEYWORDS = [
    "acute pancreatitis",
    "severe acute pancreatitis",
    "machine learning",
    "deep learning",
    "random forest",
    "LSTM",
    "FHIR R4",
    "clinical prediction model",
]

MANUSCRIPT_PATH = Path(__file__).parent / "manuscript_cureus.md"


def log(msg):
    print(f"[submit] {msg}", flush=True)


def run(email: str, password: str, headless: bool = False):
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=200)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900})
        page = ctx.new_page()

        # ── 1. Sign in ──────────────────────────────────────────────────────
        log("Navigating to Cureus sign-in…")
        page.goto("https://www.cureus.com/sign_in", wait_until="networkidle")
        page.fill('input[name="user[email]"]', email)
        page.fill('input[name="user[password]"]', password)
        page.click('input[type="submit"]')
        page.wait_for_url("**/dashboard**", timeout=20000)
        log("Signed in successfully.")

        # ── 2. Start new submission ──────────────────────────────────────────
        log("Opening new article submission…")
        page.goto("https://www.cureus.com/submit", wait_until="networkidle")

        # Click "Submit New Article" if present
        try:
            page.click("text=Submit New Article", timeout=8000)
            page.wait_for_load_state("networkidle")
        except PWTimeout:
            log("'Submit New Article' button not found — may already be on form.")

        # ── 3. Article type ─────────────────────────────────────────────────
        log("Selecting article type: Original Article…")
        try:
            page.select_option("select#article_type", label="Original Article")
        except Exception:
            try:
                page.click("text=Original Article")
            except Exception:
                log("WARNING: Could not select article type — select manually.")

        # ── 4. Title ────────────────────────────────────────────────────────
        log("Entering title…")
        try:
            title_sel = 'input[name="article[title]"], textarea[name="article[title]"], #article_title'
            page.fill(title_sel, TITLE)
        except Exception:
            log("WARNING: Could not fill title field — fill manually.")

        # ── 5. Abstract ─────────────────────────────────────────────────────
        log("Entering abstract…")
        try:
            abs_sel = 'textarea[name="article[abstract]"], #article_abstract, .abstract-field'
            page.fill(abs_sel, ABSTRACT)
        except Exception:
            log("WARNING: Could not fill abstract — paste manually.")

        # ── 6. Keywords ─────────────────────────────────────────────────────
        log("Adding keywords…")
        for kw in KEYWORDS:
            try:
                kw_input = page.locator('input[placeholder*="keyword"], input[placeholder*="tag"]').first
                kw_input.fill(kw)
                kw_input.press("Enter")
                time.sleep(0.4)
            except Exception:
                log(f"  Could not add keyword '{kw}' — add manually.")
                break

        # ── 7. Manuscript body ───────────────────────────────────────────────
        log("Pasting manuscript body…")
        body_text = MANUSCRIPT_PATH.read_text(encoding="utf-8")
        try:
            body_sel = '.ql-editor, #article_body, textarea[name="article[body]"]'
            editor = page.locator(body_sel).first
            editor.click()
            editor.fill(body_text)
        except Exception:
            log("WARNING: Could not fill body — paste manuscript_cureus.md manually.")

        # ── 8. Take screenshot ──────────────────────────────────────────────
        screenshot_path = Path(__file__).parent / "submission_screenshot.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        log(f"Screenshot saved: {screenshot_path}")

        # ── 9. Pause for manual review before submitting ────────────────────
        log("")
        log("=" * 60)
        log("PAUSED — Review the form in the browser before submitting.")
        log("Press ENTER here to click Submit, or Ctrl+C to abort.")
        log("=" * 60)
        input()

        # ── 10. Submit ──────────────────────────────────────────────────────
        log("Submitting…")
        try:
            page.click('input[type="submit"][value*="Submit"], button:has-text("Submit")', timeout=10000)
            page.wait_for_load_state("networkidle", timeout=30000)
            log("Submission complete.")
            final_screenshot = Path(__file__).parent / "submission_confirmation.png"
            page.screenshot(path=str(final_screenshot), full_page=True)
            log(f"Confirmation screenshot: {final_screenshot}")
        except PWTimeout:
            log("Submit button not found — click Submit manually in the browser.")
            input("Press ENTER after you have submitted…")

        browser.close()
        log("Done.")


def main():
    ap = argparse.ArgumentParser(description="Automate Cureus manuscript submission.")
    ap.add_argument("--email", required=True, help="Cureus account email")
    ap.add_argument("--password", required=True, help="Cureus account password")
    ap.add_argument("--headless", action="store_true", help="Run browser headless (no UI)")
    args = ap.parse_args()
    run(args.email, args.password, headless=args.headless)


if __name__ == "__main__":
    main()
