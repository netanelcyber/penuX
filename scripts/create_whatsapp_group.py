#!/usr/bin/env python3
"""
Create a WhatsApp group with PenuX collaboration contacts.
Run once — scans QR code, then creates the group automatically.

Usage:
    pip install playwright
    python3 create_whatsapp_group.py
"""

import time, sys
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

CONTACTS = [
    {"name": "Prof. Michael Kochman",  "phone": "+12153498354"},
    {"name": "Dr. Saurabh Chawla",     "phone": "+14047782714"},
    {"name": "Prof. Timna Naftali",    "phone": "+97235028500"},
    {"name": "Dr. Vera Dreizin",       "phone": "+972545558570"},
]

GROUP_NAME    = "PenuX Collaboration"
GROUP_MESSAGE = (
    "Hello everyone 👋\n"
    "This group brings together the clinical collaborators of the PenuX project — "
    "an AI tool for early prediction of severe acute pancreatitis.\n\n"
    "We are currently at Stage 0: building hospital data partnerships.\n"
    "Looking forward to working with you all!\n\n"
    "— Netanel Stern | penux.uk | +972-55-970-8708"
)

def wait(ms=800):
    time.sleep(ms / 1000)

def create_group(page):
    print("⏳ Waiting for WhatsApp to load...")

    # Wait for QR scan / main page
    page.wait_for_selector('[data-testid="chatlist-header"], [data-testid="qrcode"]', timeout=120_000)

    if page.query_selector('[data-testid="qrcode"]'):
        print("📱 Scan the QR code in the browser window, then press ENTER here...")
        input()
        page.wait_for_selector('[data-testid="chatlist-header"]', timeout=60_000)

    print("✅ WhatsApp ready.")
    wait(1500)

    # Click new chat / menu button
    try:
        # Click the three-dot menu → New group
        menu = page.wait_for_selector('[data-testid="menu"], [title="Menu"]', timeout=10_000)
        menu.click()
        wait(600)
        page.get_by_text("New group").click()
    except PWTimeout:
        # Alternative: click pencil/new chat icon
        page.wait_for_selector('[data-testid="new-chat-btn"]', timeout=10_000).click()
        wait(600)
        page.get_by_text("New group").click()

    wait(800)
    print("👥 Adding contacts to group...")

    # Add contacts one by one
    search_box = page.wait_for_selector('[data-testid="add-participants-search-input"], input[type="text"]', timeout=10_000)

    added = 0
    for c in CONTACTS:
        print(f"   Adding {c['name']} ({c['phone']})...")
        search_box.click()
        search_box.fill(c['phone'])
        wait(1500)

        # Try to click the first search result
        try:
            result = page.wait_for_selector('[data-testid="cell-frame-container"]', timeout=5_000)
            result.click()
            wait(600)
            search_box.fill("")
            added += 1
            print(f"   ✓ Added {c['name']}")
        except PWTimeout:
            print(f"   ⚠️  Could not find {c['name']} ({c['phone']}) — skipping")
            search_box.fill("")
            wait(400)

    if added == 0:
        print("⚠️  No contacts added — make sure they are in your WhatsApp contacts first.")
        print("   Add them manually and re-run, or create the group manually.")
        return

    # Click Next / arrow
    wait(600)
    try:
        page.wait_for_selector('[data-testid="arrow-forward"]', timeout=5_000).click()
    except PWTimeout:
        page.keyboard.press("Enter")
    wait(1000)

    # Set group name
    print(f"✏️  Setting group name: {GROUP_NAME}")
    name_input = page.wait_for_selector(
        '[data-testid="group-name-input"], input[placeholder*="group" i], input[placeholder*="שם" ]',
        timeout=8_000
    )
    name_input.click()
    name_input.fill(GROUP_NAME)
    wait(600)

    # Click Create / checkmark
    try:
        page.wait_for_selector('[data-testid="checkmark"]', timeout=5_000).click()
    except PWTimeout:
        try:
            page.get_by_role("button", name="Create").click()
        except Exception:
            page.keyboard.press("Enter")
    wait(2000)

    print(f"🎉 Group '{GROUP_NAME}' created with {added} contacts!")

    # Send opening message
    print("💬 Sending opening message...")
    try:
        msg_box = page.wait_for_selector(
            '[data-testid="conversation-compose-box-input"]', timeout=8_000
        )
        msg_box.click()
        # Type with newlines
        for line in GROUP_MESSAGE.split('\n'):
            msg_box.type(line)
            msg_box.press('Shift+Enter')
        wait(400)
        page.keyboard.press('Enter')
        wait(1000)
        print("✅ Opening message sent.")
    except Exception as e:
        print(f"⚠️  Could not auto-send message: {e}")
        print("   Copy and paste this manually:")
        print(GROUP_MESSAGE)

    print()
    print("=" * 55)
    print(f"✅ DONE — WhatsApp group '{GROUP_NAME}' is live.")
    print("=" * 55)

def main():
    print("PenuX WhatsApp Group Creator")
    print("Contacts to add:")
    for c in CONTACTS:
        print(f"  • {c['name']:30s} {c['phone']}")
    print()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path="/opt/pw-browsers/chromium",
            headless=False,
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        ctx  = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )
        page = ctx.new_page()
        page.goto("https://web.whatsapp.com")

        try:
            create_group(page)
        except Exception as e:
            print(f"❌ Error: {e}")
            print("WhatsApp Web's UI may have changed. Try running manually.")

        input("Press ENTER to close the browser...")
        browser.close()

if __name__ == "__main__":
    main()
