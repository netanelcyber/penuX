#!/usr/bin/env python3
"""
Check which phone numbers have active WhatsApp accounts.
Opens WhatsApp Web, scans QR once, then checks each number.

Usage:
    python3 scripts/check_whatsapp.py
"""

import time
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

CONTACTS = [
    {"name": "Prof. Michael Kochman",  "phone": "+12153498354"},
    {"name": "Dr. Saurabh Chawla",     "phone": "+14047782714"},
    {"name": "Prof. Timna Naftali",    "phone": "+97235028500"},
    {"name": "Dr. Vera Dreizin",       "phone": "+972545558570"},
]

def wait(ms=800):
    time.sleep(ms / 1000)

def check_numbers(page):
    print("⏳ Waiting for WhatsApp Web to load...")
    page.wait_for_selector(
        '[data-testid="chatlist-header"], canvas',
        timeout=120_000
    )

    # QR code present → wait for user to scan
    if page.query_selector('canvas') and not page.query_selector('[data-testid="chatlist-header"]'):
        print("📱 Scan the QR code in the browser window, then press ENTER here...")
        input()
        page.wait_for_selector('[data-testid="chatlist-header"]', timeout=60_000)

    print("✅ WhatsApp ready. Checking numbers...\n")
    wait(1500)

    results = []

    for c in CONTACTS:
        phone = c["phone"]
        name  = c["name"]
        print(f"🔍 Checking {name} ({phone})...")

        # Navigate directly to the number via wa.me equivalent in Web
        # WhatsApp Web: open new chat and search the phone number
        try:
            # Click new chat button
            new_chat = page.wait_for_selector(
                '[data-testid="new-chat-btn"], [title="New chat"]',
                timeout=8_000
            )
            new_chat.click()
            wait(800)

            # Type number in search
            search = page.wait_for_selector(
                '[data-testid="chat-list-search"], [data-testid="add-participants-search-input"], input[type="text"]',
                timeout=6_000
            )
            search.fill(phone)
            wait(2000)

            # Check for "Phone number shared via url" or contact found
            page_text = page.inner_text("body")

            has_wa = False

            # If a chat result appears (profile picture, name) → has WhatsApp
            if page.query_selector('[data-testid="cell-frame-container"]'):
                has_wa = True
            # "not on WhatsApp" message
            elif any(x in page_text for x in [
                "not on WhatsApp", "אינו ב-WhatsApp", "isn't on WhatsApp",
                "No results found", "אין תוצאות"
            ]):
                has_wa = False
            else:
                # Try clicking the number-based result (phone number link)
                try:
                    phone_result = page.locator(f'text="{phone}"').first
                    if phone_result.is_visible(timeout=2000):
                        has_wa = True
                except Exception:
                    has_wa = None  # unknown

            status = "✅ Has WhatsApp" if has_wa is True else (
                     "❌ NOT on WhatsApp" if has_wa is False else
                     "❓ Unknown")

            results.append({**c, "has_whatsapp": has_wa, "status": status})
            print(f"   {status}")

            # Close search / go back
            page.keyboard.press("Escape")
            wait(600)

        except Exception as e:
            print(f"   ⚠️  Error checking {name}: {e}")
            results.append({**c, "has_whatsapp": None, "status": "⚠️ Error"})
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
            wait(800)

    # Summary
    print()
    print("=" * 55)
    print("RESULTS")
    print("=" * 55)
    for r in results:
        print(f"{r['status']:25s}  {r['name']:30s}  {r['phone']}")

    wa_contacts = [r for r in results if r["has_whatsapp"] is True]
    print()
    print(f"📱 {len(wa_contacts)}/{len(results)} numbers have WhatsApp.")
    if wa_contacts:
        print("\nGroup-eligible contacts:")
        for r in wa_contacts:
            print(f"  • {r['name']} — {r['phone']}")
    print("=" * 55)

    return results

def main():
    print("PenuX — WhatsApp Number Checker")
    print("Numbers to check:")
    for c in CONTACTS:
        print(f"  • {c['name']:30s}  {c['phone']}")
    print()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            executable_path="/opt/pw-browsers/chromium",
            headless=False,
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        ctx = browser.new_context(
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
            check_numbers(page)
        except Exception as e:
            print(f"❌ Fatal error: {e}")

        input("\nPress ENTER to close the browser...")
        browser.close()

if __name__ == "__main__":
    main()
