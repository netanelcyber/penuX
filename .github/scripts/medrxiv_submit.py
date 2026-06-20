import asyncio, os
from playwright.async_api import async_playwright

EMAIL = os.environ["MEDRXIV_EMAIL"]
PASS  = os.environ["MEDRXIV_PASS"]
PDF   = "outreach/sap_manuscript.pdf"

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox','--disable-dev-shm-usage','--disable-gpu']
        )
        ctx = await browser.new_context(
            viewport={'width':1280,'height':900},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        page = await ctx.new_page()

        print("Step 1: Loading medRxiv submission portal...")
        await page.goto('https://submit.medrxiv.org/', wait_until='networkidle', timeout=30000)
        await page.screenshot(path='/tmp/s1_landing.png')
        print(f"URL: {page.url}")
        content = await page.content()
        print(content[:1000])

        # Try login
        print("\nStep 2: Looking for login...")
        try:
            await page.click('text=Log In', timeout=5000)
            await page.wait_for_load_state('networkidle', timeout=10000)
        except:
            try:
                await page.click('a:has-text("Login")', timeout=3000)
                await page.wait_for_load_state('networkidle', timeout=10000)
            except:
                print("No login button found")

        await page.screenshot(path='/tmp/s2_login.png')
        print(f"After login click: {page.url}")

        # Fill login form
        for sel in ['input[name="email"]', 'input[type="email"]', '#email', 'input[name="username"]']:
            try:
                await page.fill(sel, EMAIL, timeout=3000)
                print(f"Email filled via {sel}")
                break
            except: pass

        for sel in ['input[name="password"]', 'input[type="password"]', '#password']:
            try:
                await page.fill(sel, PASS, timeout=3000)
                print("Password filled")
                break
            except: pass

        try:
            await page.click('button[type="submit"], input[type="submit"], button:has-text("Log In")', timeout=5000)
            await page.wait_for_load_state('networkidle', timeout=15000)
        except Exception as e:
            print(f"Submit click: {e}")

        await page.screenshot(path='/tmp/s3_after_login.png')
        print(f"After login: {page.url}")

        # New submission
        print("\nStep 3: Starting new submission...")
        for txt in ['Submit New Preprint', 'New Submission', 'Submit a Preprint', 'Submit']:
            try:
                await page.click(f'text={txt}', timeout=4000)
                await page.wait_for_load_state('networkidle', timeout=10000)
                print(f"Clicked: {txt}")
                break
            except: pass

        await page.screenshot(path='/tmp/s4_submission.png')
        print(f"Submission URL: {page.url}")

        # Upload PDF
        print("\nStep 4: Uploading PDF...")
        file_input = await page.query_selector('input[type="file"]')
        if file_input:
            await file_input.set_input_files(PDF)
            await page.wait_for_timeout(3000)
            print("PDF uploaded")
        else:
            print("No file input found - printing page structure:")
            inputs = await page.query_selector_all('input, button, a')
            for el in inputs[:20]:
                tag = await el.evaluate('el => el.tagName')
                txt = await el.inner_text() if tag in ['BUTTON','A'] else ''
                t = await el.get_attribute('type') or ''
                n = await el.get_attribute('name') or ''
                print(f"  {tag} type={t} name={n} text={txt[:50]}")

        await page.screenshot(path='/tmp/s5_final.png')
        await browser.close()

asyncio.run(run())
