import asyncio, os
from playwright.async_api import async_playwright

EMAIL = os.environ["MEDRXIV_EMAIL"]
PASS  = os.environ["MEDRXIV_PASS"]
PDF   = "outreach/sap_manuscript.pdf"

TITLE = "Comparative Evaluation of Machine Learning and Deep Learning Models for Early Prediction of Severe Acute Pancreatitis: A Multi-Model Study Using the 2012 Revised Atlanta Classification"

ABSTRACT = """Background: Severe acute pancreatitis (SAP) affects 10-20% of acute pancreatitis (AP) patients and carries mortality rates of 20-40%. Current scoring systems (BISAP, APACHE II, Ranson, MCTSI) require 24-48 hours of observation. Machine learning (ML) approaches using routine admission data may enable earlier prediction.

Methods: We evaluated 11 models across three families—classical ML (Logistic Regression, Random Forest, Gradient Boosting), feedforward deep learning (MLP, Residual MLP, Attention MLP), and recurrent deep learning (LSTM variants)—on a publicly available Chinese AP cohort (n=722; 585 severe, 137 mild) labeled per the 2012 Revised Atlanta Classification, using 5-fold stratified cross-validation.

Results: Random Forest achieved the best AUC of 0.877 (F1=0.917, sensitivity=96.8%, PPV=87.1%), followed by Gradient Boosting (AUC=0.874). Classical ML consistently outperformed deep learning. CNN-LSTM was the best recurrent model (AUC=0.777) but remained inferior to classical approaches.

Conclusions: Random Forest provides robust early SAP prediction from routine admission data. External prospective validation is required before clinical use."""

KEYWORDS = ["acute pancreatitis", "machine learning", "severity prediction", "random forest", "deep learning"]

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

        # Step 1: Go to Cureus
        print("Step 1: Loading Cureus...")
        await page.goto('https://www.cureus.com/sign_in', wait_until='networkidle', timeout=30000)
        await page.screenshot(path='/tmp/c1_landing.png')
        print(f"URL: {page.url}")

        # Step 2: Login
        print("\nStep 2: Logging in...")
        try:
            await page.fill('input[name="email"], input[type="email"], #user_email', EMAIL, timeout=5000)
            await page.fill('input[name="password"], input[type="password"], #user_password', PASS, timeout=5000)
            await page.click('input[type="submit"], button[type="submit"]', timeout=5000)
            await page.wait_for_load_state('networkidle', timeout=15000)
            await page.screenshot(path='/tmp/c2_after_login.png')
            print(f"After login: {page.url}")
        except Exception as e:
            print(f"Login error: {e}")
            # Try registration
            print("Trying registration...")
            await page.goto('https://www.cureus.com/sign_up', wait_until='networkidle', timeout=20000)
            await page.screenshot(path='/tmp/c2_register.png')
            print(f"Register URL: {page.url}")
            # Fill registration
            for sel in ['input[name="user[first_name]"]', 'input[placeholder*="first" i]', '#user_first_name']:
                try:
                    await page.fill(sel, 'Netanel', timeout=3000)
                    print("First name filled")
                    break
                except: pass
            for sel in ['input[name="user[last_name]"]', 'input[placeholder*="last" i]', '#user_last_name']:
                try:
                    await page.fill(sel, 'Stern', timeout=3000)
                    print("Last name filled")
                    break
                except: pass
            for sel in ['input[name="user[email]"]', 'input[type="email"]', '#user_email']:
                try:
                    await page.fill(sel, EMAIL, timeout=3000)
                    print("Email filled")
                    break
                except: pass
            for sel in ['input[name="user[password]"]', 'input[type="password"]', '#user_password']:
                try:
                    await page.fill(sel, PASS, timeout=3000)
                    print("Password filled")
                    break
                except: pass
            await page.screenshot(path='/tmp/c2b_register_filled.png')
            try:
                await page.click('input[type="submit"], button[type="submit"]', timeout=5000)
                await page.wait_for_load_state('networkidle', timeout=15000)
            except Exception as e2:
                print(f"Register submit: {e2}")

        await page.screenshot(path='/tmp/c3_logged_in.png')
        print(f"Current URL: {page.url}")

        # Step 3: Start new submission
        print("\nStep 3: Starting submission...")
        await page.goto('https://www.cureus.com/articles/new', wait_until='networkidle', timeout=20000)
        await page.screenshot(path='/tmp/c4_new_article.png')
        print(f"New article URL: {page.url}")

        # Fill title
        for sel in ['input[name="article[title]"]', '#article_title', 'input[placeholder*="title" i]']:
            try:
                await page.fill(sel, TITLE, timeout=5000)
                print("Title filled")
                break
            except: pass

        # Select article type
        for sel in ['select[name="article[article_type]"]', '#article_article_type']:
            try:
                await page.select_option(sel, label='Original Article', timeout=3000)
                print("Article type: Original Article")
                break
            except:
                try:
                    await page.select_option(sel, value='original', timeout=3000)
                    break
                except: pass

        await page.screenshot(path='/tmp/c5_title_filled.png')

        # Click next / create
        for txt in ['Create Article', 'Next', 'Continue', 'Submit']:
            try:
                await page.click(f'button:has-text("{txt}"), input[value*="{txt}"]', timeout=4000)
                await page.wait_for_load_state('networkidle', timeout=15000)
                print(f"Clicked: {txt}")
                break
            except: pass

        await page.screenshot(path='/tmp/c6_created.png')
        print(f"After create: {page.url}")

        # Step 4: Upload file
        print("\nStep 4: Uploading PDF...")
        file_input = await page.query_selector('input[type="file"]')
        if file_input:
            await file_input.set_input_files(PDF)
            await page.wait_for_timeout(5000)
            await page.screenshot(path='/tmp/c7_uploaded.png')
            print("PDF uploaded")
        else:
            print("No file input — printing page elements:")
            content = await page.content()
            print(content[:2000])

        print(f"\nFinal URL: {page.url}")
        await browser.close()

asyncio.run(run())
