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

# Advisers must be filled in manually by the author (Cureus requires consent)
# Set CUREUS_ADVISER_1 through CUREUS_ADVISER_5 as env vars: "Name <email>"
ADVISERS = []
for i in range(1, 6):
    raw = os.environ.get(f"CUREUS_ADVISER_{i}", "")
    if raw:
        import re
        m = re.match(r"(.+?)\s*<(.+?)>", raw)
        if m:
            ADVISERS.append({"name": m.group(1).strip(), "email": m.group(2).strip()})

async def safe_fill(page, selectors, value, label="field"):
    for sel in selectors:
        try:
            await page.fill(sel, value, timeout=3000)
            print(f"  Filled {label}")
            return True
        except: pass
    print(f"  Could not fill {label}")
    return False

async def safe_click(page, selectors, label="button"):
    for sel in selectors:
        try:
            await page.click(sel, timeout=4000)
            print(f"  Clicked {label}")
            return True
        except: pass
    print(f"  Could not click {label}")
    return False

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

        # Step 1: Login
        print("Step 1: Logging in to Cureus...")
        await page.goto('https://www.cureus.com/sign_in', wait_until='networkidle', timeout=30000)
        await page.screenshot(path='/tmp/c1_landing.png')
        print(f"URL: {page.url}")

        await safe_fill(page, ['input[name="email"]','input[type="email"]','#user_email'], EMAIL, "email")
        await safe_fill(page, ['input[name="password"]','input[type="password"]','#user_password'], PASS, "password")
        await safe_click(page, ['input[type="submit"]','button[type="submit"]'], "login")
        await page.wait_for_load_state('networkidle', timeout=15000)
        await page.screenshot(path='/tmp/c2_after_login.png')
        print(f"After login: {page.url}")

        # Step 2: Start new article
        print("\nStep 2: Starting new submission...")
        await page.goto('https://www.cureus.com/articles/new', wait_until='networkidle', timeout=20000)
        await page.screenshot(path='/tmp/c3_new_article.png')
        print(f"New article URL: {page.url}")

        # Fill title
        await safe_fill(page, ['input[name="article[title]"]','#article_title','input[placeholder*="title" i]'], TITLE, "title")

        # Select article type = Original Article
        for sel in ['select[name="article[article_type]"]','#article_article_type']:
            try:
                await page.select_option(sel, label='Original Article', timeout=3000)
                print("  Article type: Original Article")
                break
            except:
                try:
                    await page.select_option(sel, value='original', timeout=3000)
                    print("  Article type: original")
                    break
                except: pass

        await page.screenshot(path='/tmp/c4_title_filled.png')

        # Click create
        for txt in ['Create Article','Next','Continue']:
            try:
                await page.click(f'button:has-text("{txt}"), input[value*="{txt}"]', timeout=4000)
                await page.wait_for_load_state('networkidle', timeout=15000)
                print(f"  Clicked: {txt}")
                break
            except: pass

        await page.screenshot(path='/tmp/c5_created.png')
        print(f"After create: {page.url}")

        # Step 3: Fill abstract
        print("\nStep 3: Filling abstract...")
        await safe_fill(page,
            ['textarea[name="article[abstract]"]','#article_abstract','textarea[placeholder*="abstract" i]'],
            ABSTRACT, "abstract")

        # Fill keywords
        kw_str = ", ".join(KEYWORDS)
        await safe_fill(page,
            ['input[name="article[keywords]"]','#article_keywords','input[placeholder*="keyword" i]'],
            kw_str, "keywords")

        await page.screenshot(path='/tmp/c6_abstract.png')

        # Step 4: IRB / ethics declaration
        print("\nStep 4: IRB declaration...")
        # Try to find IRB/ethics section and select Exempt
        for sel in ['select[name*="irb"]','select[name*="ethics"]','select[id*="irb"]','#article_irb_status']:
            try:
                await page.select_option(sel, label='Exempt', timeout=3000)
                print("  IRB: Exempt selected")
                break
            except:
                try:
                    await page.select_option(sel, value='exempt', timeout=3000)
                    print("  IRB: exempt selected")
                    break
                except: pass

        # Check exempt/not applicable radio/checkbox
        for sel in ['input[value="exempt"]','input[value="Exempt"]','input[value="not_applicable"]',
                    'label:has-text("Exempt") input','label:has-text("Not applicable") input']:
            try:
                await page.check(sel, timeout=2000)
                print("  IRB checkbox: checked exempt")
                break
            except: pass

        await page.screenshot(path='/tmp/c7_irb.png')

        # Step 5: Upload PDF
        print("\nStep 5: Uploading PDF...")
        file_input = await page.query_selector('input[type="file"]')
        if file_input:
            await file_input.set_input_files(PDF)
            await page.wait_for_timeout(5000)
            print("  PDF uploaded")
            await page.screenshot(path='/tmp/c8_uploaded.png')
        else:
            print("  No file input found — printing inputs:")
            inputs = await page.query_selector_all('input')
            for inp in inputs:
                t = await inp.get_attribute('type')
                n = await inp.get_attribute('name')
                print(f"    type={t} name={n}")

        # Step 6: Add advisers
        print("\nStep 6: Adding 5 advisers...")
        for i, adviser in enumerate(ADVISERS):
            print(f"  Adviser {i+1}: {adviser['name']} <{adviser['email']}>")
            # Try to find adviser search/add field
            for sel in [
                'input[placeholder*="adviser" i]',
                'input[placeholder*="advisor" i]',
                'input[placeholder*="reviewer" i]',
                f'#adviser_email_{i}',
                '.adviser-search input',
                '.advisor-search input',
            ]:
                try:
                    await page.fill(sel, adviser['email'], timeout=3000)
                    await page.wait_for_timeout(1000)
                    # Try to click suggestion or add button
                    for btn in [f'text={adviser["name"]}', '.autocomplete-result', 'button:has-text("Add")']:
                        try:
                            await page.click(btn, timeout=2000)
                            print(f"    Added via autocomplete")
                            break
                        except: pass
                    break
                except: pass

            # Alternative: "Add Adviser" button pattern
            try:
                await page.click('button:has-text("Add Adviser"), a:has-text("Add Adviser"), button:has-text("Add Advisor")', timeout=2000)
                await page.wait_for_timeout(500)
                # Fill modal
                await safe_fill(page, ['input[placeholder*="email" i]','.modal input[type="email"]'], adviser['email'], f"adviser {i+1} email")
                await safe_click(page, ['button:has-text("Search")','input[value="Search"]'], "search")
                await page.wait_for_timeout(2000)
                # Select first result
                try:
                    await page.click('.search-result:first-child, .adviser-result:first-child', timeout=3000)
                except: pass
            except: pass

            await page.wait_for_timeout(500)

        await page.screenshot(path='/tmp/c9_advisers.png')

        # Step 7: Save / Submit
        print("\nStep 7: Saving submission...")
        for txt in ['Save','Save and Continue','Submit for Review','Submit']:
            try:
                await page.click(f'button:has-text("{txt}"), input[value*="{txt}"]', timeout=4000)
                await page.wait_for_load_state('networkidle', timeout=20000)
                print(f"  Clicked: {txt}")
                await page.screenshot(path=f'/tmp/c10_after_{txt.replace(" ","_")}.png')
                break
            except: pass

        print(f"\nFinal URL: {page.url}")
        content = await page.content()
        print("=== Page excerpt ===")
        print(content[:1000])
        await browser.close()

asyncio.run(run())
