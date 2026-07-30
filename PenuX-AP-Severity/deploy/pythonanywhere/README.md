# Deploying PenuX-AP-Severity to PythonAnywhere

I don't have a PythonAnywhere account myself, so I can't create or configure
the service directly — these are the exact steps for you to run it there
as an alternative to Render.

PythonAnywhere's free tier only hosts **WSGI** apps (no native ASGI, no
Docker, no Procfile). FastAPI is ASGI-only, so `wsgi.py` in this folder
wraps it with [a2wsgi](https://pypi.org/project/a2wsgi/)'s `ASGIMiddleware`
to make it servable as a plain WSGI callable — this is the standard way to
run FastAPI on WSGI-only hosts.

## Steps

1. **Sign up** at pythonanywhere.com (free "Beginner" account is enough to try this).

2. **Open a Bash console** (Consoles tab -> Bash) and clone the minimal deploy branch:
   ```bash
   git clone --branch render-ap-severity-deploy --single-branch \
     https://github.com/netanelcyber/penuX.git penux-api
   ```
   This branch is a lightweight mirror (~340KB) with just the files needed
   to run the API — same content that's deployed to Render, kept in sync
   from `PenuX-AP-Severity/` on `claude/pensive-pascal-a0l7a8`.

3. **Create a virtualenv and install dependencies**:
   ```bash
   mkvirtualenv --python=/usr/bin/python3.11 penux-api-venv
   pip install -r ~/penux-api/requirements.txt a2wsgi
   ```

4. **Web tab** -> "Add a new web app" -> choose **Manual configuration**
   (not the Flask/Django wizard) -> pick a Python version matching your venv.

5. Under **Virtualenv** on the Web tab, set the path to the venv you created
   (e.g. `/home/yourusername/.virtualenvs/penux-api-venv`).

6. Under **Code**, click the **WSGI configuration file** link — it opens an
   editor for a file PythonAnywhere generated. **Replace its entire contents**
   with:
   ```python
   import sys
   sys.path.insert(0, "/home/yourusername/penux-api")
   from deploy.pythonanywhere.wsgi import application
   ```
   (adjust the path to wherever you cloned the repo in step 2).

7. In `penux-api/deploy/pythonanywhere/wsgi.py`, edit `PROJECT_HOME` near the
   top to match the same path.

8. Click **Reload** on the Web tab. Your API should now be live at
   `https://yourusername.pythonanywhere.com` — try `/docs` and `/health`
   first.

## Known limitations vs. Render

- Free PythonAnywhere apps sleep after 3 months of no login (not per-request
  like Render's free tier) but also throttle outbound internet access unless
  you're on a paid plan — this doesn't matter here since the API makes no
  outbound calls itself.
- No custom domain support on the free tier (`api.penux.uk` would need a
  paid plan's "Custom domain" feature, priced separately from hosting).
- Updating the deployed code means re-running `git pull` in the PythonAnywhere
  console and clicking Reload — no auto-deploy-on-push like Render.
