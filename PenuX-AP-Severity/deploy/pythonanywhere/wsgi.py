# PythonAnywhere WSGI entry point for the PenuX-AP-Severity FastAPI app.
#
# PythonAnywhere's free tier only supports WSGI web apps (no native ASGI
# hosting like Render/Railway/Fly), so this wraps the FastAPI (ASGI) app
# with a2wsgi's ASGIMiddleware to make it servable as a plain WSGI callable.
#
# Setup on PythonAnywhere:
#   1. Open a Bash console, clone the deploy branch:
#        git clone --branch render-ap-severity-deploy --single-branch \
#          https://github.com/netanelcyber/penuX.git penux-api
#   2. In a virtualenv: pip install -r penux-api/requirements.txt a2wsgi
#   3. Web tab -> Add a new web app -> Manual configuration -> your Python version
#   4. Set the virtualenv path to the one you created in step 2
#   5. Edit the WSGI configuration file PythonAnywhere generated (path shown
#      on the Web tab, e.g. /var/www/yourusername_pythonanywhere_com_wsgi.py)
#      to import everything from THIS file — replace its entire contents with:
#        from penux_api.deploy.pythonanywhere.wsgi import application
#      (adjust the import path to wherever you cloned the repo)
#   6. Update PROJECT_HOME below to your actual clone path, then hit Reload
#      on the Web tab.

import os
import sys

# EDIT THIS — the directory you cloned the repo into (step 1 above).
PROJECT_HOME = "/home/yourusername/penux-api"

for path in (PROJECT_HOME, os.path.join(PROJECT_HOME, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

os.chdir(PROJECT_HOME)

from a2wsgi import ASGIMiddleware  # noqa: E402

from api.main import app as _fastapi_app  # noqa: E402

application = ASGIMiddleware(_fastapi_app)
