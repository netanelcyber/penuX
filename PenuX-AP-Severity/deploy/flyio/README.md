# Deploying PenuX-AP-Severity to Fly.io

I don't have a Fly.io account or the `fly` CLI available in my environment
(and this sandbox's network policy blocks pulling `python:3.11-slim` from
Docker Hub, so I couldn't fully build+run the image myself) — these are
the exact steps for you to run it there as another alternative to Render.

**What I verified**: every file path `Dockerfile.fly` copies actually
exists (`requirements.txt`, `pyproject.toml`, `README.md`, `api/`, `src/`,
the two data files) — but not a full `docker build && docker run`, since
the base image pull was blocked here. Worth a first-deploy sanity check on
your end.

## Steps

1. Install the Fly CLI: https://fly.io/docs/flyctl/install/, then `fly auth login`.

2. From `PenuX-AP-Severity/`, copy this directory's `fly.toml` up one level
   (Fly expects `fly.toml` in the directory you deploy from):
   ```bash
   cd PenuX-AP-Severity
   cp deploy/flyio/fly.toml ./fly.toml
   ```

3. Edit `fly.toml`: change `app = "penux-ap-severity"` to a globally-unique
   name (Fly app names are shared across all users), and `primary_region`
   to whichever Fly region is closest to you (list: `fly platform regions`).

4. Launch (this reads `fly.toml` + `Dockerfile.fly` and creates the app):
   ```bash
   fly launch --no-deploy --copy-config
   ```
   Say no to Postgres/Redis prompts — this app needs neither.

5. Deploy:
   ```bash
   fly deploy
   ```

6. Check it's up:
   ```bash
   curl https://<your-app-name>.fly.dev/health
   ```
   Then try `/docs` and `/models/analysis` in a browser.

## Custom domain (api.penux.uk)

```bash
fly certs add api.penux.uk
```
Fly will show you a CNAME target to point `api.penux.uk` at (similar to
the Render setup) — once that resolves, `fly certs check api.penux.uk`
confirms the TLS cert issued.

## Known differences vs. Render

- Fly's free allowance is ~3 shared-cpu-1x 256MB VMs — comparable to
  Render's free tier, but billing kicks in past that if you scale up.
- `auto_stop_machines`/`min_machines_running = 0` in `fly.toml` means it
  scales to zero on idle (like Render free tier sleeping) and cold-starts
  on the next request — same tradeoff as Render, not better or worse.
- Deploys are `fly deploy` from your machine (or CI), not auto-deploy-on-
  push unless you wire up GitHub Actions yourself with a Fly API token.
