#!/usr/bin/env python3
"""
Cloudflare DDNS updater for mail.penux.uk.
Runs every 5 min via systemd timer.
Reads CF_TOKEN from environment (EnvironmentFile=/etc/penux-imap/imap.env).
"""
from __future__ import annotations
import json, logging, os, sys, urllib.request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s penux-ddns %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

CF_TOKEN  = os.environ.get("CF_TOKEN", "")
DOMAIN    = "penux.uk"
HOSTNAME  = "mail.penux.uk"
CACHE_FILE = "/var/cache/penux-ddns/last_ip"
CF_BASE   = "https://api.cloudflare.com/client/v4"


def _public_ip() -> str:
    for url in [
        "https://ifconfig.me",
        "https://ipv4.icanhazip.com",
        "https://api4.my-ip.io/ip",
    ]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "penux-ddns/1.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                ip = r.read().decode().strip()
                if ip:
                    return ip
        except Exception:
            continue
    raise RuntimeError("Cannot determine public IP from any source")


def _cf(method: str, path: str, body: dict | None = None) -> dict:
    url  = CF_BASE + path
    data = json.dumps(body).encode() if body else None
    req  = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "Authorization": f"Bearer {CF_TOKEN}",
            "Content-Type":  "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def _zone_id() -> str:
    d = _cf("GET", f"/zones?name={DOMAIN}")
    if not d.get("success") or not d["result"]:
        raise RuntimeError(f"Cloudflare zone not found for {DOMAIN}")
    return d["result"][0]["id"]


def _a_record(zone_id: str) -> dict | None:
    d = _cf("GET", f"/zones/{zone_id}/dns_records?type=A&name={HOSTNAME}")
    return d["result"][0] if d.get("result") else None


def _upsert(zone_id: str, record: dict | None, ip: str) -> None:
    payload = {"type": "A", "name": HOSTNAME, "content": ip, "proxied": False, "ttl": 120}
    if record:
        _cf("PUT", f"/zones/{zone_id}/dns_records/{record['id']}", payload)
    else:
        _cf("POST", f"/zones/{zone_id}/dns_records", payload)


def _read_cache() -> str:
    try:
        return open(CACHE_FILE).read().strip()
    except OSError:
        return ""


def _write_cache(ip: str) -> None:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        f.write(ip)


def main() -> None:
    if not CF_TOKEN:
        log.error("CF_TOKEN not set — cannot update DNS")
        sys.exit(1)

    current_ip = _public_ip()
    cached_ip  = _read_cache()

    if current_ip == cached_ip:
        log.debug("IP unchanged (%s) — nothing to do", current_ip)
        return

    log.info("IP changed: %s → %s — updating Cloudflare", cached_ip or "?", current_ip)

    zone_id  = _zone_id()
    record   = _a_record(zone_id)
    cf_ip    = record["content"] if record else None

    if cf_ip == current_ip:
        log.info("Cloudflare already has %s — updating local cache only", current_ip)
    else:
        _upsert(zone_id, record, current_ip)
        log.info("DNS updated: %s → %s", HOSTNAME, current_ip)

    _write_cache(current_ip)


if __name__ == "__main__":
    main()
