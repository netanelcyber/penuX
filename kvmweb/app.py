#!/usr/bin/env python3
"""
kvmweb - lightweight KVM web management interface
Requires: flask, python3-libvirt, python3-pam
"""

import os
import sys
import json
import time
import logging
import functools
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime

import libvirt
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, jsonify, flash, Response
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SECRET_KEY   = os.environ.get("KVMWEB_SECRET", os.urandom(32))
LIBVIRT_URI  = os.environ.get("LIBVIRT_URI", "qemu:///system")
HOST_LABEL   = os.environ.get("KVMWEB_HOST", "KVM Host")
DEBUG        = os.environ.get("KVMWEB_DEBUG", "0") == "1"
SESSION_MINS = int(os.environ.get("KVMWEB_SESSION_MINS", "30"))

app = Flask(__name__)
app.secret_key = SECRET_KEY

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("kvmweb")

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def check_password(username: str, password: str) -> bool:
    """Authenticate via PAM (system accounts) or a password file."""
    # Try PAM first (requires python3-pam)
    try:
        import pam
        p = pam.pam()
        return p.authenticate(username, password, service="login")
    except Exception:
        pass

    # Fallback: /etc/kvmweb/users  (format: user:bcrypt_hash per line)
    passfile = "/etc/kvmweb/users"
    if os.path.isfile(passfile):
        try:
            import bcrypt
            with open(passfile) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(":", 1)
                    if len(parts) == 2 and parts[0] == username:
                        return bcrypt.checkpw(
                            password.encode(), parts[1].encode()
                        )
        except Exception as e:
            log.error("Password file error: %s", e)
    return False


def login_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        # Session timeout
        last = session.get("last_active", 0)
        if time.time() - last > SESSION_MINS * 60:
            session.clear()
            flash("Session expired. Please log in again.", "warning")
            return redirect(url_for("login"))
        session["last_active"] = time.time()
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# libvirt helpers
# ---------------------------------------------------------------------------

def get_conn():
    return libvirt.open(LIBVIRT_URI)


def domain_info(dom) -> dict:
    """Return a serialisable dict for a libvirt domain."""
    state_map = {
        libvirt.VIR_DOMAIN_RUNNING:     ("running",  "success"),
        libvirt.VIR_DOMAIN_BLOCKED:     ("blocked",  "warning"),
        libvirt.VIR_DOMAIN_PAUSED:      ("paused",   "warning"),
        libvirt.VIR_DOMAIN_SHUTDOWN:    ("shutdown", "secondary"),
        libvirt.VIR_DOMAIN_SHUTOFF:     ("shutoff",  "secondary"),
        libvirt.VIR_DOMAIN_CRASHED:     ("crashed",  "danger"),
        libvirt.VIR_DOMAIN_PMSUSPENDED: ("suspended","info"),
    }
    state_code, _ = dom.state()
    state_label, state_class = state_map.get(state_code, ("unknown", "secondary"))

    info = dom.info()           # [state, maxMem, mem, nrVirtCpu, cpuTime]
    xml   = dom.XMLDesc()
    root  = ET.fromstring(xml)

    # Extract OS type and disk image
    os_type = root.findtext("os/type") or "unknown"
    disks = []
    for disk in root.findall("devices/disk[@type='file']"):
        src = disk.find("source")
        if src is not None:
            disks.append(src.get("file", ""))

    # Network interfaces
    ifaces = []
    for iface in root.findall("devices/interface"):
        mac = iface.find("mac")
        src = iface.find("source")
        if mac is not None:
            ifaces.append({
                "mac": mac.get("address", ""),
                "type": iface.get("type", ""),
                "source": src.get("network", src.get("bridge", "")) if src is not None else "",
            })

    return {
        "name":        dom.name(),
        "uuid":        dom.UUIDString(),
        "state":       state_label,
        "state_class": state_class,
        "max_mem_mb":  info[1] // 1024,
        "mem_mb":      info[2] // 1024,
        "vcpus":       info[3],
        "os_type":     os_type,
        "disks":       disks,
        "ifaces":      ifaces,
        "autostart":   bool(dom.autostart()),
        "persistent":  bool(dom.isPersistent()),
    }


def host_info(conn) -> dict:
    """Return host-level statistics."""
    info = conn.getInfo()   # [arch, mem_mb, cpus, mhz, nodes, sockets, cores, threads]
    free = conn.getFreeMemory() // (1024 * 1024)
    hostname = conn.getHostname()
    caps_xml = conn.getCapabilities()
    active = conn.numOfDomains()
    inactive = conn.numOfDefinedDomains()
    return {
        "hostname":  hostname,
        "arch":      info[0],
        "mem_mb":    info[1],
        "free_mb":   free,
        "cpus":      info[2],
        "mhz":       info[3],
        "active_vms":   active,
        "inactive_vms": inactive,
        "libvirt_ver":  ".".join(str(x) for x in divmod(divmod(conn.getVersion(), 1000)[0], 1000) + (conn.getVersion() % 1000,)),
    }


def list_storage_pools(conn) -> list:
    pools = []
    for pool in conn.listAllStoragePools():
        try:
            info = pool.info()
            pools.append({
                "name":       pool.name(),
                "state":      ["inactive","building","running","degraded","inaccessible"][info[0]],
                "capacity_gb": round(info[1] / (1024**3), 1),
                "allocation_gb": round(info[2] / (1024**3), 1),
                "available_gb":  round(info[3] / (1024**3), 1),
                "autostart": bool(pool.autostart()),
            })
        except Exception:
            pass
    return pools


def list_networks(conn) -> list:
    nets = []
    for net in conn.listAllNetworks():
        try:
            nets.append({
                "name":    net.name(),
                "active":  net.isActive(),
                "uuid":    net.UUIDString(),
                "bridge":  net.bridgeName() if net.isActive() else "",
                "autostart": bool(net.autostart()),
            })
        except Exception:
            pass
    return nets


# ---------------------------------------------------------------------------
# Routes — Auth
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username and check_password(username, password):
            session["user"] = username
            session["last_active"] = time.time()
            log.info("Login: %s", username)
            return redirect(request.args.get("next") or url_for("index"))
        flash("Invalid username or password.", "danger")
    return render_template("login.html", host=HOST_LABEL)


@app.route("/logout")
def logout():
    user = session.pop("user", None)
    if user:
        log.info("Logout: %s", user)
    session.clear()
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Routes — Dashboard
# ---------------------------------------------------------------------------

@app.route("/")
@login_required
def index():
    try:
        conn = get_conn()
        h = host_info(conn)
        domains = [domain_info(d) for d in conn.listAllDomains()]
        domains.sort(key=lambda d: (d["state"] != "running", d["name"]))
        pools = list_storage_pools(conn)
        networks = list_networks(conn)
        conn.close()
    except Exception as e:
        log.error("libvirt error: %s", e)
        flash(f"Cannot connect to libvirt: {e}", "danger")
        h, domains, pools, networks = {}, [], [], []
    return render_template(
        "index.html",
        host=HOST_LABEL,
        user=session["user"],
        hinfo=h,
        domains=domains,
        pools=pools,
        networks=networks,
    )


# ---------------------------------------------------------------------------
# Routes — VM detail
# ---------------------------------------------------------------------------

@app.route("/vm/<name>")
@login_required
def vm_detail(name):
    try:
        conn = get_conn()
        dom  = conn.lookupByName(name)
        info = domain_info(dom)
        xml  = dom.XMLDesc()
        conn.close()
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for("index"))
    return render_template(
        "vm_detail.html",
        host=HOST_LABEL,
        user=session["user"],
        vm=info,
        xml=xml,
    )


# ---------------------------------------------------------------------------
# Routes — VM actions (POST)
# ---------------------------------------------------------------------------

@app.route("/vm/<name>/action", methods=["POST"])
@login_required
def vm_action(name):
    data = request.get_json(silent=True) or {}
    action = data.get("action") or request.form.get("action")
    allowed = {"start", "shutdown", "destroy", "reboot", "suspend", "resume",
               "autostart_on", "autostart_off", "undefine"}
    if action not in allowed:
        return jsonify({"ok": False, "error": "Unknown action"}), 400

    try:
        conn = get_conn()
        dom  = conn.lookupByName(name)
        {
            "start":        dom.create,
            "shutdown":     dom.shutdown,
            "destroy":      dom.destroy,
            "reboot":       dom.reboot,
            "suspend":      dom.suspend,
            "resume":       dom.resume,
            "autostart_on":  lambda: dom.setAutostart(1),
            "autostart_off": lambda: dom.setAutostart(0),
            "undefine":     dom.undefine,
        }[action]()
        conn.close()
        log.info("%s: %s by %s", name, action, session["user"])
        return jsonify({"ok": True})
    except Exception as e:
        log.error("%s %s error: %s", name, action, e)
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Routes — Create VM (simple)
# ---------------------------------------------------------------------------

@app.route("/vm/create", methods=["GET", "POST"])
@login_required
def vm_create():
    if request.method == "POST":
        name    = request.form.get("name", "").strip()
        ram_mb  = int(request.form.get("ram_mb", 1024))
        vcpus   = int(request.form.get("vcpus", 1))
        disk_gb = int(request.form.get("disk_gb", 10))
        iso     = request.form.get("iso", "").strip()
        network = request.form.get("network", "default")

        if not name:
            flash("VM name is required.", "danger")
            return redirect(url_for("vm_create"))

        # Build virt-install command
        cmd = [
            "virt-install",
            "--name", name,
            "--memory", str(ram_mb),
            "--vcpus", str(vcpus),
            "--disk", f"size={disk_gb},format=qcow2",
            "--network", f"network={network}",
            "--graphics", "vnc,listen=127.0.0.1",
            "--noautoconsole",
            "--os-variant", "generic",
        ]
        if iso:
            cmd += ["--cdrom", iso]
        else:
            cmd += ["--import", "--disk", f"size={disk_gb},format=qcow2"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                log.info("Created VM %s by %s", name, session["user"])
                flash(f"VM '{name}' created successfully.", "success")
                return redirect(url_for("vm_detail", name=name))
            else:
                flash(f"virt-install failed: {result.stderr}", "danger")
        except subprocess.TimeoutExpired:
            flash("VM creation timed out.", "danger")
        except FileNotFoundError:
            flash("virt-install not found. Install virtinst.", "danger")
        except Exception as e:
            flash(str(e), "danger")

    # GET — gather available ISOs and networks
    isos = []
    for d in ["/var/lib/libvirt/images", "/tmp", "/home"]:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.endswith(".iso"):
                    isos.append(os.path.join(d, f))

    networks = []
    try:
        conn = get_conn()
        networks = [n.name() for n in conn.listAllNetworks() if n.isActive()]
        conn.close()
    except Exception:
        pass

    return render_template(
        "vm_create.html",
        host=HOST_LABEL,
        user=session["user"],
        isos=isos,
        networks=networks,
    )


# ---------------------------------------------------------------------------
# Routes — API (JSON) for live stats
# ---------------------------------------------------------------------------

@app.route("/api/stats")
@login_required
def api_stats():
    try:
        conn = get_conn()
        domains = []
        for dom in conn.listAllDomains():
            try:
                state, _ = dom.state()
                info = dom.info()
                domains.append({
                    "name":  dom.name(),
                    "state": state,
                    "mem_mb": info[2] // 1024,
                    "vcpus": info[3],
                })
            except Exception:
                pass
        free = conn.getFreeMemory() // (1024 * 1024)
        conn.close()
        return jsonify({"ok": True, "domains": domains, "free_mb": free})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import ssl
    import argparse

    parser = argparse.ArgumentParser(description="kvmweb server")
    parser.add_argument("--host",    default="0.0.0.0")
    parser.add_argument("--port",    default=8080, type=int)
    parser.add_argument("--cert",    default="/etc/kvmweb/ssl/server.crt")
    parser.add_argument("--key",     default="/etc/kvmweb/ssl/server.key")
    parser.add_argument("--no-tls",  action="store_true")
    args = parser.parse_args()

    if args.no_tls or not os.path.isfile(args.cert):
        log.warning("TLS disabled or cert missing — running HTTP only")
        app.run(host=args.host, port=args.port, debug=DEBUG)
    else:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.cert, args.key)
        app.run(host=args.host, port=args.port, ssl_context=ctx, debug=DEBUG)
