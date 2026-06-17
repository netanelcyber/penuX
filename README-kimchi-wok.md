# Kimchi + Wok on Modern Debian/Ubuntu (Python 3.9+)

A zero-manual-GUI installation of the Wok web framework with the Kimchi
KVM/QEMU management plugin, ported to run on Python 3.9+ modern Linux hosts.

---

## Supported OS Versions

| OS | Version | Python | Status |
|----|---------|--------|--------|
| Ubuntu | 22.04 LTS (Jammy) | 3.10 | ✅ Tested |
| Ubuntu | 24.04 LTS (Noble) | 3.12 | ✅ Tested |
| Debian | 12 (Bookworm)     | 3.11 | ✅ Tested |
| Debian | 11 (Bullseye)     | 3.9  | ✅ Tested |

> **Not supported:** Ubuntu 18.04/20.04 with Python 2 defaults,
> or any host that does not support KVM (`/dev/kvm` must exist).

---

## Prerequisites

- A bare-metal or nested-virt server with KVM support:
  ```bash
  egrep -c '(vmx|svm)' /proc/cpuinfo   # must be > 0
  ls /dev/kvm                           # must exist
  ```
- Root / sudo access.
- Internet access to clone from GitHub (≈ 50 MB each for Wok and Kimchi).

---

## Required System Packages (auto-installed by the script)

| Package | Purpose |
|---------|---------|
| `qemu-kvm` | KVM hypervisor |
| `libvirt-daemon-system` | libvirt daemon |
| `libvirt-clients` | `virsh` CLI |
| `bridge-utils` | VM network bridges |
| `virtinst` | `virt-install` helper |
| `python3-libvirt` | Python libvirt bindings |
| `python3-lxml` | XML processing |
| `python3-m2crypto` | TLS / crypto (replaces `python-m2crypto`) |
| `python3-magic` | File-type detection |
| `python3-psutil` | System metrics |
| `python3-pam` | PAM authentication |
| `python3-ldap` | LDAP auth (optional) |
| `CherryPy >= 18` | Web server (via pip, replaces `python-cherrypy3`) |
| `Cheetah3` | Template engine (via pip, replaces `python-cheetah`) |

---

## Install Command

```bash
wget -O install-kimchi-wok-py39.sh \
  https://raw.githubusercontent.com/netanelcyber/penuX/main/install-kimchi-wok-py39.sh
sudo bash install-kimchi-wok-py39.sh
```

Or if you have this repository cloned:

```bash
sudo bash /path/to/penuX/install-kimchi-wok-py39.sh
```

The script takes 3–8 minutes depending on network speed. It will:

1. Install all system packages.
2. Clone Wok and Kimchi from GitHub to `/opt/wok` and `/opt/kimchi`.
3. Apply Python 3 compatibility patches in-place.
4. Build and install both projects.
5. Generate a self-signed TLS certificate at `/etc/wok/ssl/`.
6. Write `/etc/wok/wok.conf`.
7. Create and enable the `wokd` systemd unit.
8. Add you to the `libvirt` and `kvm` groups.

---

## Login URL

```
https://SERVER-IP:8001
```

Replace `SERVER-IP` with your server's IP address. The script prints
the exact URL at the end.

**Credentials:** your Linux system username and password (PAM auth).

> The TLS certificate is self-signed. Accept the browser security warning
> or import `wok.crt` into your browser/OS trust store.

---

## Firewall Recommendation

Port 8001 should **not** be open to the public internet. Restrict access:

```bash
# Allow only your workstation IP
sudo ufw allow from 192.168.1.0/24 to any port 8001 proto tcp
sudo ufw enable
```

Or use an SSH tunnel:

```bash
# On your workstation
ssh -L 8001:localhost:8001 user@server-ip
# Then open https://localhost:8001
```

---

## Post-Install Verification

Run these checks immediately after installation:

```bash
# 1. Service is running
systemctl status wokd --no-pager

# 2. Port is listening
ss -tulpn | grep 8001

# 3. libvirt is running
systemctl status libvirtd --no-pager
virsh list --all

# 4. User is in the right groups
groups

# 5. Python 3 imports OK
python3 -c "import cherrypy; import libvirt; import M2Crypto; print('OK')"
```

---

## Troubleshooting

### wokd fails to start

```bash
journalctl -u wokd -xe --no-pager
systemctl status wokd --no-pager
```

**Common issues:**

| Error | Fix |
|-------|-----|
| `ImportError: No module named 'cherrypy'` | `pip3 install CherryPy` |
| `ImportError: No module named 'M2Crypto'` | `apt install python3-m2crypto` |
| `ImportError: No module named 'Cheetah'` | `pip3 install Cheetah3` |
| `ImportError: No module named 'libvirt'` | `apt install python3-libvirt` |
| `Address already in use` on port 8001 | `fuser -k 8001/tcp` then restart |
| `Permission denied` on `/dev/kvm` | `usermod -aG kvm $USER && newgrp kvm` |

### Wok loads but Kimchi plugin is missing

```bash
ls -la /usr/share/wok/plugins/
ls /opt/kimchi/src/kimchi/
```

If the symlink is broken, re-link:

```bash
sudo ln -sfn /opt/kimchi/src /usr/share/wok/plugins/kimchi
sudo systemctl restart wokd
```

### Template compilation errors (Cheetah)

```bash
cd /opt/kimchi
find . -name "*.tmpl" -exec python3 -m Cheetah compile --nobackup {} \;
sudo systemctl restart wokd
```

### libvirt connection refused

```bash
sudo systemctl restart libvirtd
sudo virsh -c qemu:///system list --all
```

### CherryPy 18.x API errors

If you see `AttributeError: module 'cherrypy' has no attribute 'quickstart'`,
the Wok server.py needs a manual CherryPy 18 API update. See
`wok-kimchi-py3.patch` — Patch Set 5.

---

## Manual Update

To pull latest source and re-apply patches without full reinstall:

```bash
sudo git -C /opt/wok pull
sudo git -C /opt/kimchi pull
sudo bash install-kimchi-wok-py39.sh   # idempotent — safe to re-run
```

---

## Rollback / Uninstall

```bash
sudo bash uninstall-kimchi-wok.sh
```

Or manually:

```bash
# Stop and remove service
sudo systemctl stop wokd
sudo systemctl disable wokd
sudo rm -f /etc/systemd/system/wokd.service
sudo systemctl daemon-reload

# Remove source and config
sudo rm -rf /opt/wok /opt/kimchi
sudo rm -rf /etc/wok /var/log/wok /var/run/wok

# Remove launcher (if created by script)
sudo rm -f /usr/local/bin/wokd

# Optionally remove system packages (careful — may affect other tools)
# sudo apt remove --purge qemu-kvm libvirt-daemon-system
```

---

## Final Test Checklist

- [ ] `systemctl status wokd` shows **active (running)**
- [ ] `ss -tulpn | grep 8001` shows port listening
- [ ] Browser opens `https://SERVER-IP:8001` without connection error
- [ ] Login with Linux credentials succeeds
- [ ] Kimchi dashboard loads (VM list visible)
- [ ] `virsh list --all` returns output (no permission errors)
- [ ] Creating a new VM template does not produce a Python traceback
- [ ] `journalctl -u wokd -xe` shows no `ImportError` or `SyntaxError`
- [ ] `groups` output includes `libvirt` and `kvm` for your user

---

## Architecture Overview

```
Browser (HTTPS :8001)
       │
       ▼
   [CherryPy 18 WSGI server]  ← wokd systemd unit
       │
       ▼
   [Wok framework]  /opt/wok
       │  plugin discovery
       ▼
   [Kimchi plugin]  /opt/kimchi → /usr/share/wok/plugins/kimchi
       │
       ▼
   [python3-libvirt]
       │
       ▼
   [libvirtd]
       │
       ▼
   [QEMU/KVM]  ← /dev/kvm
```

---

## Patch Summary

See `wok-kimchi-py3.patch` for the full diff. Key changes:

| Change | Reason |
|--------|--------|
| `#!/usr/bin/env python` → `python3` | Python 3 shebang |
| `ConfigParser` → `configparser` | Renamed in Python 3 |
| `urllib2` → `urllib.request` | Renamed in Python 3 |
| `urlparse` → `urllib.parse` | Renamed in Python 3 |
| `httplib` → `http.client` | Renamed in Python 3 |
| `StringIO` → `io.StringIO` | Merged into `io` in Python 3 |
| `python-m2crypto` → `python3-m2crypto` | Debian package rename |
| `python-cherrypy3` → `CherryPy` | PyPI name for Python 3 port |
| `python-cheetah` → `Cheetah3` | PyPI name for Python 3 port |

---

## References

- Wok source: https://github.com/kimchi-project/wok
- Kimchi source: https://github.com/kimchi-project/kimchi
- libvirt Python bindings: https://libvirt.org/python.html
- CherryPy docs: https://docs.cherrypy.dev/
- Cheetah3 docs: https://cheetahtemplate.org/
