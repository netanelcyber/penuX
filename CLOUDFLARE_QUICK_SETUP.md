# ⚡ Cloudflare DNS Quick Setup Card

**Print this page or keep it open while configuring Cloudflare**

---

## 🔴 CRITICAL: Before You Start

⚠️ **Have these ready:**
- [ ] Cloudflare account login
- [ ] Your public IP address (from `curl ifconfig.me`)
- [ ] GitHub Pages domain (yourusername.github.io)
- [ ] Vercel API domain (your-api.vercel.app)
- [ ] Access to your domain registrar

---

## 📝 Your Information

Fill in your details here:

```
Domain:                penux.uk
Public IP:             _________________ (from curl ifconfig.me)
GitHub username:       _________________ (for yourusername.github.io)
Vercel project name:   _________________ (for your-api.vercel.app)
Registrar:             _________________ (GoDaddy, Namecheap, etc.)
Registrar login:       _________________
```

---

## 5️⃣ Quick Setup (5 minutes)

### 1️⃣ Add Domain to Cloudflare
```
https://dash.cloudflare.com
→ Add a Site
→ Enter: penux.uk
→ Select: Free plan
```

### 2️⃣ Update Nameservers at Registrar
```
Copy Cloudflare nameservers:
  NS1: [something].ns.cloudflare.com
  NS2: [something].ns.cloudflare.com

Your registrar:
  → Domain settings
  → Change nameservers
  → Paste above
  → Save
```

### 3️⃣ Add DNS Records in Cloudflare

**Copy-paste these exactly:**

```
Type: A          Name: @         Value: [YOUR_PUBLIC_IP]
Type: A          Name: www       Value: [YOUR_PUBLIC_IP]
Type: CNAME      Name: ldap      Value: yourusername.github.io
Type: CNAME      Name: api.ldap  Value: your-api.vercel.app
```

⚠️ **ALL CNAMEs: Set to DNS only (NOT Proxied)**

### 4️⃣ Configure SSL/TLS
```
Cloudflare Dashboard
→ SSL/TLS
→ Overview
→ Encryption Mode: Full
→ Always Use HTTPS: ON
```

### 5️⃣ GitHub Pages Custom Domain
```
Your GitHub Pages repo
→ Settings → Pages
→ Custom domain: ldap.penux.uk
→ Enforce HTTPS: checked
→ Save
```

---

## ✅ Verification (2 minutes)

**From your terminal:**

```bash
# Test DNS is working
nslookup penux.uk
nslookup ldap.penux.uk
nslookup api.ldap.penux.uk

# Should resolve to:
#   penux.uk → YOUR_PUBLIC_IP
#   ldap.penux.uk → GitHub IP
#   api.ldap.penux.uk → Vercel IP
```

**In your browser:**
- [ ] https://ldap.penux.uk (loads LDAP interface)
- [ ] https://api.ldap.penux.uk/api/health (returns JSON)
- [ ] 🔒 lock icon shows (HTTPS is working)

---

## 🔴 TROUBLESHOOTING

| Problem | Solution |
|---------|----------|
| **DNS not resolving** | Wait 15-60 min, then `ipconfig /flushdns` (Windows) or `sudo dscacheutil -flushcache` (Mac) |
| **CNAME error** | Make sure it says **DNS only**, not **Proxied** |
| **Certificate pending** | Wait up to 24 hours, refresh browser |
| **GitHub 404** | Check custom domain is set in GitHub Pages settings |
| **Vercel 404** | Wait 5 min, redeploy: `vercel --prod` |

---

## 📊 DNS Records Checklist

```
☐ A @ = YOUR_PUBLIC_IP (DNS only)
☐ A www = YOUR_PUBLIC_IP (DNS only)
☐ CNAME ldap = yourusername.github.io (DNS only)
☐ CNAME api.ldap = your-api.vercel.app (DNS only)

SSL/TLS:
☐ Encryption: Full
☐ Always HTTPS: ON
☐ Min TLS: 1.2
☐ Security: High
```

---

## 🧪 Final Test

```bash
# 1. Test DNS resolution
dig penux.uk          # Should show your IP
dig ldap.penux.uk     # Should show GitHub IP
dig api.ldap.penux.uk # Should show Vercel IP

# 2. Test HTTPS
curl -I https://ldap.penux.uk
curl -I https://api.ldap.penux.uk

# 3. Test API
curl https://api.ldap.penux.uk/api/health
# Should return: {"status":"ok",...}
```

**All tests pass? ✅ You're done!**

---

## 🆘 CRITICAL ISSUES

### "CNAME Conflicts with Existing Record"
→ Delete the A record at `ldap`, keep only CNAME

### "Too Many Redirects"
→ Check if GitHub Pages also has a redirect
→ Keep only one redirect active

### "ERR_TOO_MANY_REDIRECTS"
→ Clear browser cache: `Ctrl+Shift+R`
→ Try incognito/private mode

### "Certificate Error / Untrusted"
→ Wait 24 hours for GitHub to issue cert
→ Don't use custom domain yet

### DNS Still Shows Old Nameservers
→ Wait 30 minutes
→ Check registrar confirms change was saved
→ Try: `nslookup penux.uk 1.1.1.1` (forces Cloudflare DNS)

---

## 🎯 Key Remember

✅ **GitHub Pages CNAME**: DNS only (never Proxied)  
✅ **Vercel CNAME**: DNS only (never Proxied)  
✅ **A records**: Can be Proxied or DNS only (doesn't matter)  
✅ **SSL/TLS**: Full mode  
✅ **HTTPS**: Always ON  

---

## ⏱️ Timeline

| Task | Time | Notes |
|------|------|-------|
| Create Cloudflare account | 2 min | Free plan |
| Add domain | 2 min | Scan existing records |
| Update nameservers | 1 min | At your registrar |
| **Nameserver propagation** | **15-60 min** | ⏳ Wait here |
| Add DNS records | 3 min | Copy-paste above |
| Configure SSL/TLS | 2 min | Full + Always HTTPS |
| GitHub Pages domain | 2 min | In Pages settings |
| **Certificate issuance** | **5-24 hours** | ⏳ Can take time |
| Test | 2 min | Run verification commands |
| **TOTAL** | **~30 min** | (+ DNS propagation) |

---

## 📞 Support Resources

- **Cloudflare Docs**: https://developers.cloudflare.com/dns/
- **GitHub Pages**: https://pages.github.com
- **Vercel Domains**: https://vercel.com/docs/concepts/projects/domains
- **DNS Tester**: https://dnschecker.org/
- **Certificate Check**: https://www.ssllabs.com/ssltest/

---

## ✨ Success Indicators

You'll know it's working when:

✅ `nslookup ldap.penux.uk` resolves to GitHub IP  
✅ `nslookup api.ldap.penux.uk` resolves to Vercel IP  
✅ https://ldap.penux.uk loads your LDAP interface  
✅ https://api.ldap.penux.uk/api/health returns JSON  
✅ Browser shows 🔒 secure connection  
✅ No certificate warnings  

---

**You're all set! Access your LDAP directory at: https://ldap.penux.uk** 🎉

