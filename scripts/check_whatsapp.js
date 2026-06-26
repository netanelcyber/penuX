const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');

const CONTACTS = [
  { name: 'Prof. Michael Kochman',  phone: '12153498354'  },  // +1
  { name: 'Dr. Saurabh Chawla',     phone: '14047782714'  },  // +1
  { name: 'Prof. Timna Naftali',    phone: '97235028500'  },  // +972
  { name: 'Dr. Vera Dreizin',       phone: '972545558570' },  // +972
];

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: '/tmp/wwa_session' }),
  puppeteer: {
    executablePath: '/opt/pw-browsers/chromium',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage'],
    headless: true,
  }
});

client.on('qr', qr => {
  console.log('\n📱 Scan this QR code with WhatsApp on your phone:\n');
  qrcode.generate(qr, { small: true });
});

client.on('authenticated', () => console.log('\n✅ Authenticated.\n'));
client.on('auth_failure', msg => { console.error('❌ Auth failed:', msg); process.exit(1); });

client.on('ready', async () => {
  console.log('✅ WhatsApp ready. Checking numbers...\n');

  const results = [];

  for (const c of CONTACTS) {
    try {
      const waId   = c.phone + '@c.us';
      const exists = await client.isRegisteredUser(waId);
      const status = exists ? '✅ Has WhatsApp' : '❌ NOT on WhatsApp';
      results.push({ ...c, has_whatsapp: exists, status });
      console.log(`  ${status.padEnd(20)} ${c.name.padEnd(30)} +${c.phone}`);
    } catch (e) {
      console.log(`  ⚠️  Error        ${c.name} — ${e.message}`);
      results.push({ ...c, has_whatsapp: null, status: '⚠️ Error' });
    }
  }

  console.log('\n' + '='.repeat(60));
  const ok = results.filter(r => r.has_whatsapp);
  console.log(`📱 ${ok.length}/${results.length} numbers have WhatsApp.\n`);

  if (ok.length > 0) {
    console.log('Group-eligible contacts:');
    ok.forEach(r => console.log(`  • ${r.name} — +${r.phone}`));

    // Write result for the group creator script
    const fs = require('fs');
    fs.writeFileSync(
      '/tmp/wa_verified_contacts.json',
      JSON.stringify(ok, null, 2)
    );
    console.log('\nSaved to /tmp/wa_verified_contacts.json');
  }

  console.log('='.repeat(60));
  await client.destroy();
  process.exit(0);
});

console.log('Starting WhatsApp client...');
client.initialize();
