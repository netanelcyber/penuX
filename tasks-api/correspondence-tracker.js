/**
 * Polls Gmail (via IMAP, same credentials as reply-watcher.js) for the
 * latest message to/from each tracked outreach contact, and maintains a
 * `correspondence` table that always reflects the real current state of
 * each conversation — direction of the last message, its date/snippet,
 * and a derived "waiting on you" / "waiting on them" status.
 *
 * This replaces manually-maintained, easily-stale task descriptions (the
 * exact problem found earlier: a task board seeded with hardcoded text
 * like "awaiting reply" that silently goes wrong the moment someone
 * actually replies) with a table that re-derives its own state from the
 * mailbox on every poll — nothing to remember to update by hand.
 */
import { ImapFlow } from 'imapflow';
import cron from 'node-cron';

// Contacts to track: match = lowercase substring of their email address.
// Add new outreach contacts here — no code changes needed elsewhere.
export const TRACKED_CONTACTS = [
  { match: 'james.buxbaum', name: 'Dr. James Buxbaum (USC)', category: 'meeting' },
  { match: 'saurabh.chawla', name: 'Dr. Saurabh Chawla', category: 'follow' },
  { match: 'hegyi2009', name: 'Prof. Peter Hegyi (Semmelweis)', category: 'meeting' },
  { match: 'luca.frulloni', name: 'Prof. Luca Frulloni (Verona)', category: 'follow' },
  { match: 'panu.mentula', name: 'Panu Mentula (Helsinki)', category: 'follow' },
  { match: 'aldis.pukitis', name: 'Prof. Aldis Puķītis (Riga)', category: 'follow' },
  { match: 'j.windsor', name: 'Prof. John Windsor (Auckland)', category: 'follow' },
  { match: 'rohatakmd', name: 'Dr. Rohatak', category: 'follow' },
  { match: 'klaus.sahora', name: 'Dr. Klaus Sahora (Vienna)', category: 'follow' },
  { match: 'carolyn.calfee', name: 'Dr. Carolyn Calfee (UCSF)', category: 'follow' },
  { match: 'verad@wmc.gov.il', name: 'Dr. Vera Dreizin (Wolfson)', category: 'meeting' },
  { match: 'timnan@wmc.gov.il', name: 'Prof. Timna Naftali (Wolfson)', category: 'meeting' },
];

function matchContact(address) {
  const lower = (address || '').toLowerCase();
  return TRACKED_CONTACTS.find(c => lower.includes(c.match));
}

function textSnippet(buffer) {
  return buffer.toString('utf8').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 300);
}

async function findSentMailbox(client) {
  const list = await client.list();
  const sent = list.find(m => m.specialUse === '\\Sent');
  return sent ? sent.path : '[Gmail]/Sent Mail';
}

async function scanMailbox(client, mailboxPath, direction, latestByContact, lookbackDays) {
  const lock = await client.getMailboxLock(mailboxPath);
  try {
    const since = new Date(Date.now() - lookbackDays * 24 * 60 * 60 * 1000);
    const field = direction === 'received' ? 'from' : 'to';
    const messages = client.fetch({ since }, { envelope: true, uid: true });

    for await (const msg of messages) {
      const parties = direction === 'received' ? msg.envelope?.from : msg.envelope?.to;
      const address = parties?.[0]?.address || '';
      const contact = matchContact(address);
      if (!contact) continue;

      const date = msg.envelope?.date ? new Date(msg.envelope.date) : null;
      if (!date) continue;

      const existing = latestByContact.get(contact.match);
      if (!existing || date > existing.date) {
        let snippet = msg.envelope?.subject || '';
        try {
          const full = await client.download(msg.uid, undefined, { uid: true });
          const chunks = [];
          for await (const chunk of full.content) chunks.push(chunk);
          snippet = textSnippet(Buffer.concat(chunks)) || snippet;
        } catch {
          // keep subject-only snippet
        }
        latestByContact.set(contact.match, {
          contact,
          date,
          direction,
          subject: msg.envelope?.subject || '',
          snippet,
        });
      }
    }
  } finally {
    lock.release();
  }
}

function deriveStatus(entry) {
  if (!entry) return { status: 'no_correspondence_found', label: 'No emails found in lookback window' };
  if (entry.direction === 'received') {
    return { status: 'waiting_on_you', label: 'They replied — waiting on you' };
  }
  return { status: 'waiting_on_them', label: 'Waiting on their reply' };
}

export function startCorrespondenceTracker(db, { schedule = '*/10 * * * *', lookbackDays = 60 } = {}) {
  const user = process.env.GMAIL_USER;
  const pass = process.env.GMAIL_APP_PASSWORD;

  if (!user || !pass) {
    console.warn('⚠️  Correspondence tracker not started — GMAIL_USER / GMAIL_APP_PASSWORD not set.');
    return null;
  }

  db.run(`
    CREATE TABLE IF NOT EXISTS correspondence (
      contact_match TEXT PRIMARY KEY,
      contact_name TEXT,
      category TEXT,
      last_direction TEXT,
      last_date DATETIME,
      last_subject TEXT,
      last_snippet TEXT,
      status TEXT,
      status_label TEXT,
      updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  async function pollOnce() {
    const client = new ImapFlow({
      host: 'imap.gmail.com',
      port: 993,
      secure: true,
      auth: { user, pass },
      logger: false,
    });

    try {
      await client.connect();
      const latestByContact = new Map();

      await scanMailbox(client, 'INBOX', 'received', latestByContact, lookbackDays);
      const sentPath = await findSentMailbox(client);
      await scanMailbox(client, sentPath, 'sent', latestByContact, lookbackDays);

      for (const contact of TRACKED_CONTACTS) {
        const entry = latestByContact.get(contact.match);
        const { status, label } = deriveStatus(entry);

        db.run(
          `INSERT INTO correspondence
             (contact_match, contact_name, category, last_direction, last_date, last_subject, last_snippet, status, status_label, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(contact_match) DO UPDATE SET
             contact_name=excluded.contact_name, category=excluded.category,
             last_direction=excluded.last_direction, last_date=excluded.last_date,
             last_subject=excluded.last_subject, last_snippet=excluded.last_snippet,
             status=excluded.status, status_label=excluded.status_label,
             updated_at=CURRENT_TIMESTAMP`,
          [
            contact.match, contact.name, contact.category,
            entry?.direction || null, entry?.date?.toISOString() || null,
            entry?.subject || null, entry?.snippet || null,
            status, label,
          ]
        );
      }

      await client.logout();
      console.log(`📊 Correspondence tracker: updated ${latestByContact.size}/${TRACKED_CONTACTS.length} contacts.`);
    } catch (err) {
      console.error('Correspondence tracker poll failed:', err.message);
      try { await client.close(); } catch {}
    }
  }

  const task = cron.schedule(schedule, pollOnce);
  pollOnce();

  console.log(`📊 Correspondence tracker started — polling mailbox on schedule "${schedule}".`);
  return task;
}
