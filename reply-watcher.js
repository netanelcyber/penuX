import { ImapFlow } from 'imapflow';
import cron from 'node-cron';

// Contacts we're tracking follow-ups for. Maps a sender email (or domain
// fragment) to a task title match and a template key used to draft a reply.
const TRACKED_CONTACTS = [
  { match: 'james.buxbaum', taskMatch: 'Buxbaum', key: 'buxbaum' },
  { match: 'saurabh.chawla', taskMatch: 'Chawla', key: 'chawla' },
  { match: 'hegyi2009', taskMatch: 'Hegyi', key: 'hegyi' },
  { match: 'luca.frulloni', taskMatch: 'Frulloni', key: 'frulloni' },
  { match: 'panu.mentula', taskMatch: 'Mentula', key: 'mentula' },
];

// All PenuX availability (for scheduling Zoom meetings) is constrained to
// this window, Israel time. Referenced by both the rule-based drafts below
// and the Claude prompt instruction.
const AVAILABILITY_WINDOW = '9:00 AM–7:00 PM Israel time';

// Simple rule-based reply drafting from keywords in the incoming message.
// If ANTHROPIC_API_KEY is set, callAnthropicForReply() is used instead for
// genuinely generated (not templated) replies.
function draftReplyRuleBased(fromName, incomingText) {
  const lower = (incomingText || '').toLowerCase();
  let tone = 'Thank you for your reply.';
  let meetingLine = 'Please let us know how you would like to proceed, and we will follow up promptly.';

  if (lower.includes('yes') || lower.includes('available') || lower.includes('works for me')) {
    tone = 'Wonderful — thank you for confirming!';
    meetingLine = `We will send over a Zoom invite shortly — all our meetings are conducted via Zoom, and we are available ${AVAILABILITY_WINDOW}. Please let us know if the proposed time falls within that window, or suggest an alternative that does.`;
  } else if (lower.includes('cannot') || lower.includes("can't") || lower.includes('unable') || lower.includes('not available')) {
    tone = 'Thank you for letting us know — no problem at all, let\'s find another time.';
    meetingLine = `Please share a time that works better for you within ${AVAILABILITY_WINDOW}, and we will send a Zoom invite to match. If your schedule genuinely does not overlap with that window, we are also very happy to continue this conversation by email instead.`;
  } else if (lower.includes('question') || lower.includes('?')) {
    tone = 'Thank you for your question — happy to clarify.';
  }

  return `Dear ${fromName || 'Colleague'},

${tone}

We appreciate you taking the time to respond regarding the PenuX collaboration. ${meetingLine}

Best regards,
PenuX Research Team`;
}

async function callAnthropicForReply(fromName, incomingText) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return null;

  try {
    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
      },
      body: JSON.stringify({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 400,
        messages: [{
          role: 'user',
          content: `You are drafting a short, professional follow-up reply on behalf of the PenuX research team, replying to ${fromName || 'a medical collaborator'}. Their message was:\n\n"""${incomingText}"""\n\nWrite only the reply email body (no subject line), in English, warm and professional, 3-5 sentences, signed "Best regards,\\nPenuX Research Team". If a meeting is being scheduled or confirmed, state that all PenuX meetings are conducted via Zoom, and that availability is limited to ${AVAILABILITY_WINDOW}. If the contact proposed a time outside that window, politely ask them to pick a time within it instead of accepting it as-is. If their timezone genuinely does not overlap with that window at all, offer to continue the conversation by email instead of insisting on a Zoom call.`,
        }],
      }),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data?.content?.[0]?.text || null;
  } catch {
    return null;
  }
}

export async function draftReply(fromName, incomingText) {
  const aiDraft = await callAnthropicForReply(fromName, incomingText);
  return aiDraft || draftReplyRuleBased(fromName, incomingText);
}

function matchContact(fromAddress) {
  const lower = (fromAddress || '').toLowerCase();
  return TRACKED_CONTACTS.find(c => lower.includes(c.match));
}

export function startReplyWatcher(db, { schedule = '*/10 * * * *' } = {}) {
  const user = process.env.GMAIL_USER;
  const pass = process.env.GMAIL_APP_PASSWORD;

  if (!user || !pass) {
    console.warn('⚠️  Reply watcher not started — GMAIL_USER / GMAIL_APP_PASSWORD not set.');
    return null;
  }

  db.run(`
    CREATE TABLE IF NOT EXISTS draft_replies (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      contact_key TEXT,
      from_addr TEXT,
      subject TEXT,
      incoming_snippet TEXT,
      draft_body TEXT,
      status TEXT DEFAULT 'pending',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  db.run(`
    CREATE TABLE IF NOT EXISTS seen_messages (
      uid TEXT PRIMARY KEY,
      seen_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
      const lock = await client.getMailboxLock('INBOX');
      try {
        // Only look at messages from the last 3 days to keep polling cheap.
        const since = new Date(Date.now() - 3 * 24 * 60 * 60 * 1000);
        const messages = client.fetch(
          { since },
          { envelope: true, source: false, uid: true, bodyStructure: true }
        );

        for await (const msg of messages) {
          const fromAddr = msg.envelope?.from?.[0]?.address || '';
          const fromName = msg.envelope?.from?.[0]?.name || fromAddr;
          const contact = matchContact(fromAddr);
          if (!contact) continue;

          const uidKey = `${msg.uid}`;
          const already = await new Promise((resolve) => {
            db.get('SELECT uid FROM seen_messages WHERE uid = ?', [uidKey], (err, row) => resolve(!!row));
          });
          if (already) continue;

          // Fetch a plain-text snippet of the body.
          let snippet = msg.envelope?.subject || '';
          try {
            const full = await client.download(msg.uid, undefined, { uid: true });
            const chunks = [];
            for await (const chunk of full.content) chunks.push(chunk);
            const text = Buffer.concat(chunks).toString('utf8');
            snippet = text.replace(/<[^>]+>/g, ' ').slice(0, 1000);
          } catch {
            // fall back to subject-only snippet
          }

          const draftBody = await draftReply(fromName, snippet);

          db.run(
            `INSERT INTO draft_replies (contact_key, from_addr, subject, incoming_snippet, draft_body)
             VALUES (?, ?, ?, ?, ?)`,
            [contact.key, fromAddr, msg.envelope?.subject || '', snippet, draftBody]
          );
          db.run('INSERT OR IGNORE INTO seen_messages (uid) VALUES (?)', [uidKey]);

          console.log(`✉️  New reply detected from ${fromAddr} — draft generated.`);
        }
      } finally {
        lock.release();
      }
      await client.logout();
    } catch (err) {
      console.error('Reply watcher poll failed:', err.message);
      try { await client.close(); } catch {}
    }
  }

  const task = cron.schedule(schedule, pollOnce);
  // Run once immediately on startup too.
  pollOnce();

  console.log(`📬 Reply watcher started — polling inbox on schedule "${schedule}".`);
  return task;
}
