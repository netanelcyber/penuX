import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import crypto from 'crypto';
import nodemailer from 'nodemailer';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { startReplyWatcher } from './reply-watcher.js';
import { startCorrespondenceTracker, startTaskBoardSync } from './correspondence-tracker.js';
import { getClient, wrapCompat } from './db.js';
import predictRoutes from './predict-routes.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;
const PASSWORD_HASH = '5b1a03055e62917d46aa4e7377050da381f80a98a8e14c40a72677f04c32c9a2'; // SHA-256 of "penux2026"

// Uses a local file (tasks.db) by default. Set TURSO_DATABASE_URL +
// TURSO_AUTH_TOKEN to use a free Turso database instead, so task data
// survives restarts/redeploys on hosts without a persistent disk (e.g.
// Render's free tier). See README.md > "Deployment (free tier)".
const dbClient = getClient();
const db = wrapCompat(dbClient);

// ── EMAIL SENDING (Gmail SMTP via App Password) ────────────────────────────
// Requires env vars: GMAIL_USER, GMAIL_APP_PASSWORD
// Setup: enable 2FA on the Gmail account, then create an App Password at
// https://myaccount.google.com/apppasswords and set it as GMAIL_APP_PASSWORD.
let transporter = null;
if (process.env.GMAIL_USER && process.env.GMAIL_APP_PASSWORD) {
  transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.GMAIL_USER,
      pass: process.env.GMAIL_APP_PASSWORD,
    },
  });
} else {
  console.warn('⚠️  GMAIL_USER / GMAIL_APP_PASSWORD not set — /api/tasks/:id/send-email will return 503.');
}

app.use(cors());
app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

// PenuX-AP-Severity prediction endpoints (/predict, /predict/sepsis,
// /predict/deterioration, /predict/mortality, /predict/saps2,
// /predict/polynomial-logit, /models/sweep) — ported from the Python
// FastAPI service (PenuX-AP-Severity/api/main.py) so this single Node
// deployment serves both the task board and the prediction API. See
// predict-routes.js for the full port with source-of-truth notes.
app.use(predictRoutes);

// Initialize database (properly sequenced with awaits — libsql's promise
// API doesn't have sqlite3's implicit serialize() queueing behavior).
async function initDb() {
  await dbClient.execute(`
    CREATE TABLE IF NOT EXISTS tasks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      sub TEXT,
      due TEXT,
      cat TEXT,
      status TEXT,
      talking_points TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // ALTER for pre-existing databases created before talking_points existed —
  // libsql throws "duplicate column" if it's already there, which we ignore.
  try {
    await dbClient.execute('ALTER TABLE tasks ADD COLUMN talking_points TEXT');
  } catch {
    // column already exists
  }

  await dbClient.execute(`
    CREATE TABLE IF NOT EXISTS sent_emails (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      task_id INTEGER,
      to_addr TEXT,
      subject TEXT,
      sent_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  // Created here (not just inside correspondence-tracker.js's startup path)
  // so GET /api/correspondence never 500s with "no such table" even when
  // GMAIL_USER/GMAIL_APP_PASSWORD aren't set and the tracker never runs —
  // it just serves an empty array instead.
  await dbClient.execute(`
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

  const countResult = await dbClient.execute('SELECT COUNT(*) as count FROM tasks');
  if (countResult.rows[0].count === 0) {
    const defaultTasks = [
      { title: 'Zoom — Prof. Michael Kochman (UPenn Gastro)', sub: 'Confirmed: Tue 10 Jul, 08:00–09:00 EST (15:00 IL)', due: '2026-07-10', cat: 'meeting', status: 'pending' },
      { title: 'Zoom — Dr. Nauzer Forbes (Calgary)', sub: 'Confirmed by Dr. Forbes — meeting set for Aug 10', due: '2026-08-10', cat: 'meeting', status: 'pending' },
      { title: 'Meeting — Prof. Tamara Naftali (Wolfson)', sub: 'Tuesday 30.6, 10:00 — Wolfson Gastro Institute', due: '2026-06-30', cat: 'meeting', status: 'done' },
      { title: 'פגישת הכרות — ד"ר ורה דרייזין (וולפסון)', sub: 'התקיימה — פגישת הכרות עם ד"ר ורה דרייזין (מנהלת שירות הכבד), בהשתתפות פרופ\' תמנע נפתלי ואורית ש.', due: null, cat: 'meeting', status: 'done' },
      { title: 'Meeting חוזרת — וולפסון (מתאמת מחקר + צוות)', sub: 'נקבע: 4.8 — עם מתאמת המחקר והצוות; תרגום המאמר לעברית נשלח 6.7', due: '2026-08-04', cat: 'meeting', status: 'pending' },
      { title: 'Meeting — Dr. James Buxbaum (USC)', sub: 'Held via Teams, Mon 6 Jul 10:00 PST (confirmed by Cindy Lee on behalf of Dr. Buxbaum) — meeting has taken place', due: '2026-07-06', cat: 'meeting', status: 'done' },
      { title: 'Follow-Up — Dr. Saurabh Chawla', sub: 'Call took place 6/29 2pm EST — confirm outcome / next steps', due: '2026-06-30', cat: 'follow', status: 'waiting' },
      { title: 'Zoom scheduling — Prof. Peter Hegyi (Semmelweis)', sub: 'Real address confirmed (hegyi2009@gmail.com). Original slot (Mon 20.7 13:00 IL) fell through — needs to be rescheduled from scratch', due: null, cat: 'meeting', status: 'urgent' },
      { title: 'Follow-Up — Prof. Luca Frulloni (Verona)', sub: 'Back from OOO since 29.6 — followed up 6/25, 6/29, 7/1 — still no personal reply', due: '2026-07-01', cat: 'follow', status: 'waiting' },
      { title: 'Follow-Up — Panu Mentula (Helsinki)', sub: 'Back from OOO since 13.7 — followed up 6/25, 6/29 — still no personal reply', due: '2026-06-29', cat: 'follow', status: 'waiting' },
      { title: 'Follow-Up — Prof. Aldis Puķītis (Riga, Latvia)', sub: 'Replied 15.7 — interested, asked deadlines/aims, wants Zoom — you replied 16.7 — awaiting his response on scheduling', due: '2026-07-16', cat: 'follow', status: 'waiting' },
      { title: 'Follow-Up — Prof. John Windsor (Auckland)', sub: 'Extended back-and-forth through 23.6 (spot-on assessment, discussing what he can help with) — last message was yours, no reply since', due: '2026-06-23', cat: 'follow', status: 'waiting' },
      { title: 'Follow-Up — Dr. Rohatak (rohatakmd@gmail.com)', sub: 'Replied 27.7 — appreciated your comments, said your research looks highly relevant — propose a concrete Zoom time', due: '2026-07-27', cat: 'follow', status: 'waiting' },
      { title: 'Follow-Up — Dr. Klaus Sahora (MedUni Vienna)', sub: 'Outreach sent 10.7 (AG Leber/Galle/Pankreas mailing list request) — no reply yet', due: '2026-07-10', cat: 'follow', status: 'waiting' },
      { title: 'Follow-Up — Dr. Carolyn Calfee (UCSF)', sub: 'Followed up 10.7 after her OOO auto-reply expired (was back 6.7) — no reply since', due: '2026-07-10', cat: 'follow', status: 'waiting' },
      { title: 'Submit to OSF Preprints', sub: 'run: OSF_TOKEN=<token> python3 outreach/submit_osf.py', due: '2026-06-28', cat: 'research', status: 'urgent' },
      { title: 'Stage I Protocol — IRB Submission', sub: 'Prepare IRB package from PenuX_SAP_Stage1_Study_Protocol.pdf', due: '2026-07-15', cat: 'research', status: 'pending' },
      { title: 'Fix bad email — Prof. Lévy (Paris)', sub: 'No personal email published — try dept. secretariat: secretariat.pancreato.bjn@aphp.fr', due: '2026-06-30', cat: 'admin', status: 'pending' },
      { title: 'Follow-Up — Dr. Amir Dagan (Shaare Zedek)', sub: 'Personalized email sent (Liver & Pancreatic Surgery Unit) — 3.7 — awaiting reply', due: '2026-07-03', cat: 'follow', status: 'waiting' },
      { title: 'Follow-Up — Dr. Michael Neuman (Shaare Zedek)', sub: 'Email sent (cardiac surgeon, not GI/pancreas — asked for referral) — 3.7 — awaiting reply', due: '2026-07-03', cat: 'follow', status: 'waiting' },
    ];

    for (const task of defaultTasks) {
      await dbClient.execute({
        sql: 'INSERT INTO tasks (title, sub, due, cat, status) VALUES (?, ?, ?, ?, ?)',
        args: [task.title, task.sub, task.due, task.cat, task.status],
      });
    }
  }
}

await initDb();

// Auth middleware
function validatePassword(pw) {
  const buf = crypto.createHash('sha256').update(pw).digest('hex');
  return buf === PASSWORD_HASH;
}

// Routes
app.post('/api/auth', (req, res) => {
  const { password } = req.body;
  if (validatePassword(password)) {
    res.json({ success: true, token: 'auth_ok' });
  } else {
    res.status(401).json({ success: false, error: 'Incorrect password' });
  }
});

app.get('/api/tasks', (req, res) => {
  db.all('SELECT * FROM tasks ORDER BY status, due', (err, rows) => {
    if (err) {
      res.status(500).json({ error: err.message });
    } else {
      res.json(rows);
    }
  });
});

app.post('/api/tasks', (req, res) => {
  const { title, sub, due, cat, status } = req.body;
  db.run(
    'INSERT INTO tasks (title, sub, due, cat, status) VALUES (?, ?, ?, ?, ?)',
    [title, sub, due, cat, status],
    function(err) {
      if (err) {
        res.status(500).json({ error: err.message });
      } else {
        res.json({ id: this.lastID, title, sub, due, cat, status });
      }
    }
  );
});

app.put('/api/tasks/:id', (req, res) => {
  const { status } = req.body;
  db.run(
    'UPDATE tasks SET status = ? WHERE id = ?',
    [status, req.params.id],
    (err) => {
      if (err) {
        res.status(500).json({ error: err.message });
      } else {
        res.json({ success: true });
      }
    }
  );
});

app.delete('/api/tasks/:id', (req, res) => {
  db.run('DELETE FROM tasks WHERE id = ?', [req.params.id], (err) => {
    if (err) {
      res.status(500).json({ error: err.message });
    } else {
      res.json({ success: true });
    }
  });
});

// All PenuX Zoom availability is limited to this window, Israel time. If a
// contact's timezone/schedule genuinely doesn't overlap, templates fall back
// to offering email-only correspondence instead of insisting on a call.
const AVAILABILITY_WINDOW = '9:00 AM–7:00 PM Israel time';

const EMAIL_TEMPLATES = {
  'buxbaum': {
    to: 'james.buxbaum@usc.edu',
    subject: 'Re: PenuX Collaboration — Timeline Coordination',
    body: `Dear Dr. Buxbaum,

Thank you for your enthusiastic response to the PenuX severe acute pancreatitis severity project. We are very excited about the potential collaboration with USC.

Following up on our conversation, Cindy will be in touch shortly to schedule a Zoom meeting at your earliest convenience. All our meetings are conducted via Zoom, and we are available ${AVAILABILITY_WINDOW} — if that doesn't overlap well with your schedule, we're also very happy to continue by email instead.

In the meantime, please let us know if you would like us to send over any preliminary data or methodology details for your review.

Looking forward to connecting soon.

Best regards,
PenuX Research Team`
  },
  'chawla': {
    to: 'saurabh.chawla@emory.edu',
    subject: 'Re: PenuX Study — Meeting Confirmation',
    body: `Dear Dr. Chawla,

Thank you for your interest in the PenuX project. Following your suggestion, we would like to schedule a Zoom meeting for Monday at your convenience.

Could you please confirm a time that falls within ${AVAILABILITY_WINDOW}? We realize a typical US afternoon may fall outside that window — if a suitable overlap isn't possible, we are also very happy to continue this conversation by email instead of Zoom.

All our meetings are conducted via Zoom — once we hear from you, we will send a Zoom invite with the link.

Looking forward to discussing how we can work together.

Best regards,
PenuX Research Team`
  },
  'hegyi': {
    to: 'hegyi2009@gmail.com',
    subject: 'Re: PenuX Collaboration — Follow-up',
    body: `Dear Prof. Hegyi,

Thank you for your positive response to the PenuX project. We understand you are working on gathering available dates for a potential Zoom meeting.

We are very interested in collaborating with Semmelweis University on this important research. All our meetings are conducted via Zoom, and we are available ${AVAILABILITY_WINDOW} — please let us know a time within that window and we will send a Zoom invite. If that doesn't work for you, we're also happy to continue by email.

If you need any additional information about the project to facilitate scheduling, please do not hesitate to reach out.

Best regards,
PenuX Research Team`
  },
  'frulloni': {
    to: 'luca.frulloni@univr.it',
    subject: 'Re: PenuX Collaboration — Welcome Back',
    body: `Dear Prof. Frulloni,

Welcome back! We hope you had a restful time away.

We wanted to follow up on the PenuX severe acute pancreatitis severity project and see if you had a chance to review our previous message. We would love to schedule a brief Zoom call to discuss a potential collaboration — we are available ${AVAILABILITY_WINDOW}, but are just as happy to continue this conversation by email if that's easier for your schedule.

Please let us know your availability.

Best regards,
PenuX Research Team`
  }
};

app.post('/api/tasks/:id/email', (req, res) => {
  const { template } = req.body;
  const email = EMAIL_TEMPLATES[template];

  if (!email) {
    return res.status(400).json({ error: 'Template not found' });
  }

  res.json({
    success: true,
    email: {
      to: email.to,
      subject: email.subject,
      body: email.body,
      template
    },
    instruction: 'Email draft prepared. Click "Send Now" to send it, or "Open Gmail" to send manually.'
  });
});

// Actually sends the email via Gmail SMTP — this is a real, irreversible send.
app.post('/api/tasks/:id/send-email', async (req, res) => {
  if (!transporter) {
    return res.status(503).json({
      error: 'Email sending is not configured. Set GMAIL_USER and GMAIL_APP_PASSWORD env vars on the server.'
    });
  }

  const { to, subject, body } = req.body;
  if (!to || !subject || !body) {
    return res.status(400).json({ error: 'to, subject, and body are required' });
  }

  const taskId = req.params.id;

  try {
    await transporter.sendMail({
      from: process.env.GMAIL_USER,
      to,
      subject,
      text: body,
    });

    db.run(
      'INSERT INTO sent_emails (task_id, to_addr, subject) VALUES (?, ?, ?)',
      [taskId, to, subject]
    );

    // Auto-mark the task done if it corresponds to a real task (not the placeholder id 0)
    if (taskId && taskId !== '0') {
      db.run('UPDATE tasks SET status = ? WHERE id = ?', ['done', taskId]);
    }

    res.json({ success: true, message: `Email sent to ${to}` });
  } catch (err) {
    res.status(500).json({ error: 'Failed to send email: ' + err.message });
  }
});

// ── AUTO-DRAFTED REPLIES (from inbox watcher) ──────────────────────────────
// The reply-watcher polls Gmail's inbox for new messages from tracked
// contacts and stores an AI/rule-drafted response here for human review.
// Nothing is auto-sent — a person must click "Send Now" to actually send it.

app.get('/api/draft-replies', (req, res) => {
  db.all(
    `SELECT * FROM draft_replies WHERE status = 'pending' ORDER BY created_at DESC`,
    (err, rows) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json(rows);
    }
  );
});

app.put('/api/draft-replies/:id', (req, res) => {
  const { draft_body } = req.body;
  db.run(
    'UPDATE draft_replies SET draft_body = ? WHERE id = ?',
    [draft_body, req.params.id],
    (err) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json({ success: true });
    }
  );
});

app.delete('/api/draft-replies/:id', (req, res) => {
  db.run(
    `UPDATE draft_replies SET status = 'discarded' WHERE id = ?`,
    [req.params.id],
    (err) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json({ success: true });
    }
  );
});

app.post('/api/draft-replies/:id/send', async (req, res) => {
  if (!transporter) {
    return res.status(503).json({
      error: 'Email sending is not configured. Set GMAIL_USER and GMAIL_APP_PASSWORD env vars on the server.'
    });
  }

  db.get('SELECT * FROM draft_replies WHERE id = ?', [req.params.id], async (err, draft) => {
    if (err) return res.status(500).json({ error: err.message });
    if (!draft) return res.status(404).json({ error: 'Draft not found' });

    try {
      await transporter.sendMail({
        from: process.env.GMAIL_USER,
        to: draft.from_addr,
        subject: draft.subject?.startsWith('Re:') ? draft.subject : `Re: ${draft.subject}`,
        text: draft.draft_body,
      });

      db.run(`UPDATE draft_replies SET status = 'sent' WHERE id = ?`, [draft.id]);
      db.run(
        'INSERT INTO sent_emails (task_id, to_addr, subject) VALUES (?, ?, ?)',
        [null, draft.from_addr, draft.subject]
      );

      res.json({ success: true, message: `Reply sent to ${draft.from_addr}` });
    } catch (sendErr) {
      res.status(500).json({ error: 'Failed to send reply: ' + sendErr.message });
    }
  });
});

// ── AUTO-UPDATING CORRESPONDENCE TABLE ─────────────────────────────────────
// Backed by correspondence-tracker.js, which polls the mailbox on a
// schedule and re-derives each contact's status from the actual latest
// message — no manually-maintained text to go stale.
app.get('/api/correspondence', (req, res) => {
  db.all('SELECT * FROM correspondence ORDER BY last_date DESC', (err, rows) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(rows);
  });
});

app.listen(PORT, () => {
  console.log(`🎯 PenuX Tasks API running on http://localhost:${PORT}`);
  startReplyWatcher(db);
  startCorrespondenceTracker(db);
  startTaskBoardSync(db);
});
