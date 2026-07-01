import 'dotenv/config';
import express from 'express';
import sqlite3 from 'sqlite3';
import cors from 'cors';
import crypto from 'crypto';
import nodemailer from 'nodemailer';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;
const PASSWORD_HASH = '5b1a03055e62917d46aa4e7377050da381f80a98a8e14c40a72677f04c32c9a2'; // SHA-256 of "penux2026"

const db = new sqlite3.Database(join(__dirname, 'tasks.db'));

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

// Initialize database
db.serialize(() => {
  db.run(`
    CREATE TABLE IF NOT EXISTS tasks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      sub TEXT,
      due TEXT,
      cat TEXT,
      status TEXT,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  db.run(`
    CREATE TABLE IF NOT EXISTS sent_emails (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      task_id INTEGER,
      to_addr TEXT,
      subject TEXT,
      sent_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
  `);

  db.all('SELECT COUNT(*) as count FROM tasks', (err, rows) => {
    if (rows[0].count === 0) {
      const defaultTasks = [
        { title: 'Zoom — Prof. Michael Kochman (UPenn Gastro)', sub: 'Confirmed: Tue 10 Jul, 08:00–09:00 EST (15:00 IL)', due: '2026-07-10', cat: 'meeting', status: 'pending' },
        { title: 'Zoom — Dr. Nauzer Forbes (Calgary)', sub: 'Confirmed: Sun 6 Jul, 12:00 MDT (21:00 IL)', due: '2026-07-06', cat: 'meeting', status: 'pending' },
        { title: 'Meeting — Prof. Tamara Naftali (Wolfson)', sub: 'Tuesday 30.6, 10:00 — Wolfson Gastro Institute', due: '2026-06-30', cat: 'meeting', status: 'urgent' },
        { title: 'Follow-Up — Dr. James Buxbaum (USC)', sub: '"Very exciting" — wait for Cindy to coordinate time', due: '2026-06-30', cat: 'follow', status: 'waiting' },
        { title: 'Follow-Up — Dr. Saurabh Chawla', sub: '"Monday afternoon EST" — confirm specific day and time', due: '2026-06-30', cat: 'follow', status: 'urgent' },
        { title: 'Follow-Up — Prof. Peter Hegyi (Semmelweis)', sub: '"Will come back with dates" — wait, send reminder if not by 7.7', due: '2026-07-07', cat: 'follow', status: 'waiting' },
        { title: 'OOO Return — Prof. Luca Frulloni', sub: 'Returns 29.6 — send PenuX follow-up', due: '2026-06-29', cat: 'follow', status: 'pending' },
        { title: 'OOO Return — Panu Mentula', sub: 'Returns 13.7 — send PenuX follow-up', due: '2026-07-13', cat: 'follow', status: 'pending' },
        { title: 'OOO Return — Dutch recipient (STARRED)', sub: 'Returns 12.7 — send PenuX follow-up', due: '2026-07-12', cat: 'follow', status: 'pending' },
        { title: 'Submit to OSF Preprints', sub: 'run: OSF_TOKEN=<token> python3 outreach/submit_osf.py', due: '2026-06-28', cat: 'research', status: 'urgent' },
        { title: 'Stage I Protocol — IRB Submission', sub: 'Prepare IRB package from PenuX_SAP_Stage1_Study_Protocol.pdf', due: '2026-07-15', cat: 'research', status: 'pending' },
        { title: 'Fix bad email — Prof. Lévy (Paris)', sub: 'philippe.levy@bjn.ap-hop-paris.fr bounced — find correct address', due: '2026-06-30', cat: 'admin', status: 'pending' },
      ];

      defaultTasks.forEach(task => {
        db.run(
          'INSERT INTO tasks (title, sub, due, cat, status) VALUES (?, ?, ?, ?, ?)',
          [task.title, task.sub, task.due, task.cat, task.status]
        );
      });
    }
  });
});

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

// Email templates for follow-ups
const EMAIL_TEMPLATES = {
  'buxbaum': {
    to: 'james.buxbaum@usc.edu',
    subject: 'Re: PenuX Collaboration — Timeline Coordination',
    body: `Dear Dr. Buxbaum,

Thank you for your enthusiastic response to the PenuX severe acute pancreatitis severity project. We are very excited about the potential collaboration with USC.

Following up on our conversation, Cindy will be in touch shortly to schedule a meeting at your earliest convenience. We are flexible with timing and happy to work around your schedule.

In the meantime, please let us know if you would like us to send over any preliminary data or methodology details for your review.

Looking forward to connecting soon.

Best regards,
PenuX Research Team`
  },
  'chawla': {
    to: 'saurabh.chawla@[institution].edu',
    subject: 'Re: PenuX Study — Meeting Confirmation',
    body: `Dear Dr. Chawla,

Thank you for your interest in the PenuX project. Following your suggestion, we would like to schedule a meeting for Monday afternoon EST at your convenience.

Could you please confirm:
• Which Monday works best for you (we are flexible with dates)?
• Your preferred time window (morning, early afternoon, or late afternoon)?
• Preferred format (Zoom, phone, or in-person if you are in the New York area)?

Once we hear from you, we will send a calendar invite.

Looking forward to discussing how we can work together.

Best regards,
PenuX Research Team`
  },
  'hegyi': {
    to: 'peter.hegyi@semmelweis.hu',
    subject: 'Re: PenuX Collaboration — Follow-up',
    body: `Dear Prof. Hegyi,

Thank you for your positive response to the PenuX project. We understand you are working on gathering available dates for a potential meeting.

We are very interested in collaborating with Semmelweis University on this important research. Please let us know your availability at your earliest convenience — we are happy to accommodate your schedule.

If you need any additional information about the project to facilitate scheduling, please do not hesitate to reach out.

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

app.listen(PORT, () => {
  console.log(`🎯 PenuX Tasks API running on http://localhost:${PORT}`);
});
