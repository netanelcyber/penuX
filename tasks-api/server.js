import express from 'express';
import sqlite3 from 'sqlite3';
import cors from 'cors';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;
const PASSWORD_HASH = '5b1a03055e62917d46aa4e7377050da381f80a98a8e14c40a72677f04c32c9a2'; // SHA-256 of "penux2026"

const db = new sqlite3.Database(join(__dirname, 'tasks.db'));

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

app.listen(PORT, () => {
  console.log(`🎯 PenuX Tasks API running on http://localhost:${PORT}`);
});
