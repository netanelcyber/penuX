# PenuX Task Board API

Dynamic task management backend with Node.js + Express + SQLite.

## Setup

```bash
cd tasks-api
npm install
npm start
```

Server runs on `http://localhost:3000`

## API Endpoints

- `POST /api/auth` — Password authentication
- `GET /api/tasks` — Fetch all tasks
- `POST /api/tasks` — Create task
- `PUT /api/tasks/:id` — Update task status
- `DELETE /api/tasks/:id` — Delete task
- `POST /api/tasks/:id/email` — Get a pre-filled email template
- `POST /api/tasks/:id/send-email` — Actually send the email via Gmail SMTP (requires setup below)

## Sending Emails ("Send Now" button)

The task board's ✉️ button opens a modal with a "🚀 Send Now" option that sends the
email for real via Gmail, from the server. This requires a Gmail App Password:

1. Enable 2-Step Verification on the sending Gmail account.
2. Go to https://myaccount.google.com/apppasswords and create an App Password
   (choose "Mail" / "Other").
3. Set these env vars on the server (or in `.env`, see `.env.example`):
   ```
   GMAIL_USER=your-address@gmail.com
   GMAIL_APP_PASSWORD=xxxxxxxxxxxxxxxx
   ```
4. Restart the server. Without these vars, `/api/tasks/:id/send-email` returns
   `503` and the "Copy to Clipboard" / "Open Gmail" buttons still work as a
   manual fallback.

Sending a task's follow-up email automatically marks that task **done**.

## Authentication

Default password: `penux2026` (SHA-256 hash)

## Database

SQLite database (`tasks.db`) auto-created on startup with default tasks.

## Deployment

### Railway

```bash
railway login
railway link
railway up
```

### Render

1. Connect GitHub repo
2. Select `tasks-api` as root directory
3. Set build command: `npm install`
4. Set start command: `npm start`

## Password Reset

To change password, update `PASSWORD_HASH` in `server.js` with new SHA-256 hash.

Generate hash:
```bash
node -e "console.log(require('crypto').createHash('sha256').update('newpassword').digest('hex'))"
```
