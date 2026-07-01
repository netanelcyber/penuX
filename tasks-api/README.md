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
