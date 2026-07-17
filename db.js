/**
 * Database layer using @libsql/client instead of sqlite3.
 *
 * Locally (no env vars set) this transparently uses a local file
 * (tasks.db) via libSQL's embedded mode — behaves just like sqlite3 for
 * development, no Turso account needed.
 *
 * In production, set TURSO_DATABASE_URL (libsql://...) and
 * TURSO_AUTH_TOKEN to point at a free Turso database instead — this is
 * what makes task data survive restarts/redeploys on free hosts like
 * Render, which don't offer a persistent local disk on their free tier.
 *
 * wrapCompat() exposes a sqlite3.Database-compatible callback API
 * (.run/.get/.all) so the rest of the app (server.js, reply-watcher.js)
 * didn't need to be rewritten to promises.
 */
import { createClient } from '@libsql/client';

export function getClient() {
  const url = process.env.TURSO_DATABASE_URL || 'file:tasks.db';
  const authToken = process.env.TURSO_AUTH_TOKEN;
  return createClient(authToken ? { url, authToken } : { url });
}

function normalizeArgs(params, cb) {
  if (typeof params === 'function') return { args: [], callback: params };
  return { args: params || [], callback: cb };
}

export function wrapCompat(client) {
  return {
    run(sql, params, cb) {
      const { args, callback } = normalizeArgs(params, cb);
      client.execute({ sql, args })
        .then(result => {
          if (callback) {
            callback.call(
              { lastID: Number(result.lastInsertRowid ?? 0), changes: result.rowsAffected ?? 0 },
              null
            );
          }
        })
        .catch(err => {
          if (callback) callback(err);
          else console.error('DB run error:', err);
        });
    },
    get(sql, params, cb) {
      const { args, callback } = normalizeArgs(params, cb);
      client.execute({ sql, args })
        .then(result => callback(null, result.rows[0]))
        .catch(err => callback(err));
    },
    all(sql, params, cb) {
      const { args, callback } = normalizeArgs(params, cb);
      client.execute({ sql, args })
        .then(result => callback(null, result.rows))
        .catch(err => callback(err));
    },
  };
}
