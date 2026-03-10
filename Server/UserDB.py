"""User database backed by SQLite with bcrypt password hashing.

Schema:
    users(id INTEGER PK, username TEXT UNIQUE, password_hash TEXT,
          display_name TEXT, role TEXT DEFAULT 'user', created_at TEXT)
    refresh_tokens(id INTEGER PK, user_id INTEGER FK, token TEXT UNIQUE,
                   expires_at TEXT, revoked INTEGER DEFAULT 0)

Usage:
    db = UserDB("/path/to/users.db")
    db.create_user("alice", "s3cret", display_name="Alice", role="admin")
    user = db.authenticate("alice", "s3cret")  # returns dict or None
"""

import sqlite3
import bcrypt
import os
import datetime
import threading
import Constants


class UserDB:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = os.environ.get("USER_DB_PATH", Constants.GetUserDatabasePath())
        if not os.path.exists(os.path.dirname(db_path) or "."):
            raise RuntimeError(
                f"User database directory does not exist: {os.path.dirname(db_path)}\n"
                f"Expected database at: {db_path}\n"
                f"Set DATA_BASE_DIRECTORY environment variable or pass db_path explicitly."
            )
        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread connection (SQLite is not thread-safe by default)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_schema(self):
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                display_name TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS refresh_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                revoked INTEGER NOT NULL DEFAULT 0
            );
        """)
        conn.commit()

    # ---- User CRUD ----

    def create_user(self, username: str, password: str, *, display_name: str | None = None, role: str = "user") -> dict:
        """Create a new user. Returns the user dict. Raises ValueError on duplicate."""
        pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn = self._conn()
        try:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
                (username, pw_hash, display_name or username, role),
            )
            conn.commit()
            return self.get_user_by_id(cur.lastrowid)
        except sqlite3.IntegrityError:
            raise ValueError(f"User '{username}' already exists")

    def get_user_by_id(self, user_id: int) -> dict | None:
        row = self._conn().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def get_user_by_username(self, username: str) -> dict | None:
        row = self._conn().execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,)).fetchone()
        return dict(row) if row else None

    def list_users(self) -> list[dict]:
        rows = self._conn().execute("SELECT id, username, display_name, role, created_at FROM users ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    def delete_user(self, username: str) -> bool:
        conn = self._conn()
        cur = conn.execute("DELETE FROM users WHERE username = ? COLLATE NOCASE", (username,))
        conn.commit()
        return cur.rowcount > 0

    def reset_password(self, username: str, new_password: str) -> bool:
        pw_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        conn = self._conn()
        cur = conn.execute("UPDATE users SET password_hash = ? WHERE username = ? COLLATE NOCASE", (pw_hash, username))
        conn.commit()
        return cur.rowcount > 0

    # ---- Authentication ----

    def authenticate(self, username: str, password: str) -> dict | None:
        """Verify credentials. Returns user dict (without password_hash) or None."""
        user = self.get_user_by_username(username)
        if user is None:
            return None
        if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
            return None
        safe = {k: v for k, v in user.items() if k != "password_hash"}
        return safe

    # ---- Refresh token management ----

    def store_refresh_token(self, user_id: int, token: str, expires_at: datetime.datetime):
        conn = self._conn()
        conn.execute(
            "INSERT INTO refresh_tokens (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at.isoformat()),
        )
        conn.commit()

    def validate_refresh_token(self, token: str) -> dict | None:
        """Return the associated user if token is valid and not revoked/expired."""
        conn = self._conn()
        row = conn.execute(
            "SELECT rt.*, u.id as uid, u.username, u.display_name, u.role "
            "FROM refresh_tokens rt JOIN users u ON rt.user_id = u.id "
            "WHERE rt.token = ? AND rt.revoked = 0",
            (token,),
        ).fetchone()
        if row is None:
            return None
        expires = datetime.datetime.fromisoformat(row["expires_at"])
        now = datetime.datetime.now(datetime.timezone.utc)
        # Ensure both are timezone-aware for comparison
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=datetime.timezone.utc)
        if expires < now:
            return None
        return {"id": row["uid"], "username": row["username"], "display_name": row["display_name"], "role": row["role"]}

    def revoke_refresh_token(self, token: str):
        conn = self._conn()
        conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE token = ?", (token,))
        conn.commit()

    def revoke_all_user_tokens(self, user_id: int):
        conn = self._conn()
        conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ?", (user_id,))
        conn.commit()

    def cleanup_expired_tokens(self):
        """Remove expired or revoked tokens to keep the table small."""
        conn = self._conn()
        conn.execute(
            "DELETE FROM refresh_tokens WHERE revoked = 1 OR expires_at < datetime('now')"
        )
        conn.commit()
