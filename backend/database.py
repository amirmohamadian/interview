import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "credits.db")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                api_key TEXT UNIQUE,
                created_at TEXT NOT NULL,
                CHECK (email IS NOT NULL OR api_key IS NOT NULL)
            );

            CREATE TABLE IF NOT EXISTS credit_balances (
                user_id INTEGER PRIMARY KEY REFERENCES users(id),
                balance INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                plan_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                credits_per_month INTEGER NOT NULL,
                current_period_start TEXT NOT NULL,
                current_period_end TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS free_trial_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_type TEXT NOT NULL,
                identity_value TEXT NOT NULL,
                claimed_at TEXT NOT NULL,
                UNIQUE(identity_type, identity_value)
            );

            CREATE TABLE IF NOT EXISTS credit_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                delta INTEGER NOT NULL,
                balance_after INTEGER NOT NULL,
                reason TEXT NOT NULL,
                reference_id TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                product_id TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                credits_granted INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'completed',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
            CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
            CREATE INDEX IF NOT EXISTS idx_ledger_user ON credit_ledger(user_id);
        """)


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_or_create_user(
    email: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    if not email and not api_key:
        raise ValueError("email or api_key required")

    with get_connection() as conn:
        row = None
        if email:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ?", (email,)
            ).fetchone()
        if not row and api_key:
            row = conn.execute(
                "SELECT * FROM users WHERE api_key = ?", (api_key,)
            ).fetchone()

        if row:
            user_id = row["id"]
            if email and not row["email"]:
                conn.execute(
                    "UPDATE users SET email = ? WHERE id = ?", (email, user_id)
                )
            if api_key and not row["api_key"]:
                conn.execute(
                    "UPDATE users SET api_key = ? WHERE id = ?", (api_key, user_id)
                )
        else:
            now = _utcnow()
            cur = conn.execute(
                "INSERT INTO users (email, api_key, created_at) VALUES (?, ?, ?)",
                (email, api_key, now),
            )
            user_id = cur.lastrowid
            conn.execute(
                "INSERT INTO credit_balances (user_id, balance, updated_at) VALUES (?, 0, ?)",
                (user_id, now),
            )

        return _user_row(conn, user_id)


def _user_row(conn: sqlite3.Connection, user_id: int) -> dict:
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    balance = conn.execute(
        "SELECT balance FROM credit_balances WHERE user_id = ?", (user_id,)
    ).fetchone()
    sub = conn.execute(
        """SELECT * FROM subscriptions
           WHERE user_id = ? AND status = 'active'
           ORDER BY id DESC LIMIT 1""",
        (user_id,),
    ).fetchone()
    return {
        "id": user["id"],
        "email": user["email"],
        "api_key": user["api_key"],
        "balance": balance["balance"] if balance else 0,
        "subscription": dict(sub) if sub else None,
    }


def has_claimed_free_trial(identity_type: str, identity_value: str) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            """SELECT 1 FROM free_trial_claims
               WHERE identity_type = ? AND identity_value = ?""",
            (identity_type, identity_value),
        ).fetchone()
        return row is not None


def claim_free_trial(identity_type: str, identity_value: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """INSERT OR IGNORE INTO free_trial_claims
               (identity_type, identity_value, claimed_at) VALUES (?, ?, ?)""",
            (identity_type, identity_value, _utcnow()),
        )


def add_credits(
    user_id: int,
    delta: int,
    reason: str,
    reference_id: Optional[str] = None,
) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT balance FROM credit_balances WHERE user_id = ?", (user_id,)
        ).fetchone()
        new_balance = (row["balance"] if row else 0) + delta
        now = _utcnow()
        conn.execute(
            """INSERT INTO credit_ledger
               (user_id, delta, balance_after, reason, reference_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, delta, new_balance, reason, reference_id, now),
        )
        conn.execute(
            """INSERT INTO credit_balances (user_id, balance, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET balance = ?, updated_at = ?""",
            (user_id, new_balance, now, new_balance, now),
        )
        return new_balance


def deduct_credit(user_id: int, reason: str = "exam") -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT balance FROM credit_balances WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row or row["balance"] < 1:
            raise ValueError("insufficient_credits")
        new_balance = row["balance"] - 1
        now = _utcnow()
        conn.execute(
            """INSERT INTO credit_ledger
               (user_id, delta, balance_after, reason, reference_id, created_at)
               VALUES (?, -1, ?, ?, NULL, ?)""",
            (user_id, new_balance, reason, now),
        )
        conn.execute(
            "UPDATE credit_balances SET balance = ?, updated_at = ? WHERE user_id = ?",
            (new_balance, now, user_id),
        )
        return new_balance


def record_purchase(
    user_id: int,
    product_id: str,
    amount_cents: int,
    credits_granted: int,
) -> int:
    with get_connection() as conn:
        now = _utcnow()
        cur = conn.execute(
            """INSERT INTO purchases
               (user_id, product_id, amount_cents, credits_granted, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, product_id, amount_cents, credits_granted, now),
        )
        return cur.lastrowid


def activate_subscription(
    user_id: int,
    plan_id: str,
    credits_per_month: int,
) -> None:
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    period_end = now + timedelta(days=30)
    with get_connection() as conn:
        conn.execute(
            "UPDATE subscriptions SET status = 'cancelled' WHERE user_id = ? AND status = 'active'",
            (user_id,),
        )
        conn.execute(
            """INSERT INTO subscriptions
               (user_id, plan_id, credits_per_month, current_period_start,
                current_period_end, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                plan_id,
                credits_per_month,
                now.isoformat(),
                period_end.isoformat(),
                now.isoformat(),
            ),
        )


def get_user_by_id(user_id: int) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return _user_row(conn, user_id)
