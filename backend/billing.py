"""
Billing, credits and conversion-optimized pricing for the exam platform.

Monetization model (designed for maximum conversion):

1. FREE TRIAL  - 1 free exam per email AND per IP/API key (no credit card).
                 Dedupe on both email and IP prevents farming while keeping
                 sign-up friction at the absolute minimum.

2. SUBSCRIPTIONS - 3 tiers with anchoring. The middle "Pro" tier is the
                 conversion target ("Most popular"). "Unlimited" is capped
                 at a fair-use maximum of 20 exams per month.

3. SUBSCRIBER TOP-UPS - when a subscriber burns through their monthly
                 credits they can buy extra exams at a discounted per-exam
                 price, cheaper than any bundle and cheaper the higher
                 their tier. This rewards subscribing instead of punishing
                 heavy usage.

4. CREDIT BUNDLES - one-time packs for users who don't want a subscription.
                 Sizes 3 / 8 / 15. The maximum bundle is 15: large enough
                 for a strong "best value" anchor, small enough that heavy
                 users are nudged toward a subscription (which converts to
                 recurring revenue). Bundle credits never expire.
"""

import os
import re
import time
import uuid
import sqlite3
import secrets
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.environ.get("BILLING_DB", os.path.join(os.path.dirname(__file__), "billing.db"))

BILLING_CYCLE_SECONDS = 30 * 24 * 3600  # monthly cycle
SINGLE_EXAM_ANCHOR = 9.00               # price anchor shown crossed-out in UI

# ---- Subscription plans -------------------------------------------------
# "topup_price" = discounted per-exam price available ONLY to subscribers
# of that plan once their monthly credits run out (always cheaper than the
# cheapest bundle rate of $6.00/exam).
PLANS = {
    "starter": {
        "id": "starter",
        "name": "Starter",
        "price_month": 19.00,
        "exams_per_month": 5,
        "topup_price": 4.50,
        "badge": None,
        "tagline": "Perfect to get interview-ready",
    },
    "pro": {
        "id": "pro",
        "name": "Pro",
        "price_month": 39.00,
        "exams_per_month": 12,
        "topup_price": 4.00,
        "badge": "MOST POPULAR",
        "tagline": "Serious prep for serious offers",
    },
    "unlimited": {
        "id": "unlimited",
        "name": "Unlimited",
        "price_month": 59.00,
        "exams_per_month": 20,  # fair-use cap: max 20 exams / month
        "topup_price": 3.50,
        "badge": "BEST FOR HEAVY PREP",
        "tagline": "Practice as much as you want*",
        "fair_use_note": "*Fair use: up to 20 exams per month. Need more? Top up at our lowest rate ever, $3.50/exam.",
    },
}

# ---- One-time credit bundles (no subscription required) ------------------
# Max bundle size = 15 (chosen over 10/20: 15 gives the strongest
# save-% anchor while keeping 16+ exam users pointed at subscriptions).
BUNDLES = {
    "bundle_3": {
        "id": "bundle_3",
        "name": "Quick Pack",
        "credits": 3,
        "price": 24.00,   # $8.00 / exam
        "badge": None,
        "tagline": "Try it properly",
    },
    "bundle_8": {
        "id": "bundle_8",
        "name": "Prep Pack",
        "credits": 8,
        "price": 56.00,   # $7.00 / exam
        "badge": "POPULAR",
        "tagline": "Cover every topic",
    },
    "bundle_15": {
        "id": "bundle_15",
        "name": "Max Pack",
        "credits": 15,
        "price": 90.00,   # $6.00 / exam
        "badge": "BEST VALUE \u2013 SAVE 33%",
        "tagline": "Our biggest pack",
    },
}

MAX_TOPUP_PER_PURCHASE = 20

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            api_key TEXT UNIQUE NOT NULL,
            free_credits INTEGER NOT NULL DEFAULT 0,
            extra_credits INTEGER NOT NULL DEFAULT 0,   -- bundles + top-ups, never expire
            plan TEXT,                                  -- starter / pro / unlimited / NULL
            cycle_start REAL,                           -- start of current billing cycle
            monthly_used INTEGER NOT NULL DEFAULT 0,    -- exams used this cycle
            created_at REAL NOT NULL
        );
        -- one free exam per email AND per IP: both recorded here
        CREATE TABLE IF NOT EXISTS free_grants (
            kind TEXT NOT NULL,        -- 'email' or 'ip'
            value TEXT NOT NULL,
            granted_at REAL NOT NULL,
            PRIMARY KEY (kind, value)
        );
        CREATE TABLE IF NOT EXISTS purchases (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES accounts(id),
            kind TEXT NOT NULL,        -- 'subscription' / 'bundle' / 'topup'
            item TEXT NOT NULL,
            credits INTEGER NOT NULL,
            amount REAL NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS exam_sessions (
            token TEXT PRIMARY KEY,
            account_id TEXT NOT NULL REFERENCES accounts(id),
            source TEXT NOT NULL,      -- which credit pool paid for it
            created_at REAL NOT NULL,
            used INTEGER NOT NULL DEFAULT 0
        );
        """)


# ---------------------------------------------------------------------------


class BillingError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def _renew_cycle_if_due(conn, acc) -> dict:
    """Reset monthly usage when a new billing cycle starts (simulated renewal)."""
    acc = dict(acc)
    if acc["plan"] and acc["cycle_start"] is not None:
        now = time.time()
        if now - acc["cycle_start"] >= BILLING_CYCLE_SECONDS:
            # advance whole cycles, reset usage
            cycles = int((now - acc["cycle_start"]) // BILLING_CYCLE_SECONDS)
            acc["cycle_start"] = acc["cycle_start"] + cycles * BILLING_CYCLE_SECONDS
            acc["monthly_used"] = 0
            conn.execute(
                "UPDATE accounts SET cycle_start=?, monthly_used=0 WHERE id=?",
                (acc["cycle_start"], acc["id"]),
            )
    return acc


def _get_account(conn, api_key: str) -> dict:
    row = conn.execute("SELECT * FROM accounts WHERE api_key=?", (api_key,)).fetchone()
    if not row:
        raise BillingError("invalid_key", "Unknown API key. Sign up to get your free exam.")
    return _renew_cycle_if_due(conn, row)


def signup_or_login(email: str, ip: str) -> dict:
    """Create (or fetch) an account. Grants 1 free exam if neither this
    email nor this IP has claimed one before."""
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise BillingError("bad_email", "Please enter a valid email address.")

    with db() as conn:
        row = conn.execute("SELECT * FROM accounts WHERE email=?", (email,)).fetchone()
        if row:
            acc = _renew_cycle_if_due(conn, row)
            return account_summary(acc, new=False)

        now = time.time()
        email_used = conn.execute(
            "SELECT 1 FROM free_grants WHERE kind='email' AND value=?", (email,)
        ).fetchone()
        ip_used = conn.execute(
            "SELECT 1 FROM free_grants WHERE kind='ip' AND value=?", (ip,)
        ).fetchone()
        free = 0
        if not email_used and not ip_used:
            free = 1
            conn.execute("INSERT INTO free_grants VALUES ('email', ?, ?)", (email, now))
            conn.execute(
                "INSERT OR IGNORE INTO free_grants VALUES ('ip', ?, ?)", (ip, now)
            )

        acc_id = str(uuid.uuid4())
        api_key = "exm_" + secrets.token_urlsafe(24)
        conn.execute(
            "INSERT INTO accounts (id, email, api_key, free_credits, created_at) VALUES (?,?,?,?,?)",
            (acc_id, email, api_key, free, now),
        )
        acc = conn.execute("SELECT * FROM accounts WHERE id=?", (acc_id,)).fetchone()
        return account_summary(dict(acc), new=True)


def account_summary(acc: dict, new: bool = False) -> dict:
    acc = dict(acc)
    plan = PLANS.get(acc["plan"]) if acc["plan"] else None
    monthly_left = 0
    if plan:
        monthly_left = max(0, plan["exams_per_month"] - acc["monthly_used"])
    total = acc["free_credits"] + acc["extra_credits"] + monthly_left
    return {
        "new_account": new,
        "email": acc["email"],
        "api_key": acc["api_key"],
        "plan": acc["plan"],
        "plan_name": plan["name"] if plan else None,
        "free_credits": acc["free_credits"],
        "extra_credits": acc["extra_credits"],
        "monthly_credits_left": monthly_left,
        "monthly_used": acc["monthly_used"],
        "monthly_allowance": plan["exams_per_month"] if plan else 0,
        "total_credits": total,
        "topup_price": plan["topup_price"] if plan else None,
        "cycle_renews_at": (acc["cycle_start"] + BILLING_CYCLE_SECONDS) if acc["cycle_start"] else None,
    }


def get_summary(api_key: str) -> dict:
    with db() as conn:
        return account_summary(_get_account(conn, api_key))


def subscribe(api_key: str, plan_id: str) -> dict:
    plan = PLANS.get(plan_id)
    if not plan:
        raise BillingError("bad_plan", "Unknown plan.")
    with db() as conn:
        acc = _get_account(conn, api_key)
        if acc["plan"] == plan_id:
            raise BillingError("already_subscribed", f"You're already on {plan['name']}.")
        now = time.time()
        # NOTE: integrate Stripe Checkout here in production; demo charges instantly.
        conn.execute(
            "UPDATE accounts SET plan=?, cycle_start=?, monthly_used=0 WHERE id=?",
            (plan_id, now, acc["id"]),
        )
        conn.execute(
            "INSERT INTO purchases VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), acc["id"], "subscription", plan_id,
             plan["exams_per_month"], plan["price_month"], now),
        )
        return account_summary(_get_account(conn, api_key))


def buy_bundle(api_key: str, bundle_id: str) -> dict:
    bundle = BUNDLES.get(bundle_id)
    if not bundle:
        raise BillingError("bad_bundle", "Unknown bundle.")
    with db() as conn:
        acc = _get_account(conn, api_key)
        now = time.time()
        conn.execute(
            "UPDATE accounts SET extra_credits = extra_credits + ? WHERE id=?",
            (bundle["credits"], acc["id"]),
        )
        conn.execute(
            "INSERT INTO purchases VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), acc["id"], "bundle", bundle_id,
             bundle["credits"], bundle["price"], now),
        )
        return account_summary(_get_account(conn, api_key))


def buy_topup(api_key: str, quantity: int) -> dict:
    """Discounted extra exams, ONLY for active subscribers (the 'cheaper
    when you run out' perk)."""
    if not isinstance(quantity, int) or quantity < 1 or quantity > MAX_TOPUP_PER_PURCHASE:
        raise BillingError("bad_quantity", f"Top-ups are 1-{MAX_TOPUP_PER_PURCHASE} exams per purchase.")
    with db() as conn:
        acc = _get_account(conn, api_key)
        plan = PLANS.get(acc["plan"]) if acc["plan"] else None
        if not plan:
            raise BillingError(
                "subscribers_only",
                "Discounted top-ups are a subscriber perk. Subscribe, or grab a credit bundle instead.",
            )
        now = time.time()
        amount = round(plan["topup_price"] * quantity, 2)
        conn.execute(
            "UPDATE accounts SET extra_credits = extra_credits + ? WHERE id=?",
            (quantity, acc["id"]),
        )
        conn.execute(
            "INSERT INTO purchases VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), acc["id"], "topup", f"topup_x{quantity}", quantity, amount, now),
        )
        summary = account_summary(_get_account(conn, api_key))
        summary["charged"] = amount
        return summary


def start_exam(api_key: str) -> dict:
    """Consume one credit (free -> monthly -> extra) and mint a session token
    that authorizes one websocket exam session."""
    with db() as conn:
        acc = _get_account(conn, api_key)
        plan = PLANS.get(acc["plan"]) if acc["plan"] else None
        monthly_left = max(0, plan["exams_per_month"] - acc["monthly_used"]) if plan else 0

        if acc["free_credits"] > 0:
            source = "free"
            conn.execute("UPDATE accounts SET free_credits = free_credits - 1 WHERE id=?", (acc["id"],))
        elif monthly_left > 0:
            source = "subscription"
            conn.execute("UPDATE accounts SET monthly_used = monthly_used + 1 WHERE id=?", (acc["id"],))
        elif acc["extra_credits"] > 0:
            source = "extra"
            conn.execute("UPDATE accounts SET extra_credits = extra_credits - 1 WHERE id=?", (acc["id"],))
        else:
            if plan:
                raise BillingError(
                    "out_of_credits_subscriber",
                    f"You've used all {plan['exams_per_month']} exams this month. "
                    f"Top up at your subscriber rate of ${plan['topup_price']:.2f}/exam.",
                )
            raise BillingError(
                "out_of_credits",
                "You're out of exam credits. Subscribe or grab a credit bundle to keep practicing.",
            )

        token = "sess_" + secrets.token_urlsafe(24)
        conn.execute(
            "INSERT INTO exam_sessions (token, account_id, source, created_at) VALUES (?,?,?,?)",
            (token, acc["id"], source, time.time()),
        )
        summary = account_summary(_get_account(conn, api_key))
        summary["session_token"] = token
        summary["credit_source"] = source
        return summary


def validate_session(token: str) -> bool:
    """Validate and consume an exam session token for the websocket."""
    if not token:
        return False
    with db() as conn:
        row = conn.execute(
            "SELECT used FROM exam_sessions WHERE token=?", (token,)
        ).fetchone()
        if not row or row["used"]:
            return False
        conn.execute("UPDATE exam_sessions SET used=1 WHERE token=?", (token,))
        return True


def pricing_catalog() -> dict:
    return {
        "single_exam_anchor": SINGLE_EXAM_ANCHOR,
        "free_trial": {"exams": 1, "per": "email / API key", "card_required": False},
        "plans": [
            {**p, "per_exam": round(p["price_month"] / p["exams_per_month"], 2)}
            for p in PLANS.values()
        ],
        "bundles": [
            {**b, "per_exam": round(b["price"] / b["credits"], 2),
             "save_pct": round(100 * (1 - (b["price"] / b["credits"]) / SINGLE_EXAM_ANCHOR))}
            for b in BUNDLES.values()
        ],
        "topup": {
            "subscribers_only": True,
            "max_per_purchase": MAX_TOPUP_PER_PURCHASE,
            "prices": {pid: p["topup_price"] for pid, p in PLANS.items()},
        },
    }
