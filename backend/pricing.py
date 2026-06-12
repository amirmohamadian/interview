"""
Pricing & entitlement engine for the Resume ↔ JD "exam" product.

One "exam" = one resume/JD match analysis run.
Underlying cost ≈ $0.10 / exam, so every plan below keeps an 80%+ margin while
staying simple and conversion-friendly.

This module is intentionally self-contained (no Vosk / model dependency) so it
can run and be tested on its own:

    uvicorn backend.pricing:app --reload

Design goals (from product owner):
  * Highest possible conversion — a free, no-card hook + an obvious "best value".
  * 1 free exam for every new email OR API key.
  * Subscription is the main funnel; when a subscriber burns their monthly
    credits they can top-up *cheaper* than the pay-as-you-go bundles.
  * An "unlimited" option, protected by a fair-use cap of 20 exams/day.
  * Credit bundles for people who just want to test without subscribing.
  * Nothing ever exceeds a value of 20 (max bundle = 20, monthly allowance = 20,
    daily fair-use cap = 20).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Pricing catalog — single source of truth (backend + frontend both use this). #
# --------------------------------------------------------------------------- #

COST_PER_EXAM = 0.10          # what *we* pay per analysis (USD)
MAX_VALUE = 20                # hard cap: no plan/bundle/day exceeds this
UNLIMITED_DAILY_CAP = 20      # fair-use cap for the Unlimited plan
FREE_EXAMS = 1                # free exams granted per new email / API key
TOPUP_PRICE = 0.50            # subscriber top-up price per exam (cheaper than bundles)

CURRENCY = "USD"

PRICING = {
    "currency": CURRENCY,
    "cost_per_exam": COST_PER_EXAM,
    "max_value": MAX_VALUE,
    "free": {
        "id": "free",
        "name": "Free trial",
        "price": 0.0,
        "exams": FREE_EXAMS,
        "tagline": "1 free exam — no credit card needed",
        "perks": [
            "1 full resume ↔ job match analysis",
            "Detailed score breakdown",
            "No card, no commitment",
        ],
    },
    # One-time credit bundles for people who want to test without subscribing.
    "bundles": [
        {
            "id": "bundle_starter",
            "name": "Starter pack",
            "price": 5.0,
            "exams": 5,
            "per_exam": 1.00,
            "tagline": "Just testing things out",
            "perks": ["5 exams", "Credits valid 12 months", "No subscription"],
        },
        {
            "id": "bundle_value",
            "name": "Value pack",
            "price": 15.0,
            "exams": 20,            # max value
            "per_exam": 0.75,
            "badge": "Best value bundle",
            "tagline": "Best price without subscribing",
            "perks": ["20 exams", "Credits valid 12 months", "Save 25% vs Starter"],
        },
    ],
    # Recurring subscriptions — the main conversion target.
    "subscriptions": [
        {
            "id": "sub_pro",
            "name": "Pro",
            "price_monthly": 9.0,
            "price_annual": 90.0,           # 2 months free
            "monthly_exams": 20,            # max value
            "per_exam_effective": 0.45,
            "topup_price": TOPUP_PRICE,     # cheaper than any bundle once exhausted
            "badge": "Most popular",
            "tagline": "For active job seekers",
            "perks": [
                "20 exams every month",
                f"Out of credits? Top-up at ${TOPUP_PRICE:.2f}/exam (cheaper than bundles)",
                "Priority processing",
                "Cancel anytime",
            ],
        },
        {
            "id": "sub_unlimited",
            "name": "Unlimited",
            "price_monthly": 19.0,
            "price_annual": 190.0,          # 2 months free
            "monthly_exams": None,          # unlimited
            "daily_cap": UNLIMITED_DAILY_CAP,
            "tagline": "For coaches & power users",
            "perks": [
                f"Unlimited exams (fair use: {UNLIMITED_DAILY_CAP}/day)",
                "Priority processing",
                "Cancel anytime",
            ],
        },
    ],
    "topup": {
        "id": "topup",
        "name": "Subscriber top-up",
        "price_per_exam": TOPUP_PRICE,
        "tagline": "Ran out this month? Keep going for less.",
        "requires_subscription": True,
    },
}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _now_ts() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# Account model + entitlement engine                                          #
# --------------------------------------------------------------------------- #


@dataclass
class Account:
    """An account, identified by email and/or API key."""

    id: str
    email: Optional[str] = None
    api_key: Optional[str] = None

    free_used: bool = False              # has the 1 free exam been spent?
    credits: int = 0                     # pay-as-you-go bundle / top-up credits

    plan: Optional[str] = None           # None | "sub_pro" | "sub_unlimited"
    period_start_ts: float = 0.0         # when the current monthly period began
    monthly_used: int = 0                # exams used this billing period

    today: str = field(default_factory=_today)
    today_used: int = 0                  # exams used today (for daily cap)

    history: list = field(default_factory=list)

    # --- period helpers ----------------------------------------------------- #
    def _roll_period(self) -> None:
        """Reset the monthly counter every 30 days."""
        if self.plan and self.period_start_ts:
            if _now_ts() - self.period_start_ts >= 30 * 24 * 3600:
                self.period_start_ts = _now_ts()
                self.monthly_used = 0

    def _roll_day(self) -> None:
        if self.today != _today():
            self.today = _today()
            self.today_used = 0

    # --- read-only views ----------------------------------------------------- #
    def monthly_allowance(self) -> Optional[int]:
        if self.plan == "sub_pro":
            return PRICING["subscriptions"][0]["monthly_exams"]
        if self.plan == "sub_unlimited":
            return None  # unlimited
        return 0

    def remaining_monthly(self) -> Optional[int]:
        allowance = self.monthly_allowance()
        if allowance is None:
            return None  # unlimited
        return max(0, allowance - self.monthly_used)

    def can_consume(self) -> tuple[bool, str]:
        """Return (allowed, source) without mutating state."""
        self._roll_period()
        self._roll_day()

        # Unlimited plan: only the daily fair-use cap applies.
        if self.plan == "sub_unlimited":
            if self.today_used >= UNLIMITED_DAILY_CAP:
                return False, "daily_cap_reached"
            return True, "subscription"

        # Subscription with a monthly allowance still available.
        if self.plan == "sub_pro" and (self.remaining_monthly() or 0) > 0:
            return True, "subscription"

        # Free exam.
        if not self.free_used:
            return True, "free"

        # Pay-as-you-go / top-up credits.
        if self.credits > 0:
            return True, "credits"

        return False, "no_entitlement"


class Store:
    """Tiny persistence layer. JSON file when a path is given, else in-memory."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.accounts: Dict[str, Account] = {}
        self._by_email: Dict[str, str] = {}
        self._by_api: Dict[str, str] = {}
        if path and os.path.exists(path):
            self._load()

    # -- lookup --------------------------------------------------------------- #
    def find(self, email: Optional[str] = None, api_key: Optional[str] = None) -> Optional[Account]:
        if email and email.lower() in self._by_email:
            return self.accounts[self._by_email[email.lower()]]
        if api_key and api_key in self._by_api:
            return self.accounts[self._by_api[api_key]]
        return None

    def get(self, account_id: str) -> Account:
        acc = self.accounts.get(account_id)
        if not acc:
            raise KeyError(account_id)
        return acc

    def get_or_create(self, email: Optional[str] = None, api_key: Optional[str] = None) -> Account:
        if not email and not api_key:
            raise ValueError("email or api_key required")
        existing = self.find(email=email, api_key=api_key)
        if existing:
            # Attach a new identifier if one was missing.
            if email and not existing.email:
                existing.email = email
                self._by_email[email.lower()] = existing.id
            if api_key and not existing.api_key:
                existing.api_key = api_key
                self._by_api[api_key] = existing.id
            self._save()
            return existing

        acc = Account(id=uuid.uuid4().hex, email=email, api_key=api_key)
        self.accounts[acc.id] = acc
        if email:
            self._by_email[email.lower()] = acc.id
        if api_key:
            self._by_api[api_key] = acc.id
        self._save()
        return acc

    # -- mutations ------------------------------------------------------------ #
    def consume(self, acc: Account) -> str:
        allowed, source = acc.can_consume()
        if not allowed:
            raise PermissionError(source)
        if source == "free":
            acc.free_used = True
        elif source == "subscription":
            acc.monthly_used += 1
            acc.today_used += 1
        elif source == "credits":
            acc.credits -= 1
            acc.today_used += 1
        acc.history.append({"ts": _now_ts(), "source": source})
        self._save()
        return source

    def add_credits(self, acc: Account, n: int) -> None:
        acc.credits += n
        self._save()

    def subscribe(self, acc: Account, plan: str) -> None:
        if plan not in ("sub_pro", "sub_unlimited"):
            raise ValueError("unknown plan")
        acc.plan = plan
        acc.period_start_ts = _now_ts()
        acc.monthly_used = 0
        self._save()

    # -- persistence ---------------------------------------------------------- #
    def _save(self) -> None:
        if not self.path:
            return
        data = {aid: asdict(a) for aid, a in self.accounts.items()}
        with open(self.path, "w") as fh:
            json.dump(data, fh, indent=2)

    def _load(self) -> None:
        with open(self.path) as fh:
            data = json.load(fh)
        for aid, a in data.items():
            acc = Account(**a)
            self.accounts[aid] = acc
            if acc.email:
                self._by_email[acc.email.lower()] = aid
            if acc.api_key:
                self._by_api[acc.api_key] = aid


# --------------------------------------------------------------------------- #
# FastAPI surface                                                             #
# --------------------------------------------------------------------------- #

STORE = Store(path=os.environ.get("PRICING_DB"))

app = FastAPI(title="Resume↔JD Pricing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Identity(BaseModel):
    email: Optional[str] = None
    api_key: Optional[str] = None


class BundleReq(Identity):
    bundle_id: str


class SubscribeReq(Identity):
    plan: str


class TopupReq(Identity):
    exams: int = 10


def _account_view(acc: Account) -> dict:
    acc._roll_period()
    acc._roll_day()
    allowed, source = acc.can_consume()
    return {
        "id": acc.id,
        "email": acc.email,
        "has_api_key": bool(acc.api_key),
        "free_used": acc.free_used,
        "credits": acc.credits,
        "plan": acc.plan,
        "remaining_monthly": acc.remaining_monthly(),
        "today_used": acc.today_used,
        "next_exam_allowed": allowed,
        "next_exam_source": source,
    }


@app.get("/pricing")
def get_pricing():
    return PRICING


@app.post("/signup")
def signup(idy: Identity):
    if not idy.email and not idy.api_key:
        raise HTTPException(400, "email or api_key required")
    acc = STORE.get_or_create(email=idy.email, api_key=idy.api_key)
    return _account_view(acc)


@app.get("/account")
def account(email: Optional[str] = None, api_key: Optional[str] = None):
    acc = STORE.find(email=email, api_key=api_key)
    if not acc:
        raise HTTPException(404, "account not found")
    return _account_view(acc)


@app.post("/consume")
def consume(idy: Identity):
    acc = STORE.get_or_create(email=idy.email, api_key=idy.api_key)
    try:
        source = STORE.consume(acc)
    except PermissionError as exc:
        # 402 Payment Required — out of entitlements, nudge to buy/subscribe.
        raise HTTPException(402, detail={"reason": str(exc), "account": _account_view(acc)})
    return {"ok": True, "charged_to": source, "account": _account_view(acc)}


@app.post("/purchase/bundle")
def purchase_bundle(req: BundleReq):
    bundle = next((b for b in PRICING["bundles"] if b["id"] == req.bundle_id), None)
    if not bundle:
        raise HTTPException(404, "unknown bundle")
    acc = STORE.get_or_create(email=req.email, api_key=req.api_key)
    STORE.add_credits(acc, bundle["exams"])
    return {"ok": True, "added": bundle["exams"], "account": _account_view(acc)}


@app.post("/subscribe")
def subscribe(req: SubscribeReq):
    acc = STORE.get_or_create(email=req.email, api_key=req.api_key)
    try:
        STORE.subscribe(acc, req.plan)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": True, "account": _account_view(acc)}


@app.post("/topup")
def topup(req: TopupReq):
    acc = STORE.get_or_create(email=req.email, api_key=req.api_key)
    if not acc.plan:
        raise HTTPException(403, "top-up is only available to subscribers")
    n = max(1, min(req.exams, MAX_VALUE))  # respect the max-value cap
    STORE.add_credits(acc, n)
    return {
        "ok": True,
        "added": n,
        "charged": round(n * TOPUP_PRICE, 2),
        "account": _account_view(acc),
    }
