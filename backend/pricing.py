"""
Pricing configuration optimized for conversion.

Funnel: Free trial → Subscribe (primary) or Credit bundles (try without commitment)
        → Subscriber refills at ~40% off when credits run out.

Bundle ladder uses 10 / 15 / 20 credits (3-tier "good / better / best" for conversion).
Unlimited subscription = up to 20 exams/month fair-use cap.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


MAX_EXAMS = 20
FREE_EXAMS_PER_IDENTITY = 1

# Bundle sizes chosen for conversion: 3 tiers, middle highlighted, max = 20.
BUNDLE_SIZES = (10, 15, 20)
REFILL_SIZES = (5, 10, 15, 20)  # includes 5-credit impulse top-up when credits hit zero


class ProductType(str, Enum):
    FREE = "free"
    BUNDLE = "bundle"
    SUBSCRIPTION = "subscription"
    SUBSCRIBER_REFILL = "subscriber_refill"


@dataclass(frozen=True)
class Product:
    id: str
    name: str
    product_type: ProductType
    credits: int
    price_cents: int
    interval: Optional[str] = None  # "month" for subscriptions
    badge: Optional[str] = None
    description: str = ""
    highlight: bool = False

    @property
    def price_display(self) -> str:
        return f"${self.price_cents / 100:.2f}"

    @property
    def per_exam_cents(self) -> int:
        if self.credits == 0:
            return 0
        return self.price_cents // self.credits

    @property
    def per_exam_display(self) -> str:
        if self.credits == 0:
            return "Free"
        return f"${self.per_exam_cents / 100:.2f}/exam"


# ── Free tier ──────────────────────────────────────────────────────────────
FREE_PRODUCT = Product(
    id="free",
    name="Free Trial",
    product_type=ProductType.FREE,
    credits=FREE_EXAMS_PER_IDENTITY,
    price_cents=0,
    description="1 free exam per email or API key — no card required",
)

# ── Credit bundles (no subscription — test without commitment) ─────────────
# Per-exam: $2.20 → $2.00 → $1.90 (subscriptions stay clearly cheaper)
CREDIT_BUNDLES: list[Product] = [
    Product(
        id="bundle_10",
        name="Try Pack",
        product_type=ProductType.BUNDLE,
        credits=10,
        price_cents=2200,
        description="Enough to evaluate the platform before committing",
    ),
    Product(
        id="bundle_15",
        name="Practice Pack",
        product_type=ProductType.BUNDLE,
        credits=15,
        price_cents=3000,
        badge="Popular",
        description="Sweet spot for a focused interview prep sprint",
        highlight=True,
    ),
    Product(
        id="bundle_20",
        name="Max Pack",
        product_type=ProductType.BUNDLE,
        credits=MAX_EXAMS,
        price_cents=3800,
        badge="Best Value",
        description="Maximum credits without a subscription",
    ),
]

# ── Subscriptions (primary conversion path) ────────────────────────────────
SUBSCRIPTIONS: list[Product] = [
    Product(
        id="sub_starter",
        name="Starter",
        product_type=ProductType.SUBSCRIPTION,
        credits=5,
        price_cents=900,
        interval="month",
        description="5 mock exams per month — great for steady practice",
    ),
    Product(
        id="sub_pro",
        name="Pro",
        product_type=ProductType.SUBSCRIPTION,
        credits=10,
        price_cents=1500,
        interval="month",
        badge="Most Popular",
        description="10 exams/month — ideal for active job seekers",
        highlight=True,
    ),
    Product(
        id="sub_unlimited",
        name="Unlimited",
        product_type=ProductType.SUBSCRIPTION,
        credits=MAX_EXAMS,
        price_cents=2500,
        interval="month",
        badge="Power User",
        description=f"Up to {MAX_EXAMS} exams/month — unlimited feel, fair-use cap",
    ),
]

# ── Subscriber refills (~40% cheaper than bundles when credits run out) ───
SUBSCRIBER_REFILLS: list[Product] = [
    Product(
        id="refill_5",
        name="Quick Refill",
        product_type=ProductType.SUBSCRIBER_REFILL,
        credits=5,
        price_cents=600,
        description="Instant top-up when you run out mid-cycle",
    ),
    Product(
        id="refill_10",
        name="Standard Refill",
        product_type=ProductType.SUBSCRIBER_REFILL,
        credits=10,
        price_cents=1300,
        description="~41% cheaper per exam than Try Pack bundles",
    ),
    Product(
        id="refill_15",
        name="Power Refill",
        product_type=ProductType.SUBSCRIBER_REFILL,
        credits=15,
        price_cents=1800,
        badge="Best Refill",
        description="Best per-exam rate for active subscribers",
        highlight=True,
    ),
    Product(
        id="refill_20",
        name="Max Refill",
        product_type=ProductType.SUBSCRIBER_REFILL,
        credits=MAX_EXAMS,
        price_cents=2200,
        description="Maximum refill — lowest per-exam price available",
    ),
]

ALL_PRODUCTS: dict[str, Product] = {
    p.id: p
    for p in [FREE_PRODUCT, *CREDIT_BUNDLES, *SUBSCRIPTIONS, *SUBSCRIBER_REFILLS]
}


def get_product(product_id: str) -> Optional[Product]:
    return ALL_PRODUCTS.get(product_id)


def bundle_savings_vs_refill(bundle: Product, refill: Product) -> int:
    """Return percentage savings subscribers get on refills vs bundles."""
    if bundle.per_exam_cents == 0:
        return 0
    return round((1 - refill.per_exam_cents / bundle.per_exam_cents) * 100)


def subscriber_refill_savings_percent(refill: Product) -> Optional[int]:
    """Savings vs the closest matching bundle size."""
    match = next((b for b in CREDIT_BUNDLES if b.credits == refill.credits), None)
    if not match:
        return None
    return bundle_savings_vs_refill(match, refill)


def pricing_catalog() -> dict:
    """Full pricing catalog for API / frontend."""
    return {
        "max_exams": MAX_EXAMS,
        "free_per_identity": FREE_EXAMS_PER_IDENTITY,
        "free": _product_dict(FREE_PRODUCT),
        "bundles": [_product_dict(p) for p in CREDIT_BUNDLES],
        "subscriptions": [_product_dict(p) for p in SUBSCRIPTIONS],
        "subscriber_refills": [_product_dict(p) for p in SUBSCRIBER_REFILLS],
        "conversion_notes": {
            "primary_cta": "subscription",
            "free_hook": "1 free exam per email or API key",
            "subscriber_benefit": "Refills are ~40% cheaper than credit bundles",
            "unlimited_cap": f"Unlimited plan includes up to {MAX_EXAMS} exams/month",
            "bundle_ladder": "10 / 15 / 20 credit packs for no-commitment testing",
            "refill_impulse": "5-credit quick refill available when you run out mid-cycle",
        },
    }


def _product_dict(p: Product) -> dict:
    savings = (
        subscriber_refill_savings_percent(p)
        if p.product_type == ProductType.SUBSCRIBER_REFILL
        else None
    )
    return {
        "id": p.id,
        "name": p.name,
        "type": p.product_type.value,
        "credits": p.credits,
        "price_cents": p.price_cents,
        "price_display": p.price_display,
        "per_exam_display": p.per_exam_display,
        "per_exam_cents": p.per_exam_cents,
        "interval": p.interval,
        "badge": p.badge,
        "description": p.description,
        "highlight": p.highlight,
        "subscriber_savings_percent": savings,
    }
