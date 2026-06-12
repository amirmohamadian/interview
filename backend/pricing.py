"""
Pricing configuration optimized for conversion.

Funnel: Free trial → Subscribe (primary) or Credit bundles (try without commitment)
        → Subscriber refills at discounted rates when credits run out.

All paid options cap at 20 exams (max value tier).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


MAX_EXAMS = 20
FREE_EXAMS_PER_IDENTITY = 1


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
CREDIT_BUNDLES: list[Product] = [
    Product(
        id="bundle_3",
        name="Try Pack",
        product_type=ProductType.BUNDLE,
        credits=3,
        price_cents=1200,
        description="Perfect to see if it fits your prep style",
    ),
    Product(
        id="bundle_5",
        name="Practice Pack",
        product_type=ProductType.BUNDLE,
        credits=5,
        price_cents=1800,
        description="A full week of focused interview practice",
    ),
    Product(
        id="bundle_10",
        name="Prepare Pack",
        product_type=ProductType.BUNDLE,
        credits=10,
        price_cents=3200,
        badge="Popular",
        description="Best value for one-time prep before your interview",
        highlight=True,
    ),
    Product(
        id="bundle_20",
        name="Max Pack",
        product_type=ProductType.BUNDLE,
        credits=MAX_EXAMS,
        price_cents=5600,
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
        price_cents=1000,
        interval="month",
        description="5 mock exams per month — great for steady practice",
    ),
    Product(
        id="sub_pro",
        name="Pro",
        product_type=ProductType.SUBSCRIPTION,
        credits=10,
        price_cents=1700,
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
        price_cents=2900,
        interval="month",
        badge="Power User",
        description="Up to 20 exams/month — unlimited feel, fair-use cap",
    ),
]

# ── Subscriber refills (cheaper when credits run out mid-cycle) ─────────────
SUBSCRIBER_REFILLS: list[Product] = [
    Product(
        id="refill_3",
        name="Quick Refill",
        product_type=ProductType.SUBSCRIBER_REFILL,
        credits=3,
        price_cents=700,
        description="~42% cheaper than non-subscriber packs",
    ),
    Product(
        id="refill_5",
        name="Standard Refill",
        product_type=ProductType.SUBSCRIBER_REFILL,
        credits=5,
        price_cents=1000,
        description="Top up without waiting for next billing cycle",
    ),
    Product(
        id="refill_10",
        name="Power Refill",
        product_type=ProductType.SUBSCRIBER_REFILL,
        credits=10,
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
        price_cents=3200,
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
        },
    }


def _product_dict(p: Product) -> dict:
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
    }
