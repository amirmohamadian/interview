from typing import Optional

from database import (
    activate_subscription,
    add_credits,
    claim_free_trial,
    deduct_credit,
    get_or_create_user,
    get_user_by_id,
    has_claimed_free_trial,
    record_purchase,
)
from pricing import (
    FREE_EXAMS_PER_IDENTITY,
    ProductType,
    get_product,
)


class CreditError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def register_user(email: Optional[str] = None, api_key: Optional[str] = None) -> dict:
    user = get_or_create_user(email=email, api_key=api_key)
    free_granted = []

    for identity_type, value in [("email", email), ("api_key", api_key)]:
        if not value:
            continue
        if not has_claimed_free_trial(identity_type, value):
            claim_free_trial(identity_type, value)
            add_credits(
                user["id"],
                FREE_EXAMS_PER_IDENTITY,
                reason=f"free_trial_{identity_type}",
                reference_id=value,
            )
            free_granted.append({"type": identity_type, "credits": FREE_EXAMS_PER_IDENTITY})

    user = get_user_by_id(user["id"])
    return {
        "user": _public_user(user),
        "free_granted": free_granted,
        "message": _free_message(free_granted),
    }


def _free_message(free_granted: list) -> str:
    if not free_granted:
        return "Welcome back! Your free trial was already claimed for this email/API key."
    total = sum(g["credits"] for g in free_granted)
    return f"Welcome! {total} free exam credit{'s' if total != 1 else ''} added to your account."


def get_account(email: Optional[str] = None, api_key: Optional[str] = None) -> dict:
    if not email and not api_key:
        raise CreditError("missing_identity", "Provide email or api_key")
    user = get_or_create_user(email=email, api_key=api_key)
    return _account_summary(user)


def purchase_product(
    product_id: str,
    email: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    product = get_product(product_id)
    if not product:
        raise CreditError("invalid_product", f"Unknown product: {product_id}")
    if product.product_type == ProductType.FREE:
        raise CreditError("invalid_product", "Free tier is claimed automatically on signup")

    user = get_or_create_user(email=email, api_key=api_key)

    if product.product_type == ProductType.SUBSCRIBER_REFILL:
        if not user.get("subscription"):
            raise CreditError(
                "subscription_required",
                "Subscriber refills are only available to active subscribers. Subscribe first to unlock cheaper top-ups.",
            )

    purchase_id = record_purchase(
        user["id"],
        product.id,
        product.price_cents,
        product.credits,
    )

    if product.product_type == ProductType.SUBSCRIPTION:
        activate_subscription(user["id"], product.id, product.credits)
        add_credits(
            user["id"],
            product.credits,
            reason="subscription_start",
            reference_id=str(purchase_id),
        )
    else:
        add_credits(
            user["id"],
            product.credits,
            reason=product.product_type.value,
            reference_id=str(purchase_id),
        )

    user = get_user_by_id(user["id"])
    return {
        "purchase": {
            "id": purchase_id,
            "product_id": product.id,
            "product_name": product.name,
            "amount_cents": product.price_cents,
            "credits_granted": product.credits,
        },
        "account": _account_summary(user),
    }


def consume_exam_credit(
    email: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    user = get_or_create_user(email=email, api_key=api_key)
    try:
        remaining = deduct_credit(user["id"], reason="exam")
    except ValueError:
        suggestions = _upsell_suggestions(user)
        raise CreditError(
            "insufficient_credits",
            "No exam credits remaining. Subscribe or buy a refill to continue.",
        ) from None

    user = get_user_by_id(user["id"])
    return {
        "consumed": 1,
        "remaining": remaining,
        "account": _account_summary(user),
        "upsell": suggestions if remaining <= 2 else None,
    }


def _upsell_suggestions(user: dict) -> dict:
    is_subscriber = bool(user.get("subscription"))
    return {
        "is_subscriber": is_subscriber,
        "recommended": "subscriber_refills" if is_subscriber else "subscriptions",
        "message": (
            "As a subscriber, you get refills ~40% cheaper than credit bundles."
            if is_subscriber
            else "Subscribe to unlock cheaper refills when you run out of credits."
        ),
    }


def _account_summary(user: dict) -> dict:
    sub = user.get("subscription")
    return {
        "user": _public_user(user),
        "can_take_exam": user["balance"] > 0,
        "low_credit_warning": 0 < user["balance"] <= 2,
        "upsell": _upsell_suggestions(user) if user["balance"] <= 2 else None,
    }


def _public_user(user: dict) -> dict:
    sub = user.get("subscription")
    return {
        "id": user["id"],
        "email": user["email"],
        "has_api_key": bool(user.get("api_key")),
        "balance": user["balance"],
        "subscription": (
            {
                "plan_id": sub["plan_id"],
                "status": sub["status"],
                "credits_per_month": sub["credits_per_month"],
                "period_end": sub["current_period_end"],
            }
            if sub
            else None
        ),
    }
