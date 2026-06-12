from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from credits import CreditError, consume_exam_credit, get_account, purchase_product, register_user
from pricing import pricing_catalog

router = APIRouter(prefix="/api", tags=["billing"])


class IdentityRequest(BaseModel):
    email: Optional[str] = None
    api_key: Optional[str] = None


class RegisterRequest(IdentityRequest):
    api_key: Optional[str] = Field(None, description="Optional API key for programmatic access")


class PurchaseRequest(IdentityRequest):
    product_id: str


@router.get("/pricing")
def get_pricing():
    return pricing_catalog()


@router.post("/register")
def api_register(body: RegisterRequest):
    if not body.email and not body.api_key:
        raise HTTPException(400, "Provide email or api_key")
    return register_user(email=body.email, api_key=body.api_key)


@router.get("/account")
def api_account(email: Optional[str] = None, api_key: Optional[str] = None):
    try:
        return get_account(email=email, api_key=api_key)
    except CreditError as e:
        raise HTTPException(400, e.message) from e


@router.post("/purchase")
def api_purchase(body: PurchaseRequest):
    if not body.email and not body.api_key:
        raise HTTPException(400, "Provide email or api_key")
    try:
        return purchase_product(body.product_id, email=body.email, api_key=body.api_key)
    except CreditError as e:
        status = 403 if e.code == "subscription_required" else 400
        raise HTTPException(status, e.message) from e


@router.post("/exam/consume")
def api_consume_exam(body: IdentityRequest):
    if not body.email and not body.api_key:
        raise HTTPException(400, "Provide email or api_key")
    try:
        return consume_exam_credit(email=body.email, api_key=body.api_key)
    except CreditError as e:
        raise HTTPException(402, detail={"code": e.code, "message": e.message}) from e
