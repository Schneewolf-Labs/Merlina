"""Authenticated hosted billing routes; no user-accessible credit/settle API."""
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from .auth import Principal, require_principal
from .billing import BillingError, BillingService, BillingUnavailable, StripeCheckout, verify_stripe_event

router = APIRouter(prefix="/api/hosted")


class CheckoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    amount_minor: int = Field(gt=0, le=100_000, strict=True)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quote_id: str = Field(min_length=1, max_length=100)


def service(request):
    if not getattr(request.app.state, "billing", None):
        request.app.state.billing = BillingService(request.app.state.store, request.app.state.settings)
    return request.app.state.billing


def invoke(fn, *args):
    try:
        return fn(*args)
    except BillingUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except BillingError as exc:
        raise HTTPException(409, str(exc)) from exc


def execution(request):
    result = getattr(request.app.state, "execution", None)
    if result is None:
        raise HTTPException(503, "Managed execution is not configured")
    return result


@router.get("/billing/balance")
def balance(request: Request, principal: Principal = Depends(require_principal)):
    return service(request).balance(principal.account_id)


@router.post("/billing/checkout")
def checkout(body: CheckoutRequest, request: Request, principal: Principal = Depends(require_principal)):
    billing = service(request)
    transport = getattr(request.app.state, "checkout_transport", None) or StripeCheckout(request.app.state.settings)
    return invoke(billing.create_checkout, principal.account_id, body.amount_minor, transport)


@router.post("/billing/webhook")
async def webhook(request: Request):
    # Stripe's raw bytes must be verified BEFORE parsing or touching the ledger.
    body = await request.body()
    if len(body) > 1_000_000:
        raise HTTPException(413, "Webhook too large")
    try:
        event = verify_stripe_event(body, request.headers.get("stripe-signature", ""),
                                    getattr(request.app.state.settings, "stripe_webhook_secret", ""))
        applied = service(request).confirmed_payment(event)
        return {"received": True, "applied": applied}
    except BillingUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except BillingError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/projects/{project_id}/jobs/{job_id}/quote")
def quote(project_id: str, job_id: str, request: Request, principal: Principal = Depends(require_principal)):
    return invoke(execution(request).quote, principal.account_id, project_id, job_id)


@router.post("/projects/{project_id}/jobs/{job_id}/approve")
def approve(project_id: str, job_id: str, body: ApprovalRequest, request: Request,
            principal: Principal = Depends(require_principal)):
    return invoke(execution(request).approve, principal.account_id, project_id, job_id, body.quote_id)


@router.post("/projects/{project_id}/jobs/{job_id}/cancel")
def cancel(project_id: str, job_id: str, request: Request, principal: Principal = Depends(require_principal)):
    return invoke(execution(request).cancel, principal.account_id, project_id, job_id)
