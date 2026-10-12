"""Prepaid pilot ledger. Amounts are integer minor currency units throughout.

Only verified processor events credit accounts. Transactions serialize reservations;
provider cleanup must be confirmed before settlement releases any held money.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Any


class BillingError(ValueError):
    pass


class BillingUnavailable(BillingError):
    pass


class BudgetExceeded(BillingError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def config_digest(config: dict) -> str:
    return hashlib.sha256(canonical(config).encode()).hexdigest()


def minor_amount(value: int, *, zero: bool = False) -> int:
    if type(value) is not int or value < (0 if zero else 1) or value > 10**12:
        raise BillingError("Invalid integer money amount")
    return value


@dataclass(frozen=True)
class StageQuote:
    """Trusted server evidence, never accepted from an HTTP request.

    Rate includes GPU, disk and any other billed instance resources. Measurement
    identifies a recorded workload benchmark for this exact config and shape.
    """
    name: str
    provider: str
    gpu_type_id: str
    gpu_count: int
    cloud_type: str
    rate_minor_per_hour: int
    rate_observed_at: int
    measurement_id: str
    measured_seconds: int
    runtime_seconds: int
    boot_seconds: int
    shutdown_seconds: int
    billing_granularity_seconds: int = 60
    container_disk_gb: int = 50
    volume_gb: int = 0
    worker_image: str = ""
    execution_config_hash: str = ""

    def validate(self, now: int) -> None:
        minor_amount(self.rate_minor_per_hour)
        numbers = (self.gpu_count, self.measured_seconds, self.runtime_seconds,
                   self.boot_seconds, self.shutdown_seconds, self.billing_granularity_seconds)
        if any(type(n) is not int or n <= 0 for n in numbers):
            raise BillingError("Quote requires positive runtime and billing bounds")
        if self.runtime_seconds < self.measured_seconds or self.runtime_seconds > 86400:
            raise BillingError("Runtime bound must cover the measurement and be at most one day")
        if not self.measurement_id or not self.provider or not self.gpu_type_id:
            raise BillingUnavailable("Measured workload and live provider pricing required")
        if not 0 <= now - self.rate_observed_at <= 60:
            raise BillingUnavailable("Provider price is stale")

    @property
    def total_seconds(self):
        return self.boot_seconds + self.runtime_seconds + self.shutdown_seconds

    def charge(self, seconds: float) -> int:
        units = math.ceil(max(0, seconds) / self.billing_granularity_seconds)
        return (units * self.billing_granularity_seconds * self.rate_minor_per_hour + 3599) // 3600

    @property
    def reserved_minor(self):
        return self.charge(self.total_seconds)


SCHEMA = (
    "CREATE TABLE IF NOT EXISTS billing_accounts (account_id TEXT PRIMARY KEY, balance INTEGER NOT NULL DEFAULT 0 CHECK(balance>=0), reserved INTEGER NOT NULL DEFAULT 0 CHECK(reserved>=0 AND reserved<=balance))",
    "CREATE TABLE IF NOT EXISTS billing_checkouts (id TEXT PRIMARY KEY, account_id TEXT NOT NULL, amount INTEGER NOT NULL, currency TEXT NOT NULL, session_id TEXT UNIQUE, payment_id TEXT UNIQUE, state TEXT NOT NULL DEFAULT 'pending')",
    "CREATE TABLE IF NOT EXISTS billing_events (id TEXT PRIMARY KEY, kind TEXT NOT NULL, account_id TEXT NOT NULL, amount INTEGER NOT NULL, reference TEXT NOT NULL, created_at INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS billing_quotes (id TEXT PRIMARY KEY, account_id TEXT NOT NULL, project_id TEXT NOT NULL, job_id TEXT NOT NULL, config_hash TEXT NOT NULL, payload TEXT NOT NULL, expires_at INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS billing_runs (job_id TEXT PRIMARY KEY, account_id TEXT NOT NULL, project_id TEXT NOT NULL, quote_id TEXT NOT NULL UNIQUE, amount INTEGER NOT NULL, state TEXT NOT NULL, claimed_at INTEGER, charged INTEGER, provider_cost INTEGER, cancel_requested INTEGER NOT NULL DEFAULT 0, last_error TEXT, failure_code TEXT)",
)


class BillingService:
    def __init__(self, store, settings, *, clock=time.time):
        self.store, self.settings, self.clock = store, settings, clock
        self.currency = getattr(settings, "billing_currency", "usd").lower()
        with store.transaction() as db:
            for sql in SCHEMA:
                db.execute(sql)
            if "failure_code" not in {r[1] for r in db.execute("PRAGMA table_info(billing_runs)")}:
                db.execute("ALTER TABLE billing_runs ADD COLUMN failure_code TEXT")

    def enabled(self):
        s = self.settings
        if self.currency != "usd":
            raise BillingUnavailable("Pilot execution supports USD only")
        if not getattr(s, "paid_execution_enabled", False):
            raise BillingUnavailable("Paid execution is disabled")
        if (not getattr(s, "billing_policy_version", None)
                or getattr(s, "billing_failure_policy", None) != "charge_metered"
                or getattr(s, "billing_overage_policy", None) != "operator_absorbs"):
            raise BillingUnavailable("Operator must configure pricing, failure and overage policy")

    def event(self, account_id, project_id, job_id, kind, payload):
        if hasattr(self.store, "add_event"):
            self.store.add_event(account_id, project_id, kind, payload, job_id=job_id)

    def balance(self, account_id):
        with self.store.transaction() as db:
            row = db.execute("SELECT balance,reserved FROM billing_accounts WHERE account_id=?", (account_id,)).fetchone()
        balance, reserved = (row[0], row[1]) if row else (0, 0)
        return {"balance_minor": balance, "reserved_minor": reserved,
                "available_minor": balance-reserved, "currency": self.currency}

    def create_checkout(self, account_id, amount_minor, transport):
        self.enabled()
        minor_amount(amount_minor)
        if amount_minor > 100_000:
            raise BillingError("Pilot top-up limit is 100000 minor units")
        checkout_id = secrets.token_hex(16)
        with self.store.transaction() as db:
            db.execute("INSERT INTO billing_checkouts(id,account_id,amount,currency) VALUES(?,?,?,?)",
                       (checkout_id, account_id, amount_minor, self.currency))
        # Persist the expected amount BEFORE contacting Stripe. Never credit on redirect.
        session = transport.create_checkout(checkout_id, amount_minor, self.currency)
        with self.store.transaction() as db:
            db.execute("UPDATE billing_checkouts SET session_id=? WHERE id=?", (session["id"], checkout_id))
        return {"checkout_url": session["url"], "checkout_id": checkout_id}

    def confirmed_payment(self, event: dict):
        """Internal method: caller MUST verify Stripe's raw-body signature first."""
        if event.get("type") not in {"checkout.session.completed", "checkout.session.async_payment_succeeded"}:
            return False
        obj = event.get("data", {}).get("object", {})
        if obj.get("payment_status") != "paid":
            return False
        event_id, payment_id = event.get("id"), obj.get("payment_intent")
        if not isinstance(event_id, str) or not isinstance(payment_id, str):
            raise BillingError("Missing processor identifiers")
        with self.store.transaction() as db:
            if db.execute("SELECT 1 FROM billing_events WHERE id=?", (event_id,)).fetchone():
                return False
            row = db.execute("SELECT * FROM billing_checkouts WHERE session_id=?", (obj.get("id"),)).fetchone()
            if row is None:
                raise BillingError("Unknown checkout session")
            if (type(obj.get("amount_total")) is not int or obj["amount_total"] != row["amount"]
                    or obj.get("currency") != row["currency"] or obj.get("mode") != "payment"):
                raise BillingError("Checkout amount/currency/mode mismatch")
            if row["state"] == "paid":
                return False
            if db.execute("SELECT 1 FROM billing_checkouts WHERE payment_id=?", (payment_id,)).fetchone():
                raise BillingError("Payment already applied")
            db.execute("INSERT OR IGNORE INTO billing_accounts(account_id) VALUES(?)", (row["account_id"],))
            db.execute("UPDATE billing_accounts SET balance=balance+? WHERE account_id=?", (row["amount"], row["account_id"]))
            db.execute("UPDATE billing_checkouts SET state='paid',payment_id=? WHERE id=?", (payment_id, row["id"]))
            db.execute("INSERT INTO billing_events VALUES(?,?,?,?,?,?)", (event_id, "payment", row["account_id"], row["amount"], payment_id, int(self.clock())))
        return True

    def quote(self, account_id, project_id, job_id, config, stages: list[StageQuote], *, fixed_minor: int):
        self.enabled()
        minor_amount(fixed_minor, zero=True)
        now = int(self.clock())
        if not stages or len({s.name for s in stages}) != len(stages):
            raise BillingUnavailable("Every execution stage needs unique measured pricing")
        for stage in stages:
            stage.validate(now)
        amount = minor_amount(sum(s.reserved_minor for s in stages) + fixed_minor)
        quote = {"quote_id": secrets.token_hex(16), "amount_minor": amount,
                 "max_cost_minor": amount, "currency": self.currency, "expires_at": now+300,
                 "config_hash": config_digest(config), "stages": [asdict(s) for s in stages],
                 "fixed_minor": fixed_minor, "policy_version": self.settings.billing_policy_version,
                 "failure_policy": "charge_metered", "overage_policy": "operator_absorbs"}
        with self.store.transaction() as db:
            db.execute("INSERT INTO billing_quotes VALUES(?,?,?,?,?,?,?)", (quote["quote_id"], account_id, project_id, job_id, quote["config_hash"], canonical(quote), quote["expires_at"]))
        return quote

    def reserve(self, account_id, project_id, job_id, quote_id, config):
        self.enabled()
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM billing_quotes WHERE id=? AND account_id=? AND project_id=? AND job_id=?", (quote_id, account_id, project_id, job_id)).fetchone()
            if row is None:
                raise BillingError("Quote not found")
            existing = db.execute("SELECT * FROM billing_runs WHERE job_id=?", (job_id,)).fetchone()
            if existing:
                if existing["quote_id"] != quote_id:
                    raise BillingError("Job already has an approved quote")
                return {"job_id": job_id, "state": existing["state"], "reserved_minor": existing["amount"]}
            quote = json.loads(row["payload"])
            if row["expires_at"] <= self.clock() or row["config_hash"] != config_digest(config):
                raise BillingError("Quote expired or configuration changed")
            if quote["policy_version"] != self.settings.billing_policy_version:
                raise BillingError("Billing policy changed; request another quote")
            amount = quote["amount_minor"]
            result = db.execute("UPDATE billing_accounts SET reserved=reserved+? WHERE account_id=? AND balance-reserved>=?", (amount, account_id, amount))
            if result.rowcount != 1:
                raise BillingError("Insufficient prepaid balance")
            db.execute("INSERT INTO billing_runs(job_id,account_id,project_id,quote_id,amount,state) VALUES(?,?,?,?,?,'approved')", (job_id, account_id, project_id, quote_id, amount))
        return {"job_id": job_id, "state": "approved", "reserved_minor": amount}

    def run(self, job_id):
        with self.store.transaction() as db:
            row = db.execute("SELECT r.*,q.payload FROM billing_runs r JOIN billing_quotes q ON q.id=r.quote_id WHERE r.job_id=?", (job_id,)).fetchone()
        if not row:
            raise BillingError("Job has no approved reservation")
        result = dict(row)
        result["quote"] = json.loads(result.pop("payload"))
        return result

    def pending(self):
        with self.store.transaction() as db:
            return [dict(r) for r in db.execute("SELECT * FROM billing_runs WHERE state!='settled'")]

    def claim(self, job_id):
        with self.store.transaction() as db:
            return db.execute("UPDATE billing_runs SET state='dispatching',claimed_at=? WHERE job_id=? AND state='approved'", (int(self.clock()), job_id)).rowcount == 1

    def record_failure(self, job_id, code):
        if code not in {"budget_exhausted"}:
            raise BillingError("Unknown trusted failure code")
        with self.store.transaction() as db:
            db.execute("UPDATE billing_runs SET failure_code=? WHERE job_id=?", (code, job_id))

    def cancel(self, account_id, project_id, job_id):
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM billing_runs WHERE job_id=? AND account_id=? AND project_id=?", (job_id, account_id, project_id)).fetchone()
            if not row:
                raise BillingError("Approved job not found")
            db.execute("UPDATE billing_runs SET cancel_requested=1 WHERE job_id=?", (job_id,))
        return {"job_id": job_id, "state": "cancellation_requested"}

    def abort_before_launch(self, job_id, reason):
        """Internal proof: called only before handing control to a provider runner."""
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM billing_runs WHERE job_id=?", (job_id,)).fetchone()
            if not row or row["state"] != "dispatching":
                raise BillingError("No unlaunched dispatch to abort")
            db.execute("UPDATE billing_accounts SET reserved=reserved-? WHERE account_id=?", (row["amount"], row["account_id"]))
            db.execute("UPDATE billing_runs SET state='settled',charged=0,provider_cost=0,last_error=? WHERE job_id=?", (reason, job_id))
            db.execute("INSERT INTO billing_events VALUES(?,?,?,?,?,?)", ("settle:"+job_id, "no_launch", row["account_id"], 0, job_id, int(self.clock())))

    def settle(self, job_id, provider_cost_minor, *, cleanup_confirmed: bool):
        minor_amount(provider_cost_minor, zero=True)
        if not cleanup_confirmed:
            raise BillingError("Provider cleanup is unconfirmed; reservation retained")
        with self.store.transaction() as db:
            row = db.execute("SELECT * FROM billing_runs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                raise BillingError("Reservation not found")
            if row["state"] == "settled":
                return row["charged"]
            # Explicit pilot policy: SWL absorbs costs above approved reservation.
            charge = min(provider_cost_minor, row["amount"])
            db.execute("UPDATE billing_accounts SET balance=balance-?,reserved=reserved-? WHERE account_id=?", (charge, row["amount"], row["account_id"]))
            db.execute("UPDATE billing_runs SET state='settled',charged=?,provider_cost=? WHERE job_id=?", (charge, provider_cost_minor, job_id))
            db.execute("INSERT INTO billing_events VALUES(?,?,?,?,?,?)", ("settle:"+job_id, "settlement", row["account_id"], -charge, job_id, int(self.clock())))
        self.event(row["account_id"], row["project_id"], job_id, "billing.settled", {"charged_minor": charge, "currency": self.currency})
        return charge


def verify_stripe_event(body: bytes, signature: str, secret: str, *, now=None) -> dict:
    if not secret:
        raise BillingUnavailable("Stripe webhook is not configured")
    fields = {}
    for part in signature.split(","):
        key, sep, value = part.partition("=")
        if sep:
            fields.setdefault(key, []).append(value)
    try:
        timestamp = int(fields["t"][0])
        if len(fields["t"]) != 1 or abs((time.time() if now is None else now)-timestamp) > 300:
            raise ValueError()
        expected = hmac.new(secret.encode(), str(timestamp).encode()+b"."+body, hashlib.sha256).hexdigest()
        if not any(hmac.compare_digest(expected, value) for value in fields.get("v1", [])):
            raise ValueError()
        event = json.loads(body)
        if not isinstance(event, dict):
            raise ValueError()
        return event
    except (KeyError, ValueError, TypeError) as exc:
        raise BillingError("Invalid Stripe webhook signature or payload") from exc


class StripeCheckout:
    def __init__(self, settings):
        self.settings = settings

    def create_checkout(self, checkout_id, amount, currency):
        import requests
        s = self.settings
        if not getattr(s, "stripe_secret_key", None):
            raise BillingUnavailable("Stripe checkout is not configured")
        urls = [getattr(s, key, "") for key in ("checkout_success_url", "checkout_cancel_url")]
        if not all(url and url.startswith("https://") for url in urls):
            raise BillingUnavailable("HTTPS checkout return URLs must be configured")
        response = requests.post("https://api.stripe.com/v1/checkout/sessions",
            auth=(s.stripe_secret_key, ""), timeout=30,
            headers={"Idempotency-Key": checkout_id}, data={
                "mode": "payment", "client_reference_id": checkout_id,
                "success_url": urls[0], "cancel_url": urls[1],
                "line_items[0][price_data][currency]": currency,
                "line_items[0][price_data][unit_amount]": amount,
                "line_items[0][price_data][product_data][name]": "Merlina prepaid compute credits",
                "line_items[0][quantity]": 1,
            })
        response.raise_for_status()
        return response.json()
