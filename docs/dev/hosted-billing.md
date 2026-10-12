# Prepaid execution pilot

Paid execution is **off by default**. This implementation is an operator-controlled
pilot foundation, not a published pricing/refund policy or an unattended service.
The provider bills SWL. A prepaid reservation limits what the customer is charged;
it does not impose a hard provider-side spending cap or guarantee timely shutdown.

## Ledger and authorization

`BillingService` stores money as integer USD cents in transactional SQLite tables
alongside the hosted account database. `BEGIN IMMEDIATE` serializes reservations.
Payments credit only a known checkout session with matching amount/currency after
raw-body Stripe webhook verification. Payment intent IDs, webhook event IDs,
per-job reservations and settlement IDs are unique. Browser return URLs never
credit a balance. There is no public credit or settlement route.

Quotes expire after five minutes and bind the account, project, job, exact config,
policy version, measurement ID, live provider price, worker image/shape, runtime,
boot allowance, shutdown allowance and billing granularity. Approval atomically
reserves the whole amount across all stages. Changing the profile after approval
blocks launch. Repeat approval of the same quote returns the original reservation.

Billing policy must explicitly configure `billing_failure_policy=charge_metered`
and `billing_overage_policy=operator_absorbs`, with a version string. This pilot
supports that policy only: failed jobs pay metered instance time; SWL absorbs costs
above the approved maximum. Set these only after adopting and disclosing the
policy. Refunds, chargebacks, taxes, invoicing and multi-currency settlement are
not implemented. Do not enable general public payments without those procedures.

## Runtime configuration

`install_runtime(app, artifact_store_factory)` binds the hosted app to RunPod and
its durable lifecycle. Run `app.state.execution.tick()` periodically from a
supervised process; it starts approved jobs in background threads and reconciles
claimed jobs after restart. Also supervise the independent lifecycle cleanup
service. Claim ambiguity never resets automatically: uncertain creates retain
reservations until provider inventory and cleanup are resolved. A dispatch claim
without a lifecycle record requires operator investigation; it is never replayed.

Required operator inputs:

- `RUNPOD_API_KEY`: platform credential, never included in a user job or worker.
- `MERLINA_MEASUREMENT_CATALOG`: local JSON object keyed by the SHA-256 canonical
  hosted draft config. Each value includes `measurement_id`, `measured_seconds`,
  `runtime_seconds`, `boot_seconds`, `shutdown_seconds`,
  `billing_granularity_seconds`, `fixed_minor`, and a complete `training_config`
  operator profile. The profile pins `remote.worker_image` by `@sha256:`,
  `gpu_type_id`, `gpu_count` and `container_disk_gb`. Record benchmark evidence for
  this precise workload and shape before adding a catalog entry. For hosted HF
  sources, mutable upstream revisions still need an operator-pinned input policy
  before real paid pilot validation.
- `MERLINA_LIVE_PRICING_URL`: operator-controlled HTTPS service returning current
  total provider prices. Required because the existing RunPod GPU catalog does
  not price disks/other resources completely. JSON format: `{"currency":"usd",
  "observed_at":<epoch>,"rates":[{"provider":"runpod","gpu_type_id":"...",
  "gpu_count":1,"cloud_type":"secure","container_disk_gb":100,"volume_gb":0,
  "rate_minor_per_hour":120}]}`. Evidence older than 60 seconds is rejected.
  Rates are cross-checked against fresh RunPod GPU offers, with no static fallback.
- `artifact_store_factory(job)`: must supply a per-job scoped worker capability;
  a plain store containing platform AWS keys is rejected.

Hosted identity supplies safe dataset/config fields, overriding operator fields
that could enable local execution, remote code, publishing, retained instances or
arbitrary output paths. The first runtime binding supports text adapter-only jobs;
local merging is disabled. The ledger supports multiple priced remote stages but
hosted merge is not advertised until a measured and fully priced binding exists.

`fixed_minor` is the disclosed storage/transfer charge for a launched run. Instance
usage includes boot through confirmed termination, rounded to the configured
provider granularity. Agent use is excluded from this runner. Unknown provider
rates or unconfirmed cleanup retain the reservation for investigation; they do not
produce an invented zero-cost settlement. Runtime and estimated spend exhaustion
request cleanup through lifecycle hooks with reserved shutdown headroom. The
spend check rounds elapsed time plus shutdown allowance once, as the reservation
does, and stops only when that projection exceeds the reservation.

A dispatch claim whose worker never took lifecycle ownership (for example, a
crash right after the claim) is released at zero after 10 minutes: the
dispatcher takes the insert-only lifecycle claim itself, so a late worker cannot
provision, then marks the job failed. Provisioning always requires a lifecycle
intent first, so no provider resource can exist for such a claim.

## Routes

All authenticated routes use `/api/hosted`:

- `GET /billing/balance`
- `POST /billing/checkout` with integer `amount_minor`
- `POST /projects/{project_id}/jobs/{job_id}/quote`
- `POST /projects/{project_id}/jobs/{job_id}/approve` with `quote_id`
- `POST /projects/{project_id}/jobs/{job_id}/cancel`

`POST /billing/webhook` uses Stripe signature authentication instead of an account
session. Configure server-owned HTTPS checkout success/cancel URLs and both Stripe
keys. Rotate the webhook secret carefully and retain processor event records.

Test mode and mocked provider exercises cover concurrency, insufficient balance,
duplicate events, tampering, stale quotes, restart settlement and uncertain
provisioning. No live paid compute or Stripe charge is made by the test suite.
Before opening a pilot, exercise end-to-end pricing, measured workload matching,
worker scoped access, cleanup and processor reconciliation with real services.

Operator environment keys use `MERLINA_HOSTED_` for settings fields, including
`MERLINA_HOSTED_PAID_EXECUTION_ENABLED`, `MERLINA_HOSTED_STRIPE_SECRET_KEY`,
`MERLINA_HOSTED_STRIPE_WEBHOOK_SECRET`, `MERLINA_HOSTED_CHECKOUT_SUCCESS_URL`,
`MERLINA_HOSTED_CHECKOUT_CANCEL_URL`, `MERLINA_HOSTED_BILLING_POLICY_VERSION`,
`MERLINA_HOSTED_BILLING_FAILURE_POLICY` and `MERLINA_HOSTED_BILLING_OVERAGE_POLICY`.

After recording benchmark evidence, use `python -m src.hosted.measurement --help`
to create/update a catalog entry from an existing hosted job. The command reads
its stored config, computes the exact digest, and writes the minimal safe remote
profile atomically. Every allowance, GPU/disk shape, fixed charge, image digest,
and measurement reference is explicitly required. A new uploaded dataset ID or
changed hyperparameter requires another entry; it never silently inherits a
benchmark. The worker defaults supply other fields, so operators do not need to
hand-copy the complete local training configuration. The job-specific output path
is added later and bound into the immutable quote's execution hash.

Known preflight failure before the provider runner is called records a durable
`no_launch` ledger event and releases the reservation atomically. A lost response
after a provider launch intent remains unresolved until resource reconciliation.
User-visible failures contain a generic message and job ID; internal provider
errors and paths stay in operator logs. A completed adapter is registered as a
deterministic ZIP with its checksum; download cache loss is recoverable from the
verified per-job object-store artifact. Account ownership is checked before the
download callback runs.

Result synchronization has a separate durable queue from billing settlement.
A completed job whose archive registration fails remains completed with
`artifact_status=pending`; dispatcher ticks retry after restart, including after
the reservation has settled. Recovery can rebuild the archive from the verified
remote adapter if local output was lost. Customer results include finite recorded
loss/evaluation loss, step counts, and observed instance runtime. Failure codes
come from trusted service evidence (`budget_exhausted`) or the generic
`training_failed`; raw error strings never become advisor evidence.
