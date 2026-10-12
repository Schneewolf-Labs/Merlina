# Wishcraft hosted pilot runbook

This is an operator-controlled pilot, not an unattended paid service. The hosted
application is separate from the local GPU API. Serve only the hosted application
on its public origin; do not proxy the self-hosted API, MCP server or worker port
through that origin. TLS is mandatory. The hosted studio is served at `/` and
static assets at `/hosted-static`; all customer API requests use `/api/hosted`.

The sign-in throttle is per client IP. `X-Forwarded-For` is honored only when the
direct peer is listed in `MERLINA_HOSTED_TRUSTED_PROXIES` (empty by default).
Behind the local TLS reverse proxy, set `MERLINA_HOSTED_TRUSTED_PROXIES=127.0.0.1`
(or the proxy's address), and make the proxy overwrite rather than append to
client-supplied `X-Forwarded-For`. Without it every client shares the proxy's
throttle bucket, so one attacker can lock everyone out of sign-in.

## Launch gate

Before enabling paid execution, record the operator, environment, image digest,
Git revision and evidence for each gate below. A mock test is not live evidence.

- Validate a small text SFT workload end to end on the selected rented GPU,
  including data transfer, final checksums, checkpoint recovery and termination.
- Verify the selected S3-compatible service. S3 and R2 each need their own live
  integration exercise before being advertised as supported.
- Configure identity, provider, artifact storage and payment secrets server-side;
  scope each key to its service. Never put credentials in browser config or logs.
- Configure current rate lookup and measured workload estimates, worker image
  digests, margins, currency, storage/transfer policy, and conservative shutdown
  headroom. Quote unavailability must block launch. The pilot does not promise a
  model-quality outcome or arbitrary model compatibility.
- Use processor test mode to verify signed webhook replay, concurrent balance
  reservations, stale-quote rejection and settlement retries. Refunds and ledger
  adjustments are not implemented (see below); agree how a participant would be
  made whole before taking payment.
- Run the cleanup service independently from the web API. Alert on its heartbeat,
  overdue instances, termination errors, provider inventory failures, unsettled
  reservations and negative or mismatched balances.
- Complete the restore and rollback exercises below. Nominate an on-call owner
  who can pause launches (below) and terminate resources through the provider.

## Operator switches

Both switches are environment settings read at process start. Change them, then
restart every hosted process (the web app and any standalone
`python -m src.hosted.runtime` dispatcher); there is no runtime toggle.

- `MERLINA_HOSTED_LAUNCHES_PAUSED=true` is the incident switch. Quote and approve
  return 503 ("New launches are paused"), and the dispatcher stops claiming
  approved runs, so no new compute is provisioned. Runs approved before the
  pause keep their reservation and wait in `approved` until launches resume, or
  until the participant cancels them, which releases the reservation without
  provisioning. Cancellation, status, results and downloads stay available, and
  the dispatcher keeps reconciling, cleaning up and settling running or
  uncertain runs. Payment checkout and webhooks are not affected.
- `MERLINA_HOSTED_PAID_EXECUTION_ENABLED=false` removes managed execution
  entirely: quote, approve **and cancel** return 503 and the in-process
  dispatcher does not run, so reconciliation and settlement of existing runs stop
  and their reservations stay held. Use it only when there are no approved,
  dispatching or running jobs, never as an incident response.

There is no maintenance mode that blocks payment processing or other
mutations, and no in-app way to stop sign-in, uploads or drafts. If those must
stop, take the public origin offline at the proxy (which also blocks
cancellation) while keeping the dispatcher running.

First-pilot boundaries: operator-created accounts, USD prepaid execution, approved
text SFT workloads, bounded JSONL uploads, one deployed control-plane writer,
explicit quote approval on every run. VLM, diffusion, automatic experiment loops,
public signup and agent-initiated spending are not pilot features. Publish exact
retention and charging policies to participants before taking payment. Failed
runs can incur infrastructure charges; provider overages above the customer's
approved cap belong to SWL unless a separately agreed policy states otherwise.

## Backup and restore

1. Pause launches (`MERLINA_HOSTED_LAUNCHES_PAUSED=true`) and keep cleanup and
   the dispatcher running until active runs settle. There is no maintenance mode
   for payments or other mutations; for a consistent snapshot, stop the web app
   at the proxy once no runs are active. Record active instances and unsettled
   jobs.
2. Use SQLite's online backup API (not a bare copy of a live `.db` file). Back up
   the hosted identity/ledger database and job/lifecycle databases at the same
   maintenance boundary. Encrypt backups with a key separate from the snapshot.
   Include config revisions and schema versions; never put secrets in the repo.
3. Enable bucket versioning or equivalent protected retention. Retain committed
   manifests and the exact object versions they name. Database backup alone does
   not restore model artifacts or uploaded datasets.
4. Restore to an isolated staging environment with payments and provisioning
   disabled and separate credentials. Run `PRAGMA integrity_check`, compare
   account/project/job counts, reconcile ledger totals and reservations, then
   verify a sampled artifact against its manifest checksum.
5. Record recovery time, recovery point and checks. Only after live provider and
   processor reconciliation should a restored deployment accept mutations.
   Rotate/revoke existing sessions if a database snapshot may have been exposed.

Perform the restore drill before the pilot and regularly thereafter. Choose and
publish RPO/RTO, backup frequency and retention before production; this document
makes no untested recovery guarantee.

## Deployment and rollback

Build a pinned image and record its revision. Back up before schema migration.
Deploy to staging, run auth/isolation and billing tests, then a no-spend smoke
through sign-in, upload, draft, quote and cancellation. Drain mutation requests
before production deployment (pause launches, wait for active runs to settle,
then stop the web app at the proxy). Keep the previous image and its compatible schema
available. Roll back only to a schema-compatible release. Otherwise keep the
origin offline and perform a rehearsed forward fix or restore followed by external
reconciliation. Never roll the ledger back blindly: the provider and payment
processor may have accepted operations after the snapshot.

## Stuck or orphaned jobs

1. Record account/project/job, provider instance ID, stage attempt, heartbeat and
   deadline. Inspect control-plane and cleanup events; do not rely on UI silence.
2. Request cancellation through the participant-scoped cancel route (there is
   no operator cancel route). Check independent cleanup is
   alive and that termination is confirmed by provider inventory.
3. If cleanup cannot reach the provider, escalate and terminate the known
   instance through the provider console. Preserve the failure event and retry
   reconciliation. Do not mark cleanup complete on a timeout or an ambiguous
   response. Do not launch a replacement while ownership is uncertain.
4. Preserve uploaded checkpoints/manifests. Resume only from a committed,
   verified checkpoint on a supported backend, after a new explicit budget
   approval. An interrupted machine is not resumed by local worker reattachment.
5. Finalize usage only after authoritative provider evidence; settle once and
   release unused reservations. Inform the participant of the outcome and any
   artifact or credit recovery steps through the operator support process.

## Billing reconciliation

Compare each confirmed processor event ID with one ledger credit. Compare each
job reservation with its quote/config hash, approved cap, provider resource,
usage window and final debit. Duplicate events and retries must not duplicate
credits or debits. Investigate unresolved reservations and disputed usage; keep
their evidence rather than force-settling to zero.

Refunds, credits, adjustments and chargeback handling are not implemented (see
[prepaid execution](../dev/hosted-billing.md)): there is no adjustment or refund
API, and the ledger only changes through verified payments, reservations and
settlement. Hand-editing the ledger database is out of scope for this runbook
and not a supported procedure. Record any amount owed to a participant and
resolve it outside the application (for example a processor refund recorded
with its evidence) until an audited adjustment operation exists. Processor-side
refunds and chargebacks are not reflected in the ledger; track them separately.

Pause launches (`MERLINA_HOSTED_LAUNCHES_PAUSED=true`) when current pricing,
measured estimates, payment delivery, accounting or cleanup is unhealthy. Do not
turn paid execution off for this: that also disables cancellation and stops
settlement. Record reconciliation discrepancies and ownership until resolved.

## Signals and customer support

Monitor authentication failures, cross-account denials, quote failures, ledger
conflicts, provider timeouts, missing cleanup heartbeats, oldest active stage,
termination latency, storage upload failures, checksum failures and advisor
errors. Alert on sustained failures and any resource that continues billing after
its deadline. Never include passwords, bearer tokens, presigned URLs or raw
customer datasets in telemetry. Support access needs an explicit operational
policy and audit trail; customer authorization is not replaced by an LLM prompt.

## Advisor configuration and boundaries

Set `MERLINA_ADVISOR_ENABLED=true`, `MERLINA_ADVISOR_MODEL`,
`MERLINA_ADVISOR_BASE_URL` (an HTTPS OpenAI-compatible `/v1` endpoint), and
`MERLINA_ADVISOR_API_KEY` in the hosted server's secret environment. Do not set
these values in frontend assets. Choose a provider whose retention and privacy
terms match the pilot agreement. The advisor is explicitly unavailable until
configured; an outage returns an error, never a fabricated LLM answer.

This first pilot sponsors advisor calls from the operator's account; they are
not debited from customer compute credits. Calls are limited to six per account
per minute, 2,000 output tokens and a 45-second provider timeout. Add provider
account spending caps before enabling it. Automated agent billing and adaptive
experiment loops require separate implementation and authorization.

The advisor receives project goals, bounded recent conversation, selected hosted
configuration, verified dataset shape/counts, allowlisted numeric results and
failure codes, and user-reported evaluation observations. It does not receive raw
dataset examples, raw worker logs or infrastructure secrets. Explain this data
flow to pilot participants. Project goals, field names and observations remain
untrusted input. The model receives no tools; service ownership checks and strict
hosted configuration validation apply to every proposal and draft approval.

Conversation requests, immutable proposals, accepted plans and evaluations are
stored in the hosted database and included in its backup. An accepted plan can
create a draft job; only the separate billing quote approval can authorize paid
execution. Repeated plan approval returns the same draft. Recorded observations
are labeled user-reported; this milestone does not execute an independent model
evaluation or claim quality from training loss alone.

A provider interruption leaves a visible failed/pending conversation turn. Retry
with a new request key after investigating sustained errors; duplicate keys do
not repeat a completed call. Pending requests are not automatically replayed
across restarts, to avoid unbounded provider charges. Set conversation retention
and deletion policy before public production use.

## Starting the assembled services

Use the [identity setup](../dev/hosted-identity.md) to create pilot accounts.
Provide configuration through the supervised process environment (the hosted
factory does not load `.env` itself). For paid dispatch install the repository's
runtime dependencies, configure [artifact storage](../dev/artifact-storage.md),
and supply the [billing inputs](../dev/hosted-billing.md), including the measured
catalog and your current total-price service. The price service is an operator
integration, not a bundled hosted endpoint.

Set `MERLINA_HOSTED_PUBLIC_URL` to the HTTPS public origin. The app composes the
S3 store, per-job capability broker, RunPod dispatcher and billing guard at startup
when `MERLINA_HOSTED_PAID_EXECUTION_ENABLED=true`; incomplete configuration fails
startup or quote creation. With the flag false, sign-in, drafts and optional advisor
remain available without initializing paid execution. `MERLINA_HOSTED_MAX_ARTIFACT_BYTES`
sets each worker grant's cumulative allocation ceiling (default 100 GiB).

Run one hosted web process behind TLS:

```sh
uvicorn src.hosted.app:create_hosted_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Its lifespan dispatches approved work and retries result publication every
`MERLINA_HOSTED_DISPATCH_INTERVAL_SECONDS` (default 10). Supervise cleanup as a
separate OS service using the same persistent remote job database and platform
RunPod credential; it must survive web process failure:

```sh
python -m src.remote.cleanup --db /var/lib/merlina-hosted/remote-jobs.sqlite3 --interval 30
```

Match that path to `MERLINA_HOSTED_DATA_DIR`. The cleanup service terminates owned
resources and preserves unresolved evidence; hosted reconciliation settles the
ledger after the web service returns. Monitor the cleanup JSON/alerts rather than
assuming a process heartbeat proves termination. During API maintenance an
operator can run `python -m src.hosted.runtime --once` with the same environment
for one dispatch/reconciliation pass, or supervise that module continuously.
Choose one dispatch host and retain independent cleanup throughout maintenance.
