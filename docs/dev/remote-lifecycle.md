# Durable remote execution

Remote runs now claim their job ID exactly once in SQLite before work begins.
A stage records a unique deployment-scoped provider name before provisioning;
provider timeouts never authorize an automatic retry. Instance IDs, observed
hourly rates, attempt outcomes, artifact references, cleanup confirmation and
timestamps survive a controller restart. Worker/provider credentials are not
stored in these records.

The runtime limit applies to the entire job, including boot and multiple stages.
The controller renews a 120-second lease. A controller that loses its lease
cannot provision another stage. A restarted dispatcher leaves existing execution
records alone: this is deliberately **fail and clean up**, not remote training
resume. Backend checkpoint resume remains unsupported. Durable artifacts remain
available independently of a failed execution.

`run_training_remote(..., execution_guard=...)` accepts a hosted billing guard:
`before_provision(job_id, stage_name, spec)`, `usage(job_id, resources)`, and
`settle(job_id, resources)`. The first two may raise to prevent/stop paid work.
Settlement must retain funds while cleanup is unconfirmed. Ambiguous provisioning
with no discovered instance skips settlement pending operator reconciliation.
A provider raises `ProvisionRejected` only when its response proves nothing was
created (RunPod: any 4xx, or an error body without a pod id such as "no
instances available"); that attempt is recorded `failed`, not ambiguous.
Timeouts, gateway errors and unparseable responses stay `provisioning_unknown`.
Resource timestamps conservatively start at provisioning intent; these are
estimates, not the provider's authoritative invoice.

A provider DELETE response does not prove termination. Pending cleanup is stored
and surfaced in job metrics until a subsequent status check confirms it. Debug
retention is bounded by the job deadline and is disabled when a billing guard is
present. The API process reconciles periodically on its own; an independently
deployed reconciler additionally enforces cleanup while the API is down.

## Cleanup service

When RunPod is configured, the API process runs the reconciler on a background
thread: a pass at startup, then every 30 s while any run, unconfirmed resource or
ambiguous attempt is outstanding (idle installs make no provider calls). A run
whose lease was still valid when the API restarted is therefore cleaned up once
the lease lapses. Runs this process drives keep a live lease and are left to
their controller. A single self-hosted API is safe on its own.

To keep cleanup running while the API is down, also run it as a separate
process/service with the same persistent database volume
and server-side `RUNPOD_API_KEY` (never a GPU worker's credentials):

```sh
python -m src.remote.cleanup --db ./data/jobs.db --interval 30
python -m src.remote.cleanup --db ./data/jobs.db --once
```

A service supervisor must restart this process and alert on missed runs. The
one-shot form exits nonzero for unresolved alerts. JSON output includes durable
alerts and resource state; integrate it into your monitoring. These commands
perform real provider termination, so use the correct deployment database.

Each pass expires stale controller leases and total deadlines, discovers pods
whose names exactly match persisted provisioning intents, retries pending
termination, and confirms deletion through provider status. It cleans cancelled
or deleted jobs and duplicated instances. Active leases are preserved; the API and a
separate service may both run passes concurrently.
Other deployments and arbitrary `merlina-*` pods are not touched. A name from
this deployment without a matching intent triggers manual-review alerts.

Provider timeouts, failed deletes and unknown status remain visible and retry
on subsequent passes. Unknown provisioning is not resolved by one empty
listing, because inventory is eventually consistent. It resolves to `failed`
(nothing to terminate) only after its exact name is absent from 3 consecutive
successful inventory scans spanning at least 10 minutes (failed scans do not
count; a sighting resets the count). A discovered and terminated instance also
resolves it. Definite provider refusals (`ProvisionRejected`) never become
ambiguous. As a last resort an operator can settle it by hand, recorded with
their name in `remote_manual_resolutions`:

```sh
python -m src.remote.cleanup --db ./data/jobs.db --resolve <job_id|label> \
    --as failed|terminated --by <operator> --note "checked RunPod console"
```

`failed` means nothing was created; `terminated` means you confirmed the
attempt's provider resources are gone (it marks them terminated). Live runs are
refused.

Debug-retained pods are deleted at the original total job deadline. A stopped
pod is not treated as terminated: attached storage may still bill. Observed
resource durations and rates are estimates; authoritative invoice reconciliation
is still required. Network/provider outages can delay shutdown beyond a budget
limit. The hosted billing policy must reserve shutdown headroom and absorb any
remaining overage rather than claiming an absolute provider-spend guarantee.

SQLite must be shared by the API and cleanup process on a durable local volume;
this is a single-host control plane, not a distributed database architecture.
Back up the complete database (including namespace and lifecycle tables). After
a restore, audit provider inventory before enabling new paid executions.

## Cancellation and startup

Normal cancellation persists a single checkpoint grace deadline shared by the
controller and cleanup service (`REMOTE_STOP_GRACE_MINUTES`, default 15). The
window cannot extend when either process restarts and cannot exceed the job's
total deadline. Cleanup forwards an authenticated stop request when provider
metadata contains the scoped worker token; it never stores that token in the
lifecycle database. If the token is unavailable, the live controller still
forwards stop, and independent cleanup enforces the persisted deadline. Finished
or failed stages, pending emergency cleanup and explicit emergency cancellation
can terminate immediately. A late cleanup result cannot regress an already
confirmed termination.

Local worker reattachment skips remote jobs; the API's periodic reconciler (see
above) handles them, and live remote leases remain owned by their controller.
