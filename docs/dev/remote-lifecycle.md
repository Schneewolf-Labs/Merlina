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
Resource timestamps conservatively start at provisioning intent; these are
estimates, not the provider's authoritative invoice.

A provider DELETE response does not prove termination. Pending cleanup is stored
and surfaced in job metrics until a subsequent status check confirms it. Debug
retention is bounded by the job deadline and is disabled when a billing guard is
present. An independently deployed reconciler is required to enforce cleanup
while the API is unavailable; see the following cleanup milestone.
