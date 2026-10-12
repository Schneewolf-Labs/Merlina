# Merlina 3 — Wishcraft ✨

Status: development plan, started 2026-10-11. The source version is `3.0.0`;
this document describes planned work, not a published release or available
hosted service. The release name is **Wishcraft**: a friendly magician helping
turn a goal into a working model. Preserve the cute, magical personality while
making execution dependable and the hosted experience easier to use.

## Product direction

SWL.ai helps people accomplish work with AI agents instead of waiting for a
human expert. Merlina is its model-building product: describe a goal, provide
data, agree on a plan and budget, run an experiment, and evaluate the result
with an agent that understands the project.

The intended hosted experience at **merlina.swl.ai** manages infrastructure
for the user. Users pay Merlina for compute; they do not need their own RunPod
account, provider key, or bucket to get started. RunPod can remain the first
compute backend. S3-compatible storage, including R2, should hold datasets,
checkpoints, logs, and model outputs. Hugging Face remains an optional source
and publishing destination rather than the required artifact handoff.

Self-hosting and storage ownership remain useful. The repository currently
uses the MIT license; this plan does not change it. The proposed business
model is a useful open-source engine plus a paid hosted service, with platform
fees and metered execution. Exact pricing and the hosted/self-hosted feature
boundary still need decisions. Customer-owned buckets can be an advanced
option later rather than a prerequisite for the hosted workflow.

## What exists today

| Area | Existing implementation | Work needed for the hosted experience |
| --- | --- | --- |
| Jobs | SQLite job history, priority queues, detached workers, local worker reattachment (`src/job_manager.py`, `src/job_queue.py`, `src/worker_monitor.py`, `src/worker_reattach.py`) | Durable remote instance/stage ownership, recovery after control-plane failure, and independently enforced cleanup |
| Rented compute | RunPod provider, sizing, plan preview, remote execution and runtime/hourly-price limits (`src/remote/`) | Platform-owned provider credentials, total budget enforcement, authoritative usage accounting and orphan reconciliation |
| Datasets | S3/MinIO/R2 store with pinned `swl://` revisions (`dataset_handlers/internal_store.py`) | Remote planning and workers must accept and stage these sources and uploads; current remote validation still requires Hugging Face sources |
| Artifacts | `ArtifactStore` interface with local and HF Hub implementations (`src/remote/artifacts.py`) | S3/R2 implementation for outputs and checkpoints, verified uploads, retention and portable manifests; HF is currently the default remote store |
| Agent tools | MCP client for training, validation, previews and job inspection (`mcp_server.py`) | Project-aware advisor, scoped tool access, explicit budget approval, evaluations and an action history |
| Hosted product | Existing local web UI and API | Accounts, project ownership, tenant isolation, billing, managed onboarding and operational support |

Existing remote training is API/config driven and supports text LLM modes.
The first hosted pilot should use that path; hosted VLM and diffusion support
need separate validation. Local training support remains valuable throughout.

## Intended user journey

1. Describe the desired model and upload or select data.
2. The agent checks data quality and proposes a model, training approach and
   evaluation. Users can inspect and edit the configuration.
3. Show an estimated cost range, included storage, and a spending limit.
   The user approves the concrete plan before paid resources are provisioned.
4. Merlina runs the job, displays progress and spend, and supports cancellation.
   Closing the browser does not affect execution.
5. Save durable outputs and evaluate them against the agreed goal. The agent
   explains results and proposes a next experiment; additional spending needs
   authorization within an explicit budget policy.
6. Download or export the model and its reproducibility record.

## Implementation sequence and acceptance criteria

### 0. Release identity and planning — this change

- Set the source version and release name, keep version consumers consistent,
  and link this plan from the project docs.
- Keep future capabilities out of the released-feature list. Track progress
  below and document compatibility changes as implementation lands.
- Leave publication and release tagging to the release process after validation.
  This identity change alone introduces no API or config migration.

### 1. Durable storage and portable remote inputs

- Implement S3-compatible `ArtifactStore` push/pull/exists/description support
  and wire reconstruction into both the orchestrator and worker. Keep secrets
  out of store descriptions and job history; deliver scoped access separately.
- Define versioned artifact manifests with checksums, dataset/model revisions,
  training config without secrets, runtime versions, and checkpoint metadata.
  Make partial uploads distinguishable from complete artifacts.
- Extend remote validation and staging to pinned internal datasets and uploads.
  Keep HF import/export as an explicit integration.
- Define retention, deletion and recovery behavior for checkpoints and outputs.

Acceptance: a remote text-training run reads an object-store dataset and saves
an adapter/checkpoint to S3 or R2 without an HF output repository. Downloads
verify integrity; interrupted uploads can retry without exposing a partial
artifact as complete. Existing local and HF workflows still work. Validate
against both S3 and R2 before advertising both as supported.

### 2. Reliable remote job lifecycle

- Persist provider instance IDs, stage attempts, deadlines and artifact
  references before relying on in-memory orchestration state.
- Add idempotent stage transitions and startup reconciliation so retries or
  process restarts cannot launch duplicate paid jobs.
- Add a cleanup service independent of the API process, with provider inventory
  reconciliation, termination retries and alerts for resources still billing.
- Define checkpoint/resume support per training backend. Local worker
  reattachment is not recovery from loss of a rented machine.

Acceptance: exercise API restart, worker loss, provider timeout, cancellation
and upload failure. Each run reaches a truthful state, committed artifacts
remain accessible, and provisioned resources are reconciled and terminated.
Keep cleanup faults visible until the provider confirms termination.

### 3. Hosted accounts and prepaid execution

- Scope projects, jobs, datasets, artifacts, events and agent tools to an
  authenticated account. Cover API, WebSocket and object-store access.
- Separate the hosted app/API from GPU workers. Store provider credentials
  server-side and give workers only the access required by their job.
- Design a prepaid ledger: confirmed payment credits, atomic reservations
  before provisioning, metered settlement and release of unused reservations.
  Payment events, retries and settlement must be idempotent.
- Quote from current provider rates and measured workloads. Define charges
  for boot time, training, merging, storage, transfer, agent usage and failures.
  Stale/static catalog estimates are not sufficient to authorize paid work.
- Enforce total approved spend across stages and concurrent jobs. Reserve
  headroom for billing granularity and shutdown delay; document who absorbs
  overages and how failed runs are credited. The provider still bills SWL when
  infrastructure runs under SWL's account, even with prepaid customer credits.
- Add backups with restore checks, deployment rollback, service metrics and
  operating procedures for stuck jobs and billing discrepancies.

Acceptance: one account cannot access another's resources. Concurrent starts
cannot spend the same balance twice. Insufficient funds prevent provisioning;
duplicate payment events do not duplicate credits. Restarted jobs settle once,
and simulated budget exhaustion triggers cleanup. Run an operator-controlled
pilot before allowing unattended paid use.

### 4. Agent-guided model creation

- Build on the REST/MCP tools for dataset inspection, plan generation,
  validation, execution, evaluation and diagnosis.
- Persist project goals, approved plans, results and an auditable action
  history. Ground recommendations in actual job data and measured results.
- Treat dataset content and logs as untrusted input. Tool permissions and
  spending authorization are enforced by the service, not by model prompts.
- Start with an existing LLM powering the advisor. Training an SWL-specific
  advisor is a later experiment, not a dependency for the initial product.
- Keep configuration, logs, cost and cancellation accessible alongside chat.

Acceptance: a user completes the goal-to-model journey without provider setup.
The agent explains a failed run using its evidence and proposes a reviewable
next step. It cannot bypass the account boundary or approved spending limit.

## Initial backlog

All items below are planned, not implemented by this version change.

| ID | Deliverable | Depends on |
| --- | --- | --- |
| M3-01 | Artifact manifest and S3/R2 store implementation | Storage contract decision |
| M3-02 | Remote object-store datasets and upload staging | M3-01 |
| M3-03 | Durable remote stage/instance records and restart reconciliation | M3-01 |
| M3-04 | Independent resource cleanup and failure/recovery exercises | M3-03 |
| M3-05 | Account/project ownership and scoped worker access | Hosted identity/data design |
| M3-06 | Quotes, prepaid ledger, reservations and usage settlement | M3-03, M3-05, billing policy |
| M3-07 | Managed launch/progress/results UI and operating procedures | M3-02, M3-04, M3-05, M3-06 |
| M3-08 | Advisor with approved actions and evaluation workflow | M3-05, M3-06, M3-07 |

Start implementation with **M3-01**: it removes the required HF artifact
dependency and establishes the durable output contract used by later work.

## Decisions to resolve before the hosted pilot

- First supported model family, workload and evaluation criteria.
- Initial provider/region, supported GPU shapes and pinned worker images.
- Hosted bucket layout, retention defaults, export/deletion policy and pricing.
- Account identity, production database and worker authentication design.
- Payment processor, margin, refunds, failure charges and overage responsibility.
- Advisor model/provider, data handling, and which actions need explicit approval.
- Hosted/self-hosted feature boundary and migration support for existing installs.

The hosted pilot is ready only when the storage, recovery, isolation and
billing acceptance criteria pass together in a real rented-compute run.
Record its observed costs, cleanup outcome and artifact integrity before
promoting 3.0.0 as ready for paid hosted use.
