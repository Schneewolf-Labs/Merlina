# Durable remote artifacts

S3-compatible artifact storage is opt-in. Existing HF handoffs and local stores
remain available. The S3 implementation uses the existing `s3fs` dependency;
AWS S3 and Cloudflare R2 production compatibility still require live validation.
The automated suite covers the protocol using an in-memory transport, not a
paid remote training run.

## Self-hosted configuration

Set these on the control plane:

```dotenv
MERLINA_ARTIFACT_STORE=s3
MERLINA_ARTIFACT_S3_BUCKET=your-bucket
MERLINA_ARTIFACT_S3_PREFIX=merlina/runs
# Omit endpoint for AWS; for R2 use your account's HTTPS endpoint.
MERLINA_ARTIFACT_S3_ENDPOINT=https://ACCOUNT.r2.cloudflarestorage.com
MERLINA_ARTIFACT_S3_REGION=auto
MERLINA_ARTIFACT_ACCESS_KEY=YOUR_SCOPED_KEY
MERLINA_ARTIFACT_SECRET_KEY=YOUR_SCOPED_SECRET
# Optional for temporary credentials:
MERLINA_ARTIFACT_SESSION_TOKEN=YOUR_SESSION_TOKEN
```

Each job is stored below `<prefix>/<job_id>/`. Store descriptions and history
contain only the bucket, prefix, endpoint and region. Dedicated credentials
travel separately in worker environment variables. Ambient AWS credentials
may be used by the control plane but are **never copied** to workers. A worker
can also use its own instance role. Merlina does not create buckets or broaden
permissions. Restrict credentials to the job prefix with GetObject/PutObject;
large multipart transfers also need permission to abort multipart uploads.
These settings alone do not mint job-scoped credentials: hosted deployments
need a scoped capability broker before launching untrusted workers.

S3 handoff does not require an HF token or output repo. A gated HF input model
may still need a token. `push_to_hub` remains a separate optional publishing
step. A remote merge publishes the merged model to exactly one place: the HF
target repo when `push_to_hub` is set, otherwise the artifact store. A local
merge keeps its result on the control plane's disk and does not re-upload it.

## Publication and integrity contract

An artifact has a stable `<name>/manifest.json` pointer. Its version-1 document
contains immutable generation ID, creation time, file paths, byte lengths and
SHA-256 hashes, redacted config, model/dataset revision fields when supplied,
runtime package versions, and checkpoint directory names. An absent revision
is unknown; Merlina does not invent a resolved model revision.

Data is uploaded below `<name>/generations/<uuid>/`. A local snapshot is hashed;
each uploaded object is read back and checked before the manifest is written.
The final single-object manifest PUT publishes the generation. Failed uploads
cannot replace a committed manifest. Retrying creates a fresh generation.
Concurrent writers publish independently; the last completed manifest PUT wins.
Readers pin one manifest and keep using its generation even if a later write
publishes a new one. `exists()` means a valid committed manifest exists; full
content verification happens on download, and permission errors are surfaced.

Downloads reject traversal, duplicate/conflicting paths and invalid schema,
verify every file in a temporary directory, and only then replace the requested
destination. Verification failure preserves previous local files. Publication
assumes atomic single-object writes and read-after-write consistency. No bucket
listing, ETag checksum assumptions or cross-object transaction is required.

Uploads use a local snapshot plus an additional download for verification. Budget
for this disk space and network transfer. SDK transient request retries are
bounded; a failed complete operation can be retried. Training checkpoints inside
the final output directory are included. Remote text runs using S3 also publish
the stable `checkpoint` artifact after each completed Grimoire checkpoint save.
The callback runs after `accelerator.save_state`, before checkpoint rotation;
DDP ranks synchronize around publication so all rank state files are complete.
Only rank zero uploads. The existing `save_steps` policy controls cadence;
`save_steps=0` disables periodic saves. Local/HF training behavior is unchanged.

Publication uses an isolated child process with a 120-second timeout, configurable
with `MERLINA_CHECKPOINT_TIMEOUT_SECONDS` (clamped to 1–900 seconds; the control
plane forwards its value to S3 workers). Training
pauses for that bounded interval; slow/failed publication records a warning and
continues, preserving previous committed generations. The status API mirrors
`checkpoint_publication` into control-plane job metrics, including the last
committed step. A failed/timed-out attempt is never reported as durable; after
one, the worker rereads the committed manifest (bounded) so `last_committed_step`
still matches the store if the manifest PUT landed before the failure. Final
adapter publication proceeds independently and still fails the job if it cannot
commit. Include periodic upload/download verification costs in budgets.

To recover files, call `store.pull_dir("checkpoint", Path("checkpoint-N"))`,
using the committed step N from its manifest metadata. This verifies the full
Accelerate checkpoint, including optimizer and per-rank RNG files. Restoring
files is distinct from restarting training: backend-specific resume and a
compatible model/runtime/data topology must be configured separately. This
change does not automatically launch a replacement paid job.

## Retention, deletion and recovery

Default retention is indefinite: this implementation never deletes committed
or uncommitted generations automatically. Enable bucket versioning to recover a
previous manifest after accidental replacement. Configure the bucket to abort
incomplete multipart uploads after a suitable grace period.

To delete an artifact, first stop its job and all consumers, then delete its
manifest and every object below its name prefix. Do not apply a blanket age
expiration to generation objects: a still-current manifest can reference old
objects. A future garbage collector must retain every referenced generation
(and honor active readers) before removing abandoned generations. Operators can
remove abandoned uploads after comparing them with committed manifests and
waiting for active job/read leases to expire. No automatic retention or recovery
service is implied by this first storage contract.

Before advertising AWS/R2 support, run a real text training smoke test on each:
use a pinned object-store dataset, interrupt and retry an upload, overwrite an
existing artifact, download and verify it, and record cleanup plus observed
compute/storage/transfer charges. No such paid run is part of the unit suite.

## Hosted worker capabilities

The hosted runtime creates a separate account/project/job prefix and issues an
opaque, hashed, expiring broker token. The worker uses that token only at the
hosted capability endpoint. The endpoint checks exact prefix ownership and a
transactional cumulative byte quota before issuing a short-lived GET/PUT URL;
the worker sends data directly to the bucket without forwarding the broker token.
There is no worker list/delete permission or platform AWS key. Access may be
revoked by job ID; already signed URLs remain valid until their short expiry.

The current hosted transport uses single-object PUT, so individual files must
fit the provider's single-PUT limit (5 GiB on S3). Configure checkpoint sharding
accordingly before measuring a workload. Self-hosted s3fs transfers can use
multipart uploads. Hosted quotas conservatively count allocated keys across
checkpoint generations and failed attempts; they do not reclaim bytes during a
run. Choose checkpoint cadence, shards and quota together. A full quota preserves
prior committed artifacts and reports upload failure; it does not delete them.
