# Portable remote datasets

Remote text training accepts Hugging Face sources, uploaded files, local files,
and pinned non-private internal datasets (`swl://name@revision`). Primary,
additional and explicit evaluation sources all use the same staging and privacy
checks. Planning
checks that each source can be staged; execution verifies availability and
finishes staging before provisioning compute. Local training is unchanged.

Uploaded, local and internal sources are staged only when the control plane
uses an operator-controlled S3/R2 artifact store (`MERLINA_ARTIFACT_STORE=s3`);
with the HF artifact store they are refused, so operator data never lands in a
third-party repo. Remote local files must resolve (symlinks followed) inside a
directory listed in `REMOTE_LOCAL_DATA_ROOTS` (comma or `os.pathsep`
separated); the default, empty, refuses every remote local file. Local paths,
upload IDs and upload filenames matching the private-name policy are refused.
Planning/preflight reports these problems, and staging enforces them again.

Local files and uploads support JSON, JSONL, CSV, and Parquet. Internal sources
must use a concrete revision, Parquet, and the `train` split. Internal shards
are downloaded by the control plane and combined in bounded-size Arrow batches.
Workers therefore receive only access to the job's artifact store, never the
internal dataset bucket's credentials. This deliberately uses control-plane
disk and transfer bandwidth; direct scoped object transfer is a future
optimization. Provision control-plane temporary disk for the combined dataset
plus one shard, and worker disk for the downloaded inputs as well as the model.

Internal manifests marked private, or matching the existing private-name
policy, cannot be exported. This restriction applies even when the destination
artifact store is configured: changing storage does not authorize promotion of
private datasets. Publish an approved non-private revision through the existing
dataset promotion process first. Shard references must remain inside the
requested dataset revision; arbitrary object keys in a manifest are rejected.

Each staged source is an artifact named `input-<index>-<manifest-sha256>`.
The reference pins provenance and content metadata together. It contains
`data.<format>` and a version-1 `input.json` manifest recording format, byte
count, SHA-256, and minimal provenance (source type and pinned internal URI).
The portable training configuration uses `artifact://<name>` references.
Original upload names, upload IDs, local filenames and paths, source bucket
addresses, and credentials are omitted from these manifests and references,
including the artifact provenance metadata recorded with each snapshot.
Source-specific column mappings are preserved. HF sources continue loading
through the existing HF workflow.

Workers reconstruct their artifact store through the usual store description
and separate credentials. Before training they download each input, validate
its manifest, size, checksum and content-addressed reference, then substitute a
private temporary local path in the runtime configuration. Persisted worker
configuration keeps the portable reference. Temporary inputs are cleaned up
when training finishes or fails. Unstaged filesystem paths are refused by the
worker; missing uploads and staging failures prevent compute provisioning.

Input snapshots have the same retention and access policy as their job's
artifacts. Keep them while the run must remain reproducible; deleting the job's
artifact prefix removes the snapshots too. A failed submission may leave an
unused committed snapshot; it is safe to include it in normal job retention
cleanup. Snapshots survive expiration of the API's in-memory upload after they
have been staged. Legacy uploads still need to be available when a queued job
starts; hosted durable upload storage supplies that recovery boundary.

Validation uses fake stores and providers, real Parquet shards, and worker
training stubs. It covers portable upload/local/internal handoffs, integrity
failure, manifest path attacks, unavailable uploads, privacy restrictions,
cleanup, and provisioning order. Live S3/R2 and rented-worker acceptance remains
an operator pilot requirement; these tests do not claim provider certification.
