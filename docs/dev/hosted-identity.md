# Hosted identity and isolation (M3-05)

This is an opt-in **operator-controlled pilot**, separate from the local Merlina
API. It creates project and training drafts; identity alone never provisions
compute. Later stack layers add prepaid execution and the advisor. Do not
advertise unattended paid readiness until the integrated pilot gates in
[the roadmap](merlina-3.md) pass.

## Start on a CPU host

From a repository checkout, install the minimal control-plane dependencies:

```sh
python -m pip install 'fastapi>=0.104.0' 'pydantic>=2.0.0' 'uvicorn>=0.24.0'
export MERLINA_HOSTED_DATA_DIR=/var/lib/merlina-hosted
export MERLINA_HOSTED_TRUSTED_PROXIES=127.0.0.1  # the local TLS reverse proxy
python -m src.hosted.cli create-account pilot@example.com
uvicorn src.hosted.app:create_hosted_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Run behind a TLS reverse proxy with a request size limit, request timeout,
per-IP login throttling, and access logs that omit authorization headers and
bodies. Account creation prompts twice for the password; never pass passwords
as command arguments. A dedicated OS user must own the private data directory.
The hosted app does not import the local `merlina` app, torch, transformers,
GPU training engines, or their unauthenticated API/WebSocket routes. Never
mount the local app under the same public origin or proxy its routes.

The factory reads `HostedSettings.from_env()`. Settings use the
`MERLINA_HOSTED_` prefix: `DATA_DIR`, `DATABASE_PATH`, `SESSION_TTL_SECONDS`,
`MAX_DATASET_BYTES`, `CORS_ORIGINS` (comma-separated explicit origins),
`TRUSTED_PROXIES` (comma-separated peer addresses whose `X-Forwarded-For` is
honoured; empty by default, so the TCP peer address is used),
`PAID_EXECUTION_ENABLED` (strict `true` or `false`, default false),
`STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `CHECKOUT_SUCCESS_URL`,
`CHECKOUT_CANCEL_URL`, `BILLING_CURRENCY` (USD only), `BILLING_POLICY_VERSION`,
`BILLING_FAILURE_POLICY`, and `BILLING_OVERAGE_POLICY`. The database defaults
to `$MERLINA_HOSTED_DATA_DIR/hosted.sqlite3`. Billing/runtime layers document
required policy values and additional gates; setting the flag alone is not
sufficient to provision resources.

## Authentication and API

An operator bootstraps accounts. Passwords use scrypt (N=32768, r=8, p=3),
random 128-bit salts, and constant-time comparison. Login is
`POST /api/hosted/session` with `{email,password}`. It returns a random opaque
bearer token, expiry, and account. Store only the SHA-256 token hash in SQLite;
raw tokens are returned once, never stored in jobs or logs. Sessions default
to 24 hours and survive restarts. `DELETE /api/hosted/session` revokes the current
token. `GET /api/hosted/session` returns the authenticated account.

All remaining API calls require `Authorization: Bearer TOKEN`. Cookies and
URL query tokens are not accepted. Browser clients use session storage. The
server atomically reserves a failure before password verification and refunds
it on success, so only failed logins count. Limits are per client IP (100
failures per 15 minutes) and per client IP plus email (10 per 15 minutes,
cleared on success). A known email is never locked from everywhere: a correct
password from an address that is not itself throttled still signs in. The
client IP is the TCP peer unless that peer is listed in `TRUSTED_PROXIES`, in
which case the rightmost untrusted `X-Forwarded-For` hop is used. Behind the
recommended local reverse proxy, set `TRUSTED_PROXIES=127.0.0.1`; otherwise
every client shares the proxy's bucket. Password verification runs outside
database transactions so unauthenticated requests cannot hold the billing
writer lock during the KDF. Hashed throttle buckets expire after 15 minutes
and the table is capped at 10000 rows by evicting the oldest buckets, so
sprays cannot grow it without bound or refuse new clients.
Use a reverse proxy or WAF for distributed brute-force limits. There is no public
signup, password-reset email, MFA or SSO yet. A public rollout requires an
identity provider or those account lifecycle controls.

- `/api/hosted/projects`: list or create `{name,goal}`.
- `/api/hosted/projects/{project_id}`: retrieve, or PATCH any of `{name,goal}` (omitted fields are kept).
- `/api/hosted/projects/{project_id}/datasets`: list or POST `{name,content}`,
  where `content` is JSONL, limited to 10 MiB UTF-8 by default.
- `/api/hosted/projects/{project_id}/jobs`: list or POST `{config}` to save a draft.
- `/api/hosted/projects/{project_id}/jobs/{job_id}`: retrieve a draft/run.
- Project `/events` and job `/events`: durable action/progress history.
- Job `/artifacts`: metadata without bucket keys; `/artifacts/{id}/download`
  calls an operator-installed artifact service only after ownership checks.
  Without that service, download returns 503, never a public object URL.

Every resource query checks its account and project; substituting another
account's ID returns 404. Hosted progress uses authenticated polling, with no
unauthenticated WebSocket. CORS is disabled unless specific HTTPS origins are
configured; wildcards are rejected. Responses are not cacheable and use a
same-origin content-security policy. Dataset content and goals are untrusted
text, never instructions to grant access or execute tools.

## Worker boundary

`HostedJobConfig` is an allowlist, separate from the broad local training
schema. Users cannot provide provider tokens, worker images, local paths,
artifact bucket keys, arbitrary internal datasets, executable templates,
remote-code flags, local merge operations, or unknown fields. Base models and
HF datasets accept repository identifiers only. Upload sources use an opaque
owned dataset ID. Outputs use server-derived account/project/job prefixes.
Only text modes are accepted. An operator must additionally approve model
families and pin worker images before running the pilot.

`build_execution_config` supplies the safe remote configuration;
`resolve_uploaded_datasets` loads only owned files and verifies their saved
checksum. Remote staging snapshots uploads into the job's artifact store.
Provider credentials stay with the control plane. The runtime must supply a
short-lived, job-prefix-only artifact grant to each worker; identity provides
no shared storage or provider key to the caller. The execution/billing layer
must enforce approval and reserves before queueing this configuration.

## Database, backup and restore

SQLite schema version 1 is installed transactionally, with foreign keys,
WAL journaling, 30-second busy timeout, and `BEGIN IMMEDIATE` write boundaries.
A new database is mode 0600, with newly created private data directory mode
0700. Future migrations must increment `PRAGMA user_version`; a newer schema
causes startup to fail rather than silently downgrade. Operate a single
control-plane host; this is not a multi-region database architecture.

Create a consistent online database backup with the SQLite backup API:

```sh
python -m src.hosted.cli backup /private/backups/hosted-2026-10-11.sqlite3
```

The destination must not exist. The command runs `PRAGMA integrity_check` on
the backup. Also back up the `accounts/` dataset tree and any separately
configured remote lifecycle databases. Quiesce new uploads and jobs while
capturing the database plus dataset tree so the snapshots agree; checkpoint
artifacts have their own object-store retention/replication policy. Encrypt
backups and protect them like production data: they contain password hashes,
project data and session hashes. Do not copy an active SQLite main file alone,
since committed transactions may still be in its WAL.

Restore rehearsal:

1. Stop the API and dispatchers; independent cleanup must continue supervising
   already provisioned resources. Preserve the current data directory.
2. Restore the backup and matching `accounts/` tree into a new private directory.
   Do not reuse old `-wal`/`-shm` files alongside the restored database.
3. Start a test factory pointing at that directory, with paid execution off.
   Check integrity, log in, list projects/jobs, and verify upload checksums.
   Revoke restored sessions (`DELETE FROM sessions` in an operator transaction)
   if rollback would revive a session revoked after the backup.
4. Reconcile provider inventory, prepaid reservations and artifact records
   before re-enabling execution. Record recovery time and any data loss.

Rollback uses the previous application only if its supported schema matches;
otherwise restore the known matching backup with the above reconciliation.
Tests exercise backup restoration, concurrent transactions, session revocation,
project/job/dataset/artifact/event isolation, strict configuration, and CPU-only
startup. Real object-store grants, rented-worker isolation and TLS deployment
remain integrated pilot checks rather than claims of this identity change.
