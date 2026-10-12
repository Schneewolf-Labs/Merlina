# Storage tiers

Merlina talks to up to two S3-compatible object stores, the **internal** and
**external** tiers. The storage-tiers feature lets you manage them from one
place, through the `merlina storage` CLI, the `/storage/*` API or the web UI's
**Object storage** section. You can list, create and delete buckets, browse and
preview their contents, check usage, copy or move prefixes between tiers and
buckets, and delete prefixes.

| Tier       | Settings              | Privacy | Typical backend          | Holds                                   |
|------------|-----------------------|---------|--------------------------|-----------------------------------------|
| `internal` | `STORAGE_INTERNAL_*`  | private | self-hosted MinIO (LAN)  | datasets (`datasets/<name>/<rev>/...`)  |
| `external` | `STORAGE_EXTERNAL_*`  | shared  | Cloudflare R2, any S3    | promoted datasets, remote-run artifacts |

Remote-run artifacts live **in a tier** (`ARTIFACT_STORE=external`). They sit
under `ARTIFACT_PREFIX` in the tier's bucket, or in `ARTIFACT_BUCKET` on the
same endpoint, which then shows up as a protected default bucket of that tier.
The UI opens the artifact prefix directly. Only a legacy custom artifact
location (see below) still appears as a separate `artifacts` tier.

## Configuration

Both tiers have the same shape:

```dotenv
STORAGE_INTERNAL_ENDPOINT=http://minio.lan:9000
STORAGE_INTERNAL_BUCKET=datasets
STORAGE_INTERNAL_ACCESS_KEY=...
STORAGE_INTERNAL_SECRET_KEY=...
STORAGE_INTERNAL_REGION=auto                  # optional, default auto
STORAGE_INTERNAL_ADMIN_ACCESS_KEY=...         # optional, bucket operations only
STORAGE_INTERNAL_ADMIN_SECRET_KEY=...

STORAGE_EXTERNAL_ENDPOINT=https://<account>.r2.cloudflarestorage.com
STORAGE_EXTERNAL_BUCKET=merlina-shared
STORAGE_EXTERNAL_ACCESS_KEY=...
STORAGE_EXTERNAL_SECRET_KEY=...
# ...same optional _REGION / _ADMIN_* keys

DATASET_TIER=internal        # which tier this process resolves swl:// against
DATASET_MIRROR=false         # mirror HF datasets into the internal tier on first use

ARTIFACT_STORE=external      # hf_hub (default) | external | internal
ARTIFACT_BUCKET=             # optional; defaults to the tier's bucket
ARTIFACT_PREFIX=merlina/runs
```

A tier name in `ARTIFACT_STORE` reuses that tier's endpoint and keys, so
credentials are never duplicated. `ARTIFACT_STORE=internal` is fine for local
use. Remote (RunPod) runs are refused at plan and preflight time with it,
because remote workers can't reach a LAN-only tier; they need `external`.

Remote workers still receive the artifact keys under the original names
`MERLINA_ARTIFACT_ACCESS_KEY`, `MERLINA_ARTIFACT_SECRET_KEY` and
`MERLINA_ARTIFACT_SESSION_TOKEN`, built from the new settings. That is the
worker wire format, so already-deployed worker images keep working.

### Renamed settings

The old names still work as deprecated aliases. Merlina logs one warning
listing any that are set (names only, never values). When both an old and a
new name are set, the new one wins. `merlina config migrate-env` renames them
in a `.env` file.

| Old | New |
|-----|-----|
| `S3_ENDPOINT_URL` | `STORAGE_INTERNAL_ENDPOINT` |
| `S3_DATASET_BUCKET` | `STORAGE_INTERNAL_BUCKET` |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | `STORAGE_INTERNAL_ACCESS_KEY` / `STORAGE_INTERNAL_SECRET_KEY` |
| `S3_REGION` | `STORAGE_INTERNAL_REGION` |
| `R2_ENDPOINT_URL` | `STORAGE_EXTERNAL_ENDPOINT` |
| `R2_DATASET_BUCKET` | `STORAGE_EXTERNAL_BUCKET` |
| `R2_ACCESS_KEY` / `R2_SECRET_KEY` | `STORAGE_EXTERNAL_ACCESS_KEY` / `STORAGE_EXTERNAL_SECRET_KEY` |
| `R2_REGION` | `STORAGE_EXTERNAL_REGION` |
| `DATASET_STORE` | `DATASET_TIER` |
| `PREFER_INTERNAL` | `DATASET_MIRROR` |
| `MERLINA_ARTIFACT_STORE=hf_hub` | `ARTIFACT_STORE=hf_hub` |
| `MERLINA_ARTIFACT_STORE=s3` + `_S3_ENDPOINT` / `_S3_BUCKET` / `_S3_REGION` / `_ACCESS_KEY` / `_SECRET_KEY` | `ARTIFACT_STORE=<tier>` (+ `ARTIFACT_BUCKET` if the bucket differs). This applies when the endpoint and keys match that tier's; otherwise see "Custom artifact location" below |
| `MERLINA_ARTIFACT_S3_PREFIX` | `ARTIFACT_PREFIX` |

`MERLINA_CHECKPOINT_TIMEOUT_SECONDS` is unchanged.

### Migrating a .env

```bash
merlina config migrate-env            # dry run on ./.env: prints planned key changes only
merlina config migrate-env path/.env --write
```

- **Dry run:** lists renames, removals and replacements by **key name only**
  and flags conflicts, for example an old and a new name both set with
  different values. A conflicting pair is left in place for you to resolve.
  Values are compared in memory and never printed.
- **Duplicate artifact settings:** a `MERLINA_ARTIFACT_*` block whose endpoint
  and keys equal a tier's collapses to `ARTIFACT_STORE=<tier>`, and the
  duplicate keys are removed.
- **`--write`:**
  - saves a timestamped backup next to the file (`.env.bak-YYYYmmdd-HHMMSS`,
    mode 600);
  - rewrites the file in place, keeping comments, ordering and unrelated lines
    byte-for-byte;
  - leaves the file at mode 600.

  Running it again reports nothing to do.

### Custom artifact location (legacy)

A `MERLINA_ARTIFACT_STORE=s3` block can't be collapsed onto a tier when its
keys differ from every tier's, for example dedicated keys scoped to the job
prefix, or a session token. It is then kept as a custom artifact location, and
those keys are what workers receive. This is the only remaining use of the
`MERLINA_ARTIFACT_*` operator settings. The migration tool leaves such a block
untouched.

A tier is one endpoint plus one **default bucket** (its configured dataset or
artifact bucket). Every command, route and UI action can also target **any
other bucket on the same endpoint**. That bucket keeps the tier's privacy: any
bucket on the internal endpoint counts as private and any bucket on the
external one as shared. So the privacy guard also covers cross-bucket syncs.

## CLI

```bash
merlina storage tiers [--usage]                  # configured tiers (+ object count / bytes)
merlina storage ls internal datasets/            # one level: sub-prefixes and objects
merlina storage ls internal runs/ --sort size --desc --search adapter --limit 50
merlina storage du external runs/                # object count and total bytes
merlina storage buckets internal [--sizes]       # buckets on the endpoint (* = default)
merlina storage mb internal scratch-bucket       # make a bucket
merlina storage rb internal scratch-bucket       # preview: object count and size
merlina storage rb internal scratch-bucket --yes [--empty]   # remove (--empty if not empty)

# Any command can target another bucket on the tier's endpoint
merlina storage ls internal --bucket scratch-bucket
merlina storage sync internal external runs/ --src-bucket scratch-bucket --dst-bucket other-bucket

# Copy everything under runs/ from the shared tier down into the private one
merlina storage sync external internal runs/ --dry-run
merlina storage sync external internal runs/

# Options: --verify (hash both sides), --overwrite (recopy), --move (delete source after verify)
merlina storage sync external internal runs/job_123/ --move

merlina storage rm external scratch/old-run/          # preview only
merlina storage rm external scratch/old-run/ --yes    # delete
```

## How sync works

`sync SRC DST PREFIX` copies every object under `PREFIX` in SRC to the **same
key** in DST.

- **Streaming.** Objects are streamed in 32 MiB chunks, so multi-GB artifacts
  never sit in memory or on local disk. Objects larger than one chunk become
  multipart uploads.
- **SHA-256 metadata.** The digest is computed while streaming and stored on the
  destination object as S3 user metadata (`x-amz-meta-sha256`).
  - Small objects get it in the same PUT.
  - Multipart uploads get it afterwards through a server-side, metadata-only
    `CopyObject`, so no data is transferred twice.
  - Providers cap `CopyObject` at 5 GiB. Larger objects are kept without the
    metadata and listed as `meta_missing`. Later runs compare them by size, or
    by hash with `--verify`.
  - No sidecar files are written.
- **Skipping.** An object is skipped when DST already has the same size and:
  - both sides carry `sha256` metadata and the digests match; or
  - metadata is missing on either side. With `--verify`, both sides are hashed
    first and the object is skipped only if the hashes match.

  Otherwise the object is copied. `--overwrite` always recopies.
- **Post-write check.** The size is checked after every write. With `--verify`,
  the written object is read back and hashed.
- **Move.** `--move` implies `--verify`. Source objects are deleted only after
  every copy (and every skipped "already there" object) has been verified by
  hash and the run had no errors. If anything fails, nothing is deleted.
- **Plan.** A dry run (and every result) reports `to_copy`, `skipped`,
  `refused` and `errors`, with byte totals.

## The privacy guard

When SRC is private (`internal`) and DST is shared, sync **refuses**, reports
and never copies:

- any key whose path matches a private marker (the same list as
  `dataset_handlers/internal_store.PRIVATE_NAME_MARKERS`, for example
  `incident` or `private/`);
- any key under `datasets/<name>/` when **any** revision manifest of `<name>`
  has `"private": true`, or when a manifest can't be read (the guard fails
  closed). The `latest` pointer is refused too.
- any key inside a **remote-training artifact** unless its manifest proves the
  training data is shareable. This covers adapters, merged models, checkpoints
  and staged inputs, for example under `runs/<job>/<artifact>/`. The artifact
  root is the nearest ancestor directory whose `manifest.json` is in the
  artifact-store format (`files` / `generation` / `metadata`, see
  `src/remote/s3_artifacts.py`). Every key under that root is refused, including
  the manifest itself and `generations/...`, when:
  - any dataset source (`source`, `additional_sources[]`, `eval_source` in
    `metadata.datasets` or `metadata.training_config.dataset`) has a `repo_id`,
    `name`, `uri` or path that matches a private marker;
  - an `internal` / `swl://` source points to a dataset that is private (by
    manifest flag or name), or that is not in the source tier, so it can't be
    checked;
  - a source is `local_file` or `upload`. Their contents can't be known from the
    artifact, so they are treated as private. Datasets staged for remote
    workers are recorded this way, so artifacts trained on them are refused.
  - the base model id matches a private marker;
  - the manifest records no dataset metadata at all, or can't be read.

  The only artifacts that can be shared are those whose sources are all
  public HuggingFace repos or non-private internal datasets. Artifact
  generation data with no manifest (for example, an orphaned upload) is
  refused too.

The only override is the CLI's `--allow-private` flag, which works like
`merlina datasets promote --allow-private`. The API rejects an `allow_private`
field (HTTP 422), and the web UI never offers it. In the UI, Apply appears only
after a preview, so a private-to-shared action always shows the guard result
(including the refused keys) before anything is copied. A `--move` never
deletes refused objects.

## Delete

`rm` / `POST /storage/delete` delete every object under a prefix. They:

- refuse an empty prefix (the whole bucket);
- require `--yes` / `confirm: true`;
- treat the prefix as a directory, so `runs/a` matches `runs/a` and `runs/a/...`
  but not `runs/ab`.

In the web UI, you confirm by typing the prefix. While a sync involving a bucket
is running, the API refuses deletes in that bucket.

## Buckets

`buckets` lists the buckets visible to the tier's key, marking the configured
**default** buckets. That includes the artifact bucket when it lives on the
same endpoint. `mb` creates a bucket. New names are checked against the S3
naming rules, in both the backend and the UI. `rb` deletes a bucket, with these
guards:

- **configured buckets** (dataset or artifact) are always refused;
- the **bucket name must be typed back** (`confirm_name`; the CLI's `--yes`
  stands in for it);
- a **non-empty bucket** is refused unless you choose to empty it first
  (`empty_first` / `--empty`). The confirmation shows its object count and size
  before you do.

The API refuses a bucket delete while a sync involving that bucket is running.

### Admin keys (optional)

The normal keys are usually scoped to one bucket, so they can't list, create or
delete buckets. When that happens, Merlina still shows the configured bucket
and an inline hint naming the setting that would allow it. You can add an
optional **admin key per tier**. It is used **only** for bucket-level
operations: listing, creating and deleting buckets, and browsing buckets other
than the default. Everything else keeps using the normal key, which is also the
fallback when no admin key is set.

| Tier | Settings |
|------|----------|
| `internal` | `STORAGE_INTERNAL_ADMIN_ACCESS_KEY`, `STORAGE_INTERNAL_ADMIN_SECRET_KEY` |
| `external` | `STORAGE_EXTERNAL_ADMIN_ACCESS_KEY`, `STORAGE_EXTERNAL_ADMIN_SECRET_KEY` |

What each key needs:

- **MinIO:** a service account (for example from `mc admin user svcacct add`)
  whose policy allows `s3:ListAllMyBuckets`, `s3:CreateBucket`,
  `s3:DeleteBucket` and `s3:GetBucketLocation`. To browse or empty other
  buckets, it also needs object operations on them: `s3:ListBucket`,
  `s3:GetObject`, `s3:PutObject` and `s3:DeleteObject`.
- **Cloudflare R2:** an account API token with **Admin Read & Write**. Bucket
  creation and deletion are account-level operations, which a token scoped to
  one bucket can't do.

The API reports only whether an admin key is configured
(`admin_configured`), never the key itself.

## Browsing and previews

The web UI lists one folder level at a time, with breadcrumbs, a filter for the
current folder, sorting by name, size or modified time, and pagination (100
entries per page). The API equivalent is `GET /storage/ls`, with `search`,
`sort`, `order`, `offset` and `limit`. Clicking an object opens its details:

- size, modified time, content type, ETag and user metadata, including the
  `sha256` written by sync;
- an **inline preview**, read with range requests and never the whole object:
  - **text-like files** (json, jsonl, md, txt, jinja, yaml, csv, and files with
    no extension): the first 256 KiB, with JSON pretty-printed when it's
    complete; larger files show a "truncated" note;
  - **`.safetensors`**: only the header (an 8-byte length read, then the header
    itself), showing tensor names, dtypes, shapes and total parameters;
  - **`.parquet`**: the schema and the first 20 rows, read through pyarrow over
    range requests (the footer plus the first pages);
- a **download** button. Downloads are always streamed through the Merlina
  server, never through a presigned URL. The private tier is LAN-only, and a
  presigned URL would hand the browser the key ID.

## API

All routes are under the **Storage** tag (see `/docs`).

| Method | Path | Notes |
|--------|------|-------|
| GET  | `/storage/tiers?usage=` | tiers with default bucket(s), endpoint host, privacy, `configured`, `admin_configured` (+ usage) |
| GET  | `/storage/buckets?tier=` | buckets on the endpoint (`default` flagged). If listing is denied: the configured bucket(s), `ok: false` and a `hint` |
| GET  | `/storage/buckets/info?tier=&bucket=` | region, creation date, default flag (each best effort) |
| POST | `/storage/buckets` | `{tier, name}`: create a bucket |
| POST | `/storage/buckets/delete` | `{tier, name, confirm_name, empty_first, dry_run}` |
| GET  | `/storage/ls?tier=&bucket=&prefix=&search=&sort=&order=&offset=&limit=` | one folder level, filtered, sorted, paged |
| GET  | `/storage/du?tier=&bucket=&prefix=` | object count and bytes |
| GET  | `/storage/object?tier=&bucket=&key=` | object details and user metadata |
| GET  | `/storage/preview?tier=&bucket=&key=` | text, safetensors header or parquet head |
| GET  | `/storage/download?tier=&bucket=&key=` | streamed download |
| POST | `/storage/sync` | `{src, src_bucket, dst, dst_bucket, prefix, dry_run=true, move, verify, overwrite}`. A dry run returns `{plan}`; a real run returns `{id}` |
| GET  | `/storage/sync` | recent sync jobs |
| GET  | `/storage/sync/{id}` | `state`, `progress` (phase, objects and bytes done/total, current key), `result` |
| POST | `/storage/sync/{id}/cancel` | stops after the current object; a move then deletes nothing |
| POST | `/storage/delete` | `{tier, bucket, prefix, confirm, dry_run}` |

Only one sync runs at a time per source/destination bucket pair. A second request gets
HTTP 409. Sync jobs live in memory and are lost when the server restarts. A
restart mid-copy leaves no partial object, because an interrupted multipart
upload is never completed. Re-running the sync skips everything already copied.

## Limits

- Objects are copied one at a time. The metadata HEAD requests made while
  planning run in parallel.
- The largest object that can be written is about 320 GiB (10,000 parts of
  32 MiB).
- Comparisons use only `sha256` user metadata, never ETags, because multipart
  ETags are not checksums.
