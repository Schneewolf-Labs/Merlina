# Storage tiers

Merlina can talk to up to three S3-compatible object stores. The storage-tiers
feature lets you manage them from one place, through the `merlina storage` CLI,
the `/storage/*` API or the web UI's **Object storage** section. You can browse
buckets, check usage, copy or move prefixes between tiers, and delete prefixes.

| Tier        | Settings              | Privacy | Typical backend          | Holds                                    |
|-------------|-----------------------|---------|--------------------------|------------------------------------------|
| `internal`  | `S3_*`                | private | self-hosted MinIO (LAN)  | datasets (`datasets/<name>/<rev>/...`)   |
| `external`  | `R2_*`                | shared  | Cloudflare R2, any S3    | promoted datasets, remote-run artifacts  |
| `artifacts` | `MERLINA_ARTIFACT_*`  | shared  | any S3                   | `runs/<job_id>/...` adapters and merged models |

`artifacts` shows up only when it is configured **and** points at a different
endpoint or bucket than `external`. When both use the same bucket (a common
setup), `external` already covers the artifact prefix and the UI opens it there.

There are no new settings. The tiers reuse the variables described in
[internal-dataset-store.md](internal-dataset-store.md) and
[dev/artifact-storage.md](dev/artifact-storage.md). Credentials are never
returned by the API or printed by the CLI.

## CLI

```bash
merlina storage tiers [--usage]                  # configured tiers (+ object count / bytes)
merlina storage ls internal datasets/            # one level: sub-prefixes and objects
merlina storage du external runs/                # object count and total bytes
merlina storage buckets external [--create NAME] # list or create buckets (see "Buckets")

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

In the web UI, you confirm by typing the prefix. While a sync involving a tier
is running, the API refuses deletes on that tier.

## Buckets

`buckets` lists the buckets the tier's key can see, and `--create` makes one.
Keys scoped to a single bucket usually can't do either. Merlina then reports
that access was denied and doesn't raise an error. Create buckets with an admin
key in the provider's console or CLI, then grant the scoped key access.

## API

All routes are under the **Storage** tag (see `/docs`).

| Method | Path | Notes |
|--------|------|-------|
| GET  | `/storage/tiers?usage=` | tiers with bucket, endpoint host, privacy, `configured` (+ usage) |
| GET  | `/storage/ls?tier=&prefix=` | delimiter-style listing |
| GET  | `/storage/du?tier=&prefix=` | object count and bytes |
| POST | `/storage/sync` | `{src, dst, prefix, dry_run=true, move, verify, overwrite}`. A dry run returns `{plan}`; a real run returns `{id}` |
| GET  | `/storage/sync` | recent sync jobs |
| GET  | `/storage/sync/{id}` | `state`, `progress` (phase, objects and bytes done/total, current key), `result` |
| POST | `/storage/sync/{id}/cancel` | stops after the current object; a move then deletes nothing |
| POST | `/storage/delete` | `{tier, prefix, confirm, dry_run}` |

Only one sync runs at a time per source/destination pair. A second request gets
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
