# Internal dataset store

Train from an S3-compatible object store instead of depending on the
HuggingFace Hub. The store is **backend-agnostic** — the same code talks to a
self-hosted **MinIO** (the private primary) or to **Cloudflare R2** (a later,
shareable tier); only the endpoint URL and credentials differ.

## Why

- **Resilience** — a training run no longer breaks when the Hub rate-limits, is
  down, or a dataset gets pulled.
- **Reproducibility** — every revision is pinned; two runs of the same config
  train on byte-identical data.
- **Privacy** — datasets that must not live on a public hub (internal IR data,
  DPO pairs, agent traces) have a first-class home.

## Addressing

Datasets are referenced by an `swl://` URI:

```
swl://athanorlite-dpo             # the 'latest' revision
swl://athanorlite-dpo@2026-10-08  # a specific, pinned revision
```

Use it as a dataset source in either of two ways:

```python
# explicit
create_loader(source_type="internal", uri="swl://athanorlite-dpo@2026-10-08", store=store)

# shorthand: an swl:// repo_id is recognized as an internal source
create_loader_from_config({"repo_id": "swl://athanorlite-dpo"}, store=store)
```

`store` is an `InternalStore`, built from server settings:

```python
from config import settings
from dataset_handlers.internal_store import InternalStore, InternalStoreConfig

cfg = InternalStoreConfig.from_settings(settings)   # None if not configured
store = InternalStore(cfg) if cfg else None
```

## Bucket layout

```
datasets/<name>/<rev>/manifest.json   # describes the revision
datasets/<name>/<rev>/data-*.parquet  # the shards
datasets/<name>/latest                # text file holding the current <rev>
```

### Manifest

```json
{
  "name": "athanorlite-dpo",
  "rev": "2026-10-08",
  "format": "parquet",
  "data_files": ["s3://datasets/datasets/athanorlite-dpo/2026-10-08/data-0.parquet"],
  "num_rows": 12345,
  "source": { "hf_repo": "schneewolflabs/Athanorlite-DPO", "hf_revision": "<commit-hash>" },
  "created_at": "2026-10-08T00:00:00Z"
}
```

`source.hf_revision` records the exact upstream commit a mirror came from, so a
run can prove what it trained on. Our own datasets omit the HF fields.

## Configuration

Set in the server's `.env` (never in code or version control):

```
STORAGE_INTERNAL_ENDPOINT=http://minio.lan:9000
STORAGE_INTERNAL_BUCKET=datasets
STORAGE_INTERNAL_ACCESS_KEY=...
STORAGE_INTERNAL_SECRET_KEY=...
STORAGE_INTERNAL_REGION=auto
```

(The older `S3_*` / `R2_*` / `DATASET_STORE` names still work as deprecated
aliases; see [Storage tiers](storage-tiers.md#renamed-settings).)

Leave unset to disable — Merlina then loads straight from HuggingFace as before.

### External tier (R2)

The internal MinIO store is LAN-only, so a rented GPU box (RunPod) can't reach
it. The **external tier** is a second S3 backend — same code, same `swl://`
names — reachable off-LAN. It's configured with a `STORAGE_EXTERNAL_*` block
mirroring the internal one, and a per-process selector:

```
STORAGE_EXTERNAL_ENDPOINT=https://<acct>.r2.cloudflarestorage.com
STORAGE_EXTERNAL_BUCKET=merlina-datasets
STORAGE_EXTERNAL_ACCESS_KEY=...
STORAGE_EXTERNAL_SECRET_KEY=...
STORAGE_EXTERNAL_REGION=auto

# which tier THIS process resolves swl:// against: internal (default) | external
DATASET_TIER=internal
```

On sabre everything stays `internal` (MinIO). A rented GPU box sets
`DATASET_TIER=external` and supplies the `STORAGE_EXTERNAL_*` vars; the loaders, mirror, and
`swl://` names are unchanged — only the backend differs.

Data reaches the external tier **only** via an explicit promote:

```bash
merlina datasets promote swl://hf/org/ds/train --to external --dry-run  # cost estimate, writes nothing
merlina datasets promote swl://hf/org/ds/train --to external            # actually copy
merlina datasets info swl://hf/org/ds/train --store external            # verify
```

Promotion is one-way (internal is the source of truth), content-addressed (so a
copy is byte-identical and re-promoting a present rev is a no-op unless
`--overwrite`), and **refuses private data**: any revision whose manifest is
flagged `private: true`, or whose name matches a private marker
(`vps-incident`, `schneewolf-west`, `incident`, `private/`, …), is blocked.
West/IR data never leaves the internal store. `--allow-private` exists as a
deliberate override but is not for that data.

## Status / roadmap

- **Phase 1 (this PR): read path.** `swl://` addressing, manifests, the loader,
  factory dispatch, config. Reads revisions that already exist in the store.
- **Phase 2: stand up MinIO on sabre** (see below).
- **Phase 3 (done): mirror-on-first-use.** An `huggingface` source with
  `prefer_internal` is served from the store if mirrored; on a miss, pulled from
  HF once, snapshotted as parquet, and a sha-pinned manifest written. Streaming
  sources are left direct. See `dataset_handlers/mirror.py`.
- **Phase 3.5 (this PR): external tier + promote.** A second `R2_*` backend
  selected by `DATASET_STORE`, a one-way `merlina datasets promote` (with
  `--dry-run` cost estimate) to copy revisions internal→external, and a private
  guard so west/IR data can't be promoted. See `dataset_handlers/promote.py`.
- **Phase 4: migrate** the datasets we actually train on and cut configs over.

## Phase 2 — standing up MinIO on sabre (operator steps)

MinIO lives next to Forgejo on `/mnt/ZABA`, LAN-only. Run on **sabre**:

```bash
# /mnt/ZABA/minio/compose.yaml
mkdir -p /mnt/ZABA/minio/data
cat > /mnt/ZABA/minio/compose.yaml <<'YAML'
services:
  minio:
    image: quay.io/minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    ports:
      - "9000:9000"   # S3 API
      - "9001:9001"   # console
    volumes:
      - /mnt/ZABA/minio/data:/data
    restart: unless-stopped
YAML

# credentials in /mnt/ZABA/minio/.env (you create this; do not commit it):
#   MINIO_ROOT_USER=...
#   MINIO_ROOT_PASSWORD=...
docker compose -f /mnt/ZABA/minio/compose.yaml --env-file /mnt/ZABA/minio/.env up -d

# create the bucket + a scoped access key via the console at http://sabre:9001
# then put STORAGE_INTERNAL_* into the Merlina server's .env
```

Keep it LAN-only (like Forgejo). R2 is added later, as a second tier, only for
datasets that need to reach rented GPUs — and private/IR data is never synced there.
