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
S3_ENDPOINT_URL=http://sabre:9000
S3_DATASET_BUCKET=datasets
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_REGION=auto
```

Leave unset to disable — Merlina then loads straight from HuggingFace as before.

## Status / roadmap

- **Phase 1 (this PR): read path.** `swl://` addressing, manifests, the loader,
  factory dispatch, config. Reads revisions that already exist in the store.
- **Phase 2: stand up MinIO on sabre** (see below).
- **Phase 3: mirror-on-first-use.** An `huggingface` source with `prefer_internal`
  is served from the store if mirrored; on a miss, pulled from HF once, snapshotted
  as parquet, and a revision-pinned manifest written.
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
# then put S3_* into the Merlina server's .env
```

Keep it LAN-only (like Forgejo). R2 is added later, as a second tier, only for
datasets that need to reach rented GPUs — and private/IR data is never synced there.
