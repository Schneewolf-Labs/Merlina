# Models in object storage (`store:` references)

Merlina can load base models from your own object storage instead of the
HuggingFace Hub. Push a model once, then refer to it as `store:<name>`
anywhere a base model goes: training (`base_model`), the Playground, layer
detection, the tokenizer preview, and merges and GGUF export after training.

```bash
merlina models push /mnt/models/C0 C0 --dry-run   # what would upload
merlina models push /mnt/models/C0 C0             # -> <artifact bucket>/models/C0
```

```json
{"base_model": "store:C0"}
```

## How it resolves

- Models live in the artifact store (`ARTIFACT_STORE=internal` for MinIO on
  the LAN, or `external` for R2) under `models/<name>`. Each push publishes
  a new generation atomically.
- The first job that needs a model pulls the current generation into
  `<DATA_DIR>/model-cache/<bucket>/<name>/<generation>`, checking each
  file's sha256. Later jobs reuse that copy. A newer push is pulled on next
  use, and the superseded copy is removed. Only one pull of a model runs at
  a time, even across jobs and DDP ranks.
- **Offline:** if the store can't be reached, the newest complete cached
  copy is used, with a warning. Pre-flight accepts a cached copy too.
- **Metadata:**
  - Job records, model cards and uploads keep the `store:` reference, never
    the cache path.
  - Saved adapters get the reference written back into
    `adapter_config.json`, so they stay portable.
  - On the Hub, a model card credits a mirrored Hub model
    (`store:hf/<org>/<repo>`) as its `base_model`. Your own stored models
    get no Hub link.
- **Names:** letters, digits, `.`, `_` and `-`, with up to three `/`
  separators (`hf/meta-llama/Llama-3.1-8B`).

## Remote runs

Remote training, remote merges and remote LoRA serving all accept a
`store:` base model when the artifact store is the **external** tier (R2).
The worker gets the model store's address in its job payload, uses the
artifact credentials it already has, and pulls the model into its own
cache. Pre-flight refuses the internal (MinIO) tier for remote runs,
because a rented instance can't reach the LAN.

A whole stored model can be served without training anything:
`model.store_model` (see [remote inference](remote-inference.md)).
