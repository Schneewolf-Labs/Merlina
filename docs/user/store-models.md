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

## Publishing trained models to the store

- **During training:** set `publish_to_store: true` (or tick **Publish to the
  model store** in the form). When training finishes, the model goes to
  `models/<output_name>` (or `store_model_name`). A LoRA run is merged
  first when `merge_lora_before_upload` is on (the default), reusing the
  same merge as a Hub upload or GGUF export. Otherwise the adapter is
  published as is.
- **After training:** `POST /models/{name}/publish`
  (`{"store_name": ..., "merged": false, "include_gguf": true}`) publishes
  an existing model directory as a background job.
- **From the CLI:** `merlina models push ./models/<name> <name>` pushes any
  directory: a full model, an adapter or GGUF files.

**What gets uploaded**
- The model files, plus `README.md`. The model card is also written into
  the local output directory.
- `gguf/` exports if they exist.
- Never `checkpoint-*` folders or the upload sidecar.

**Where the result is recorded**
- The job's `store_publish` metric.
- The model's upload history (`upload_state.json`), with
  `destination: "store"`.

**Remote runs**
- With a remote merge, the merge worker publishes straight to R2, so the
  merged model never travels through your machine.
- With a local merge, or for an adapter-only run, the control plane
  publishes.
- Remote publishing needs `ARTIFACT_STORE=external`.

## Mirroring Hub models

```bash
merlina models mirror Qwen/Qwen3-8B            # -> store:hf/Qwen/Qwen3-8B
merlina models mirror Qwen/Qwen3-8B --revision v1.0
```

This downloads the model once at the Hub's current commit (or
`--revision`) and publishes it as `hf/<org>/<repo>`. The manifest records
the repo and commit sha. Running it again re-publishes only if the Hub
moved (`--force` re-publishes anyway). Formats Merlina never loads are
skipped (`.pth`, `.pt`, `.gguf`, `.onnx`, `original/`), and so is `.bin`
when the repo also has safetensors.

Set `MODEL_MIRROR=true` to do this automatically. Every Hub id used as a
base model is then mirrored on first use and loaded from the store after
that, while configs and cards keep the plain Hub id. When neither the Hub
nor the store can be reached, a cached mirror still loads. If mirroring
fails for any other reason, Merlina logs a warning and loads from the Hub
as before. The setting only applies on the control plane; remote workers
download Hub ids directly unless you pass `store:hf/...`.

Model cards credit a `store:hf/<org>/<repo>` base to its Hub id.

## Remote runs

Remote training, remote merges and remote LoRA serving all accept a
`store:` base model when the artifact store is the **external** tier (R2).
The worker gets the model store's address in its job payload, uses the
artifact credentials it already has, and pulls the model into its own
cache. Pre-flight refuses the internal (MinIO) tier for remote runs,
because a rented instance can't reach the LAN.

A whole stored model can be served without training anything:
`model.store_model` (see [remote inference](remote-inference.md)).
