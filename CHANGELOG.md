# Changelog

All notable changes to Merlina will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-05-24 "Workshop"

### Added
- **Diffusion training is now a first-class Merlina backend** via [Atelier](https://github.com/Schneewolf-Labs/atelier). Three new training modes — `diffusion_qwen_image` (Qwen-Image text-to-image LoRA), `diffusion_qwen_edit` (Qwen-Image-Edit image-to-image LoRA), `diffusion_sdxl` (SDXL LoRA) — accepted by the existing `TrainingConfig` and dispatched to a new `src/training_runner_diffusion.py` runner. Same job lifecycle / WebSocket / preflight surface as the text + VLM paths.
- **`src/training_runner_diffusion.py`**: parallel narrow path mirroring `run_vlm_training_sync`'s contract but the body is Atelier-driven — loads an Atelier `ModelAdapter` (`QwenImageAdapter` / `QwenEditAdapter` / `SDXLAdapter`), runs `cache_embeddings` to pre-compute text + image embeddings (with the staged-load pattern so encoders + transformer don't have to coexist in VRAM during the cache pass), frees encoders, moves the transformer onto the GPU, then drives `AtelierTrainer` with `FlowMatchingLoss` + LoRA. Reuses the existing `WebSocketCallback` unchanged — Atelier's callback fire loop is duck-typed so the same callback object works across both engines.
- **Generalized dispatch**: the `training_mode.startswith("vlm_")` if-statement in `run_training_sync` is now a `_resolve_sibling_runner(config)` helper that picks a parallel runner by (a) explicit `model_type` (new `'diffusion'` value joins `'vlm'`) or (b) legacy `training_mode` prefix sniffing for backward compatibility. Easy to extend further: add a new `model_type` value and a `training_runner_<x>.py`, register both in the helper.
- **`TrainingConfig` (Pydantic) gains diffusion fields** — all `Optional` so text-mode / VLM-mode requests are unaffected:
  - `model_name` (HF repo id of the diffusion model; falls back to `base_model` if unset)
  - `image_resolution` (square cache + train resolution, default 1024)
  - `lora_rank` (overrides `lora_r` for diffusion runs)
  - `lora_target_modules` (DiT/UNet target list, e.g. `["to_k","to_q","to_v","to_out.0"]`)
  - `dataset_jsonl_path` / `dataset_name` / `dataset_split` (three ways to source the image+caption corpus)
- **Frontend: diffusion picker + form**: `🧠 Model Type` gains a `Diffusion / Image LoRA` option, the `Training Method` dropdown groups text vs diffusion modes under `<optgroup>`s, and a new `🎨 Diffusion Settings` panel (image resolution, LoRA rank override, target modules, dataset JSONL path, HF dataset name) shows when a `diffusion_*` mode is selected.
- **`atelier` + `diffusers>=0.34` added to `requirements.txt`** — pulled via git URL the same way `grimoire-rl` is.
- **`tests/test_diffusion_dispatch.py`** — lightweight unit tests for the `_resolve_sibling_runner` routing, the diffusion mode registry shape, and the new Pydantic field surface. No model weights needed (heavy end-to-end tests live elsewhere and skip under pytest, mirroring `test_artemis_runner.py`).

### Changed
- **Major version bump (1.8.0 → 2.0.0)**. The job dispatcher now reads `model_type` as a first-class engine selector (was: `training_mode` prefix only). Existing text-mode + VLM jobs (no `model_type` set, or `model_type='auto'`/`'causal_lm'`/`'vlm'`) continue to dispatch to the correct runner unchanged — no breaking change for stored configs or external API consumers using the v1 surface. The prefix-sniff fallback (`vlm_*`, `diffusion_*`) means in-flight VLM configs keep working.

### Notes
- Diffusion training currently produces a LoRA at `./models/<output_name>/` in diffusers `pytorch_lora_weights.safetensors` format — directly loadable by `pipeline.load_lora_weights(...)` for inference, drag-droppable into ComfyUI / A1111. GGUF export and HF Hub upload for diffusion LoRAs are out-of-scope for 2.0 (the existing post-hoc upload endpoint can ship the saved directory manually).
- The Atelier QwenImage adapter loads the 38 GiB Qwen-Image transformer to CPU first by default (`defer_transformer=True`) and only moves it to GPU after `free_encoders()` reclaims the ~14 GiB Qwen-VL text encoder. This keeps peak VRAM = max(encoders, transformer) instead of their sum — important on 48 GB cards. Merlina's diffusion runner honors this dance.

## [1.8.0] - 2026-05-24 "Artemis Extract"

### Changed
- **Artemis VLM module extracted to a standalone package**: the model classes (`ArtemisVLMConfig`, `ArtemisVLMForConditionalGeneration`, `ArtemisVLMProjector`, `ArtemisVLMProcessor`, `ArtemisDataCollator`, `artemis_loss_fn`) now live in the dedicated [`Schneewolf-Labs/Artemis`](https://github.com/Schneewolf-Labs/Artemis) repo (package name `artemis-vlm`, apache-2.0). Merlina's `src/artemis_vlm.py` is deleted; `src/training_runner_vlm.py` now imports from the external package (`from artemis_vlm import ...`).
- **Merlina's `requirements.txt` gains `artemis-vlm @ git+https://github.com/Schneewolf-Labs/Artemis.git`** as a dependency. The package is only loaded when `training_mode` starts with `vlm_`, so text-mode users pay no runtime cost beyond the install.
- **Tests reorganized**: the four model-class hardware-smoke tests (`test_artemis_vlm.py`, `test_artemis_processor.py`, `test_artemis_collator.py`, `test_artemis_stage_gen.py`) moved to the Artemis repo's `tests/`. The Merlina-side runner-glue smoke (`tests/test_artemis_runner.py`) stays here — it tests `src/training_runner_vlm.py`, which is Merlina-specific orchestration.
- **`CLAUDE.md` Artemis section** updated to point at the external repo for model/architecture documentation, and now describes only the Merlina-side integration surface (`run_vlm_training_sync`, dispatch hook, Pydantic fields, runner glue).

### Why
Merlina is training infrastructure (FastAPI app, job queue, WebSocket); the VLM model classes are a library others should be able to consume without depending on Merlina. The original consolidation in v1.6.0 was the fastest path to a working pipeline, but it bound the model definitions to internal infra. Spinning them out is the right long-term shape: `flammenai/Mahou-2-VLM` (planned) can `pip install artemis-vlm` and graft onto Mahou-1.5 without pulling in Merlina's FastAPI app.

### Migration notes
- No API-level change: `POST /train` with `training_mode: "vlm_stage1"` or `"vlm_stage2"` works exactly the same. All Artemis Pydantic fields (`vision_model_id`, `stage`, `image_column`, `streaming`, etc.) are unchanged.
- The currently-running Stage-1 process (started under v1.7.1) is unaffected — `src.artemis_vlm` was already loaded into its Python memory before this refactor. Future restarts pick up the external package automatically via the new requirement.
- Users with a local clone should `pip install -r requirements.txt` after pulling to install the `artemis-vlm` dependency.

## [1.7.1] - 2026-05-20

### Added
- **Streaming support in the VLM dataset loader** (`_load_image_caption_dataset`): new `streaming: Optional[bool]` field on the Pydantic `TrainingConfig`. When set with a bounded `dataset.max_samples`, the runner uses `load_dataset(..., streaming=True).take(N)` and materializes only those N rows into an Arrow-backed `Dataset` via `from_generator`, with image bytes stored once through the `Image()` feature so PIL decode stays lazy at training time. The whole-corpus shard download is bypassed entirely.

### Fixed
- **Huge sharded webdataset corpora are now actually usable**: previously, pointing `vlm_stage1` at `BLIP3o/BLIP3o-Pretrain-Long-Caption` (2,891 webdataset shards) caused `load_dataset()` to download every shard before `.select(range(max_samples))` could take effect — projected ~7 hours of download for 25k usable samples, with ~1TB of disk burned on shards we'd never see. With `streaming: true`, that same 25k materializes in ~36 seconds. The non-streaming path is unchanged (still the right choice for small, single-file datasets).

## [1.7.0] - 2026-05-20 "Artemis Runner"

### Added
- **Artemis VLM training is now wired through Merlina's `/train` API**: two new training modes — `vlm_stage1` (projector-only alignment on image-caption corpora) and `vlm_stage2` (full multimodal instruction FFT) — accepted by the existing `TrainingConfig` and dispatched to a new `src/training_runner_vlm.py` runner. The text-mode `run_training_sync` is unchanged; VLM modes are handled by an early-dispatch branch that delegates to `run_vlm_training_sync(...)`.
- **`src/training_runner_vlm.py`**: parallel narrow path mirroring `run_training_sync`'s contracts (WebSocketCallback, JobManager status updates, GPU cleanup in all exit paths) but the body is multimodal-specific. Assembles `ArtemisVLMForConditionalGeneration.from_a2_and_vision(...)`, builds `ArtemisVLMProcessor`, lazy-decodes images through an internal `_ArtemisImageCaptionAdapter` (so BLIP3o-scale corpora don't have to fit in RAM), hands a custom `ArtemisDataCollator` to `GrimoireTrainer`, and uses `artemis_loss_fn` so the model's internal CE drives training. `set_training_stage()` is honored via grimoire's `requires_grad`-filtered optimizer — no manual param-group plumbing.
- **`TrainingConfig` (Pydantic, API layer) gains Artemis fields** — all `Optional` with sensible defaults so text-mode requests are unaffected:
  - `vision_model_id` (default `Qwen/Qwen3-VL-2B-Instruct`)
  - `stage` (`stage1` | `stage2`, default `stage1`)
  - `unfreeze_vision_top_n` (default 0)
  - `image_token_id` (default 22, matching A1/A2's repurposed-reserved-token layout)
  - `min_pixels` / `max_pixels` (dynamic-resolution caps)
  - `image_column` / `caption_column` (dataset column overrides)
  - `instruction` (user-side prompt paired with each image)
- **`tests/test_artemis_runner.py`** — end-to-end smoke for the runner glue: a synthetic 16-row image+caption set, the full `run_vlm_training_sync` path, asserts finite decreasing loss + `models/<name>/` checkpoint + terminal `status=completed`. Same `pytest`-collection skip pattern as the other Artemis tests; meant to be run as `python tests/test_artemis_runner.py` on the GB10.

### Notes
- Stage-1/Stage-2 are full-FT runs (no LoRA), so the post-training LoRA-merge / GGUF-export / HF-upload pipelines from the text runner are intentionally NOT wired into the VLM path yet — the saved checkpoint at `./models/<output_name>/` is the final artifact and can be uploaded with the existing post-hoc `/models/{name}/upload` endpoint. Wiring the full upload pipeline into the VLM runner is a follow-up after a real Stage-1 / Stage-2 pair has been shipped.

## [1.6.0] - 2026-05-20 "Project Artemis"

### Added
- **Project Artemis — multimodal VLM scaffolding** (`src/artemis_vlm.py`): a LLaVA-style graft adding vision-language capability to A-series (or any `MistralForCausalLM`-class) decoders. Path B by design — vision is composed *around* the decoder rather than modifying it, so the underlying text capabilities (reasoning, tools, identity) are preserved by construction.
  - `ArtemisVLMConfig` — composite config (`Qwen3VLVisionConfig` + text config + `image_token_id`/`video_token_id`).
  - `ArtemisVLMForConditionalGeneration` — composes Qwen3-VL's vision tower + a fresh 2-layer MLP projector (`vision out_hidden → text hidden`) + an unmodified language model. `from_a2_and_vision(a2_path, vision_model=...)` helper assembles without double-instantiating the decoder. `set_training_stage("stage1"|"stage2", unfreeze_vision_top_n=0)` toggles requires_grad for the two-stage recipe. `prepare_inputs_for_generation` injects the image only at step 0 so `generate()` works through standard `GenerationMixin`.
  - `ArtemisVLMProcessor` — wraps `Qwen2VLImageProcessor` with `patch/temporal/merge` sourced from the model's `vision_config` (prevents the Qwen2-VL `patch=14` vs Qwen3-VL `patch=16` per-patch-dim drift). Mirrors `Qwen3VLProcessor.__call__` expansion: each `<|image_pad|>` is replaced with `grid_thw.prod() // merge_size**2` copies, matching the model's merged feature count. `max_pixels` caps dynamic-resolution token blow-up on large images.
  - `ArtemisDataCollator` — multimodal batching: per-example processor run, prefix-trick label masking (prompt + every `<|image_pad|>` → -100), batch padding, flat `pixel_values` concat, per-image `image_grid_thw` stack. Produces a dict ready for `model(**batch)` with `labels`.
  - `artemis_loss_fn(model, batch, training)` — grimoire-compatible `(loss, metrics)` adapter; the collator's labels drive the standard CausalLM loss.
- **Four standalone tests** in `tests/test_artemis_*.py` validate (1) model assembly + merged vision-path shape + forward, (2) chat-template ↔ `<|image_pad|>` expansion round-trip on a real image, (3) batched multimodal training step with finite loss, (4) staged-freeze param counts + multimodal `generate()` + the loss adapter.
- **`CLAUDE.md`** gains an Artemis VLM section documenting the Path-B architecture, public API, and two-stage training recipe, plus a LoRA-merge / `modules_to_save` / repurposed-tokens note in *Important Implementation Notes* (lessons surfaced during the v1.5.x VLM-fix line of work).

### Notes
- The `training_runner` multimodal switch (selecting `ArtemisVLM` + `ArtemisDataCollator` + `artemis_loss_fn` + `set_training_stage` in the live training path) and image loading in `dataset_handlers` are intentionally **not in this PR** — they are data-coupled and will ship alongside the first real Stage-1/Stage-2 runs.

## [1.5.1] - 2026-05-15

### Fixed
- **VLM processor save no longer silently writes a bare tokenizer**: `AutoProcessor.from_pretrained` on an adapter directory (no `preprocessor_config.json`) returns just the tokenizer wrapped as a "processor". The merge fallback chain in `gguf_exporter.merge_lora_to_directory` and `training_runner._save_processor` now rejects results that lack an `image_processor`/`video_processor` and falls through to the base model, so VLM uploads actually contain processor config.
- **Post-hoc upload + GGUF export endpoints auto-detect VLMs**: `/models/{name}/upload` and `/models/{name}/export-gguf` previously used a strict `request.model_type == "vlm"` check, which silently treated `model_type="auto"` (the UI default) as text-only — skipping the VLM processor save and the visual-weight grafting in the merge. They now share the training-time `_get_auto_model_class` detection via a new `_resolve_is_vlm()` helper.
- **`generate_model_readme` no longer crashes on post-hoc uploads**: the function used to access training-only fields (`batch_size`, `learning_rate`, etc.) directly, which raised `AttributeError` on the minimal `SimpleNamespace` built by post-hoc endpoints. Each config row is now guarded with `getattr` and skipped when absent, so post-hoc readmes contain just the fields we know.
- **Silent failures are now visible**: `_save_processor` failures (e.g., missing `Pillow`/`torchvision`) log at WARNING instead of DEBUG.

### Added
- **`Pillow` listed in `requirements.txt`** and `torchvision` called out as a VLM requirement in `README.md`. Without these, image processors fail to load.
- **Post-hoc upload enriches READMEs from `adapter_config.json`**: LoRA rank, alpha, dropout, and target modules are pulled from the adapter config when generating the model card for a re-uploaded LoRA checkpoint.



## [1.5.0] - 2026-04-18 "Liger Familiar"

### Added
- **Multi-GPU DDP training**: When multiple GPUs are available, training now launches as a subprocess via `accelerate launch` for proper Distributed Data Parallel. New `multi_gpu_strategy` config field (`auto`/`ddp`/`single`) and UI selector. Includes a standalone `src/train_worker.py` entry point with file-based progress reporting and graceful SIGTERM stop.
- **Upload / re-upload to HuggingFace Hub for finished jobs**: New `POST /jobs/{job_id}/upload` endpoint and "Upload to Hub" button in the job monitor lets you push trained artifacts after a job has completed or stopped — no retraining required. Configurable token, repo name, merge preference, and privacy per upload.
- **Muon optimizer**: Select `muon` from the optimizer dropdown to use Grimoire's native MomentUm Orthogonalized by Newton-Schulz implementation, with `muon_momentum` config field (default `0.95`).
- **Full Adafactor configuration**: Exposes `relative_step`, `scale_parameter`, `warmup_init`, `decay_rate`, `beta1`, and `clip_threshold` through Pydantic config, training runners, and frontend UI with conditional show/hide based on optimizer selection.
- **Liger Kernel support**: New `use_liger` flag patches the model with Grimoire's Liger Kernel fused ops (RMSNorm, RoPE, SwiGLU) for faster, lower-VRAM training on Llama, Mistral, Qwen, Gemma, and Phi family models. Pre-flight check verifies the `liger-kernel` package is installed when enabled.
- **torch.compile integration**: New `torch_compile` flag wraps the model with `torch.compile` for fused PyTorch 2.x kernels.
- **NEFTune regularization**: New `neftune_alpha` field exposes Grimoire's embedding-noise regularization (set e.g. `5.0` to enable, leave empty to disable).
- **Eval-on-start**: New `eval_on_start` flag runs an initial baseline evaluation pass before training begins.
- New "🦁 Grimoire Optimizations" section in the training UI exposing all of the above.
- W&B run name suffixes (`-liger`, `-compile`) when these features are enabled.
- **Server-side secrets**: HF and W&B tokens can now be set in `.env` and clients may omit them from request bodies. New `GET /env/secrets` endpoint (booleans only) and frontend hint marking token inputs optional when the server already has them.
- **System prompt override**: New per-job system prompt override with `fill_empty` (only fill missing) and `replace_all` (overwrite every sample) modes.
- **Persistent job names**: `/jobs` now returns the stored `output_name` so the sidebar shows friendly names (e.g. `Hemlock2-Coder-7B`) instead of raw job IDs after page or server reloads.
- **Torch version pre-flight checks**: Startup warning and pre-flight check verify `torch >= 2.5.0` and torchvision compatibility.
- **Decoupled upload errors**: Upload failures no longer mark a successful training run as `failed`. Upload errors are stored in a new `upload_error` field (DB migration included), surfaced as a warning banner in the UI with a retry button.
- Favicon generated from `merlina.png`.

### Changed
- HuggingFace Hub uploads now save the merged model to disk first, then push via `upload_folder`, instead of streaming through `push_to_hub`. This enables the new VLM state-dict repair pass before upload.
- `torch`, `torchvision`, `torchaudio`, and `xformers` removed from `requirements.txt`. GPU environments (RunPod, Colab, etc.) ship a CUDA-matched torch build; letting pip upgrade torch independently broke `torchvision::nms`. README documents the correct `--index-url` install.
- Model cards now include `pipeline_tag` in YAML frontmatter (`image-text-to-text` for VLMs, `text-generation` for LLMs) plus VLM-specific tags so HuggingFace classifies and surfaces models correctly.
- Replaced deprecated `torch_dtype` with `dtype` in `from_pretrained` calls.
- Added `.env` to `.gitignore`.

### Fixed
- **VLM state dict on merge+upload**: After PEFT `merge_and_unload` on VLM architectures (e.g. Qwen3.5-VL), `save_pretrained` could produce triple-nested `language_model` keys, misplaced visual prefixes, and missing MTP/visual weights. New `fix_vlm_state_dict_on_disk()` validates and repairs the saved safetensors before upload.
- **VLMs unloadable with `AutoProcessor.from_pretrained()`**: Processor and `preprocessor_config.json` are now copied from the base model and saved/uploaded alongside the merged model.
- **VLM `generation_config.json`** is now saved from the base model so merged VLMs use the correct `eos_token_id` and other generation defaults.
- **VLMs misclassified as text-only on HuggingFace** even with vision weights present — fixed via the model card `pipeline_tag` change above.
- **HF Hub `404 Repository Not Found` on upload**: `upload_folder()` was being called with the bare `output_name` instead of the namespace-prefixed `repo_id` returned by `create_repo()`.
- **Adafactor kwargs leaked to other optimizers**: Adafactor-specific keyword arguments are now only passed to `TrainingConfig` when the optimizer is actually `adafactor`, preventing crashes with `adamw_8bit_paged` and friends.
- **Multi-GPU was doubling steps and producing NaN loss**: Direct multi-GPU runs without `accelerate launch` caused incorrect optimization-step accounting and NaN loss from `device_map="auto"` sharding. Fixed by the new DDP subprocess path.
- **`train_worker` crash modes**: Guarded `import wandb` for environments without it, propagate `was_stopped` to the upload step so stopped jobs aren't marked `completed`, guarded `eval_dataset.map()` against `None` when `test_size` produces no eval split, and wrapped config loading in `try/except` so failures land in the progress file/DB instead of crashing silently.
- **`UnboundLocalError` masquerading as a training failure**: `del trainer, model` left names unbound for the `finally` block, causing the outer `except` to record a successful run as failed. Variables are now reassigned to `None` after `del` and upload setup is wrapped in its own `try/except`.
- **Upload button invisible**: Replaced undefined `--accent-gold` CSS variable with `--wizard-yellow`, and ensured all jobs are tracked in `activeJobs` on load so status is available when opening details from the sidebar.
- **`multi_gpu_strategy` accepted invalid values**: Now typed as `Literal["auto", "ddp", "single"]` so Pydantic rejects bad input at validation time. Shared `_make_training_callback()` between `/train` and `/jobs/{job_id}/retry` to eliminate divergent behavior.



## [1.4.1] - 2026-03-28

### Fixed
- **VLM loading on transformers v5+**: `AutoModelForVision2Seq` was removed in transformers v5. Now imports `AutoModelForImageTextToText` (v5+) with fallback to `AutoModelForVision2Seq` (v4).
- Bumped minimum `transformers` requirement to `>=5.0.0`.

## [1.4.0] - 2026-03-24 "Crystal Ball"

### Added
- **VLM Support**: Auto-detects vision-language models (Qwen-VL, LLaVA, etc.) and loads with correct `AutoModelForVision2Seq` class, preserving vision capabilities. Manual override via `model_type` dropdown (Auto/CausalLM/VLM).
- **Suggested Settings Presets**: Paper-backed recommended hyperparameters for all 7 training methods. "Apply Suggested Settings" button fills in optimal learning rate, beta, epochs, etc. per method. Available via `GET /presets/{mode}` API.
- **Multi-Dataset Concatenation**: Add multiple HuggingFace datasets to combine for training. Each additional source supports its own column mapping. All sources concatenated before formatting and splitting.
- **Dataset Deduplication**: Remove duplicate samples before training with configurable strategies (prompt, chosen, prompt+chosen, exact match). Toggle in Advanced Options.
- **Dataset Previewer Navigation**: Browse dataset samples one at a time with Prev/Next buttons, jump-to-index, and position indicator. Preview endpoints support `offset`/`limit` pagination.
- **Section Navigation UI**: Sticky tabbed banner (Model, Dataset, Training, Jobs) replaces scrolling through one long page. Auto-switches to Jobs tab after submitting training.
- **Output Name Generator**: "Generate" button creates model names from base model + dataset + training method (e.g. `Qwen2.5-7B-Instruct-MyDataset-ORPO`).
- Per-source `column_mapping` field on `DatasetSource` for multi-dataset workflows.
- `GET /presets` endpoint listing all available presets.

### Changed
- SimPO suggested beta is 2.0 (was defaulting to 0.1, matching DPO — 20x too low per the SimPO paper).
- IPO suggested beta is 0.01 (was 0.1 — IPO's beta has inverted meaning, smaller = stronger margin).
- SFT suggested learning rate is 2e-4 for LoRA (was 5e-6 — LoRA rates should be ~10x full fine-tuning).
- KTO suggested batch_size is 4 (per-step batch must be >= 4 for stable KL estimation per the KTO paper).
- All preference methods suggest 1 epoch and lora_dropout=0 (paper consensus: preference methods overfit fast, dropout interferes with preference signal).
- Jobs section is always accessible via nav tab (no longer auto-hidden when empty).

### Fixed
- VLM models (e.g. `Qwen3_5ForConditionalGeneration`) were silently loaded as text-only `Qwen3_5ForCausalLM`, stripping vision components. Fixed in both training and LoRA merge paths.



## [1.3.0] - 2026-03-14 "Seven Spells"

### Added
- **Messages Format Support**: Automatic detection and conversion of the common "messages" chat dataset format
  - Multi-turn conversation support (user/assistant turns combined with double newlines)
  - System message extraction into dedicated `system` field
  - Toggleable via UI checkbox or `convert_messages_format` API parameter (enabled by default)
  - New module: `dataset_handlers/messages_converter.py`
- **DPO Mode** (Direct Preference Optimization): Log-ratio preference learning with `beta` and `label_smoothing` parameters
- **SimPO Mode** (Simple Preference Optimization): Reference-free DPO variant with length-normalized rewards and configurable `gamma` margin
- **CPO Mode** (Contrastive Preference Optimization): Reference-free contrastive learning with `label_smoothing` support
- **IPO Mode** (Identity Preference Optimization): Squared-loss DPO variant, more robust to noisy preferences
- **KTO Mode** (Kahneman-Tversky Optimization): Binary feedback optimization using prospect theory; works with unpaired data and optional rejected responses
- Dynamic hyperparameter UI: `gamma` field appears for SimPO, `label_smoothing` for DPO/CPO
- Messages format detection banner in dataset preview UI
- Example scripts for messages format usage
- Test coverage for messages format conversion and toggle behavior

### Changed
- Training mode selector expanded from 2 modes to 7 (ORPO, SFT, DPO, SimPO, CPO, IPO, KTO)
- `TrainingConfig` now accepts `beta`, `label_smoothing`, and `gamma` hyperparameters
- Dataset preview endpoints enhanced with messages format detection and conversion
- Frontend dynamically adapts parameter fields based on selected training mode



### Planned
- Semantic versioning system
- Version tracking and changelog management
- Automated version bumping tools

## [1.2.0] - 2024-12-15 "Magical Memories"

### Added
- **SFT Mode**: New Supervised Fine-Tuning mode alongside ORPO
  - Train with only chosen responses (rejected field not required)
  - Traditional supervised learning for instruction following
  - Configurable via `training_mode` parameter
- Dynamic UI that adapts based on selected training mode
- `SFTTrainer` integration from TRL library

### Changed
- Dataset requirements now flexible based on training mode
- UI displays relevant fields based on ORPO vs SFT mode selection
- Training pipeline automatically selects appropriate trainer

### Documentation
- Updated CLAUDE.md with training mode documentation
- Added SFT vs ORPO usage guidelines

## [1.1.0] - 2024-11-30 "Persistent Power"

### Added
- **Persistent Job Storage**: SQLite database for job history
  - `JobManager` class for CRUD operations
  - Job history survives server restarts
  - Training metrics time-series storage
- **Real-time WebSocket Updates**: Live training progress
  - WebSocket connections per job
  - Real-time status, metrics, and GPU memory updates
  - `WebSocketManager` for connection handling
- **Pre-flight Validation**: Configuration validation before training
  - GPU availability and VRAM checks
  - Disk space validation
  - Model access and gating checks
  - Dataset configuration validation
  - Hyperparameter sanity checks
- **Job Queue System**: Priority-based job queue
  - Configurable concurrent job limit
  - Priority levels: LOW, NORMAL, HIGH
  - Queue position tracking
  - Job cancellation support
- **Local Model Support**: Load models from local directories
  - Absolute and relative path support
  - Automatic path detection
  - Pre-flight validation for local paths
- **HuggingFace Hub Privacy Control**: Public/private repository selection
  - `hf_hub_private` parameter
  - Configurable during push_to_hub

### Changed
- Training logic moved to modular `training_runner.py`
- Enhanced error handling and logging
- GPU memory tracking during training
- Configuration system centralized in `config.py`

### Added - API Endpoints
- `POST /validate` - Validate configuration
- `GET /jobs/history` - Paginated job history
- `GET /jobs/{job_id}/metrics` - Detailed training metrics
- `GET /stats` - Database and system statistics
- `WebSocket /ws/{job_id}` - Real-time training updates
- `GET /queue/status` - Queue statistics
- `POST /jobs/{job_id}/stop` - Cancel/stop jobs

### Added - Files
- `src/job_manager.py` - Job persistence layer
- `src/websocket_manager.py` - WebSocket management
- `src/preflight_checks.py` - Configuration validation
- `src/training_runner.py` - Enhanced training logic
- `src/job_queue.py` - Job queue management
- `data/jobs.db` - SQLite database (runtime)

### Documentation
- Comprehensive v1.1 feature documentation in CLAUDE.md
- Example scripts for new features
- WebSocket integration guide

## [1.0.0] - 2024-11-01 "Magical Beginnings"

### Added
- **Initial Release**: ORPO training system
- **Dataset Pipeline**: Modular loader and formatter system
  - `DatasetLoader` abstraction (HuggingFace, Local, Upload)
  - `DatasetFormatter` abstraction (ChatML, Llama3, Mistral, Tokenizer)
  - `DatasetPipeline` orchestration
- **Web UI**: Wizard-themed interface
  - Dataset configuration and preview
  - Training job submission
  - Status polling and progress tracking
- **Training Features**:
  - ORPO (Odds Ratio Preference Optimization)
  - LoRA adapter support
  - 4-bit quantization for memory efficiency
  - Flash Attention support (Ampere GPUs and newer)
- **Model Support**:
  - HuggingFace Hub model loading
  - Automatic tokenizer chat template detection
  - Multi-format support (ChatML, Llama3, Mistral, etc.)
- **Dataset Sources**:
  - HuggingFace Hub datasets
  - Local JSON/JSONL files
  - Direct file upload
- **API Endpoints**:
  - `POST /train` - Submit training job
  - `GET /status/{job_id}` - Job status
  - `POST /dataset/preview` - Raw dataset preview
  - `POST /dataset/preview-formatted` - Formatted preview
  - `POST /dataset/upload-file` - File upload
- **Configuration**:
  - Environment variable support (.env)
  - Flexible training parameters
  - GPU selection and optimization settings

### Core Components
- `merlina.py` - FastAPI application
- `dataset_handlers/` - Dataset pipeline
  - `base.py` - Abstract interfaces
  - `loaders.py` - Dataset loaders
  - `formatters.py` - Format strategies
  - `validators.py` - Validation logic
- `frontend/` - Web interface (HTML/CSS/JS)
- `config.py` - Configuration management

### Documentation
- `CLAUDE.md` - Comprehensive development guide
- `README.md` - User documentation
- `.env.example` - Configuration template
- Example training scripts

---

## Version Categories

- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

## Semantic Versioning Guide

- **MAJOR (X.0.0)**: Breaking changes, incompatible API changes
- **MINOR (1.X.0)**: New features, backwards-compatible additions
- **PATCH (1.0.X)**: Bug fixes, backwards-compatible fixes

## Links

- [Keep a Changelog](https://keepachangelog.com/)
- [Semantic Versioning](https://semver.org/)
- [Merlina Repository](https://github.com/Schneewolf-Labs/Merlina)
