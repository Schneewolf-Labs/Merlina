"""Project Artemis — multimodal training entry point.

Parallel narrow path to `run_training_sync` for Artemis VLM training:
the Qwen3-VL ViT + learned projector + A-series text decoder graft built
in the standalone `artemis_vlm` package
(https://github.com/Schneewolf-Labs/Artemis). Two modes:

  - **vlm_stage1** — projector-only alignment on image-caption corpora
    (vision tower + decoder frozen). The big upstream win is that
    grimoire's optimizer is `requires_grad`-filtered, so we just call
    `model.set_training_stage("stage1")` and the optimizer Just Works.
  - **vlm_stage2** — multimodal instruction FFT (everything trainable;
    paged 8-bit AdamW + grad-checkpointing to fit a 12B decoder + 2B ViT
    on a 128GB unified memory node).

The flow mirrors the post-tokenize portion of `run_training_sync` but
swaps the text-mode tokenization/collator pair for ArtemisVLMProcessor
+ ArtemisDataCollator and uses `artemis_loss_fn` (which delegates to the
model's internal CE — labels are already correctly masked by the
collator). Post-training LoRA-merge / GGUF / HF upload paths from the
text runner are intentionally NOT wired here: Stage-1 is projector-only
full-FT (no LoRA to merge), and the resulting checkpoint is a directory
that the same upload machinery in the API layer can ship if requested.
"""
import os
import gc
import json
import logging
from typing import Any, Optional, Tuple

import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

from grimoire import GrimoireTrainer

from artemis_vlm import (
    ArtemisVLMForConditionalGeneration,
    ArtemisVLMProcessor,
    ArtemisDataCollator,
    artemis_loss_fn,
)
from src.training_runner import (
    WebSocketCallback,
    send_websocket_update,
    _cleanup_training_resources,
)
from src.job_manager import JobManager
from src.websocket_manager import websocket_manager
from src.utils import build_grimoire_config, get_num_gpus
from src.model_card import generate_wandb_run_name
from grimoire import TrainingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset adapter
# ---------------------------------------------------------------------------

class _ArtemisImageCaptionAdapter:
    """Lazy adapter: image-caption HF row -> {images, messages} for the collator.

    Built to be the thing the DataLoader iterates. Images stay lazy (HF
    `datasets` decodes PIL on access, so BLIP3o-scale corpora don't have
    to fit in RAM); only the rows the loader actually pulls get decoded.
    """

    def __init__(
        self,
        ds: Dataset,
        image_column: str,
        caption_column: str,
        instruction: str,
    ) -> None:
        self.ds = ds
        self.image_column = image_column
        self.caption_column = caption_column
        self.instruction = instruction

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        return {
            "images": [row[self.image_column]],
            "messages": [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.instruction},
                ]},
                {"role": "assistant", "content": row[self.caption_column]},
            ],
        }


def _load_image_caption_dataset(config: Any) -> Tuple[Dataset, Optional[Dataset]]:
    """Load the image-caption corpus from `config.dataset`.

    Returns (train, eval-or-None). Uses HF Hub by default; honors the
    same `dataset.repo_id` / `dataset.split` config layout as text modes.
    Train/eval split derived from `test_size` like the text path.

    For huge sharded webdataset corpora (e.g. BLIP3o-Pretrain-Long-Caption
    with 2891 shards), non-streaming `load_dataset()` downloads every shard
    before any `.select()` can take effect — set `config.streaming=True`
    with a bounded `dataset.max_samples` to pull only the shards needed
    for the first N samples and materialize them to an Arrow-backed
    Dataset (image bytes stored once, PIL decode stays lazy at train time).
    """
    dataset_cfg = config.dataset
    repo_id = getattr(dataset_cfg.source, "repo_id", None) or dataset_cfg.source.path
    split = getattr(dataset_cfg.source, "split", None) or "train"
    token = config.hf_token or os.environ.get("HF_TOKEN")
    max_samples = getattr(dataset_cfg, "max_samples", None)
    streaming = bool(getattr(config, "streaming", False))
    image_column = config.image_column or "image"
    caption_column = config.caption_column or "caption"

    logger.info(
        f"Loading image-caption dataset: {repo_id} split={split} "
        f"streaming={streaming} max_samples={max_samples}"
    )

    if streaming and max_samples:
        # Stream the corpus, materialize first N samples to Arrow.
        # Only the shards that hold those N samples get downloaded.
        # JPEG bytes are stored once via the Image() feature so PIL
        # decode stays lazy at training time — RAM footprint stays at
        # ~jpeg_bytes × N (a few GB for 50k).
        import io
        from datasets import Features, Image as ImageFeature, Value
        stream = load_dataset(repo_id, split=split, streaming=True, token=token)
        stream = stream.take(max_samples)

        def _gen():
            n = 0
            for row in stream:
                pil = row[image_column]
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=92)
                yield {
                    image_column: {"bytes": buf.getvalue(), "path": None},
                    caption_column: row[caption_column],
                }
                n += 1
                if n % 1000 == 0:
                    logger.info(f"  streamed {n}/{max_samples} samples")

        features = Features({
            image_column: ImageFeature(),
            caption_column: Value("string"),
        })
        raw = Dataset.from_generator(_gen, features=features)
        logger.info(f"  streaming materialized {len(raw)} samples")
    else:
        raw = load_dataset(repo_id, split=split, token=token)
        if max_samples and max_samples < len(raw):
            raw = raw.select(range(max_samples))
            logger.info(f"  capped to {max_samples} samples")

    # Train/eval split
    test_size = getattr(dataset_cfg, "test_size", 0.01) or 0.01
    if len(raw) > 100 and test_size > 0:
        split_out = raw.train_test_split(test_size=test_size, seed=config.seed)
        return split_out["train"], split_out["test"]
    return raw, None


# ---------------------------------------------------------------------------
# Stage-2: unified ArtemisMix corpus
# ---------------------------------------------------------------------------

class _ArtemisStage2Adapter:
    """Lazy adapter: ArtemisMix unified row -> {images, messages} for the collator.

    ArtemisMix rows carry `messages` (JSON-serialized chat with content lists)
    and a nullable `image` (text-only L4 rows have image=None). Multimodal rows
    embed `{"type":"image"}` placeholders in the user turn; text rows are plain
    strings. The collator already tolerates image-less rows (no pixel_values).
    """

    def __init__(self, ds: Dataset, image_column: str = "image",
                 messages_column: str = "messages") -> None:
        self.ds = ds
        self.image_column = image_column
        self.messages_column = messages_column

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[idx]
        msgs = row[self.messages_column]
        if isinstance(msgs, str):            # ArtemisMix stores messages as JSON
            msgs = json.loads(msgs)
        img = row.get(self.image_column)
        return {
            "messages": msgs,
            "images": [img] if img is not None else [],
        }


def _load_stage2_dataset(config: Any) -> Tuple[Dataset, Optional[Dataset]]:
    """Load the ArtemisMix unified corpus for vlm_stage2.

    Accepts either a local `save_to_disk` directory or an HF Hub repo id
    (e.g. `schneewolflabs/ArtemisMix-v1`). Rows: `messages` (JSON str),
    nullable `image`, plus `layer`/`source`/`reasoning` (ignored here).
    """
    dataset_cfg = config.dataset
    repo_id = getattr(dataset_cfg.source, "repo_id", None) or dataset_cfg.source.path
    split = getattr(dataset_cfg.source, "split", None) or "train"
    token = config.hf_token or os.environ.get("HF_TOKEN")
    max_samples = getattr(dataset_cfg, "max_samples", None)

    logger.info(f"Loading Stage-2 (ArtemisMix) dataset: {repo_id} split={split}")

    if os.path.isdir(repo_id) and os.path.exists(os.path.join(repo_id, "dataset_info.json")):
        from datasets import load_from_disk
        raw = load_from_disk(repo_id)
        if hasattr(raw, "keys"):             # DatasetDict -> pick split
            raw = raw[split]
    else:
        raw = load_dataset(repo_id, split=split, token=token)

    # Drop pathologically long rows. A few % of L4 (competitive-programming
    # code dumps, long GLM reasoning) reach 50k-90k tokens and would OOM a
    # full-FFT batch. Cheap char proxy (~4 chars/token) avoids tokenizing all
    # rows just to measure. Tune via config.max_seq_length (default 4096).
    max_seq_len = getattr(config, "max_seq_length", None) or 4096
    max_chars = max_seq_len * 4
    before = len(raw)
    raw = raw.filter(lambda r: len(r["messages"]) <= max_chars, num_proc=4)
    logger.info(
        f"  length filter (<= ~{max_seq_len} tok / {max_chars} chars): "
        f"{before} -> {len(raw)} rows ({before - len(raw)} dropped)"
    )

    if max_samples and max_samples < len(raw):
        raw = raw.shuffle(seed=config.seed).select(range(max_samples))
        logger.info(f"  capped to {max_samples} samples (shuffled across layers)")

    test_size = getattr(dataset_cfg, "test_size", 0.01) or 0.01
    if len(raw) > 100 and test_size > 0:
        split_out = raw.train_test_split(test_size=test_size, seed=config.seed)
        return split_out["train"], split_out["test"]
    return raw, None


# ---------------------------------------------------------------------------
# Model + processor assembly
# ---------------------------------------------------------------------------

def _load_artemis_model_and_processor(
    config: Any,
    torch_dtype: torch.dtype,
) -> Tuple[ArtemisVLMForConditionalGeneration, ArtemisVLMProcessor, Any]:
    """Build Artemis VLM (text + vision + projector) and matching processor.

    `config.base_model` is the A-series text decoder; `config.vision_model_id`
    is the source of the pretrained Qwen3-VL ViT (the decoder of that model
    is dropped). The projector is fresh, the vision tower is the pretrained
    ViT, the language model is the A-series decoder unchanged.
    """
    from transformers import Qwen3VLForConditionalGeneration

    text_path = config.base_model
    vision_id = config.vision_model_id or "Qwen/Qwen3-VL-2B-Instruct"
    image_token_id = config.image_token_id or 22

    logger.info(f"Assembling Artemis: text={text_path} | vision={vision_id}")
    qv = Qwen3VLForConditionalGeneration.from_pretrained(vision_id, dtype=torch_dtype)
    vision = qv.model.visual
    # Drop the Qwen3VL decoder we don't need (~2B params freed)
    qv.model.language_model = None
    del qv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = ArtemisVLMForConditionalGeneration.from_a2_and_vision(
        text_path,
        vision_model=vision,
        image_token_id=image_token_id,
        torch_dtype=torch_dtype,
    )

    tokenizer = AutoTokenizer.from_pretrained(text_path, trust_remote_code=True)
    processor = ArtemisVLMProcessor(
        tokenizer=tokenizer,
        vision_config=vision.config,
        min_pixels=(config.min_pixels or 32 * 32),
        max_pixels=(config.max_pixels or 512 * 512),
    )
    return model, processor, tokenizer


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_vlm_training_sync(
    job_id: str,
    config: Any,
    job_manager: JobManager,
    uploaded_datasets: dict,
    event_loop=None,
) -> None:
    """Run a vlm_stage1 / vlm_stage2 Artemis training job.

    Contract matches `run_training_sync` (job/WebSocket lifecycle, GPU
    cleanup in all exit paths) but the body is a narrow multimodal path:
    no LoRA, no chat-template grafting (the A-series decoder already has
    the Qwen3 template baked in from A1), no preference-mode dataset
    rendering.
    """
    model = None
    trainer = None

    try:
        job_manager.update_job(job_id, status="initializing", progress=0.0)
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id, status="initializing", progress=0.0,
            ),
            event_loop,
        )

        # GPU pinning (mirrors text path)
        if config.gpu_ids is not None:
            from src.gpu_utils import get_gpu_manager
            get_gpu_manager().set_visible_devices(config.gpu_ids)
            logger.info(f"Using GPUs: {config.gpu_ids}")

        # Wandb (let grimoire's accelerator drive init via init_trackers)
        wandb_run_name = None
        wandb_project = None
        if config.use_wandb:
            if config.wandb_key:
                wandb.login(key=config.wandb_key)
            wandb_run_name = config.wandb_run_name or generate_wandb_run_name(config)
            wandb_project = config.wandb_project or "merlina-training"
            logger.info(f"W&B Project: {wandb_project}, Run: {wandb_run_name}")

        if config.hf_token:
            os.environ["HF_TOKEN"] = config.hf_token

        # dtype: bf16 on Ampere+, fp16 fallback
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
            mixed_precision = "bf16"
        else:
            torch_dtype = torch.float16
            mixed_precision = "fp16"

        # ---- Build model + processor ----
        job_manager.update_job(job_id, status="loading_model", progress=0.05)
        model, processor, tokenizer = _load_artemis_model_and_processor(config, torch_dtype)

        # Stage gating — grimoire's requires_grad-filtered optimizer picks up
        # only what set_training_stage marks trainable.
        stage = (config.stage or "stage1").lower()
        unfreeze_top_n = config.unfreeze_vision_top_n or 0
        trainable, total = model.set_training_stage(stage, unfreeze_vision_top_n=unfreeze_top_n)
        logger.info(
            f"Stage {stage}: trainable={trainable/1e6:.1f}M / total={total/1e9:.2f}B "
            f"({100*trainable/total:.3f}%)"
        )

        if config.gradient_checkpointing and stage == "stage2":
            # Stage-1 is projector-only — checkpointing the frozen decoder
            # buys nothing and breaks no-grad paths in some configurations.
            model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled on language_model (stage2)")

        # ---- Build dataset ----
        job_manager.update_job(job_id, status="loading_dataset", progress=0.1)
        if stage == "stage2":
            # Unified ArtemisMix corpus: messages (JSON) + nullable image.
            train_raw, eval_raw = _load_stage2_dataset(config)
            image_column = config.image_column or "image"
            messages_column = getattr(config, "messages_column", None) or "messages"
            train_dataset = _ArtemisStage2Adapter(train_raw, image_column, messages_column)
            eval_dataset = (
                _ArtemisStage2Adapter(eval_raw, image_column, messages_column)
                if eval_raw is not None else None
            )
        else:
            # Stage-1: image-caption corpus with a fixed instruction.
            train_raw, eval_raw = _load_image_caption_dataset(config)
            instruction = config.instruction or "Describe this image."
            image_column = config.image_column or "image"
            caption_column = config.caption_column or "caption"
            train_dataset = _ArtemisImageCaptionAdapter(
                train_raw, image_column, caption_column, instruction
            )
            eval_dataset = (
                _ArtemisImageCaptionAdapter(eval_raw, image_column, caption_column, instruction)
                if eval_raw is not None else None
            )
        logger.info(
            f"Dataset: train={len(train_dataset)} "
            f"eval={len(eval_dataset) if eval_dataset else 0}"
        )

        collator = ArtemisDataCollator(processor)

        # ---- eval_steps: <1 means ratio, >=1 absolute (same as text path) ----
        output_dir = f"./results/{job_id}"
        eval_steps = None
        if config.eval_steps:
            if config.eval_steps < 1:
                import math
                num_gpus = get_num_gpus()
                effective_batch = config.batch_size * config.gradient_accumulation_steps * num_gpus
                steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
                total_steps = steps_per_epoch * config.num_epochs
                eval_steps = max(1, int(total_steps * config.eval_steps))
            else:
                eval_steps = int(config.eval_steps)

        # ---- Grimoire config ----
        grimoire_config = build_grimoire_config(
            TrainingConfig,
            output_dir=output_dir,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=config.max_grad_norm,
            mixed_precision=mixed_precision,
            gradient_checkpointing=False,  # handled manually above (stage2 only)
            optimizer=config.optimizer_type,
            lr_scheduler=config.lr_scheduler_type,
            logging_steps=config.logging_steps,
            eval_steps=eval_steps,
            eval_on_start=config.eval_on_start,
            eval_batch_size=getattr(config, "eval_batch_size", None),
            save_steps=eval_steps,
            save_total_limit=2,
            seed=config.seed,
            run_name=wandb_run_name if config.use_wandb else config.output_name,
            log_with="wandb" if config.use_wandb else None,
            project_name=wandb_project,
            wandb_tags=config.wandb_tags or [],
            wandb_notes=config.wandb_notes,
            dataloader_pin_memory=False,  # PIL images don't pin meaningfully
        )

        # ---- Trainer ----
        trainer = GrimoireTrainer(
            model=model,
            tokenizer=tokenizer,
            config=grimoire_config,
            loss_fn=artemis_loss_fn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,           # the prerequisite from grimoire 1.1.x
            peft_config=None,                 # Stage-1/2 are full-FT (no LoRA)
            callbacks=[WebSocketCallback(job_id, job_manager, event_loop)],
        )

        # Capture W&B URL after accelerator/grimoire init
        if config.use_wandb and wandb.run is not None:
            wandb_url = wandb.run.get_url()
            logger.info(f"W&B run URL: {wandb_url}")
            job_manager.update_job(job_id, wandb_url=wandb_url)

        logger.info(f"Starting Artemis {stage} training")
        trainer.train()
        was_stopped = trainer.stopped_early

        # ---- Save ----
        job_manager.update_job(
            job_id,
            status="saving_stopped" if was_stopped else "saving",
            progress=0.9,
        )
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id, status="saving", progress=0.9,
            ),
            event_loop,
        )

        final_output_dir = f"./models/{config.output_name}"
        trainer.save_model(final_output_dir)

        # Save processor + image processor + tokenizer alongside the model
        # so the checkpoint is self-contained and reloadable for inference.
        try:
            processor.image_processor.save_pretrained(final_output_dir)
            tokenizer.save_pretrained(final_output_dir)
            logger.info(f"Processor + tokenizer saved to {final_output_dir}")
        except Exception as e:
            logger.warning(f"Processor save failed (non-fatal): {e}")

        final_step = trainer.global_step
        final_max_steps = trainer.max_steps

        # Free VRAM before any post-training upload pipeline picks up
        logger.info("🧹 Cleaning up Artemis training resources...")
        del trainer, model
        model = None
        trainer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ VRAM freed")

        # Stage-1/2 don't use LoRA, so the saved directory is already the
        # final merged model. If push_to_hub is set, the existing
        # post-training upload threads in the text runner can be reused
        # in a follow-up — keep this path narrow for now.
        if config.push_to_hub:
            logger.info(
                "push_to_hub set for VLM run — upload from this path is a "
                "follow-up; the saved checkpoint at "
                f"{final_output_dir} is ready to upload manually."
            )

        # Mark done
        job_manager.update_job(
            job_id,
            status="stopped" if was_stopped else "completed",
            progress=1.0,
            current_step=final_step,
            total_steps=final_max_steps,
        )
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id,
                status="stopped" if was_stopped else "completed",
                progress=1.0,
                current_step=final_step,
                total_steps=final_max_steps,
            ),
            event_loop,
        )

    except Exception as e:
        logger.exception(f"Artemis VLM training failed: {e}")
        job_manager.update_job(job_id, status="failed", error=str(e))
        send_websocket_update(
            websocket_manager.send_status_update(
                job_id=job_id, status="failed", message=str(e),
            ),
            event_loop,
        )
        raise
    finally:
        _cleanup_training_resources(model, trainer)
