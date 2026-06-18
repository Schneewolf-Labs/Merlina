// Form Config Module — single source of truth for building a TrainingConfig
// payload from the form. Used by BOTH "Cast Spell" (training) and
// "Save Configuration" so the two never drift.
//
// Anything that lands in the backend Pydantic TrainingConfig must be read
// here. If you add a field to TrainingConfig, add it here too — and add a
// row to tests/test_form_config_parity.py so the regression is caught.

/**
 * Read a numeric input, returning the fallback if empty or NaN.
 */
function num(id, fallback, parser = parseFloat) {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const raw = el.value;
    if (raw === '' || raw === null || raw === undefined) return fallback;
    const parsed = parser(raw);
    return Number.isNaN(parsed) ? fallback : parsed;
}

/**
 * Read a numeric input that may be empty (returns null when unset).
 */
function numOrNull(id, parser = parseFloat) {
    const el = document.getElementById(id);
    if (!el || el.value === '' || el.value === null || el.value === undefined) {
        return null;
    }
    const parsed = parser(el.value);
    return Number.isNaN(parsed) ? null : parsed;
}

/**
 * Read a checkbox value, returning fallback when the element is missing.
 */
function bool(id, fallback) {
    const el = document.getElementById(id);
    return el ? !!el.checked : fallback;
}

/**
 * Read a text input, returning the fallback when missing/empty.
 */
function str(id, fallback = '') {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const v = el.value;
    return (v === undefined || v === null) ? fallback : v;
}

/**
 * Read a comma-separated input as a list (trimmed, no empties).
 */
function strList(id) {
    const raw = str(id, '');
    return raw.split(',').map(s => s.trim()).filter(s => s.length > 0);
}

/**
 * Read target-modules from either the hidden field or the manual fallback.
 * Returns a normalized array of module names.
 */
function readTargetModules() {
    let v = str('target-modules', '');
    if (!v) v = str('target-modules-manual', '');
    return v.split(',').map(s => s.trim()).filter(s => s.length > 0);
}

/**
 * Read the {sourceCol: standardName} mapping from a dataset card.
 */
function readCardColumnMapping(card) {
    const mapping = {};
    const pairs = [
        ['.ds-map-prompt', 'prompt'],
        ['.ds-map-chosen', 'chosen'],
        ['.ds-map-rejected', 'rejected'],
        ['.ds-map-system', 'system'],
        ['.ds-map-reasoning', 'reasoning'],
    ];
    for (const [selector, target] of pairs) {
        const val = card.querySelector(selector)?.value;
        if (val) mapping[val] = target;
    }
    return mapping;
}

/**
 * Read a dataset source object from a single card. We read the raw DOM so
 * unfilled fields still round-trip — saving must NOT enforce validation that
 * the user can address later.
 */
function readCardSource(card) {
    const t = card.querySelector('.ds-source-type')?.value || 'huggingface';
    const src = { source_type: t };
    if (t === 'huggingface') {
        src.repo_id = card.querySelector('.ds-repo')?.value || '';
        src.split = card.querySelector('.ds-split')?.value || 'train';
        const configName = card.querySelector('.ds-config')?.value?.trim();
        if (configName) src.config_name = configName;
    } else if (t === 'local_file') {
        src.file_path = card.querySelector('.ds-local-path')?.value || '';
        src.file_format = card.querySelector('.ds-local-format')?.value || '';
    } else if (t === 'upload') {
        src.dataset_id = card.dataset.uploadId || '';
    }
    const mapping = readCardColumnMapping(card);
    if (Object.keys(mapping).length > 0) src.column_mapping = mapping;
    return src;
}

/**
 * Build the dataset block of TrainingConfig from the DOM. Mirrors the
 * structure DatasetManager.getDatasetConfig() builds for /train so saving
 * a config keeps every dataset knob (max_samples, deduplicate, etc.).
 *
 * @param {object} options
 * @param {boolean} options.includeTrainingMode — include training_mode in
 *     the dataset block. /train sends it; preset save can include it too.
 */
export function buildDatasetConfig({ includeTrainingMode = true } = {}) {
    const cards = Array.from(document.querySelectorAll('#datasets-list .dataset-card'));

    let source;
    let additionalSources = [];
    if (cards.length > 0) {
        source = readCardSource(cards[0]);
        additionalSources = cards.slice(1).map(readCardSource);
    } else {
        source = { source_type: 'huggingface', repo_id: '', split: 'train' };
    }

    const formatType = str('dataset-format-type', 'tokenizer');
    const format = { format_type: formatType };
    if (formatType === 'qwen3') {
        format.enable_thinking = bool('enable-thinking', true);
    } else if (formatType === 'custom') {
        format.custom_templates = {
            prompt_template: str('custom-prompt-template', ''),
            chosen_template: str('custom-chosen-template', ''),
            rejected_template: str('custom-rejected-template', ''),
        };
    }
    // Auto-detect-thinking: surfaces whenever the chat template supports the
    // enable_thinking kwarg (qwen3 + tokenizer). Independent of static
    // enable_thinking — when on, formatter picks per row from reasoning signal.
    if (formatType === 'qwen3' || formatType === 'tokenizer') {
        format.auto_detect_thinking = bool('auto-detect-thinking', true);
    }

    const config = {
        source,
        additional_sources: additionalSources,
        format,
        test_size: num('test-size', 0.01),
        convert_messages_format: bool('convert-messages-checkbox', true),
        deduplicate: bool('deduplicate', false),
        dedupe_strategy: str('dedupe-strategy', 'prompt_chosen'),
    };

    const maxSamples = numOrNull('max-samples', parseInt);
    if (maxSamples !== null) config.max_samples = maxSamples;

    const baseModel = str('base-model', '').trim();
    if (baseModel) config.model_name = baseModel;

    const systemPrompt = str('system-prompt-override', '').trim();
    if (systemPrompt) {
        config.system_prompt = systemPrompt;
        const modeRadio = document.querySelector('input[name="system-prompt-mode"]:checked');
        config.system_prompt_mode = modeRadio ? modeRadio.value : 'fill_empty';
    } else {
        // Still record the chosen mode so loading a config restores the radio
        const modeRadio = document.querySelector('input[name="system-prompt-mode"]:checked');
        if (modeRadio) config.system_prompt_mode = modeRadio.value;
    }

    if (includeTrainingMode) {
        config.training_mode = str('training-mode', 'orpo');
    }

    return config;
}

/**
 * Build the complete TrainingConfig payload from the form.
 *
 * This is the SINGLE SOURCE OF TRUTH for collecting training settings from
 * the UI. Both /train submission and /configs/save call this so saved
 * configs round-trip cleanly and the two paths can never drift.
 *
 * @param {object} options
 * @param {object} options.gpuManager — GPUManager instance (optional). When
 *     provided, its selected GPU IDs are included.
 * @param {boolean} options.includeSecrets — when true, hf_token and wandb_key
 *     are read from the form and included. Defaults to true. Saved presets
 *     should NOT include secrets unless explicitly authorized.
 */
export function buildTrainingConfig({ gpuManager = null, includeSecrets = true } = {}) {
    const useLora = bool('use-lora', true);

    const config = {
        // Model
        base_model: str('base-model', ''),
        output_name: str('output-name', ''),
        model_type: str('model-type', 'auto'),

        // Dataset
        dataset: buildDatasetConfig({ includeTrainingMode: true }),

        // LoRA
        use_lora: useLora,
        lora_r: num('lora-r', 64, parseInt),
        lora_alpha: num('lora-alpha', 32, parseInt),
        lora_dropout: num('lora-dropout', 0.05),
        target_modules: readTargetModules(),
        modules_to_save: strList('modules-to-save'),
        lora_task_type: str('lora-task-type', 'CAUSAL_LM'),

        // Training
        training_mode: str('training-mode', 'orpo'),
        learning_rate: num('learning-rate', 0.000005),
        num_epochs: num('epochs', 2, parseInt),
        batch_size: num('batch-size', 1, parseInt),
        gradient_accumulation_steps: num('grad-accum', 16, parseInt),
        max_length: num('max-length', 2048, parseInt),
        max_prompt_length: num('max-prompt-length', 1024, parseInt),
        beta: num('beta', 0.1),
        gamma: num('gamma', 0.5),
        label_smoothing: num('label-smoothing', 0.0),
        seed: num('seed', 42, parseInt),
        max_grad_norm: num('max-grad-norm', 0.3),
        warmup_ratio: num('warmup-ratio', 0.05),
        eval_steps: num('eval-steps', 0.2),
        shuffle_dataset: bool('shuffle-dataset', true),
        weight_decay: num('weight-decay', 0.01),
        lr_scheduler_type: str('lr-scheduler-type', 'cosine'),
        gradient_checkpointing: bool('gradient-checkpointing', false),
        logging_steps: num('logging-steps', 1, parseInt),

        // Optimizer
        optimizer_type: str('optimizer-type', 'paged_adamw_8bit'),
        adam_beta1: num('adam-beta1', 0.9),
        adam_beta2: num('adam-beta2', 0.999),
        adam_epsilon: num('adam-epsilon', 1e-8),
        adafactor_relative_step: bool('adafactor-relative-step', false),
        adafactor_scale_parameter: bool('adafactor-scale-parameter', false),
        adafactor_warmup_init: bool('adafactor-warmup-init', false),
        adafactor_decay_rate: num('adafactor-decay-rate', -0.8),
        adafactor_beta1: numOrNull('adafactor-beta1'),
        adafactor_clip_threshold: num('adafactor-clip-threshold', 1.0),

        // Attention
        attn_implementation: str('attn-implementation', 'auto'),

        // Grimoire kernel/regularization features
        use_liger: bool('use-liger', false),
        torch_compile: bool('torch-compile', false),
        neftune_alpha: numOrNull('neftune-alpha'),
        eval_on_start: bool('eval-on-start', false),

        // GPU
        multi_gpu_strategy: str('multi-gpu-strategy', 'auto'),

        // Options
        use_4bit: bool('use-4bit', true),
        use_wandb: bool('use-wandb', false),
        push_to_hub: bool('push-hub', false),
        merge_lora_before_upload: bool('merge-lora-before-upload', true),
        hf_hub_private: bool('hf-hub-private', true),

        // GGUF export — Cast Spell does not trigger GGUF (it's done from
        // the dedicated Export tab on a finished model). We still emit
        // every GGUF field so saved presets round-trip through
        // TrainingConfig with defaults intact.
        export_gguf: false,
        gguf_quant_types: ['Q4_K_M'],
        keep_gguf_fp16: false,

        // W&B settings (always serialize so a saved preset round-trips)
        wandb_project: str('wandb-project', '') || null,
        wandb_run_name: str('wandb-run-name', '') || null,
        wandb_tags: (() => {
            const raw = str('wandb-tags', '');
            const tags = raw.split(',').map(t => t.trim()).filter(t => t);
            return tags.length > 0 ? tags : null;
        })(),
        wandb_notes: str('wandb-notes', '') || null,

        // Sharing — when true, the training config is embedded in the
        // uploaded model's README so others can reproduce the training run.
        share_config: bool('share-config', true),

        // Publish a scannable QR image (merlina_config.png) that also carries
        // the full config in its PNG metadata. Independent of share_config.
        share_config_image: bool('share-config-image', false),

        // Diffusion / image-LoRA fields. All Optional in the backend
        // Pydantic schema so text/VLM jobs ignore them entirely. Read
        // unconditionally; saved presets still capture them so a
        // diffusion-trained preset round-trips.
        model_name:          null,  // diffusion runs reuse base_model; runner falls back when null
        image_resolution:    numOrNull('diffusion-image-resolution', parseInt),
        lora_rank:           numOrNull('diffusion-lora-rank', parseInt),
        lora_target_modules: (strList('diffusion-lora-target-modules').length
                              ? strList('diffusion-lora-target-modules') : null),
        lora_use_dora:       bool('diffusion-lora-use-dora', false),
        mid_training_samples: bool('diffusion-mid-training-samples', true),
        dataset_jsonl_path:  str('diffusion-dataset-jsonl', '') || null,
        dataset_name:        str('diffusion-dataset-name', '') || null,
        dataset_split:       null,  // wired when an explicit-split UI lands; null = HF default
        sample_prompts:      null,  // power-user override for post-training preview prompts
        sample_num_steps:    null,  // power-user override for preview denoise steps

        // Artemis VLM fields. All Optional; UI inputs don't exist yet
        // (VLM mode currently reuses the text dataset section, with
        // these knobs driven by API clients). Null placeholders here
        // satisfy the form_config↔TrainingConfig parity contract so
        // saved presets that DID set them via the API round-trip.
        vision_model_id:        null,
        stage:                  null,
        unfreeze_vision_top_n:  null,
        image_token_id:         null,
        min_pixels:             null,
        max_pixels:             null,
        image_column:           null,
        caption_column:         null,
        instruction:            null,
        streaming:              null,

        // Secrets — always present in the dict so the contract is
        // deterministic. The values come from the form (or null) when
        // includeSecrets=true; saved presets force them to null and the
        // ConfigManager strips them at write time as defense in depth.
        hf_token: null,
        wandb_key: null,
    };

    // GPU selection — only attach when a manager is supplied
    if (gpuManager && typeof gpuManager.getSelectedGPUs === 'function') {
        config.gpu_ids = gpuManager.getSelectedGPUs();
    }

    // Saved presets default to includeSecrets=false so tokens never reach
    // disk. /train always includes them so the backend can resolve them
    // (or fall back to .env).
    if (includeSecrets) {
        const hfToken = str('hf-token', '') || str('hf-token-preload', '');
        config.hf_token = hfToken || null;

        const wandbKey = str('wandb-key', '');
        config.wandb_key = wandbKey || null;
    }

    return config;
}

// Names of fields stripped from saved presets. Re-exported so tests can
// assert the contract between the UI and ConfigManager.
export const SECRET_FIELDS = Object.freeze(['hf_token', 'wandb_key']);
