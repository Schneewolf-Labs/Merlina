// Main Application Module - Initializes and coordinates all modules

import { JobManager } from './jobs.js';
import { DatasetManager } from './dataset.js';
import { ConfigManager } from './config.js';
import { GPUManager } from './gpu.js';
import { ModelManager } from './model.js';
import { Toast, FormUI, Tooltip, createSparkle } from './ui.js';
import { Validator, ValidationRules, debounce } from './validation.js';
import { ThemeManager } from './theme.js';
import { InferenceManager } from './inference.js';

/**
 * Toggle visibility of optimizer-specific settings based on selected optimizer.
 */
function updateOptimizerSettingsVisibility() {
    const optimizerType = document.getElementById('optimizer-type')?.value || '';
    const adamSettings = document.getElementById('adam-settings');
    const adafactorSettings = document.getElementById('adafactor-settings');
    const isAdafactor = optimizerType === 'adafactor';

    if (adamSettings) adamSettings.style.display = isAdafactor ? 'none' : '';
    if (adafactorSettings) adafactorSettings.style.display = isAdafactor ? 'block' : 'none';
}

// Expose globally so config.js can call it when restoring saved configs
window.updateOptimizerSettingsVisibility = updateOptimizerSettingsVisibility;

/**
 * Main Application Class
 */
class MerlinaApp {
    constructor() {
        // Initialize theme first to prevent flash of wrong theme
        this.themeManager = new ThemeManager();

        // Initialize managers
        this.jobManager = new JobManager();
        this.datasetManager = new DatasetManager();
        this.configManager = new ConfigManager();
        this.gpuManager = new GPUManager();
        this.modelManager = new ModelManager();
        this.inferenceManager = new InferenceManager();

        this.toast = new Toast();

        // Make managers available globally for debugging and onclick handlers
        window.jobManager = this.jobManager;
        window.datasetManager = this.datasetManager;
        window.configManager = this.configManager;
        window.gpuManager = this.gpuManager;
        window.modelManager = this.modelManager;
        window.themeManager = this.themeManager;
        window.inferenceManager = this.inferenceManager;

        // Initialize the app
        this.init();
    }

    /**
     * Initialize application
     */
    init() {
        console.log('🧙 Initializing Merlina...');

        // Load existing jobs
        this.jobManager.loadJobs();

        // Setup form submission
        this.setupFormSubmission();

        // Setup LoRA toggle
        this.setupLoRAToggle();

        // Setup layer detection
        this.setupLayerDetection();

        // Setup training mode toggle
        this.setupTrainingModeToggle();

        // Setup preset button
        this.setupPresetButton();

        // Setup section navigation
        this.setupSectionNav();

        // Setup name generator
        this.setupNameGenerator();

        // Setup advanced settings toggle
        this.setupAdvancedToggle();

        // Setup conditional config visibility
        this.setupConditionalVisibility();

        // Setup input validation
        this.setupInputValidation();

        // Setup tooltips
        Tooltip.init();

        // Setup magical interactions
        this.setupMagicalInteractions();

        // Setup auto-save
        this.setupAutoSave();

        // Setup keyboard shortcuts
        this.setupKeyboardShortcuts();

        // Load version information
        this.loadVersion();

        // Load server-side secret availability to hint which tokens can be omitted
        this.loadEnvSecretStatus();

        console.log('✨ Merlina initialized successfully!');
    }

    /**
     * Fetch the server's .env secret status and update token inputs so
     * users know which tokens are already configured and can be omitted.
     */
    async loadEnvSecretStatus() {
        try {
            const { MerlinaAPI } = await import('./api.js');
            const status = await MerlinaAPI.getEnvSecrets();

            const applyHint = (inputId, label) => {
                const input = document.getElementById(inputId);
                if (!input) return;
                input.placeholder = `Using ${label} from server .env (leave blank) — or paste to override`;
                input.removeAttribute('required');
                const small = input.parentElement?.querySelector('small');
                if (small) {
                    small.innerHTML = `✓ ${label} is configured on the server via <code>.env</code>. ` +
                                      `Leave this field blank to use it, or paste a token to override.`;
                    small.style.color = '#2e7d32';
                }
            };

            if (status?.hf_token) {
                ['hf-token-preload', 'hf-token', 'inference-hf-token', 'upload-hf-token'].forEach(
                    id => applyHint(id, 'HF_TOKEN')
                );
            }
            if (status?.wandb_api_key) {
                applyHint('wandb-key', 'WANDB_API_KEY');
            }
        } catch (error) {
            console.warn('Failed to load env secret status:', error);
        }
    }

    /**
     * Setup form submission
     */
    setupFormSubmission() {
        const form = document.getElementById('training-form');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleTrainingSubmit();
        });
    }

    /**
     * Handle training form submission
     */
    async handleTrainingSubmit() {
        // Collect form data
        const config = this.collectTrainingConfig();

        // Validate configuration
        const errors = this.validateTrainingConfig(config);
        if (Object.keys(errors).length > 0) {
            this.displayValidationErrors(errors);
            return;
        }

        // Show VRAM estimate
        const vramEstimate = Validator.estimateVRAM(config);
        const proceed = confirm(
            `🔮 Ready to cast training spell?\n\n` +
            `Estimated VRAM usage:\n` +
            `  Base model: ${vramEstimate.base} GB\n` +
            `  Training overhead: ${vramEstimate.training} GB\n` +
            `  Total: ${vramEstimate.total} GB\n\n` +
            `Proceed with training?`
        );

        if (!proceed) return;

        // Disable form
        const form = document.getElementById('training-form');
        FormUI.setEnabled(form, false);

        try {
            // Submit job
            await this.jobManager.submitJob(config);

            // Clear only output name field
            const outputNameInput = document.getElementById('output-name');
            if (outputNameInput) {
                outputNameInput.value = '';
            }

            // Switch to Jobs tab
            this.showSection('jobs-section');

        } catch (error) {
            console.error('Failed to submit training:', error);
        } finally {
            FormUI.setEnabled(form, true);
        }
    }

    /**
     * Collect training configuration from form
     */
    collectTrainingConfig() {
        // Parse target modules - check hidden field first, then manual input
        let targetModulesStr = document.getElementById('target-modules')?.value || '';
        // Fallback to manual input if hidden field is empty
        if (!targetModulesStr) {
            targetModulesStr = document.getElementById('target-modules-manual')?.value || '';
        }
        const targetModules = targetModulesStr.split(',').map(s => s.trim()).filter(s => s.length > 0);

        // Parse modules to save
        const modulesToSaveStr = document.getElementById('modules-to-save')?.value || '';
        const modulesToSave = modulesToSaveStr.split(',').map(s => s.trim()).filter(s => s.length > 0);

        // Get dataset configuration
        let datasetConfig;
        try {
            datasetConfig = this.datasetManager.getDatasetConfig();
        } catch (error) {
            this.toast.error(`Dataset config error: ${error.message}`);
            throw error;
        }

        // Build configuration
        const config = {
            base_model: document.getElementById('base-model')?.value || '',
            output_name: document.getElementById('output-name')?.value || '',
            model_type: document.getElementById('model-type')?.value || 'auto',

            // Dataset
            dataset: datasetConfig,

            // LoRA
            use_lora: document.getElementById('use-lora')?.checked ?? true,
            lora_r: parseInt(document.getElementById('lora-r')?.value || 64),
            lora_alpha: parseInt(document.getElementById('lora-alpha')?.value || 32),
            lora_dropout: parseFloat(document.getElementById('lora-dropout')?.value || 0.05),
            target_modules: targetModules,
            modules_to_save: modulesToSave,

            // Training
            training_mode: document.getElementById('training-mode')?.value || 'orpo',
            learning_rate: parseFloat(document.getElementById('learning-rate')?.value || 0.000005),
            num_epochs: parseInt(document.getElementById('epochs')?.value || 2),
            batch_size: parseInt(document.getElementById('batch-size')?.value || 1),
            gradient_accumulation_steps: parseInt(document.getElementById('grad-accum')?.value || 16),
            max_length: parseInt(document.getElementById('max-length')?.value || 2048),
            max_prompt_length: parseInt(document.getElementById('max-prompt-length')?.value || 1024),
            beta: parseFloat(document.getElementById('beta')?.value || 0.1),
            gamma: parseFloat(document.getElementById('gamma')?.value || 0.5),
            label_smoothing: parseFloat(document.getElementById('label-smoothing')?.value || 0.0),
            seed: parseInt(document.getElementById('seed')?.value || 42),
            max_grad_norm: parseFloat(document.getElementById('max-grad-norm')?.value || 0.3),
            warmup_ratio: parseFloat(document.getElementById('warmup-ratio')?.value || 0.05),
            eval_steps: parseFloat(document.getElementById('eval-steps')?.value || 0.2),
            shuffle_dataset: document.getElementById('shuffle-dataset')?.checked ?? true,
            weight_decay: parseFloat(document.getElementById('weight-decay')?.value || 0.01),
            lr_scheduler_type: document.getElementById('lr-scheduler-type')?.value || 'cosine',
            gradient_checkpointing: document.getElementById('gradient-checkpointing')?.checked ?? false,
            logging_steps: parseInt(document.getElementById('logging-steps')?.value || 1),

            // Optimizer
            optimizer_type: document.getElementById('optimizer-type')?.value || 'paged_adamw_8bit',
            adam_beta1: parseFloat(document.getElementById('adam-beta1')?.value || 0.9),
            adam_beta2: parseFloat(document.getElementById('adam-beta2')?.value || 0.999),
            adam_epsilon: parseFloat(document.getElementById('adam-epsilon')?.value || 1e-8),
            adafactor_relative_step: document.getElementById('adafactor-relative-step')?.checked ?? false,
            adafactor_scale_parameter: document.getElementById('adafactor-scale-parameter')?.checked ?? false,
            adafactor_warmup_init: document.getElementById('adafactor-warmup-init')?.checked ?? false,
            adafactor_decay_rate: parseFloat(document.getElementById('adafactor-decay-rate')?.value || -0.8),
            adafactor_beta1: document.getElementById('adafactor-beta1')?.value ? parseFloat(document.getElementById('adafactor-beta1').value) : null,
            adafactor_clip_threshold: parseFloat(document.getElementById('adafactor-clip-threshold')?.value || 1.0),

            // Attention
            attn_implementation: document.getElementById('attn-implementation')?.value || 'auto',

            // Grimoire kernel/regularization features
            use_liger: document.getElementById('use-liger')?.checked ?? false,
            torch_compile: document.getElementById('torch-compile')?.checked ?? false,
            neftune_alpha: document.getElementById('neftune-alpha')?.value
                ? parseFloat(document.getElementById('neftune-alpha').value)
                : null,
            eval_on_start: document.getElementById('eval-on-start')?.checked ?? false,

            // GPU
            gpu_ids: this.gpuManager.getSelectedGPUs(),
            multi_gpu_strategy: document.getElementById('multi-gpu-strategy')?.value || 'auto',

            // Training backend (local vs HuggingFace Jobs)
            training_backend: document.getElementById('training-backend')?.value || 'local',
            hf_jobs_flavor: document.getElementById('hf-jobs-flavor')?.value || 'a10g-large',
            hf_jobs_image: document.getElementById('hf-jobs-image')?.value || 'ghcr.io/schneewolf-labs/merlina-hfjobs:latest',
            hf_jobs_timeout: document.getElementById('hf-jobs-timeout')?.value || '6h',

            // Options
            use_4bit: document.getElementById('use-4bit')?.checked ?? true,
            use_wandb: document.getElementById('use-wandb')?.checked ?? false,
            push_to_hub: document.getElementById('push-hub')?.checked ?? false,
            merge_lora_before_upload: document.getElementById('merge-lora-before-upload')?.checked ?? true,
            hf_hub_private: document.getElementById('hf-hub-private')?.checked ?? true,

            // API Keys
            wandb_key: document.getElementById('wandb-key')?.value || null,
            hf_token: document.getElementById('hf-token')?.value || document.getElementById('hf-token-preload')?.value || null,

            // W&B settings
            wandb_project: document.getElementById('wandb-project')?.value || null,
            wandb_run_name: document.getElementById('wandb-run-name')?.value || null,
            wandb_tags: document.getElementById('wandb-tags')?.value ?
                document.getElementById('wandb-tags').value.split(',').map(tag => tag.trim()).filter(tag => tag) :
                null,
            wandb_notes: document.getElementById('wandb-notes')?.value || null
        };

        return config;
    }

    /**
     * Validate training configuration
     */
    validateTrainingConfig(config) {
        const formData = {
            'base-model': config.base_model,
            'output-name': config.output_name,
            'learning-rate': config.learning_rate,
            'epochs': config.num_epochs,
            'batch-size': config.batch_size,
            'grad-accum': config.gradient_accumulation_steps,
            'max-length': config.max_length,
            'max-prompt-length': config.max_prompt_length,
            'beta': config.beta,
            'lora-r': config.lora_r,
            'lora-alpha': config.lora_alpha,
            'lora-dropout': config.lora_dropout,
            'test-size': config.dataset.test_size
        };

        return Validator.validateForm(formData);
    }

    /**
     * Display validation errors
     */
    displayValidationErrors(errors) {
        for (const [fieldId, fieldErrors] of Object.entries(errors)) {
            Validator.showFieldError(fieldId, fieldErrors);
        }

        // Show toast with first error
        const firstError = Object.values(errors)[0][0];
        this.toast.error(`Validation error: ${firstError}`);

        // Scroll to first error
        const firstField = document.getElementById(Object.keys(errors)[0]);
        if (firstField) {
            firstField.scrollIntoView({ behavior: 'smooth', block: 'center' });
            firstField.focus();
        }
    }

    /**
     * Setup LoRA toggle
     */
    setupLoRAToggle() {
        const useLora = document.getElementById('use-lora');
        const loraSettings = document.getElementById('lora-settings');
        const mergeLoraConfig = document.getElementById('merge-lora-config');

        if (!useLora) return;

        useLora.addEventListener('change', (e) => {
            if (loraSettings) {
                loraSettings.style.display = e.target.checked ? 'block' : 'none';
            }
            if (mergeLoraConfig) {
                mergeLoraConfig.style.display = e.target.checked ? 'block' : 'none';
            }
        });

        // Initialize
        if (useLora.checked) {
            if (loraSettings) loraSettings.style.display = 'block';
            if (mergeLoraConfig) mergeLoraConfig.style.display = 'block';
        }
    }

    /**
     * Setup layer detection for LoRA target modules
     */
    setupLayerDetection() {
        const detectBtn = document.getElementById('detect-layers-btn');
        const statusDiv = document.getElementById('layer-detection-status');
        const layersContainer = document.getElementById('detected-layers-container');
        const manualInput = document.getElementById('manual-layers-input');
        const targetModulesHidden = document.getElementById('target-modules');
        const targetModulesManual = document.getElementById('target-modules-manual');

        if (!detectBtn) return;

        // Store detected layers data
        this.detectedLayers = null;
        this.recommendedLayers = [];

        // Detect layers button click
        detectBtn.addEventListener('click', async () => {
            await this.detectModelLayers();
        });

        // Setup action buttons
        const selectAllBtn = document.getElementById('select-all-layers');
        const selectRecommendedBtn = document.getElementById('select-recommended-layers');
        const clearBtn = document.getElementById('clear-layers');

        if (selectAllBtn) {
            selectAllBtn.addEventListener('click', () => this.selectAllLayers());
        }
        if (selectRecommendedBtn) {
            selectRecommendedBtn.addEventListener('click', () => this.selectRecommendedLayers());
        }
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearLayerSelection());
        }

        // Setup category toggle (collapse/expand)
        document.querySelectorAll('.category-header').forEach(header => {
            header.addEventListener('click', (e) => {
                const category = header.closest('.layer-category');
                const layersDiv = category.querySelector('.category-layers');
                const toggle = header.querySelector('.category-toggle');

                if (layersDiv.style.display === 'none') {
                    layersDiv.style.display = 'block';
                    toggle.textContent = '▼';
                } else {
                    layersDiv.style.display = 'none';
                    toggle.textContent = '▶';
                }
            });
        });

        // Sync manual input to hidden field
        if (targetModulesManual) {
            targetModulesManual.addEventListener('input', () => {
                if (targetModulesHidden) {
                    targetModulesHidden.value = targetModulesManual.value;
                }
            });
        }
    }

    /**
     * Detect model layers via API
     */
    async detectModelLayers() {
        const modelName = document.getElementById('base-model')?.value?.trim();
        const hfToken = document.getElementById('hf-token')?.value || document.getElementById('hf-token-preload')?.value;

        if (!modelName) {
            this.toast.error('Please enter a model name first');
            return;
        }

        const detectBtn = document.getElementById('detect-layers-btn');
        const statusDiv = document.getElementById('layer-detection-status');
        const layersContainer = document.getElementById('detected-layers-container');
        const manualInput = document.getElementById('manual-layers-input');

        // Show loading state
        detectBtn.disabled = true;
        detectBtn.querySelector('.detect-layers-text').textContent = 'Detecting...';
        statusDiv.style.display = 'block';
        layersContainer.style.display = 'none';

        try {
            const response = await fetch('/model/layers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: modelName,
                    hf_token: hfToken || null
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to detect layers');
            }

            const data = await response.json();
            this.detectedLayers = data;
            this.recommendedLayers = data.recommended || [];

            // Populate the UI
            this.populateLayerCategories(data);

            // Show layers container, hide manual input
            layersContainer.style.display = 'block';
            manualInput.style.display = 'none';
            statusDiv.style.display = 'none';

            // Select recommended layers by default
            this.selectRecommendedLayers();

            const linearCount = data.total_linear || data.total_layers || 0;
            const embeddingCount = data.total_embedding || 0;
            const msg = embeddingCount > 0
                ? `Detected ${linearCount} linear + ${embeddingCount} embedding layers`
                : `Detected ${linearCount} layer types`;
            this.toast.success(msg);

        } catch (error) {
            console.error('Layer detection failed:', error);
            this.toast.error(`Layer detection failed: ${error.message}`);
            statusDiv.style.display = 'none';
        } finally {
            detectBtn.disabled = false;
            detectBtn.querySelector('.detect-layers-text').textContent = 'Detect Layers';
        }
    }

    /**
     * Populate layer categories in the UI
     */
    populateLayerCategories(data) {
        const categories = {
            attention: document.querySelector('#category-attention .category-layers'),
            mlp: document.querySelector('#category-mlp .category-layers'),
            embedding: document.querySelector('#category-embedding .category-layers'),
            other: document.querySelector('#category-other .category-layers')
        };

        const categoryContainers = {
            attention: document.getElementById('category-attention'),
            mlp: document.getElementById('category-mlp'),
            embedding: document.getElementById('category-embedding'),
            other: document.getElementById('category-other')
        };

        // Clear existing layers
        Object.values(categories).forEach(cat => {
            if (cat) cat.innerHTML = '';
        });

        // Populate each category
        for (const [catName, layers] of Object.entries(data.categories)) {
            const container = categories[catName];
            const categoryContainer = categoryContainers[catName];

            if (!container || !categoryContainer) continue;

            if (layers.length === 0) {
                categoryContainer.style.display = 'none';
                continue;
            }

            categoryContainer.style.display = 'block';

            // Update count
            const countSpan = categoryContainer.querySelector('.category-count');
            if (countSpan) {
                countSpan.textContent = `(${layers.length})`;
            }

            // Create checkboxes for each layer
            layers.forEach(layerName => {
                const layerDiv = document.createElement('div');
                layerDiv.className = 'layer-item';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `layer-${layerName}`;
                checkbox.value = layerName;
                checkbox.className = 'layer-checkbox';
                checkbox.addEventListener('change', () => this.updateSelectedLayers());

                const label = document.createElement('label');
                label.htmlFor = `layer-${layerName}`;
                label.textContent = layerName;

                // Get layer details
                const detail = data.layer_details.find(d => d.name === layerName);
                if (detail) {
                    const info = document.createElement('span');
                    info.className = 'layer-info';
                    const layerClass = detail.layer_class === 'embedding' ? '📦' : '';
                    info.textContent = `${layerClass}${detail.in_features}→${detail.out_features}`;
                    info.title = detail.layer_class === 'embedding'
                        ? 'Embedding layer (vocab_size→hidden_dim)'
                        : 'Linear layer (in→out)';
                    label.appendChild(info);
                }

                // Mark recommended layers
                if (this.recommendedLayers.includes(layerName)) {
                    layerDiv.classList.add('recommended');
                }

                layerDiv.appendChild(checkbox);
                layerDiv.appendChild(label);
                container.appendChild(layerDiv);
            });
        }
    }

    /**
     * Update selected layers and sync to hidden input
     */
    updateSelectedLayers() {
        const checkboxes = document.querySelectorAll('.layer-checkbox:checked');
        const selectedLayers = Array.from(checkboxes).map(cb => cb.value);

        // Update hidden input
        const targetModulesHidden = document.getElementById('target-modules');
        if (targetModulesHidden) {
            targetModulesHidden.value = selectedLayers.join(',');
        }

        // Update count display
        const countSpan = document.getElementById('selected-layers-count');
        if (countSpan) {
            countSpan.textContent = `${selectedLayers.length} layer${selectedLayers.length !== 1 ? 's' : ''} selected`;
        }
    }

    /**
     * Select all detected layers
     */
    selectAllLayers() {
        document.querySelectorAll('.layer-checkbox').forEach(cb => {
            cb.checked = true;
        });
        this.updateSelectedLayers();
    }

    /**
     * Select only recommended layers
     */
    selectRecommendedLayers() {
        document.querySelectorAll('.layer-checkbox').forEach(cb => {
            cb.checked = this.recommendedLayers.includes(cb.value);
        });
        this.updateSelectedLayers();
    }

    /**
     * Clear all layer selections
     */
    clearLayerSelection() {
        document.querySelectorAll('.layer-checkbox').forEach(cb => {
            cb.checked = false;
        });
        this.updateSelectedLayers();
    }

    /**
     * Setup training mode toggle (show/hide relevant parameter fields)
     */
    setupTrainingModeToggle() {
        const trainingMode = document.getElementById('training-mode');
        const betaField = document.getElementById('beta-field');
        const gammaField = document.getElementById('gamma-field');
        const labelSmoothingField = document.getElementById('label-smoothing-field');
        const descriptionEl = document.getElementById('training-mode-description');

        if (!trainingMode) return;

        const PREFERENCE_MODES = ['orpo', 'dpo', 'simpo', 'cpo', 'ipo', 'kto'];

        const MODE_DESCRIPTIONS = {
            sft: '<strong>SFT:</strong> Learn from good examples. Uses only the "chosen" response for each prompt — great for teaching your model a new style or task. No rejected responses needed.',
            orpo: '<strong>ORPO:</strong> A simple, all-in-one preference method. Combines supervised learning with odds-ratio preference optimization in a single pass — no reference model needed. Good default choice for preference training.',
            dpo: '<strong>DPO:</strong> The most popular preference method. Directly optimizes the policy to prefer chosen over rejected responses using a clever log-ratio trick. Stable and well-studied.',
            simpo: '<strong>SimPO:</strong> A streamlined DPO variant that skips the reference model entirely. Uses length-normalized rewards and a configurable margin (gamma) to separate good from bad responses.',
            cpo: '<strong>CPO:</strong> Reference-free contrastive learning on preference pairs. Similar to DPO but simpler — directly contrasts chosen vs. rejected without needing a frozen copy of the model.',
            kto: '<strong>KTO:</strong> Uses prospect theory to align models with binary feedback (good/bad) instead of paired preferences. Works with unpaired data — if you have rejected responses they\'re split into separate negative examples. Great when you only have thumbs-up/thumbs-down signals.',
            ipo: '<strong>IPO:</strong> A squared-loss variant of DPO that avoids overfitting to noisy preferences. More robust when your chosen/rejected labels may not be perfectly clean.',
        };

        const updateFields = (mode) => {
            const isPreference = PREFERENCE_MODES.includes(mode);
            // Beta: shown for all preference methods
            if (betaField) betaField.style.display = isPreference ? 'block' : 'none';
            // Gamma: SimPO only
            if (gammaField) gammaField.style.display = mode === 'simpo' ? 'block' : 'none';
            // Label smoothing: DPO and CPO
            if (labelSmoothingField) labelSmoothingField.style.display = (mode === 'dpo' || mode === 'cpo') ? 'block' : 'none';
            // Description
            if (descriptionEl) descriptionEl.innerHTML = MODE_DESCRIPTIONS[mode] || '';
        };

        trainingMode.addEventListener('change', (e) => updateFields(e.target.value));
        updateFields(trainingMode.value);
    }

    /**
     * Setup section navigation banner
     */
    setupSectionNav() {
        const navBtns = document.querySelectorAll('.section-nav-btn');
        if (!navBtns.length) return;

        navBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const targetId = btn.dataset.section;
                this.showSection(targetId);
            });
        });
    }

    /**
     * Switch to a specific section
     */
    showSection(sectionId) {
        // Hide all nav sections
        document.querySelectorAll('.nav-section').forEach(section => {
            section.style.display = 'none';
        });

        // Show target section
        const target = document.getElementById(sectionId);
        if (target) {
            target.style.display = 'block';
        }

        // Update nav button active state
        document.querySelectorAll('.section-nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.section === sectionId);
        });

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    /**
     * Setup "Apply Suggested Settings" button
     */
    setupPresetButton() {
        const btn = document.getElementById('apply-preset-btn');
        if (!btn) return;

        btn.addEventListener('click', async () => {
            const mode = document.getElementById('training-mode')?.value;
            if (!mode) return;

            btn.disabled = true;
            btn.textContent = '⏳ Loading...';

            try {
                const response = await fetch(`${window.location.origin}/presets/${mode}`);
                if (!response.ok) {
                    throw new Error(`No preset available for ${mode}`);
                }
                const preset = await response.json();
                const settings = preset.settings;

                // Map preset keys to form element IDs
                const fieldMap = {
                    learning_rate: 'learning-rate',
                    num_epochs: 'epochs',
                    batch_size: 'batch-size',
                    gradient_accumulation_steps: 'grad-accum',
                    warmup_ratio: 'warmup-ratio',
                    weight_decay: 'weight-decay',
                    lora_dropout: 'lora-dropout',
                    max_grad_norm: 'max-grad-norm',
                    lr_scheduler_type: 'lr-scheduler-type',
                    beta: 'beta',
                    gamma: 'gamma',
                    label_smoothing: 'label-smoothing',
                };

                const applied = [];
                for (const [key, elementId] of Object.entries(fieldMap)) {
                    if (settings[key] !== undefined) {
                        const el = document.getElementById(elementId);
                        if (el) {
                            el.value = settings[key];
                            applied.push(key);
                        }
                    }
                }

                // Show notes
                const notesEl = document.getElementById('preset-notes');
                if (notesEl && preset.notes) {
                    notesEl.textContent = preset.notes;
                    notesEl.style.display = 'block';
                }

                this.toast.success(`Applied ${applied.length} suggested settings for ${mode.toUpperCase()}`);
            } catch (error) {
                this.toast.error(`Failed to load preset: ${error.message}`);
            } finally {
                btn.disabled = false;
                btn.textContent = '📋 Apply Suggested Settings';
            }
        });
    }

    /**
     * Generate an output model name from base model, dataset, and training method
     */
    setupNameGenerator() {
        const btn = document.getElementById('generate-name-btn');
        if (!btn) return;

        btn.addEventListener('click', () => {
            const baseModel = document.getElementById('base-model')?.value?.trim() || '';
            const repoId = document.getElementById('hf-repo-id')?.value?.trim() || '';
            const mode = document.getElementById('training-mode')?.value || 'sft';

            // Extract short model name: "org/Model-Name-7B-Instruct" -> "Model-Name-7B-Instruct"
            const modelPart = baseModel.split('/').pop() || 'model';

            // Extract short dataset name: "org/dataset-name" -> "dataset-name"
            const datasetPart = repoId.split('/').pop() || '';

            const parts = [modelPart];
            if (datasetPart) parts.push(datasetPart);
            parts.push(mode.toUpperCase());

            const name = parts.join('-');
            document.getElementById('output-name').value = name;
        });
    }

    /**
     * Setup advanced settings toggle
     */
    setupAdvancedToggle() {
        const toggleBtn = document.getElementById('toggle-advanced');
        const advancedSections = document.querySelectorAll('.advanced-section');

        if (!toggleBtn) return;

        let advancedVisible = false;

        toggleBtn.addEventListener('click', () => {
            advancedVisible = !advancedVisible;

            advancedSections.forEach(section => {
                section.style.display = advancedVisible ? 'block' : 'none';
            });

            const span = toggleBtn.querySelector('span');
            if (span) {
                span.textContent = advancedVisible ?
                    '⚙️ Hide Advanced Settings' :
                    '⚙️ Show Advanced Settings';
            }
        });
    }

    /**
     * Setup conditional visibility for W&B and HF Hub configs
     */
    setupConditionalVisibility() {
        // W&B config
        const useWandb = document.getElementById('use-wandb');
        const wandbConfig = document.getElementById('wandb-config');

        if (useWandb && wandbConfig) {
            useWandb.addEventListener('change', (e) => {
                wandbConfig.style.display = e.target.checked ? 'block' : 'none';
            });

            // Initialize
            if (useWandb.checked) {
                wandbConfig.style.display = 'block';
            }
        }

        // Optimizer-specific settings visibility
        const optimizerType = document.getElementById('optimizer-type');
        if (optimizerType) {
            optimizerType.addEventListener('change', () => updateOptimizerSettingsVisibility());
            updateOptimizerSettingsVisibility();
        }

        // HF Hub config
        const pushHub = document.getElementById('push-hub');
        const hfHubConfig = document.getElementById('hf-hub-config');

        if (pushHub && hfHubConfig) {
            pushHub.addEventListener('change', (e) => {
                hfHubConfig.style.display = e.target.checked ? 'block' : 'none';
            });

            // Initialize
            if (pushHub.checked) {
                hfHubConfig.style.display = 'block';
            }
        }

        // Training backend toggle (local vs HF Jobs)
        const trainingBackend = document.getElementById('training-backend');
        const hfJobsOptions = document.getElementById('hf-jobs-options');
        if (trainingBackend && hfJobsOptions) {
            const syncBackend = () => {
                hfJobsOptions.style.display = trainingBackend.value === 'hf_jobs' ? 'block' : 'none';
            };
            trainingBackend.addEventListener('change', syncBackend);
            syncBackend();
        }
    }

    /**
     * Setup input validation
     */
    setupInputValidation() {
        // Add validation on blur for all inputs with rules
        for (const fieldId of Object.keys(ValidationRules)) {
            const field = document.getElementById(fieldId);
            if (!field) continue;

            // Debounced validation
            const validateField = debounce(() => {
                const rules = ValidationRules[fieldId];
                const errors = Validator.validateField(fieldId, field.value, rules);

                if (errors.length > 0) {
                    Validator.showFieldError(fieldId, errors);
                } else {
                    Validator.clearFieldError(fieldId);
                }
            }, 500);

            field.addEventListener('input', validateField);
            field.addEventListener('blur', () => {
                // Clear debounce and validate immediately
                validateField.cancel?.();
                const rules = ValidationRules[fieldId];
                const errors = Validator.validateField(fieldId, field.value, rules);

                if (errors.length > 0) {
                    Validator.showFieldError(fieldId, errors);
                } else {
                    Validator.clearFieldError(fieldId);
                }
            });
        }
    }

    /**
     * Setup magical interactions (sparkles, etc.)
     */
    setupMagicalInteractions() {
        // Add sparkle effect on input focus
        const inputs = document.querySelectorAll('.magic-input, .magic-select');
        inputs.forEach(input => {
            input.addEventListener('focus', (e) => {
                createSparkle(e.target);
            });
        });

        // Animate wizard hat on hover
        const wizardHat = document.querySelector('.wizard-hat');
        if (wizardHat) {
            wizardHat.addEventListener('mouseenter', () => {
                wizardHat.style.animation = 'wiggle 0.5s ease-in-out';
                setTimeout(() => {
                    wizardHat.style.animation = 'wiggle 2s ease-in-out infinite';
                }, 500);
            });
        }

        // Easter egg: Konami code
        this.setupKonamiCode();
    }

    /**
     * Setup Konami code easter egg
     */
    setupKonamiCode() {
        let konamiCode = [];
        const konamiPattern = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];

        document.addEventListener('keydown', (e) => {
            konamiCode.push(e.key);
            konamiCode = konamiCode.slice(-10);

            if (konamiCode.join(',') === konamiPattern.join(',')) {
                this.activateSuperMagic();
            }
        });
    }

    /**
     * Activate super magic mode (easter egg)
     */
    activateSuperMagic() {
        this.toast.success('🌟 SUPER MAGIC MODE ACTIVATED! 🌟');
        document.body.style.animation = 'rainbow 3s linear infinite';

        const style = document.createElement('style');
        style.textContent = `
            @keyframes rainbow {
                0% { filter: hue-rotate(0deg); }
                100% { filter: hue-rotate(360deg); }
            }
        `;
        document.head.appendChild(style);

        setTimeout(() => {
            document.body.style.animation = '';
            style.remove();
        }, 5000);
    }

    /**
     * Setup auto-save to localStorage
     */
    setupAutoSave() {
        const form = document.getElementById('training-form');
        if (!form) return;

        // Load saved state on init
        this.loadFormState();

        // Auto-save every 30 seconds
        setInterval(() => {
            this.saveFormState();
        }, 30000);

        // Save before unload
        window.addEventListener('beforeunload', (e) => {
            this.saveFormState();

            // Warn if form has been modified
            if (this.isFormModified()) {
                e.preventDefault();
                e.returnValue = '';
            }
        });
    }

    /**
     * Save form state to localStorage
     */
    saveFormState() {
        try {
            const form = document.getElementById('training-form');
            if (!form) return;

            const state = FormUI.getData(form);
            localStorage.setItem('merlina_form_state', JSON.stringify(state));
            console.log('📝 Form state auto-saved');
        } catch (error) {
            console.error('Failed to save form state:', error);
        }
    }

    /**
     * Load form state from localStorage
     */
    loadFormState() {
        try {
            const savedState = localStorage.getItem('merlina_form_state');
            if (!savedState) return;

            const state = JSON.parse(savedState);
            const form = document.getElementById('training-form');
            if (!form) return;

            FormUI.setData(form, state);

            // Dispatch change events for toggle elements to update visibility
            const toggleElements = [
                'use-lora',
                'use-wandb',
                'push-hub',
                'training-mode',
                'dataset-source-type',
                'dataset-format-type'
            ];

            toggleElements.forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    element.dispatchEvent(new Event('change'));
                }
            });

            console.log('📝 Form state restored from auto-save');
        } catch (error) {
            console.error('Failed to load form state:', error);
        }
    }

    /**
     * Check if form has been modified
     */
    isFormModified() {
        const outputName = document.getElementById('output-name')?.value;
        return outputName && outputName.trim().length > 0;
    }

    /**
     * Setup keyboard shortcuts
     */
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+S / Cmd+S: Save config
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                this.configManager.showSaveModal();
            }

            // Ctrl+O / Cmd+O: Load config
            if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
                e.preventDefault();
                this.configManager.showLoadModal();
            }

            // Ctrl+Enter / Cmd+Enter: Submit form
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                const form = document.getElementById('training-form');
                if (form) {
                    form.dispatchEvent(new Event('submit'));
                }
            }
        });
    }

    /**
     * Load and display version information
     */
    async loadVersion() {
        try {
            // Import MerlinaAPI from api.js
            const { MerlinaAPI } = await import('./api.js');

            const versionInfo = await MerlinaAPI.getVersion();
            const versionElement = document.getElementById('version-info');

            if (versionElement && versionInfo) {
                const versionText = versionInfo.release_name
                    ? `Merlina v${versionInfo.version} - ${versionInfo.release_name}`
                    : `Merlina v${versionInfo.version}`;

                versionElement.textContent = versionText;
                versionElement.title = `Released: ${versionInfo.release_date}`;

                console.log(`📦 ${versionText}`);
            }
        } catch (error) {
            console.warn('Failed to load version information:', error);
            const versionElement = document.getElementById('version-info');
            if (versionElement) {
                versionElement.textContent = 'Merlina';
            }
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.merlinaApp = new MerlinaApp();
});

export { MerlinaApp };
