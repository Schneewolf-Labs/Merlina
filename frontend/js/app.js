// Main Application Module - Initializes and coordinates all modules

import { JobManager } from './jobs.js';
import { DatasetManager } from './dataset.js';
import { ConfigManager } from './config.js';
import { GPUManager } from './gpu.js';
import { ModelManager } from './model.js';
import { Toast, FormUI, Tooltip, createSparkle } from './ui.js';
import { Validator, debounce } from './validation.js';
import { ThemeManager } from './theme.js';

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

        this.toast = new Toast();

        // Make managers available globally for debugging and onclick handlers
        window.jobManager = this.jobManager;
        window.datasetManager = this.datasetManager;
        window.configManager = this.configManager;
        window.gpuManager = this.gpuManager;
        window.modelManager = this.modelManager;
        window.themeManager = this.themeManager;

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

        console.log('✨ Merlina initialized successfully!');
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

            // Attention
            attn_implementation: document.getElementById('attn-implementation')?.value || 'auto',

            // GPU
            gpu_ids: this.gpuManager.getSelectedGPUs(),

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
     * Setup training mode toggle (show/hide ORPO beta field)
     */
    setupTrainingModeToggle() {
        const trainingMode = document.getElementById('training-mode');
        const betaField = document.getElementById('beta-field');

        if (!trainingMode || !betaField) return;

        trainingMode.addEventListener('change', (e) => {
            const mode = e.target.value;
            // Show beta field only for ORPO mode
            betaField.style.display = mode === 'orpo' ? 'block' : 'none';
        });

        // Initialize based on current value
        const currentMode = trainingMode.value;
        betaField.style.display = currentMode === 'orpo' ? 'block' : 'none';
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
    }

    /**
     * Setup input validation
     */
    setupInputValidation() {
        // Add validation on blur for all inputs with rules
        for (const fieldId of Object.keys(Validator.validationRules || {})) {
            const field = document.getElementById(fieldId);
            if (!field) continue;

            // Debounced validation
            const validateField = debounce(() => {
                const rules = Validator.validationRules[fieldId];
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
                const rules = Validator.validationRules[fieldId];
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
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.merlinaApp = new MerlinaApp();
});

export { MerlinaApp };
