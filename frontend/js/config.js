// Config Management Module - Save/load configurations

import { MerlinaAPI } from './api.js';
import { Toast, Modal, LoadingManager } from './ui.js';
import { sanitizeHTML } from './validation.js';

/**
 * Configuration Manager - handles saving/loading configs
 */
class ConfigManager {
    constructor() {
        this.toast = new Toast();
        this.saveModal = new Modal('save-config-modal');
        this.loadModal = new Modal('load-config-modal');
        this.loadFromJobModal = new Modal('load-from-job-modal');
        this.manageModal = new Modal('manage-configs-modal');

        this.setupEventListeners();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Save config button
        const saveBtn = document.getElementById('save-config-btn');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => this.showSaveModal());
        }

        // Load config button
        const loadBtn = document.getElementById('load-config-btn');
        if (loadBtn) {
            loadBtn.addEventListener('click', () => this.showLoadModal());
        }

        // Load from job button
        const loadFromJobBtn = document.getElementById('load-from-job-btn');
        if (loadFromJobBtn) {
            loadFromJobBtn.addEventListener('click', () => this.showLoadFromJobModal());
        }

        // Manage configs button
        const manageBtn = document.getElementById('manage-configs-btn');
        if (manageBtn) {
            manageBtn.addEventListener('click', () => this.showManageModal());
        }
    }

    /**
     * Show save config modal
     */
    showSaveModal() {
        this.saveModal.show();
    }

    /**
     * Show load config modal
     */
    async showLoadModal() {
        this.loadModal.show();
        await this.loadConfigsList();
    }

    /**
     * Show load from job modal
     */
    async showLoadFromJobModal() {
        this.loadFromJobModal.show();
        await this.loadJobHistoryList();
    }

    /**
     * Load job history list for the load-from-job modal
     */
    async loadJobHistoryList() {
        const listEl = document.getElementById('job-history-list');
        if (!listEl) return;

        LoadingManager.showSkeleton(listEl, 3);

        try {
            const result = await MerlinaAPI.getJobHistory(50, 0);
            const jobs = result.jobs;

            if (jobs.length === 0) {
                listEl.innerHTML = '<p class="empty-state">No previous jobs found.</p>';
                return;
            }

            listEl.innerHTML = jobs.map(job => {
                const summary = job.config_summary || {};
                const statusClass = this.getJobStatusClass(job.status);
                const date = new Date(job.created_at).toLocaleString();

                return `
                    <div class="config-item job-history-item" data-job-id="${sanitizeHTML(job.job_id)}">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4>${sanitizeHTML(summary.output_name || job.job_id)}</h4>
                            <span class="job-status-badge ${statusClass}">${sanitizeHTML(job.status)}</span>
                        </div>
                        ${summary.base_model ? `<p style="margin: 4px 0; font-size: 0.9em; opacity: 0.8;">Model: ${sanitizeHTML(summary.base_model)}</p>` : ''}
                        <div class="config-meta">
                            ${summary.training_mode ? `<span class="config-tag">${sanitizeHTML(summary.training_mode.toUpperCase())}</span>` : ''}
                            <span class="config-date">${date}</span>
                        </div>
                    </div>
                `;
            }).join('');

            // Add click handlers
            listEl.querySelectorAll('.job-history-item').forEach(item => {
                item.addEventListener('click', () => {
                    const jobId = item.dataset.jobId;
                    this.loadConfigFromJob(jobId);
                });
            });

        } catch (error) {
            console.error('Failed to load job history:', error);
            listEl.innerHTML = '<p class="error-state">Error loading job history</p>';
        }
    }

    /**
     * Get CSS class for job status badge
     */
    getJobStatusClass(status) {
        const classMap = {
            completed: 'status-completed',
            failed: 'status-failed',
            stopped: 'status-stopped',
            training: 'status-running',
            running: 'status-running',
            queued: 'status-queued'
        };
        return classMap[status] || 'status-default';
    }

    /**
     * Load config from a previous job and populate form
     */
    async loadConfigFromJob(jobId) {
        try {
            const result = await MerlinaAPI.getJobConfig(jobId);
            const config = this.normalizeJobConfig(result.config);

            this.populateForm(config);

            this.toast.success('Configuration loaded from previous job!');
            this.loadFromJobModal.hide();

        } catch (error) {
            console.error('Failed to load job config:', error);
            this.toast.error(`Failed to load job configuration: ${error.message}`);
        }
    }

    /**
     * Normalize a stored job config (Pydantic field names) to the format populateForm expects.
     * The stored config uses Pydantic model field names (e.g., num_epochs, batch_size)
     * while populateForm expects the names from getCurrentConfig (e.g., num_train_epochs,
     * per_device_train_batch_size).
     */
    normalizeJobConfig(config) {
        const normalized = { ...config };

        // Map Pydantic field names to form-expected names
        if ('num_epochs' in normalized && !('num_train_epochs' in normalized)) {
            normalized.num_train_epochs = normalized.num_epochs;
        }
        if ('batch_size' in normalized && !('per_device_train_batch_size' in normalized)) {
            normalized.per_device_train_batch_size = normalized.batch_size;
        }

        // Handle target_modules: stored as array, form expects comma-separated string
        if (Array.isArray(normalized.target_modules)) {
            normalized.target_modules = normalized.target_modules.join(', ');
        }
        if (Array.isArray(normalized.modules_to_save)) {
            normalized.modules_to_save = normalized.modules_to_save.join(', ');
        }

        return normalized;
    }

    /**
     * Show manage configs modal
     */
    async showManageModal() {
        this.manageModal.show();
        await this.loadManageConfigsList();
    }

    /**
     * Get current configuration from form
     */
    getCurrentConfig() {
        // Gather dataset cards from the unified datasets list. The first card
        // becomes DatasetConfig.source; any additional cards become
        // additional_sources[]. We read the raw DOM so unfilled fields still
        // round-trip (saving shouldn't enforce validation).
        const cards = Array.from(document.querySelectorAll('#datasets-list .dataset-card'));
        const readCardSource = (card) => {
            const t = card.querySelector('.ds-source-type')?.value || 'huggingface';
            const src = { source_type: t };
            if (t === 'huggingface') {
                src.repo_id = card.querySelector('.ds-repo')?.value || '';
                src.split = card.querySelector('.ds-split')?.value || 'train';
            } else if (t === 'local_file') {
                src.file_path = card.querySelector('.ds-local-path')?.value || '';
                src.file_format = card.querySelector('.ds-local-format')?.value || '';
            } else if (t === 'upload') {
                src.dataset_id = card.dataset.uploadId || '';
            }
            const mapping = this.readCardColumnMapping(card);
            if (Object.keys(mapping).length > 0) src.column_mapping = mapping;
            return src;
        };

        const firstCard = cards[0];
        let datasetSource = firstCard
            ? readCardSource(firstCard)
            : { source_type: 'huggingface', repo_id: '', split: 'train' };
        const additionalSources = cards.slice(1).map(readCardSource);

        const formatType = document.getElementById('dataset-format-type')?.value || 'tokenizer';
        let datasetFormat = { format_type: formatType };

        if (formatType === 'qwen3') {
            datasetFormat.enable_thinking = document.getElementById('enable-thinking')?.checked ?? true;
        } else if (formatType === 'custom') {
            datasetFormat.custom_templates = {
                prompt_template: document.getElementById('custom-prompt-template')?.value || '',
                chosen_template: document.getElementById('custom-chosen-template')?.value || '',
                rejected_template: document.getElementById('custom-rejected-template')?.value || ''
            };
        }

        // Get LoRA config
        const useLora = document.getElementById('use-lora')?.checked ?? true;
        let loraConfig = {};
        if (useLora) {
            loraConfig = {
                lora_r: parseInt(document.getElementById('lora-r')?.value || 64),
                lora_alpha: parseInt(document.getElementById('lora-alpha')?.value || 32),
                lora_dropout: parseFloat(document.getElementById('lora-dropout')?.value || 0.05),
                target_modules: document.getElementById('target-modules')?.value || '',
                modules_to_save: document.getElementById('modules-to-save')?.value || '',
                lora_task_type: document.getElementById('lora-task-type')?.value || 'CAUSAL_LM'
            };
        }

        // Get training mode
        const trainingMode = document.getElementById('training-mode')?.value || 'orpo';

        // Build complete config
        const config = {
            base_model: document.getElementById('base-model')?.value || '',
            output_name: document.getElementById('output-name')?.value || '',
            model_type: document.getElementById('model-type')?.value || 'auto',
            training_mode: trainingMode,
            use_lora: useLora,
            ...loraConfig,
            use_4bit: document.getElementById('use-4bit')?.checked ?? true,
            max_length: parseInt(document.getElementById('max-length')?.value || 2048),
            max_prompt_length: parseInt(document.getElementById('max-prompt-length')?.value || 1024),
            num_epochs: parseInt(document.getElementById('epochs')?.value || 2),
            batch_size: parseInt(document.getElementById('batch-size')?.value || 1),
            gradient_accumulation_steps: parseInt(document.getElementById('grad-accum')?.value || 16),
            learning_rate: parseFloat(document.getElementById('learning-rate')?.value || 0.000005),
            warmup_ratio: parseFloat(document.getElementById('warmup-ratio')?.value || 0.05),
            beta: parseFloat(document.getElementById('beta')?.value || 0.1),
            seed: parseInt(document.getElementById('seed')?.value || 42),
            max_grad_norm: parseFloat(document.getElementById('max-grad-norm')?.value || 0.3),
            weight_decay: parseFloat(document.getElementById('weight-decay')?.value || 0.01),
            lr_scheduler_type: document.getElementById('lr-scheduler-type')?.value || 'cosine',
            logging_steps: parseInt(document.getElementById('logging-steps')?.value || 1),
            shuffle_dataset: document.getElementById('shuffle-dataset')?.checked ?? true,
            gradient_checkpointing: document.getElementById('gradient-checkpointing')?.checked ?? false,
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
            attn_implementation: document.getElementById('attn-implementation')?.value || 'auto',
            use_liger: document.getElementById('use-liger')?.checked ?? false,
            torch_compile: document.getElementById('torch-compile')?.checked ?? false,
            neftune_alpha: document.getElementById('neftune-alpha')?.value
                ? parseFloat(document.getElementById('neftune-alpha').value)
                : null,
            eval_on_start: document.getElementById('eval-on-start')?.checked ?? false,
            eval_steps: parseFloat(document.getElementById('eval-steps')?.value || 0.2),
            dataset: {
                source: datasetSource,
                additional_sources: additionalSources,
                format: datasetFormat,
                test_size: parseFloat(document.getElementById('test-size')?.value || 0.01),
                convert_messages_format: document.getElementById('convert-messages-checkbox')?.checked ?? true,
                system_prompt: document.getElementById('system-prompt-override')?.value?.trim() || undefined,
                system_prompt_mode: document.querySelector('input[name="system-prompt-mode"]:checked')?.value || 'fill_empty'
            }
        };

        // Optional fields
        const hfToken = document.getElementById('hf-token')?.value || document.getElementById('hf-token-preload')?.value;
        if (hfToken) {
            config.hf_token = hfToken;
        }

        // W&B settings
        if (document.getElementById('use-wandb')?.checked) {
            const wandbKey = document.getElementById('wandb-key')?.value;
            if (wandbKey) {
                config.wandb_key = wandbKey;
                config.wandb_project = document.getElementById('wandb-project')?.value || null;
                config.wandb_run_name = document.getElementById('wandb-run-name')?.value || null;
                const tagsInput = document.getElementById('wandb-tags')?.value;
                if (tagsInput) {
                    config.wandb_tags = tagsInput.split(',').map(t => t.trim()).filter(t => t);
                }
                config.wandb_notes = document.getElementById('wandb-notes')?.value || null;
            }
        }

        // HF Hub settings
        if (document.getElementById('push-hub')?.checked) {
            config.push_to_hub = true;
            config.hf_hub_private = document.getElementById('hf-hub-private')?.checked ?? true;
            config.merge_lora_before_upload = document.getElementById('merge-lora-before-upload')?.checked ?? true;
        }

        return config;
    }

    /**
     * Save current configuration
     */
    async saveConfig() {
        const nameInput = document.getElementById('config-name');
        const name = nameInput?.value.trim();

        if (!name) {
            this.toast.error('Please enter a configuration name');
            return;
        }

        const description = document.getElementById('config-description')?.value.trim() || '';
        const tagsInput = document.getElementById('config-tags')?.value.trim() || '';
        const tags = tagsInput ? tagsInput.split(',').map(t => t.trim()).filter(t => t) : [];

        const config = this.getCurrentConfig();

        try {
            await MerlinaAPI.saveConfig(name, config, description, tags);

            this.toast.success(`Configuration '${name}' saved successfully!`);
            this.saveModal.hide();

            // Clear form
            if (nameInput) nameInput.value = '';
            const descInput = document.getElementById('config-description');
            if (descInput) descInput.value = '';
            const tagsInputEl = document.getElementById('config-tags');
            if (tagsInputEl) tagsInputEl.value = '';

        } catch (error) {
            console.error('Failed to save config:', error);
            this.toast.error(`Failed to save: ${error.message}`);
        }
    }

    /**
     * Load list of configs for loading
     */
    async loadConfigsList() {
        const listEl = document.getElementById('config-list');
        if (!listEl) return;

        LoadingManager.showSkeleton(listEl, 3);

        try {
            const result = await MerlinaAPI.listConfigs();
            const configs = result.configs;

            if (configs.length === 0) {
                listEl.innerHTML = '<p class="empty-state">No saved configurations found.</p>';
                return;
            }

            listEl.innerHTML = configs.map(cfg => `
                <div class="config-item" data-config-name="${sanitizeHTML(cfg.filename)}">
                    <h4>${sanitizeHTML(cfg.name)}</h4>
                    ${cfg.description ? `<p>${sanitizeHTML(cfg.description)}</p>` : ''}
                    <div class="config-meta">
                        ${cfg.tags.map(tag => `<span class="config-tag">${sanitizeHTML(tag)}</span>`).join('')}
                        <span class="config-date">Modified: ${new Date(cfg.modified_at).toLocaleString()}</span>
                    </div>
                </div>
            `).join('');

            // Add click handlers
            listEl.querySelectorAll('.config-item').forEach(item => {
                item.addEventListener('click', () => {
                    const configName = item.dataset.configName;
                    this.loadConfigByName(configName);
                });
            });

        } catch (error) {
            console.error('Failed to load configs:', error);
            listEl.innerHTML = '<p class="error-state">Error loading configurations</p>';
        }
    }

    /**
     * Load list of configs for management
     */
    async loadManageConfigsList() {
        const listEl = document.getElementById('manage-config-list');
        if (!listEl) return;

        LoadingManager.showSkeleton(listEl, 3);

        try {
            const result = await MerlinaAPI.listConfigs();
            const configs = result.configs;

            if (configs.length === 0) {
                listEl.innerHTML = '<p class="empty-state">No saved configurations found.</p>';
                return;
            }

            listEl.innerHTML = configs.map(cfg => `
                <div class="config-item">
                    <h4>${sanitizeHTML(cfg.name)}</h4>
                    ${cfg.description ? `<p>${sanitizeHTML(cfg.description)}</p>` : ''}
                    <div class="config-meta">
                        ${cfg.tags.map(tag => `<span class="config-tag">${sanitizeHTML(tag)}</span>`).join('')}
                        <span class="config-date">Modified: ${new Date(cfg.modified_at).toLocaleString()}</span>
                    </div>
                    <div class="config-actions">
                        <button class="config-load-btn" data-action="load" data-config="${sanitizeHTML(cfg.filename)}">
                            📂 Load
                        </button>
                        <button class="config-delete-btn" data-action="delete" data-config="${sanitizeHTML(cfg.filename)}">
                            🗑️ Delete
                        </button>
                    </div>
                </div>
            `).join('');

            // Add click handlers
            listEl.querySelectorAll('[data-action="load"]').forEach(btn => {
                btn.addEventListener('click', () => {
                    const configName = btn.dataset.config;
                    this.loadConfigByName(configName);
                    this.manageModal.hide();
                });
            });

            listEl.querySelectorAll('[data-action="delete"]').forEach(btn => {
                btn.addEventListener('click', () => {
                    const configName = btn.dataset.config;
                    this.deleteConfig(configName);
                });
            });

        } catch (error) {
            console.error('Failed to load configs:', error);
            listEl.innerHTML = '<p class="error-state">Error loading configurations</p>';
        }
    }

    /**
     * Load a specific config and populate form
     */
    async loadConfigByName(name) {
        try {
            const result = await MerlinaAPI.loadConfig(name);
            const config = result.config;

            // Populate form fields
            this.populateForm(config);

            this.toast.success(`Configuration '${name}' loaded successfully!`);
            this.loadModal.hide();

        } catch (error) {
            console.error('Failed to load config:', error);
            this.toast.error(`Failed to load configuration: ${error.message}`);
        }
    }

    /**
     * Populate form with config data
     */
    populateForm(config) {
        // Basic settings
        this.setInputValue('base-model', config.base_model);
        this.setInputValue('output-name', config.output_name);
        this.setInputValue('model-type', config.model_type || 'auto');
        this.setCheckboxValue('use-lora', config.use_lora);
        this.setCheckboxValue('use-4bit', config.use_4bit);

        // LoRA settings (use != null to allow falsy values like 0)
        if (config.lora_r != null) this.setInputValue('lora-r', config.lora_r);
        if (config.lora_alpha != null) this.setInputValue('lora-alpha', config.lora_alpha);
        if (config.lora_dropout != null) this.setInputValue('lora-dropout', config.lora_dropout);
        if (config.target_modules) this.setInputValue('target-modules', config.target_modules);
        if (config.modules_to_save) this.setInputValue('modules-to-save', config.modules_to_save);
        if (config.lora_task_type) this.setInputValue('lora-task-type', config.lora_task_type);

        // Training settings — setting training-mode triggers the mirror sync
        this.setInputValue('training-mode', config.training_mode || 'orpo');
        const trainingModeEl = document.getElementById('training-mode');
        if (trainingModeEl) trainingModeEl.dispatchEvent(new Event('change'));
        this.setInputValue('max-length', config.max_length || 2048);
        this.setInputValue('max-prompt-length', config.max_prompt_length || 1024);
        this.setInputValue('epochs', config.num_epochs || config.num_train_epochs || 2);
        this.setInputValue('batch-size', config.batch_size || config.per_device_train_batch_size || 1);
        this.setInputValue('grad-accum', config.gradient_accumulation_steps || 16);
        this.setInputValue('learning-rate', config.learning_rate || 0.000005);
        this.setInputValue('warmup-ratio', config.warmup_ratio || 0.05);
        this.setInputValue('beta', config.beta || 0.1);
        this.setInputValue('gamma', config.gamma || 0.5);
        this.setInputValue('label-smoothing', config.label_smoothing || 0.0);
        this.setInputValue('seed', config.seed || 42);
        this.setInputValue('max-grad-norm', config.max_grad_norm || 0.3);
        this.setInputValue('weight-decay', config.weight_decay || 0.01);
        this.setInputValue('lr-scheduler-type', config.lr_scheduler_type || 'cosine');
        this.setInputValue('logging-steps', config.logging_steps || 1);
        this.setInputValue('eval-steps', config.eval_steps || 0.2);

        this.setCheckboxValue('shuffle-dataset', config.shuffle_dataset);
        this.setCheckboxValue('gradient-checkpointing', config.gradient_checkpointing);

        // Optimizer settings
        this.setInputValue('optimizer-type', config.optimizer_type || 'paged_adamw_8bit');
        this.setInputValue('adam-beta1', config.adam_beta1 || 0.9);
        this.setInputValue('adam-beta2', config.adam_beta2 || 0.999);
        this.setInputValue('adam-epsilon', config.adam_epsilon || 1e-8);

        // Adafactor settings
        this.setCheckboxValue('adafactor-relative-step', config.adafactor_relative_step);
        this.setCheckboxValue('adafactor-scale-parameter', config.adafactor_scale_parameter);
        this.setCheckboxValue('adafactor-warmup-init', config.adafactor_warmup_init);
        this.setInputValue('adafactor-decay-rate', config.adafactor_decay_rate ?? -0.8);
        if (config.adafactor_beta1 != null) {
            this.setInputValue('adafactor-beta1', config.adafactor_beta1);
        } else {
            const beta1El = document.getElementById('adafactor-beta1');
            if (beta1El) beta1El.value = '';
        }
        this.setInputValue('adafactor-clip-threshold', config.adafactor_clip_threshold ?? 1.0);

        // Update optimizer settings visibility
        if (typeof window.updateOptimizerSettingsVisibility === 'function') {
            window.updateOptimizerSettingsVisibility();
        }

        // Attention settings
        this.setInputValue('attn-implementation', config.attn_implementation || 'auto');

        // Grimoire kernel/regularization features
        this.setCheckboxValue('use-liger', config.use_liger);
        this.setCheckboxValue('torch-compile', config.torch_compile);
        this.setCheckboxValue('eval-on-start', config.eval_on_start);
        if (config.neftune_alpha != null) {
            this.setInputValue('neftune-alpha', config.neftune_alpha);
        } else {
            const neftuneEl = document.getElementById('neftune-alpha');
            if (neftuneEl) neftuneEl.value = '';
        }

        // Dataset config
        if (config.dataset) {
            if (config.dataset.format) {
                this.setInputValue('dataset-format-type', config.dataset.format.format_type || 'tokenizer');

                const formatTypeEl = document.getElementById('dataset-format-type');
                if (formatTypeEl) {
                    formatTypeEl.dispatchEvent(new Event('change'));
                }

                if (config.dataset.format.enable_thinking !== undefined) {
                    this.setCheckboxValue('enable-thinking', config.dataset.format.enable_thinking);
                }

                if (config.dataset.format.custom_templates) {
                    const templates = config.dataset.format.custom_templates;
                    this.setInputValue('custom-prompt-template', templates.prompt_template);
                    this.setInputValue('custom-chosen-template', templates.chosen_template);
                    this.setInputValue('custom-rejected-template', templates.rejected_template);
                }
            }

            if (config.dataset.test_size) {
                this.setInputValue('test-size', config.dataset.test_size);
            }

            if (config.dataset.system_prompt) {
                this.setInputValue('system-prompt-override', config.dataset.system_prompt);
            }
            if (config.dataset.system_prompt_mode) {
                const radio = document.querySelector(`input[name="system-prompt-mode"][value="${config.dataset.system_prompt_mode}"]`);
                if (radio) radio.checked = true;
            }

            if (config.dataset.convert_messages_format !== undefined) {
                this.setCheckboxValue('convert-messages-checkbox', config.dataset.convert_messages_format);
            }

            // Rebuild dataset cards from source + additional_sources. The
            // top-level column_mapping is the historical home for the first
            // source's mapping, so merge it into source.column_mapping before
            // restoring.
            const primary = config.dataset.source ? { ...config.dataset.source } : null;
            if (primary && config.dataset.column_mapping && !primary.column_mapping) {
                primary.column_mapping = config.dataset.column_mapping;
            }
            const extras = Array.isArray(config.dataset.additional_sources)
                ? config.dataset.additional_sources
                : [];
            const allSources = primary ? [primary, ...extras] : extras;
            this.restoreDatasetCards(allSources);
        }

        // Optional fields
        if (config.hf_token) {
            this.setInputValue('hf-token', config.hf_token);
        }

        // W&B settings
        if (config.wandb_key) {
            this.setCheckboxValue('use-wandb', true);
            this.setInputValue('wandb-key', config.wandb_key);
            this.setInputValue('wandb-project', config.wandb_project || '');
            this.setInputValue('wandb-run-name', config.wandb_run_name || '');
            if (config.wandb_tags) {
                this.setInputValue('wandb-tags', config.wandb_tags.join(', '));
            }
            this.setInputValue('wandb-notes', config.wandb_notes || '');
        }

        // HF Hub settings
        if (config.push_to_hub) {
            this.setCheckboxValue('push-hub', true);
            this.setCheckboxValue('hf-hub-private', config.hf_hub_private ?? true);
            this.setCheckboxValue('merge-lora-before-upload', config.merge_lora_before_upload ?? true);
        }
    }

    /**
     * Helper to set input value
     */
    setInputValue(id, value) {
        const element = document.getElementById(id);
        if (element && value !== undefined && value !== null) {
            element.value = value;
        }
    }

    /**
     * Helper to set checkbox value
     */
    setCheckboxValue(id, value) {
        const element = document.getElementById(id);
        if (element && value !== undefined) {
            element.checked = !!value;
            // Trigger change event
            element.dispatchEvent(new Event('change'));
        }
    }

    /**
     * Read the {sourceCol: standardName} mapping from a dataset card.
     */
    readCardColumnMapping(card) {
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
     * Rebuild all dataset cards from an array of DatasetSource objects.
     */
    restoreDatasetCards(sources) {
        const listEl = document.getElementById('datasets-list');
        const dsManager = window.datasetManager;
        if (!listEl || !dsManager) return;

        listEl.innerHTML = '';
        dsManager.datasetCounter = 0;

        if (sources.length === 0) {
            // Always keep at least one card so the UI isn't empty.
            dsManager.addDataset({ canRemove: false });
            return;
        }

        sources.forEach((src, idx) => {
            const card = dsManager.addDataset({ canRemove: idx > 0 });
            if (!card) return;
            this.populateCardFromSource(card, src);
        });
    }

    /**
     * Apply a DatasetSource object to a card (source fields + column mapping).
     */
    populateCardFromSource(card, src) {
        const sourceType = src.source_type || 'huggingface';
        const sourceTypeSel = card.querySelector('.ds-source-type');
        if (sourceTypeSel) {
            sourceTypeSel.value = sourceType;
            sourceTypeSel.dispatchEvent(new Event('change'));
        }

        if (sourceType === 'huggingface') {
            const repoInput = card.querySelector('.ds-repo');
            const splitInput = card.querySelector('.ds-split');
            if (repoInput) repoInput.value = src.repo_id || '';
            if (splitInput) splitInput.value = src.split || 'train';
        } else if (sourceType === 'local_file') {
            const pathInput = card.querySelector('.ds-local-path');
            const fmtSelect = card.querySelector('.ds-local-format');
            if (pathInput) pathInput.value = src.file_path || '';
            if (fmtSelect && src.file_format) fmtSelect.value = src.file_format;
        } else if (sourceType === 'upload' && src.dataset_id) {
            card.dataset.uploadId = src.dataset_id;
        }

        if (src.column_mapping && Object.keys(src.column_mapping).length > 0) {
            const knownColumns = Object.keys(src.column_mapping);
            const pairs = [
                ['.ds-map-prompt', 'prompt'],
                ['.ds-map-chosen', 'chosen'],
                ['.ds-map-rejected', 'rejected'],
                ['.ds-map-system', 'system'],
                ['.ds-map-reasoning', 'reasoning'],
            ];
            const reverse = {};
            for (const [k, v] of Object.entries(src.column_mapping)) reverse[v] = k;
            for (const [sel, std] of pairs) {
                const dropdown = card.querySelector(sel);
                if (!dropdown) continue;
                while (dropdown.options.length > 1) dropdown.remove(1);
                for (const col of knownColumns) {
                    const opt = document.createElement('option');
                    opt.value = col;
                    opt.textContent = col;
                    dropdown.appendChild(opt);
                }
                if (reverse[std]) dropdown.value = reverse[std];
            }
            card.querySelector('.ds-columns-list').textContent = knownColumns.join(', ');
            card.querySelector('.ds-colmap-config').style.display = 'block';
        }
    }

    /**
     * Delete a configuration
     */
    async deleteConfig(name) {
        const confirmed = window.confirm(`Are you sure you want to delete the configuration '${name}'?`);
        if (!confirmed) return;

        try {
            await MerlinaAPI.deleteConfig(name);
            this.toast.success(`Configuration '${name}' deleted successfully!`);

            // Reload the list
            await this.loadManageConfigsList();

        } catch (error) {
            console.error('Failed to delete config:', error);
            this.toast.error(`Failed to delete: ${error.message}`);
        }
    }
}

// Make saveCurrentConfig available globally for onclick handlers in HTML
window.saveCurrentConfig = function() {
    if (window.configManager) {
        window.configManager.saveConfig();
    }
};

export { ConfigManager };
