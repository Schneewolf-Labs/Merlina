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
        // Get dataset config
        const sourceType = document.getElementById('dataset-source-type')?.value || 'huggingface';
        let datasetSource = { source_type: sourceType };

        if (sourceType === 'huggingface') {
            datasetSource.repo_id = document.getElementById('hf-repo-id')?.value || '';
            datasetSource.split = document.getElementById('hf-split')?.value || 'train';
        } else if (sourceType === 'local_file') {
            datasetSource.file_path = document.getElementById('local-file-path')?.value || '';
            datasetSource.file_format = document.getElementById('local-file-format')?.value || '';
        } else if (sourceType === 'upload') {
            datasetSource.dataset_id = window.uploadedDatasetId || '';
        }

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
                modules_to_save: document.getElementById('modules-to-save')?.value || ''
            };
        }

        // Get training mode
        const trainingMode = document.getElementById('training-mode')?.value || 'orpo';

        // Get column mapping if configured
        const columnMapping = this.getColumnMapping();
        if (columnMapping && Object.keys(columnMapping).length > 0) {
            datasetSource.column_mapping = columnMapping;
        }

        // Build complete config
        const config = {
            base_model: document.getElementById('base-model')?.value || '',
            output_name: document.getElementById('output-name')?.value || '',
            training_mode: trainingMode,
            use_lora: useLora,
            ...loraConfig,
            use_4bit: document.getElementById('use-4bit')?.checked ?? true,
            max_length: parseInt(document.getElementById('max-length')?.value || 2048),
            max_prompt_length: parseInt(document.getElementById('max-prompt-length')?.value || 1024),
            num_train_epochs: parseInt(document.getElementById('epochs')?.value || 2),
            per_device_train_batch_size: parseInt(document.getElementById('batch-size')?.value || 1),
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
            attn_implementation: document.getElementById('attn-implementation')?.value || 'auto',
            eval_steps: parseFloat(document.getElementById('eval-steps')?.value || 0.2),
            dataset: {
                source: datasetSource,
                format: datasetFormat,
                test_size: parseFloat(document.getElementById('test-size')?.value || 0.01)
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
        this.setCheckboxValue('use-lora', config.use_lora);
        this.setCheckboxValue('use-4bit', config.use_4bit);

        // LoRA settings
        if (config.lora_r) this.setInputValue('lora-r', config.lora_r);
        if (config.lora_alpha) this.setInputValue('lora-alpha', config.lora_alpha);
        if (config.lora_dropout) this.setInputValue('lora-dropout', config.lora_dropout);
        if (config.target_modules) this.setInputValue('target-modules', config.target_modules);
        if (config.modules_to_save) this.setInputValue('modules-to-save', config.modules_to_save);

        // Training settings
        this.setInputValue('training-mode', config.training_mode || 'orpo');
        this.setInputValue('max-length', config.max_length || 2048);
        this.setInputValue('max-prompt-length', config.max_prompt_length || 1024);
        this.setInputValue('epochs', config.num_train_epochs || 2);
        this.setInputValue('batch-size', config.per_device_train_batch_size || 1);
        this.setInputValue('grad-accum', config.gradient_accumulation_steps || 16);
        this.setInputValue('learning-rate', config.learning_rate || 0.000005);
        this.setInputValue('warmup-ratio', config.warmup_ratio || 0.05);
        this.setInputValue('beta', config.beta || 0.1);
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

        // Attention settings
        this.setInputValue('attn-implementation', config.attn_implementation || 'auto');

        // Dataset config
        if (config.dataset) {
            if (config.dataset.source) {
                const source = config.dataset.source;
                this.setInputValue('dataset-source-type', source.source_type || 'huggingface');

                if (source.repo_id) this.setInputValue('hf-repo-id', source.repo_id);
                if (source.split) this.setInputValue('hf-split', source.split);
                if (source.file_path) this.setInputValue('local-file-path', source.file_path);
                if (source.file_format) this.setInputValue('local-file-format', source.file_format);

                // Trigger source type change to show correct config
                const sourceTypeEl = document.getElementById('dataset-source-type');
                if (sourceTypeEl) {
                    sourceTypeEl.dispatchEvent(new Event('change'));
                }

                // Restore column mapping if present
                if (source.column_mapping) {
                    this.restoreColumnMapping(source.column_mapping);
                }
            }

            if (config.dataset.format) {
                this.setInputValue('dataset-format-type', config.dataset.format.format_type || 'tokenizer');

                // Trigger format type change
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
     * Get column mapping from UI (mirrors DatasetManager.getColumnMapping)
     */
    getColumnMapping() {
        const mapping = {};

        const promptCol = document.getElementById('map-prompt')?.value;
        const chosenCol = document.getElementById('map-chosen')?.value;
        const rejectedCol = document.getElementById('map-rejected')?.value;
        const systemCol = document.getElementById('map-system')?.value;
        const reasoningCol = document.getElementById('map-reasoning')?.value;

        if (promptCol) mapping[promptCol] = 'prompt';
        if (chosenCol) mapping[chosenCol] = 'chosen';
        if (rejectedCol) mapping[rejectedCol] = 'rejected';
        if (systemCol) mapping[systemCol] = 'system';
        if (reasoningCol) mapping[reasoningCol] = 'reasoning';

        return mapping;
    }

    /**
     * Restore column mapping to UI dropdowns
     * @param {Object} columnMapping - Mapping like {"original_col": "prompt", "other_col": "chosen"}
     */
    restoreColumnMapping(columnMapping) {
        // Reverse the mapping to find: standard_field -> original_column
        const reverseMapping = {};
        for (const [originalCol, standardField] of Object.entries(columnMapping)) {
            reverseMapping[standardField] = originalCol;
        }

        // Show the column mapping config section
        const columnMappingConfig = document.getElementById('column-mapping-config');
        if (columnMappingConfig) {
            columnMappingConfig.style.display = 'block';
        }

        // Get all original column names from the mapping
        const columns = Object.keys(columnMapping);

        // Populate the dropdowns with the saved column names
        const selectIds = ['map-prompt', 'map-chosen', 'map-rejected', 'map-system', 'map-reasoning'];
        selectIds.forEach(selectId => {
            const select = document.getElementById(selectId);
            if (!select) return;

            // Clear existing options except first (empty option)
            while (select.options.length > 1) {
                select.remove(1);
            }

            // Add column options
            columns.forEach(col => {
                const option = document.createElement('option');
                option.value = col;
                option.textContent = col;
                select.appendChild(option);
            });

            // Set the saved value
            const standardField = selectId.replace('map-', '');
            if (reverseMapping[standardField]) {
                select.value = reverseMapping[standardField];
            }
        });
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
