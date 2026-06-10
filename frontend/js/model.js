// Model Module - Model preloading and validation

import { MerlinaAPI } from './api.js';
import { Toast, LoadingManager } from './ui.js';
import { sanitizeHTML } from './validation.js';

/**
 * Model Manager - handles model operations
 */
class ModelManager {
    constructor() {
        this.toast = new Toast();
        this.setupEventListeners();
        this.loadLocalModels();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Preload model button
        const preloadBtn = document.getElementById('preload-model-button');
        if (preloadBtn) {
            preloadBtn.addEventListener('click', () => this.preloadModel());
        }

        // Local model picker: selecting an entry fills the base model input
        const localSelect = document.getElementById('local-model-select');
        if (localSelect) {
            localSelect.addEventListener('change', () => {
                const baseModelInput = document.getElementById('base-model');
                if (localSelect.value && baseModelInput) {
                    baseModelInput.value = localSelect.value;
                    baseModelInput.dispatchEvent(new Event('change', { bubbles: true }));
                }
            });
        }

        const refreshBtn = document.getElementById('refresh-local-models');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadLocalModels(true));
        }
    }

    /**
     * Load locally available models (HF cache + ./models) into the picker
     * so users can choose a base model without internet access.
     */
    async loadLocalModels(notify = false) {
        const localSelect = document.getElementById('local-model-select');
        const datalist = document.getElementById('local-models-list');
        const offlineBadge = document.getElementById('offline-mode-badge');

        try {
            const data = await MerlinaAPI.getLocalModels();

            if (offlineBadge) {
                offlineBadge.style.display = data.offline_mode ? 'inline-block' : 'none';
            }

            const models = data.models || [];

            if (localSelect) {
                localSelect.innerHTML = '';
                const placeholder = document.createElement('option');
                placeholder.value = '';
                placeholder.textContent = models.length
                    ? '— Select a model already on this machine —'
                    : '— No local models found —';
                localSelect.appendChild(placeholder);

                const groups = [
                    { source: 'hf_cache', label: '🤗 HuggingFace Cache' },
                    { source: 'models_dir', label: '📦 Merlina Models Folder' },
                ];
                for (const { source, label } of groups) {
                    const entries = models.filter(m => m.source === source);
                    if (!entries.length) continue;
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = label;
                    for (const m of entries) {
                        const option = document.createElement('option');
                        option.value = m.model_id;
                        option.textContent = m.name || m.model_id;
                        optgroup.appendChild(option);
                    }
                    localSelect.appendChild(optgroup);
                }
            }

            if (datalist) {
                datalist.innerHTML = '';
                for (const m of models) {
                    const option = document.createElement('option');
                    option.value = m.model_id;
                    datalist.appendChild(option);
                }
            }

            if (notify) {
                this.toast.success(`Found ${models.length} local model${models.length === 1 ? '' : 's'}`);
            }
        } catch (error) {
            console.error('Failed to load local models:', error);
            if (notify) {
                this.toast.error(`Failed to load local models: ${error.message}`);
            }
        }
    }

    /**
     * Preload and validate model
     */
    async preloadModel() {
        const baseModelInput = document.getElementById('base-model');
        const hfTokenInput = document.getElementById('hf-token-preload');
        const preloadButton = document.getElementById('preload-model-button');
        const modelStatus = document.getElementById('model-status');
        const modelInfo = document.getElementById('model-info');

        const baseModel = baseModelInput?.value.trim();
        const hfToken = hfTokenInput?.value.trim();

        if (!baseModel) {
            this.toast.error('Please enter a base model name');
            return;
        }

        try {
            LoadingManager.show(preloadButton, '⏳ Loading model tokenizer...');

            if (modelStatus) {
                modelStatus.style.display = 'none';
            }

            const data = await MerlinaAPI.preloadModel(baseModel, hfToken || null);

            // Display model info
            if (modelInfo) {
                modelInfo.innerHTML = `
                    <strong>${sanitizeHTML(data.model_name)}</strong><br/>
                    Vocab Size: ${data.vocab_size.toLocaleString()}<br/>
                    Max Length: ${data.model_max_length.toLocaleString()}<br/>
                    Chat Template: ${data.has_chat_template ? '✓ Detected' : '✗ Not found'}<br/>
                    ${data.has_chat_template ? '<span style="color: var(--primary-purple);">💡 You can now use "Tokenizer" format for accurate preview!</span>' : ''}
                `;
            }

            if (modelStatus) {
                // Reset to success styling
                const statusDiv = modelStatus.querySelector('div');
                if (statusDiv) {
                    statusDiv.style.background = '#e8f5e9';
                    statusDiv.style.borderColor = '#4caf50';

                    const header = statusDiv.querySelector('div:first-child');
                    if (header) {
                        header.textContent = '✓ Model Ready';
                        header.style.color = '#2e7d32';
                    }
                }

                modelStatus.style.display = 'block';
            }

            this.toast.success('Model tokenizer loaded successfully!');

        } catch (error) {
            console.error('Failed to preload model:', error);
            this.toast.error(`Failed to load model: ${error.message}`);

            if (modelInfo) {
                modelInfo.innerHTML = `<span style="color: var(--danger);">${sanitizeHTML(error.message)}</span>`;
            }

            if (modelStatus) {
                const statusDiv = modelStatus.querySelector('div');
                if (statusDiv) {
                    statusDiv.style.background = '#ffebee';
                    statusDiv.style.borderColor = 'var(--danger)';

                    const header = statusDiv.querySelector('div:first-child');
                    if (header) {
                        header.textContent = '✗ Error';
                        header.style.color = 'var(--danger)';
                    }
                }

                modelStatus.style.display = 'block';
            }

        } finally {
            LoadingManager.hide(preloadButton);
        }
    }
}

export { ModelManager };
