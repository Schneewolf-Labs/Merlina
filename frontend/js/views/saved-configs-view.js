/**
 * Saved Configurations View - Manage saved training configurations
 * Merlina Modular Frontend v2.0
 */

export class SavedConfigsView {
    constructor() {
        this.configs = [];
    }

    /**
     * Render saved configs view
     * @returns {string}
     */
    render() {
        return `
            <div class="saved-configs-view">
                ${this.renderHeader()}
                ${this.renderConfigsList()}
            </div>
        `;
    }

    /**
     * Render header
     * @returns {string}
     */
    renderHeader() {
        return `
            <div class="card-header">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 class="card-title">💾 Saved Configurations</h2>
                        <p class="card-subtitle">Load and manage your training configurations</p>
                    </div>
                    <button class="btn btn-danger" id="clear-all-configs-btn">
                        🗑️ Clear All
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render configs list
     * @returns {string}
     */
    renderConfigsList() {
        return `
            <div id="configs-list-container">
                <div style="text-align: center; padding: var(--space-2xl); color: var(--text-secondary);">
                    Loading configurations...
                </div>
            </div>
        `;
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Clear all button
        const clearBtn = document.getElementById('clear-all-configs-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                this.clearAllConfigs();
            });
        }

        // Load configs
        this.loadConfigs();
    }

    /**
     * Load configurations from localStorage
     */
    loadConfigs() {
        const configs = [];

        // Get all configs from localStorage
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith('merlina_config_')) {
                try {
                    const configData = localStorage.getItem(key);
                    const config = JSON.parse(configData);
                    const configName = key.replace('merlina_config_', '');

                    configs.push({
                        name: configName,
                        key: key,
                        config: config,
                        savedAt: this.getConfigTimestamp(key)
                    });
                } catch (error) {
                    console.error(`Failed to parse config ${key}:`, error);
                }
            }
        }

        // Sort by timestamp (newest first)
        configs.sort((a, b) => b.savedAt - a.savedAt);

        this.configs = configs;
        this.renderConfigsData(configs);
    }

    /**
     * Get config timestamp from name
     * @param {string} key - Config key
     * @returns {number}
     */
    getConfigTimestamp(key) {
        const match = key.match(/_(\d+)$/);
        return match ? parseInt(match[1]) : 0;
    }

    /**
     * Render configs data
     * @param {array} configs - Configs array
     */
    renderConfigsData(configs) {
        const container = document.getElementById('configs-list-container');
        if (!container) return;

        if (!configs || configs.length === 0) {
            container.innerHTML = `
                <div class="card">
                    <div class="empty-state">
                        <div class="empty-state-icon">📁</div>
                        <div class="empty-state-title">No Saved Configurations</div>
                        <div class="empty-state-message">
                            Save configurations from the Training Config page to reuse them later
                        </div>
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = configs.map(config => this.renderConfigCard(config)).join('');

        // Attach action listeners
        this.attachConfigActionListeners();
    }

    /**
     * Render individual config card
     * @param {object} configData - Config data
     * @returns {string}
     */
    renderConfigCard(configData) {
        const { name, config, savedAt } = configData;
        const mode = config.training_mode || 'orpo';
        const modeIcon = mode === 'orpo' ? '🏆' : '📖';
        const savedDate = savedAt ? new Date(savedAt).toLocaleString() : 'Unknown';

        return `
            <div class="card" style="margin-bottom: var(--space-lg);">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: var(--space-lg);">
                    <div>
                        <h3 style="margin: 0 0 var(--space-xs) 0; display: flex; align-items: center; gap: var(--space-sm);">
                            💾 ${config.output_name || name}
                            <span class="badge badge-${mode}">${modeIcon} ${mode.toUpperCase()}</span>
                        </h3>
                        <div style="font-size: var(--text-sm); color: var(--text-secondary);">
                            Saved: ${savedDate}
                        </div>
                    </div>
                </div>

                <!-- Config Details -->
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-md); margin-bottom: var(--space-lg); padding: var(--space-md); background: var(--surface-2); border-radius: var(--radius-md);">
                    <div>
                        <div style="font-size: var(--text-xs); color: var(--text-secondary);">Base Model</div>
                        <div style="font-weight: bold; font-size: var(--text-sm);">${config.base_model || 'N/A'}</div>
                    </div>
                    <div>
                        <div style="font-size: var(--text-xs); color: var(--text-secondary);">Learning Rate</div>
                        <div style="font-weight: bold;">${config.learning_rate?.toExponential(2) || 'N/A'}</div>
                    </div>
                    <div>
                        <div style="font-size: var(--text-xs); color: var(--text-secondary);">Epochs</div>
                        <div style="font-weight: bold;">${config.num_epochs || 'N/A'}</div>
                    </div>
                    <div>
                        <div style="font-size: var(--text-xs); color: var(--text-secondary);">Batch Size</div>
                        <div style="font-weight: bold;">${config.batch_size || 'N/A'}</div>
                    </div>
                    ${config.training_mode === 'orpo' ? `
                        <div>
                            <div style="font-size: var(--text-xs); color: var(--text-secondary);">ORPO Beta</div>
                            <div style="font-weight: bold;">${config.beta || 'N/A'}</div>
                        </div>
                    ` : ''}
                    <div>
                        <div style="font-size: var(--text-xs); color: var(--text-secondary);">LoRA</div>
                        <div style="font-weight: bold;">${config.use_lora ? `✅ r=${config.lora_r}` : '❌'}</div>
                    </div>
                    <div>
                        <div style="font-size: var(--text-xs); color: var(--text-secondary);">4-bit Quantization</div>
                        <div style="font-weight: bold;">${config.use_4bit ? '✅' : '❌'}</div>
                    </div>
                </div>

                <!-- Dataset Info -->
                ${config.dataset ? `
                    <div style="margin-bottom: var(--space-lg); padding: var(--space-md); background: var(--surface-1); border-radius: var(--radius-md); border-left: 3px solid var(--primary-purple);">
                        <div style="font-weight: bold; margin-bottom: var(--space-sm);">📚 Dataset</div>
                        <div style="font-size: var(--text-sm);">
                            ${config.dataset.source?.source_type === 'huggingface' ? `
                                <div>Source: HuggingFace</div>
                                <div>Repo: <code>${config.dataset.source.repo_id}</code></div>
                                <div>Split: ${config.dataset.source.split}</div>
                            ` : config.dataset.source?.source_type === 'local' ? `
                                <div>Source: Local File</div>
                                <div>Path: <code>${config.dataset.source.file_path}</code></div>
                            ` : `
                                <div>Source: ${config.dataset.source?.source_type || 'Unknown'}</div>
                            `}
                            <div>Format: ${config.dataset.format?.format_type || 'N/A'}</div>
                        </div>
                    </div>
                ` : ''}

                <!-- Actions -->
                <div style="display: flex; gap: var(--space-sm); justify-content: flex-end;">
                    <button class="btn btn-secondary btn-sm" data-config-action="view" data-config-key="${configData.key}">
                        👁️ View JSON
                    </button>
                    <button class="btn btn-primary btn-sm" data-config-action="load" data-config-key="${configData.key}">
                        📥 Load Config
                    </button>
                    <button class="btn btn-danger btn-sm" data-config-action="delete" data-config-key="${configData.key}">
                        🗑️ Delete
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Attach config action listeners
     */
    attachConfigActionListeners() {
        document.querySelectorAll('[data-config-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.configAction;
                const key = e.target.dataset.configKey;
                this.handleConfigAction(action, key);
            });
        });
    }

    /**
     * Handle config action
     * @param {string} action - Action name
     * @param {string} key - Config key
     */
    handleConfigAction(action, key) {
        switch (action) {
            case 'view':
                this.viewConfig(key);
                break;
            case 'load':
                this.loadConfig(key);
                break;
            case 'delete':
                this.deleteConfig(key);
                break;
        }
    }

    /**
     * View configuration JSON
     * @param {string} key - Config key
     */
    viewConfig(key) {
        try {
            const configData = localStorage.getItem(key);
            const config = JSON.parse(configData);

            // Create a modal or show in console
            console.log('Configuration:', config);

            window.dispatchEvent(new CustomEvent('toast', {
                detail: {
                    message: 'Configuration logged to console (F12)',
                    type: 'info'
                }
            }));
        } catch (error) {
            console.error('Failed to view config:', error);
        }
    }

    /**
     * Load configuration
     * @param {string} key - Config key
     */
    loadConfig(key) {
        try {
            const configData = localStorage.getItem(key);
            const config = JSON.parse(configData);

            // Save to session storage for the training config view to pick up
            sessionStorage.setItem('merlina_load_config', JSON.stringify(config));

            window.dispatchEvent(new CustomEvent('toast', {
                detail: {
                    message: 'Configuration loaded! Navigating to Training Config...',
                    type: 'success'
                }
            }));

            // Navigate to training config view
            setTimeout(() => {
                window.dispatchEvent(new CustomEvent('navigate', {
                    detail: { view: 'training' }
                }));
            }, 1000);
        } catch (error) {
            console.error('Failed to load config:', error);
            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: 'Failed to load configuration', type: 'danger' }
            }));
        }
    }

    /**
     * Delete configuration
     * @param {string} key - Config key
     */
    deleteConfig(key) {
        if (!confirm('Are you sure you want to delete this configuration?')) {
            return;
        }

        try {
            localStorage.removeItem(key);

            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: 'Configuration deleted', type: 'success' }
            }));

            // Reload configs
            this.loadConfigs();
        } catch (error) {
            console.error('Failed to delete config:', error);
            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: 'Failed to delete configuration', type: 'danger' }
            }));
        }
    }

    /**
     * Clear all configurations
     */
    clearAllConfigs() {
        if (!confirm('Are you sure you want to delete ALL saved configurations? This cannot be undone.')) {
            return;
        }

        try {
            // Remove all merlina configs
            const keysToRemove = [];
            for (let i = 0; i < localStorage.length; i++) {
                const key = localStorage.key(i);
                if (key && key.startsWith('merlina_config_')) {
                    keysToRemove.push(key);
                }
            }

            keysToRemove.forEach(key => localStorage.removeItem(key));

            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: `Deleted ${keysToRemove.length} configuration(s)`, type: 'success' }
            }));

            // Reload configs
            this.loadConfigs();
        } catch (error) {
            console.error('Failed to clear configs:', error);
            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: 'Failed to clear configurations', type: 'danger' }
            }));
        }
    }
}
