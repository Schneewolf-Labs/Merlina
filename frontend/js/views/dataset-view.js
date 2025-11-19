/**
 * Dataset Manager View - Dataset selection, preview, and validation
 * Merlina Modular Frontend v2.0
 */

export class DatasetView {
    constructor(trainingModeManager) {
        this.trainingModeManager = trainingModeManager;
        this.currentDataset = null;
        this.previewData = null;
    }

    /**
     * Render dataset view
     * @returns {string}
     */
    render() {
        return `
            <div class="dataset-view">
                ${this.renderHeader()}
                ${this.renderModeIndicator()}
                ${this.renderSourceSelector()}
                ${this.renderColumnMapping()}
                ${this.renderFormatting()}
                ${this.renderPreview()}
                ${this.renderActions()}
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
                <h2 class="card-title">📚 Dataset Manager</h2>
                <p class="card-subtitle">
                    Configure your dataset for ${this.trainingModeManager.getModeFullName()} training
                </p>
            </div>
        `;
    }

    /**
     * Render mode indicator with requirements
     * @returns {string}
     */
    renderModeIndicator() {
        const mode = this.trainingModeManager.getMode();
        const config = this.trainingModeManager.getModeConfig();
        const required = config.requiredColumns;

        return `
            <div class="alert alert-info">
                <strong>${config.icon} ${config.fullName} Mode Requirements:</strong>
                <div style="margin-top: var(--space-sm);">
                    Required columns: <strong>${required.join(', ')}</strong>
                    ${mode === 'orpo' ?
                        '<div style="margin-top: 4px; font-size: var(--text-sm);">💡 Switch to SFT mode if you only have chosen responses</div>' :
                        '<div style="margin-top: 4px; font-size: var(--text-sm);">💡 Rejected column will be ignored if present</div>'
                    }
                </div>
            </div>
        `;
    }

    /**
     * Render dataset source selector
     * @returns {string}
     */
    renderSourceSelector() {
        return `
            <div class="card">
                <div class="spell-section">
                    <h3>🔮 Dataset Source</h3>

                    <div class="form-group">
                        <label>Source Type</label>
                        <select id="dataset-source-type" class="select">
                            <option value="huggingface">HuggingFace Dataset</option>
                            <option value="upload">Upload File</option>
                            <option value="local">Local File Path</option>
                        </select>
                    </div>

                    <!-- HuggingFace Source -->
                    <div id="hf-source-config" class="source-config">
                        <div class="form-group">
                            <label>HuggingFace Repository ID</label>
                            <input type="text" id="hf-repo-id" class="input"
                                   value="schneewolflabs/Athanor-DPO"
                                   placeholder="username/dataset-name">
                            <small>Example: username/dataset-name</small>
                        </div>
                        <div class="form-group">
                            <label>Split</label>
                            <input type="text" id="hf-split" class="input" value="train">
                            <small>Usually "train" or "test"</small>
                        </div>
                    </div>

                    <!-- Upload Source -->
                    <div id="upload-source-config" class="source-config" style="display: none;">
                        <div class="file-input">
                            <input type="file" id="dataset-file" accept=".json,.jsonl,.csv,.parquet">
                            <label for="dataset-file" class="file-input-label">
                                <span>📤</span>
                                <span class="file-input-text">Click to upload or drag & drop</span>
                            </label>
                        </div>
                        <small>Supported: JSON, JSONL, CSV, Parquet</small>
                        <div id="upload-status"></div>
                    </div>

                    <!-- Local File Source -->
                    <div id="local-source-config" class="source-config" style="display: none;">
                        <div class="form-group">
                            <label>File Path</label>
                            <input type="text" id="local-file-path" class="input"
                                   placeholder="/path/to/dataset.json">
                        </div>
                        <div class="form-group">
                            <label>File Format</label>
                            <select id="local-file-format" class="select">
                                <option value="">Auto-detect</option>
                                <option value="json">JSON</option>
                                <option value="jsonl">JSONL</option>
                                <option value="csv">CSV</option>
                                <option value="parquet">Parquet</option>
                            </select>
                        </div>
                    </div>

                    <button class="btn btn-primary btn-block" id="load-dataset-btn">
                        🔍 Load Dataset
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render column mapping section
     * @returns {string}
     */
    renderColumnMapping() {
        const required = this.trainingModeManager.getRequiredColumns();
        const optional = this.trainingModeManager.getOptionalColumns();
        const showRejected = this.trainingModeManager.getMode() === 'orpo';

        return `
            <div class="card" id="column-mapping-section" style="display: none;">
                <div class="spell-section">
                    <h3>🔗 Column Mapping</h3>

                    <div class="alert alert-info">
                        <strong>Available Columns:</strong>
                        <div id="available-columns" style="margin-top: var(--space-sm);">
                            Not loaded yet
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Prompt Column <span style="color: var(--danger);">*</span></label>
                            <select id="map-prompt" class="select">
                                <option value="">-- Select Column --</option>
                            </select>
                            <small>User input/question</small>
                        </div>
                        <div class="form-group">
                            <label>Chosen Column <span style="color: var(--danger);">*</span></label>
                            <select id="map-chosen" class="select">
                                <option value="">-- Select Column --</option>
                            </select>
                            <small>Preferred response</small>
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group" style="display: ${showRejected ? 'block' : 'none'};" id="rejected-column-group">
                            <label>Rejected Column <span style="color: var(--danger);">*</span></label>
                            <select id="map-rejected" class="select">
                                <option value="">-- Select Column --</option>
                            </select>
                            <small>Non-preferred response (ORPO only)</small>
                        </div>
                        <div class="form-group">
                            <label>System Column (Optional)</label>
                            <select id="map-system" class="select">
                                <option value="">-- None --</option>
                            </select>
                            <small>System message</small>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Reasoning Column (Optional)</label>
                        <select id="map-reasoning" class="select">
                            <option value="">-- None --</option>
                        </select>
                        <small>Reasoning/thinking process (for Qwen3)</small>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render formatting section
     * @returns {string}
     */
    renderFormatting() {
        return `
            <div class="card">
                <div class="spell-section">
                    <h3>📝 Dataset Formatting</h3>

                    <div class="form-group">
                        <label>Format Type</label>
                        <select id="dataset-format-type" class="select">
                            <option value="tokenizer" selected>Tokenizer (Recommended)</option>
                            <option value="chatml">ChatML</option>
                            <option value="llama3">Llama 3</option>
                            <option value="mistral">Mistral Instruct</option>
                            <option value="qwen3">Qwen 3 (with thinking)</option>
                            <option value="custom">Custom Template</option>
                        </select>
                        <small>How to format conversations for the model</small>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Test Split Size</label>
                            <input type="number" id="test-size" class="input" value="0.01"
                                   min="0.001" max="0.5" step="0.01">
                            <small>Fraction for evaluation</small>
                        </div>
                        <div class="form-group">
                            <label>Max Samples (Optional)</label>
                            <input type="number" id="max-samples" class="input"
                                   placeholder="Use all">
                            <small>Limit dataset size for testing</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render preview section
     * @returns {string}
     */
    renderPreview() {
        return `
            <div class="card">
                <div class="spell-section">
                    <h3>👁️ Dataset Preview</h3>

                    <div style="display: flex; gap: var(--space-md); margin-bottom: var(--space-md);">
                        <button class="btn btn-secondary" id="preview-raw-btn" style="flex: 1;">
                            🔍 Preview Raw
                        </button>
                        <button class="btn btn-secondary" id="preview-formatted-btn" style="flex: 1;">
                            ✨ Preview Formatted
                        </button>
                    </div>

                    <div id="preview-container" style="display: none;">
                        <div id="preview-content"></div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render action buttons
     * @returns {string}
     */
    renderActions() {
        return `
            <div class="form-actions">
                <button class="btn btn-secondary" data-action="back">
                    ← Back to Dashboard
                </button>
                <button class="btn btn-primary btn-lg" data-action="next">
                    Next: Training Config →
                </button>
            </div>
        `;
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Source type switching
        const sourceType = document.getElementById('dataset-source-type');
        if (sourceType) {
            sourceType.addEventListener('change', (e) => {
                this.switchSourceType(e.target.value);
            });
        }

        // Load dataset button
        const loadBtn = document.getElementById('load-dataset-btn');
        if (loadBtn) {
            loadBtn.addEventListener('click', () => this.loadDataset());
        }

        // Preview buttons
        const rawBtn = document.getElementById('preview-raw-btn');
        if (rawBtn) {
            rawBtn.addEventListener('click', () => this.previewRaw());
        }

        const formattedBtn = document.getElementById('preview-formatted-btn');
        if (formattedBtn) {
            formattedBtn.addEventListener('click', () => this.previewFormatted());
        }

        // Action buttons
        document.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleAction(action);
            });
        });

        // Subscribe to mode changes
        this.trainingModeManager.subscribe(() => {
            this.updateRejectedColumnVisibility();
        });
    }

    /**
     * Switch source type
     * @param {string} type - Source type
     */
    switchSourceType(type) {
        // Hide all source configs
        document.querySelectorAll('.source-config').forEach(el => {
            el.style.display = 'none';
        });

        // Show selected config
        const configMap = {
            'huggingface': 'hf-source-config',
            'upload': 'upload-source-config',
            'local': 'local-source-config'
        };

        const configId = configMap[type];
        if (configId) {
            const config = document.getElementById(configId);
            if (config) config.style.display = 'block';
        }
    }

    /**
     * Update rejected column visibility
     */
    updateRejectedColumnVisibility() {
        const group = document.getElementById('rejected-column-group');
        if (group) {
            group.style.display = this.trainingModeManager.getMode() === 'orpo' ? 'block' : 'none';
        }
    }

    /**
     * Load dataset
     */
    async loadDataset() {
        console.log('Load dataset (mock implementation)');

        // Mock columns
        const mockColumns = ['prompt', 'chosen', 'rejected', 'system'];

        // Populate column selectors
        this.populateColumnSelectors(mockColumns);

        // Show column mapping section
        const section = document.getElementById('column-mapping-section');
        if (section) section.style.display = 'block';

        // Update available columns display
        const availableEl = document.getElementById('available-columns');
        if (availableEl) {
            availableEl.innerHTML = mockColumns.map(col =>
                `<span class="badge badge-secondary" style="margin: 2px;">${col}</span>`
            ).join('');
        }

        window.dispatchEvent(new CustomEvent('toast', {
            detail: { message: 'Dataset loaded successfully', type: 'success' }
        }));
    }

    /**
     * Populate column selectors
     * @param {array} columns - Available columns
     */
    populateColumnSelectors(columns) {
        const selectors = [
            'map-prompt',
            'map-chosen',
            'map-rejected',
            'map-system',
            'map-reasoning'
        ];

        selectors.forEach(id => {
            const select = document.getElementById(id);
            if (select) {
                // Keep first option
                const firstOption = select.querySelector('option:first-child');
                select.innerHTML = firstOption ? firstOption.outerHTML : '<option value="">-- Select --</option>';

                // Add columns
                columns.forEach(col => {
                    const option = document.createElement('option');
                    option.value = col;
                    option.textContent = col;

                    // Auto-select if name matches
                    if (id.includes(col)) {
                        option.selected = true;
                    }

                    select.appendChild(option);
                });
            }
        });
    }

    /**
     * Preview raw data
     */
    async previewRaw() {
        const container = document.getElementById('preview-container');
        const content = document.getElementById('preview-content');

        if (!container || !content) return;

        // Mock data
        const mockData = [
            {
                prompt: 'What is the capital of France?',
                chosen: 'The capital of France is Paris.',
                rejected: 'I think it might be London.',
                system: 'You are a helpful assistant.'
            }
        ];

        content.innerHTML = `
            <div style="background: var(--surface-1); padding: var(--space-md); border-radius: var(--radius-md); border: 1px solid var(--border-light);">
                <pre style="margin: 0; white-space: pre-wrap; font-size: var(--text-sm);">${JSON.stringify(mockData, null, 2)}</pre>
            </div>
        `;

        container.style.display = 'block';
    }

    /**
     * Preview formatted data
     */
    async previewFormatted() {
        const container = document.getElementById('preview-container');
        const content = document.getElementById('preview-content');

        if (!container || !content) return;

        content.innerHTML = `
            <div style="background: var(--surface-1); padding: var(--space-lg); border-radius: var(--radius-md); border: 1px solid var(--border-light);">
                <div style="margin-bottom: var(--space-lg);">
                    <div style="font-weight: bold; color: var(--primary-purple); margin-bottom: var(--space-sm);">Prompt:</div>
                    <div style="background: var(--surface-2); padding: var(--space-md); border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: var(--text-sm);">
                        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\\n\\nWhat is the capital of France?<|eot_id|>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-md);">
                    <div>
                        <div style="font-weight: bold; color: var(--success); margin-bottom: var(--space-sm);">✓ Chosen:</div>
                        <div style="background: var(--success-bg); padding: var(--space-md); border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: var(--text-sm);">
                            <|start_header_id|>assistant<|end_header_id|>\\n\\nThe capital of France is Paris.<|eot_id|>
                        </div>
                    </div>
                    <div>
                        <div style="font-weight: bold; color: var(--danger); margin-bottom: var(--space-sm);">✗ Rejected:</div>
                        <div style="background: var(--danger-bg); padding: var(--space-md); border-radius: var(--radius-sm); font-family: var(--font-mono); font-size: var(--text-sm);">
                            <|start_header_id|>assistant<|end_header_id|>\\n\\nI think it might be London.<|eot_id|>
                        </div>
                    </div>
                </div>
            </div>
        `;

        container.style.display = 'block';
    }

    /**
     * Handle action button clicks
     * @param {string} action - Action name
     */
    handleAction(action) {
        switch (action) {
            case 'back':
                window.dispatchEvent(new CustomEvent('navigate', {
                    detail: { view: 'dashboard' }
                }));
                break;
            case 'next':
                window.dispatchEvent(new CustomEvent('navigate', {
                    detail: { view: 'training' }
                }));
                break;
        }
    }
}
