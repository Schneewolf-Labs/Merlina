// Dataset Module - Multi-Dataset configuration and preview

import { MerlinaAPI } from './api.js';
import { Toast, LoadingManager } from './ui.js';
import { Validator } from './validation.js';
import { sanitizeHTML } from './validation.js';

/**
 * Single Dataset Configuration - manages one dataset's config
 */
class SingleDatasetUI {
    constructor(index, manager) {
        this.index = index;
        this.manager = manager;
        this.uploadedDatasetId = null;
        this.datasetColumns = null;
        this.datasetSamples = null;
        this.element = null;
    }

    /**
     * Render the dataset card HTML
     */
    render() {
        const card = document.createElement('div');
        card.className = 'dataset-card';
        card.id = `dataset-card-${this.index}`;
        card.innerHTML = `
            <div class="dataset-card-header">
                <h3>📁 Dataset ${this.index + 1}</h3>
                <button type="button" class="remove-dataset-btn" data-index="${this.index}" ${this.index === 0 ? 'style="display: none;"' : ''}>
                    ✕ Remove
                </button>
            </div>

            <!-- Dataset Source Selection -->
            <div class="spell-section">
                <h4>🔮 Dataset Source</h4>
                <div class="form-group">
                    <label>Source Type</label>
                    <select id="dataset-source-type-${this.index}" class="magic-select dataset-source-type">
                        <option value="huggingface">HuggingFace Dataset</option>
                        <option value="upload">Upload File</option>
                        <option value="local_file">Local File Path</option>
                    </select>
                </div>

                <!-- HuggingFace Source -->
                <div id="hf-source-config-${this.index}" class="source-config">
                    <div class="form-group">
                        <label>HuggingFace Repository ID</label>
                        <input type="text" id="hf-repo-id-${this.index}" value="schneewolflabs/Athanor-DPO" class="magic-input">
                        <small style="color: #888; font-size: 0.85em;">Example: username/dataset-name</small>
                    </div>
                    <div class="form-group">
                        <label>Split</label>
                        <input type="text" id="hf-split-${this.index}" value="train" class="magic-input">
                    </div>
                </div>

                <!-- Upload Source -->
                <div id="upload-source-config-${this.index}" class="source-config" style="display: none;">
                    <div class="form-group">
                        <label>Upload Dataset File</label>
                        <input type="file" id="dataset-file-${this.index}" accept=".json,.jsonl,.csv,.parquet" class="magic-input">
                        <small style="color: #888; font-size: 0.85em;">Supported: JSON, JSONL, CSV, Parquet</small>
                    </div>
                    <div id="upload-status-${this.index}"></div>
                    <button type="button" id="upload-button-${this.index}" class="action-button upload-button" data-index="${this.index}" style="margin-top: 10px;">
                        📤 Upload Dataset
                    </button>
                </div>

                <!-- Local File Source -->
                <div id="local-source-config-${this.index}" class="source-config" style="display: none;">
                    <div class="form-group">
                        <label>File Path</label>
                        <input type="text" id="local-file-path-${this.index}" placeholder="/path/to/dataset.json" class="magic-input">
                    </div>
                    <div class="form-group">
                        <label>File Format</label>
                        <select id="local-file-format-${this.index}" class="magic-select">
                            <option value="">Auto-detect</option>
                            <option value="json">JSON</option>
                            <option value="csv">CSV</option>
                            <option value="parquet">Parquet</option>
                        </select>
                    </div>
                </div>
            </div>

            <!-- Column Mapping -->
            <div class="spell-section">
                <h4>🔗 Column Mapping</h4>
                <button type="button" id="inspect-columns-button-${this.index}" class="action-button inspect-columns-button" data-index="${this.index}" style="width: 100%; margin-bottom: 15px;">
                    🔍 Inspect Dataset Columns
                </button>

                <div id="column-mapping-config-${this.index}" style="display: none;">
                    <div style="background: #f5f7fa; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                        <div style="font-weight: bold; color: var(--primary-purple); margin-bottom: 10px;">
                            Available Columns: <span id="available-columns-${this.index}"></span>
                        </div>
                        <div style="font-size: 0.85em; color: #666;">
                            Map your dataset columns to the standard format
                        </div>
                    </div>

                    <!-- Messages Format Detection -->
                    <div id="messages-format-notice-${this.index}" style="display: none; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; border-radius: 10px; margin-bottom: 15px; color: white;">
                        <div style="font-weight: bold; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                            ✨ Messages Format Detected!
                        </div>
                        <div style="font-size: 0.9em; margin-bottom: 12px; opacity: 0.95;">
                            This dataset uses the common "messages" format. Enable auto-convert to transform it into standard format.
                        </div>
                        <label style="display: flex; align-items: center; gap: 10px; cursor: pointer; font-size: 0.95em;">
                            <input type="checkbox" id="convert-messages-checkbox-${this.index}" checked style="width: 18px; height: 18px; cursor: pointer;">
                            <span>Auto-convert messages format</span>
                        </label>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Prompt Column</label>
                            <select id="map-prompt-${this.index}" class="magic-select">
                                <option value="">-- Select Column --</option>
                            </select>
                            <small style="color: #888; font-size: 0.85em;">Required: User input/question</small>
                        </div>
                        <div class="form-group">
                            <label>Chosen Column</label>
                            <select id="map-chosen-${this.index}" class="magic-select">
                                <option value="">-- Select Column --</option>
                            </select>
                            <small style="color: #888; font-size: 0.85em;">Required: Preferred response</small>
                        </div>
                    </div>

                    <div class="form-row">
                        <div class="form-group">
                            <label>Rejected Column</label>
                            <select id="map-rejected-${this.index}" class="magic-select">
                                <option value="">-- None (SFT mode) --</option>
                            </select>
                            <small style="color: #888; font-size: 0.85em;">Required for ORPO, optional for SFT</small>
                        </div>
                        <div class="form-group">
                            <label>System Column (Optional)</label>
                            <select id="map-system-${this.index}" class="magic-select">
                                <option value="">-- None --</option>
                            </select>
                        </div>
                    </div>

                    <div class="form-group">
                        <label>Reasoning Column (Optional)</label>
                        <select id="map-reasoning-${this.index}" class="magic-select">
                            <option value="">-- None --</option>
                        </select>
                        <small style="color: #888; font-size: 0.85em;">For Qwen3 thinking mode</small>
                    </div>

                    <!-- Sample Preview -->
                    <div id="column-sample-preview-${this.index}" style="margin-top: 15px; display: none;">
                        <div style="font-weight: bold; color: var(--primary-purple); margin-bottom: 10px;">
                            Sample Data (First Row):
                        </div>
                        <div style="background: white; padding: 15px; border-radius: 10px; border: 1px solid var(--light-purple); max-height: 200px; overflow-y: auto;">
                            <pre id="column-sample-content-${this.index}" style="margin: 0; font-size: 0.85em; white-space: pre-wrap;"></pre>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Dataset Format -->
            <div class="spell-section">
                <h4>📝 Dataset Format</h4>
                <div class="form-group">
                    <label>Format Type</label>
                    <select id="dataset-format-type-${this.index}" class="magic-select dataset-format-type">
                        <option value="tokenizer">Tokenizer (Model Native)</option>
                        <option value="chatml">ChatML</option>
                        <option value="llama3">Llama 3</option>
                        <option value="mistral">Mistral Instruct</option>
                        <option value="qwen3">Qwen 3 (with thinking)</option>
                        <option value="custom">Custom Template</option>
                    </select>
                </div>

                <!-- Qwen3 Config -->
                <div id="qwen3-format-config-${this.index}" style="display: none; margin-top: 15px;">
                    <label class="magic-checkbox">
                        <input type="checkbox" id="enable-thinking-${this.index}" checked>
                        <span>Enable Thinking Mode 🧠</span>
                    </label>
                </div>

                <!-- Custom Template Config -->
                <div id="custom-format-config-${this.index}" style="display: none; margin-top: 15px;">
                    <div class="form-group">
                        <label>Prompt Template</label>
                        <textarea id="custom-prompt-template-${this.index}" class="magic-input" rows="2">{system}\n\n{prompt}</textarea>
                    </div>
                    <div class="form-group">
                        <label>Chosen Template</label>
                        <input type="text" id="custom-chosen-template-${this.index}" value="{chosen}" class="magic-input">
                    </div>
                    <div class="form-group">
                        <label>Rejected Template</label>
                        <input type="text" id="custom-rejected-template-${this.index}" value="{rejected}" class="magic-input">
                    </div>
                </div>
            </div>

            <!-- Per-Dataset Max Samples -->
            <div class="spell-section">
                <div class="form-group">
                    <label>Max Samples from this Dataset (Optional)</label>
                    <input type="number" id="max-samples-${this.index}" placeholder="Use all" class="magic-input">
                    <small style="color: #888; font-size: 0.85em;">Limit samples from this specific dataset</small>
                </div>
            </div>
        `;

        this.element = card;
        this.setupEventListeners();
        return card;
    }

    /**
     * Setup event listeners for this dataset card
     */
    setupEventListeners() {
        // Source type change
        const sourceTypeSelect = this.element.querySelector(`#dataset-source-type-${this.index}`);
        sourceTypeSelect.addEventListener('change', (e) => this.handleSourceTypeChange(e));

        // Format type change
        const formatTypeSelect = this.element.querySelector(`#dataset-format-type-${this.index}`);
        formatTypeSelect.addEventListener('change', (e) => this.handleFormatTypeChange(e));

        // Upload button
        const uploadBtn = this.element.querySelector(`#upload-button-${this.index}`);
        uploadBtn.addEventListener('click', () => this.handleUpload());

        // Inspect columns button
        const inspectBtn = this.element.querySelector(`#inspect-columns-button-${this.index}`);
        inspectBtn.addEventListener('click', () => this.handleInspectColumns());

        // Remove button
        const removeBtn = this.element.querySelector('.remove-dataset-btn');
        removeBtn.addEventListener('click', () => this.manager.removeDataset(this.index));
    }

    /**
     * Handle source type change
     */
    handleSourceTypeChange(e) {
        const sourceType = e.target.value;

        // Hide all configs
        this.element.querySelector(`#hf-source-config-${this.index}`).style.display = 'none';
        this.element.querySelector(`#upload-source-config-${this.index}`).style.display = 'none';
        this.element.querySelector(`#local-source-config-${this.index}`).style.display = 'none';

        // Show selected config
        if (sourceType === 'huggingface') {
            this.element.querySelector(`#hf-source-config-${this.index}`).style.display = 'block';
        } else if (sourceType === 'upload') {
            this.element.querySelector(`#upload-source-config-${this.index}`).style.display = 'block';
        } else if (sourceType === 'local_file') {
            this.element.querySelector(`#local-source-config-${this.index}`).style.display = 'block';
        }
    }

    /**
     * Handle format type change
     */
    handleFormatTypeChange(e) {
        const formatType = e.target.value;

        const customConfig = this.element.querySelector(`#custom-format-config-${this.index}`);
        const qwen3Config = this.element.querySelector(`#qwen3-format-config-${this.index}`);

        if (customConfig) {
            customConfig.style.display = formatType === 'custom' ? 'block' : 'none';
        }

        if (qwen3Config) {
            qwen3Config.style.display = formatType === 'qwen3' ? 'block' : 'none';
        }
    }

    /**
     * Handle dataset upload
     */
    async handleUpload() {
        const fileInput = this.element.querySelector(`#dataset-file-${this.index}`);
        const file = fileInput.files[0];

        if (!file) {
            this.manager.toast.error('Please select a file to upload');
            return;
        }

        const uploadBtn = this.element.querySelector(`#upload-button-${this.index}`);
        const uploadStatus = this.element.querySelector(`#upload-status-${this.index}`);

        try {
            LoadingManager.show(uploadBtn, '⏳ Uploading...');

            const data = await MerlinaAPI.uploadDataset(file);
            this.uploadedDatasetId = data.dataset_id;

            uploadStatus.innerHTML = `
                <div class="success-message">
                    ✅ Uploaded: ${sanitizeHTML(data.filename)} (ID: ${sanitizeHTML(data.dataset_id)})
                </div>
            `;

            this.manager.toast.success('Dataset uploaded successfully!');
        } catch (error) {
            console.error('Upload failed:', error);
            this.manager.toast.error(`Upload failed: ${error.message}`);

            uploadStatus.innerHTML = `
                <div class="error-message">
                    ❌ Upload failed: ${sanitizeHTML(error.message)}
                </div>
            `;
        } finally {
            LoadingManager.hide(uploadBtn);
        }
    }

    /**
     * Handle column inspection
     */
    async handleInspectColumns() {
        const inspectBtn = this.element.querySelector(`#inspect-columns-button-${this.index}`);
        const columnMappingConfig = this.element.querySelector(`#column-mapping-config-${this.index}`);

        try {
            LoadingManager.show(inspectBtn, '⏳ Loading columns...');

            const datasetConfig = this.getDatasetSourceConfig();
            // Use dataset_index=0 since getDatasetSourceConfig returns a single-dataset config
            const data = await MerlinaAPI.getDatasetColumns(datasetConfig, 0);

            this.datasetColumns = data.columns;
            this.datasetSamples = data.samples;

            // Populate column mapping dropdowns
            this.populateColumnMappings(data.columns);

            // Show available columns
            const availableColumnsEl = this.element.querySelector(`#available-columns-${this.index}`);
            if (availableColumnsEl) {
                availableColumnsEl.textContent = data.columns.join(', ');
            }

            // Check for messages format
            const messagesFormatNotice = this.element.querySelector(`#messages-format-notice-${this.index}`);
            if (messagesFormatNotice) {
                messagesFormatNotice.style.display = data.columns.includes('messages') ? 'block' : 'none';
            }

            // Show sample data
            if (data.samples && data.samples.length > 0) {
                const samplePreview = this.element.querySelector(`#column-sample-preview-${this.index}`);
                const sampleContent = this.element.querySelector(`#column-sample-content-${this.index}`);

                if (sampleContent) {
                    sampleContent.textContent = JSON.stringify(data.samples[0], null, 2);
                }

                if (samplePreview) {
                    samplePreview.style.display = 'block';
                }
            }

            // Show column mapping UI
            if (columnMappingConfig) {
                columnMappingConfig.style.display = 'block';
            }

            this.manager.toast.success(`Found ${data.columns.length} columns in dataset`);
        } catch (error) {
            console.error('Failed to inspect columns:', error);
            this.manager.toast.error(`Failed to inspect columns: ${error.message}`);
        } finally {
            LoadingManager.hide(inspectBtn);
        }
    }

    /**
     * Populate column mapping dropdowns
     */
    populateColumnMappings(columns) {
        const selects = [
            `map-prompt-${this.index}`,
            `map-chosen-${this.index}`,
            `map-rejected-${this.index}`,
            `map-system-${this.index}`,
            `map-reasoning-${this.index}`
        ];

        selects.forEach(selectId => {
            const select = this.element.querySelector(`#${selectId}`);
            if (!select) return;

            // Clear existing options except first
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

            // Auto-select if column name matches
            const targetColumn = selectId.replace(`map-`, '').replace(`-${this.index}`, '');
            if (columns.includes(targetColumn)) {
                select.value = targetColumn;
            }
        });
    }

    /**
     * Get dataset source config (for column inspection)
     */
    getDatasetSourceConfig() {
        const sourceType = this.element.querySelector(`#dataset-source-type-${this.index}`).value;
        let source = { source_type: sourceType };

        if (sourceType === 'huggingface') {
            source.repo_id = this.element.querySelector(`#hf-repo-id-${this.index}`).value;
            source.split = this.element.querySelector(`#hf-split-${this.index}`).value;
        } else if (sourceType === 'upload') {
            if (!this.uploadedDatasetId) {
                throw new Error('Please upload a dataset first');
            }
            source.dataset_id = this.uploadedDatasetId;
        } else if (sourceType === 'local_file') {
            source.file_path = this.element.querySelector(`#local-file-path-${this.index}`).value;
            const format = this.element.querySelector(`#local-file-format-${this.index}`).value;
            if (format) source.file_format = format;
        }

        return {
            source: source,
            format: { format_type: 'chatml' },
            test_size: 0.1
        };
    }

    /**
     * Get full single dataset config
     */
    getConfig() {
        const sourceType = this.element.querySelector(`#dataset-source-type-${this.index}`).value;
        const formatType = this.element.querySelector(`#dataset-format-type-${this.index}`).value;

        // Build source config
        let source = { source_type: sourceType };

        if (sourceType === 'huggingface') {
            source.repo_id = this.element.querySelector(`#hf-repo-id-${this.index}`).value;
            source.split = this.element.querySelector(`#hf-split-${this.index}`).value;
        } else if (sourceType === 'upload') {
            if (!this.uploadedDatasetId) {
                throw new Error(`Dataset ${this.index + 1}: Please upload a dataset first`);
            }
            source.dataset_id = this.uploadedDatasetId;
        } else if (sourceType === 'local_file') {
            source.file_path = this.element.querySelector(`#local-file-path-${this.index}`).value;
            const format = this.element.querySelector(`#local-file-format-${this.index}`).value;
            if (format) source.file_format = format;
        }

        // Build format config
        let format = { format_type: formatType };

        if (formatType === 'custom') {
            format.custom_templates = {
                prompt_template: this.element.querySelector(`#custom-prompt-template-${this.index}`)?.value || '',
                chosen_template: this.element.querySelector(`#custom-chosen-template-${this.index}`)?.value || '',
                rejected_template: this.element.querySelector(`#custom-rejected-template-${this.index}`)?.value || ''
            };
        }

        if (formatType === 'qwen3') {
            const enableThinking = this.element.querySelector(`#enable-thinking-${this.index}`);
            format.enable_thinking = enableThinking ? enableThinking.checked : true;
        }

        // Build config object
        const config = {
            source: source,
            format: format,
            name: `Dataset ${this.index + 1}`
        };

        // Add column mapping if configured
        const columnMapping = this.getColumnMapping();
        if (columnMapping && Object.keys(columnMapping).length > 0) {
            config.column_mapping = columnMapping;
        }

        // Add messages format conversion setting
        const convertMessagesCheckbox = this.element.querySelector(`#convert-messages-checkbox-${this.index}`);
        config.convert_messages_format = convertMessagesCheckbox ? convertMessagesCheckbox.checked : true;

        // Add per-dataset max samples
        const maxSamples = this.element.querySelector(`#max-samples-${this.index}`)?.value;
        if (maxSamples) {
            config.max_samples = parseInt(maxSamples);
        }

        return config;
    }

    /**
     * Get column mapping from UI
     */
    getColumnMapping() {
        const mapping = {};

        const promptCol = this.element.querySelector(`#map-prompt-${this.index}`)?.value;
        const chosenCol = this.element.querySelector(`#map-chosen-${this.index}`)?.value;
        const rejectedCol = this.element.querySelector(`#map-rejected-${this.index}`)?.value;
        const systemCol = this.element.querySelector(`#map-system-${this.index}`)?.value;
        const reasoningCol = this.element.querySelector(`#map-reasoning-${this.index}`)?.value;

        if (promptCol) mapping[promptCol] = 'prompt';
        if (chosenCol) mapping[chosenCol] = 'chosen';
        if (rejectedCol) mapping[rejectedCol] = 'rejected';
        if (systemCol) mapping[systemCol] = 'system';
        if (reasoningCol) mapping[reasoningCol] = 'reasoning';

        return mapping;
    }
}


/**
 * Multi-Dataset Manager - manages multiple datasets
 */
class DatasetManager {
    constructor() {
        this.datasets = [];
        this.toast = new Toast();
        this.container = null;

        this.init();
    }

    /**
     * Initialize the dataset manager
     */
    init() {
        this.container = document.getElementById('datasets-container');
        if (!this.container) {
            console.warn('Datasets container not found');
            return;
        }

        // Add first dataset
        this.addDataset();

        // Setup add button
        const addBtn = document.getElementById('add-dataset-button');
        if (addBtn) {
            addBtn.addEventListener('click', () => this.addDataset());
        }

        // Setup preview buttons
        this.setupPreviewButtons();
    }

    /**
     * Add a new dataset
     */
    addDataset() {
        const index = this.datasets.length;
        const datasetUI = new SingleDatasetUI(index, this);
        const card = datasetUI.render();

        this.datasets.push(datasetUI);
        this.container.appendChild(card);

        // Update preview select
        this.updatePreviewSelect();

        // Show remove buttons if more than one dataset
        this.updateRemoveButtons();
    }

    /**
     * Remove a dataset
     */
    removeDataset(index) {
        if (this.datasets.length <= 1) {
            this.toast.error('Cannot remove the last dataset');
            return;
        }

        // Remove from DOM
        const card = document.getElementById(`dataset-card-${index}`);
        if (card) {
            card.remove();
        }

        // Remove from array
        this.datasets.splice(index, 1);

        // Re-render all datasets with correct indices
        this.reindexDatasets();

        // Update preview select
        this.updatePreviewSelect();

        this.toast.success('Dataset removed');
    }

    /**
     * Re-index datasets after removal
     */
    reindexDatasets() {
        // Clear container
        this.container.innerHTML = '';

        // Re-render all datasets with new indices
        const oldDatasets = [...this.datasets];
        this.datasets = [];

        oldDatasets.forEach((oldDataset, newIndex) => {
            const newDatasetUI = new SingleDatasetUI(newIndex, this);
            newDatasetUI.uploadedDatasetId = oldDataset.uploadedDatasetId;
            newDatasetUI.datasetColumns = oldDataset.datasetColumns;
            newDatasetUI.datasetSamples = oldDataset.datasetSamples;

            const card = newDatasetUI.render();
            this.datasets.push(newDatasetUI);
            this.container.appendChild(card);

            // Restore values from old dataset
            this.restoreDatasetValues(oldDataset, newDatasetUI);
        });

        this.updateRemoveButtons();
    }

    /**
     * Restore values from old dataset to new one
     */
    restoreDatasetValues(oldDataset, newDataset) {
        // This would need to copy all form values
        // For simplicity, we'll just note that users should re-configure
        // A more complete implementation would serialize/deserialize all values
    }

    /**
     * Update visibility of remove buttons
     */
    updateRemoveButtons() {
        const showRemove = this.datasets.length > 1;
        this.datasets.forEach((dataset, index) => {
            const removeBtn = document.querySelector(`#dataset-card-${index} .remove-dataset-btn`);
            if (removeBtn) {
                removeBtn.style.display = showRemove ? 'block' : 'none';
            }
        });
    }

    /**
     * Update the preview dataset select dropdown
     */
    updatePreviewSelect() {
        const select = document.getElementById('preview-dataset-select');
        if (!select) return;

        // Clear options
        select.innerHTML = '';

        // Add option for each dataset
        this.datasets.forEach((dataset, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `Dataset ${index + 1}`;
            select.appendChild(option);
        });
    }

    /**
     * Setup preview buttons
     */
    setupPreviewButtons() {
        const previewBtn = document.getElementById('preview-dataset-button');
        if (previewBtn) {
            previewBtn.addEventListener('click', () => this.handlePreview());
        }

        const previewFormattedBtn = document.getElementById('preview-formatted-button');
        if (previewFormattedBtn) {
            previewFormattedBtn.addEventListener('click', () => this.handlePreviewFormatted());
        }
    }

    /**
     * Handle dataset preview
     */
    async handlePreview() {
        const previewBtn = document.getElementById('preview-dataset-button');
        const datasetIndex = parseInt(document.getElementById('preview-dataset-select')?.value || '0');

        try {
            LoadingManager.show(previewBtn, '⏳ Loading...');

            const datasetConfig = this.getDatasetConfig(true);  // forPreview=true
            const data = await MerlinaAPI.previewDataset(datasetConfig, datasetIndex);

            // Display preview
            const previewDiv = document.getElementById('dataset-preview');
            const previewContent = document.getElementById('dataset-preview-content');
            const formattedPreview = document.getElementById('formatted-preview');

            if (previewContent) {
                previewContent.textContent = JSON.stringify(data.samples, null, 2);
            }

            if (previewDiv) {
                previewDiv.style.display = 'block';
            }

            if (formattedPreview) {
                formattedPreview.style.display = 'none';
            }

            this.toast.success(`Loaded ${data.num_samples} sample(s) from Dataset ${datasetIndex + 1}`);
        } catch (error) {
            console.error('Preview failed:', error);
            this.toast.error(`Preview failed: ${error.message}`);
        } finally {
            LoadingManager.hide(previewBtn);
        }
    }

    /**
     * Handle formatted preview
     */
    async handlePreviewFormatted() {
        const previewBtn = document.getElementById('preview-formatted-button');
        const datasetIndex = parseInt(document.getElementById('preview-dataset-select')?.value || '0');

        try {
            LoadingManager.show(previewBtn, '⏳ Loading...');

            const datasetConfig = this.getDatasetConfig(true);  // forPreview=true
            const data = await MerlinaAPI.previewFormattedDataset(datasetConfig, datasetIndex);

            if (!data.samples || data.samples.length === 0) {
                throw new Error('No samples returned');
            }

            // Get first sample
            const sample = data.samples[0];

            // Display formatted preview
            const formattedPreview = document.getElementById('formatted-preview');
            const rawPreview = document.getElementById('dataset-preview');

            const promptEl = document.getElementById('formatted-prompt');
            const chosenEl = document.getElementById('formatted-chosen');
            const rejectedEl = document.getElementById('formatted-rejected');
            const formatTypeEl = document.getElementById('format-type-display');

            if (promptEl) promptEl.textContent = sample.prompt;
            if (chosenEl) chosenEl.textContent = sample.chosen;
            if (rejectedEl) rejectedEl.textContent = sample.rejected || '(N/A for SFT mode)';

            // Get format type from the selected dataset
            const formatType = this.datasets[datasetIndex]?.element?.querySelector(`#dataset-format-type-${datasetIndex}`)?.value || 'tokenizer';
            const formatNames = {
                'tokenizer': 'Tokenizer (Model Native)',
                'chatml': 'ChatML',
                'llama3': 'Llama 3',
                'mistral': 'Mistral Instruct',
                'qwen3': 'Qwen 3',
                'custom': 'Custom Template'
            };

            if (formatTypeEl) {
                formatTypeEl.textContent = formatNames[formatType] || formatType;
            }

            if (formattedPreview) {
                formattedPreview.style.display = 'block';
            }

            if (rawPreview) {
                rawPreview.style.display = 'none';
            }

            this.toast.success(`Preview formatted from Dataset ${datasetIndex + 1}`);
        } catch (error) {
            console.error('Formatted preview failed:', error);
            this.toast.error(`Formatted preview failed: ${error.message}`);
        } finally {
            LoadingManager.hide(previewBtn);
        }
    }

    /**
     * Get full dataset config (for API calls)
     * @param {boolean} forPreview - If true, skip strict validation for previews
     */
    getDatasetConfig(forPreview = false) {
        // Get training mode
        const trainingMode = forPreview ? 'sft' : (document.getElementById('training-mode')?.value || 'orpo');

        // Get global options
        const testSize = parseFloat(document.getElementById('test-size')?.value || '0.01');
        const maxSamplesEl = document.getElementById('max-samples');
        const maxSamples = maxSamplesEl?.value ? parseInt(maxSamplesEl.value) : null;

        // Get model name for tokenizer format
        const baseModel = document.getElementById('base-model')?.value?.trim();

        // Build datasets array
        const datasets = this.datasets.map(dataset => dataset.getConfig());

        // Build config
        const config = {
            datasets: datasets,
            test_size: testSize,
            training_mode: trainingMode
        };

        if (maxSamples) {
            config.max_samples = maxSamples;
        }

        if (baseModel) {
            config.model_name = baseModel;
        }

        // Validate if not for preview
        if (!forPreview) {
            const errors = [];
            datasets.forEach((ds, idx) => {
                // Basic validation
                if (ds.source.source_type === 'huggingface' && !ds.source.repo_id) {
                    errors.push(`Dataset ${idx + 1}: HuggingFace repo ID is required`);
                }
                if (ds.source.source_type === 'upload' && !ds.source.dataset_id) {
                    errors.push(`Dataset ${idx + 1}: Please upload a dataset file`);
                }
                if (ds.source.source_type === 'local_file' && !ds.source.file_path) {
                    errors.push(`Dataset ${idx + 1}: File path is required`);
                }
            });

            if (errors.length > 0) {
                throw new Error(errors.join('; '));
            }
        }

        return config;
    }

    /**
     * Legacy method for backward compatibility
     */
    getColumnMapping() {
        // Return mapping from first dataset for backward compatibility
        if (this.datasets.length > 0) {
            return this.datasets[0].getColumnMapping();
        }
        return {};
    }
}


// For backward compatibility, also export as single DatasetManager
export { DatasetManager, SingleDatasetUI };
