// Dataset Module - Dataset configuration and preview

import { MerlinaAPI } from './api.js';
import { Toast, LoadingManager } from './ui.js';
import { Validator } from './validation.js';
import { sanitizeHTML } from './validation.js';

/**
 * Dataset Manager - handles dataset operations
 */
class DatasetManager {
    constructor() {
        this.uploadedDatasetId = null;
        this.datasetColumns = null;
        this.datasetSamples = null;
        this.additionalDatasets = []; // Array of additional dataset configs
        this.additionalDatasetCounter = 0;

        this.toast = new Toast();
        this.setupEventListeners();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Source type change
        const sourceTypeSelect = document.getElementById('dataset-source-type');
        if (sourceTypeSelect) {
            sourceTypeSelect.addEventListener('change', (e) => this.handleSourceTypeChange(e));
        }

        // Format type change
        const formatTypeSelect = document.getElementById('dataset-format-type');
        if (formatTypeSelect) {
            formatTypeSelect.addEventListener('change', (e) => this.handleFormatTypeChange(e));
        }

        // Upload button
        const uploadBtn = document.getElementById('upload-button');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => this.handleUpload());
        }

        // Column inspection
        const inspectBtn = document.getElementById('inspect-columns-button');
        if (inspectBtn) {
            inspectBtn.addEventListener('click', () => this.handleInspectColumns());
        }

        // Preview buttons
        const previewBtn = document.getElementById('preview-dataset-button');
        if (previewBtn) {
            previewBtn.addEventListener('click', () => this.handlePreview());
        }

        const previewFormattedBtn = document.getElementById('preview-formatted-button');
        if (previewFormattedBtn) {
            previewFormattedBtn.addEventListener('click', () => this.handlePreviewFormatted());
        }

        // Add additional dataset button
        const addDatasetBtn = document.getElementById('add-dataset-button');
        if (addDatasetBtn) {
            addDatasetBtn.addEventListener('click', () => this.addAdditionalDataset());
        }
    }

    /**
     * Handle source type change
     */
    handleSourceTypeChange(e) {
        const sourceType = e.target.value;

        // Hide all configs
        document.getElementById('hf-source-config').style.display = 'none';
        document.getElementById('upload-source-config').style.display = 'none';
        document.getElementById('local-source-config').style.display = 'none';

        // Show selected config
        if (sourceType === 'huggingface') {
            document.getElementById('hf-source-config').style.display = 'block';
        } else if (sourceType === 'upload') {
            document.getElementById('upload-source-config').style.display = 'block';
        } else if (sourceType === 'local_file') {
            document.getElementById('local-source-config').style.display = 'block';
        }
    }

    /**
     * Handle format type change
     */
    handleFormatTypeChange(e) {
        const formatType = e.target.value;

        const customConfig = document.getElementById('custom-format-config');
        const qwen3Config = document.getElementById('qwen3-format-config');

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
        const fileInput = document.getElementById('dataset-file');
        const file = fileInput.files[0];

        if (!file) {
            this.toast.error('Please select a file to upload');
            return;
        }

        const uploadBtn = document.getElementById('upload-button');
        const uploadStatus = document.getElementById('upload-status');

        try {
            LoadingManager.show(uploadBtn, '⏳ Uploading...');

            const data = await MerlinaAPI.uploadDataset(file);
            this.uploadedDatasetId = data.dataset_id;
            // Also expose on window for ConfigManager compatibility
            window.uploadedDatasetId = data.dataset_id;

            uploadStatus.innerHTML = `
                <div class="success-message">
                    ✅ Uploaded: ${sanitizeHTML(data.filename)} (ID: ${sanitizeHTML(data.dataset_id)})
                </div>
            `;

            this.toast.success('Dataset uploaded successfully!');
        } catch (error) {
            console.error('Upload failed:', error);
            this.toast.error(`Upload failed: ${error.message}`);

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
        const inspectBtn = document.getElementById('inspect-columns-button');
        const columnMappingConfig = document.getElementById('column-mapping-config');

        try {
            LoadingManager.show(inspectBtn, '⏳ Loading columns...');

            const datasetConfig = this.getDatasetSourceConfig();
            const data = await MerlinaAPI.getDatasetColumns(datasetConfig);

            this.datasetColumns = data.columns;
            this.datasetSamples = data.samples;

            // Populate column mapping dropdowns
            this.populateColumnMappings(data.columns);

            // Show available columns
            const availableColumnsEl = document.getElementById('available-columns');
            if (availableColumnsEl) {
                availableColumnsEl.textContent = data.columns.join(', ');
            }

            // Check for messages format
            const messagesFormatNotice = document.getElementById('messages-format-notice');
            if (messagesFormatNotice) {
                if (data.columns.includes('messages')) {
                    messagesFormatNotice.style.display = 'block';
                } else {
                    messagesFormatNotice.style.display = 'none';
                }
            }

            // Show sample data
            if (data.samples && data.samples.length > 0) {
                const samplePreview = document.getElementById('column-sample-preview');
                const sampleContent = document.getElementById('column-sample-content');

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

            this.toast.success(`Found ${data.columns.length} columns in dataset`);
        } catch (error) {
            console.error('Failed to inspect columns:', error);
            this.toast.error(`Failed to inspect columns: ${error.message}`);
        } finally {
            LoadingManager.hide(inspectBtn);
        }
    }

    /**
     * Populate column mapping dropdowns
     */
    populateColumnMappings(columns) {
        const selects = [
            'map-prompt',
            'map-chosen',
            'map-rejected',
            'map-system',
            'map-reasoning'
        ];

        selects.forEach(selectId => {
            const select = document.getElementById(selectId);
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
            const targetColumn = selectId.replace('map-', '');
            if (columns.includes(targetColumn)) {
                select.value = targetColumn;
            }
        });
    }

    /**
     * Handle dataset preview
     */
    async handlePreview() {
        const previewBtn = document.getElementById('preview-dataset-button');

        try {
            LoadingManager.show(previewBtn, '⏳ Loading...');

            const datasetConfig = this.getDatasetConfig(true);  // forPreview=true
            const data = await MerlinaAPI.previewDataset(datasetConfig);

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

            this.toast.success(`Loaded ${data.num_samples} sample(s)`);
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

        try {
            LoadingManager.show(previewBtn, '⏳ Loading...');

            const datasetConfig = this.getDatasetConfig(true);  // forPreview=true
            const data = await MerlinaAPI.previewFormattedDataset(datasetConfig);

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
            if (rejectedEl) rejectedEl.textContent = sample.rejected;

            // Display format type
            const formatType = datasetConfig.format.format_type;
            const formatNames = {
                'tokenizer': 'Tokenizer (Model Native)',
                'chatml': 'ChatML',
                'llama3': 'Llama 3',
                'mistral': 'Mistral Instruct',
                'qwen3': `Qwen 3 (thinking ${datasetConfig.format.enable_thinking ? 'enabled' : 'disabled'})`,
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

            this.toast.success(`Preview formatted with ${formatNames[formatType]}`);
        } catch (error) {
            console.error('Formatted preview failed:', error);
            this.toast.error(`Formatted preview failed: ${error.message}`);
        } finally {
            LoadingManager.hide(previewBtn);
        }
    }

    /**
     * Get dataset source config (without format)
     */
    getDatasetSourceConfig() {
        const sourceType = document.getElementById('dataset-source-type').value;
        let source = { source_type: sourceType };

        if (sourceType === 'huggingface') {
            source.repo_id = document.getElementById('hf-repo-id').value;
            source.split = document.getElementById('hf-split').value;
        } else if (sourceType === 'upload') {
            if (!this.uploadedDatasetId) {
                throw new Error('Please upload a dataset first');
            }
            source.dataset_id = this.uploadedDatasetId;
        } else if (sourceType === 'local_file') {
            source.file_path = document.getElementById('local-file-path').value;
            const format = document.getElementById('local-file-format').value;
            if (format) source.file_format = format;
        }

        return {
            source: source,
            format: { format_type: 'chatml' }, // Dummy format
            test_size: 0.1
        };
    }

    /**
     * Get full dataset config (with format and column mapping)
     * @param {boolean} forPreview - If true, skip rejected column validation for previews
     */
    getDatasetConfig(forPreview = false) {
        const sourceType = document.getElementById('dataset-source-type').value;
        const formatType = document.getElementById('dataset-format-type').value;

        // Build source config
        let source = { source_type: sourceType };

        if (sourceType === 'huggingface') {
            source.repo_id = document.getElementById('hf-repo-id').value;
            source.split = document.getElementById('hf-split').value;
        } else if (sourceType === 'upload') {
            if (!this.uploadedDatasetId) {
                throw new Error('Please upload a dataset first');
            }
            source.dataset_id = this.uploadedDatasetId;
        } else if (sourceType === 'local_file') {
            source.file_path = document.getElementById('local-file-path').value;
            const format = document.getElementById('local-file-format').value;
            if (format) source.file_format = format;
        }

        // Build format config
        let format = { format_type: formatType };

        if (formatType === 'custom') {
            format.custom_templates = {
                prompt_template: document.getElementById('custom-prompt-template')?.value || '',
                chosen_template: document.getElementById('custom-chosen-template')?.value || '',
                rejected_template: document.getElementById('custom-rejected-template')?.value || ''
            };
        }

        if (formatType === 'qwen3') {
            const enableThinking = document.getElementById('enable-thinking');
            format.enable_thinking = enableThinking ? enableThinking.checked : true;
        }

        // Build full config
        const config = {
            source: source,
            format: format,
            test_size: parseFloat(document.getElementById('test-size').value)
        };

        // Add max samples if specified
        const maxSamples = document.getElementById('max-samples')?.value;
        if (maxSamples) {
            config.max_samples = parseInt(maxSamples);
        }

        // Add model name for tokenizer format
        const baseModel = document.getElementById('base-model')?.value?.trim();
        if (baseModel) {
            config.model_name = baseModel;
        }

        // Add column mapping if configured
        const columnMapping = this.getColumnMapping();
        if (columnMapping && Object.keys(columnMapping).length > 0) {
            config.column_mapping = columnMapping;
        }

        // Add messages format conversion setting
        const convertMessagesCheckbox = document.getElementById('convert-messages-checkbox');
        if (convertMessagesCheckbox) {
            config.convert_messages_format = convertMessagesCheckbox.checked;
        } else {
            config.convert_messages_format = true;  // Default to true
        }

        // Add additional datasets for concatenation
        const additionalConfigs = this.getAdditionalDatasetsConfig();
        if (additionalConfigs.length > 0) {
            config.additional_datasets = additionalConfigs;
        }

        // Get training mode for validation and include in config
        // For previews, use 'sft' mode to skip rejected column requirement
        const trainingMode = forPreview ? 'sft' : (document.getElementById('training-mode')?.value || 'orpo');
        config.training_mode = trainingMode;

        // Validate dataset config
        const errors = Validator.validateDatasetConfig(config, trainingMode);
        if (errors.length > 0) {
            throw new Error(errors.join('; '));
        }

        return config;
    }

    /**
     * Add an additional dataset entry to the UI
     */
    addAdditionalDataset() {
        const id = this.additionalDatasetCounter++;
        const listEl = document.getElementById('additional-datasets-list');
        if (!listEl) return;

        const entry = document.createElement('div');
        entry.className = 'additional-dataset-entry';
        entry.id = `additional-dataset-${id}`;
        entry.style.cssText = 'background: #f5f7fa; border: 1px solid var(--light-purple); border-radius: 10px; padding: 15px; margin-bottom: 10px; position: relative;';

        entry.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong style="color: var(--primary-purple);">Dataset #${id + 2}</strong>
                <button type="button" class="remove-dataset-btn" data-id="${id}"
                    style="background: var(--danger); color: white; border: none; border-radius: 5px; padding: 4px 10px; cursor: pointer; font-size: 0.85em;">
                    Remove
                </button>
            </div>
            <div class="form-group" style="margin-bottom: 10px;">
                <label>Source Type</label>
                <select class="magic-select additional-source-type" data-id="${id}">
                    <option value="huggingface">HuggingFace Dataset</option>
                    <option value="upload">Upload File</option>
                    <option value="local_file">Local File Path</option>
                </select>
            </div>
            <div class="additional-hf-config" data-id="${id}">
                <div class="form-group" style="margin-bottom: 10px;">
                    <label>HuggingFace Repository ID</label>
                    <input type="text" class="magic-input additional-hf-repo" data-id="${id}" placeholder="username/dataset-name">
                </div>
                <div class="form-group" style="margin-bottom: 10px;">
                    <label>Split</label>
                    <input type="text" class="magic-input additional-hf-split" data-id="${id}" value="train">
                </div>
            </div>
            <div class="additional-upload-config" data-id="${id}" style="display: none;">
                <div class="form-group" style="margin-bottom: 10px;">
                    <label>Upload Dataset File</label>
                    <input type="file" class="magic-input additional-upload-file" data-id="${id}" accept=".json,.jsonl,.csv,.parquet">
                </div>
                <button type="button" class="action-button additional-upload-btn" data-id="${id}" style="margin-bottom: 10px;">
                    Upload
                </button>
                <div class="additional-upload-status" data-id="${id}"></div>
            </div>
            <div class="additional-local-config" data-id="${id}" style="display: none;">
                <div class="form-group" style="margin-bottom: 10px;">
                    <label>File Path</label>
                    <input type="text" class="magic-input additional-local-path" data-id="${id}" placeholder="/path/to/dataset.json">
                </div>
                <div class="form-group" style="margin-bottom: 10px;">
                    <label>File Format</label>
                    <select class="magic-select additional-local-format" data-id="${id}">
                        <option value="">Auto-detect</option>
                        <option value="json">JSON</option>
                        <option value="csv">CSV</option>
                        <option value="parquet">Parquet</option>
                    </select>
                </div>
            </div>
            <div class="form-row" style="gap: 10px;">
                <div class="form-group" style="flex: 1;">
                    <label>Column Mapping (Optional)</label>
                    <input type="text" class="magic-input additional-col-mapping" data-id="${id}"
                        placeholder='e.g. {"input":"prompt","output":"chosen"}'>
                    <small style="color: #888; font-size: 0.85em;">JSON mapping of source columns to standard names</small>
                </div>
                <div class="form-group" style="flex: 0 0 auto; display: flex; align-items: center; padding-top: 20px;">
                    <label style="display: flex; align-items: center; gap: 6px; cursor: pointer; font-size: 0.9em;">
                        <input type="checkbox" class="additional-convert-messages" data-id="${id}" checked>
                        <span>Auto-convert messages</span>
                    </label>
                </div>
            </div>
        `;

        listEl.appendChild(entry);
        this.additionalDatasets.push({ id, uploadedDatasetId: null });

        // Setup source type toggle for this entry
        const sourceTypeSelect = entry.querySelector('.additional-source-type');
        sourceTypeSelect.addEventListener('change', (e) => {
            const entryId = e.target.dataset.id;
            entry.querySelector(`.additional-hf-config[data-id="${entryId}"]`).style.display =
                e.target.value === 'huggingface' ? 'block' : 'none';
            entry.querySelector(`.additional-upload-config[data-id="${entryId}"]`).style.display =
                e.target.value === 'upload' ? 'block' : 'none';
            entry.querySelector(`.additional-local-config[data-id="${entryId}"]`).style.display =
                e.target.value === 'local_file' ? 'block' : 'none';
        });

        // Setup remove button
        entry.querySelector('.remove-dataset-btn').addEventListener('click', () => {
            this.removeAdditionalDataset(id);
        });

        // Setup upload button
        const uploadBtn = entry.querySelector('.additional-upload-btn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => this.handleAdditionalUpload(id));
        }
    }

    /**
     * Remove an additional dataset entry
     */
    removeAdditionalDataset(id) {
        const entry = document.getElementById(`additional-dataset-${id}`);
        if (entry) entry.remove();
        this.additionalDatasets = this.additionalDatasets.filter(d => d.id !== id);
    }

    /**
     * Handle upload for an additional dataset
     */
    async handleAdditionalUpload(id) {
        const fileInput = document.querySelector(`.additional-upload-file[data-id="${id}"]`);
        const statusEl = document.querySelector(`.additional-upload-status[data-id="${id}"]`);
        const file = fileInput?.files[0];

        if (!file) {
            this.toast.error('Please select a file to upload');
            return;
        }

        try {
            const data = await MerlinaAPI.uploadDataset(file);
            const entry = this.additionalDatasets.find(d => d.id === id);
            if (entry) entry.uploadedDatasetId = data.dataset_id;

            if (statusEl) {
                statusEl.innerHTML = `<div style="color: green; font-size: 0.85em;">Uploaded: ${sanitizeHTML(data.filename)} (ID: ${sanitizeHTML(data.dataset_id)})</div>`;
            }
            this.toast.success('Additional dataset uploaded!');
        } catch (error) {
            if (statusEl) {
                statusEl.innerHTML = `<div style="color: red; font-size: 0.85em;">Upload failed: ${sanitizeHTML(error.message)}</div>`;
            }
            this.toast.error(`Upload failed: ${error.message}`);
        }
    }

    /**
     * Get additional datasets configuration array
     */
    getAdditionalDatasetsConfig() {
        const configs = [];

        for (const dataset of this.additionalDatasets) {
            const id = dataset.id;
            const entry = document.getElementById(`additional-dataset-${id}`);
            if (!entry) continue;

            const sourceType = entry.querySelector('.additional-source-type')?.value;
            let source = { source_type: sourceType };

            if (sourceType === 'huggingface') {
                const repoId = entry.querySelector('.additional-hf-repo')?.value?.trim();
                if (!repoId) continue; // Skip empty entries
                source.repo_id = repoId;
                source.split = entry.querySelector('.additional-hf-split')?.value || 'train';
            } else if (sourceType === 'upload') {
                if (!dataset.uploadedDatasetId) continue;
                source.dataset_id = dataset.uploadedDatasetId;
            } else if (sourceType === 'local_file') {
                const filePath = entry.querySelector('.additional-local-path')?.value?.trim();
                if (!filePath) continue;
                source.file_path = filePath;
                const format = entry.querySelector('.additional-local-format')?.value;
                if (format) source.file_format = format;
            }

            // Parse column mapping
            let columnMapping = null;
            const colMappingStr = entry.querySelector('.additional-col-mapping')?.value?.trim();
            if (colMappingStr) {
                try {
                    columnMapping = JSON.parse(colMappingStr);
                } catch (e) {
                    this.toast.error(`Invalid column mapping JSON for Dataset #${id + 2}`);
                    throw new Error(`Invalid column mapping JSON for Dataset #${id + 2}`);
                }
            }

            const convertMessages = entry.querySelector('.additional-convert-messages')?.checked ?? true;

            configs.push({
                source: source,
                column_mapping: columnMapping,
                convert_messages_format: convertMessages
            });
        }

        return configs;
    }

    /**
     * Get column mapping from UI
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
}

export { DatasetManager };
