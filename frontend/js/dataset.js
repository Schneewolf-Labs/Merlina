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
