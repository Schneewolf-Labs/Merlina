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

        // Preview navigation state
        this.previewOffset = 0;
        this.previewLimit = 1;
        this.previewTotalCount = 0;
        this.previewType = null; // 'raw' or 'formatted'

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

        // Add dataset button
        const addDatasetBtn = document.getElementById('add-dataset-btn');
        if (addDatasetBtn) {
            addDatasetBtn.addEventListener('click', () => this.addAdditionalDataset());
        }

        // Navigation controls
        const prevBtn = document.getElementById('preview-prev-button');
        if (prevBtn) {
            prevBtn.addEventListener('click', () => this.handlePrevious());
        }

        const nextBtn = document.getElementById('preview-next-button');
        if (nextBtn) {
            nextBtn.addEventListener('click', () => this.handleNext());
        }

        const jumpBtn = document.getElementById('jump-to-index-button');
        if (jumpBtn) {
            jumpBtn.addEventListener('click', () => this.handleJumpToIndex());
        }

        const limitSelect = document.getElementById('preview-limit');
        if (limitSelect) {
            limitSelect.addEventListener('change', (e) => this.handleLimitChange(e));
        }

        // Dataset stats button
        const statsBtn = document.getElementById('dataset-stats-button');
        if (statsBtn) {
            statsBtn.addEventListener('click', () => this.handleDatasetStats());
        }

        // Enter key on index input
        const indexInput = document.getElementById('preview-index');
        if (indexInput) {
            indexInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.handleJumpToIndex();
                }
            });
        }

        this.additionalDatasetCounter = 0;

        // React to training-mode changes for column mapping UI
        const trainingModeEl = document.getElementById('training-mode');
        if (trainingModeEl) {
            trainingModeEl.addEventListener('change', (e) => {
                this.updateColumnMappingForMode(e.target.value);
            });
            // Apply once on init
            this.updateColumnMappingForMode(trainingModeEl.value);
        }
    }

    /**
     * Update the rejected-column UI based on training mode.
     * Paired preference modes: Required (red badge, dropdown enabled)
     * KTO: Optional (gray badge, dropdown enabled)
     * SFT: Not used (gray badge, dropdown disabled)
     */
    updateColumnMappingForMode(mode) {
        const PAIRED = ['orpo', 'dpo', 'simpo', 'cpo', 'ipo'];
        const badge = document.getElementById('rejected-badge');
        const group = document.getElementById('map-rejected-group');
        const select = document.getElementById('map-rejected');
        const hint = document.getElementById('rejected-column-hint');
        if (!badge || !select || !group || !hint) return;

        badge.classList.remove('required', 'optional', 'not-used');

        if (mode === 'sft') {
            badge.textContent = 'Not used';
            badge.classList.add('not-used');
            group.classList.add('field-disabled');
            select.disabled = true;
            hint.textContent = 'SFT only uses the chosen response — this field is ignored.';
        } else if (mode === 'kto') {
            badge.textContent = 'Optional';
            badge.classList.add('optional');
            group.classList.remove('field-disabled');
            select.disabled = false;
            hint.textContent = 'Optional for KTO — provide to split into negative examples, or leave empty.';
        } else if (PAIRED.includes(mode)) {
            badge.textContent = 'Required';
            badge.classList.add('required');
            group.classList.remove('field-disabled');
            select.disabled = false;
            hint.textContent = 'Non-preferred response — required for preference optimization.';
        } else {
            badge.textContent = 'Optional';
            badge.classList.add('optional');
            group.classList.remove('field-disabled');
            select.disabled = false;
            hint.textContent = 'Non-preferred response.';
        }

        // Propagate to any additional-dataset cards
        document.querySelectorAll('#additional-datasets-list .additional-dataset-entry')
            .forEach(card => this.applyModeToCard(card, mode));
    }

    /**
     * Add an additional dataset source entry to the UI as a full card
     * with its own source config, inspect button, and column mapping.
     */
    addAdditionalDataset() {
        this.additionalDatasetCounter++;
        const id = this.additionalDatasetCounter;
        const container = document.getElementById('additional-datasets-list');
        if (!container) return;

        // Primary dataset is implicitly #1, so additional cards start at #2
        const displayNum = container.querySelectorAll('.additional-dataset-entry').length + 2;

        const card = document.createElement('div');
        card.className = 'dataset-card additional-dataset-entry';
        card.dataset.id = id;
        card.innerHTML = `
            <div class="dataset-card-header">
                <span class="dataset-card-title">📦 Dataset ${displayNum}</span>
                <button type="button" class="remove-dataset-btn" title="Remove dataset">&times;</button>
            </div>
            <div class="dataset-card-body">
                <div class="form-group">
                    <label>Source Type</label>
                    <select class="magic-select ds-source-type">
                        <option value="huggingface">HuggingFace Dataset</option>
                        <option value="local_file">Local File Path</option>
                    </select>
                </div>

                <div class="ds-hf-config">
                    <div class="form-row">
                        <div class="form-group" style="flex: 2;">
                            <label>Repository ID</label>
                            <input type="text" class="magic-input ds-repo" placeholder="username/dataset-name">
                        </div>
                        <div class="form-group" style="flex: 1;">
                            <label>Split</label>
                            <input type="text" class="magic-input ds-split" value="train">
                        </div>
                    </div>
                </div>

                <div class="ds-local-config" style="display: none;">
                    <div class="form-row">
                        <div class="form-group" style="flex: 2;">
                            <label>File Path</label>
                            <input type="text" class="magic-input ds-local-path" placeholder="/path/to/dataset.json">
                        </div>
                        <div class="form-group" style="flex: 1;">
                            <label>Format</label>
                            <select class="magic-select ds-local-format">
                                <option value="">Auto-detect</option>
                                <option value="json">JSON</option>
                                <option value="csv">CSV</option>
                                <option value="parquet">Parquet</option>
                            </select>
                        </div>
                    </div>
                </div>

                <button type="button" class="action-button ds-inspect-btn" style="width: 100%; margin-top: 8px;">
                    🔍 Inspect Columns
                </button>

                <div class="ds-colmap-config" style="display: none; margin-top: 12px;">
                    <div class="ds-columns-summary" style="background: #f5f7fa; padding: 10px; border-radius: 8px; margin-bottom: 10px; font-size: 0.85em;">
                        <strong style="color: var(--primary-purple);">Columns:</strong>
                        <span class="ds-columns-list"></span>
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label>Prompt <span class="col-badge required">Required</span></label>
                            <select class="magic-select ds-map-prompt">
                                <option value="">-- Select --</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>Chosen <span class="col-badge required">Required</span></label>
                            <select class="magic-select ds-map-chosen">
                                <option value="">-- Select --</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-row">
                        <div class="form-group ds-map-rejected-group">
                            <label>Rejected <span class="col-badge ds-rejected-badge required">Required</span></label>
                            <select class="magic-select ds-map-rejected">
                                <option value="">-- None --</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label>System <span class="col-badge optional">Optional</span></label>
                            <select class="magic-select ds-map-system">
                                <option value="">-- None --</option>
                            </select>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Reasoning <span class="col-badge optional">Optional</span></label>
                        <select class="magic-select ds-map-reasoning">
                            <option value="">-- None --</option>
                        </select>
                    </div>
                </div>
            </div>
        `;
        container.appendChild(card);

        // Wire remove button
        card.querySelector('.remove-dataset-btn').addEventListener('click', () => {
            card.remove();
            this.renumberAdditionalCards();
        });

        // Wire source-type toggle
        const sourceTypeSel = card.querySelector('.ds-source-type');
        sourceTypeSel.addEventListener('change', (e) => {
            const t = e.target.value;
            card.querySelector('.ds-hf-config').style.display = t === 'huggingface' ? 'block' : 'none';
            card.querySelector('.ds-local-config').style.display = t === 'local_file' ? 'block' : 'none';
        });

        // Wire inspect button
        card.querySelector('.ds-inspect-btn').addEventListener('click', () => {
            this.inspectCardColumns(card);
        });

        // Reflect current training mode on the card's rejected field
        const mode = document.getElementById('training-mode')?.value || 'orpo';
        this.applyModeToCard(card, mode);
    }

    /**
     * Re-number the "Dataset N" titles after a card is removed.
     */
    renumberAdditionalCards() {
        const cards = document.querySelectorAll('#additional-datasets-list .additional-dataset-entry');
        cards.forEach((card, idx) => {
            const titleEl = card.querySelector('.dataset-card-title');
            if (titleEl) titleEl.textContent = `📦 Dataset ${idx + 2}`;
        });
    }

    /**
     * Build a minimal DatasetConfig for a card to pass to /dataset/columns.
     */
    buildCardSourceConfig(card) {
        const sourceType = card.querySelector('.ds-source-type').value;
        const source = { source_type: sourceType };

        if (sourceType === 'huggingface') {
            const repoId = card.querySelector('.ds-repo').value.trim();
            if (!repoId) throw new Error('Repository ID is required');
            source.repo_id = repoId;
            source.split = card.querySelector('.ds-split').value.trim() || 'train';
        } else if (sourceType === 'local_file') {
            const filePath = card.querySelector('.ds-local-path').value.trim();
            if (!filePath) throw new Error('File path is required');
            source.file_path = filePath;
            const fmt = card.querySelector('.ds-local-format').value;
            if (fmt) source.file_format = fmt;
        }

        return {
            source: source,
            format: { format_type: 'chatml' },  // Dummy format — only source matters for /dataset/columns
            test_size: 0.1
        };
    }

    /**
     * Inspect columns for a single additional-dataset card.
     */
    async inspectCardColumns(card) {
        const btn = card.querySelector('.ds-inspect-btn');
        try {
            LoadingManager.show(btn, '⏳ Loading columns...');
            const cfg = this.buildCardSourceConfig(card);
            const data = await MerlinaAPI.getDatasetColumns(cfg);

            // Populate mapping dropdowns
            const selects = [
                ['.ds-map-prompt', 'prompt'],
                ['.ds-map-chosen', 'chosen'],
                ['.ds-map-rejected', 'rejected'],
                ['.ds-map-system', 'system'],
                ['.ds-map-reasoning', 'reasoning'],
            ];
            selects.forEach(([selector, target]) => {
                const sel = card.querySelector(selector);
                if (!sel) return;
                // Clear options except first
                while (sel.options.length > 1) sel.remove(1);
                data.columns.forEach(col => {
                    const opt = document.createElement('option');
                    opt.value = col;
                    opt.textContent = col;
                    sel.appendChild(opt);
                });
                // Auto-select if column name matches
                if (data.columns.includes(target)) sel.value = target;
            });

            // Show columns list + mapping UI
            card.querySelector('.ds-columns-list').textContent = data.columns.join(', ');
            card.querySelector('.ds-colmap-config').style.display = 'block';

            this.toast.success(`Found ${data.columns.length} columns`);
        } catch (error) {
            console.error('Card inspect failed:', error);
            this.toast.error(`Inspect failed: ${error.message}`);
        } finally {
            LoadingManager.hide(btn);
        }
    }

    /**
     * Apply the current training mode to a single card's rejected-column UI.
     */
    applyModeToCard(card, mode) {
        const PAIRED = ['orpo', 'dpo', 'simpo', 'cpo', 'ipo'];
        const badge = card.querySelector('.ds-rejected-badge');
        const group = card.querySelector('.ds-map-rejected-group');
        const select = card.querySelector('.ds-map-rejected');
        if (!badge || !group || !select) return;

        badge.classList.remove('required', 'optional', 'not-used');
        if (mode === 'sft') {
            badge.textContent = 'Not used';
            badge.classList.add('not-used');
            group.classList.add('field-disabled');
            select.disabled = true;
        } else if (mode === 'kto') {
            badge.textContent = 'Optional';
            badge.classList.add('optional');
            group.classList.remove('field-disabled');
            select.disabled = false;
        } else if (PAIRED.includes(mode)) {
            badge.textContent = 'Required';
            badge.classList.add('required');
            group.classList.remove('field-disabled');
            select.disabled = false;
        } else {
            badge.textContent = 'Optional';
            badge.classList.add('optional');
            group.classList.remove('field-disabled');
            select.disabled = false;
        }
    }

    /**
     * Read a card's column mapping into a {sourceCol: standardName} object.
     */
    getCardColumnMapping(card) {
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
     * Get additional dataset sources from the UI
     */
    getAdditionalSources() {
        const cards = document.querySelectorAll('#additional-datasets-list .additional-dataset-entry');
        const sources = [];
        for (const card of cards) {
            const sourceType = card.querySelector('.ds-source-type')?.value || 'huggingface';
            const source = { source_type: sourceType };

            if (sourceType === 'huggingface') {
                const repoId = card.querySelector('.ds-repo')?.value?.trim();
                if (!repoId) continue;
                source.repo_id = repoId;
                source.split = card.querySelector('.ds-split')?.value?.trim() || 'train';
            } else if (sourceType === 'local_file') {
                const filePath = card.querySelector('.ds-local-path')?.value?.trim();
                if (!filePath) continue;
                source.file_path = filePath;
                const fmt = card.querySelector('.ds-local-format')?.value;
                if (fmt) source.file_format = fmt;
            }

            const mapping = this.getCardColumnMapping(card);
            if (Object.keys(mapping).length > 0) {
                source.column_mapping = mapping;
            }

            sources.push(source);
        }
        return sources;
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
        this.previewType = 'raw';
        this.previewOffset = 0;
        this.previewLimit = parseInt(document.getElementById('preview-limit').value) || 1;
        await this.loadPreview();
    }

    /**
     * Load preview with current offset and limit
     */
    async loadPreview() {
        if (this.previewType === 'raw') {
            await this.loadRawPreview();
        } else if (this.previewType === 'formatted') {
            await this.loadFormattedPreview();
        }
    }

    /**
     * Load raw dataset preview
     */
    async loadRawPreview() {
        const previewBtn = document.getElementById('preview-dataset-button');

        try {
            LoadingManager.show(previewBtn, '⏳ Loading...');

            const datasetConfig = this.getDatasetConfig(true);  // forPreview=true
            const data = await MerlinaAPI.previewDataset(datasetConfig, this.previewOffset, this.previewLimit);

            this.previewTotalCount = data.total_count;

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

            // Update position info
            this.updatePositionInfo('raw');

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
        this.previewType = 'formatted';
        this.previewOffset = 0;
        this.previewLimit = parseInt(document.getElementById('preview-limit').value) || 1;
        await this.loadFormattedPreview();
    }

    /**
     * Load formatted dataset preview
     */
    async loadFormattedPreview() {
        const previewBtn = document.getElementById('preview-formatted-button');

        try {
            LoadingManager.show(previewBtn, '⏳ Loading...');

            const datasetConfig = this.getDatasetConfig(true);  // forPreview=true
            const data = await MerlinaAPI.previewFormattedDataset(datasetConfig, this.previewOffset, this.previewLimit);

            if (!data.samples || data.samples.length === 0) {
                throw new Error('No samples returned');
            }

            this.previewTotalCount = data.total_count;

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

            // Update position info
            this.updatePositionInfo('formatted');

            this.toast.success(`Preview formatted with ${formatNames[formatType]}`);
        } catch (error) {
            console.error('Formatted preview failed:', error);
            this.toast.error(`Formatted preview failed: ${error.message}`);
        } finally {
            LoadingManager.hide(previewBtn);
        }
    }

    /**
     * Handle dataset stats computation
     */
    async handleDatasetStats() {
        const statsBtn = document.getElementById('dataset-stats-button');

        try {
            LoadingManager.show(statsBtn, '⏳ Analyzing...');

            const datasetConfig = this.getDatasetConfig(true);
            const data = await MerlinaAPI.getDatasetStats(datasetConfig);

            this.renderDatasetStats(data);

            const panel = document.getElementById('dataset-stats-panel');
            if (panel) panel.style.display = 'block';

            this.toast.success(`Dataset analyzed: ${data.total_rows} rows`);
        } catch (error) {
            console.error('Dataset stats failed:', error);
            this.toast.error(`Stats failed: ${error.message}`);
        } finally {
            LoadingManager.hide(statsBtn);
        }
    }

    /**
     * Render dataset statistics into the stats panel
     */
    renderDatasetStats(data) {
        const container = document.getElementById('dataset-stats-content');
        if (!container) return;

        const fmt = (n) => typeof n === 'number' ? n.toLocaleString() : n;

        let html = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 15px;">
                <div style="background: #f5f7fa; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.4em; font-weight: bold; color: var(--primary-purple);">${fmt(data.total_rows)}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 2px;">Total Rows</div>
                </div>
                <div style="background: #f5f7fa; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.4em; font-weight: bold; color: var(--primary-purple);">${fmt(data.est_total_tokens)}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 2px;">Est. Tokens</div>
                </div>
                <div style="background: #f5f7fa; padding: 12px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 1.4em; font-weight: bold; color: var(--primary-purple);">${data.columns.length}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 2px;">Columns</div>
                </div>
            </div>
        `;

        // Field-level stats table
        const fields = data.field_stats;
        if (fields && Object.keys(fields).length > 0) {
            html += `
                <h4 style="color: var(--secondary-purple); margin-bottom: 8px; font-size: 0.95em;">Field Statistics</h4>
                <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85em;">
                    <thead>
                        <tr style="background: #f0ebff; text-align: left;">
                            <th style="padding: 8px; border-bottom: 2px solid var(--light-purple);">Field</th>
                            <th style="padding: 8px; border-bottom: 2px solid var(--light-purple);">Avg Len</th>
                            <th style="padding: 8px; border-bottom: 2px solid var(--light-purple);">Min</th>
                            <th style="padding: 8px; border-bottom: 2px solid var(--light-purple);">Med</th>
                            <th style="padding: 8px; border-bottom: 2px solid var(--light-purple);">Max</th>
                            <th style="padding: 8px; border-bottom: 2px solid var(--light-purple);">Est. Tokens</th>
                            <th style="padding: 8px; border-bottom: 2px solid var(--light-purple);">Empty</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            for (const [field, fs] of Object.entries(fields)) {
                const emptyWarning = fs.empty_count > 0
                    ? `<span style="color: var(--danger);" title="${fs.empty_count} empty values">${fmt(fs.empty_count)}</span>`
                    : '<span style="color: #999;">0</span>';

                html += `
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 8px; font-weight: 600; color: var(--primary-purple);">${field}</td>
                        <td style="padding: 8px;">${fmt(fs.avg_length)}</td>
                        <td style="padding: 8px;">${fmt(fs.min_length)}</td>
                        <td style="padding: 8px;">${fmt(fs.median_length)}</td>
                        <td style="padding: 8px;">${fmt(fs.max_length)}</td>
                        <td style="padding: 8px;">${fmt(fs.est_avg_tokens)}</td>
                        <td style="padding: 8px;">${emptyWarning}</td>
                    </tr>
                `;
            }

            html += `</tbody></table></div>`;
        }

        // Class balance for preference modes
        if (data.length_balance) {
            const lb = data.length_balance;
            const barWidth = Math.min(lb.chosen_longer_pct, 100);
            const barColor = lb.chosen_longer_pct >= 50 ? 'var(--success)' : 'var(--danger)';

            html += `
                <h4 style="color: var(--secondary-purple); margin: 15px 0 8px; font-size: 0.95em;">Length Balance (Chosen vs Rejected)</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                    <div style="background: #e8f5e9; padding: 10px; border-radius: 8px; text-align: center;">
                        <div style="font-weight: bold; color: var(--success);">${fmt(lb.avg_chosen_length)}</div>
                        <div style="font-size: 0.8em; color: #666;">Avg Chosen</div>
                    </div>
                    <div style="background: #ffebee; padding: 10px; border-radius: 8px; text-align: center;">
                        <div style="font-weight: bold; color: var(--danger);">${fmt(lb.avg_rejected_length)}</div>
                        <div style="font-size: 0.8em; color: #666;">Avg Rejected</div>
                    </div>
                    <div style="background: #f5f7fa; padding: 10px; border-radius: 8px; text-align: center;">
                        <div style="font-weight: bold; color: var(--primary-purple);">${lb.ratio}x</div>
                        <div style="font-size: 0.8em; color: #666;">Length Ratio</div>
                    </div>
                </div>
                <div style="background: #f5f7fa; padding: 10px; border-radius: 8px;">
                    <div style="font-size: 0.85em; color: #666; margin-bottom: 5px;">Chosen longer than rejected: <strong>${lb.chosen_longer_pct}%</strong></div>
                    <div style="background: #e0e0e0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: ${barColor}; height: 100%; width: ${barWidth}%; border-radius: 4px; transition: width 0.3s;"></div>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
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

        // Deduplication
        config.deduplicate = document.getElementById('deduplicate')?.checked ?? false;
        config.dedupe_strategy = document.getElementById('dedupe-strategy')?.value || 'prompt_chosen';

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

        // System prompt override
        const systemPrompt = document.getElementById('system-prompt-override')?.value?.trim();
        if (systemPrompt) {
            config.system_prompt = systemPrompt;
            const modeRadio = document.querySelector('input[name="system-prompt-mode"]:checked');
            config.system_prompt_mode = modeRadio ? modeRadio.value : 'fill_empty';
        }

        // Add additional dataset sources
        const additionalSources = this.getAdditionalSources();
        if (additionalSources.length > 0) {
            config.additional_sources = additionalSources;
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

    /**
     * Handle previous button click
     */
    async handlePrevious() {
        if (!this.previewType) {
            this.toast.error('Please load a preview first');
            return;
        }

        const newOffset = Math.max(0, this.previewOffset - this.previewLimit);
        if (newOffset !== this.previewOffset) {
            this.previewOffset = newOffset;
            await this.loadPreview();
        }
    }

    /**
     * Handle next button click
     */
    async handleNext() {
        if (!this.previewType) {
            this.toast.error('Please load a preview first');
            return;
        }

        const newOffset = this.previewOffset + this.previewLimit;
        if (newOffset < this.previewTotalCount) {
            this.previewOffset = newOffset;
            await this.loadPreview();
        }
    }

    /**
     * Handle jump to index
     */
    async handleJumpToIndex() {
        if (!this.previewType) {
            this.toast.error('Please load a preview first');
            return;
        }

        const indexInput = document.getElementById('preview-index');
        const index = parseInt(indexInput.value);

        if (isNaN(index) || index < 1) {
            this.toast.error('Please enter a valid index (starting from 1)');
            return;
        }

        // Convert 1-based index to 0-based offset
        const newOffset = index - 1;

        if (newOffset >= this.previewTotalCount) {
            this.toast.error(`Index out of range (max: ${this.previewTotalCount})`);
            return;
        }

        this.previewOffset = newOffset;
        await this.loadPreview();
    }

    /**
     * Handle limit change
     */
    async handleLimitChange(e) {
        if (!this.previewType) {
            return;
        }

        this.previewLimit = parseInt(e.target.value) || 1;
        // Reset to beginning when changing limit
        this.previewOffset = 0;
        await this.loadPreview();
    }

    /**
     * Update position information display
     */
    updatePositionInfo(type) {
        const startIdx = this.previewOffset + 1;
        const endIdx = Math.min(this.previewOffset + this.previewLimit, this.previewTotalCount);

        const infoText = `(Showing ${startIdx}-${endIdx} of ${this.previewTotalCount})`;
        const positionText = `Samples ${startIdx}-${endIdx} of ${this.previewTotalCount}`;

        if (type === 'raw') {
            const infoEl = document.getElementById('raw-preview-info');
            const positionEl = document.getElementById('raw-preview-position');
            if (infoEl) infoEl.textContent = infoText;
            if (positionEl) positionEl.textContent = positionText;
        } else if (type === 'formatted') {
            const infoEl = document.getElementById('formatted-preview-info');
            const positionEl = document.getElementById('formatted-preview-position');
            if (infoEl) infoEl.textContent = infoText;
            if (positionEl) positionEl.textContent = positionText;
        }

        // Update index input to show current position
        const indexInput = document.getElementById('preview-index');
        if (indexInput) {
            indexInput.value = startIdx;
            indexInput.max = this.previewTotalCount;
        }

        // Update button states
        this.updateNavigationButtons();
    }

    /**
     * Update navigation button states (enable/disable)
     */
    updateNavigationButtons() {
        const prevBtn = document.getElementById('preview-prev-button');
        const nextBtn = document.getElementById('preview-next-button');

        if (prevBtn) {
            prevBtn.disabled = this.previewOffset === 0;
            prevBtn.style.opacity = this.previewOffset === 0 ? '0.5' : '1';
        }

        if (nextBtn) {
            const isAtEnd = this.previewOffset + this.previewLimit >= this.previewTotalCount;
            nextBtn.disabled = isAtEnd;
            nextBtn.style.opacity = isAtEnd ? '0.5' : '1';
        }
    }
}

export { DatasetManager };
