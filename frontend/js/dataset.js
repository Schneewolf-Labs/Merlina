// Dataset Module - Dataset configuration and preview

import { MerlinaAPI } from './api.js';
import { Toast, LoadingManager } from './ui.js';
import { Validator } from './validation.js';
import { sanitizeHTML } from './validation.js';

const PAIRED_MODES = ['orpo', 'dpo', 'simpo', 'cpo', 'ipo'];

/**
 * Dataset Manager — unified handling of one or more dataset cards.
 *
 * There is no "primary" vs "additional" distinction in the UI: every dataset
 * is a card in #datasets-list. When building a config for the backend, the
 * first card maps to DatasetConfig.source + column_mapping, and subsequent
 * cards map to DatasetConfig.additional_sources[].
 */
class DatasetManager {
    constructor() {
        this.datasetCounter = 0;

        // Preview navigation state (operates on the concatenated dataset)
        this.previewOffset = 0;
        this.previewLimit = 1;
        this.previewTotalCount = 0;
        this.previewType = null; // 'raw' or 'formatted'

        this.toast = new Toast();
        this.setupEventListeners();
        this.initializeFirstCard();
    }

    setupEventListeners() {
        document.getElementById('dataset-format-type')
            ?.addEventListener('change', (e) => this.handleFormatTypeChange(e));

        document.getElementById('preview-dataset-button')
            ?.addEventListener('click', () => this.handlePreview());
        document.getElementById('preview-formatted-button')
            ?.addEventListener('click', () => this.handlePreviewFormatted());

        document.getElementById('add-dataset-btn')
            ?.addEventListener('click', () => this.addDataset());

        document.getElementById('preview-prev-button')
            ?.addEventListener('click', () => this.handlePrevious());
        document.getElementById('preview-next-button')
            ?.addEventListener('click', () => this.handleNext());
        document.getElementById('jump-to-index-button')
            ?.addEventListener('click', () => this.handleJumpToIndex());
        document.getElementById('preview-limit')
            ?.addEventListener('change', (e) => this.handleLimitChange(e));
        document.getElementById('preview-index')
            ?.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleJumpToIndex();
            });

        document.getElementById('dataset-stats-button')
            ?.addEventListener('click', () => this.handleDatasetStats());

        const trainingModeEl = document.getElementById('training-mode');
        if (trainingModeEl) {
            trainingModeEl.addEventListener('change', (e) => {
                this.applyModeToAllCards(e.target.value);
            });
        }
    }

    /**
     * Start with a single card prefilled with the legacy default dataset
     * so existing behavior is preserved on first load.
     */
    initializeFirstCard() {
        const card = this.addDataset({ canRemove: false });
        if (!card) return;
        const repoInput = card.querySelector('.ds-repo');
        if (repoInput && !repoInput.value) {
            repoInput.value = 'schneewolflabs/Athanorlite-DPO';
        }
    }

    getCards() {
        return Array.from(document.querySelectorAll('#datasets-list .dataset-card'));
    }

    /**
     * Append a new dataset card. Returns the card element.
     */
    addDataset({ canRemove = true } = {}) {
        const container = document.getElementById('datasets-list');
        if (!container) return null;

        this.datasetCounter++;
        const id = this.datasetCounter;
        const card = document.createElement('div');
        card.className = 'dataset-card';
        card.dataset.id = String(id);
        card.innerHTML = this.cardTemplate();
        container.appendChild(card);

        this.wireCard(card, canRemove);
        this.renumberCards();
        this.applyModeToCard(card, document.getElementById('training-mode')?.value || 'orpo');
        return card;
    }

    cardTemplate() {
        return `
            <div class="dataset-card-header">
                <span class="dataset-card-title">📦 Dataset</span>
                <button type="button" class="remove-dataset-btn" title="Remove dataset">&times;</button>
            </div>
            <div class="dataset-card-body">
                <div class="form-group">
                    <label>Source Type</label>
                    <select class="magic-select ds-source-type">
                        <option value="huggingface">HuggingFace Dataset</option>
                        <option value="upload">Upload File</option>
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

                <div class="ds-upload-config" style="display: none;">
                    <div class="form-group">
                        <label>Upload Dataset File</label>
                        <input type="file" class="magic-input ds-file" accept=".json,.jsonl,.csv,.parquet">
                        <small style="color: #888; font-size: 0.85em;">Supported: JSON, JSONL, CSV, Parquet</small>
                    </div>
                    <div class="ds-upload-status"></div>
                    <button type="button" class="action-button ds-upload-btn" style="margin-top: 10px;">
                        📤 Upload Dataset
                    </button>
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

                <div class="ds-sample-preview" style="margin-top: 12px; display: none;">
                    <div style="font-weight: bold; color: var(--primary-purple); margin-bottom: 8px; font-size: 0.9em;">
                        Sample Data (First Row):
                    </div>
                    <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid var(--light-purple); max-height: 250px; overflow-y: auto;">
                        <pre class="ds-sample-content" style="margin: 0; font-size: 0.8em; white-space: pre-wrap;"></pre>
                    </div>
                </div>
            </div>
        `;
    }

    wireCard(card, canRemove) {
        const removeBtn = card.querySelector('.remove-dataset-btn');
        if (!canRemove) {
            removeBtn.style.display = 'none';
        }
        removeBtn.addEventListener('click', () => {
            if (this.getCards().length <= 1) {
                this.toast.error('At least one dataset is required');
                return;
            }
            card.remove();
            this.renumberCards();
            this.refreshMessagesFormatNotice();
        });

        const sourceTypeSel = card.querySelector('.ds-source-type');
        sourceTypeSel.addEventListener('change', () => this.updateCardSourceVisibility(card));
        this.updateCardSourceVisibility(card);

        card.querySelector('.ds-upload-btn')
            .addEventListener('click', () => this.handleCardUpload(card));

        card.querySelector('.ds-inspect-btn')
            .addEventListener('click', () => this.inspectCard(card));
    }

    updateCardSourceVisibility(card) {
        const t = card.querySelector('.ds-source-type').value;
        card.querySelector('.ds-hf-config').style.display = t === 'huggingface' ? 'block' : 'none';
        card.querySelector('.ds-upload-config').style.display = t === 'upload' ? 'block' : 'none';
        card.querySelector('.ds-local-config').style.display = t === 'local_file' ? 'block' : 'none';
    }

    renumberCards() {
        const cards = this.getCards();
        cards.forEach((card, idx) => {
            const title = card.querySelector('.dataset-card-title');
            if (title) title.textContent = `📦 Dataset ${idx + 1}`;
        });
    }

    /**
     * Apply the current training mode to every card's rejected-field UI.
     */
    applyModeToAllCards(mode) {
        this.getCards().forEach(card => this.applyModeToCard(card, mode));
    }

    applyModeToCard(card, mode) {
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
        } else if (PAIRED_MODES.includes(mode)) {
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
     * Upload a file for this card. Stores the returned id on the card.
     */
    async handleCardUpload(card) {
        const fileInput = card.querySelector('.ds-file');
        const statusEl = card.querySelector('.ds-upload-status');
        const btn = card.querySelector('.ds-upload-btn');
        const file = fileInput?.files?.[0];

        if (!file) {
            this.toast.error('Please select a file to upload');
            return;
        }

        try {
            LoadingManager.show(btn, '⏳ Uploading...');
            const data = await MerlinaAPI.uploadDataset(file);
            card.dataset.uploadId = data.dataset_id;
            // First card's upload id is exposed globally for ConfigManager compatibility.
            if (this.getCards()[0] === card) {
                window.uploadedDatasetId = data.dataset_id;
            }
            statusEl.innerHTML = `
                <div class="success-message">
                    ✅ Uploaded: ${sanitizeHTML(data.filename)} (ID: ${sanitizeHTML(data.dataset_id)})
                </div>
            `;
            this.toast.success('Dataset uploaded successfully!');
        } catch (error) {
            console.error('Upload failed:', error);
            this.toast.error(`Upload failed: ${error.message}`);
            statusEl.innerHTML = `
                <div class="error-message">
                    ❌ Upload failed: ${sanitizeHTML(error.message)}
                </div>
            `;
        } finally {
            LoadingManager.hide(btn);
        }
    }

    /**
     * Build a minimal DatasetConfig for a single card — used by the columns
     * endpoint during inspection.
     */
    buildCardSourceConfig(card) {
        return {
            source: this.readCardSource(card),
            format: { format_type: 'chatml' },
            test_size: 0.1
        };
    }

    /**
     * Extract a DatasetSource object from a card's inputs.
     */
    readCardSource(card) {
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
        } else if (sourceType === 'upload') {
            const datasetId = card.dataset.uploadId;
            if (!datasetId) throw new Error('Please upload a dataset first');
            source.dataset_id = datasetId;
        }

        return source;
    }

    /**
     * Inspect a card's dataset columns and populate its mapping dropdowns.
     */
    async inspectCard(card) {
        const btn = card.querySelector('.ds-inspect-btn');
        try {
            LoadingManager.show(btn, '⏳ Loading columns...');
            const cfg = this.buildCardSourceConfig(card);
            const data = await MerlinaAPI.getDatasetColumns(cfg);

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
                while (sel.options.length > 1) sel.remove(1);
                data.columns.forEach(col => {
                    const opt = document.createElement('option');
                    opt.value = col;
                    opt.textContent = col;
                    sel.appendChild(opt);
                });
                if (data.columns.includes(target)) sel.value = target;
            });

            card.querySelector('.ds-columns-list').textContent = data.columns.join(', ');
            card.querySelector('.ds-colmap-config').style.display = 'block';
            card.dataset.hasMessages = data.columns.includes('messages') ? '1' : '0';

            if (data.samples && data.samples.length > 0) {
                const sampleEl = card.querySelector('.ds-sample-content');
                if (sampleEl) sampleEl.textContent = JSON.stringify(data.samples[0], null, 2);
                card.querySelector('.ds-sample-preview').style.display = 'block';
            }

            this.refreshMessagesFormatNotice();
            this.toast.success(`Found ${data.columns.length} columns`);
        } catch (error) {
            console.error('Inspect failed:', error);
            this.toast.error(`Inspect failed: ${error.message}`);
        } finally {
            LoadingManager.hide(btn);
        }
    }

    /**
     * Show/hide the global messages-format notice based on whether any card
     * has a "messages" column.
     */
    refreshMessagesFormatNotice() {
        const notice = document.getElementById('messages-format-notice');
        if (!notice) return;
        const anyHasMessages = this.getCards().some(card => card.dataset.hasMessages === '1');
        notice.style.display = anyHasMessages ? 'block' : 'none';
    }

    /**
     * Read a card's column mapping into a {sourceCol: standardName} object.
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

    handleFormatTypeChange(e) {
        const formatType = e.target.value;
        const customConfig = document.getElementById('custom-format-config');
        const qwen3Config = document.getElementById('qwen3-format-config');
        if (customConfig) customConfig.style.display = formatType === 'custom' ? 'block' : 'none';
        if (qwen3Config) qwen3Config.style.display = formatType === 'qwen3' ? 'block' : 'none';
    }

    async handlePreview() {
        this.previewType = 'raw';
        this.previewOffset = 0;
        this.previewLimit = parseInt(document.getElementById('preview-limit').value) || 1;
        await this.loadPreview();
    }

    async loadPreview() {
        if (this.previewType === 'raw') await this.loadRawPreview();
        else if (this.previewType === 'formatted') await this.loadFormattedPreview();
    }

    async loadRawPreview() {
        const previewBtn = document.getElementById('preview-dataset-button');
        try {
            LoadingManager.show(previewBtn, '⏳ Loading...');
            const datasetConfig = this.getDatasetConfig(true);
            const data = await MerlinaAPI.previewDataset(datasetConfig, this.previewOffset, this.previewLimit);

            this.previewTotalCount = data.total_count;

            const previewDiv = document.getElementById('dataset-preview');
            const previewContent = document.getElementById('dataset-preview-content');
            const formattedPreview = document.getElementById('formatted-preview');

            if (previewContent) previewContent.textContent = JSON.stringify(data.samples, null, 2);
            if (previewDiv) previewDiv.style.display = 'block';
            if (formattedPreview) formattedPreview.style.display = 'none';

            this.updatePositionInfo('raw');
            this.toast.success(`Loaded ${data.num_samples} sample(s)`);
        } catch (error) {
            console.error('Preview failed:', error);
            this.toast.error(`Preview failed: ${error.message}`);
        } finally {
            LoadingManager.hide(previewBtn);
        }
    }

    async handlePreviewFormatted() {
        this.previewType = 'formatted';
        this.previewOffset = 0;
        this.previewLimit = parseInt(document.getElementById('preview-limit').value) || 1;
        await this.loadFormattedPreview();
    }

    async loadFormattedPreview() {
        const previewBtn = document.getElementById('preview-formatted-button');
        try {
            LoadingManager.show(previewBtn, '⏳ Loading...');
            const datasetConfig = this.getDatasetConfig(true);
            const data = await MerlinaAPI.previewFormattedDataset(datasetConfig, this.previewOffset, this.previewLimit);

            if (!data.samples || data.samples.length === 0) {
                throw new Error('No samples returned');
            }

            this.previewTotalCount = data.total_count;
            const sample = data.samples[0];

            const formattedPreview = document.getElementById('formatted-preview');
            const rawPreview = document.getElementById('dataset-preview');
            const promptEl = document.getElementById('formatted-prompt');
            const chosenEl = document.getElementById('formatted-chosen');
            const rejectedEl = document.getElementById('formatted-rejected');
            const formatTypeEl = document.getElementById('format-type-display');

            if (promptEl) promptEl.textContent = sample.prompt;
            if (chosenEl) chosenEl.textContent = sample.chosen;
            if (rejectedEl) rejectedEl.textContent = sample.rejected;

            const formatType = datasetConfig.format.format_type;
            const formatNames = {
                'tokenizer': 'Tokenizer (Model Native)',
                'chatml': 'ChatML',
                'llama3': 'Llama 3',
                'mistral': 'Mistral Instruct',
                'qwen3': `Qwen 3 (thinking ${datasetConfig.format.enable_thinking ? 'enabled' : 'disabled'})`,
                'custom': 'Custom Template'
            };
            if (formatTypeEl) formatTypeEl.textContent = formatNames[formatType] || formatType;

            if (formattedPreview) formattedPreview.style.display = 'block';
            if (rawPreview) rawPreview.style.display = 'none';

            this.updatePositionInfo('formatted');
            this.toast.success(`Preview formatted with ${formatNames[formatType]}`);
        } catch (error) {
            console.error('Formatted preview failed:', error);
            this.toast.error(`Formatted preview failed: ${error.message}`);
        } finally {
            LoadingManager.hide(previewBtn);
        }
    }

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
     * Build the full DatasetConfig payload for the backend. The first card
     * becomes `source` + `column_mapping`; remaining cards become
     * `additional_sources[]` (each with its own column_mapping).
     *
     * @param {boolean} forPreview - If true, skip the rejected-column
     *     requirement (preview doesn't care about training mode).
     */
    getDatasetConfig(forPreview = false) {
        const cards = this.getCards();
        if (cards.length === 0) throw new Error('Please add at least one dataset');

        const [firstCard, ...rest] = cards;
        const source = this.readCardSource(firstCard);
        const firstMapping = this.readCardColumnMapping(firstCard);

        const formatType = document.getElementById('dataset-format-type').value;
        const format = { format_type: formatType };
        if (formatType === 'custom') {
            format.custom_templates = {
                prompt_template: document.getElementById('custom-prompt-template')?.value || '',
                chosen_template: document.getElementById('custom-chosen-template')?.value || '',
                rejected_template: document.getElementById('custom-rejected-template')?.value || ''
            };
        }
        if (formatType === 'qwen3') {
            format.enable_thinking = document.getElementById('enable-thinking')?.checked ?? true;
        }

        const config = {
            source,
            format,
            test_size: parseFloat(document.getElementById('test-size').value)
        };

        const maxSamples = document.getElementById('max-samples')?.value;
        if (maxSamples) config.max_samples = parseInt(maxSamples);

        config.deduplicate = document.getElementById('deduplicate')?.checked ?? false;
        config.dedupe_strategy = document.getElementById('dedupe-strategy')?.value || 'prompt_chosen';

        const baseModel = document.getElementById('base-model')?.value?.trim();
        if (baseModel) config.model_name = baseModel;

        if (Object.keys(firstMapping).length > 0) {
            config.column_mapping = firstMapping;
        }

        const convertCheckbox = document.getElementById('convert-messages-checkbox');
        config.convert_messages_format = convertCheckbox ? convertCheckbox.checked : true;

        const systemPrompt = document.getElementById('system-prompt-override')?.value?.trim();
        if (systemPrompt) {
            config.system_prompt = systemPrompt;
            const modeRadio = document.querySelector('input[name="system-prompt-mode"]:checked');
            config.system_prompt_mode = modeRadio ? modeRadio.value : 'fill_empty';
        }

        if (rest.length > 0) {
            config.additional_sources = rest.map(card => {
                const src = this.readCardSource(card);
                const mapping = this.readCardColumnMapping(card);
                if (Object.keys(mapping).length > 0) src.column_mapping = mapping;
                return src;
            });
        }

        const trainingMode = forPreview ? 'sft' : (document.getElementById('training-mode')?.value || 'orpo');
        config.training_mode = trainingMode;

        const errors = Validator.validateDatasetConfig(config, trainingMode);
        if (errors.length > 0) throw new Error(errors.join('; '));

        return config;
    }

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
        const newOffset = index - 1;
        if (newOffset >= this.previewTotalCount) {
            this.toast.error(`Index out of range (max: ${this.previewTotalCount})`);
            return;
        }
        this.previewOffset = newOffset;
        await this.loadPreview();
    }

    async handleLimitChange(e) {
        if (!this.previewType) return;
        this.previewLimit = parseInt(e.target.value) || 1;
        this.previewOffset = 0;
        await this.loadPreview();
    }

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

        const indexInput = document.getElementById('preview-index');
        if (indexInput) {
            indexInput.value = startIdx;
            indexInput.max = this.previewTotalCount;
        }

        this.updateNavigationButtons();
    }

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
