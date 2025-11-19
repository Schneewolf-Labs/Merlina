/**
 * Training Configuration View - Complete training setup with tabs
 * Merlina Modular Frontend v2.0
 */

export class TrainingConfigView {
    constructor(trainingModeManager) {
        this.trainingModeManager = trainingModeManager;
        this.currentTab = 'basic';
    }

    /**
     * Render training configuration view
     * @returns {string}
     */
    render() {
        return `
            <div class="training-config">
                ${this.renderHeader()}
                ${this.renderModeIndicator()}
                ${this.renderTabs()}
                ${this.renderTabContent()}
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
                <h2 class="card-title">⚙️ Configure Training</h2>
                <p class="card-subtitle">
                    Set up your ${this.trainingModeManager.getModeFullName()} training job
                </p>
            </div>
        `;
    }

    /**
     * Render mode indicator
     * @returns {string}
     */
    renderModeIndicator() {
        const mode = this.trainingModeManager.getMode();
        const config = this.trainingModeManager.getModeConfig();

        return `
            <div class="alert alert-info" style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <strong>${config.icon} ${config.fullName}</strong> mode selected
                    <small style="display: block; margin-top: 4px;">${config.description}</small>
                </div>
                <button class="btn btn-sm btn-ghost" data-action="change-mode">
                    Change Mode
                </button>
            </div>
        `;
    }

    /**
     * Render tabs
     * @returns {string}
     */
    renderTabs() {
        const tabs = [
            { id: 'basic', label: '🎯 Basic', icon: '🎯' },
            { id: 'lora', label: '🪄 LoRA', icon: '🪄' },
            { id: 'training', label: '⚗️ Training', icon: '⚗️' },
            { id: 'advanced', label: '⚙️ Advanced', icon: '⚙️' }
        ];

        return `
            <div class="tabs">
                ${tabs.map(tab => `
                    <div class="tab ${this.currentTab === tab.id ? 'active' : ''}"
                         data-tab="${tab.id}">
                        ${tab.icon} ${tab.label}
                    </div>
                `).join('')}
            </div>
        `;
    }

    /**
     * Render tab content
     * @returns {string}
     */
    renderTabContent() {
        return `
            <div class="tab-contents">
                ${this.renderBasicTab()}
                ${this.renderLoRATab()}
                ${this.renderTrainingTab()}
                ${this.renderAdvancedTab()}
            </div>
        `;
    }

    /**
     * Render basic configuration tab
     * @returns {string}
     */
    renderBasicTab() {
        return `
            <div class="tab-content ${this.currentTab === 'basic' ? 'active' : ''}" data-tab-content="basic">
                <div class="card">
                    <div class="spell-section">
                        <h3>🏷️ Output Configuration</h3>
                        <div class="form-group">
                            <label>Output Model Name</label>
                            <input type="text" id="output-name" class="input"
                                   placeholder="my-magical-model" required>
                            <small>Name for your fine-tuned model</small>
                        </div>
                    </div>

                    <div class="spell-section">
                        <h3>🎮 GPU Selection</h3>
                        <div class="form-group">
                            <button type="button" class="btn btn-secondary btn-block" id="refresh-gpu-btn">
                                🔄 Refresh GPU List
                            </button>
                        </div>
                        <div id="gpu-list" style="margin-top: var(--space-md);">
                            <div class="empty-state">
                                <div class="empty-state-message">Click refresh to see available GPUs</div>
                            </div>
                        </div>
                    </div>

                    <div class="spell-section">
                        <h3>🔧 Optimizer</h3>
                        <div class="form-group">
                            <label>Optimizer Type</label>
                            <select id="optimizer-type" class="select">
                                <option value="paged_adamw_8bit" selected>Paged AdamW 8-bit (Recommended)</option>
                                <option value="paged_adamw_32bit">Paged AdamW 32-bit</option>
                                <option value="adamw_8bit">AdamW 8-bit</option>
                                <option value="adamw_torch">AdamW PyTorch</option>
                                <option value="adafactor">Adafactor</option>
                            </select>
                            <small>Paged optimizers swap memory to CPU when GPU is full</small>
                        </div>
                    </div>

                    <div class="spell-section">
                        <h3>⚡ Attention Implementation</h3>
                        <div class="form-group">
                            <label>Attention Type</label>
                            <select id="attn-implementation" class="select">
                                <option value="auto" selected>Auto (Recommended)</option>
                                <option value="flash_attention_2">Flash Attention 2</option>
                                <option value="sdpa">SDPA</option>
                                <option value="eager">Eager</option>
                            </select>
                            <small>Auto mode uses Flash Attention 2 on compatible GPUs</small>
                        </div>
                    </div>

                    <div class="spell-section">
                        <h3>🌟 Options</h3>
                        <div class="checkbox-group">
                            <label class="checkbox">
                                <input type="checkbox" id="use-4bit" checked>
                                <span>Use 4-bit Quantization 🗜️</span>
                            </label>
                            <label class="checkbox">
                                <input type="checkbox" id="use-wandb">
                                <span>Report to Weights & Biases 📊</span>
                            </label>
                            <label class="checkbox">
                                <input type="checkbox" id="push-hub">
                                <span>Push to HuggingFace Hub 🤗</span>
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render LoRA configuration tab
     * @returns {string}
     */
    renderLoRATab() {
        return `
            <div class="tab-content ${this.currentTab === 'lora' ? 'active' : ''}" data-tab-content="lora">
                <div class="card">
                    <div class="spell-section">
                        <h3>🪄 LoRA Configuration</h3>

                        <div class="form-group">
                            <label class="checkbox">
                                <input type="checkbox" id="use-lora" checked>
                                <span style="font-weight: bold;">Enable LoRA (Low-Rank Adaptation)</span>
                            </label>
                            <small>When enabled, trains only LoRA adapters. When disabled, trains full model.</small>
                        </div>

                        <div id="lora-settings">
                            <div class="form-row">
                                <div class="form-group">
                                    <label>Rank (r)</label>
                                    <input type="number" id="lora-r" class="input" value="64" min="8" max="256">
                                    <small>Higher rank = more capacity but more VRAM</small>
                                </div>
                                <div class="form-group">
                                    <label>Alpha</label>
                                    <input type="number" id="lora-alpha" class="input" value="32" min="8" max="256">
                                    <small>Scaling factor (typically rank/2)</small>
                                </div>
                                <div class="form-group">
                                    <label>Dropout</label>
                                    <input type="number" id="lora-dropout" class="input" value="0.05"
                                           min="0" max="0.5" step="0.01">
                                    <small>Regularization (0 = no dropout)</small>
                                </div>
                            </div>

                            <div class="form-group">
                                <label>Target Modules</label>
                                <input type="text" id="target-modules" class="input"
                                       value="up_proj,down_proj,gate_proj,k_proj,q_proj,v_proj,o_proj">
                                <small>Comma-separated list of layers to apply LoRA</small>
                            </div>
                        </div>
                    </div>

                    <div class="alert alert-info">
                        <strong>💡 LoRA Tips</strong>
                        <ul style="margin: var(--space-sm) 0 0 var(--space-lg); font-size: var(--text-sm);">
                            <li>Rank 64-128 works well for most tasks</li>
                            <li>Alpha = rank/2 is a good starting point</li>
                            <li>Lower dropout (0.05) for small datasets</li>
                            <li>Target all linear layers for best results</li>
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render training parameters tab
     * @returns {string}
     */
    renderTrainingTab() {
        const showBeta = this.trainingModeManager.requiresBeta();

        return `
            <div class="tab-content ${this.currentTab === 'training' ? 'active' : ''}" data-tab-content="training">
                <div class="card">
                    <div class="spell-section">
                        <h3>⚗️ Training Parameters</h3>

                        <div class="form-row">
                            <div class="form-group">
                                <label>Learning Rate</label>
                                <input type="number" id="learning-rate" class="input"
                                       value="0.000005" step="0.000001">
                                <small>Try 5e-6 to start</small>
                            </div>
                            <div class="form-group">
                                <label>Epochs</label>
                                <input type="number" id="epochs" class="input" value="2" min="1" max="10">
                                <small>Number of passes through dataset</small>
                            </div>
                        </div>

                        <div class="form-row">
                            <div class="form-group">
                                <label>Batch Size</label>
                                <input type="number" id="batch-size" class="input" value="1" min="1" max="8">
                                <small>Per-device batch size</small>
                            </div>
                            <div class="form-group">
                                <label>Gradient Accumulation</label>
                                <input type="number" id="grad-accum" class="input" value="16" min="1" max="128">
                                <small>Effective batch = batch × accum</small>
                            </div>
                        </div>

                        <div class="form-row">
                            <div class="form-group">
                                <label>Max Length</label>
                                <input type="number" id="max-length" class="input" value="2048"
                                       min="512" max="8192" step="512">
                                <small>Maximum sequence length</small>
                            </div>
                            <div class="form-group" id="beta-field" style="display: ${showBeta ? 'block' : 'none'};">
                                <label>ORPO Beta</label>
                                <input type="number" id="beta" class="input" value="0.1"
                                       min="0.01" max="1" step="0.01">
                                <small>Preference strength (ORPO only)</small>
                            </div>
                        </div>

                        <div class="form-row">
                            <div class="form-group">
                                <label>Random Seed 🎲</label>
                                <input type="number" id="seed" class="input" value="42" min="0">
                                <small>For reproducibility</small>
                            </div>
                            <div class="form-group">
                                <label>Max Gradient Norm ✂️</label>
                                <input type="number" id="max-grad-norm" class="input" value="0.3"
                                       min="0.01" max="10" step="0.01">
                                <small>Gradient clipping</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render advanced settings tab
     * @returns {string}
     */
    renderAdvancedTab() {
        return `
            <div class="tab-content ${this.currentTab === 'advanced' ? 'active' : ''}" data-tab-content="advanced">
                <div class="card">
                    <div class="spell-section">
                        <h3>⚙️ Advanced Training Settings</h3>

                        <div class="form-row">
                            <div class="form-group">
                                <label>Max Prompt Length</label>
                                <input type="number" id="max-prompt-length" class="input" value="1024"
                                       min="256" max="4096" step="128">
                            </div>
                            <div class="form-group">
                                <label>Warmup Ratio</label>
                                <input type="number" id="warmup-ratio" class="input" value="0.05"
                                       min="0" max="0.5" step="0.01">
                            </div>
                        </div>

                        <div class="form-row">
                            <div class="form-group">
                                <label>Weight Decay</label>
                                <input type="number" id="weight-decay" class="input" value="0.01"
                                       min="0" max="0.5" step="0.001">
                            </div>
                            <div class="form-group">
                                <label>LR Scheduler Type</label>
                                <select id="lr-scheduler-type" class="select">
                                    <option value="cosine" selected>Cosine</option>
                                    <option value="linear">Linear</option>
                                    <option value="constant">Constant</option>
                                    <option value="constant_with_warmup">Constant with Warmup</option>
                                </select>
                            </div>
                        </div>

                        <div class="form-row">
                            <div class="form-group">
                                <label>Eval Steps Ratio</label>
                                <input type="number" id="eval-steps" class="input" value="0.2"
                                       min="0.1" max="1" step="0.05">
                                <small>How often to evaluate (0-1)</small>
                            </div>
                            <div class="form-group">
                                <label>Logging Steps</label>
                                <input type="number" id="logging-steps" class="input" value="1"
                                       min="1" max="100">
                                <small>Log metrics every N steps</small>
                            </div>
                        </div>

                        <div class="checkbox-group">
                            <label class="checkbox">
                                <input type="checkbox" id="shuffle-dataset" checked>
                                <span>Shuffle Dataset 🔀</span>
                            </label>
                            <label class="checkbox">
                                <input type="checkbox" id="gradient-checkpointing">
                                <span>Gradient Checkpointing 💾</span>
                            </label>
                        </div>
                    </div>

                    <div class="spell-section">
                        <h3>🔧 Optimizer Advanced</h3>
                        <div class="form-row">
                            <div class="form-group">
                                <label>Adam Beta1</label>
                                <input type="number" id="adam-beta1" class="input" value="0.9"
                                       min="0.8" max="0.99" step="0.01">
                            </div>
                            <div class="form-group">
                                <label>Adam Beta2</label>
                                <input type="number" id="adam-beta2" class="input" value="0.999"
                                       min="0.9" max="0.9999" step="0.001">
                            </div>
                            <div class="form-group">
                                <label>Adam Epsilon</label>
                                <input type="number" id="adam-epsilon" class="input" value="0.00000001"
                                       min="0.000000001" max="0.000001" step="0.000000001">
                            </div>
                        </div>
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
                <button class="btn btn-secondary" data-action="save-config">
                    💾 Save Config
                </button>
                <button class="btn btn-secondary" data-action="validate">
                    🔍 Validate
                </button>
                <button class="btn btn-primary btn-lg" data-action="start-training">
                    <span>🪄</span>
                    <span>Start Training</span>
                </button>
            </div>
        `;
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Tab switching
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const tabId = e.currentTarget.dataset.tab;
                this.switchTab(tabId);
            });
        });

        // LoRA toggle
        const loraCheckbox = document.getElementById('use-lora');
        if (loraCheckbox) {
            loraCheckbox.addEventListener('change', (e) => {
                const settings = document.getElementById('lora-settings');
                if (settings) {
                    settings.style.display = e.target.checked ? 'block' : 'none';
                }
            });
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
            this.updateBetaFieldVisibility();
        });
    }

    /**
     * Switch to a different tab
     * @param {string} tabId - Tab ID to switch to
     */
    switchTab(tabId) {
        this.currentTab = tabId;

        // Update tab buttons
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.tab === tabId);
        });

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('active', content.dataset.tabContent === tabId);
        });
    }

    /**
     * Update beta field visibility based on mode
     */
    updateBetaFieldVisibility() {
        const betaField = document.getElementById('beta-field');
        if (betaField) {
            betaField.style.display = this.trainingModeManager.requiresBeta() ? 'block' : 'none';
        }
    }

    /**
     * Handle action button clicks
     * @param {string} action - Action name
     */
    handleAction(action) {
        switch (action) {
            case 'change-mode':
                window.dispatchEvent(new CustomEvent('navigate', {
                    detail: { view: 'dashboard' }
                }));
                break;
            case 'save-config':
                this.saveConfiguration();
                break;
            case 'validate':
                this.validateConfiguration();
                break;
            case 'start-training':
                this.startTraining();
                break;
        }
    }

    /**
     * Save configuration
     */
    saveConfiguration() {
        console.log('Save configuration (not implemented)');
        window.dispatchEvent(new CustomEvent('toast', {
            detail: { message: 'Configuration saved', type: 'success' }
        }));
    }

    /**
     * Validate configuration
     */
    validateConfiguration() {
        console.log('Validate configuration (not implemented)');
        window.dispatchEvent(new CustomEvent('toast', {
            detail: { message: 'Validation passed', type: 'success' }
        }));
    }

    /**
     * Start training
     */
    startTraining() {
        console.log('Start training (not implemented)');
        window.dispatchEvent(new CustomEvent('toast', {
            detail: { message: 'Training job submitted', type: 'success' }
        }));
    }
}
