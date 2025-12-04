/**
 * Training Mode Manager - Handles ORPO vs SFT mode logic
 * Merlina Modular Frontend v2.0
 */

export class TrainingModeManager {
    constructor() {
        this.mode = 'orpo'; // Default mode
        this.listeners = [];
        this.modeConfig = {
            orpo: {
                name: 'ORPO',
                fullName: 'Odds Ratio Preference Optimization',
                icon: '🏆',
                color: '#9b42ff',
                requiredColumns: ['prompt', 'chosen', 'rejected'],
                optionalColumns: ['system', 'reasoning'],
                requiresBeta: true,
                description: 'Trains model to prefer chosen responses over rejected responses',
                useCases: [
                    'DPO/RLHF datasets',
                    'Preference pairs',
                    'Quality rankings'
                ]
            },
            sft: {
                name: 'SFT',
                fullName: 'Supervised Fine-Tuning',
                icon: '📖',
                color: '#4299ff',
                requiredColumns: ['prompt', 'chosen'],
                optionalColumns: ['system', 'reasoning'],
                requiresBeta: false,
                description: 'Traditional fine-tuning on chosen responses only',
                useCases: [
                    'Instruction datasets',
                    'Chat templates',
                    'Standard fine-tuning'
                ]
            }
        };
    }

    /**
     * Set training mode
     * @param {string} mode - 'orpo' or 'sft'
     */
    setMode(mode) {
        if (!this.isValidMode(mode)) {
            throw new Error(`Invalid training mode: ${mode}. Must be 'orpo' or 'sft'.`);
        }

        const previousMode = this.mode;
        this.mode = mode;

        console.log(`Training mode changed: ${previousMode} → ${mode}`);

        // Notify all listeners
        this.notifyListeners(mode, previousMode);

        // Update UI elements
        this.updateUI();

        // Save to localStorage
        this.saveMode();

        return this;
    }

    /**
     * Get current training mode
     * @returns {string} Current mode ('orpo' or 'sft')
     */
    getMode() {
        return this.mode;
    }

    /**
     * Get mode configuration
     * @param {string} mode - Optional specific mode, defaults to current
     * @returns {object} Mode configuration
     */
    getModeConfig(mode = null) {
        const targetMode = mode || this.mode;
        return this.modeConfig[targetMode];
    }

    /**
     * Check if mode is valid
     * @param {string} mode - Mode to check
     * @returns {boolean}
     */
    isValidMode(mode) {
        return mode === 'orpo' || mode === 'sft';
    }

    /**
     * Get required dataset columns for current mode
     * @returns {array} Required column names
     */
    getRequiredColumns() {
        return this.getModeConfig().requiredColumns;
    }

    /**
     * Get optional dataset columns for current mode
     * @returns {array} Optional column names
     */
    getOptionalColumns() {
        return this.getModeConfig().optionalColumns;
    }

    /**
     * Check if beta parameter is required for current mode
     * @returns {boolean}
     */
    requiresBeta() {
        return this.getModeConfig().requiresBeta;
    }

    /**
     * Get mode display name
     * @returns {string}
     */
    getModeName() {
        return this.getModeConfig().name;
    }

    /**
     * Get mode full name
     * @returns {string}
     */
    getModeFullName() {
        return this.getModeConfig().fullName;
    }

    /**
     * Get mode icon emoji
     * @returns {string}
     */
    getModeIcon() {
        return this.getModeConfig().icon;
    }

    /**
     * Get mode color
     * @returns {string}
     */
    getModeColor() {
        return this.getModeConfig().color;
    }

    /**
     * Get mode description
     * @returns {string}
     */
    getModeDescription() {
        return this.getModeConfig().description;
    }

    /**
     * Get mode use cases
     * @returns {array}
     */
    getModeUseCases() {
        return this.getModeConfig().useCases;
    }

    /**
     * Validate dataset for current mode
     * @param {array} columns - Available dataset columns
     * @returns {object} Validation result
     */
    validateDataset(columns) {
        const required = this.getRequiredColumns();
        const missing = required.filter(col => !columns.includes(col));

        if (missing.length > 0) {
            return {
                valid: false,
                errors: [`Missing required columns for ${this.getModeName()}: ${missing.join(', ')}`],
                suggestion: this.mode === 'orpo'
                    ? 'Switch to SFT mode if you only have chosen responses'
                    : 'Ensure your dataset has both prompt and chosen columns'
            };
        }

        return {
            valid: true,
            message: `Dataset is compatible with ${this.getModeName()} mode`
        };
    }

    /**
     * Subscribe to mode changes
     * @param {function} callback - Callback function (newMode, oldMode) => void
     */
    subscribe(callback) {
        this.listeners.push(callback);
        return () => this.unsubscribe(callback);
    }

    /**
     * Unsubscribe from mode changes
     * @param {function} callback - Callback to remove
     */
    unsubscribe(callback) {
        this.listeners = this.listeners.filter(cb => cb !== callback);
    }

    /**
     * Notify all listeners of mode change
     * @param {string} newMode - New mode
     * @param {string} oldMode - Previous mode
     */
    notifyListeners(newMode, oldMode) {
        this.listeners.forEach(callback => {
            try {
                callback(newMode, oldMode);
            } catch (error) {
                console.error('Error in mode change listener:', error);
            }
        });
    }

    /**
     * Update UI elements based on current mode
     */
    updateUI() {
        // Update beta field visibility
        const betaField = document.getElementById('beta-field');
        if (betaField) {
            betaField.style.display = this.requiresBeta() ? 'block' : 'none';
        }

        // Update mode indicators
        document.querySelectorAll('[data-mode-indicator]').forEach(el => {
            el.textContent = this.getModeName();
            el.style.color = this.getModeColor();
        });

        // Update mode badges
        document.querySelectorAll('.mode-badge').forEach(badge => {
            badge.textContent = this.getModeName();
            badge.className = `badge mode-badge badge-${this.mode}`;
        });

        // Dispatch custom event
        window.dispatchEvent(new CustomEvent('trainingModeChanged', {
            detail: {
                mode: this.mode,
                config: this.getModeConfig()
            }
        }));
    }

    /**
     * Save mode to localStorage
     */
    saveMode() {
        try {
            localStorage.setItem('merlina_training_mode', this.mode);
        } catch (error) {
            console.error('Failed to save training mode:', error);
        }
    }

    /**
     * Load mode from localStorage
     */
    loadMode() {
        try {
            const saved = localStorage.getItem('merlina_training_mode');
            if (saved && this.isValidMode(saved)) {
                this.setMode(saved);
            }
        } catch (error) {
            console.error('Failed to load training mode:', error);
        }
    }

    /**
     * Get mode badge HTML
     * @returns {string} HTML for mode badge
     */
    getModeBadgeHTML() {
        return `<span class="badge badge-${this.mode}">${this.getModeIcon()} ${this.getModeName()}</span>`;
    }

    /**
     * Get mode info card HTML
     * @param {string} mode - Mode to get card for
     * @returns {string} HTML for mode card
     */
    getModeCardHTML(mode) {
        const config = this.modeConfig[mode];
        const isActive = this.mode === mode;

        return `
            <div class="mode-card mode-card-${mode} ${isActive ? 'active' : ''}" data-mode="${mode}">
                <div class="mode-card-badge">Selected</div>
                <div class="mode-card-icon">${config.icon}</div>
                <h3 class="mode-card-title">${config.fullName}</h3>
                <p class="mode-card-subtitle">${config.description}</p>

                <ul class="mode-card-requirements">
                    <li>✅ ${config.requiredColumns.join(', ')}</li>
                    ${mode === 'orpo' ? '<li>❌ Rejected responses required</li>' : '<li>⊗ Rejected responses not needed</li>'}
                    ${config.requiresBeta ? '<li>🎚️ Beta parameter tuning</li>' : '<li>🎯 Direct learning</li>'}
                </ul>

                <div class="mode-card-benefits">
                    <div class="mode-card-benefits-title">Best for:</div>
                    <ul class="mode-card-benefits-list">
                        ${config.useCases.map(useCase => `<li>• ${useCase}</li>`).join('')}
                    </ul>
                </div>

                <button class="btn btn-primary btn-block mode-card-button" data-select-mode="${mode}">
                    ${isActive ? '✓ Selected' : `Select ${config.name}`}
                </button>
            </div>
        `;
    }
}
