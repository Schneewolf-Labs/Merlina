/**
 * Dashboard View - Main landing page with overview and quick actions
 * Merlina Modular Frontend v2.0
 */

export class DashboardView {
    constructor(trainingModeManager) {
        this.trainingModeManager = trainingModeManager;
    }

    /**
     * Render dashboard view
     * @returns {string} HTML for dashboard
     */
    render() {
        return `
            <div class="dashboard">
                ${this.renderWelcomeBanner()}
                ${this.renderQuickStats()}
                ${this.renderModeSelector()}
                ${this.renderActionCards()}
                ${this.renderRecentJobs()}
            </div>
        `;
    }

    /**
     * Render welcome banner
     * @returns {string}
     */
    renderWelcomeBanner() {
        return `
            <div class="welcome-banner">
                <div class="welcome-banner-content">
                    <h2 class="welcome-banner-title">Welcome to Merlina! 🪄</h2>
                    <p class="welcome-banner-subtitle">
                        Magical LLM fine-tuning made simple. Train with ORPO or SFT in just a few clicks.
                    </p>
                    <div class="welcome-banner-actions">
                        <button class="btn btn-lg" style="background: white; color: var(--primary-purple);" data-action="new-training">
                            🎯 Start New Training
                        </button>
                        <button class="btn btn-lg btn-ghost" style="color: white; border: 2px solid white;" data-action="load-config">
                            💾 Load Config
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render quick stats cards
     * @returns {string}
     */
    renderQuickStats() {
        return `
            <div class="quick-stats">
                <div class="stat-card" data-stat="active-jobs">
                    <div class="stat-card-icon">🔮</div>
                    <div class="stat-card-label">Active Jobs</div>
                    <div class="stat-card-value" id="stat-active-jobs">0</div>
                </div>
                <div class="stat-card" data-stat="completed-jobs">
                    <div class="stat-card-icon">✅</div>
                    <div class="stat-card-label">Completed</div>
                    <div class="stat-card-value" id="stat-completed-jobs">0</div>
                </div>
                <div class="stat-card" data-stat="saved-configs">
                    <div class="stat-card-icon">💾</div>
                    <div class="stat-card-label">Saved Configs</div>
                    <div class="stat-card-value" id="stat-saved-configs">0</div>
                </div>
                <div class="stat-card" data-stat="gpus">
                    <div class="stat-card-icon">🎮</div>
                    <div class="stat-card-label">GPUs Available</div>
                    <div class="stat-card-value" id="stat-gpus">-</div>
                </div>
            </div>
        `;
    }

    /**
     * Render training mode selector
     * @returns {string}
     */
    renderModeSelector() {
        return `
            <div class="mode-selector">
                <div class="mode-selector-header">
                    <h2 class="mode-selector-title">Choose Your Training Mode</h2>
                    <p class="mode-selector-subtitle">
                        Select between preference optimization (ORPO) or traditional fine-tuning (SFT)
                    </p>
                </div>
                <div class="mode-selector-grid">
                    ${this.trainingModeManager.getModeCardHTML('orpo')}
                    ${this.trainingModeManager.getModeCardHTML('sft')}
                </div>
            </div>
        `;
    }

    /**
     * Render action cards
     * @returns {string}
     */
    renderActionCards() {
        return `
            <div class="action-cards">
                <div class="action-card" data-action="new-training">
                    <div class="action-card-icon">🎯</div>
                    <h3 class="action-card-title">New Experiment</h3>
                    <p class="action-card-description">
                        Configure and start a new training job from scratch with full control over parameters
                    </p>
                    <button class="btn btn-primary btn-block action-card-button">
                        Configure Training
                    </button>
                </div>

                <div class="action-card" data-action="load-config">
                    <div class="action-card-icon">📁</div>
                    <h3 class="action-card-title">Load Configuration</h3>
                    <p class="action-card-description">
                        Resume from a previously saved configuration to quickly start training
                    </p>
                    <button class="btn btn-primary btn-block action-card-button">
                        Browse Configs
                    </button>
                </div>

                <div class="action-card" data-action="view-analytics">
                    <div class="action-card-icon">📊</div>
                    <h3 class="action-card-title">View Analytics</h3>
                    <p class="action-card-description">
                        Explore training metrics, loss curves, and performance visualizations
                    </p>
                    <button class="btn btn-primary btn-block action-card-button">
                        View Charts
                    </button>
                </div>

                <div class="action-card" data-action="gpu-manager">
                    <div class="action-card-icon">🎮</div>
                    <h3 class="action-card-title">GPU Manager</h3>
                    <p class="action-card-description">
                        Monitor GPU usage, temperature, and memory across all available devices
                    </p>
                    <button class="btn btn-primary btn-block action-card-button">
                        Manage GPUs
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render recent jobs widget
     * @returns {string}
     */
    renderRecentJobs() {
        return `
            <div class="recent-jobs">
                <div class="recent-jobs-header">
                    <h3 class="recent-jobs-title">Recent Training Jobs</h3>
                    <button class="btn btn-sm btn-ghost" data-action="view-all-jobs">
                        View All →
                    </button>
                </div>
                <div class="recent-jobs-list" id="recent-jobs-list">
                    <div class="empty-state">
                        <div class="empty-state-icon">🔮</div>
                        <div class="empty-state-title">No training jobs yet</div>
                        <div class="empty-state-message">
                            Start your first training job to see it here
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Setup event listeners after render
     */
    attachEventListeners() {
        // Mode selection
        document.querySelectorAll('[data-select-mode]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.target.dataset.selectMode;
                this.trainingModeManager.setMode(mode);
                this.updateModeCards();
            });
        });

        // Action cards
        document.querySelectorAll('[data-action]').forEach(card => {
            card.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleAction(action);
            });
        });

        // Subscribe to mode changes
        this.trainingModeManager.subscribe((newMode, oldMode) => {
            console.log(`Dashboard: Mode changed from ${oldMode} to ${newMode}`);
            this.updateModeCards();
        });
    }

    /**
     * Update mode cards after selection
     */
    updateModeCards() {
        const container = document.querySelector('.mode-selector-grid');
        if (container) {
            container.innerHTML = `
                ${this.trainingModeManager.getModeCardHTML('orpo')}
                ${this.trainingModeManager.getModeCardHTML('sft')}
            `;

            // Reattach listeners
            document.querySelectorAll('[data-select-mode]').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const mode = e.target.dataset.selectMode;
                    this.trainingModeManager.setMode(mode);
                    this.updateModeCards();
                });
            });
        }
    }

    /**
     * Handle action button clicks
     * @param {string} action - Action name
     */
    handleAction(action) {
        const viewMap = {
            'new-training': 'training',
            'load-config': 'configs',
            'view-analytics': 'analytics',
            'gpu-manager': 'gpu',
            'view-all-jobs': 'job-history'
        };

        const view = viewMap[action];
        if (view) {
            window.dispatchEvent(new CustomEvent('navigate', {
                detail: { view }
            }));
        }
    }

    /**
     * Update stats (called periodically)
     * @param {object} stats - Stats object
     */
    updateStats(stats) {
        if (stats.activeJobs !== undefined) {
            const el = document.getElementById('stat-active-jobs');
            if (el) el.textContent = stats.activeJobs;
        }

        if (stats.completedJobs !== undefined) {
            const el = document.getElementById('stat-completed-jobs');
            if (el) el.textContent = stats.completedJobs;
        }

        if (stats.savedConfigs !== undefined) {
            const el = document.getElementById('stat-saved-configs');
            if (el) el.textContent = stats.savedConfigs;
        }

        if (stats.gpus !== undefined) {
            const el = document.getElementById('stat-gpus');
            if (el) el.textContent = stats.gpus;
        }
    }

    /**
     * Update recent jobs list
     * @param {array} jobs - Array of job objects
     */
    updateRecentJobs(jobs) {
        const container = document.getElementById('recent-jobs-list');
        if (!container) return;

        if (!jobs || jobs.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🔮</div>
                    <div class="empty-state-title">No training jobs yet</div>
                    <div class="empty-state-message">
                        Start your first training job to see it here
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = jobs.slice(0, 5).map(job => this.renderJobItem(job)).join('');
    }

    /**
     * Render a job item
     * @param {object} job - Job data
     * @returns {string}
     */
    renderJobItem(job) {
        const statusIcons = {
            running: '⚡',
            completed: '✅',
            failed: '❌',
            stopped: '⏸️',
            queued: '⏳'
        };

        const icon = statusIcons[job.status] || '🔮';
        const mode = job.training_mode || 'orpo';

        return `
            <div class="job-item" data-job-id="${job.job_id}">
                <div class="job-item-header">
                    <div class="job-item-name">
                        ${icon} ${job.output_name || job.job_id}
                        <span class="job-item-mode ${mode}">${mode.toUpperCase()}</span>
                    </div>
                    <span class="badge badge-${job.status}">${job.status}</span>
                </div>

                ${job.progress !== undefined ? `
                    <div class="job-item-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${job.progress}%"></div>
                        </div>
                    </div>
                ` : ''}

                <div class="job-item-stats">
                    ${job.current_step ? `<div class="job-item-stat">Step ${job.current_step}/${job.total_steps || '?'}</div>` : ''}
                    ${job.loss ? `<div class="job-item-stat">Loss: ${job.loss.toFixed(3)}</div>` : ''}
                    ${job.gpu_memory ? `<div class="job-item-stat">VRAM: ${job.gpu_memory}</div>` : ''}
                </div>

                <div class="job-item-actions">
                    <button class="btn btn-sm btn-ghost" data-job-action="view">
                        📊 Details
                    </button>
                    ${job.status === 'running' ? `
                        <button class="btn btn-sm btn-danger" data-job-action="stop">
                            🛑 Stop
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }
}
