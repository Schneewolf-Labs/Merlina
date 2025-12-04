/**
 * Active Jobs View - Monitor all running and queued jobs
 * Merlina Modular Frontend v2.0
 */

export class ActiveJobsView {
    constructor() {
        this.jobs = [];
        this.refreshInterval = null;
    }

    /**
     * Render active jobs view
     * @returns {string}
     */
    render() {
        return `
            <div class="active-jobs-view">
                ${this.renderHeader()}
                ${this.renderQueueStatus()}
                ${this.renderJobsList()}
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
                        <h2 class="card-title">🔮 Active Jobs</h2>
                        <p class="card-subtitle">Monitor running and queued training jobs</p>
                    </div>
                    <button class="btn btn-secondary" id="refresh-jobs-btn">
                        🔄 Refresh
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render queue status
     * @returns {string}
     */
    renderQueueStatus() {
        return `
            <div class="card" style="margin-bottom: var(--space-lg);">
                <h3>📊 Queue Status</h3>
                <div id="queue-status-container">
                    <div style="text-align: center; padding: var(--space-lg); color: var(--text-secondary);">
                        Loading queue status...
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render jobs list
     * @returns {string}
     */
    renderJobsList() {
        return `
            <div class="card">
                <h3>🎯 Jobs</h3>
                <div id="jobs-list-container">
                    <div style="text-align: center; padding: var(--space-lg); color: var(--text-secondary);">
                        Loading jobs...
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Refresh button
        const refreshBtn = document.getElementById('refresh-jobs-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadJobs();
            });
        }

        // Initial load
        this.loadJobs();

        // Auto-refresh every 2 seconds
        this.refreshInterval = setInterval(() => {
            this.loadJobs();
        }, 2000);
    }

    /**
     * Load jobs from API
     */
    async loadJobs() {
        try {
            // Load queue status
            const queueResponse = await fetch('/queue/status');
            const queueData = await queueResponse.json();
            this.renderQueueStatusData(queueData);

            // Load jobs
            const jobsResponse = await fetch('/jobs');
            const jobsData = await jobsResponse.json();

            // Filter for active jobs
            this.jobs = jobsData.jobs?.filter(job =>
                ['queued', 'initializing', 'loading_model', 'loading_dataset', 'training', 'running'].includes(job.status)
            ) || [];

            this.renderJobsListData(this.jobs);
        } catch (error) {
            console.error('Failed to load jobs:', error);
        }
    }

    /**
     * Render queue status data
     * @param {object} data - Queue status data
     */
    renderQueueStatusData(data) {
        const container = document.getElementById('queue-status-container');
        if (!container) return;

        container.innerHTML = `
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: var(--space-md);">
                <div class="stat-card">
                    <div class="stat-card-label">Running</div>
                    <div class="stat-card-value">${data.running_count || 0}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-card-label">Queued</div>
                    <div class="stat-card-value">${data.queued_count || 0}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-card-label">Total Active</div>
                    <div class="stat-card-value">${data.total_active || 0}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-card-label">Max Concurrent</div>
                    <div class="stat-card-value">${data.max_concurrent || 1}</div>
                </div>
            </div>
        `;
    }

    /**
     * Render jobs list data
     * @param {array} jobs - Jobs array
     */
    renderJobsListData(jobs) {
        const container = document.getElementById('jobs-list-container');
        if (!container) return;

        if (!jobs || jobs.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">🌟</div>
                    <div class="empty-state-title">No Active Jobs</div>
                    <div class="empty-state-message">Start a training job to see it here</div>
                </div>
            `;
            return;
        }

        container.innerHTML = jobs.map(job => this.renderJobCard(job)).join('');

        // Attach job action listeners
        document.querySelectorAll('[data-job-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.jobAction;
                const jobId = e.target.dataset.jobId;
                this.handleJobAction(action, jobId);
            });
        });
    }

    /**
     * Render individual job card
     * @param {object} job - Job data
     * @returns {string}
     */
    renderJobCard(job) {
        const statusIcon = this.getStatusIcon(job.status);
        const mode = job.config?.training_mode || 'orpo';
        const modeIcon = mode === 'orpo' ? '🏆' : '📖';
        const progress = job.progress || 0;

        return `
            <div class="card" style="margin-bottom: var(--space-md); padding: var(--space-lg);">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: var(--space-md);">
                    <div>
                        <h4 style="margin: 0 0 var(--space-xs) 0;">
                            ${statusIcon} ${job.config?.output_name || job.job_id}
                        </h4>
                        <div style="font-size: var(--text-sm); color: var(--text-secondary);">
                            Job ID: <code>${job.job_id}</code>
                        </div>
                    </div>
                    <div style="display: flex; gap: var(--space-sm); align-items: center;">
                        <span class="badge badge-${mode}">${modeIcon} ${mode.toUpperCase()}</span>
                        <span class="badge badge-${this.getStatusBadgeClass(job.status)}">${job.status}</span>
                    </div>
                </div>

                ${job.status !== 'queued' ? `
                    <div style="margin-bottom: var(--space-md);">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${progress}%"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: var(--space-xs); font-size: var(--text-sm);">
                            <span>${progress.toFixed(1)}%</span>
                            <span>${job.current_step || 0} / ${job.total_steps || '?'} steps</span>
                        </div>
                    </div>
                ` : ''}

                ${job.queue_position !== undefined ? `
                    <div style="margin-bottom: var(--space-md); padding: var(--space-sm); background: var(--surface-2); border-radius: var(--radius-sm);">
                        📍 Queue Position: <strong>${job.queue_position + 1}</strong>
                    </div>
                ` : ''}

                ${job.status === 'training' || job.status === 'running' ? `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: var(--space-md); margin-bottom: var(--space-md);">
                        ${job.loss !== undefined ? `
                            <div>
                                <div style="font-size: var(--text-xs); color: var(--text-secondary);">Loss</div>
                                <div style="font-size: var(--text-lg); font-weight: bold;">${job.loss.toFixed(4)}</div>
                            </div>
                        ` : ''}
                        ${job.learning_rate ? `
                            <div>
                                <div style="font-size: var(--text-xs); color: var(--text-secondary);">Learning Rate</div>
                                <div style="font-size: var(--text-lg); font-weight: bold;">${job.learning_rate.toExponential(2)}</div>
                            </div>
                        ` : ''}
                        ${job.gpu_memory ? `
                            <div>
                                <div style="font-size: var(--text-xs); color: var(--text-secondary);">GPU Memory</div>
                                <div style="font-size: var(--text-lg); font-weight: bold;">${job.gpu_memory}</div>
                            </div>
                        ` : ''}
                        ${job.eta ? `
                            <div>
                                <div style="font-size: var(--text-xs); color: var(--text-secondary);">ETA</div>
                                <div style="font-size: var(--text-lg); font-weight: bold;">${job.eta}</div>
                            </div>
                        ` : ''}
                    </div>
                ` : ''}

                <div style="display: flex; gap: var(--space-sm); justify-content: flex-end;">
                    <button class="btn btn-sm btn-secondary" data-job-action="details" data-job-id="${job.job_id}">
                        📊 View Details
                    </button>
                    ${['queued', 'running', 'training'].includes(job.status) ? `
                        <button class="btn btn-sm btn-danger" data-job-action="stop" data-job-id="${job.job_id}">
                            🛑 Stop
                        </button>
                    ` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Get status icon
     * @param {string} status - Job status
     * @returns {string}
     */
    getStatusIcon(status) {
        const icons = {
            queued: '⏳',
            initializing: '🔄',
            loading_model: '📥',
            loading_dataset: '📚',
            training: '⚡',
            running: '⚡',
            saving: '💾'
        };
        return icons[status] || '🔮';
    }

    /**
     * Get status badge class
     * @param {string} status - Job status
     * @returns {string}
     */
    getStatusBadgeClass(status) {
        if (['training', 'running'].includes(status)) return 'primary';
        if (status === 'queued') return 'secondary';
        return 'info';
    }

    /**
     * Handle job action
     * @param {string} action - Action name
     * @param {string} jobId - Job ID
     */
    async handleJobAction(action, jobId) {
        switch (action) {
            case 'details':
                window.dispatchEvent(new CustomEvent('navigate', {
                    detail: { view: 'job-history', jobId }
                }));
                break;
            case 'stop':
                await this.stopJob(jobId);
                break;
        }
    }

    /**
     * Stop a job
     * @param {string} jobId - Job ID
     */
    async stopJob(jobId) {
        if (!confirm('Are you sure you want to stop this job?')) return;

        try {
            const response = await fetch(`/jobs/${jobId}/stop`, { method: 'POST' });
            const result = await response.json();

            if (response.ok) {
                window.dispatchEvent(new CustomEvent('toast', {
                    detail: { message: 'Job stop requested', type: 'success' }
                }));
                this.loadJobs();
            } else {
                throw new Error(result.detail || 'Failed to stop job');
            }
        } catch (error) {
            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: `Failed to stop job: ${error.message}`, type: 'danger' }
            }));
        }
    }

    /**
     * Cleanup on view change
     */
    destroy() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
}
