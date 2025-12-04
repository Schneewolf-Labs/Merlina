/**
 * Job Monitoring Panel Component - Real-time job monitoring
 * Merlina Modular Frontend v2.0
 */

export class JobPanel {
    constructor() {
        this.activeJobs = new Map();
        this.wsConnections = new Map();
    }

    /**
     * Render job panel content
     * @param {array} jobs - Array of active jobs
     * @returns {string}
     */
    render(jobs = []) {
        if (!jobs || jobs.length === 0) {
            return this.renderEmpty();
        }

        return `
            <div class="job-panel">
                ${jobs.map(job => this.renderJobCard(job)).join('')}
            </div>
        `;
    }

    /**
     * Render empty state
     * @returns {string}
     */
    renderEmpty() {
        return `
            <div class="empty-state">
                <div class="empty-state-icon">🌟</div>
                <div class="empty-state-title">No Active Jobs</div>
                <div class="empty-state-message">
                    Start a training job to see live updates here
                </div>
            </div>
        `;
    }

    /**
     * Render individual job card
     * @param {object} job - Job data
     * @returns {string}
     */
    renderJobCard(job) {
        const mode = job.training_mode || 'orpo';
        const modeIcon = mode === 'orpo' ? '🏆' : '📖';
        const progress = job.progress || 0;
        const status = job.status || 'queued';

        return `
            <div class="job-card job-card-compact" data-job-id="${job.job_id}">
                <div class="job-card-header">
                    <div class="job-card-title">
                        ${this.getStatusIcon(status)} ${job.output_name || job.job_id}
                    </div>
                    <span class="badge badge-${mode}">${modeIcon} ${mode.toUpperCase()}</span>
                </div>

                <div class="job-card-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${progress}%"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: var(--space-xs); font-size: var(--text-sm);">
                        <span>${progress.toFixed(1)}%</span>
                        <span>${job.current_step || 0} / ${job.total_steps || '?'}</span>
                    </div>
                </div>

                <div class="job-card-metrics">
                    ${job.loss !== undefined ? `
                        <div class="metric-item">
                            <span class="metric-label">Loss:</span>
                            <span class="metric-value">${job.loss.toFixed(4)}</span>
                        </div>
                    ` : ''}
                    ${job.learning_rate ? `
                        <div class="metric-item">
                            <span class="metric-label">LR:</span>
                            <span class="metric-value">${job.learning_rate.toExponential(2)}</span>
                        </div>
                    ` : ''}
                    ${job.gpu_memory ? `
                        <div class="metric-item">
                            <span class="metric-label">VRAM:</span>
                            <span class="metric-value">${job.gpu_memory}</span>
                        </div>
                    ` : ''}
                    ${job.eta ? `
                        <div class="metric-item">
                            <span class="metric-label">ETA:</span>
                            <span class="metric-value">${job.eta}</span>
                        </div>
                    ` : ''}
                </div>

                ${status === 'running' || status === 'training' ? `
                    <div class="job-card-actions">
                        <button class="btn btn-sm btn-ghost" data-job-action="details" data-job-id="${job.job_id}">
                            📊 Details
                        </button>
                        <button class="btn btn-sm btn-danger" data-job-action="stop" data-job-id="${job.job_id}">
                            🛑 Stop
                        </button>
                    </div>
                ` : ''}
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
            saving: '💾',
            uploading: '📤',
            completed: '✅',
            failed: '❌',
            stopped: '⏸️',
            cancelled: '🚫'
        };
        return icons[status] || '🔮';
    }

    /**
     * Update job card with new data
     * @param {string} jobId - Job ID
     * @param {object} data - Updated job data
     */
    updateJob(jobId, data) {
        const card = document.querySelector(`[data-job-id="${jobId}"]`);
        if (!card) return;

        // Update progress
        if (data.progress !== undefined) {
            const progressFill = card.querySelector('.progress-fill');
            if (progressFill) {
                progressFill.style.width = `${data.progress}%`;
            }

            const progressText = card.querySelector('.progress-bar + div span:first-child');
            if (progressText) {
                progressText.textContent = `${data.progress.toFixed(1)}%`;
            }
        }

        // Update steps
        if (data.current_step !== undefined) {
            const stepText = card.querySelector('.progress-bar + div span:last-child');
            if (stepText) {
                stepText.textContent = `${data.current_step} / ${data.total_steps || '?'}`;
            }
        }

        // Update metrics
        this.updateMetrics(card, data);
    }

    /**
     * Update metrics in job card
     * @param {HTMLElement} card - Job card element
     * @param {object} data - Job data
     */
    updateMetrics(card, data) {
        const metricsContainer = card.querySelector('.job-card-metrics');
        if (!metricsContainer) return;

        const metrics = [];

        if (data.loss !== undefined) {
            metrics.push(`
                <div class="metric-item">
                    <span class="metric-label">Loss:</span>
                    <span class="metric-value">${data.loss.toFixed(4)}</span>
                </div>
            `);
        }

        if (data.learning_rate) {
            metrics.push(`
                <div class="metric-item">
                    <span class="metric-label">LR:</span>
                    <span class="metric-value">${data.learning_rate.toExponential(2)}</span>
                </div>
            `);
        }

        if (data.gpu_memory) {
            metrics.push(`
                <div class="metric-item">
                    <span class="metric-label">VRAM:</span>
                    <span class="metric-value">${data.gpu_memory}</span>
                </div>
            `);
        }

        if (data.eta) {
            metrics.push(`
                <div class="metric-item">
                    <span class="metric-label">ETA:</span>
                    <span class="metric-value">${data.eta}</span>
                </div>
            `);
        }

        metricsContainer.innerHTML = metrics.join('');
    }

    /**
     * Add job to panel
     * @param {object} job - Job data
     */
    addJob(job) {
        this.activeJobs.set(job.job_id, job);
        this.refreshPanel();
        this.connectWebSocket(job.job_id);
    }

    /**
     * Remove job from panel
     * @param {string} jobId - Job ID
     */
    removeJob(jobId) {
        this.activeJobs.delete(jobId);
        this.disconnectWebSocket(jobId);
        this.refreshPanel();
    }

    /**
     * Refresh panel content
     */
    refreshPanel() {
        const content = document.getElementById('panel-content');
        if (!content) return;

        const jobs = Array.from(this.activeJobs.values());
        content.innerHTML = this.render(jobs);

        // Reattach event listeners
        this.attachEventListeners();
    }

    /**
     * Attach event listeners to job cards
     */
    attachEventListeners() {
        document.querySelectorAll('[data-job-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.target.dataset.jobAction;
                const jobId = e.target.dataset.jobId;
                this.handleJobAction(action, jobId);
            });
        });
    }

    /**
     * Handle job action
     * @param {string} action - Action name
     * @param {string} jobId - Job ID
     */
    handleJobAction(action, jobId) {
        switch (action) {
            case 'details':
                window.dispatchEvent(new CustomEvent('showJobDetails', {
                    detail: { jobId }
                }));
                break;
            case 'stop':
                this.stopJob(jobId);
                break;
        }
    }

    /**
     * Stop job
     * @param {string} jobId - Job ID
     */
    async stopJob(jobId) {
        if (!confirm('Are you sure you want to stop this training job?')) {
            return;
        }

        try {
            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: 'Stopping job...', type: 'info' }
            }));

            const response = await fetch(`/jobs/${jobId}/stop`, {
                method: 'POST'
            });

            const result = await response.json();

            if (response.ok) {
                window.dispatchEvent(new CustomEvent('toast', {
                    detail: { message: 'Job stop requested', type: 'success' }
                }));

                // Update job status
                const job = this.activeJobs.get(jobId);
                if (job) {
                    job.status = 'stopping';
                    this.updateJob(jobId, job);
                }
            } else {
                throw new Error(result.detail || 'Failed to stop job');
            }
        } catch (error) {
            console.error('Failed to stop job:', error);
            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: `Failed to stop job: ${error.message}`, type: 'danger' }
            }));
        }
    }

    /**
     * Connect WebSocket for job updates
     * @param {string} jobId - Job ID
     */
    connectWebSocket(jobId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsURL = `${protocol}//${window.location.host}/ws/${jobId}`;

        try {
            const ws = new WebSocket(wsURL);

            ws.onopen = () => {
                console.log(`WebSocket connected for job ${jobId}`);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(jobId, data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            ws.onerror = (error) => {
                console.error(`WebSocket error for job ${jobId}:`, error);
            };

            ws.onclose = () => {
                console.log(`WebSocket closed for job ${jobId}`);
                this.wsConnections.delete(jobId);
            };

            this.wsConnections.set(jobId, ws);
        } catch (error) {
            console.error(`Failed to connect WebSocket for job ${jobId}:`, error);
        }
    }

    /**
     * Disconnect WebSocket
     * @param {string} jobId - Job ID
     */
    disconnectWebSocket(jobId) {
        const ws = this.wsConnections.get(jobId);
        if (ws) {
            ws.close();
            this.wsConnections.delete(jobId);
        }
    }

    /**
     * Handle WebSocket message
     * @param {string} jobId - Job ID
     * @param {object} data - Message data
     */
    handleWebSocketMessage(jobId, data) {
        const job = this.activeJobs.get(jobId);
        if (!job) return;

        // Update job data
        Object.assign(job, data);

        // Update UI
        this.updateJob(jobId, data);

        // Handle completion
        if (data.status === 'completed' || data.status === 'failed' || data.status === 'stopped') {
            setTimeout(() => {
                this.removeJob(jobId);
            }, 5000); // Remove after 5 seconds
        }
    }

    /**
     * Cleanup on destroy
     */
    destroy() {
        // Disconnect all WebSockets
        this.wsConnections.forEach((ws, jobId) => {
            this.disconnectWebSocket(jobId);
        });
    }
}
