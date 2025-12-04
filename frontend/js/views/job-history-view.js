/**
 * Job History View - Browse and analyze past training runs
 * Merlina Modular Frontend v2.0
 */

export class JobHistoryView {
    constructor() {
        this.jobs = [];
        this.filter = 'all'; // all, completed, failed
        this.limit = 20;
        this.offset = 0;
    }

    /**
     * Render job history view
     * @returns {string}
     */
    render() {
        return `
            <div class="job-history-view">
                ${this.renderHeader()}
                ${this.renderFilters()}
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
                        <h2 class="card-title">📊 Job History</h2>
                        <p class="card-subtitle">Browse past training runs and view metrics</p>
                    </div>
                    <button class="btn btn-secondary" id="refresh-history-btn">
                        🔄 Refresh
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render filters
     * @returns {string}
     */
    renderFilters() {
        return `
            <div class="card" style="margin-bottom: var(--space-lg);">
                <div style="display: flex; gap: var(--space-md); align-items: center; flex-wrap: wrap;">
                    <span style="font-weight: bold;">Filter:</span>
                    <button class="btn btn-sm ${this.filter === 'all' ? 'btn-primary' : 'btn-secondary'}"
                            data-filter="all">
                        All Jobs
                    </button>
                    <button class="btn btn-sm ${this.filter === 'completed' ? 'btn-primary' : 'btn-secondary'}"
                            data-filter="completed">
                        ✅ Completed
                    </button>
                    <button class="btn btn-sm ${this.filter === 'failed' ? 'btn-primary' : 'btn-secondary'}"
                            data-filter="failed">
                        ❌ Failed
                    </button>
                    <button class="btn btn-sm ${this.filter === 'stopped' ? 'btn-primary' : 'btn-secondary'}"
                            data-filter="stopped">
                        ⏸️ Stopped
                    </button>
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
                <div id="jobs-history-container">
                    <div style="text-align: center; padding: var(--space-2xl); color: var(--text-secondary);">
                        Loading job history...
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
        const refreshBtn = document.getElementById('refresh-history-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadJobs();
            });
        }

        // Filter buttons
        document.querySelectorAll('[data-filter]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.filter = e.target.dataset.filter;
                this.offset = 0;

                // Update button styles
                document.querySelectorAll('[data-filter]').forEach(b => {
                    b.className = 'btn btn-sm btn-secondary';
                });
                e.target.className = 'btn btn-sm btn-primary';

                this.loadJobs();
            });
        });

        // Initial load
        this.loadJobs();
    }

    /**
     * Load jobs from API
     */
    async loadJobs() {
        try {
            const status = this.filter === 'all' ? '' : this.filter;
            const response = await fetch(`/jobs/history?limit=${this.limit}&offset=${this.offset}&status=${status}`);
            const data = await response.json();

            this.jobs = data.jobs || [];
            this.renderJobsData(this.jobs, data.total_count || 0);
        } catch (error) {
            console.error('Failed to load job history:', error);
            const container = document.getElementById('jobs-history-container');
            if (container) {
                container.innerHTML = `
                    <div class="alert alert-danger">
                        Failed to load job history. Please try again.
                    </div>
                `;
            }
        }
    }

    /**
     * Render jobs data
     * @param {array} jobs - Jobs array
     * @param {number} totalCount - Total number of jobs
     */
    renderJobsData(jobs, totalCount) {
        const container = document.getElementById('jobs-history-container');
        if (!container) return;

        if (!jobs || jobs.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-icon">📭</div>
                    <div class="empty-state-title">No Jobs Found</div>
                    <div class="empty-state-message">Try changing the filter or start a new training job</div>
                </div>
            `;
            return;
        }

        container.innerHTML = `
            <div style="margin-bottom: var(--space-lg); color: var(--text-secondary);">
                Showing ${jobs.length} of ${totalCount} jobs
            </div>
            ${jobs.map(job => this.renderJobRow(job)).join('')}
            ${this.renderPagination(jobs.length, totalCount)}
        `;

        // Attach action listeners
        this.attachJobActionListeners();
    }

    /**
     * Render individual job row
     * @param {object} job - Job data
     * @returns {string}
     */
    renderJobRow(job) {
        const statusIcon = this.getStatusIcon(job.status);
        const mode = job.config?.training_mode || 'orpo';
        const modeIcon = mode === 'orpo' ? '🏆' : '📖';
        const createdAt = job.created_at ? new Date(job.created_at).toLocaleString() : 'N/A';
        const duration = job.training_time ? this.formatDuration(job.training_time) : 'N/A';

        return `
            <div class="card" style="margin-bottom: var(--space-md); padding: var(--space-lg); cursor: pointer;"
                 data-job-row="${job.job_id}">
                <div style="display: grid; grid-template-columns: 1fr auto; gap: var(--space-lg);">
                    <div>
                        <div style="display: flex; align-items: center; gap: var(--space-md); margin-bottom: var(--space-sm);">
                            <h4 style="margin: 0;">${statusIcon} ${job.config?.output_name || job.job_id}</h4>
                            <span class="badge badge-${mode}">${modeIcon} ${mode.toUpperCase()}</span>
                            <span class="badge badge-${this.getStatusBadgeClass(job.status)}">${job.status}</span>
                        </div>
                        <div style="font-size: var(--text-sm); color: var(--text-secondary);">
                            <div>Job ID: <code>${job.job_id}</code></div>
                            <div>Created: ${createdAt}</div>
                            ${job.config?.base_model ? `<div>Model: ${job.config.base_model}</div>` : ''}
                        </div>
                    </div>
                    <div style="display: flex; flex-direction: column; align-items: flex-end; gap: var(--space-sm);">
                        ${job.final_loss !== null && job.final_loss !== undefined ? `
                            <div style="text-align: right;">
                                <div style="font-size: var(--text-xs); color: var(--text-secondary);">Final Loss</div>
                                <div style="font-size: var(--text-xl); font-weight: bold; color: var(--success);">${job.final_loss.toFixed(4)}</div>
                            </div>
                        ` : ''}
                        <div style="text-align: right;">
                            <div style="font-size: var(--text-xs); color: var(--text-secondary);">Duration</div>
                            <div style="font-weight: bold;">${duration}</div>
                        </div>
                        ${job.status === 'completed' ? `
                            <button class="btn btn-sm btn-secondary" data-job-action="metrics" data-job-id="${job.job_id}">
                                📈 View Metrics
                            </button>
                        ` : ''}
                        ${job.error_message ? `
                            <button class="btn btn-sm btn-danger" data-job-action="error" data-job-id="${job.job_id}">
                                ⚠️ View Error
                            </button>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Render pagination controls
     * @param {number} currentCount - Current number of jobs shown
     * @param {number} totalCount - Total number of jobs
     * @returns {string}
     */
    renderPagination(currentCount, totalCount) {
        const hasMore = this.offset + currentCount < totalCount;
        const hasPrev = this.offset > 0;

        if (!hasMore && !hasPrev) return '';

        return `
            <div style="display: flex; justify-content: center; gap: var(--space-md); margin-top: var(--space-lg); padding-top: var(--space-lg); border-top: 1px solid var(--border-light);">
                <button class="btn btn-secondary" id="prev-page-btn" ${!hasPrev ? 'disabled' : ''}>
                    ← Previous
                </button>
                <span style="padding: var(--space-md); color: var(--text-secondary);">
                    ${this.offset + 1} - ${this.offset + currentCount} of ${totalCount}
                </span>
                <button class="btn btn-secondary" id="next-page-btn" ${!hasMore ? 'disabled' : ''}>
                    Next →
                </button>
            </div>
        `;
    }

    /**
     * Attach job action listeners
     */
    attachJobActionListeners() {
        // Job actions
        document.querySelectorAll('[data-job-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const action = e.target.dataset.jobAction;
                const jobId = e.target.dataset.jobId;
                this.handleJobAction(action, jobId);
            });
        });

        // Job row click
        document.querySelectorAll('[data-job-row]').forEach(row => {
            row.addEventListener('click', (e) => {
                const jobId = e.currentTarget.dataset.jobRow;
                this.showJobDetails(jobId);
            });
        });

        // Pagination
        const prevBtn = document.getElementById('prev-page-btn');
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                this.offset = Math.max(0, this.offset - this.limit);
                this.loadJobs();
            });
        }

        const nextBtn = document.getElementById('next-page-btn');
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                this.offset += this.limit;
                this.loadJobs();
            });
        }
    }

    /**
     * Handle job action
     * @param {string} action - Action name
     * @param {string} jobId - Job ID
     */
    async handleJobAction(action, jobId) {
        switch (action) {
            case 'metrics':
                await this.showMetrics(jobId);
                break;
            case 'error':
                await this.showError(jobId);
                break;
        }
    }

    /**
     * Show job details
     * @param {string} jobId - Job ID
     */
    async showJobDetails(jobId) {
        try {
            const response = await fetch(`/status/${jobId}`);
            const job = await response.json();

            const details = `
                📊 **Job Details**

                **Job ID:** ${job.job_id}
                **Status:** ${job.status}
                **Output Name:** ${job.config?.output_name || 'N/A'}
                **Base Model:** ${job.config?.base_model || 'N/A'}
                **Training Mode:** ${job.config?.training_mode || 'N/A'}

                **Progress:** ${job.progress?.toFixed(1) || 0}%
                **Current Step:** ${job.current_step || 0} / ${job.total_steps || '?'}
                ${job.final_loss !== null ? `**Final Loss:** ${job.final_loss?.toFixed(4)}` : ''}

                **Created:** ${job.created_at ? new Date(job.created_at).toLocaleString() : 'N/A'}
                **Training Time:** ${job.training_time ? this.formatDuration(job.training_time) : 'N/A'}
            `;

            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: details, type: 'info' }
            }));
        } catch (error) {
            console.error('Failed to load job details:', error);
        }
    }

    /**
     * Show metrics for a job
     * @param {string} jobId - Job ID
     */
    async showMetrics(jobId) {
        try {
            const response = await fetch(`/jobs/${jobId}/metrics`);
            const data = await response.json();

            if (data.metrics && data.metrics.length > 0) {
                const lastMetric = data.metrics[data.metrics.length - 1];
                const message = `
                    📈 Training Metrics for ${jobId}

                    Total Steps: ${data.metrics.length}
                    Final Loss: ${lastMetric.loss?.toFixed(4) || 'N/A'}
                    ${lastMetric.eval_loss ? `Eval Loss: ${lastMetric.eval_loss.toFixed(4)}` : ''}

                    View full metrics in the Analytics page (coming soon!)
                `;

                window.dispatchEvent(new CustomEvent('toast', {
                    detail: { message, type: 'success' }
                }));
            } else {
                window.dispatchEvent(new CustomEvent('toast', {
                    detail: { message: 'No metrics available for this job', type: 'info' }
                }));
            }
        } catch (error) {
            console.error('Failed to load metrics:', error);
            window.dispatchEvent(new CustomEvent('toast', {
                detail: { message: 'Failed to load metrics', type: 'danger' }
            }));
        }
    }

    /**
     * Show error for a job
     * @param {string} jobId - Job ID
     */
    async showError(jobId) {
        const job = this.jobs.find(j => j.job_id === jobId);
        if (job && job.error_message) {
            window.dispatchEvent(new CustomEvent('toast', {
                detail: {
                    message: `❌ Error in ${jobId}:\n\n${job.error_message}`,
                    type: 'danger'
                }
            }));
        }
    }

    /**
     * Format duration
     * @param {number} seconds - Duration in seconds
     * @returns {string}
     */
    formatDuration(seconds) {
        if (!seconds) return 'N/A';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    }

    /**
     * Get status icon
     * @param {string} status - Job status
     * @returns {string}
     */
    getStatusIcon(status) {
        const icons = {
            completed: '✅',
            failed: '❌',
            stopped: '⏸️',
            cancelled: '🚫'
        };
        return icons[status] || '🔮';
    }

    /**
     * Get status badge class
     * @param {string} status - Job status
     * @returns {string}
     */
    getStatusBadgeClass(status) {
        if (status === 'completed') return 'success';
        if (status === 'failed') return 'danger';
        if (status === 'stopped' || status === 'cancelled') return 'warning';
        return 'secondary';
    }
}
