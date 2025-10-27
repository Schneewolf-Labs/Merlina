// Jobs Module - Job management and monitoring

import { MerlinaAPI, WebSocketManager } from './api.js';
import { Toast, Modal, ProgressBar, JobCardRenderer, MetricsDisplay } from './ui.js';

/**
 * Job Manager - handles job lifecycle and monitoring
 */
class JobManager {
    constructor() {
        this.activeJobs = {};
        this.currentJobId = null;
        this.pollInterval = null;
        this.wsManager = new WebSocketManager();
        this.useWebSocket = true; // Try WebSocket first, fall back to polling

        this.toast = new Toast();
        this.modal = new Modal('job-modal');
        this.progressBar = new ProgressBar('job-modal');

        this.setupEventListeners();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Stop button
        const stopBtn = document.getElementById('stop-button');
        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopCurrentJob());
        }

        // Clear all jobs button
        const clearAllBtn = document.getElementById('clear-all-jobs-btn');
        if (clearAllBtn) {
            clearAllBtn.addEventListener('click', () => this.clearAllJobs());
        }

        // Event delegation for job cards
        const jobsContainer = document.getElementById('jobs-container');
        if (jobsContainer) {
            jobsContainer.addEventListener('click', (e) => {
                const card = e.target.closest('.job-card');
                if (!card) return;

                const jobId = card.dataset.jobId;
                const action = e.target.closest('[data-action]')?.dataset.action;

                if (action === 'delete-job') {
                    e.stopPropagation();
                    this.deleteJob(jobId);
                } else if (action === 'view-job' || !action) {
                    this.showJobDetails(jobId);
                }
            });
        }
    }

    /**
     * Load all jobs from server
     */
    async loadJobs() {
        try {
            const jobs = await MerlinaAPI.getJobs();

            const jobsSection = document.getElementById('jobs-section');
            const jobsContainer = document.getElementById('jobs-container');

            if (Object.keys(jobs).length > 0) {
                jobsSection.style.display = 'block';
                jobsContainer.innerHTML = '';

                for (const [jobId, job] of Object.entries(jobs)) {
                    // Update local tracking
                    if (this.activeJobs[jobId]) {
                        this.activeJobs[jobId].status = job.status;
                        this.activeJobs[jobId].progress = job.progress;
                    }

                    // Create or update job card
                    const jobName = this.activeJobs[jobId]?.name || jobId;
                    const jobCard = JobCardRenderer.create(jobId, job, jobName);
                    jobsContainer.appendChild(jobCard);
                }
            } else {
                jobsSection.style.display = 'none';
            }
        } catch (error) {
            console.error('Failed to load jobs:', error);
            this.toast.error('Failed to load jobs');
        }
    }

    /**
     * Submit new training job
     */
    async submitJob(config) {
        try {
            const data = await MerlinaAPI.submitTraining(config);

            this.toast.success(`Training spell cast! Job ID: ${data.job_id}`);

            // Track job
            this.activeJobs[data.job_id] = {
                name: config.output_name,
                status: 'started',
                config: config
            };

            // Reload jobs list
            await this.loadJobs();

            // Open monitoring modal
            this.showJobDetails(data.job_id);

            return data.job_id;
        } catch (error) {
            console.error('Failed to submit job:', error);
            this.toast.error(`Failed to cast spell: ${error.message}`);
            throw error;
        }
    }

    /**
     * Show job details modal with real-time monitoring
     */
    showJobDetails(jobId) {
        this.currentJobId = jobId;
        this.modal.show();

        // Update modal header
        const jobName = this.activeJobs[jobId]?.name || jobId;
        document.getElementById('job-name').textContent = jobName;
        document.getElementById('job-id').textContent = jobId;

        // Try WebSocket first
        if (this.useWebSocket) {
            this.startWebSocketMonitoring(jobId);
        } else {
            this.startPollingMonitoring(jobId);
        }
    }

    /**
     * Start WebSocket monitoring for job
     */
    startWebSocketMonitoring(jobId) {
        console.log('🔌 Starting WebSocket monitoring for job:', jobId);

        this.wsManager.connect(jobId, {
            onStatus: (data) => {
                this.updateJobUI(data);
            },
            onMetrics: (data) => {
                this.updateMetrics(data);
            },
            onCompleted: (data) => {
                this.handleJobCompleted(data);
            },
            onError: (message) => {
                console.warn('WebSocket error, falling back to polling:', message);
                this.useWebSocket = false;
                this.startPollingMonitoring(jobId);
            }
        });

        // Also do initial polling as fallback
        this.updateJobStatus();
    }

    /**
     * Start polling monitoring for job
     */
    startPollingMonitoring(jobId) {
        console.log('📊 Starting polling monitoring for job:', jobId);

        // Clear any existing interval
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        // Initial update
        this.updateJobStatus();

        // Poll every 3 seconds
        this.pollInterval = setInterval(() => {
            this.updateJobStatus();
        }, 3000);
    }

    /**
     * Update job status from API
     */
    async updateJobStatus() {
        if (!this.currentJobId) return;

        try {
            const status = await MerlinaAPI.getJobStatus(this.currentJobId);
            this.updateJobUI(status);

            // If job is complete, stop monitoring
            if (['completed', 'failed', 'stopped'].includes(status.status)) {
                this.stopMonitoring();
            }

            // Update jobs list
            await this.loadJobs();
        } catch (error) {
            console.error('Failed to update job status:', error);
        }
    }

    /**
     * Update job UI elements
     */
    updateJobUI(status) {
        // Update progress bar
        const progressPercent = Math.round((status.progress || 0) * 100);
        this.progressBar.update(progressPercent);

        // Update status text
        const statusText = document.getElementById('status-text');
        if (statusText) {
            statusText.textContent = `Status: ${status.status}`;
        }

        // Update metrics
        MetricsDisplay.update(status);

        // Update W&B link
        const wandbLink = document.getElementById('wandb-link');
        if (wandbLink && this.activeJobs[this.currentJobId]?.config?.use_wandb) {
            if (status.wandb_url) {
                wandbLink.style.display = 'block';
                wandbLink.href = status.wandb_url;
            } else {
                wandbLink.style.display = 'none';
            }
        }

        // Handle stop button state
        const stopButton = document.getElementById('stop-button');
        if (stopButton) {
            if (status.status === 'completed') {
                stopButton.disabled = true;
                this.toast.success('Training completed successfully!');
            } else if (status.status === 'failed') {
                stopButton.disabled = true;
                this.toast.error(`Training failed: ${status.error || 'Unknown error'}`);
            } else if (status.status === 'stopped') {
                stopButton.disabled = true;
                this.toast.warning(`Training stopped at step ${status.current_step || '?'}`);
            } else if (status.status === 'stopping') {
                stopButton.disabled = true;
                stopButton.textContent = '⏸️ Stopping...';
            } else if (['training', 'initializing', 'loading_model', 'loading_dataset'].includes(status.status)) {
                stopButton.disabled = false;
                stopButton.textContent = '🛑 Stop Training';
            }
        }
    }

    /**
     * Update metrics display
     */
    updateMetrics(data) {
        MetricsDisplay.update(data.metrics || data);
    }

    /**
     * Handle job completed event
     */
    handleJobCompleted(data) {
        this.toast.success('Training completed successfully!');
        this.stopMonitoring();
        this.loadJobs();
    }

    /**
     * Stop monitoring current job
     */
    stopMonitoring() {
        // Stop WebSocket
        if (this.wsManager.isConnected()) {
            this.wsManager.disconnect();
        }

        // Stop polling
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    /**
     * Close job modal
     */
    closeJobModal() {
        this.stopMonitoring();
        this.modal.hide();
        this.currentJobId = null;

        // Reset stop button
        const stopButton = document.getElementById('stop-button');
        if (stopButton) {
            stopButton.disabled = false;
            stopButton.textContent = '🛑 Stop Training';
        }
    }

    /**
     * Stop current job
     */
    async stopCurrentJob() {
        if (!this.currentJobId) return;

        const confirmed = window.confirm(
            '⚠️ Stop training gracefully?\n\n' +
            'The training will:\n' +
            '• Complete the current step\n' +
            '• Save a checkpoint\n' +
            '• Exit cleanly\n\n' +
            'This cannot be undone.'
        );

        if (!confirmed) return;

        try {
            const result = await MerlinaAPI.stopJob(this.currentJobId);

            if (result.status === 'success') {
                this.toast.success('Stop request sent. Training will stop after current step.');

                // Disable stop button
                const stopBtn = document.getElementById('stop-button');
                if (stopBtn) {
                    stopBtn.disabled = true;
                    stopBtn.textContent = '⏸️ Stopping...';
                }
            } else {
                this.toast.warning(result.message);
            }
        } catch (error) {
            console.error('Failed to stop job:', error);
            this.toast.error(`Failed to stop training: ${error.message}`);
        }
    }

    /**
     * Delete a job
     */
    async deleteJob(jobId) {
        const confirmed = window.confirm(`Are you sure you want to delete job ${jobId}?`);
        if (!confirmed) return;

        try {
            await MerlinaAPI.deleteJob(jobId);
            this.toast.success('Job deleted successfully');

            // Remove from active jobs
            delete this.activeJobs[jobId];

            // Close modal if this job was open
            if (this.currentJobId === jobId) {
                this.closeJobModal();
            }

            // Reload jobs
            await this.loadJobs();
        } catch (error) {
            console.error('Failed to delete job:', error);
            this.toast.error(`Failed to delete job: ${error.message}`);
        }
    }

    /**
     * Clear all jobs
     */
    async clearAllJobs() {
        const confirmed = window.confirm(
            '⚠️ Are you sure you want to delete ALL jobs?\n\n' +
            'This action cannot be undone!'
        );

        if (!confirmed) return;

        try {
            await MerlinaAPI.clearAllJobs();
            this.toast.success('All jobs cleared successfully');

            // Clear local state
            this.activeJobs = {};
            this.closeJobModal();

            // Reload jobs (will hide section)
            await this.loadJobs();
        } catch (error) {
            console.error('Failed to clear jobs:', error);
            this.toast.error(`Failed to clear jobs: ${error.message}`);
        }
    }
}

export { JobManager };
