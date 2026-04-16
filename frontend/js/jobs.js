// Jobs Module - Job management and monitoring

import { MerlinaAPI, WebSocketManager } from './api.js';
import { Toast, Modal, ProgressBar, JobCardRenderer, MetricsDisplay, LossChart } from './ui.js';

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
        this.lossChart = new LossChart('loss-chart');

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

        // Retry button
        const retryBtn = document.getElementById('retry-button');
        if (retryBtn) {
            retryBtn.addEventListener('click', () => this.retryCurrentJob());
        }

        // Upload to Hub button
        const uploadHubBtn = document.getElementById('upload-hub-button');
        if (uploadHubBtn) {
            uploadHubBtn.addEventListener('click', () => this.showUploadModal());
        }

        // Upload confirm button
        const uploadConfirmBtn = document.getElementById('upload-hub-confirm');
        if (uploadConfirmBtn) {
            uploadConfirmBtn.addEventListener('click', () => this.uploadCurrentJob());
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

            jobsContainer.innerHTML = '';

            if (Object.keys(jobs).length > 0) {
                for (const [jobId, job] of Object.entries(jobs)) {
                    // Update or create local tracking, preserving name from API
                    if (!this.activeJobs[jobId]) {
                        this.activeJobs[jobId] = { name: job.name || jobId };
                    } else if (job.name && this.activeJobs[jobId].name === jobId) {
                        // Update name if we only had the job_id as fallback
                        this.activeJobs[jobId].name = job.name;
                    }
                    this.activeJobs[jobId].status = job.status;
                    this.activeJobs[jobId].progress = job.progress;

                    // Create or update job card
                    const jobName = this.activeJobs[jobId]?.name || jobId;
                    const jobCard = JobCardRenderer.create(jobId, job, jobName);
                    jobsContainer.appendChild(jobCard);
                }
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

        // Reset chart for new job view
        this.lossChart.reset();

        // Update modal header
        const jobName = this.activeJobs[jobId]?.name || jobId;
        document.getElementById('job-name').textContent = jobName;
        document.getElementById('job-id').textContent = jobId;

        // Load job config for upload modal pre-fill (if not already cached)
        if (!this.activeJobs[jobId]?.config) {
            MerlinaAPI.getJobConfig(jobId).then(data => {
                if (data && data.config) {
                    if (!this.activeJobs[jobId]) this.activeJobs[jobId] = {};
                    this.activeJobs[jobId].config = data.config;
                }
            }).catch(() => {});
        }

        // Load historical metrics for the chart
        this.loadJobMetrics(jobId);

        // Try WebSocket first
        if (this.useWebSocket) {
            this.startWebSocketMonitoring(jobId);
        } else {
            this.startPollingMonitoring(jobId);
        }
    }

    /**
     * Load historical metrics for chart display
     */
    async loadJobMetrics(jobId) {
        try {
            const data = await MerlinaAPI.getJobMetrics(jobId);
            if (data && data.metrics && data.metrics.length > 0) {
                this.lossChart.loadHistory(data.metrics);
            }
        } catch (error) {
            // Non-critical - chart will populate from real-time updates
            console.debug('Could not load historical metrics:', error.message);
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
                // Update the job card in the sidebar from WebSocket data
                this.updateJobCard(data.job_id || jobId, data);
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

        // Do a single initial status fetch (no polling)
        this.fetchInitialStatus();
    }

    /**
     * Fetch initial job status once (no repeated polling)
     */
    async fetchInitialStatus() {
        if (!this.currentJobId) return;

        try {
            const status = await MerlinaAPI.getJobStatus(this.currentJobId);
            this.updateJobUI(status);

            // If job is already in a terminal state, stop monitoring
            if (['completed', 'failed', 'stopped'].includes(status.status)) {
                this.stopMonitoring();
            }
        } catch (error) {
            console.error('Failed to fetch initial job status:', error);
        }
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
     * Update job status from API (used only during polling fallback)
     */
    async updateJobStatus() {
        if (!this.currentJobId) return;

        try {
            const status = await MerlinaAPI.getJobStatus(this.currentJobId);
            this.updateJobUI(status);

            // Update the job card in the sidebar
            this.updateJobCard(this.currentJobId, status);

            // If job is complete, stop monitoring
            if (['completed', 'failed', 'stopped'].includes(status.status)) {
                this.stopMonitoring();
                // Refresh full job list on terminal state
                await this.loadJobs();
            }
        } catch (error) {
            console.error('Failed to update job status:', error);
        }
    }

    /**
     * Update a single job card in the sidebar without fetching /jobs
     */
    updateJobCard(jobId, data) {
        // Use the existing JobCardRenderer.update utility
        JobCardRenderer.update(jobId, {
            status: data.status,
            progress: data.progress
        });

        // Update local tracking
        if (this.activeJobs[jobId]) {
            if (data.status) this.activeJobs[jobId].status = data.status;
            if (data.progress !== undefined) this.activeJobs[jobId].progress = data.progress;
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

        // Feed data to loss chart
        if (status.current_step && status.loss) {
            this.lossChart.addPoint(
                status.current_step,
                status.loss,
                status.eval_loss || null,
                status.learning_rate || null,
                status.gpu_memory || null
            );
        }

        // Update W&B link - only change visibility when wandb_url is explicitly present
        // WebSocket updates don't include wandb_url, so skip hiding on those
        const wandbLink = document.getElementById('wandb-link');
        if (wandbLink && 'wandb_url' in status) {
            if (status.wandb_url) {
                wandbLink.style.display = 'block';
                wandbLink.href = status.wandb_url;
            } else {
                wandbLink.style.display = 'none';
            }
        }

        // Handle stop button state
        const stopButton = document.getElementById('stop-button');
        const retryButton = document.getElementById('retry-button');
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

        // Show retry button for failed or stopped jobs
        if (retryButton) {
            retryButton.style.display = ['failed', 'stopped'].includes(status.status) ? '' : 'none';
        }

        // Show upload button for completed or stopped jobs (not during active upload)
        const uploadHubButton = document.getElementById('upload-hub-button');
        if (uploadHubButton) {
            uploadHubButton.style.display = ['completed', 'stopped'].includes(status.status) ? '' : 'none';
        }

        // Show upload error warning if present
        const uploadErrorEl = document.getElementById('upload-error-message');
        if (uploadErrorEl) {
            if (status.upload_error && status.status === 'completed') {
                uploadErrorEl.textContent = `Upload failed: ${status.upload_error}`;
                uploadErrorEl.style.display = 'block';
            } else {
                uploadErrorEl.style.display = 'none';
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
        if (data.upload_error) {
            this.toast.warning(`Training completed, but upload failed: ${data.upload_error}`);
        } else {
            this.toast.success('Training completed successfully!');
        }
        this.stopMonitoring();
        // Refresh job list once on completion to get final state
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
        this.lossChart.reset();

        // Reset stop button
        const stopButton = document.getElementById('stop-button');
        if (stopButton) {
            stopButton.disabled = false;
            stopButton.textContent = '🛑 Stop Training';
        }

        // Hide retry button
        const retryButton = document.getElementById('retry-button');
        if (retryButton) {
            retryButton.style.display = 'none';
        }

        // Hide upload error
        const uploadErrorEl = document.getElementById('upload-error-message');
        if (uploadErrorEl) uploadErrorEl.style.display = 'none';

        // Hide upload button
        const uploadHubButton = document.getElementById('upload-hub-button');
        if (uploadHubButton) {
            uploadHubButton.style.display = 'none';
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
     * Retry a failed or stopped job with the same config
     */
    async retryCurrentJob() {
        if (!this.currentJobId) return;

        try {
            const result = await MerlinaAPI.retryJob(this.currentJobId);

            this.toast.success(`Retrying! New job: ${result.job_id}`);

            // Track the new job
            this.activeJobs[result.job_id] = {
                name: result.job_id,
                status: 'queued'
            };

            // Close current modal and reload jobs
            this.closeJobModal();
            await this.loadJobs();

            // Open the new job's monitoring modal
            this.showJobDetails(result.job_id);
        } catch (error) {
            console.error('Failed to retry job:', error);
            this.toast.error(`Failed to retry: ${error.message}`);
        }
    }

    /**
     * Show upload modal with pre-filled values from job config
     */
    showUploadModal() {
        if (!this.currentJobId) return;

        // Pre-fill HF token from the original job config if available
        const jobConfig = this.activeJobs[this.currentJobId]?.config;
        const tokenInput = document.getElementById('upload-hf-token');
        const repoInput = document.getElementById('upload-repo-name');
        const mergeCheckbox = document.getElementById('upload-merge-lora');
        const privateCheckbox = document.getElementById('upload-private');

        if (tokenInput) {
            // Try to get token from original config or from the main form
            const mainToken = document.getElementById('hf-token')?.value || '';
            tokenInput.value = jobConfig?.hf_token || mainToken;
        }
        if (repoInput) repoInput.value = '';
        if (repoInput) repoInput.placeholder = jobConfig?.output_name || 'Leave empty to use original output name';
        if (mergeCheckbox) mergeCheckbox.checked = jobConfig?.merge_lora_before_upload ?? true;
        if (privateCheckbox) privateCheckbox.checked = jobConfig?.hf_hub_private ?? true;

        document.getElementById('upload-hub-modal').style.display = 'flex';
    }

    /**
     * Upload current job's model to HuggingFace Hub
     */
    async uploadCurrentJob() {
        if (!this.currentJobId) return;

        const hfToken = document.getElementById('upload-hf-token')?.value?.trim() || null;
        // Token may be blank if HF_TOKEN is configured in the server's .env —
        // the backend resolves it and returns 400 if neither source has one.

        const outputName = document.getElementById('upload-repo-name')?.value?.trim() || null;
        const mergeLora = document.getElementById('upload-merge-lora')?.checked ?? true;
        const isPrivate = document.getElementById('upload-private')?.checked ?? true;

        // Close upload modal
        document.getElementById('upload-hub-modal').style.display = 'none';

        try {
            const result = await MerlinaAPI.uploadJob(this.currentJobId, {
                hf_token: hfToken,
                output_name: outputName,
                merge_lora_before_upload: mergeLora,
                hf_hub_private: isPrivate
            });

            this.toast.success(result.message || 'Upload started!');

            // Hide upload button, show uploading state
            const uploadBtn = document.getElementById('upload-hub-button');
            if (uploadBtn) uploadBtn.style.display = 'none';

            // Start monitoring for upload progress
            if (this.useWebSocket) {
                this.startWebSocketMonitoring(this.currentJobId);
            } else {
                this.startPollingMonitoring(this.currentJobId);
            }
        } catch (error) {
            console.error('Failed to upload job:', error);
            this.toast.error(`Upload failed: ${error.message}`);
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
