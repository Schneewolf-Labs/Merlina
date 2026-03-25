// UI Module - Handles UI updates, notifications, and visual feedback

import { sanitizeHTML } from './validation.js';

/**
 * Toast notification manager
 */
class Toast {
    constructor() {
        this.toastElement = document.getElementById('toast');
    }

    /**
     * Show toast notification
     */
    show(message, type = 'info', duration = 5000) {
        if (!this.toastElement) return;

        this.toastElement.textContent = message;
        this.toastElement.className = `toast ${type} show`;

        if (this.timeout) {
            clearTimeout(this.timeout);
        }

        this.timeout = setTimeout(() => {
            this.toastElement.classList.remove('show');
        }, duration);
    }

    success(message, duration) {
        this.show(message, 'success', duration);
    }

    error(message, duration) {
        this.show(message, 'error', duration);
    }

    warning(message, duration) {
        this.show(message, 'warning', duration);
    }

    info(message, duration) {
        this.show(message, 'info', duration);
    }
}

/**
 * Loading state manager
 */
class LoadingManager {
    /**
     * Show loading state on element
     */
    static show(element, text = 'Loading...') {
        if (!element) return;

        element.dataset.originalText = element.textContent;
        element.dataset.originalHtml = element.innerHTML;
        element.disabled = true;
        element.classList.add('loading');
        element.innerHTML = `<span class="loading-spinner"></span> ${text}`;
        element.setAttribute('aria-busy', 'true');
    }

    /**
     * Hide loading state
     */
    static hide(element) {
        if (!element) return;

        element.disabled = false;
        element.classList.remove('loading');
        element.setAttribute('aria-busy', 'false');

        if (element.dataset.originalHtml) {
            element.innerHTML = element.dataset.originalHtml;
            delete element.dataset.originalHtml;
        } else if (element.dataset.originalText) {
            element.textContent = element.dataset.originalText;
        }
        delete element.dataset.originalText;
    }

    /**
     * Show skeleton loading for container
     */
    static showSkeleton(container, rows = 3, type = 'default') {
        if (!container) return;

        const skeleton = document.createElement('div');
        skeleton.className = 'skeleton-loader';
        skeleton.setAttribute('role', 'status');
        skeleton.setAttribute('aria-label', 'Loading content');

        let content = '';
        switch (type) {
            case 'card':
                content = `
                    <div class="skeleton-card">
                        <div class="skeleton-line" style="width: 60%; height: 24px;"></div>
                        <div class="skeleton-line" style="width: 40%; height: 16px;"></div>
                        <div class="skeleton-line short"></div>
                    </div>
                `;
                break;
            case 'table':
                content = Array(rows).fill(`
                    <div class="skeleton-row">
                        <div class="skeleton-cell" style="width: 30%;"></div>
                        <div class="skeleton-cell" style="width: 50%;"></div>
                        <div class="skeleton-cell" style="width: 20%;"></div>
                    </div>
                `).join('');
                break;
            case 'text':
                content = Array(rows).fill(0).map((_, i) =>
                    `<div class="skeleton-line ${i === rows - 1 ? 'short' : ''}"></div>`
                ).join('');
                break;
            default:
                content = Array(rows).fill('<div class="skeleton-line"></div>').join('');
        }

        skeleton.innerHTML = `<span class="sr-only">Loading...</span>${content}`;
        container.innerHTML = '';
        container.appendChild(skeleton);
    }

    /**
     * Create inline loading spinner
     */
    static createSpinner(size = 'small') {
        const spinner = document.createElement('span');
        spinner.className = `loading-spinner loading-spinner-${size}`;
        spinner.setAttribute('role', 'status');
        spinner.innerHTML = '<span class="sr-only">Loading...</span>';
        return spinner;
    }
}

/**
 * Connection status indicator
 */
class ConnectionStatus {
    constructor(containerId = null) {
        this.container = containerId ? document.getElementById(containerId) : null;
        this.status = 'connected';
    }

    /**
     * Create status indicator element
     */
    createIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'connection-status';
        indicator.setAttribute('role', 'status');
        indicator.setAttribute('aria-live', 'polite');
        indicator.innerHTML = `
            <span class="connection-dot"></span>
            <span class="connection-text">Connected</span>
        `;
        return indicator;
    }

    /**
     * Update connection status
     */
    update(status, message = null) {
        this.status = status;

        const indicators = document.querySelectorAll('.connection-status');
        indicators.forEach(indicator => {
            const dot = indicator.querySelector('.connection-dot');
            const text = indicator.querySelector('.connection-text');

            indicator.className = `connection-status connection-${status}`;
            if (dot) {
                dot.className = `connection-dot connection-dot-${status}`;
            }
            if (text) {
                text.textContent = message || this.getStatusText(status);
            }
        });
    }

    /**
     * Get status text
     */
    getStatusText(status) {
        switch (status) {
            case 'connected': return 'Connected';
            case 'connecting': return 'Connecting...';
            case 'reconnecting': return 'Reconnecting...';
            case 'disconnected': return 'Disconnected';
            case 'error': return 'Connection Error';
            default: return 'Unknown';
        }
    }
}

/**
 * Modal manager with shared escape key handler to prevent memory leaks
 */
class Modal {
    // Static set to track all modal instances
    static instances = new Set();
    static escapeHandlerInitialized = false;

    static initGlobalEscapeHandler() {
        if (Modal.escapeHandlerInitialized) return;

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                // Find the topmost visible modal and close it
                for (const modal of Modal.instances) {
                    if (modal.isVisible()) {
                        modal.hide();
                        break;
                    }
                }
            }
        });

        Modal.escapeHandlerInitialized = true;
    }

    constructor(modalId) {
        this.modal = document.getElementById(modalId);
        this.closeBtn = this.modal?.querySelector('.close');

        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.hide());
        }

        // Close on outside click - use bound function for potential cleanup
        this._outsideClickHandler = (e) => {
            if (e.target === this.modal) {
                this.hide();
            }
        };
        window.addEventListener('click', this._outsideClickHandler);

        // Register with static set and initialize global escape handler
        Modal.instances.add(this);
        Modal.initGlobalEscapeHandler();
    }

    show() {
        if (this.modal) {
            this.modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    }

    hide() {
        if (this.modal) {
            this.modal.style.display = 'none';
            document.body.style.overflow = '';
        }
    }

    isVisible() {
        if (!this.modal) return false;
        // Check computed style to handle both inline and CSS-hidden modals
        return window.getComputedStyle(this.modal).display !== 'none';
    }

    /**
     * Cleanup method to remove event listeners and unregister instance
     */
    destroy() {
        if (this._outsideClickHandler) {
            window.removeEventListener('click', this._outsideClickHandler);
        }
        Modal.instances.delete(this);
    }
}

/**
 * Progress bar manager
 */
class ProgressBar {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.fill = this.container?.querySelector('.progress-fill');
        this.text = this.container?.querySelector('.progress-text');
    }

    /**
     * Update progress (0-100)
     */
    update(percent, text = null) {
        if (!this.fill || !this.text) return;

        const clampedPercent = Math.max(0, Math.min(100, percent));
        this.fill.style.width = `${clampedPercent}%`;
        this.text.textContent = text || `${Math.round(clampedPercent)}%`;
    }

    /**
     * Set indeterminate state
     */
    setIndeterminate() {
        if (this.fill) {
            this.fill.classList.add('indeterminate');
        }
    }

    /**
     * Clear indeterminate state
     */
    clearIndeterminate() {
        if (this.fill) {
            this.fill.classList.remove('indeterminate');
        }
    }
}

/**
 * Form utilities
 */
class FormUI {
    /**
     * Enable/disable entire form
     */
    static setEnabled(formElement, enabled) {
        if (!formElement) return;

        const inputs = formElement.querySelectorAll('input, select, button, textarea');
        inputs.forEach(input => {
            input.disabled = !enabled;
        });
    }

    /**
     * Clear form
     */
    static clear(formElement) {
        if (!formElement) return;

        const inputs = formElement.querySelectorAll('input, textarea');
        inputs.forEach(input => {
            if (input.type === 'checkbox' || input.type === 'radio') {
                input.checked = false;
            } else {
                input.value = '';
            }
        });

        const selects = formElement.querySelectorAll('select');
        selects.forEach(select => {
            select.selectedIndex = 0;
        });
    }

    /**
     * Get form data as object
     */
    static getData(formElement) {
        if (!formElement) return {};

        const data = {};
        const inputs = formElement.querySelectorAll('input, select, textarea');

        inputs.forEach(input => {
            if (input.type === 'checkbox') {
                data[input.id] = input.checked;
            } else if (input.type === 'number') {
                data[input.id] = input.value ? parseFloat(input.value) : null;
            } else {
                data[input.id] = input.value;
            }
        });

        return data;
    }

    /**
     * Set form data from object
     */
    static setData(formElement, data) {
        if (!formElement || !data) return;

        for (const [id, value] of Object.entries(data)) {
            const input = formElement.querySelector(`#${id}`);
            if (!input) continue;

            if (input.type === 'checkbox') {
                input.checked = !!value;
            } else {
                input.value = value;
            }
        }
    }
}

/**
 * Job card renderer
 */
class JobCardRenderer {
    /**
     * Create job card element
     */
    static create(jobId, job, jobName) {
        const card = document.createElement('div');
        card.className = 'job-card';
        card.dataset.jobId = jobId;

        const statusClass = `status-${job.status}`;
        const progressPercent = Math.round((job.progress || 0) * 100);

        card.innerHTML = `
            <div class="job-card-header">
                <div style="flex: 1; cursor: pointer;" data-action="view-job">
                    <h4>${sanitizeHTML(jobName || jobId)}</h4>
                    <span class="job-status-badge ${statusClass}">${sanitizeHTML(job.status)}</span>
                </div>
                <button class="delete-job-btn" data-action="delete-job" title="Delete job" aria-label="Delete job ${jobId}">
                    🗑️
                </button>
            </div>
            <div class="progress-bar" style="cursor: pointer;" data-action="view-job">
                <div class="progress-fill" style="width: ${progressPercent}%"></div>
                <span class="progress-text">${progressPercent}%</span>
            </div>
        `;

        return card;
    }

    /**
     * Update existing job card
     */
    static update(jobId, job) {
        const card = document.querySelector(`[data-job-id="${jobId}"]`);
        if (!card) return;

        const statusBadge = card.querySelector('.job-status-badge');
        const progressFill = card.querySelector('.progress-fill');
        const progressText = card.querySelector('.progress-text');

        if (statusBadge) {
            statusBadge.className = `job-status-badge status-${job.status}`;
            statusBadge.textContent = sanitizeHTML(job.status);
        }

        if (progressFill && progressText) {
            const progressPercent = Math.round((job.progress || 0) * 100);
            progressFill.style.width = `${progressPercent}%`;
            progressText.textContent = `${progressPercent}%`;
        }
    }
}

/**
 * Metrics display
 */
class MetricsDisplay {
    /**
     * Update metrics grid
     */
    static update(metricsData) {
        const elements = {
            'current-step': metricsData.current_step || '-',
            'loss-value': metricsData.loss ? metricsData.loss.toFixed(4) : '-',
            'job-status': sanitizeHTML(metricsData.status || '-'),
            'gpu-memory-value': metricsData.gpu_memory ? `${metricsData.gpu_memory.toFixed(2)} GB` : '-'
        };

        for (const [id, value] of Object.entries(elements)) {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        }
    }

    /**
     * Format metric value
     */
    static formatValue(value, type = 'number') {
        if (value === null || value === undefined) return '-';

        switch (type) {
            case 'number':
                return typeof value === 'number' ? value.toFixed(4) : value;
            case 'percent':
                return `${(value * 100).toFixed(2)}%`;
            case 'memory':
                return `${value.toFixed(2)} GB`;
            case 'time':
                return this.formatDuration(value);
            default:
                return value.toString();
        }
    }

    /**
     * Format duration in seconds
     */
    static formatDuration(seconds) {
        if (!seconds) return '-';

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
}

/**
 * Loss chart visualization using Chart.js
 */
class LossChart {
    constructor(canvasId = 'loss-chart') {
        this.canvasId = canvasId;
        this.chart = null;
        this.data = {
            steps: [],
            loss: [],
            evalLoss: [],
            learningRate: [],
            gpuMemory: []
        };
        this.showLR = false;
        this.showGPU = false;

        this.setupToggleButtons();
    }

    setupToggleButtons() {
        const lrBtn = document.getElementById('chart-toggle-lr');
        const gpuBtn = document.getElementById('chart-toggle-gpu');

        if (lrBtn) {
            lrBtn.addEventListener('click', () => {
                this.showLR = !this.showLR;
                lrBtn.classList.toggle('active', this.showLR);
                this.updateVisibility();
            });
        }
        if (gpuBtn) {
            gpuBtn.addEventListener('click', () => {
                this.showGPU = !this.showGPU;
                gpuBtn.classList.toggle('active', this.showGPU);
                this.updateVisibility();
            });
        }
    }

    getThemeColors() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        return {
            loss: '#c042ff',
            evalLoss: '#ff6b9d',
            lr: '#4caf50',
            gpu: '#2196f3',
            grid: isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.08)',
            text: isDark ? '#c0c0c0' : '#555'
        };
    }

    initChart() {
        if (this.chart) return;

        const canvas = document.getElementById(this.canvasId);
        if (!canvas || typeof Chart === 'undefined') return;

        const colors = this.getThemeColors();

        this.chart = new Chart(canvas, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: colors.loss,
                        backgroundColor: colors.loss + '20',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHitRadius: 8,
                        tension: 0.3,
                        fill: true,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Eval Loss',
                        data: [],
                        borderColor: colors.evalLoss,
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 3,
                        pointBackgroundColor: colors.evalLoss,
                        tension: 0.3,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Learning Rate',
                        data: [],
                        borderColor: colors.lr,
                        backgroundColor: 'transparent',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.3,
                        hidden: true,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'GPU Memory (GB)',
                        data: [],
                        borderColor: colors.gpu,
                        backgroundColor: colors.gpu + '15',
                        borderWidth: 1.5,
                        pointRadius: 0,
                        tension: 0.3,
                        fill: true,
                        hidden: true,
                        yAxisID: 'y2'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: colors.text,
                            usePointStyle: true,
                            pointStyle: 'line',
                            padding: 12,
                            font: { size: 11 },
                            filter: (item) => !item.hidden || item.datasetIndex <= 1
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0,0,0,0.8)',
                        titleFont: { size: 12 },
                        bodyFont: { size: 11 },
                        padding: 10,
                        callbacks: {
                            title: (items) => `Step ${items[0].label}`,
                            label: (ctx) => {
                                const val = ctx.parsed.y;
                                if (val === null || val === undefined) return null;
                                if (ctx.datasetIndex === 2) return `LR: ${val.toExponential(2)}`;
                                if (ctx.datasetIndex === 3) return `GPU: ${val.toFixed(2)} GB`;
                                return `${ctx.dataset.label}: ${val.toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Step', color: colors.text },
                        ticks: { color: colors.text, maxTicksLimit: 10 },
                        grid: { color: colors.grid }
                    },
                    y: {
                        title: { display: true, text: 'Loss', color: colors.text },
                        ticks: { color: colors.text },
                        grid: { color: colors.grid },
                        beginAtZero: false
                    },
                    y1: {
                        display: false,
                        position: 'right',
                        title: { display: true, text: 'Learning Rate', color: colors.text },
                        ticks: { color: colors.text },
                        grid: { drawOnChartArea: false }
                    },
                    y2: {
                        display: false,
                        position: 'right',
                        title: { display: true, text: 'GPU (GB)', color: colors.text },
                        ticks: { color: colors.text },
                        grid: { drawOnChartArea: false },
                        beginAtZero: true
                    }
                }
            }
        });
    }

    updateVisibility() {
        if (!this.chart) return;

        // LR dataset (index 2)
        this.chart.data.datasets[2].hidden = !this.showLR;
        this.chart.options.scales.y1.display = this.showLR;

        // GPU dataset (index 3)
        this.chart.data.datasets[3].hidden = !this.showGPU;
        this.chart.options.scales.y2.display = this.showGPU;

        this.chart.update('none');
    }

    /**
     * Load historical metrics from API response
     */
    loadHistory(metrics) {
        if (!metrics || metrics.length === 0) return;

        this.data = { steps: [], loss: [], evalLoss: [], learningRate: [], gpuMemory: [] };

        for (const m of metrics) {
            this.data.steps.push(m.step);
            this.data.loss.push(m.loss);
            this.data.evalLoss.push(m.eval_loss);
            this.data.learningRate.push(m.learning_rate);
            this.data.gpuMemory.push(m.gpu_memory_used);
        }

        this.initChart();
        this.syncChart();
        this.showChart();
    }

    /**
     * Add a single data point from real-time updates
     */
    addPoint(step, loss, evalLoss, learningRate, gpuMemory) {
        if (step === null || step === undefined) return;
        // Avoid duplicate steps
        if (this.data.steps.length > 0 && step <= this.data.steps[this.data.steps.length - 1]) return;

        this.data.steps.push(step);
        this.data.loss.push(loss ?? null);
        this.data.evalLoss.push(evalLoss ?? null);
        this.data.learningRate.push(learningRate ?? null);
        this.data.gpuMemory.push(gpuMemory ?? null);

        this.initChart();
        this.syncChart();
        this.showChart();
    }

    syncChart() {
        if (!this.chart) return;

        this.chart.data.labels = this.data.steps;
        this.chart.data.datasets[0].data = this.data.loss;
        this.chart.data.datasets[1].data = this.data.evalLoss;
        this.chart.data.datasets[2].data = this.data.learningRate;
        this.chart.data.datasets[3].data = this.data.gpuMemory;

        this.chart.update('none');
    }

    showChart() {
        const emptyMsg = document.getElementById('loss-chart-empty');
        const canvas = document.getElementById(this.canvasId);
        if (this.data.steps.length > 0) {
            if (emptyMsg) emptyMsg.style.display = 'none';
            if (canvas) canvas.style.display = 'block';
        }
    }

    /**
     * Reset chart for a new job
     */
    reset() {
        this.data = { steps: [], loss: [], evalLoss: [], learningRate: [], gpuMemory: [] };

        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }

        const emptyMsg = document.getElementById('loss-chart-empty');
        const canvas = document.getElementById(this.canvasId);
        if (emptyMsg) emptyMsg.style.display = 'block';
        if (canvas) canvas.style.display = 'none';
    }
}

/**
 * GPU display utilities
 */
class GPUDisplay {
    /**
     * Render GPU list
     */
    static renderList(gpus, container) {
        if (!container) return;

        if (!gpus || gpus.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <p>No GPUs detected</p>
                </div>
            `;
            return;
        }

        container.innerHTML = gpus.map(gpu => `
            <div class="gpu-card">
                <div class="gpu-header">
                    <div class="gpu-name">
                        <strong>GPU ${gpu.index}:</strong> ${sanitizeHTML(gpu.name)}
                    </div>
                    <div class="gpu-badge">
                        ${sanitizeHTML(gpu.compute_capability || 'N/A')}
                    </div>
                </div>
                <div class="gpu-stats">
                    <div class="gpu-stat">
                        <span class="gpu-stat-label">Memory</span>
                        <span class="gpu-stat-value">${gpu.used_memory_mb} / ${gpu.total_memory_mb} MB</span>
                        <span class="gpu-stat-detail">${gpu.memory_utilization_percent}% used</span>
                    </div>
                    ${gpu.gpu_utilization_percent !== null ? `
                    <div class="gpu-stat">
                        <span class="gpu-stat-label">GPU Util</span>
                        <span class="gpu-stat-value">${gpu.gpu_utilization_percent}%</span>
                    </div>
                    ` : ''}
                    ${gpu.temperature_c !== null ? `
                    <div class="gpu-stat">
                        <span class="gpu-stat-label">Temperature</span>
                        <span class="gpu-stat-value">${gpu.temperature_c}°C</span>
                    </div>
                    ` : ''}
                    ${gpu.power_usage_w !== null ? `
                    <div class="gpu-stat">
                        <span class="gpu-stat-label">Power</span>
                        <span class="gpu-stat-value">${gpu.power_usage_w}W</span>
                    </div>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }

    /**
     * Populate GPU select dropdown
     */
    static populateSelect(gpus, selectElement) {
        if (!selectElement) return;

        selectElement.innerHTML = '<option value="">Auto (use all available)</option>';

        if (gpus && gpus.length > 0) {
            gpus.forEach(gpu => {
                const option = document.createElement('option');
                option.value = gpu.index;
                option.textContent = `GPU ${gpu.index}: ${gpu.name} (${gpu.free_memory_mb} MB free)`;
                selectElement.appendChild(option);
            });
            selectElement.disabled = false;
        } else {
            selectElement.innerHTML = '<option value="">No GPUs available</option>';
            selectElement.disabled = true;
        }
    }
}

/**
 * Tooltip manager
 */
class Tooltip {
    /**
     * Initialize tooltips for elements with data-tooltip attribute
     */
    static init() {
        document.querySelectorAll('[data-tooltip]').forEach(element => {
            this.attach(element);
        });
    }

    /**
     * Attach tooltip to element
     */
    static attach(element) {
        const text = element.dataset.tooltip;
        if (!text) return;

        let tooltip = null;

        element.addEventListener('mouseenter', () => {
            tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = text;
            document.body.appendChild(tooltip);

            const rect = element.getBoundingClientRect();
            tooltip.style.left = `${rect.left + rect.width / 2}px`;
            tooltip.style.top = `${rect.top - tooltip.offsetHeight - 8}px`;
        });

        element.addEventListener('mouseleave', () => {
            if (tooltip) {
                tooltip.remove();
                tooltip = null;
            }
        });
    }
}

/**
 * Confirmation dialog
 */
function confirm(message, title = 'Confirm') {
    return window.confirm(`${title}\n\n${message}`);
}

/**
 * Create sparkle effect (Easter egg)
 */
function createSparkle(element) {
    const sparkle = document.createElement('div');
    sparkle.className = 'sparkle-effect';

    const rect = element.getBoundingClientRect();
    sparkle.style.left = `${rect.left + Math.random() * rect.width}px`;
    sparkle.style.top = `${rect.top + rect.height}px`;

    document.body.appendChild(sparkle);

    setTimeout(() => sparkle.remove(), 1000);
}

export {
    Toast,
    LoadingManager,
    ConnectionStatus,
    Modal,
    ProgressBar,
    FormUI,
    JobCardRenderer,
    MetricsDisplay,
    LossChart,
    GPUDisplay,
    Tooltip,
    confirm,
    createSparkle
};
