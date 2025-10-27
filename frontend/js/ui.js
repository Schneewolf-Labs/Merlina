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
        element.disabled = true;
        element.classList.add('loading');
        element.textContent = text;
    }

    /**
     * Hide loading state
     */
    static hide(element) {
        if (!element) return;

        element.disabled = false;
        element.classList.remove('loading');
        if (element.dataset.originalText) {
            element.textContent = element.dataset.originalText;
            delete element.dataset.originalText;
        }
    }

    /**
     * Show skeleton loading for container
     */
    static showSkeleton(container, rows = 3) {
        if (!container) return;

        const skeleton = document.createElement('div');
        skeleton.className = 'skeleton-loader';
        skeleton.innerHTML = Array(rows).fill('<div class="skeleton-line"></div>').join('');

        container.innerHTML = '';
        container.appendChild(skeleton);
    }
}

/**
 * Modal manager
 */
class Modal {
    constructor(modalId) {
        this.modal = document.getElementById(modalId);
        this.closeBtn = this.modal?.querySelector('.close');

        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.hide());
        }

        // Close on outside click
        window.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.hide();
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isVisible()) {
                this.hide();
            }
        });
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
        return this.modal && this.modal.style.display !== 'none';
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
    Modal,
    ProgressBar,
    FormUI,
    JobCardRenderer,
    MetricsDisplay,
    GPUDisplay,
    Tooltip,
    confirm,
    createSparkle
};
