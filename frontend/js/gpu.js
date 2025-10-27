// GPU Module - GPU management and monitoring

import { MerlinaAPI } from './api.js';
import { Toast, LoadingManager, GPUDisplay } from './ui.js';

/**
 * GPU Manager - handles GPU operations
 */
class GPUManager {
    constructor() {
        this.toast = new Toast();
        this.setupEventListeners();
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Refresh GPU button
        const refreshBtn = document.getElementById('refresh-gpu-button');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.refreshGPUs());
        }
    }

    /**
     * Refresh and display GPU list
     */
    async refreshGPUs() {
        const container = document.getElementById('gpu-list-container');
        const gpuSelect = document.getElementById('gpu-selection');
        const refreshButton = document.getElementById('refresh-gpu-button');

        if (!container) return;

        LoadingManager.show(refreshButton, '🔄 Loading...');
        LoadingManager.showSkeleton(container, 2);

        try {
            const data = await MerlinaAPI.getGPUList();

            if (data.status === 'no_cuda') {
                container.innerHTML = `
                    <div class="warning-message">
                        <div class="warning-header">⚠️ No CUDA Available</div>
                        <div class="warning-body">${data.message}</div>
                    </div>
                `;

                if (gpuSelect) {
                    gpuSelect.innerHTML = '<option value="">No GPUs available</option>';
                    gpuSelect.disabled = true;
                }
            } else if (data.gpus && data.gpus.length > 0) {
                // Display GPU cards
                GPUDisplay.renderList(data.gpus, container);

                // Update select dropdown
                if (gpuSelect) {
                    GPUDisplay.populateSelect(data.gpus, gpuSelect);
                }

                this.toast.success(`Found ${data.gpus.length} GPU(s)`);
            } else {
                container.innerHTML = `
                    <div class="empty-state">
                        No GPUs detected
                    </div>
                `;

                if (gpuSelect) {
                    gpuSelect.innerHTML = '<option value="">No GPUs available</option>';
                    gpuSelect.disabled = true;
                }
            }
        } catch (error) {
            console.error('Failed to load GPUs:', error);
            container.innerHTML = `
                <div class="error-message">
                    <div class="error-header">❌ Error Loading GPUs</div>
                    <div class="error-body">${error.message}</div>
                </div>
            `;
            this.toast.error('Failed to load GPU information');
        } finally {
            LoadingManager.hide(refreshButton);
        }
    }

    /**
     * Get selected GPU IDs for training config
     */
    getSelectedGPUs() {
        const gpuSelect = document.getElementById('gpu-selection');
        if (!gpuSelect) return null;

        const selected = Array.from(gpuSelect.selectedOptions)
            .map(opt => opt.value)
            .filter(val => val !== ''); // Filter out "Auto" option

        // If nothing selected or "Auto" is selected, return null (use all GPUs)
        return selected.length > 0 ? selected.map(Number) : null;
    }

    /**
     * Refresh GPU stats while monitoring a job
     */
    async refreshJobGPUs() {
        const container = document.getElementById('gpu-monitor-cards');
        if (!container) return;

        LoadingManager.showSkeleton(container, 1);

        try {
            const data = await MerlinaAPI.getGPUList();

            if (data.status === 'success' && data.gpus && data.gpus.length > 0) {
                GPUDisplay.renderList(data.gpus, container);

                // Show the section if hidden
                const section = document.getElementById('gpu-monitor-section');
                if (section) {
                    section.style.display = 'block';
                }
            } else {
                container.innerHTML = '<div class="empty-state">No GPU data available</div>';
            }
        } catch (error) {
            console.error('Failed to refresh GPU stats:', error);
            container.innerHTML = `<div class="error-message">Error: ${error.message}</div>`;
        }
    }
}

// Make refreshJobGPUs available globally for onclick handler in HTML
window.refreshJobGPUs = function() {
    if (window.gpuManager) {
        window.gpuManager.refreshJobGPUs();
    }
};

export { GPUManager };
