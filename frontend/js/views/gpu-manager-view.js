/**
 * GPU Manager View - Monitor and manage GPU resources
 * Merlina Modular Frontend v2.0
 */

export class GPUManagerView {
    constructor() {
        this.gpus = [];
        this.refreshInterval = null;
    }

    /**
     * Render GPU manager view
     * @returns {string}
     */
    render() {
        return `
            <div class="gpu-manager-view">
                ${this.renderHeader()}
                ${this.renderGPUList()}
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
                        <h2 class="card-title">🎮 GPU Manager</h2>
                        <p class="card-subtitle">Monitor GPU utilization and memory usage</p>
                    </div>
                    <button class="btn btn-secondary" id="refresh-gpus-btn">
                        🔄 Refresh
                    </button>
                </div>
            </div>
        `;
    }

    /**
     * Render GPU list
     * @returns {string}
     */
    renderGPUList() {
        return `
            <div id="gpu-list-container">
                <div style="text-align: center; padding: var(--space-2xl); color: var(--text-secondary);">
                    Loading GPUs...
                </div>
            </div>
        `;
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Refresh button
        const refreshBtn = document.getElementById('refresh-gpus-btn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => {
                this.loadGPUs();
            });
        }

        // Initial load
        this.loadGPUs();

        // Auto-refresh every 3 seconds
        this.refreshInterval = setInterval(() => {
            this.loadGPUs();
        }, 3000);
    }

    /**
     * Load GPUs from API
     */
    async loadGPUs() {
        try {
            const response = await fetch('/gpu/list');
            const data = await response.json();

            this.gpus = data.gpus || [];
            this.renderGPUData(this.gpus);
        } catch (error) {
            console.error('Failed to load GPUs:', error);
            const container = document.getElementById('gpu-list-container');
            if (container) {
                container.innerHTML = `
                    <div class="card">
                        <div class="alert alert-danger">
                            Failed to load GPU information. Make sure CUDA is available.
                        </div>
                    </div>
                `;
            }
        }
    }

    /**
     * Render GPU data
     * @param {array} gpus - GPUs array
     */
    renderGPUData(gpus) {
        const container = document.getElementById('gpu-list-container');
        if (!container) return;

        if (!gpus || gpus.length === 0) {
            container.innerHTML = `
                <div class="card">
                    <div class="empty-state">
                        <div class="empty-state-icon">🔌</div>
                        <div class="empty-state-title">No GPUs Detected</div>
                        <div class="empty-state-message">
                            Make sure CUDA-capable GPUs are installed and drivers are up to date
                        </div>
                    </div>
                </div>
            `;
            return;
        }

        container.innerHTML = gpus.map((gpu, index) => this.renderGPUCard(gpu, index)).join('');
    }

    /**
     * Render individual GPU card
     * @param {object} gpu - GPU data
     * @param {number} index - GPU index
     * @returns {string}
     */
    renderGPUCard(gpu, index) {
        const memoryPercent = ((gpu.memory_used_gb / gpu.memory_total_gb) * 100).toFixed(1);
        const isBusy = gpu.memory_free_gb < 2;
        const statusColor = isBusy ? 'var(--danger)' : 'var(--success)';

        return `
            <div class="card" style="margin-bottom: var(--space-lg);">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: var(--space-lg);">
                    <div>
                        <h3 style="margin: 0 0 var(--space-xs) 0; display: flex; align-items: center; gap: var(--space-sm);">
                            🎮 GPU ${index}
                            <span class="badge" style="background: ${statusColor}; color: white;">
                                ${isBusy ? 'BUSY' : 'AVAILABLE'}
                            </span>
                        </h3>
                        <div style="font-size: var(--text-lg); font-weight: bold; margin-bottom: var(--space-sm);">
                            ${gpu.name}
                        </div>
                        <div style="font-size: var(--text-sm); color: var(--text-secondary);">
                            Compute Capability: ${gpu.compute_capability || 'N/A'}
                        </div>
                    </div>
                </div>

                <!-- Memory Usage -->
                <div style="margin-bottom: var(--space-lg);">
                    <div style="display: flex; justify-content: space-between; margin-bottom: var(--space-sm);">
                        <span style="font-weight: bold;">💾 Memory Usage</span>
                        <span>${memoryPercent}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${memoryPercent}%; background: ${this.getMemoryColor(memoryPercent)};"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-top: var(--space-xs); font-size: var(--text-sm); color: var(--text-secondary);">
                        <span>Used: ${gpu.memory_used_gb.toFixed(1)} GB</span>
                        <span>Free: ${gpu.memory_free_gb.toFixed(1)} GB</span>
                        <span>Total: ${gpu.memory_total_gb.toFixed(1)} GB</span>
                    </div>
                </div>

                ${gpu.utilization_percent !== undefined ? `
                    <!-- GPU Utilization -->
                    <div style="margin-bottom: var(--space-lg);">
                        <div style="display: flex; justify-content: space-between; margin-bottom: var(--space-sm);">
                            <span style="font-weight: bold;">⚡ GPU Utilization</span>
                            <span>${gpu.utilization_percent}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${gpu.utilization_percent}%; background: ${this.getUtilizationColor(gpu.utilization_percent)};"></div>
                        </div>
                    </div>
                ` : ''}

                ${gpu.temperature_c !== undefined ? `
                    <!-- Temperature -->
                    <div style="margin-bottom: var(--space-lg);">
                        <div style="display: flex; justify-content: space-between; margin-bottom: var(--space-sm);">
                            <span style="font-weight: bold;">🌡️ Temperature</span>
                            <span>${gpu.temperature_c}°C</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${Math.min(gpu.temperature_c, 100)}%; background: ${this.getTemperatureColor(gpu.temperature_c)};"></div>
                        </div>
                    </div>
                ` : ''}

                <!-- Stats Grid -->
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: var(--space-md); padding: var(--space-md); background: var(--surface-2); border-radius: var(--radius-md);">
                    <div>
                        <div style="font-size: var(--text-xs); color: var(--text-secondary);">Status</div>
                        <div style="font-weight: bold;">${isBusy ? '🔴 Busy' : '🟢 Available'}</div>
                    </div>
                    <div>
                        <div style="font-size: var(--text-xs); color: var(--text-secondary);">Free Memory</div>
                        <div style="font-weight: bold;">${gpu.memory_free_gb.toFixed(1)} GB</div>
                    </div>
                    ${gpu.power_draw_w !== undefined ? `
                        <div>
                            <div style="font-size: var(--text-xs); color: var(--text-secondary);">Power Draw</div>
                            <div style="font-weight: bold;">${gpu.power_draw_w} W</div>
                        </div>
                    ` : ''}
                    ${gpu.fan_speed_percent !== undefined ? `
                        <div>
                            <div style="font-size: var(--text-xs); color: var(--text-secondary);">Fan Speed</div>
                            <div style="font-weight: bold;">${gpu.fan_speed_percent}%</div>
                        </div>
                    ` : ''}
                </div>

                ${gpu.processes && gpu.processes.length > 0 ? `
                    <div style="margin-top: var(--space-lg);">
                        <div style="font-weight: bold; margin-bottom: var(--space-sm);">🔄 Running Processes (${gpu.processes.length})</div>
                        <div style="background: var(--surface-2); border-radius: var(--radius-md); padding: var(--space-sm);">
                            ${gpu.processes.map(proc => `
                                <div style="padding: var(--space-xs); font-family: var(--font-mono); font-size: var(--text-sm);">
                                    PID ${proc.pid}: ${proc.used_memory_mb} MB
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    /**
     * Get color for memory usage
     * @param {number} percent - Memory usage percentage
     * @returns {string}
     */
    getMemoryColor(percent) {
        if (percent < 60) return 'var(--success)';
        if (percent < 85) return 'var(--warning)';
        return 'var(--danger)';
    }

    /**
     * Get color for utilization
     * @param {number} percent - Utilization percentage
     * @returns {string}
     */
    getUtilizationColor(percent) {
        if (percent < 50) return 'var(--info)';
        if (percent < 80) return 'var(--primary-purple)';
        return 'var(--success)';
    }

    /**
     * Get color for temperature
     * @param {number} temp - Temperature in Celsius
     * @returns {string}
     */
    getTemperatureColor(temp) {
        if (temp < 60) return 'var(--success)';
        if (temp < 75) return 'var(--warning)';
        return 'var(--danger)';
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
