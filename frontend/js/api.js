// API Module - Handles all API communication
// Dynamically detect API URL from current page location
const API_URL = '';  // Empty string = relative URLs (same origin)
const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

console.log('🔧 Merlina API Configuration:');
console.log(`  API URL: ${API_URL || window.location.origin} (relative)`);
console.log(`  WebSocket URL: ${WS_URL}`);

/**
 * API Client for Merlina backend
 */
class MerlinaAPI {
    /**
     * Generic fetch wrapper with error handling
     */
    static async fetch(endpoint, options = {}) {
        try {
            const response = await fetch(`${API_URL}${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    }

    // Training endpoints
    static async submitTraining(config) {
        return this.fetch('/train', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    static async getJobStatus(jobId) {
        return this.fetch(`/status/${jobId}`);
    }

    static async stopJob(jobId) {
        return this.fetch(`/jobs/${jobId}/stop`, { method: 'POST' });
    }

    static async getJobs() {
        return this.fetch('/jobs');
    }

    static async deleteJob(jobId) {
        return this.fetch(`/jobs/${jobId}`, { method: 'DELETE' });
    }

    static async clearAllJobs() {
        return this.fetch('/jobs', { method: 'DELETE' });
    }

    // Model endpoints
    static async preloadModel(modelName, hfToken = null) {
        return this.fetch('/model/preload', {
            method: 'POST',
            body: JSON.stringify({
                model_name: modelName,
                hf_token: hfToken
            })
        });
    }

    // Dataset endpoints
    static async uploadDataset(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_URL}/dataset/upload-file`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Upload failed: ${response.status}`);
        }

        return await response.json();
    }

    static async getDatasetColumns(config) {
        return this.fetch('/dataset/columns', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    static async previewDataset(config) {
        return this.fetch('/dataset/preview', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    static async previewFormattedDataset(config) {
        return this.fetch('/dataset/preview-formatted', {
            method: 'POST',
            body: JSON.stringify(config)
        });
    }

    // GPU endpoints
    static async getGPUList() {
        return this.fetch('/gpu/list');
    }

    // Config management endpoints
    static async saveConfig(name, config, description = '', tags = []) {
        return this.fetch('/configs/save', {
            method: 'POST',
            body: JSON.stringify({ name, config, description, tags })
        });
    }

    static async listConfigs() {
        return this.fetch('/configs/list');
    }

    static async loadConfig(name) {
        return this.fetch(`/configs/${name}`);
    }

    static async deleteConfig(name) {
        return this.fetch(`/configs/${name}`, { method: 'DELETE' });
    }
}

/**
 * WebSocket Manager for real-time updates
 */
class WebSocketManager {
    constructor() {
        this.socket = null;
        this.jobId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.callbacks = {
            onStatus: null,
            onMetrics: null,
            onCompleted: null,
            onError: null
        };
    }

    /**
     * Connect to WebSocket for a specific job
     */
    connect(jobId, callbacks = {}) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.disconnect();
        }

        this.jobId = jobId;
        this.callbacks = { ...this.callbacks, ...callbacks };

        const wsUrl = `${WS_URL}/ws/${jobId}`;
        console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);

        try {
            this.socket = new WebSocket(wsUrl);

            this.socket.onopen = () => {
                console.log(`✅ WebSocket connected for job ${jobId}`);
                this.reconnectAttempts = 0;
            };

            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            this.socket.onclose = (event) => {
                console.log(`🔌 WebSocket closed (code: ${event.code})`);

                // Attempt reconnection if not a normal closure
                if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    console.log(`🔄 Reconnecting... (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                    setTimeout(() => this.connect(jobId, callbacks), this.reconnectDelay);
                }
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            // Fall back to polling if WebSocket fails
            if (this.callbacks.onError) {
                this.callbacks.onError('WebSocket not available, using polling');
            }
        }
    }

    /**
     * Handle incoming WebSocket message
     */
    handleMessage(data) {
        switch (data.type) {
            case 'status_update':
                if (this.callbacks.onStatus) {
                    this.callbacks.onStatus(data);
                }
                break;

            case 'metrics':
                if (this.callbacks.onMetrics) {
                    this.callbacks.onMetrics(data);
                }
                break;

            case 'completed':
                if (this.callbacks.onCompleted) {
                    this.callbacks.onCompleted(data);
                }
                break;

            case 'error':
                if (this.callbacks.onError) {
                    this.callbacks.onError(data.message || 'Unknown error');
                }
                break;

            default:
                console.log('Unknown message type:', data.type);
        }
    }

    /**
     * Disconnect from WebSocket
     */
    disconnect() {
        if (this.socket) {
            this.socket.close(1000, 'Client closed connection');
            this.socket = null;
            this.jobId = null;
        }
    }

    /**
     * Check if connected
     */
    isConnected() {
        return this.socket && this.socket.readyState === WebSocket.OPEN;
    }
}

export { MerlinaAPI, WebSocketManager, API_URL, WS_URL };
