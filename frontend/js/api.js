// API Module - Handles all API communication
// Dynamically detect API URL from current page location
const API_URL = '';  // Empty string = relative URLs (same origin)
const WS_URL = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;

// Default timeout for API requests (30 seconds)
const DEFAULT_TIMEOUT = 30000;

// Longer timeout for operations that might take more time
const LONG_TIMEOUT = 120000; // 2 minutes

console.log('Merlina API Configuration:');
console.log(`  API URL: ${API_URL || window.location.origin} (relative)`);
console.log(`  WebSocket URL: ${WS_URL}`);

/**
 * Error types for better error categorization
 */
const ErrorType = {
    NETWORK: 'network',
    TIMEOUT: 'timeout',
    SERVER: 'server',
    VALIDATION: 'validation',
    AUTH: 'auth',
    NOT_FOUND: 'not_found',
    UNKNOWN: 'unknown'
};

/**
 * Custom API Error with categorization
 */
class APIError extends Error {
    constructor(message, type = ErrorType.UNKNOWN, statusCode = null, details = null) {
        super(message);
        this.name = 'APIError';
        this.type = type;
        this.statusCode = statusCode;
        this.details = details;
    }

    /**
     * Get a user-friendly error message
     */
    getUserMessage() {
        switch (this.type) {
            case ErrorType.NETWORK:
                return 'Network error. Please check your internet connection and try again.';
            case ErrorType.TIMEOUT:
                return 'The request timed out. The server might be busy. Please try again.';
            case ErrorType.SERVER:
                return this.statusCode >= 500
                    ? 'Server error. Please try again later or contact support.'
                    : this.message;
            case ErrorType.VALIDATION:
                return `Validation error: ${this.message}`;
            case ErrorType.AUTH:
                return 'Authentication required. Please check your API tokens.';
            case ErrorType.NOT_FOUND:
                return 'The requested resource was not found.';
            default:
                return this.message || 'An unexpected error occurred.';
        }
    }
}

/**
 * Create an AbortController with timeout
 */
function createTimeoutController(timeout) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    return { controller, timeoutId };
}

/**
 * Determine error type from response or error
 */
function categorizeError(error, response = null) {
    // Network errors
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return { type: ErrorType.NETWORK, message: 'Network connection failed' };
    }

    // Timeout
    if (error.name === 'AbortError') {
        return { type: ErrorType.TIMEOUT, message: 'Request timed out' };
    }

    // HTTP status code based categorization
    if (response) {
        const status = response.status;
        if (status === 401 || status === 403) {
            return { type: ErrorType.AUTH, message: 'Authentication failed' };
        }
        if (status === 404) {
            return { type: ErrorType.NOT_FOUND, message: 'Resource not found' };
        }
        if (status === 422) {
            return { type: ErrorType.VALIDATION, message: 'Validation error' };
        }
        if (status >= 500) {
            return { type: ErrorType.SERVER, message: 'Server error' };
        }
        if (status >= 400) {
            return { type: ErrorType.SERVER, message: 'Request failed' };
        }
    }

    return { type: ErrorType.UNKNOWN, message: error.message || 'Unknown error' };
}

/**
 * API Client for Merlina backend
 */
class MerlinaAPI {
    /**
     * Generic fetch wrapper with error handling and timeout
     */
    static async fetch(endpoint, options = {}, timeout = DEFAULT_TIMEOUT) {
        const { controller, timeoutId } = createTimeoutController(timeout);

        try {
            const response = await fetch(`${API_URL}${endpoint}`, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                signal: controller.signal,
                ...options
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                let errorMessage = `HTTP ${response.status}`;
                let errorDetails = null;

                try {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorData.message || errorMessage;
                    errorDetails = errorData;
                } catch {
                    // Response is not JSON
                }

                const { type } = categorizeError(new Error(errorMessage), response);
                throw new APIError(errorMessage, type, response.status, errorDetails);
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);

            // Re-throw APIError as-is
            if (error instanceof APIError) {
                throw error;
            }

            // Categorize and wrap other errors
            const { type, message } = categorizeError(error);
            throw new APIError(message, type, null, { originalError: error.message });
        }
    }

    // Training endpoints
    static async submitTraining(config) {
        return this.fetch('/train', {
            method: 'POST',
            body: JSON.stringify(config)
        }, LONG_TIMEOUT);
    }

    static async getJobStatus(jobId) {
        return this.fetch(`/status/${jobId}`);
    }

    static async stopJob(jobId) {
        return this.fetch(`/jobs/${jobId}/stop`, { method: 'POST' }, LONG_TIMEOUT);
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
        }, LONG_TIMEOUT); // Model preloading can take a while
    }

    // Dataset endpoints
    static async uploadDataset(file, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);

        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            // Set up progress tracking
            if (onProgress && xhr.upload) {
                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable) {
                        onProgress(Math.round((e.loaded / e.total) * 100));
                    }
                });
            }

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        resolve(JSON.parse(xhr.responseText));
                    } catch {
                        reject(new APIError('Invalid response from server', ErrorType.SERVER));
                    }
                } else {
                    let message = `Upload failed: ${xhr.status}`;
                    try {
                        const error = JSON.parse(xhr.responseText);
                        message = error.detail || message;
                    } catch {
                        // Response is not JSON
                    }
                    reject(new APIError(message, ErrorType.SERVER, xhr.status));
                }
            });

            xhr.addEventListener('error', () => {
                reject(new APIError('Network error during upload', ErrorType.NETWORK));
            });

            xhr.addEventListener('timeout', () => {
                reject(new APIError('Upload timed out', ErrorType.TIMEOUT));
            });

            xhr.timeout = LONG_TIMEOUT;
            xhr.open('POST', `${API_URL}/dataset/upload-file`);
            xhr.send(formData);
        });
    }

    static async getDatasetColumns(config) {
        return this.fetch('/dataset/columns', {
            method: 'POST',
            body: JSON.stringify(config)
        }, LONG_TIMEOUT);
    }

    static async previewDataset(config, offset = 0, limit = 10) {
        return this.fetch(`/dataset/preview?offset=${offset}&limit=${limit}`, {
            method: 'POST',
            body: JSON.stringify(config)
        }, LONG_TIMEOUT);
    }

    static async previewFormattedDataset(config, offset = 0, limit = 5) {
        return this.fetch(`/dataset/preview-formatted?offset=${offset}&limit=${limit}`, {
            method: 'POST',
            body: JSON.stringify(config)
        }, LONG_TIMEOUT);
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

    // Version endpoint
    static async getVersion() {
        return this.fetch('/version');
    }
}

/**
 * WebSocket Manager for real-time updates with exponential backoff
 */
class WebSocketManager {
    constructor() {
        this.socket = null;
        this.jobId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.baseReconnectDelay = 1000; // Start with 1 second
        this.maxReconnectDelay = 30000; // Max 30 seconds
        this.reconnectTimeoutId = null;
        this.isIntentionalClose = false;
        this.callbacks = {
            onStatus: null,
            onMetrics: null,
            onCompleted: null,
            onError: null,
            onConnectionChange: null
        };
    }

    /**
     * Calculate reconnect delay with exponential backoff and jitter
     */
    getReconnectDelay() {
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s, capped at maxReconnectDelay
        const exponentialDelay = Math.min(
            this.baseReconnectDelay * Math.pow(2, this.reconnectAttempts),
            this.maxReconnectDelay
        );

        // Add jitter (random variation) to prevent thundering herd
        const jitter = Math.random() * 0.3 * exponentialDelay;
        return Math.floor(exponentialDelay + jitter);
    }

    /**
     * Connect to WebSocket for a specific job
     */
    connect(jobId, callbacks = {}) {
        // Clean up existing connection
        if (this.socket) {
            this.isIntentionalClose = true;
            this.socket.close();
            this.socket = null;
        }

        // Clear any pending reconnect
        if (this.reconnectTimeoutId) {
            clearTimeout(this.reconnectTimeoutId);
            this.reconnectTimeoutId = null;
        }

        this.jobId = jobId;
        this.callbacks = { ...this.callbacks, ...callbacks };
        this.isIntentionalClose = false;
        this.reconnectAttempts = 0;

        this._createConnection();
    }

    /**
     * Internal method to create WebSocket connection
     */
    _createConnection() {
        const wsUrl = `${WS_URL}/ws/${this.jobId}`;
        console.log(`Connecting to WebSocket: ${wsUrl} (attempt ${this.reconnectAttempts + 1})`);

        try {
            this.socket = new WebSocket(wsUrl);

            this.socket.onopen = () => {
                console.log(`WebSocket connected for job ${this.jobId}`);
                this.reconnectAttempts = 0;

                if (this.callbacks.onConnectionChange) {
                    this.callbacks.onConnectionChange({ connected: true, reconnecting: false });
                }
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
                console.log(`WebSocket closed (code: ${event.code}, reason: ${event.reason || 'none'})`);

                // Notify about disconnection
                if (this.callbacks.onConnectionChange) {
                    this.callbacks.onConnectionChange({
                        connected: false,
                        reconnecting: !this.isIntentionalClose && this.reconnectAttempts < this.maxReconnectAttempts
                    });
                }

                // Don't reconnect if intentionally closed or job completed normally
                if (this.isIntentionalClose || event.code === 1000) {
                    return;
                }

                // Attempt reconnection with exponential backoff
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    const delay = this.getReconnectDelay();
                    this.reconnectAttempts++;

                    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

                    this.reconnectTimeoutId = setTimeout(() => {
                        this._createConnection();
                    }, delay);
                } else {
                    console.log('Max reconnection attempts reached. Falling back to polling.');
                    if (this.callbacks.onError) {
                        this.callbacks.onError('WebSocket connection lost. Updates may be delayed.');
                    }
                }
            };
        } catch (error) {
            console.error('Failed to create WebSocket:', error);

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
                // Job completed, no need to reconnect
                this.isIntentionalClose = true;
                break;

            case 'error':
                if (this.callbacks.onError) {
                    this.callbacks.onError(data.message || 'Unknown error');
                }
                break;

            case 'ping':
                // Respond to ping with pong to keep connection alive
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    this.socket.send(JSON.stringify({ type: 'pong' }));
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
        this.isIntentionalClose = true;

        if (this.reconnectTimeoutId) {
            clearTimeout(this.reconnectTimeoutId);
            this.reconnectTimeoutId = null;
        }

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

    /**
     * Get connection status
     */
    getStatus() {
        if (!this.socket) return 'disconnected';

        switch (this.socket.readyState) {
            case WebSocket.CONNECTING: return 'connecting';
            case WebSocket.OPEN: return 'connected';
            case WebSocket.CLOSING: return 'closing';
            case WebSocket.CLOSED: return 'disconnected';
            default: return 'unknown';
        }
    }
}

export { MerlinaAPI, WebSocketManager, APIError, ErrorType, API_URL, WS_URL };
