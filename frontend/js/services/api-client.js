/**
 * API Client - Base HTTP client for backend communication
 * Merlina Modular Frontend v2.0
 */

export class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL || this.detectBaseURL();
    }

    /**
     * Auto-detect base URL from current location
     * @returns {string}
     */
    detectBaseURL() {
        const { protocol, hostname, port } = window.location;
        return `${protocol}//${hostname}${port ? `:${port}` : ''}`;
    }

    /**
     * Make HTTP request
     * @param {string} endpoint - API endpoint
     * @param {object} options - Fetch options
     * @returns {Promise}
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;

        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        };

        const config = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, config);

            // Handle non-JSON responses
            const contentType = response.headers.get('content-type');
            const isJSON = contentType && contentType.includes('application/json');

            if (!response.ok) {
                const error = isJSON ? await response.json() : { detail: response.statusText };
                throw new APIError(error.detail || 'Request failed', response.status, error);
            }

            return isJSON ? await response.json() : await response.text();
        } catch (error) {
            if (error instanceof APIError) {
                throw error;
            }
            throw new APIError(`Network error: ${error.message}`, 0, error);
        }
    }

    /**
     * GET request
     * @param {string} endpoint - API endpoint
     * @param {object} options - Additional options
     * @returns {Promise}
     */
    async get(endpoint, options = {}) {
        return this.request(endpoint, { ...options, method: 'GET' });
    }

    /**
     * POST request
     * @param {string} endpoint - API endpoint
     * @param {object} data - Request body
     * @param {object} options - Additional options
     * @returns {Promise}
     */
    async post(endpoint, data, options = {}) {
        return this.request(endpoint, {
            ...options,
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * PUT request
     * @param {string} endpoint - API endpoint
     * @param {object} data - Request body
     * @param {object} options - Additional options
     * @returns {Promise}
     */
    async put(endpoint, data, options = {}) {
        return this.request(endpoint, {
            ...options,
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE request
     * @param {string} endpoint - API endpoint
     * @param {object} options - Additional options
     * @returns {Promise}
     */
    async delete(endpoint, options = {}) {
        return this.request(endpoint, { ...options, method: 'DELETE' });
    }

    /**
     * Upload file
     * @param {string} endpoint - API endpoint
     * @param {File} file - File to upload
     * @param {object} options - Additional options
     * @returns {Promise}
     */
    async upload(endpoint, file, options = {}) {
        const formData = new FormData();
        formData.append('file', file);

        return this.request(endpoint, {
            ...options,
            method: 'POST',
            headers: {
                // Don't set Content-Type, let browser set it with boundary
                ...options.headers
            },
            body: formData
        });
    }
}

/**
 * API Error class
 */
export class APIError extends Error {
    constructor(message, status, data) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.data = data;
    }
}

/**
 * Training API - Job management endpoints
 */
export class TrainingAPI extends APIClient {
    /**
     * Submit training job
     * @param {object} config - Training configuration
     * @param {string} priority - Job priority (low/normal/high)
     * @returns {Promise}
     */
    async submitJob(config, priority = 'normal') {
        return this.post(`/train?priority=${priority}`, config);
    }

    /**
     * Get job status
     * @param {string} jobId - Job ID
     * @returns {Promise}
     */
    async getJobStatus(jobId) {
        return this.get(`/status/${jobId}`);
    }

    /**
     * List all jobs
     * @param {string} status - Filter by status
     * @param {number} limit - Max results
     * @param {number} offset - Offset for pagination
     * @returns {Promise}
     */
    async listJobs(status = null, limit = 50, offset = 0) {
        const params = new URLSearchParams();
        if (status) params.append('status', status);
        params.append('limit', limit);
        params.append('offset', offset);

        return this.get(`/jobs/history?${params}`);
    }

    /**
     * Stop job
     * @param {string} jobId - Job ID
     * @returns {Promise}
     */
    async stopJob(jobId) {
        return this.post(`/jobs/${jobId}/stop`);
    }

    /**
     * Get job metrics
     * @param {string} jobId - Job ID
     * @returns {Promise}
     */
    async getJobMetrics(jobId) {
        return this.get(`/jobs/${jobId}/metrics`);
    }

    /**
     * Validate configuration
     * @param {object} config - Training configuration
     * @returns {Promise}
     */
    async validateConfig(config) {
        return this.post('/validate', config);
    }
}

/**
 * Dataset API - Dataset management endpoints
 */
export class DatasetAPI extends APIClient {
    /**
     * Preview raw dataset
     * @param {object} config - Dataset configuration
     * @returns {Promise}
     */
    async previewRaw(config) {
        return this.post('/dataset/preview', config);
    }

    /**
     * Preview formatted dataset
     * @param {object} config - Dataset configuration
     * @returns {Promise}
     */
    async previewFormatted(config) {
        return this.post('/dataset/preview-formatted', config);
    }

    /**
     * Upload dataset file
     * @param {File} file - Dataset file
     * @returns {Promise}
     */
    async uploadFile(file) {
        return this.upload('/dataset/upload-file', file);
    }

    /**
     * List uploaded datasets
     * @returns {Promise}
     */
    async listUploads() {
        return this.get('/dataset/uploads');
    }
}

/**
 * GPU API - GPU monitoring endpoints
 */
export class GPUAPI extends APIClient {
    /**
     * Get GPU information
     * @returns {Promise}
     */
    async getGPUs() {
        return this.get('/gpu');
    }

    /**
     * Monitor GPU stats
     * @returns {Promise}
     */
    async monitorGPUs() {
        return this.get('/gpu/monitor');
    }
}

/**
 * Model API - Model management endpoints
 */
export class ModelAPI extends APIClient {
    /**
     * Preload/validate model
     * @param {string} modelName - Model name or path
     * @param {string} hfToken - HuggingFace token (optional)
     * @returns {Promise}
     */
    async preloadModel(modelName, hfToken = null) {
        return this.post('/model/preload', {
            base_model: modelName,
            hf_token: hfToken
        });
    }
}

/**
 * Stats API - Application statistics
 */
export class StatsAPI extends APIClient {
    /**
     * Get application stats
     * @returns {Promise}
     */
    async getStats() {
        return this.get('/stats');
    }

    /**
     * Get queue status
     * @returns {Promise}
     */
    async getQueueStatus() {
        return this.get('/queue/status');
    }
}

// Export singleton instances
export const trainingAPI = new TrainingAPI();
export const datasetAPI = new DatasetAPI();
export const gpuAPI = new GPUAPI();
export const modelAPI = new ModelAPI();
export const statsAPI = new StatsAPI();
