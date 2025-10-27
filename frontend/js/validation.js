// Validation Module - Input validation and sanitization

/**
 * Validation rules for form inputs
 */
const ValidationRules = {
    base_model: {
        required: true,
        pattern: /^[a-zA-Z0-9\-_\/\.]+$/,
        message: 'Please enter a valid model name (alphanumeric, hyphens, underscores, slashes)'
    },
    output_name: {
        required: true,
        pattern: /^[a-zA-Z0-9\-_]+$/,
        minLength: 3,
        maxLength: 100,
        message: 'Model name must be 3-100 characters (alphanumeric, hyphens, underscores only)'
    },
    learning_rate: {
        required: true,
        type: 'number',
        min: 0.000001,
        max: 0.1,
        message: 'Learning rate must be between 0.000001 and 0.1'
    },
    epochs: {
        required: true,
        type: 'number',
        min: 1,
        max: 100,
        message: 'Epochs must be between 1 and 100'
    },
    batch_size: {
        required: true,
        type: 'number',
        min: 1,
        max: 32,
        message: 'Batch size must be between 1 and 32'
    },
    grad_accum: {
        required: true,
        type: 'number',
        min: 1,
        max: 256,
        message: 'Gradient accumulation must be between 1 and 256'
    },
    max_length: {
        required: true,
        type: 'number',
        min: 128,
        max: 32768,
        message: 'Max length must be between 128 and 32768'
    },
    max_prompt_length: {
        required: true,
        type: 'number',
        min: 64,
        max: 16384,
        message: 'Max prompt length must be between 64 and 16384'
    },
    beta: {
        required: true,
        type: 'number',
        min: 0.01,
        max: 10,
        message: 'Beta must be between 0.01 and 10'
    },
    lora_r: {
        type: 'number',
        min: 8,
        max: 512,
        message: 'LoRA rank must be between 8 and 512'
    },
    lora_alpha: {
        type: 'number',
        min: 8,
        max: 512,
        message: 'LoRA alpha must be between 8 and 512'
    },
    lora_dropout: {
        type: 'number',
        min: 0,
        max: 0.5,
        message: 'LoRA dropout must be between 0 and 0.5'
    },
    test_size: {
        type: 'number',
        min: 0.001,
        max: 0.5,
        message: 'Test size must be between 0.001 and 0.5'
    }
};

/**
 * Validator class for form validation
 */
class Validator {
    /**
     * Validate a single field
     */
    static validateField(fieldId, value, rules) {
        const errors = [];

        // Required check
        if (rules.required && (!value || value.toString().trim() === '')) {
            errors.push('This field is required');
            return errors;
        }

        // Skip other checks if not required and empty
        if (!value || value.toString().trim() === '') {
            return errors;
        }

        // Type check
        if (rules.type === 'number') {
            const num = parseFloat(value);
            if (isNaN(num)) {
                errors.push('Must be a valid number');
                return errors;
            }

            // Min/max checks
            if (rules.min !== undefined && num < rules.min) {
                errors.push(`Must be at least ${rules.min}`);
            }
            if (rules.max !== undefined && num > rules.max) {
                errors.push(`Must be at most ${rules.max}`);
            }
        }

        // String checks
        if (typeof value === 'string') {
            // Length checks
            if (rules.minLength && value.length < rules.minLength) {
                errors.push(`Must be at least ${rules.minLength} characters`);
            }
            if (rules.maxLength && value.length > rules.maxLength) {
                errors.push(`Must be at most ${rules.maxLength} characters`);
            }

            // Pattern check
            if (rules.pattern && !rules.pattern.test(value)) {
                errors.push(rules.message || 'Invalid format');
            }
        }

        return errors;
    }

    /**
     * Validate entire form
     */
    static validateForm(formData) {
        const errors = {};

        for (const [fieldId, value] of Object.entries(formData)) {
            const rules = ValidationRules[fieldId];
            if (rules) {
                const fieldErrors = this.validateField(fieldId, value, rules);
                if (fieldErrors.length > 0) {
                    errors[fieldId] = fieldErrors;
                }
            }
        }

        return errors;
    }

    /**
     * Show validation error on field
     */
    static showFieldError(fieldId, errors) {
        const field = document.getElementById(fieldId);
        if (!field) return;

        // Remove existing error
        this.clearFieldError(fieldId);

        // Add error styling
        field.classList.add('input-error');

        // Create error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error-message';
        errorDiv.textContent = errors.join(', ');

        // Insert after field
        field.parentNode.insertBefore(errorDiv, field.nextSibling);
    }

    /**
     * Clear validation error from field
     */
    static clearFieldError(fieldId) {
        const field = document.getElementById(fieldId);
        if (!field) return;

        field.classList.remove('input-error');

        const errorDiv = field.parentNode.querySelector('.field-error-message');
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    /**
     * Validate dataset configuration
     */
    static validateDatasetConfig(config) {
        const errors = [];

        // Check source configuration
        if (config.source.source_type === 'huggingface') {
            if (!config.source.repo_id) {
                errors.push('HuggingFace repository ID is required');
            }
        } else if (config.source.source_type === 'local_file') {
            if (!config.source.file_path) {
                errors.push('Local file path is required');
            }
        } else if (config.source.source_type === 'upload') {
            if (!config.source.dataset_id) {
                errors.push('Please upload a dataset file first');
            }
        }

        // Check column mapping
        if (config.column_mapping) {
            const hasPrompt = Object.values(config.column_mapping).includes('prompt');
            const hasChosen = Object.values(config.column_mapping).includes('chosen');
            const hasRejected = Object.values(config.column_mapping).includes('rejected');

            if (!hasPrompt) errors.push('Prompt column must be mapped');
            if (!hasChosen) errors.push('Chosen column must be mapped');
            if (!hasRejected) errors.push('Rejected column must be mapped');
        }

        return errors;
    }

    /**
     * Estimate VRAM usage based on config
     */
    static estimateVRAM(config) {
        // Base model size estimates (in GB)
        const modelSizes = {
            '1b': 2,
            '3b': 6,
            '7b': 14,
            '8b': 16,
            '13b': 26,
            '70b': 140
        };

        // Try to extract model size from name
        let baseSize = 10; // Default fallback
        for (const [size, gb] of Object.entries(modelSizes)) {
            if (config.base_model.toLowerCase().includes(size)) {
                baseSize = gb;
                break;
            }
        }

        // Adjust for quantization
        if (config.use_4bit) {
            baseSize = baseSize * 0.35; // 4-bit is roughly 35% of full size
        }

        // Add overhead for gradients and optimizer states
        const batchSize = config.batch_size * config.gradient_accumulation_steps;
        const seqLength = config.max_length || 2048;

        // Rough estimate: base model + (batch * seq * 0.001)
        const trainingOverhead = batchSize * (seqLength / 1000) * 0.5;

        const totalVRAM = baseSize + trainingOverhead;

        return {
            base: baseSize.toFixed(1),
            training: trainingOverhead.toFixed(1),
            total: totalVRAM.toFixed(1)
        };
    }
}

/**
 * Sanitize user input to prevent XSS
 */
function sanitizeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Debounce function for input validation
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

export { Validator, ValidationRules, sanitizeHTML, debounce };
