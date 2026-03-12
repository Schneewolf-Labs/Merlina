// Unit tests for validation.js - Pure logic tests using Node's built-in test runner
// Run with: node --test tests/frontend/test_validation.mjs

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// ─── Minimal DOM shim for sanitizeHTML ───────────────────────────────────────
// sanitizeHTML uses document.createElement('div') + textContent/innerHTML
const elementShim = () => ({
    _text: '',
    set textContent(val) { this._text = val; },
    get innerHTML() {
        return this._text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
});

globalThis.document = {
    createElement: () => elementShim(),
    getElementById: () => null,
    querySelectorAll: () => [],
    addEventListener: () => {},
};
globalThis.window = { addEventListener: () => {}, confirm: () => true };

// ─── Import after shim is in place ──────────────────────────────────────────
const { Validator, ValidationRules, sanitizeHTML, debounce } = await import(
    '../../frontend/js/validation.js'
);

// ═════════════════════════════════════════════════════════════════════════════
// ValidationRules structure tests
// ═════════════════════════════════════════════════════════════════════════════

describe('ValidationRules', () => {
    it('has rules for all expected fields', () => {
        const expectedFields = [
            'base-model', 'output-name', 'learning-rate', 'epochs',
            'batch-size', 'grad-accum', 'max-length', 'max-prompt-length',
            'beta', 'lora-r', 'lora-alpha', 'lora-dropout', 'test-size'
        ];
        for (const field of expectedFields) {
            assert.ok(ValidationRules[field], `Missing rule for '${field}'`);
        }
    });

    it('marks required fields correctly', () => {
        assert.equal(ValidationRules['base-model'].required, true);
        assert.equal(ValidationRules['output-name'].required, true);
        assert.equal(ValidationRules['learning-rate'].required, true);
        // LoRA fields are optional
        assert.equal(ValidationRules['lora-r'].required, undefined);
    });

    it('has valid min/max ranges', () => {
        for (const [field, rules] of Object.entries(ValidationRules)) {
            if (rules.min !== undefined && rules.max !== undefined) {
                assert.ok(rules.min < rules.max,
                    `${field}: min (${rules.min}) should be less than max (${rules.max})`);
            }
        }
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Validator.validateField tests
// ═════════════════════════════════════════════════════════════════════════════

describe('Validator.validateField', () => {

    describe('required fields', () => {
        it('returns error for empty required field', () => {
            const errors = Validator.validateField('base-model', '', { required: true });
            assert.ok(errors.length > 0);
            assert.match(errors[0], /required/i);
        });

        it('returns error for null required field', () => {
            const errors = Validator.validateField('base-model', null, { required: true });
            assert.ok(errors.length > 0);
        });

        it('returns error for whitespace-only required field', () => {
            const errors = Validator.validateField('test', '   ', { required: true });
            assert.ok(errors.length > 0);
        });

        it('passes for non-empty required field', () => {
            const errors = Validator.validateField('base-model', 'meta-llama/Llama-3', { required: true });
            assert.equal(errors.length, 0);
        });

        it('passes for empty optional field', () => {
            const errors = Validator.validateField('lora-r', '', { type: 'number', min: 8, max: 512 });
            assert.equal(errors.length, 0);
        });
    });

    describe('number validation', () => {
        it('rejects non-numeric value', () => {
            const errors = Validator.validateField('lr', 'abc', { type: 'number' });
            assert.ok(errors.length > 0);
            assert.match(errors[0], /valid number/i);
        });

        it('rejects value below minimum', () => {
            const errors = Validator.validateField('lr', '0.0000001', { type: 'number', min: 0.000001, max: 0.1 });
            assert.ok(errors.length > 0);
            assert.match(errors[0], /at least/i);
        });

        it('rejects value above maximum', () => {
            const errors = Validator.validateField('lr', '0.5', { type: 'number', min: 0.000001, max: 0.1 });
            assert.ok(errors.length > 0);
            assert.match(errors[0], /at most/i);
        });

        it('accepts value within range', () => {
            const errors = Validator.validateField('lr', '0.00005', { type: 'number', min: 0.000001, max: 0.1 });
            assert.equal(errors.length, 0);
        });

        it('accepts exact minimum', () => {
            const errors = Validator.validateField('epochs', '1', { type: 'number', min: 1, max: 100 });
            assert.equal(errors.length, 0);
        });

        it('accepts exact maximum', () => {
            const errors = Validator.validateField('epochs', '100', { type: 'number', min: 1, max: 100 });
            assert.equal(errors.length, 0);
        });
    });

    describe('string validation', () => {
        it('rejects string below minimum length', () => {
            const errors = Validator.validateField('name', 'ab', { minLength: 3 });
            assert.ok(errors.length > 0);
            assert.match(errors[0], /at least 3 characters/i);
        });

        it('rejects string above maximum length', () => {
            const errors = Validator.validateField('name', 'a'.repeat(101), { maxLength: 100 });
            assert.ok(errors.length > 0);
            assert.match(errors[0], /at most 100 characters/i);
        });

        it('rejects string not matching pattern', () => {
            const errors = Validator.validateField('name', 'invalid name!@#', {
                pattern: /^[a-zA-Z0-9\-_]+$/,
                message: 'Invalid characters'
            });
            assert.ok(errors.length > 0);
        });

        it('accepts valid pattern match', () => {
            const errors = Validator.validateField('name', 'my-model_v1', {
                pattern: /^[a-zA-Z0-9\-_]+$/,
                message: 'Invalid characters'
            });
            assert.equal(errors.length, 0);
        });
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Validator.validateForm tests
// ═════════════════════════════════════════════════════════════════════════════

describe('Validator.validateForm', () => {
    it('returns empty object for valid form data', () => {
        const formData = {
            'base-model': 'meta-llama/Llama-3-8B',
            'output-name': 'my-model',
            'learning-rate': '0.00005',
            'epochs': '3',
            'batch-size': '2',
            'grad-accum': '16',
            'max-length': '2048',
            'max-prompt-length': '1024',
            'beta': '0.1',
        };
        const errors = Validator.validateForm(formData);
        assert.equal(Object.keys(errors).length, 0);
    });

    it('returns errors for multiple invalid fields', () => {
        const formData = {
            'base-model': '',
            'output-name': 'ab',  // too short
            'learning-rate': '999',  // too high
        };
        const errors = Validator.validateForm(formData);
        assert.ok('base-model' in errors);
        assert.ok('output-name' in errors);
        assert.ok('learning-rate' in errors);
    });

    it('skips fields without defined rules', () => {
        const formData = {
            'unknown-field': 'anything',
            'base-model': 'valid/model',
        };
        const errors = Validator.validateForm(formData);
        assert.ok(!('unknown-field' in errors));
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Validator.validateDatasetConfig tests
// ═════════════════════════════════════════════════════════════════════════════

describe('Validator.validateDatasetConfig', () => {
    it('validates huggingface source requires repo_id', () => {
        const config = {
            source: { source_type: 'huggingface', repo_id: '' },
            column_mapping: { prompt: 'prompt', chosen: 'chosen', rejected: 'rejected' },
        };
        const errors = Validator.validateDatasetConfig(config);
        assert.ok(errors.some(e => /repository/i.test(e)));
    });

    it('validates local_file source requires file_path', () => {
        const config = {
            source: { source_type: 'local_file', file_path: '' },
            column_mapping: { prompt: 'prompt', chosen: 'chosen', rejected: 'rejected' },
        };
        const errors = Validator.validateDatasetConfig(config);
        assert.ok(errors.some(e => /file path/i.test(e)));
    });

    it('validates upload source requires dataset_id', () => {
        const config = {
            source: { source_type: 'upload' },
            column_mapping: { prompt: 'prompt', chosen: 'chosen', rejected: 'rejected' },
        };
        const errors = Validator.validateDatasetConfig(config);
        assert.ok(errors.some(e => /upload/i.test(e)));
    });

    it('requires prompt and chosen column mapping', () => {
        const config = {
            source: { source_type: 'huggingface', repo_id: 'test/dataset' },
            column_mapping: { some_col: 'other' },
        };
        const errors = Validator.validateDatasetConfig(config);
        assert.ok(errors.some(e => /prompt/i.test(e)));
        assert.ok(errors.some(e => /chosen/i.test(e)));
    });

    it('requires rejected column in ORPO mode', () => {
        const config = {
            source: { source_type: 'huggingface', repo_id: 'test/dataset' },
            column_mapping: { prompt_col: 'prompt', chosen_col: 'chosen' },
        };
        const errors = Validator.validateDatasetConfig(config, 'orpo');
        assert.ok(errors.some(e => /rejected/i.test(e)));
    });

    it('does NOT require rejected column in SFT mode', () => {
        const config = {
            source: { source_type: 'huggingface', repo_id: 'test/dataset' },
            column_mapping: { prompt_col: 'prompt', chosen_col: 'chosen' },
        };
        const errors = Validator.validateDatasetConfig(config, 'sft');
        assert.ok(!errors.some(e => /rejected/i.test(e)));
    });

    it('passes for valid ORPO config', () => {
        const config = {
            source: { source_type: 'huggingface', repo_id: 'test/dataset' },
            column_mapping: { p: 'prompt', c: 'chosen', r: 'rejected' },
        };
        const errors = Validator.validateDatasetConfig(config, 'orpo');
        assert.equal(errors.length, 0);
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Validator.estimateVRAM tests
// ═════════════════════════════════════════════════════════════════════════════

describe('Validator.estimateVRAM', () => {
    it('estimates VRAM for 7B model', () => {
        const result = Validator.estimateVRAM({
            base_model: 'meta-llama/Llama-3-7b',
            use_4bit: false,
            batch_size: 1,
            gradient_accumulation_steps: 16,
            max_length: 2048,
        });
        assert.ok(parseFloat(result.base) > 0);
        assert.ok(parseFloat(result.total) > parseFloat(result.base));
    });

    it('reduces VRAM estimate with 4-bit quantization', () => {
        const config = {
            base_model: 'meta-llama/Llama-3-7b',
            batch_size: 1,
            gradient_accumulation_steps: 16,
            max_length: 2048,
        };
        const full = Validator.estimateVRAM({ ...config, use_4bit: false });
        const quantized = Validator.estimateVRAM({ ...config, use_4bit: true });
        assert.ok(parseFloat(quantized.base) < parseFloat(full.base));
    });

    it('uses fallback size for unknown model', () => {
        const result = Validator.estimateVRAM({
            base_model: 'unknown/model-xyz',
            use_4bit: false,
            batch_size: 1,
            gradient_accumulation_steps: 1,
            max_length: 2048,
        });
        // Default fallback is 10 GB
        assert.equal(parseFloat(result.base), 10.0);
    });

    it('larger batch increases training overhead', () => {
        const config = {
            base_model: 'meta-llama/Llama-3-7b',
            use_4bit: true,
            gradient_accumulation_steps: 1,
            max_length: 2048,
        };
        const small = Validator.estimateVRAM({ ...config, batch_size: 1 });
        const large = Validator.estimateVRAM({ ...config, batch_size: 8 });
        assert.ok(parseFloat(large.training) > parseFloat(small.training));
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// sanitizeHTML tests
// ═════════════════════════════════════════════════════════════════════════════

describe('sanitizeHTML', () => {
    it('escapes HTML tags', () => {
        const result = sanitizeHTML('<script>alert("xss")</script>');
        assert.ok(!result.includes('<script>'));
        assert.ok(result.includes('&lt;script&gt;'));
    });

    it('escapes ampersands', () => {
        const result = sanitizeHTML('foo & bar');
        assert.ok(result.includes('&amp;'));
    });

    it('escapes quotes', () => {
        const result = sanitizeHTML('he said "hello"');
        assert.ok(result.includes('&quot;'));
    });

    it('preserves normal text', () => {
        const result = sanitizeHTML('Hello World 123');
        assert.equal(result, 'Hello World 123');
    });

    it('handles empty string', () => {
        assert.equal(sanitizeHTML(''), '');
    });

    it('handles nested injection attempts', () => {
        const result = sanitizeHTML('<img src=x onerror="alert(1)">');
        assert.ok(!result.includes('<img'));
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// debounce tests
// ═════════════════════════════════════════════════════════════════════════════

describe('debounce', () => {
    it('delays execution', async () => {
        let called = false;
        const fn = debounce(() => { called = true; }, 50);
        fn();
        assert.equal(called, false);
        await new Promise(r => setTimeout(r, 100));
        assert.equal(called, true);
    });

    it('only executes once for rapid calls', async () => {
        let count = 0;
        const fn = debounce(() => { count++; }, 50);
        fn(); fn(); fn(); fn(); fn();
        await new Promise(r => setTimeout(r, 100));
        assert.equal(count, 1);
    });

    it('can be cancelled', async () => {
        let called = false;
        const fn = debounce(() => { called = true; }, 50);
        fn();
        fn.cancel();
        await new Promise(r => setTimeout(r, 100));
        assert.equal(called, false);
    });
});
