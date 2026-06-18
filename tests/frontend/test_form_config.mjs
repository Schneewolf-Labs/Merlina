// Unit tests for form_config.js — the centralized TrainingConfig builder.
//
// These tests guard the contract that:
//   1. Every field on the backend Pydantic TrainingConfig is present in
//      the dict produced by buildTrainingConfig().
//   2. The "save preset" path strips secrets while the "/train" path
//      keeps them.
//   3. The dataset block carries every knob (max_samples, deduplicate,
//      etc.) instead of being a thin subset.
//   4. The share_config flag round-trips.
//
// Run with: node --test tests/frontend/test_form_config.mjs

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..', '..');

// ─── DOM shim ───────────────────────────────────────────────────────────────
//
// We back the entire shim with a single `state` object so each test can
// declare exactly which inputs/checkboxes/cards the form should expose.
// `setState({...})` builds enough of the DOM to satisfy buildTrainingConfig.

let state = {};

function makeElement({ id, value = '', checked = false, type = 'text', tag = 'INPUT' }) {
    return {
        id,
        type,
        tagName: tag,
        value,
        checked,
        dataset: {},
        querySelector() { return null; },
        querySelectorAll() { return []; },
        addEventListener() {},
        dispatchEvent() {},
    };
}

function makeCard(card) {
    // card: { sourceType, repoId, split, configName, filePath, fileFormat, datasetId, mapping }
    const select = (sel) => {
        switch (sel) {
            case '.ds-source-type':
                return { value: card.sourceType ?? 'huggingface' };
            case '.ds-repo':
                return { value: card.repoId ?? '' };
            case '.ds-split':
                return { value: card.split ?? 'train' };
            case '.ds-config':
                return { value: card.configName ?? '' };
            case '.ds-local-path':
                return { value: card.filePath ?? '' };
            case '.ds-local-format':
                return { value: card.fileFormat ?? '' };
            case '.ds-map-prompt':
                return { value: card.mapping?.prompt ?? '' };
            case '.ds-map-chosen':
                return { value: card.mapping?.chosen ?? '' };
            case '.ds-map-rejected':
                return { value: card.mapping?.rejected ?? '' };
            case '.ds-map-system':
                return { value: card.mapping?.system ?? '' };
            case '.ds-map-reasoning':
                return { value: card.mapping?.reasoning ?? '' };
            default:
                return null;
        }
    };
    return {
        dataset: { uploadId: card.datasetId ?? '' },
        querySelector: select,
    };
}

function setState(s) {
    state = {
        inputs: s.inputs || {},
        cards: s.cards || [],
        radios: s.radios || {},
    };
}

globalThis.document = {
    getElementById(id) {
        if (!(id in state.inputs)) return null;
        const v = state.inputs[id];
        if (typeof v === 'boolean') {
            return makeElement({ id, checked: v, type: 'checkbox' });
        }
        return makeElement({ id, value: String(v) });
    },
    querySelectorAll(sel) {
        if (sel === '#datasets-list .dataset-card') {
            return state.cards.map(makeCard);
        }
        return [];
    },
    querySelector(sel) {
        const m = sel.match(/^input\[name="([^"]+)":?checked\]?$/);
        if (m) {
            const name = m[1];
            const val = state.radios[name];
            return val !== undefined ? { value: val } : null;
        }
        // Fall back: handle the actual selector used by buildDatasetConfig
        if (sel === 'input[name="system-prompt-mode"]:checked') {
            const val = state.radios['system-prompt-mode'];
            return val !== undefined ? { value: val } : null;
        }
        return null;
    },
};
globalThis.window = {};

// ─── Import the module under test ───────────────────────────────────────────

const { buildTrainingConfig, buildDatasetConfig, SECRET_FIELDS } =
    await import('../../frontend/js/form_config.js');

// ─── Helpers ────────────────────────────────────────────────────────────────

/**
 * Build a state object with values for *every* form field touched by
 * buildTrainingConfig, so we get a complete config back.
 */
function fullFormState({ extras = {}, cards, radios } = {}) {
    return {
        inputs: {
            'base-model': 'meta-llama/Meta-Llama-3-8B',
            'output-name': 'my-cool-model',
            'model-type': 'auto',
            'training-mode': 'orpo',
            'use-lora': true,
            'lora-r': '64',
            'lora-alpha': '32',
            'lora-dropout': '0.05',
            'target-modules': 'q_proj, v_proj, o_proj',
            'target-modules-manual': '',
            'modules-to-save': '',
            'lora-task-type': 'CAUSAL_LM',
            'use-4bit': true,
            'max-length': '2048',
            'max-prompt-length': '1024',
            'epochs': '2',
            'batch-size': '1',
            'grad-accum': '16',
            'learning-rate': '0.000005',
            'warmup-ratio': '0.05',
            'beta': '0.1',
            'gamma': '0.5',
            'label-smoothing': '0.0',
            'seed': '42',
            'max-grad-norm': '0.3',
            'weight-decay': '0.01',
            'lr-scheduler-type': 'cosine',
            'logging-steps': '1',
            'eval-steps': '0.2',
            'shuffle-dataset': true,
            'gradient-checkpointing': false,
            'optimizer-type': 'paged_adamw_8bit',
            'adam-beta1': '0.9',
            'adam-beta2': '0.999',
            'adam-epsilon': '1e-8',
            'adafactor-relative-step': false,
            'adafactor-scale-parameter': false,
            'adafactor-warmup-init': false,
            'adafactor-decay-rate': '-0.8',
            'adafactor-beta1': '',
            'adafactor-clip-threshold': '1.0',
            'attn-implementation': 'auto',
            'use-liger': false,
            'torch-compile': false,
            'neftune-alpha': '',
            'eval-on-start': false,
            'multi-gpu-strategy': 'auto',
            'use-wandb': true,
            'push-hub': false,
            'merge-lora-before-upload': true,
            'hf-hub-private': true,
            'share-config': true,
            'wandb-project': '',
            'wandb-run-name': '',
            'wandb-tags': '',
            'wandb-notes': '',
            'hf-token': '',
            'hf-token-preload': '',
            'wandb-key': '',
            'dataset-format-type': 'tokenizer',
            'enable-thinking': true,
            'custom-prompt-template': '',
            'custom-chosen-template': '',
            'custom-rejected-template': '',
            'test-size': '0.01',
            'convert-messages-checkbox': true,
            'deduplicate': false,
            'dedupe-strategy': 'prompt_chosen',
            'max-samples': '',
            'system-prompt-override': '',
            ...extras,
        },
        cards: cards ?? [{
            sourceType: 'huggingface',
            repoId: 'schneewolflabs/Athanorlite-DPO',
            split: 'train',
        }],
        radios: radios ?? { 'system-prompt-mode': 'fill_empty' },
    };
}

// ─── Parity test: every TrainingConfig field is covered ─────────────────────

describe('buildTrainingConfig — coverage of TrainingConfig fields', () => {
    /**
     * Pulls the TrainingConfig field names from merlina.py without
     * importing the module (which requires Python+heavy ML deps). We grep
     * for `field_name: TypeAnnotation = Field(...)` declarations within
     * the `class TrainingConfig(BaseModel):` block.
     */
    function extractTrainingConfigFields() {
        const py = readFileSync(join(REPO_ROOT, 'merlina.py'), 'utf8');
        const start = py.indexOf('class TrainingConfig(BaseModel):');
        assert.ok(start !== -1, 'TrainingConfig class not found in merlina.py');
        // The class body ends at the next "class JobResponse" line.
        const end = py.indexOf('\nclass JobResponse(BaseModel):', start);
        assert.ok(end !== -1, 'End of TrainingConfig class not found');
        const body = py.slice(start, end);

        // Match top-level field declarations: `    name: type = Field(`.
        // Skip @model_validator decorators and methods (they start with def).
        const fields = new Set();
        for (const line of body.split('\n')) {
            const m = line.match(/^    ([a-z_][a-z0-9_]*):\s/i);
            if (!m) continue;
            const name = m[1];
            // Skip the placeholder `_metadata` style names; TrainingConfig
            // has none today, but be defensive.
            if (name.startsWith('_')) continue;
            fields.add(name);
        }
        return fields;
    }

    it('covers every TrainingConfig field (frontend ↔ backend parity)', () => {
        const backendFields = extractTrainingConfigFields();
        setState(fullFormState());
        const config = buildTrainingConfig({ includeSecrets: true });

        const frontendFields = new Set(Object.keys(config));

        // Every backend field must be present in the frontend payload OR
        // be a documented opt-in (gpu_ids — only attached when a
        // GPUManager is supplied).
        const optional = new Set(['gpu_ids']);
        const missing = [];
        for (const f of backendFields) {
            if (!frontendFields.has(f) && !optional.has(f)) {
                missing.push(f);
            }
        }
        assert.deepEqual(missing, [], (
            `buildTrainingConfig is missing TrainingConfig fields: ${missing.join(', ')}. ` +
            `Add them to form_config.js — saved presets will lose data otherwise.`
        ));
    });

    it('does not invent fields the backend does not know', () => {
        const backendFields = extractTrainingConfigFields();
        setState(fullFormState());
        const config = buildTrainingConfig({ includeSecrets: true });

        // Top-level keys we know are not on TrainingConfig.
        // (gpu_ids is opt-in, hf_token/wandb_key are real fields.)
        const ignore = new Set();
        const unknown = [];
        for (const f of Object.keys(config)) {
            if (ignore.has(f)) continue;
            if (!backendFields.has(f)) unknown.push(f);
        }
        assert.deepEqual(unknown, [], (
            `buildTrainingConfig emits keys not on TrainingConfig: ${unknown.join(', ')}. ` +
            `Pydantic v2 silently drops unknown fields — these values would be lost.`
        ));
    });
});

// ─── Secret stripping ───────────────────────────────────────────────────────

describe('buildTrainingConfig — secret handling', () => {
    it('returns null secrets when includeSecrets=false (saved presets)', () => {
        setState(fullFormState({
            extras: {
                'hf-token': 'hf_xxx_secret',
                'wandb-key': 'wb_yyy_secret',
            },
        }));
        const config = buildTrainingConfig({ includeSecrets: false });
        // Secrets are present-as-null so the dict shape matches
        // TrainingConfig deterministically; ConfigManager.strip_secrets
        // drops them entirely on persistence.
        for (const f of SECRET_FIELDS) {
            assert.equal(config[f], null,
                `${f} must be null in saved preset (form value should not leak)`);
        }
    });

    it('includes secret values when includeSecrets=true (the /train path)', () => {
        setState(fullFormState({
            extras: {
                'hf-token': 'hf_xxx_secret',
                'wandb-key': 'wb_yyy_secret',
            },
        }));
        const config = buildTrainingConfig({ includeSecrets: true });
        assert.equal(config.hf_token, 'hf_xxx_secret');
        assert.equal(config.wandb_key, 'wb_yyy_secret');
    });

    it('falls back to hf-token-preload when hf-token is empty', () => {
        setState(fullFormState({
            extras: {
                'hf-token': '',
                'hf-token-preload': 'preload_token',
            },
        }));
        const config = buildTrainingConfig({ includeSecrets: true });
        assert.equal(config.hf_token, 'preload_token');
    });

    it('emits null when both token inputs are empty', () => {
        setState(fullFormState({ extras: { 'hf-token': '', 'hf-token-preload': '' } }));
        const config = buildTrainingConfig({ includeSecrets: true });
        assert.equal(config.hf_token, null);
        assert.equal(config.wandb_key, null);
    });

    it('SECRET_FIELDS is frozen and contains the canonical secrets', () => {
        assert.deepEqual([...SECRET_FIELDS], ['hf_token', 'wandb_key']);
        // Frozen arrays in JS are still arrays — pushing to them throws.
        assert.throws(() => SECRET_FIELDS.push('extra'));
    });
});

// ─── Dataset block coverage ─────────────────────────────────────────────────

describe('buildDatasetConfig — full dataset coverage', () => {
    it('serializes every dataset knob the backend understands', () => {
        setState(fullFormState({
            extras: {
                'max-samples': '100',
                'deduplicate': true,
                'dedupe-strategy': 'exact',
                'system-prompt-override': 'You are helpful.',
                'convert-messages-checkbox': false,
            },
            radios: { 'system-prompt-mode': 'replace_all' },
        }));
        const ds = buildDatasetConfig();

        assert.equal(ds.max_samples, 100);
        assert.equal(ds.deduplicate, true);
        assert.equal(ds.dedupe_strategy, 'exact');
        assert.equal(ds.convert_messages_format, false);
        assert.equal(ds.system_prompt, 'You are helpful.');
        assert.equal(ds.system_prompt_mode, 'replace_all');
        assert.equal(ds.training_mode, 'orpo');
        assert.equal(ds.test_size, 0.01);
        assert.equal(ds.model_name, 'meta-llama/Meta-Llama-3-8B');
        assert.deepEqual(ds.source, {
            source_type: 'huggingface',
            repo_id: 'schneewolflabs/Athanorlite-DPO',
            split: 'train',
        });
    });

    it('omits max_samples when blank', () => {
        setState(fullFormState());
        const ds = buildDatasetConfig();
        assert.equal('max_samples' in ds, false);
    });

    it('captures additional_sources from extra cards', () => {
        setState(fullFormState({
            cards: [
                { sourceType: 'huggingface', repoId: 'a/one', split: 'train' },
                { sourceType: 'huggingface', repoId: 'b/two', split: 'validation' },
                { sourceType: 'local_file', filePath: '/data/extra.json', fileFormat: 'json' },
            ],
        }));
        const ds = buildDatasetConfig();
        assert.equal(ds.source.repo_id, 'a/one');
        assert.equal(ds.additional_sources.length, 2);
        assert.equal(ds.additional_sources[0].repo_id, 'b/two');
        assert.equal(ds.additional_sources[1].file_path, '/data/extra.json');
    });

    it('includes config_name (HF subset) when set, omits it when blank', () => {
        // set on the primary source
        setState(fullFormState({
            cards: [{ sourceType: 'huggingface', repoId: 'org/ds', split: 'train',
                      configName: 'high_quality' }],
        }));
        let ds = buildDatasetConfig();
        assert.equal(ds.source.config_name, 'high_quality');

        // blank -> key omitted entirely (default config)
        setState(fullFormState({
            cards: [{ sourceType: 'huggingface', repoId: 'org/ds', split: 'train' }],
        }));
        ds = buildDatasetConfig();
        assert.equal('config_name' in ds.source, false);
    });

    it('attaches per-card column_mapping when provided', () => {
        setState(fullFormState({
            cards: [{
                sourceType: 'huggingface',
                repoId: 'a/one',
                split: 'train',
                mapping: { prompt: 'instruction', chosen: 'response' },
            }],
        }));
        const ds = buildDatasetConfig();
        assert.deepEqual(ds.source.column_mapping, {
            instruction: 'prompt',
            response: 'chosen',
        });
    });

    it('attaches qwen3 enable_thinking when format_type=qwen3', () => {
        setState(fullFormState({
            extras: {
                'dataset-format-type': 'qwen3',
                'enable-thinking': false,
            },
        }));
        const ds = buildDatasetConfig();
        assert.equal(ds.format.format_type, 'qwen3');
        assert.equal(ds.format.enable_thinking, false);
    });

    it('attaches custom_templates when format_type=custom', () => {
        setState(fullFormState({
            extras: {
                'dataset-format-type': 'custom',
                'custom-prompt-template': 'P:{prompt}',
                'custom-chosen-template': 'C:{chosen}',
                'custom-rejected-template': 'R:{rejected}',
            },
        }));
        const ds = buildDatasetConfig();
        assert.deepEqual(ds.format.custom_templates, {
            prompt_template: 'P:{prompt}',
            chosen_template: 'C:{chosen}',
            rejected_template: 'R:{rejected}',
        });
    });
});

// ─── share_config flag ──────────────────────────────────────────────────────

describe('buildTrainingConfig — share_config toggle', () => {
    it('defaults to true when checkbox is missing', () => {
        const s = fullFormState();
        delete s.inputs['share-config'];
        setState(s);
        const config = buildTrainingConfig();
        assert.equal(config.share_config, true);
    });

    it('respects unchecked state', () => {
        setState(fullFormState({ extras: { 'share-config': false } }));
        const config = buildTrainingConfig();
        assert.equal(config.share_config, false);
    });

    it('respects checked state', () => {
        setState(fullFormState({ extras: { 'share-config': true } }));
        const config = buildTrainingConfig();
        assert.equal(config.share_config, true);
    });
});

// ─── share_config_image flag (QR / PNG-metadata artifact) ───────────────────

describe('buildTrainingConfig — share_config_image toggle', () => {
    it('defaults to false when checkbox is missing', () => {
        const s = fullFormState();
        delete s.inputs['share-config-image'];
        setState(s);
        const config = buildTrainingConfig();
        assert.equal(config.share_config_image, false);
    });

    it('respects checked state', () => {
        setState(fullFormState({ extras: { 'share-config-image': true } }));
        const config = buildTrainingConfig();
        assert.equal(config.share_config_image, true);
    });

    it('is independent of share_config', () => {
        setState(fullFormState({ extras: { 'share-config': false, 'share-config-image': true } }));
        const config = buildTrainingConfig();
        assert.equal(config.share_config, false);
        assert.equal(config.share_config_image, true);
    });
});

// ─── target_modules normalization ───────────────────────────────────────────

describe('buildTrainingConfig — target_modules', () => {
    it('parses comma-separated target_modules into an array', () => {
        setState(fullFormState({
            extras: { 'target-modules': 'q_proj, v_proj, o_proj' },
        }));
        const config = buildTrainingConfig();
        assert.deepEqual(config.target_modules, ['q_proj', 'v_proj', 'o_proj']);
    });

    it('falls back to target-modules-manual when target-modules is empty', () => {
        setState(fullFormState({
            extras: {
                'target-modules': '',
                'target-modules-manual': 'gate_proj,up_proj',
            },
        }));
        const config = buildTrainingConfig();
        assert.deepEqual(config.target_modules, ['gate_proj', 'up_proj']);
    });

    it('returns an empty array when both are blank', () => {
        setState(fullFormState({
            extras: {
                'target-modules': '',
                'target-modules-manual': '',
            },
        }));
        const config = buildTrainingConfig();
        assert.deepEqual(config.target_modules, []);
    });

    it('always emits target_modules as an array (never a string)', () => {
        setState(fullFormState());
        const config = buildTrainingConfig();
        assert.ok(Array.isArray(config.target_modules),
            'target_modules must be an array — Pydantic will reject a string');
        assert.ok(Array.isArray(config.modules_to_save),
            'modules_to_save must be an array');
    });
});

// ─── GPU manager integration ────────────────────────────────────────────────

describe('buildTrainingConfig — GPU manager', () => {
    it('attaches gpu_ids when a GPU manager is supplied', () => {
        setState(fullFormState());
        const fakeMgr = { getSelectedGPUs: () => [0, 1] };
        const config = buildTrainingConfig({ gpuManager: fakeMgr });
        assert.deepEqual(config.gpu_ids, [0, 1]);
    });

    it('omits gpu_ids when no manager is supplied (preset save)', () => {
        setState(fullFormState());
        const config = buildTrainingConfig();
        assert.equal('gpu_ids' in config, false);
    });
});
