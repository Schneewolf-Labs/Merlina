// Unit tests for ui.js - UI components (MetricsDisplay, ProgressBar, FormUI, etc.)
// Run with: node --test tests/frontend/test_ui.mjs

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// ─── DOM shims ──────────────────────────────────────────────────────────────

function createElement(tag) {
    const el = {
        _tag: tag,
        _children: [],
        _classes: new Set(),
        _attrs: {},
        _style: {},
        _text: '',
        _html: '',
        _htmlExplicit: false,
        _disabled: false,
        _dataset: {},
        id: '',
        type: '',
        value: '',
        checked: false,
        selectedIndex: 0,
        options: [],
        selectedOptions: [],

        get textContent() { return this._text; },
        set textContent(val) { this._text = val; this._htmlExplicit = false; },
        get innerHTML() {
            // If innerHTML was explicitly set, return it. Otherwise encode textContent (for sanitizeHTML).
            if (this._htmlExplicit) return this._html;
            return this._text
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#039;');
        },
        set innerHTML(val) { this._html = val; this._htmlExplicit = true; },
        get disabled() { return this._disabled; },
        set disabled(val) { this._disabled = val; },
        get dataset() { return this._dataset; },

        get classList() {
            const self = this;
            return {
                add(cls) { self._classes.add(cls); },
                remove(cls) { self._classes.delete(cls); },
                contains(cls) { return self._classes.has(cls); },
                toggle(cls) {
                    if (self._classes.has(cls)) self._classes.delete(cls);
                    else self._classes.add(cls);
                }
            };
        },

        get className() { return Array.from(this._classes).join(' '); },
        set className(val) {
            this._classes.clear();
            val.split(' ').filter(Boolean).forEach(c => this._classes.add(c));
        },

        get style() { return this._style; },

        setAttribute(name, val) { this._attrs[name] = val; },
        getAttribute(name) { return this._attrs[name] ?? null; },
        removeAttribute(name) { delete this._attrs[name]; },

        appendChild(child) { this._children.push(child); return child; },
        remove() {},
        querySelector(sel) { return null; },
        querySelectorAll(sel) { return []; },

        get parentNode() {
            return {
                insertBefore: () => {},
                querySelector: () => null,
            };
        },

        addEventListener() {},
        removeEventListener() {},
        dispatchEvent() {},
    };
    return el;
}

const domElements = {};

globalThis.document = {
    createElement,
    getElementById(id) { return domElements[id] || null; },
    querySelectorAll() { return []; },
    addEventListener() {},
    body: { appendChild() {}, style: {} },
    head: { appendChild() {} },
    documentElement: { setAttribute() {}, getAttribute() { return 'light'; } },
};
globalThis.window = {
    location: { protocol: 'https:', host: 'localhost:8000' },
    addEventListener() {},
    removeEventListener() {},
    confirm: () => true,
    getComputedStyle: () => ({ display: 'none' }),
    matchMedia: () => ({ matches: false, addEventListener() {} }),
};
globalThis.console = { ...console };
globalThis.WebSocket = class WebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;
};
globalThis.AbortController = AbortController;
globalThis.FormData = class FormData { append() {} };
globalThis.XMLHttpRequest = class XMLHttpRequest {
    open() {} send() {} upload = { addEventListener() {} };
    addEventListener() {}
};
globalThis.localStorage = {
    _data: {},
    getItem(key) { return this._data[key] ?? null; },
    setItem(key, val) { this._data[key] = val; },
    removeItem(key) { delete this._data[key]; },
};

// ─── Import modules ──────────────────────────────────────────────────────────

const {
    Toast, LoadingManager, ConnectionStatus, Modal,
    ProgressBar, FormUI, JobCardRenderer, MetricsDisplay, GPUDisplay,
    LossChart, collectExtraMetricKeys, extraMetricSeries, formatMetricLabel
} = await import('../../frontend/js/ui.js');

// ═════════════════════════════════════════════════════════════════════════════
// MetricsDisplay tests
// ═════════════════════════════════════════════════════════════════════════════

describe('MetricsDisplay', () => {
    describe('formatValue', () => {
        it('formats numbers to 4 decimal places', () => {
            assert.equal(MetricsDisplay.formatValue(0.123456789, 'number'), '0.1235');
        });

        it('formats percentages', () => {
            assert.equal(MetricsDisplay.formatValue(0.756, 'percent'), '75.60%');
        });

        it('formats memory in GB', () => {
            assert.equal(MetricsDisplay.formatValue(12.345, 'memory'), '12.35 GB');
        });

        it('returns dash for null values', () => {
            assert.equal(MetricsDisplay.formatValue(null, 'number'), '-');
        });

        it('returns dash for undefined values', () => {
            assert.equal(MetricsDisplay.formatValue(undefined, 'number'), '-');
        });

        it('converts non-number to string for default type', () => {
            assert.equal(MetricsDisplay.formatValue('hello', 'other'), 'hello');
        });
    });

    describe('formatDuration', () => {
        it('formats seconds only', () => {
            assert.equal(MetricsDisplay.formatDuration(45), '45s');
        });

        it('formats minutes and seconds', () => {
            assert.equal(MetricsDisplay.formatDuration(125), '2m 5s');
        });

        it('formats hours, minutes, and seconds', () => {
            assert.equal(MetricsDisplay.formatDuration(3725), '1h 2m 5s');
        });

        it('returns dash for zero', () => {
            assert.equal(MetricsDisplay.formatDuration(0), '-');
        });

        it('returns dash for null', () => {
            assert.equal(MetricsDisplay.formatDuration(null), '-');
        });

        it('returns dash for undefined', () => {
            assert.equal(MetricsDisplay.formatDuration(undefined), '-');
        });

        it('handles exact hour boundary', () => {
            assert.equal(MetricsDisplay.formatDuration(3600), '1h 0m 0s');
        });

        it('handles exact minute boundary', () => {
            assert.equal(MetricsDisplay.formatDuration(60), '1m 0s');
        });
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// ConnectionStatus tests
// ═════════════════════════════════════════════════════════════════════════════

describe('ConnectionStatus', () => {
    it('initializes with connected status', () => {
        const cs = new ConnectionStatus();
        assert.equal(cs.status, 'connected');
    });

    it('returns correct status text for all states', () => {
        const cs = new ConnectionStatus();
        assert.equal(cs.getStatusText('connected'), 'Connected');
        assert.equal(cs.getStatusText('connecting'), 'Connecting...');
        assert.equal(cs.getStatusText('reconnecting'), 'Reconnecting...');
        assert.equal(cs.getStatusText('disconnected'), 'Disconnected');
        assert.equal(cs.getStatusText('error'), 'Connection Error');
        assert.equal(cs.getStatusText('anything_else'), 'Unknown');
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// LoadingManager tests
// ═════════════════════════════════════════════════════════════════════════════

describe('LoadingManager', () => {
    it('show sets disabled and loading class', () => {
        const el = createElement('button');
        el._text = 'Submit';
        LoadingManager.show(el, 'Loading...');
        assert.equal(el.disabled, true);
        assert.ok(el._classes.has('loading'));
        assert.equal(el._attrs['aria-busy'], 'true');
    });

    it('show stores original text', () => {
        const el = createElement('button');
        el._text = 'Submit';
        LoadingManager.show(el, 'Loading...');
        assert.equal(el.dataset.originalText, 'Submit');
    });

    it('hide restores element state', () => {
        const el = createElement('button');
        el._text = 'Submit';
        el._html = 'Submit';
        LoadingManager.show(el, 'Loading...');
        LoadingManager.hide(el);
        assert.equal(el.disabled, false);
        assert.ok(!el._classes.has('loading'));
        assert.equal(el._attrs['aria-busy'], 'false');
    });

    it('handles null element gracefully', () => {
        // Should not throw
        LoadingManager.show(null);
        LoadingManager.hide(null);
    });

    it('createSpinner returns element with correct class', () => {
        const spinner = LoadingManager.createSpinner('small');
        assert.ok(spinner._classes.has('loading-spinner'));
        assert.ok(spinner._classes.has('loading-spinner-small'));
        assert.equal(spinner._attrs['role'], 'status');
    });

    it('createSpinner defaults to small size', () => {
        const spinner = LoadingManager.createSpinner();
        assert.ok(spinner._classes.has('loading-spinner-small'));
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// FormUI tests
// ═════════════════════════════════════════════════════════════════════════════

describe('FormUI', () => {
    describe('setEnabled', () => {
        it('handles null form gracefully', () => {
            // Should not throw
            FormUI.setEnabled(null, true);
        });
    });

    describe('getData', () => {
        it('returns empty object for null form', () => {
            const result = FormUI.getData(null);
            assert.deepEqual(result, {});
        });
    });

    describe('clear', () => {
        it('handles null form gracefully', () => {
            // Should not throw
            FormUI.clear(null);
        });
    });

    describe('setData', () => {
        it('handles null form gracefully', () => {
            // Should not throw
            FormUI.setData(null, {});
        });

        it('handles null data gracefully', () => {
            const form = createElement('form');
            // Should not throw
            FormUI.setData(form, null);
        });
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// GPUDisplay tests
// ═════════════════════════════════════════════════════════════════════════════

describe('GPUDisplay', () => {
    describe('renderList', () => {
        it('handles null container gracefully', () => {
            // Should not throw
            GPUDisplay.renderList([], null);
        });

        it('shows empty state for no GPUs', () => {
            const container = createElement('div');
            GPUDisplay.renderList([], container);
            assert.ok(container._html.includes('No GPUs detected'));
        });

        it('shows empty state for null GPUs', () => {
            const container = createElement('div');
            GPUDisplay.renderList(null, container);
            assert.ok(container._html.includes('No GPUs detected'));
        });

        it('renders GPU cards for valid data', () => {
            const container = createElement('div');
            const gpus = [{
                index: 0,
                name: 'RTX 4090',
                compute_capability: '8.9',
                used_memory_mb: 1000,
                total_memory_mb: 24000,
                memory_utilization_percent: 4,
                gpu_utilization_percent: null,
                temperature_c: null,
                power_usage_w: null,
            }];
            GPUDisplay.renderList(gpus, container);
            assert.ok(container._html.includes('RTX 4090'));
            assert.ok(container._html.includes('GPU 0'));
        });

        it('renders optional stats when present', () => {
            const container = createElement('div');
            const gpus = [{
                index: 0,
                name: 'RTX 4090',
                compute_capability: '8.9',
                used_memory_mb: 1000,
                total_memory_mb: 24000,
                memory_utilization_percent: 4,
                gpu_utilization_percent: 50,
                temperature_c: 65,
                power_usage_w: 200,
            }];
            GPUDisplay.renderList(gpus, container);
            assert.ok(container._html.includes('50%'));
            assert.ok(container._html.includes('65°C'));
            assert.ok(container._html.includes('200W'));
        });
    });

    describe('populateSelect', () => {
        it('handles null select gracefully', () => {
            // Should not throw
            GPUDisplay.populateSelect([], null);
        });

        it('shows no GPUs message when empty', () => {
            const select = createElement('select');
            GPUDisplay.populateSelect([], select);
            assert.ok(select._html.includes('No GPUs available'));
            assert.equal(select.disabled, true);
        });

        it('shows auto option and GPU options', () => {
            const select = createElement('select');
            GPUDisplay.populateSelect([
                { index: 0, name: 'RTX 4090', free_memory_mb: 20000 }
            ], select);
            assert.ok(select._html.includes('Auto'));
            assert.equal(select.disabled, false);
            assert.equal(select._children.length, 1); // One GPU option appended
        });
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Modal tests
// ═════════════════════════════════════════════════════════════════════════════

describe('Modal', () => {
    it('tracks instances in static set', () => {
        const initialCount = Modal.instances.size;
        // Create a mock modal element
        domElements['test-modal-1'] = createElement('div');
        const modal = new Modal('test-modal-1');
        assert.equal(Modal.instances.size, initialCount + 1);
        assert.ok(Modal.instances.has(modal));

        // Cleanup
        modal.destroy();
        assert.ok(!Modal.instances.has(modal));
        delete domElements['test-modal-1'];
    });

    it('show sets display to flex', () => {
        const el = createElement('div');
        domElements['test-modal-2'] = el;
        const modal = new Modal('test-modal-2');
        modal.show();
        assert.equal(el._style.display, 'flex');
        modal.destroy();
        delete domElements['test-modal-2'];
    });

    it('hide sets display to none', () => {
        const el = createElement('div');
        domElements['test-modal-3'] = el;
        const modal = new Modal('test-modal-3');
        modal.show();
        modal.hide();
        assert.equal(el._style.display, 'none');
        modal.destroy();
        delete domElements['test-modal-3'];
    });

    it('handles missing modal element gracefully', () => {
        const modal = new Modal('nonexistent-modal');
        // Should not throw
        modal.show();
        modal.hide();
        modal.destroy();
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Extra (per-component) trainer metrics on the loss chart
// ═════════════════════════════════════════════════════════════════════════════

describe('extra trainer metrics', () => {
    const rows = [
        { 'train/nll_loss': 1.2, 'train/or_loss': 0.3, 'train/reward_margin': 0.01 },
        {},
        { 'train/reward_margin': 0.05, 'eval/reward_accuracy': 0.75, 'train/epoch': 0.5 },
        null,
    ];

    it('collects metric names, train before eval, alphabetical', () => {
        assert.deepEqual(collectExtraMetricKeys(rows), [
            'train/epoch', 'train/nll_loss', 'train/or_loss', 'train/reward_margin',
            'eval/reward_accuracy',
        ]);
    });

    it('returns no keys for SFT-style rows with no extras', () => {
        assert.deepEqual(collectExtraMetricKeys([{}, {}]), []);
        assert.deepEqual(collectExtraMetricKeys(undefined), []);
    });

    it('builds a step-aligned series with nulls for missing values', () => {
        assert.deepEqual(extraMetricSeries(rows, 'train/reward_margin'), [0.01, null, 0.05, null]);
        assert.deepEqual(
            extraMetricSeries([{ 'train/x': null }, { 'train/x': 'bad' }, { 'train/x': 2 }], 'train/x'),
            [null, null, 2],
        );
    });

    it('formats metric labels', () => {
        assert.equal(formatMetricLabel('train/reward_margin'), 'reward margin (train)');
        assert.equal(formatMetricLabel('custom_metric'), 'custom metric');
    });

    it('LossChart keeps extras aligned with steps and merges eval results', () => {
        const select = createElement('select');
        domElements['chart-extra-metric'] = select;
        const chart = new LossChart('no-canvas');
        chart.loadHistory([
            { step: 1, loss: 2.0, eval_loss: null, learning_rate: 1e-5, gpu_memory_used: null,
              extra: { 'train/nll_loss': 1.8 } },
            { step: 2, loss: 1.9, eval_loss: null, learning_rate: 1e-5, gpu_memory_used: null },
        ]);
        assert.deepEqual(chart.data.extra, [{ 'train/nll_loss': 1.8 }, {}]);
        assert.deepEqual(chart.extraKeys, ['train/nll_loss']);
        assert.ok(select.innerHTML.includes('value="train/nll_loss"'));
        assert.equal(select._style.display, '');

        chart.addPoint(3, 1.7, null, 1e-5, null, { 'train/nll_loss': 1.6, 'train/or_loss': 0.1 });
        chart.mergeEval(3, 1.75, { 'eval/reward_margin': 0.2 });
        assert.equal(chart.data.steps.length, 3);
        assert.equal(chart.data.evalLoss[2], 1.75);
        assert.deepEqual(chart.data.extra[2],
            { 'train/nll_loss': 1.6, 'train/or_loss': 0.1, 'eval/reward_margin': 0.2 });

        // Eval on a step with no training log appends a point without loss.
        chart.mergeEval(4, 1.7, null);
        assert.deepEqual(chart.data.steps, [1, 2, 3, 4]);
        assert.equal(chart.data.loss[3], null);

        chart.reset();
        assert.deepEqual(chart.data.extra, []);
        assert.equal(select._style.display, 'none');
        delete domElements['chart-extra-metric'];
    });
});
