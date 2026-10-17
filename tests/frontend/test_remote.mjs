// Unit tests for remote.js - Remote GPUs section (request building, plan -> start gating)
// Run with: node --test tests/frontend/test_remote.mjs

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// ─── Minimal DOM shims ──────────────────────────────────────────────────────

const elements = new Map();
function el(id) {
    if (!elements.has(id)) {
        elements.set(id, {
            id, innerHTML: '', value: '', checked: false, disabled: false, style: {},
            addEventListener() {}, focus() {},
        });
    }
    return elements.get(id);
}
let source = 'hf';

globalThis.window = {
    location: { protocol: 'http:', host: 'localhost:8000' },
    addEventListener: () => {},
};
globalThis.document = {
    getElementById: (id) => el(id),
    querySelector: (sel) => (sel.includes('remote-source') ? { value: source } : null),
    querySelectorAll: () => [],
    createElement: () => ({}),
    addEventListener: () => {},
    activeElement: null,
};

const { MerlinaAPI } = await import('../../frontend/js/api.js');
const {
    RemoteManager, buildRemoteRequest, planKey, splitArgs, sessionSpend, sessionPhase, sessionModelLabel, duration,
} = await import('../../frontend/js/remote.js');

const PLAN = {
    sizing: { gpu_type_id: 'NVIDIA H200', gpu_count: 2, est_cost_per_hr: 10.58, cloud_type: 'secure', disk_required_gb: 211 },
    vram_breakdown: { weights_gb: 160, kv_cache_gb: 20, total_gb: 185 },
    effective_max_hours: 4.73, deadline_reason: 'max_total_cost_usd', est_max_total_cost_usd: 50,
    served_model_name: 'C0', lora: false, max_model_len: 131072, max_num_seqs: 4, tensor_parallel_size: 2,
    vllm_command: ['vllm', 'serve', '<store:models/C0>'], warnings: ['<b>careful</b>'],
};

function fill(values) {
    for (const [id, v] of Object.entries(values)) el(id).value = v;
}

describe('buildRemoteRequest', () => {
    it('builds an HF request and drops empty optionals', () => {
        const { body, errors } = buildRemoteRequest({
            source: 'hf', hf_repo: ' Qwen/Qwen3-8B ', revision: '', max_runtime_hours: '2',
            max_num_seqs: '8', gpu_count: '', cloud_type: 'secure', vllm_args: '',
        });
        assert.deepEqual(errors, []);
        assert.deepEqual(body, {
            model: { hf_repo: 'Qwen/Qwen3-8B' }, max_runtime_hours: 2, max_num_seqs: 8, cloud_type: 'secure',
        });
    });

    it('builds store and job sources', () => {
        assert.deepEqual(buildRemoteRequest({ source: 'store', store_model: 'C0', max_runtime_hours: 5 }).body.model,
            { store_model: 'C0' });
        assert.deepEqual(buildRemoteRequest({ source: 'job', job_id: 'job_1', artifact: 'merged', max_runtime_hours: 1 }).body.model,
            { job_id: 'job_1', artifact: 'merged' });
    });

    it('rejects local paths, missing sources and bad runtimes', () => {
        assert.ok(buildRemoteRequest({ source: 'hf', hf_repo: '/mnt/models/x', max_runtime_hours: 1 }).errors.length);
        assert.ok(buildRemoteRequest({ source: 'store', store_model: '', max_runtime_hours: 1 }).errors.length);
        assert.ok(buildRemoteRequest({ source: 'hf', hf_repo: 'a/b', max_runtime_hours: 30 }).errors.length);
        assert.ok(buildRemoteRequest({ source: 'hf', hf_repo: 'a/b', max_runtime_hours: '' }).errors.length);
    });

    it('passes caps, flags and the token through', () => {
        const { body } = buildRemoteRequest({
            source: 'hf', hf_repo: 'a/b', max_runtime_hours: '5', max_total_cost_usd: '50', idle_timeout_minutes: '30',
            vllm_args: '--enable-prefix-caching --chat-template "a b.jinja"', keep_instance_on_failure: true,
            hf_token: 'hf_x', cloud_type: 'community',
        });
        assert.equal(body.max_total_cost_usd, 50);
        assert.equal(body.idle_timeout_minutes, 30);
        assert.deepEqual(body.vllm_args, ['--enable-prefix-caching', '--chat-template', 'a b.jinja']);
        assert.equal(body.keep_instance_on_failure, true);
        assert.equal(body.cloud_type, 'community');
        assert.equal(body.hf_token, 'hf_x');
        assert.ok(!planKey(body).includes('hf_x'), 'the token never enters the plan key');
    });
});

describe('helpers', () => {
    it('splitArgs honours quotes', () => {
        assert.deepEqual(splitArgs(`--a 1 --b 'x y' "z"`), ['--a', '1', '--b', 'x y', 'z']);
        assert.deepEqual(splitArgs(''), []);
    });

    it('sessionSpend prefers the controller figure, estimates live sessions', () => {
        assert.deepEqual(sessionSpend({ status: 'stopped', metrics: { est_cost_usd: 12.5 } }), { usd: 12.5, estimate: false });
        const live = sessionSpend({ status: 'ready', created_at: 0, metrics: { instance: { cost_per_hr: 10 } } }, 1800);
        assert.equal(live.usd, 5);
        assert.equal(live.estimate, true);
        assert.equal(sessionSpend({ status: 'queued', created_at: 0, metrics: {} }, 100).usd, null);
    });

    it('sessionPhase names the worker phase and the stop reason', () => {
        assert.equal(sessionPhase({ status: 'starting', metrics: { phase: 'preparing_model' } }), 'Downloading the model');
        assert.equal(sessionPhase({ status: 'expired', stop_reason: 'max_total_cost_usd' }), 'expired (cost cap)');
    });

    it('labels and durations', () => {
        assert.equal(sessionModelLabel({ request: { model: { store_model: 'C0' } } }), 'store: C0');
        assert.equal(sessionModelLabel({ request: { model: { hf_repo: 'a/b', revision: 'r' } } }), 'a/b@r');
        assert.equal(duration(3725), '1h 02m');
        assert.equal(duration(65), '1m 05s');
    });
});

describe('RemoteManager plan -> start gating', () => {
    let mgr, started;
    beforeEach(() => {
        elements.clear();
        source = 'store';
        started = [];
        MerlinaAPI.planRemoteInference = async () => PLAN;
        MerlinaAPI.startRemoteInference = async (body) => { started.push(body); return { session_id: 'inf_1', message: 'ok' }; };
        MerlinaAPI.listRemoteInferenceSessions = async () => ({ sessions: [] });
        MerlinaAPI.getRemoteInferenceSession = async () => null;
        mgr = new RemoteManager();
        mgr.toast = { success() {}, error() {}, info() {} };
        fill({ 'remote-store-model': 'C0', 'remote-max-hours': '5', 'remote-max-total': '50', 'remote-cloud-type': 'secure' });
    });

    it('needs a plan, then an explicit confirm, before renting', async () => {
        await mgr._plan();
        const html = el('remote-plan').innerHTML;
        assert.match(html, /2× NVIDIA H200/);
        assert.match(html, /\$50\.00 max/);
        assert.match(html, /&lt;b&gt;careful/, 'warnings are escaped');
        assert.match(html, /data-act="start"/);
        assert.doesNotMatch(html, /start-confirm/);

        await mgr._onClick({ target: { closest: () => ({ dataset: { act: 'start' } }) } });
        assert.match(el('remote-plan').innerHTML, /start-confirm/);
        assert.equal(started.length, 0, 'arming does not rent');

        await mgr._onClick({ target: { closest: () => ({ dataset: { act: 'start-confirm' } }) } });
        assert.equal(started.length, 1);
        assert.deepEqual(started[0].model, { store_model: 'C0' });
        assert.equal(started[0].max_total_cost_usd, 50);
    });

    it('a form edit after planning withdraws Start', async () => {
        await mgr._plan();
        mgr.armed = true;
        el('remote-max-total').value = '500';
        mgr._onFormChange({ target: { closest: () => true, name: '', id: 'remote-max-total' } });
        assert.doesNotMatch(el('remote-plan').innerHTML, /data-act="start/);
        assert.match(el('remote-plan').innerHTML, /Plan again/);
        await mgr._start();
        assert.equal(started.length, 0, 'a stale plan never starts');
    });

    it('shows validation errors without calling the API', async () => {
        let called = false;
        MerlinaAPI.planRemoteInference = async () => { called = true; return PLAN; };
        el('remote-store-model').value = '';
        await mgr._plan();
        assert.equal(called, false);
        assert.match(el('remote-plan').innerHTML, /Pick a model/);
    });
});
