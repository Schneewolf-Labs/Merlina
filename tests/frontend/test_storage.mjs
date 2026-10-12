// Unit tests for storage.js - Object storage section (tiers, sync preview/apply, delete confirm)
// Run with: node --test tests/frontend/test_storage.mjs

import { describe, it, beforeEach } from 'node:test';
import assert from 'node:assert/strict';

// ─── Minimal DOM shims ──────────────────────────────────────────────────────

const elements = new Map();
function el(id) {
    if (!elements.has(id)) {
        elements.set(id, {
            id, innerHTML: '', value: '', checked: false, disabled: false,
            addEventListener() {},
        });
    }
    return elements.get(id);
}

globalThis.window = {
    location: { protocol: 'http:', host: 'localhost:8000' },
    addEventListener: () => {},
};
globalThis.document = {
    getElementById: (id) => el(id),
    querySelectorAll: () => [],
    createElement: () => ({}),
    addEventListener: () => {},
};

const { MerlinaAPI } = await import('../../frontend/js/api.js');
const { StorageManager, breadcrumbs, humanBytes, esc, isGuarded } = await import('../../frontend/js/storage.js');

const TIERS = [
    { name: 'internal', bucket: 'datasets', endpoint_host: 'minio.lan:9000', private: true, configured: true, hint_prefix: '' },
    { name: 'external', bucket: 'shared', endpoint_host: 'r2.example.com', private: false, configured: true, hint_prefix: 'runs/' },
];

function plan(overrides = {}) {
    return {
        src: 'internal', dst: 'external', prefix: 'datasets/', guarded: true,
        counts: { to_copy: 1, skipped: 0, refused: 2, errors: 0, copied: 0, deleted: 0, meta_missing: 0 },
        bytes: { to_copy: 10, to_copy_human: '10 B', skipped: 0, skipped_human: '0 B', copied: 0, copied_human: '0 B' },
        to_copy: [{ key: 'datasets/open/r1/a', size: 10, reason: 'missing' }],
        skipped: [],
        refused: [
            { key: 'datasets/secret/r1/a', size: 1, reason: 'dataset manifest is private' },
            { key: 'datasets/<script>/x', size: 1, reason: 'path matches a private marker' },
        ],
        errors: [], copied: [], deleted: [], ...overrides,
    };
}

function manager() {
    elements.clear();
    const m = new StorageManager();
    m.tiers = TIERS;
    m.tier = 'internal';
    m.prefix = 'datasets/';
    m.selected = 'datasets/';
    return m;
}

describe('storage helpers', () => {
    it('breadcrumbs split a prefix into navigable steps', () => {
        assert.deepEqual(breadcrumbs('runs/job/adapter/'), [
            { label: '/', prefix: '' },
            { label: 'runs', prefix: 'runs/' },
            { label: 'job', prefix: 'runs/job/' },
            { label: 'adapter', prefix: 'runs/job/adapter/' },
        ]);
        assert.deepEqual(breadcrumbs(''), [{ label: '/', prefix: '' }]);
    });

    it('humanBytes formats sizes', () => {
        assert.equal(humanBytes(5), '5 B');
        assert.equal(humanBytes(1536), '1.5 KiB');
        assert.equal(humanBytes(8.6 * 1024 ** 3), '8.6 GiB');
        assert.equal(humanBytes(null), '—');
    });

    it('esc escapes HTML', () => {
        assert.equal(esc('<a href="x">\'</a>'), '&lt;a href=&quot;x&quot;&gt;&#39;&lt;/a&gt;');
    });

    it('isGuarded only for private -> shared', () => {
        assert.equal(isGuarded(TIERS[0], TIERS[1]), true);
        assert.equal(isGuarded(TIERS[1], TIERS[0]), false);
    });
});

describe('StorageManager actions', () => {
    let calls;
    beforeEach(() => {
        calls = [];
        MerlinaAPI.storageSync = async (req) => {
            calls.push(req);
            return req.dry_run ? { dry_run: true, plan: plan() } : { dry_run: false, id: 'sync_1' };
        };
        MerlinaAPI.storageSyncStatus = async () => ({ id: 'sync_1', state: 'running', progress: {} });
        MerlinaAPI.storageDelete = async (req) => {
            calls.push(req);
            return { objects: 3, human: '3 B', prefix: 'datasets', deleted: !!req.confirm };
        };
    });

    it('marks private -> shared actions as guarded and offers no Apply before a preview', () => {
        const m = manager();
        m._renderActions();
        const html = el('storage-actions').innerHTML;
        assert.match(html, /Sync to external 🛡️/);
        assert.match(html, /Move to external 🛡️/);
        assert.doesNotMatch(html, /data-action="apply"/);
    });

    it('shows no guard marker when browsing a shared tier', () => {
        const m = manager();
        m.tier = 'external';
        m._renderActions();
        assert.doesNotMatch(el('storage-actions').innerHTML, /🛡️/);
    });

    it('preview renders the refused objects (escaped) and an Apply button', async () => {
        const m = manager();
        m._renderActions();
        await m.preview('external', false);
        assert.equal(calls[0].dry_run, true);
        const html = el('storage-plan').innerHTML;
        assert.match(html, /2 private object\(s\) refused/);
        assert.match(html, /datasets\/secret\/r1\/a/);
        assert.match(html, /&lt;script&gt;/);
        assert.doesNotMatch(html, /<script>/);
        assert.match(html, /data-action="apply"/);
    });

    it('apply without a preview does nothing', async () => {
        const m = manager();
        m.toast = { show() {} };
        await m.apply();
        assert.equal(calls.length, 0);
    });

    it('apply replays the previewed request as a real run, once', async () => {
        const m = manager();
        m._renderActions();
        await m.preview('external', true);
        m._poll = () => {};
        await m.apply();
        assert.equal(calls.length, 2);
        assert.deepEqual({ ...calls[1], dry_run: true }, calls[0]);
        assert.equal(calls[1].dry_run, false);
        assert.equal(calls[1].move, true);
        assert.equal(calls[1].verify, true);
        assert.equal(m.plan, null);  // a second Apply needs a fresh preview
    });

    it('delete needs the prefix typed exactly', async () => {
        const m = manager();
        await m._openDelete();
        assert.match(el('storage-delete-confirm').innerHTML, /Type <code>datasets\/<\/code> to confirm/);
        const input = el('storage-delete-input');
        const go = el('storage-delete-go');
        input.value = 'datasets';
        m._onInput({ target: input });
        assert.equal(go.disabled, true);
        await m.deleteSelected();
        assert.equal(calls.filter(c => c.confirm).length, 0);
        input.value = 'datasets/';
        m._onInput({ target: input });
        assert.equal(go.disabled, false);
    });

    it('no delete is offered at the bucket root', () => {
        const m = manager();
        m.selected = '';
        m.prefix = '';
        m._renderActions();
        assert.doesNotMatch(el('storage-actions').innerHTML, /delete-open/);
    });
});
