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
const {
    StorageManager, breadcrumbs, parentPrefix, humanBytes, humanCount, esc, isGuarded, validateBucketName, renderPreview, startPrefix,
} = await import('../../frontend/js/storage.js');

const TIERS = [
    { name: 'internal', bucket: 'datasets', endpoint_host: 'minio.lan:9000', private: true, configured: true,
      hint_prefix: '', default_buckets: ['datasets'], admin_configured: false, env_prefix: 'S3' },
    { name: 'external', bucket: 'shared', endpoint_host: 'r2.example.com', private: false, configured: true,
      hint_prefix: 'runs/', default_buckets: ['shared'], admin_configured: false, env_prefix: 'R2' },
];

const BUCKETS = {
    internal: { ok: true, hint: '', message: '', buckets: [
        { name: 'datasets', default: true }, { name: 'scratch', default: false }] },
    external: { ok: false, message: 'access denied', hint: 'Set R2_ADMIN_ACCESS_KEY / R2_ADMIN_SECRET_KEY',
        buckets: [{ name: 'shared', default: true }] },
};

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
    m.buckets = JSON.parse(JSON.stringify(BUCKETS));
    m.toast = { show() {} };
    m.tier = 'internal';
    m.bucket = 'datasets';
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

    it('parentPrefix goes up one folder', () => {
        assert.equal(parentPrefix('runs/job/adapter/'), 'runs/job/');
        assert.equal(parentPrefix('runs/'), '');
        assert.equal(parentPrefix(''), '');
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

describe('startPrefix', () => {
    it('opens the artifact prefix of the bucket that hosts runs', () => {
        const t = { bucket: 'shared', hint_prefix: 'runs/', bucket_hints: { artifacts: 'merlina/runs/' } };
        assert.equal(startPrefix(t, 'shared'), 'runs/');
        assert.equal(startPrefix(t, 'artifacts'), 'merlina/runs/');
        assert.equal(startPrefix(t, 'other'), '');
        assert.equal(startPrefix(null, 'x'), '');
    });
});

describe('bucket name validation (mirrors the backend)', () => {
    it('accepts valid names', () => {
        for (const n of ['abc', 'my-bucket', 'a.b.c', 'x'.repeat(63)]) assert.equal(validateBucketName(n), null, n);
    });
    it('rejects invalid names', () => {
        for (const n of ['', 'ab', 'x'.repeat(64), 'Upper', 'under_score', '-lead', 'trail-', 'a..b',
            'a.-b', '192.168.1.1', 'xn--abc', 'foo-s3alias', 'a/b']) {
            assert.ok(validateBucketName(n), n);
        }
    });
});

describe('renderPreview', () => {
    it('escapes text and notes truncation', () => {
        const html = renderPreview({ kind: 'text', text: '<script>x</script>', truncated: true, bytes_shown: 262144 });
        assert.match(html, /&lt;script&gt;/);
        assert.doesNotMatch(html, /<script>/);
        assert.match(html, /truncated/);
    });
    it('summarises a safetensors header', () => {
        const html = renderPreview({
            kind: 'safetensors', tensor_count: 2, total_params: 131153920, header_bytes: 96920,
            dtypes: { F32: 131153920 }, metadata: null,
            tensors: [{ name: 'a.<b>', dtype: 'F32', shape: [64, 10240], params: 655360 }],
        });
        assert.match(html, /131\.2M/);
        assert.match(html, /\[64, 10240\]/);
        assert.match(html, /a\.&lt;b&gt;/);
    });
    it('renders parquet schema and rows', () => {
        const html = renderPreview({
            kind: 'parquet', num_rows: 50, num_columns: 2, num_row_groups: 1,
            schema: [{ name: 'prompt', type: 'string' }, { name: 'tags', type: 'list<string>' }],
            rows: [{ prompt: 'hi & bye', tags: ['a'] }],
        });
        assert.match(html, /<strong>50<\/strong> rows/);
        assert.match(html, /hi &amp; bye/);
        assert.match(html, /\[&quot;a&quot;\]/);
    });
    it('binary gets a download hint', () => {
        assert.match(renderPreview({ kind: 'binary' }), /download/);
    });
    it('humanCount', () => {
        assert.equal(humanCount(131153920), '131.2M');
        assert.equal(humanCount(999), '999');
    });
});

describe('StorageManager', () => {
    let calls;
    beforeEach(() => {
        calls = [];
        MerlinaAPI.storageSync = async (req) => {
            calls.push(['sync', req]);
            return req.dry_run ? { dry_run: true, plan: plan() } : { dry_run: false, id: 'sync_1' };
        };
        MerlinaAPI.storageSyncStatus = async () => ({ id: 'sync_1', state: 'running', progress: {} });
        MerlinaAPI.storageDelete = async (req) => {
            calls.push(['delete', req]);
            return { objects: 3, human: '3 B', prefix: 'datasets', deleted: !!req.confirm };
        };
        MerlinaAPI.storageLs = async (req) => {
            calls.push(['ls', req]);
            return { dirs: [], objects: [{ key: 'a.json', name: 'a.json', size: 1, mtime: null }],
                total: 250, offset: req.offset, limit: req.limit, has_more: req.offset + req.limit < 250, search: req.search };
        };
        MerlinaAPI.storageCreateBucket = async (tier, name) => {
            calls.push(['create', tier, name]);
            return { ok: false, message: 'may not create buckets', hint: 'Set R2_ADMIN_ACCESS_KEY' };
        };
        MerlinaAPI.storageDeleteBucket = async (req) => {
            calls.push(['rb', req]);
            return req.dry_run
                ? { ok: true, objects: 2, human: '5 B', bucket: req.name }
                : { ok: true, deleted: true, message: 'Deleted bucket.' };
        };
        MerlinaAPI.storageBuckets = async (tier) => BUCKETS[tier];
    });

    it('lists buckets per tier, stars defaults, and shows the denied hint', () => {
        const m = manager();
        m._renderTiers();
        const html = el('storage-tiers').innerHTML;
        assert.match(html, /data-bucket="scratch"/);
        assert.match(html, /★ default/);
        assert.match(html, /R2_ADMIN_ACCESS_KEY/);
        // default buckets never get a delete button
        assert.doesNotMatch(html, /data-action="bucket-delete" data-tier="internal" data-bucket="datasets"/);
        assert.match(html, /data-action="bucket-delete" data-tier="internal" data-bucket="scratch"/);
    });

    it('offers every other bucket as a destination and marks private -> shared as guarded', () => {
        const m = manager();
        m._renderActions();
        const html = el('storage-actions').innerHTML;
        assert.match(html, /value="internal\|scratch">internal:scratch</);
        assert.match(html, /value="external\|shared">external:shared \(guarded\)/);
        assert.doesNotMatch(html, /value="internal\|datasets"/);
        assert.doesNotMatch(html, /data-action="apply"/);
    });

    it('preview renders refused objects (escaped) and an Apply button', async () => {
        const m = manager();
        m._renderActions();
        await m.preview('external', 'shared', false);
        assert.equal(calls[0][1].dry_run, true);
        assert.equal(calls[0][1].src_bucket, 'datasets');
        assert.equal(calls[0][1].dst_bucket, 'shared');
        const html = el('storage-plan').innerHTML;
        assert.match(html, /2 private object\(s\) refused/);
        assert.match(html, /&lt;script&gt;/);
        assert.doesNotMatch(html, /<script>/);
        assert.match(html, /data-action="apply"/);
    });

    it('apply without a preview does nothing', async () => {
        const m = manager();
        await m.apply();
        assert.equal(calls.length, 0);
    });

    it('apply replays the previewed request as a real run, once', async () => {
        const m = manager();
        m._renderActions();
        await m.preview('internal', 'scratch', true);
        m._poll = () => {};
        await m.apply();
        assert.equal(calls.length, 2);
        assert.deepEqual({ ...calls[1][1], dry_run: true }, calls[0][1]);
        assert.equal(calls[1][1].dry_run, false);
        assert.equal(calls[1][1].move, true);
        assert.equal(calls[1][1].verify, true);
        assert.equal(m.plan, null);
    });

    it('browse passes bucket, search, sort and paging; pager pages', async () => {
        const m = manager();
        m.tier = null;
        await m.browse('internal', 'scratch', 'runs/');
        let req = calls.at(-1)[1];
        assert.deepEqual([req.bucket, req.prefix, req.offset, req.limit], ['scratch', 'runs/', 0, 100]);
        assert.match(el('storage-pager').innerHTML, /1–1 of 250/);
        m.view.sort = 'size';
        m.view.order = 'desc';
        m.view.search = 'json';
        await m._onClick({ target: { closest: () => ({ dataset: { action: 'page', dir: '1' } }) } });
        req = calls.at(-1)[1];
        assert.deepEqual([req.offset, req.sort, req.order, req.search], [100, 'size', 'desc', 'json']);
        // moving to another folder resets search and paging
        await m.browse('internal', 'scratch', 'runs/job/');
        req = calls.at(-1)[1];
        assert.deepEqual([req.offset, req.search], [0, '']);
        assert.match(el('storage-crumbs').innerHTML, /internal:scratch<\/button> <span aria-hidden="true">\/<\/span> /);
    });

    it('create bucket validates client-side before calling the API', async () => {
        const m = manager();
        el('storage-new-bucket-internal').value = 'Bad_Name';
        await m.createBucket('internal');
        assert.equal(calls.length, 0);
        assert.match(el('storage-new-bucket-msg-internal').innerHTML, /lower-case/);
        el('storage-new-bucket-internal').value = 'good-name';
        await m.createBucket('internal');
        assert.deepEqual(calls.at(-1), ['create', 'internal', 'good-name']);
        assert.match(el('storage-new-bucket-msg-internal').innerHTML, /R2_ADMIN_ACCESS_KEY/);
    });

    it('bucket delete shows size, needs the typed name and "empty first" when not empty', async () => {
        const m = manager();
        await m.openBucketDelete('internal', 'scratch');
        const html = el('storage-bucket-panel').innerHTML;
        assert.match(html, /2 object\(s\), 5 B/);
        assert.match(html, /Empty it first/);
        el('storage-bucket-confirm').value = 'scratch';
        el('storage-bucket-empty').checked = false;
        assert.equal(m._bucketDeleteReady(), false);
        await m.deleteBucket();
        assert.equal(calls.filter(c => c[0] === 'rb' && !c[1].dry_run).length, 0);
        el('storage-bucket-empty').checked = true;
        el('storage-bucket-confirm').value = 'scratc';
        assert.equal(m._bucketDeleteReady(), false);
        el('storage-bucket-confirm').value = 'scratch';
        assert.equal(m._bucketDeleteReady(), true);
        m.browse = async () => {};
        await m.deleteBucket();
        const real = calls.filter(c => c[0] === 'rb' && !c[1].dry_run);
        assert.equal(real.length, 1);
        assert.deepEqual([real[0][1].confirm_name, real[0][1].empty_first], ['scratch', true]);
    });

    it('prefix delete needs the prefix typed exactly and carries the bucket', async () => {
        const m = manager();
        await m._openDelete();
        assert.match(el('storage-delete-confirm').innerHTML, /Type <code>datasets\/<\/code> to confirm/);
        const input = el('storage-delete-input');
        const go = el('storage-delete-go');
        input.value = 'datasets';
        m._onInput({ target: input });
        assert.equal(go.disabled, true);
        await m.deleteSelected();
        assert.equal(calls.filter(c => c[0] === 'delete' && c[1].confirm).length, 0);
        input.value = 'datasets/';
        m._onInput({ target: input });
        assert.equal(go.disabled, false);
        assert.equal(calls.find(c => c[0] === 'delete')[1].bucket, 'datasets');
    });

    it('no delete is offered at the bucket root', () => {
        const m = manager();
        m.selected = '';
        m.prefix = '';
        m._renderActions();
        assert.doesNotMatch(el('storage-actions').innerHTML, /delete-open/);
    });
});

describe('folder navigation', () => {
    it('Up goes to the parent, Back retraces visited folders', async () => {
        const m = manager();
        const calls = [];
        const orig = MerlinaAPI.storageLs;
        MerlinaAPI.storageLs = async (q) => { calls.push(`${q.bucket}:${q.prefix}`); return { dirs: [], objects: [], total: 0, offset: 0 }; };
        try {
            await m.browse('internal', 'datasets', 'runs/job/adapter/');
            assert.match(el('storage-crumbs').innerHTML, /data-action="back"[^>]*>/);
            assert.doesNotMatch(el('storage-crumbs').innerHTML, /data-action="up"[^>]*disabled/);
            await m.up();
            assert.equal(m.prefix, 'runs/job/');
            await m.back();
            assert.equal(m.prefix, 'runs/job/adapter/');
            await m.back();
            assert.equal(m.prefix, 'datasets/');  // where manager() started
            assert.equal(m.history.length, 0);
            assert.equal(m.back(), undefined);    // nothing left
            await m.browse('internal', 'datasets', '');
            assert.match(el('storage-crumbs').innerHTML, /data-action="up"[^>]*disabled/);
            assert.equal(m.up(), undefined);
            assert.deepEqual(calls, ['datasets:runs/job/adapter/', 'datasets:runs/job/',
                'datasets:runs/job/adapter/', 'datasets:datasets/', 'datasets:']);
        } finally {
            MerlinaAPI.storageLs = orig;
        }
    });

    it('listing shows a ".." row below the root only', () => {
        const m = manager();
        m._renderListing({ dirs: [{ prefix: 'datasets/a/', name: 'a/' }], objects: [], total: 1, offset: 0 });
        assert.match(el('storage-browser').innerHTML, /data-action="up"/);
        m.prefix = '';
        m._renderListing({ dirs: [{ prefix: 'a/', name: 'a/' }], objects: [], total: 1, offset: 0 });
        assert.doesNotMatch(el('storage-browser').innerHTML, /data-action="up"/);
    });
});
