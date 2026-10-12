// Object Storage Section Manager
//
// Drives the "Object storage" section:
//   * tier cards listing every bucket the tier's key can see (default buckets
//     starred, sizes loaded lazily), with create / info / delete per bucket;
//   * a bucket browser with breadcrumbs, search, sorting and pagination;
//   * an object panel with details, metadata (incl. our sha256), an inline
//     preview (text / safetensors header / parquet head) and a download link;
//   * sync / move / delete on the selected prefix, across tiers and buckets.
//
// Every sync or move is previewed (dry run) first; Apply only appears on a
// fresh plan, so a private -> shared action is never offered without the
// server's guard result (refused objects) on screen. Destructive actions use
// inline type-the-name confirmations, never window.confirm.

import { MerlinaAPI } from './api.js';
import { Toast } from './ui.js';

export function esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

export function humanBytes(n) {
    if (n === null || n === undefined || Number.isNaN(n)) return '—';
    let v = Number(n);
    for (const unit of ['B', 'KiB', 'MiB', 'GiB', 'TiB']) {
        if (Math.abs(v) < 1024 || unit === 'TiB') {
            return unit === 'B' ? `${Math.round(v)} B` : `${v.toFixed(1)} ${unit}`;
        }
        v /= 1024;
    }
    return `${n} B`;
}

export function humanCount(n) {
    const v = Number(n) || 0;
    if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
    if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
    return String(v);
}

/** Split a prefix ("a/b/") into breadcrumb steps [{label, prefix}]. */
export function breadcrumbs(prefix) {
    const crumbs = [{ label: '/', prefix: '' }];
    let acc = '';
    for (const part of String(prefix || '').split('/').filter(Boolean)) {
        acc += `${part}/`;
        crumbs.push({ label: part, prefix: acc });
    }
    return crumbs;
}

/** The folder above ``prefix`` ("a/b/" -> "a/", "a/" -> ""). */
export function parentPrefix(prefix) {
    const parts = String(prefix || '').split('/').filter(Boolean);
    parts.pop();
    return parts.length ? `${parts.join('/')}/` : '';
}

/** True when copying from `src` to `dst` crosses from private to shared. */
export function isGuarded(src, dst) {
    return !!(src && dst && src.private && !dst.private);
}

/**
 * Why `name` is not a valid new S3 bucket name, or null if it is.
 * Mirrors storage_tiers.validate_bucket_name.
 */
export function validateBucketName(name) {
    if (!name) return 'Bucket name is required.';
    if (name.length < 3 || name.length > 63) return 'Bucket names must be 3-63 characters long.';
    if (!/^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$/.test(name)) {
        return "Use only lower-case letters, digits, '.' and '-', starting and ending with a letter or digit.";
    }
    if (name.includes('..') || name.includes('.-') || name.includes('-.')) {
        return "Bucket names may not contain '..', '.-' or '-.'.";
    }
    if (/^\d{1,3}(\.\d{1,3}){3}$/.test(name)) return 'Bucket names may not look like an IP address.';
    if (/^(xn--|sthree-)/.test(name) || /(-s3alias|--ol-s3|--x-s3)$/.test(name)) {
        return 'Bucket names may not use a reserved prefix or suffix.';
    }
    return null;
}

const POLL_MS = 1000;
const PAGE = 100;
const MAX_TENSOR_ROWS = 300;

export class StorageManager {
    constructor() {
        this.toast = new Toast();
        this.tiers = [];
        this.buckets = {};       // tier -> /storage/buckets response
        this.sizes = {};         // "tier/bucket" -> du result | 'loading' | {error}
        this.tier = null;        // tier being browsed
        this.bucket = null;      // bucket being browsed
        this.prefix = '';        // prefix being browsed
        this.selected = '';      // prefix/key the actions apply to
        this.history = [];       // previous {tier, bucket, prefix} locations, for Back
        this.view = { search: '', sort: 'name', order: 'asc', offset: 0 };
        this.listing = null;
        this.plan = null;        // last dry-run plan {request, plan}
        this.bucketDelete = null;  // {tier, bucket, preview}
        this.pollTimer = null;
        this.searchTimer = null;
        this.bound = false;
        this.loaded = false;
    }

    init() {
        if (this.bound) return;
        const refresh = document.getElementById('storage-refresh-btn');
        if (!refresh) return;  // section not on this page
        refresh.addEventListener('click', () => this.load(true));
        // Event delegation keeps the dynamically-rendered markup simple.
        const section = document.getElementById('storage-section');
        section?.addEventListener('click', e => this._onClick(e));
        section?.addEventListener('input', e => this._onInput(e));
        section?.addEventListener('change', e => this._onChange(e));
        this.bound = true;
    }

    _tier(name) {
        return this.tiers.find(t => t.name === name) || null;
    }

    _bucketList(tier) {
        return this.buckets[tier]?.buckets || [];
    }

    // ── tiers & buckets ──────────────────────────────────────────────────────

    /** Load tiers, then every tier's bucket list; open the first default bucket. */
    async load(force = false) {
        if (this.loaded && !force) return;
        this.loaded = true;
        const cards = document.getElementById('storage-tiers');
        if (cards) cards.innerHTML = '<p style="color: var(--text-muted);">Loading tiers…</p>';
        try {
            const data = await MerlinaAPI.getStorageTiers(false);
            this.tiers = data.tiers || [];
        } catch (err) {
            if (cards) cards.innerHTML = `<p style="color: var(--danger-dark);">Failed to load tiers: ${esc(err.message)}</p>`;
            return;
        }
        if (force) this.sizes = {};
        await Promise.all(this.tiers.filter(t => t.configured).map(t => this.loadBuckets(t.name, false)));
        this._renderTiers();
        // Size the default buckets in the background; others are on demand.
        for (const t of this.tiers.filter(x => x.configured)) {
            for (const b of this._bucketList(t.name).filter(x => x.default)) this.sizeBucket(t.name, b.name);
        }
        const first = this.tiers.find(t => t.configured);
        if (!first) {
            this._setBrowser('<p style="color: var(--text-muted);">No storage tier is configured. See docs/storage-tiers.md.</p>');
            return;
        }
        if (this.tier && this._tier(this.tier)?.configured) {
            await this.browse(this.tier, this.bucket, this.prefix);
        } else {
            await this.browse(first.name, first.bucket, first.hint_prefix || '');
        }
    }

    async loadBuckets(tier, render = true) {
        try {
            this.buckets[tier] = await MerlinaAPI.storageBuckets(tier);
        } catch (err) {
            const t = this._tier(tier);
            this.buckets[tier] = {
                ok: false, message: err.message, hint: '',
                buckets: (t?.default_buckets || []).map(name => ({ name, default: true })),
            };
        }
        if (render) this._renderTiers();
    }

    async sizeBucket(tier, bucket) {
        const k = `${tier}/${bucket}`;
        if (this.sizes[k] === 'loading') return;
        this.sizes[k] = 'loading';
        this._renderTiers();
        try {
            this.sizes[k] = await MerlinaAPI.storageDu(tier, '', bucket);
        } catch (err) {
            this.sizes[k] = { error: err.message };
        }
        this._renderTiers();
    }

    _sizeCell(tier, bucket) {
        const s = this.sizes[`${tier}/${bucket}`];
        if (s === 'loading') return '<span style="color: var(--text-muted);">sizing…</span>';
        if (s && s.error) return `<span style="color: var(--text-muted);" title="${esc(s.error)}">size unavailable</span>`;
        if (s) return `${esc(s.human)} · ${s.objects} obj`;
        return `<button type="button" class="link-button" data-action="bucket-size" data-tier="${esc(tier)}" data-bucket="${esc(bucket)}">Size</button>`;
    }

    _renderTiers() {
        const el = document.getElementById('storage-tiers');
        if (!el) return;
        // Re-rendering replaces the inputs: keep any half-typed bucket names.
        const typed = {};
        for (const t of this.tiers) {
            const v = document.getElementById(`storage-new-bucket-${t.name}`)?.value;
            if (v) typed[t.name] = v;
        }
        el.innerHTML = this.tiers.map(t => {
            const badge = t.private
                ? '<span class="storage-badge private" title="Private data never leaves this tier">🔒 private</span>'
                : '<span class="storage-badge shared">🌐 shared</span>';
            if (!t.configured) {
                return `<div class="metric-card storage-tier-card" style="opacity: 0.6;">
                    <span class="metric-label">${esc(t.name)}</span>
                    <span>not configured</span>
                </div>`;
            }
            const admin = t.admin_configured
                ? '<span class="storage-badge shared" title="An admin key is configured for bucket operations">🔑 admin key</span>'
                : '';
            const info = this.buckets[t.name] || { buckets: [] };
            const rows = (info.buckets || []).map(b => {
                const active = t.name === this.tier && b.name === this.bucket;
                return `<li class="storage-bucket-row${active ? ' active' : ''}">
                    <button type="button" class="link-button" data-action="open-bucket" data-tier="${esc(t.name)}"
                        data-bucket="${esc(b.name)}" title="Browse this bucket">🪣 ${esc(b.name)}</button>
                    ${b.default ? '<span class="storage-badge default" title="Configured in Merlina; cannot be deleted here">★ default</span>' : ''}
                    <span class="storage-bucket-size">${this._sizeCell(t.name, b.name)}</span>
                    <span class="storage-bucket-tools">
                        <button type="button" class="link-button" data-action="bucket-info" data-tier="${esc(t.name)}" data-bucket="${esc(b.name)}" title="Bucket info" aria-label="Info for ${esc(b.name)}">ℹ️</button>
                        ${b.default ? '' : `<button type="button" class="link-button" data-action="bucket-delete" data-tier="${esc(t.name)}" data-bucket="${esc(b.name)}" title="Delete bucket" aria-label="Delete ${esc(b.name)}">🗑️</button>`}
                    </span>
                </li>`;
            }).join('');
            const denied = info.ok === false
                ? `<p class="storage-hint">${esc(info.message)}${info.hint ? `<br><strong>To manage buckets:</strong> ${esc(info.hint)}` : ''}</p>`
                : '';
            return `<div class="metric-card storage-tier-card${t.name === this.tier ? ' active' : ''}">
                <span class="metric-label">${esc(t.name)} ${badge} ${admin}</span>
                <span style="font-size: 0.8em; color: var(--text-muted);">${esc(t.endpoint_host || 'default endpoint')}</span>
                <ul class="storage-bucket-list">${rows || '<li style="color: var(--text-muted);">No buckets.</li>'}</ul>
                ${denied}
                <div class="storage-new-bucket">
                    <input type="text" id="storage-new-bucket-${esc(t.name)}" data-tier="${esc(t.name)}" class="storage-new-bucket-input"
                        value="${esc(typed[t.name] || '')}" placeholder="new-bucket-name" autocomplete="off" spellcheck="false"
                        aria-label="New bucket name on ${esc(t.name)}">
                    <button type="button" class="action-button" style="max-width: 120px;" data-action="bucket-create" data-tier="${esc(t.name)}">＋ Create</button>
                </div>
                <div class="storage-new-bucket-msg" id="storage-new-bucket-msg-${esc(t.name)}"></div>
            </div>`;
        }).join('');
    }

    async createBucket(tier) {
        const input = document.getElementById(`storage-new-bucket-${tier}`);
        const msg = document.getElementById(`storage-new-bucket-msg-${tier}`);
        const name = (input?.value || '').trim();
        const problem = validateBucketName(name);
        if (problem) {
            if (msg) msg.innerHTML = `<span style="color: var(--danger-dark);">${esc(problem)}</span>`;
            return;
        }
        if (msg) msg.innerHTML = '<span style="color: var(--text-muted);">Creating…</span>';
        try {
            const r = await MerlinaAPI.storageCreateBucket(tier, name);
            if (!r.ok) {
                if (msg) msg.innerHTML = `<span style="color: var(--danger-dark);">${esc(r.message)}</span>${r.hint ? `<br><span class="storage-hint">${esc(r.hint)}</span>` : ''}`;
                return;
            }
            if (input) input.value = '';
            this.toast.show(r.message, 'success');
            await this.loadBuckets(tier);
        } catch (err) {
            if (msg) msg.innerHTML = `<span style="color: var(--danger-dark);">${esc(err.message)}</span>`;
        }
    }

    _setBucketPanel(html) {
        const el = document.getElementById('storage-bucket-panel');
        if (el) el.innerHTML = html;
    }

    async showBucketInfo(tier, bucket) {
        this._setBucketPanel('<p style="color: var(--text-muted);">Loading bucket info…</p>');
        try {
            const i = await MerlinaAPI.storageBucketInfo(tier, bucket);
            const notes = (i.notes || []).map(n => `<li>${esc(n)}</li>`).join('');
            this._setBucketPanel(`<div class="spell-section" style="margin: 0;">
                <h4 style="margin-top: 0;">🪣 ${esc(i.bucket)} <span style="font-weight: normal; color: var(--text-muted);">on ${esc(i.tier)}</span></h4>
                <dl class="storage-kv">
                    <dt>Privacy</dt><dd>${i.private ? '🔒 private tier' : '🌐 shared tier'}</dd>
                    <dt>Default</dt><dd>${i.default ? '★ configured in Merlina (protected)' : 'no'}</dd>
                    <dt>Region</dt><dd>${esc(i.region || '—')}</dd>
                    <dt>Created</dt><dd>${esc(i.creation_date || '—')}</dd>
                </dl>
                ${notes ? `<ul class="storage-hint">${notes}</ul>` : ''}
                <button type="button" class="link-button" data-action="panel-close">Close</button>
            </div>`);
        } catch (err) {
            this._setBucketPanel(`<p style="color: var(--danger-dark);">${esc(err.message)}</p>`);
        }
    }

    /** Inline bucket delete: count first, then type-the-name confirm. */
    async openBucketDelete(tier, bucket) {
        this.bucketDelete = null;
        this._setBucketPanel('<p style="color: var(--text-muted);">Counting objects…</p>');
        try {
            const preview = await MerlinaAPI.storageDeleteBucket({ tier, name: bucket, dry_run: true });
            if (!preview.ok) {
                this._setBucketPanel(`<p style="color: var(--danger-dark);">${esc(preview.message)}</p>${preview.hint ? `<p class="storage-hint">${esc(preview.hint)}</p>` : ''}`);
                return;
            }
            this.bucketDelete = { tier, bucket, preview };
            const nonEmpty = preview.objects > 0;
            this._setBucketPanel(`<div class="storage-danger-box">
                <p style="margin-top: 0;">Permanently delete bucket <code>${esc(bucket)}</code> on <strong>${esc(tier)}</strong>?
                   It holds <strong>${preview.objects} object(s), ${esc(preview.human)}</strong>. This cannot be undone.</p>
                ${nonEmpty ? `<label style="display: inline-flex; gap: 6px; align-items: center;">
                    <input type="checkbox" id="storage-bucket-empty" style="width: auto;">
                    Empty it first (delete all ${preview.objects} object(s), ${esc(preview.human)})</label><br>` : ''}
                <label for="storage-bucket-confirm" style="font-size: 0.9em;">Type <code>${esc(bucket)}</code> to confirm</label>
                <input type="text" id="storage-bucket-confirm" autocomplete="off" spellcheck="false" style="max-width: 420px;">
                <div style="display: flex; gap: 8px; margin-top: 8px;">
                    <button type="button" id="storage-bucket-delete-go" class="action-button danger" style="max-width: 220px;" data-action="bucket-delete-go" disabled>🗑️ Delete bucket</button>
                    <button type="button" class="action-button" style="max-width: 140px;" data-action="panel-close">Cancel</button>
                </div>
            </div>`);
        } catch (err) {
            this._setBucketPanel(`<p style="color: var(--danger-dark);">${esc(err.message)}</p>`);
        }
    }

    _bucketDeleteReady() {
        const d = this.bucketDelete;
        if (!d) return false;
        const typed = document.getElementById('storage-bucket-confirm')?.value;
        const emptied = !d.preview.objects || !!document.getElementById('storage-bucket-empty')?.checked;
        return typed === d.bucket && emptied;
    }

    async deleteBucket() {
        const d = this.bucketDelete;
        if (!d || !this._bucketDeleteReady()) return;
        try {
            const r = await MerlinaAPI.storageDeleteBucket({
                tier: d.tier, name: d.bucket, confirm_name: d.bucket,
                empty_first: !!document.getElementById('storage-bucket-empty')?.checked,
            });
            if (!r.ok) {
                this._setBucketPanel(`<p style="color: var(--danger-dark);">${esc(r.message)}</p>${r.hint ? `<p class="storage-hint">${esc(r.hint)}</p>` : ''}`);
                return;
            }
            this.toast.show(r.message, 'success');
            this.bucketDelete = null;
            this._setBucketPanel('');
            delete this.sizes[`${d.tier}/${d.bucket}`];
            await this.loadBuckets(d.tier);
            if (this.tier === d.tier && this.bucket === d.bucket) {
                await this.browse(d.tier, this._tier(d.tier)?.bucket, '');
            }
        } catch (err) {
            this.toast.show(`Delete failed: ${err.message}`, 'error');
        }
    }

    // ── browser ──────────────────────────────────────────────────────────────

    _setBrowser(html) {
        const el = document.getElementById('storage-browser');
        if (el) el.innerHTML = html;
    }

    /** Browse one level of a bucket (resets search/paging when moving). */
    async browse(tier, bucket, prefix = '', { keepView = false, fromHistory = false } = {}) {
        bucket = bucket || this._tier(tier)?.bucket;
        const moved = tier !== this.tier || bucket !== this.bucket || prefix !== this.prefix;
        if (moved && !fromHistory && this.tier) {
            this.history.push({ tier: this.tier, bucket: this.bucket, prefix: this.prefix });
            if (this.history.length > 100) this.history.shift();
        }
        this.tier = tier;
        this.bucket = bucket;
        this.prefix = prefix;
        if (moved || !keepView) {
            this.selected = prefix;
            this.plan = null;
            if (moved) this.view = { ...this.view, search: '', offset: 0 };
            this._setObject('');
        }
        this._renderTiers();
        this._renderCrumbs();
        this._renderToolbar();
        this._renderActions();
        this._setBrowser('<p style="color: var(--text-muted);">Loading…</p>');
        try {
            this.listing = await MerlinaAPI.storageLs({
                tier: this.tier, bucket: this.bucket, prefix, search: this.view.search,
                sort: this.view.sort, order: this.view.order, offset: this.view.offset, limit: PAGE,
            });
            this._renderListing(this.listing);
        } catch (err) {
            this._setBrowser(`<p style="color: var(--danger-dark);">Failed to list: ${esc(err.message)}</p>`);
        }
    }

    _reload() {
        return this.browse(this.tier, this.bucket, this.prefix, { keepView: true });
    }

    _renderCrumbs() {
        const el = document.getElementById('storage-crumbs');
        if (!el) return;
        const crumbs = breadcrumbs(this.prefix).map((c, i, all) => {
            const last = i === all.length - 1;
            const label = i === 0 ? `${esc(this.tier)}:${esc(this.bucket)}` : esc(c.label);
            return last
                ? `<strong>${label}</strong>`
                : `<button type="button" class="link-button" data-action="crumb" data-prefix="${esc(c.prefix)}">${label}</button>`;
        });
        const nav = `<button type="button" class="link-button" data-action="back" title="Back to the previous folder" ${this.history.length ? '' : 'disabled'}>← Back</button>
            <button type="button" class="link-button" data-action="up" title="Up one folder" ${this.prefix ? '' : 'disabled'}>⬆ Up</button>
            <span style="margin: 0 6px; color: var(--text-muted);">|</span> `;
        el.innerHTML = nav + crumbs.join(' <span aria-hidden="true">/</span> ');
    }

    /** Go back to the previously browsed location. */
    back() {
        const prev = this.history.pop();
        if (!prev) return undefined;
        return this.browse(prev.tier, prev.bucket, prev.prefix, { fromHistory: true });
    }

    /** Go up one folder in the current bucket. */
    up() {
        if (!this.prefix) return undefined;
        return this.browse(this.tier, this.bucket, parentPrefix(this.prefix));
    }

    _renderToolbar() {
        const el = document.getElementById('storage-toolbar');
        if (!el) return;
        const v = this.view;
        const opt = (val, label) => `<option value="${val}"${v.sort === val ? ' selected' : ''}>${label}</option>`;
        el.innerHTML = `
            <input type="search" id="storage-search" placeholder="Filter this folder…" value="${esc(v.search)}"
                aria-label="Filter names in this folder" style="max-width: 260px;">
            <label style="display: inline-flex; gap: 6px; align-items: center;">Sort
                <select id="storage-sort" style="width: auto;">${opt('name', 'Name')}${opt('size', 'Size')}${opt('mtime', 'Modified')}</select>
            </label>
            <button type="button" class="action-button" style="max-width: 110px;" data-action="order"
                title="Toggle sort order">${v.order === 'asc' ? '↑ Asc' : '↓ Desc'}</button>`;
    }

    _renderListing(data) {
        const dirs = data.dirs || [];
        const objects = data.objects || [];
        const pager = document.getElementById('storage-pager');
        if (pager) {
            const from = data.total ? data.offset + 1 : 0;
            const to = data.offset + dirs.length + objects.length;
            pager.innerHTML = data.has_more || data.offset
                ? `<button type="button" class="link-button" data-action="page" data-dir="-1" ${data.offset ? '' : 'disabled'}>‹ Prev</button>
                   <span>${from}–${to} of ${data.total}</span>
                   <button type="button" class="link-button" data-action="page" data-dir="1" ${data.has_more ? '' : 'disabled'}>Next ›</button>`
                : (data.total ? `<span>${data.total} item(s)</span>` : '');
        }
        if (!dirs.length && !objects.length) {
            this._setBrowser(`<p style="color: var(--text-muted);">${data.search ? 'Nothing matches the filter.' : 'Empty.'}</p>`);
            return;
        }
        const upRow = this.prefix && !data.offset ? `<tr>
            <td style="padding: 6px 10px;" colspan="4">
                <button type="button" class="link-button" data-action="up" title="Up one folder">📁 ..</button>
            </td>
        </tr>` : '';
        const dirRows = dirs.map(d => `<tr${d.prefix === this.selected ? ' class="selected"' : ''}>
            <td style="padding: 6px 10px;">
                <button type="button" class="link-button" data-action="open" data-prefix="${esc(d.prefix)}">📁 ${esc(d.name)}</button>
            </td>
            <td style="padding: 6px 10px; text-align: right;">—</td>
            <td style="padding: 6px 10px;"></td>
            <td style="padding: 6px 10px; text-align: right;">
                <button type="button" class="link-button" data-action="select" data-prefix="${esc(d.prefix)}">Select</button>
            </td>
        </tr>`).join('');
        const objRows = objects.map(o => `<tr${o.key === this.selected ? ' class="selected"' : ''}>
            <td style="padding: 6px 10px;">
                <button type="button" class="link-button" data-action="object" data-key="${esc(o.key)}" title="Details & preview">📄 ${esc(o.name)}</button>
            </td>
            <td style="padding: 6px 10px; text-align: right;">${esc(humanBytes(o.size))}</td>
            <td style="padding: 6px 10px; color: var(--text-muted);">${esc((o.mtime || '').slice(0, 19).replace('T', ' '))}</td>
            <td style="padding: 6px 10px; text-align: right;">
                <button type="button" class="link-button" data-action="select" data-prefix="${esc(o.key)}">Select</button>
            </td>
        </tr>`).join('');
        this._setBrowser(`<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
            <thead><tr style="text-align: left; border-bottom: 2px solid var(--border-color);">
                <th style="padding: 6px 10px;">Name</th>
                <th style="padding: 6px 10px; text-align: right;">Size</th>
                <th style="padding: 6px 10px;">Modified</th>
                <th style="padding: 6px 10px;"></th>
            </tr></thead>
            <tbody>${upRow}${dirRows}${objRows}</tbody>
        </table>`);
    }

    // ── object detail & preview ──────────────────────────────────────────────

    _setObject(html) {
        const el = document.getElementById('storage-object');
        if (el) el.innerHTML = html;
    }

    async openObject(key) {
        this.selected = key;
        this._renderActions();
        this._setObject('<p style="color: var(--text-muted);">Loading object…</p>');
        let info;
        try {
            info = await MerlinaAPI.storageObject(this.tier, key, this.bucket);
        } catch (err) {
            this._setObject(`<p style="color: var(--danger-dark);">${esc(err.message)}</p>`);
            return;
        }
        const meta = Object.entries(info.metadata || {}).map(([k, v]) =>
            `<dt>${esc(k)}</dt><dd><code>${esc(v)}</code></dd>`).join('');
        const href = MerlinaAPI.storageDownloadUrl(this.tier, key, this.bucket);
        this._setObject(`<div class="spell-section" style="margin: 0;">
            <div style="display: flex; justify-content: space-between; gap: 8px; align-items: center; flex-wrap: wrap;">
                <h4 style="margin: 0; overflow-wrap: anywhere;">📄 ${esc(info.name)}</h4>
                <span style="display: flex; gap: 8px;">
                    <a class="action-button" style="max-width: 160px; text-decoration: none; text-align: center;" href="${esc(href)}" download="${esc(info.name)}">⬇️ Download</a>
                    <button type="button" class="action-button" style="max-width: 100px;" data-action="object-close">Close</button>
                </span>
            </div>
            <dl class="storage-kv">
                <dt>Key</dt><dd><code>${esc(info.key)}</code></dd>
                <dt>Size</dt><dd>${esc(info.human)} (${esc(info.size)} bytes)</dd>
                <dt>Modified</dt><dd>${esc(info.mtime || '—')}</dd>
                <dt>Content type</dt><dd>${esc(info.content_type || '—')}</dd>
                <dt>ETag</dt><dd><code>${esc(info.etag || '—')}</code></dd>
                ${meta}
            </dl>
            ${info.metadata && info.metadata.sha256 ? '' : '<p class="storage-hint">No sha256 metadata (objects get it when Merlina syncs them).</p>'}
            <div id="storage-preview"><p style="color: var(--text-muted);">${info.preview_kind === 'binary' ? 'No inline preview for this file type.' : 'Loading preview…'}</p></div>
        </div>`);
        if (info.preview_kind === 'binary') return;
        try {
            const p = await MerlinaAPI.storagePreview(this.tier, key, this.bucket);
            const el = document.getElementById('storage-preview');
            if (el) el.innerHTML = renderPreview(p);
        } catch (err) {
            const el = document.getElementById('storage-preview');
            if (el) el.innerHTML = `<p style="color: var(--danger-dark);">Preview failed: ${esc(err.message)}</p>`;
        }
    }

    // ── actions: sync / move / delete ────────────────────────────────────────

    /** Every (tier, bucket) other than the one being browsed. */
    _destinations() {
        const out = [];
        for (const t of this.tiers.filter(x => x.configured)) {
            const names = this._bucketList(t.name).map(b => b.name);
            if (!names.length && t.bucket) names.push(t.bucket);
            for (const b of names) {
                if (t.name === this.tier && b === this.bucket) continue;
                out.push({ tier: t, bucket: b });
            }
        }
        return out;
    }

    _destination() {
        const v = document.getElementById('storage-dest')?.value || '';
        const [tier, bucket] = v.split('|');
        return tier ? { tier: this._tier(tier), bucket } : null;
    }

    /** Action bar for the selected prefix: destination + sync/move, delete. */
    _renderActions() {
        const el = document.getElementById('storage-actions');
        if (!el) return;
        const src = this._tier(this.tier);
        if (!src) { el.innerHTML = ''; return; }
        const target = this.selected || '';
        const dests = this._destinations();
        const options = dests.map(d => {
            const guard = isGuarded(src, d.tier) ? ' 🛡️' : '';
            return `<option value="${esc(d.tier.name)}|${esc(d.bucket)}">${esc(d.tier.name)}:${esc(d.bucket)}${guard}</option>`;
        }).join('');
        const del = target
            ? '<button type="button" class="action-button danger" style="max-width: 200px;" data-action="delete-open">🗑️ Delete…</button>'
            : '';
        const useFolder = target !== this.prefix
            ? `<button type="button" class="link-button" data-action="select" data-prefix="${esc(this.prefix)}">(use current folder)</button>`
            : '';
        el.innerHTML = `
            <p style="margin: 0 0 8px 0;">Selected: <code>${esc(src.name)}:${esc(this.bucket)}/${esc(target)}</code> ${useFolder}</p>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
                ${dests.length ? `<label style="display: inline-flex; gap: 6px; align-items: center;">To
                    <select id="storage-dest" style="width: auto; max-width: 320px;">${options}</select></label>
                    <button type="button" class="action-button" style="max-width: 200px;" data-action="preview" data-move="0">🔍 Preview sync</button>
                    <button type="button" class="action-button" style="max-width: 200px;" data-action="preview" data-move="1">🚚 Preview move</button>`
                  : '<span style="color: var(--text-muted);">No other bucket to sync to.</span>'}
                ${del}
            </div>
            <p id="storage-guard-note" class="storage-hint" style="margin: 6px 0 0 0;"></p>
            <label style="display: inline-flex; gap: 6px; align-items: center; margin-top: 8px; font-size: 0.9em;">
                <input type="checkbox" id="storage-verify" style="width: auto;"> Verify by hashing (slower; always on for Move)
            </label>
            <div id="storage-delete-confirm" style="margin-top: 12px;"></div>
            <div id="storage-plan" style="margin-top: 12px;"></div>
            <div id="storage-progress" style="margin-top: 12px;"></div>`;
        this._updateGuardNote();
    }

    _updateGuardNote() {
        const note = document.getElementById('storage-guard-note');
        if (!note) return;
        const d = this._destination();
        note.textContent = d && isGuarded(this._tier(this.tier), d.tier)
            ? '🛡️ Private → shared: the preview lists anything the privacy guard refuses; refused objects are never copied.'
            : '';
    }

    _verify() {
        return !!document.getElementById('storage-verify')?.checked;
    }

    async _onClick(e) {
        const btn = e.target.closest('[data-action]');
        if (!btn) return undefined;
        const d = btn.dataset;
        switch (d.action) {
            case 'open-bucket': {
                const t = this._tier(d.tier);
                return this.browse(d.tier, d.bucket, d.bucket === t?.bucket ? (t?.hint_prefix || '') : '');
            }
            case 'bucket-size': return this.sizeBucket(d.tier, d.bucket);
            case 'bucket-info': return this.showBucketInfo(d.tier, d.bucket);
            case 'bucket-delete': return this.openBucketDelete(d.tier, d.bucket);
            case 'bucket-delete-go': return this.deleteBucket();
            case 'bucket-create': return this.createBucket(d.tier);
            case 'panel-close': this.bucketDelete = null; return this._setBucketPanel('');
            case 'open':
            case 'crumb': return this.browse(this.tier, this.bucket, d.prefix || '');
            case 'back': return this.back();
            case 'up': return this.up();
            case 'object': return this.openObject(d.key);
            case 'object-close': return this._setObject('');
            case 'select': this.selected = d.prefix || ''; this.plan = null; return this._renderActions();
            case 'order':
                this.view.order = this.view.order === 'asc' ? 'desc' : 'asc';
                this.view.offset = 0;
                return this._reload();
            case 'page':
                this.view.offset = Math.max(0, this.view.offset + Number(d.dir) * PAGE);
                return this._reload();
            case 'preview': {
                const dest = this._destination();
                return dest ? this.preview(dest.tier.name, dest.bucket, d.move === '1') : undefined;
            }
            case 'apply': return this.apply();
            case 'cancel-sync': return this.cancelSync(d.id);
            case 'delete-open': return this._openDelete();
            case 'delete-cancel': {
                const el = document.getElementById('storage-delete-confirm');
                if (el) el.innerHTML = '';
                return undefined;
            }
            case 'delete-go': return this.deleteSelected();
            default: return undefined;
        }
    }

    _onInput(e) {
        const id = e.target?.id || '';
        if (id === 'storage-delete-input') {
            const go = document.getElementById('storage-delete-go');
            if (go) go.disabled = e.target.value !== this.selected;
        } else if (id === 'storage-bucket-confirm') {
            const go = document.getElementById('storage-bucket-delete-go');
            if (go) go.disabled = !this._bucketDeleteReady();
        } else if (id === 'storage-search') {
            clearTimeout(this.searchTimer);
            const value = e.target.value;
            this.searchTimer = setTimeout(() => {
                this.view.search = value;
                this.view.offset = 0;
                this._reload().then(() => document.getElementById('storage-search')?.focus?.());
            }, 300);
        } else if (e.target?.classList?.contains('storage-new-bucket-input')) {
            const tier = e.target.dataset.tier;
            const msg = document.getElementById(`storage-new-bucket-msg-${tier}`);
            const v = e.target.value.trim();
            const problem = v ? validateBucketName(v) : null;
            if (msg) msg.innerHTML = problem ? `<span style="color: var(--danger-dark);">${esc(problem)}</span>` : '';
        }
    }

    _onChange(e) {
        const id = e.target?.id || '';
        if (id === 'storage-sort') {
            this.view.sort = e.target.value;
            this.view.offset = 0;
            this._reload();
        } else if (id === 'storage-dest') {
            this.plan = null;
            const planEl = document.getElementById('storage-plan');
            if (planEl) planEl.innerHTML = '';
            this._updateGuardNote();
        } else if (id === 'storage-bucket-empty') {
            const go = document.getElementById('storage-bucket-delete-go');
            if (go) go.disabled = !this._bucketDeleteReady();
        }
    }

    /** Dry run: fetch and render the plan, then offer Apply. */
    async preview(dst, dstBucket, move) {
        const planEl = document.getElementById('storage-plan');
        const request = {
            src: this.tier, src_bucket: this.bucket, dst, dst_bucket: dstBucket,
            prefix: this.selected, move, verify: move || this._verify(), overwrite: false, dry_run: true,
        };
        if (planEl) planEl.innerHTML = '<p style="color: var(--text-muted);">Planning…</p>';
        try {
            const { plan } = await MerlinaAPI.storageSync(request);
            this.plan = { request, plan };
            this._renderPlan();
        } catch (err) {
            this.plan = null;
            if (planEl) planEl.innerHTML = `<p style="color: var(--danger-dark);">Preview failed: ${esc(err.message)}</p>`;
        }
    }

    _renderPlan() {
        const el = document.getElementById('storage-plan');
        if (!el || !this.plan) return;
        const { request, plan } = this.plan;
        const c = plan.counts;
        const b = plan.bytes;
        const verb = request.move ? 'Move' : 'Copy';
        const refused = plan.refused.length
            ? `<div class="storage-warn-box">
                    <strong>🛡️ ${c.refused} private object(s) refused</strong> — they stay in ${esc(request.src)} and are never copied:
                    <ul style="max-height: 160px; overflow: auto; font-size: 0.85em;">${plan.refused.map(r =>
                        `<li><code>${esc(r.key)}</code> — ${esc(r.reason)}</li>`).join('')}</ul>
               </div>`
            : (plan.guarded ? '<p style="font-size: 0.9em; color: var(--success);">🛡️ Privacy guard checked: nothing refused.</p>' : '');
        const list = plan.to_copy.slice(0, 50).map(i =>
            `<li><code>${esc(i.key)}</code> — ${esc(humanBytes(i.size))} (${esc(i.reason)})</li>`).join('');
        const nothing = c.to_copy === 0 && !(request.move && c.skipped);
        el.innerHTML = `
            <div class="spell-section" style="margin: 0;">
                <p style="font-weight: 600; margin-top: 0;">${verb} <code>${esc(request.src)}:${esc(request.src_bucket)}/${esc(request.prefix)}</code>
                    → <code>${esc(request.dst)}:${esc(request.dst_bucket)}</code></p>
                <div class="metrics-grid" style="margin-bottom: 8px;">
                    <div class="metric-card"><span class="metric-label">To copy</span><span class="metric-value">${c.to_copy} · ${esc(b.to_copy_human)}</span></div>
                    <div class="metric-card"><span class="metric-label">Already there</span><span class="metric-value">${c.skipped} · ${esc(b.skipped_human)}</span></div>
                    <div class="metric-card"><span class="metric-label">Refused</span><span class="metric-value">${c.refused}</span></div>
                </div>
                ${refused}
                ${list ? `<ul style="max-height: 200px; overflow: auto; font-size: 0.85em;">${list}</ul>` : ''}
                ${c.to_copy > 50 ? `<p style="font-size: 0.85em; color: var(--text-muted);">…and ${c.to_copy - 50} more.</p>` : ''}
                ${request.move ? '<p style="font-size: 0.9em;">Source objects are deleted only after every copy is hash-verified.</p>' : ''}
                ${nothing
                    ? '<p style="color: var(--success);">Nothing to do — already in sync. ✨</p>'
                    : `<button type="button" class="action-button${request.move ? ' danger' : ''}" style="max-width: 240px;" data-action="apply">✨ Apply ${verb.toLowerCase()}</button>`}
            </div>`;
    }

    /** Start the previewed sync as a background job and poll it. */
    async apply() {
        if (!this.plan) {
            this.toast.show('Preview first.', 'warning');
            return;
        }
        const request = { ...this.plan.request, dry_run: false };
        this.plan = null;
        const planEl = document.getElementById('storage-plan');
        if (planEl) planEl.innerHTML = '';
        try {
            const { id } = await MerlinaAPI.storageSync(request);
            this._poll(id);
        } catch (err) {
            this.toast.show(`Sync failed to start: ${err.message}`, 'error');
        }
    }

    _poll(id) {
        clearInterval(this.pollTimer);
        const tick = async () => {
            try {
                const job = await MerlinaAPI.storageSyncStatus(id);
                this._renderProgress(job);
                if (job.state !== 'running') {
                    clearInterval(this.pollTimer);
                    this.pollTimer = null;
                    const ok = job.state === 'completed';
                    this.toast.show(ok ? 'Sync complete ✨' : `Sync ${job.state}`, ok ? 'success' : 'error');
                    this.load(true);
                }
            } catch (err) {
                clearInterval(this.pollTimer);
                this.pollTimer = null;
                this.toast.show(`Lost track of sync: ${err.message}`, 'error');
            }
        };
        this.pollTimer = setInterval(tick, POLL_MS);
        tick();
    }

    _renderProgress(job) {
        const el = document.getElementById('storage-progress');
        if (!el) return;
        const p = job.progress || {};
        const total = p.bytes_total || 0;
        const pct = total ? Math.min(100, Math.round((p.bytes_done / total) * 100)) : (job.state === 'running' ? 0 : 100);
        const r = job.result;
        const summary = r
            ? `<p style="font-size: 0.9em;">Copied ${r.counts.copied} (${esc(r.bytes.copied_human)}), skipped ${r.counts.skipped},
                refused ${r.counts.refused}${r.counts.deleted ? `, deleted ${r.counts.deleted} from source` : ''}.
                ${r.move_aborted ? `<br><strong>Move aborted:</strong> ${esc(r.move_aborted)}` : ''}</p>
               ${r.errors.length ? `<ul style="color: var(--danger-dark); font-size: 0.85em;">${r.errors.map(e => `<li><code>${esc(e.key)}</code>: ${esc(e.error)}</li>`).join('')}</ul>` : ''}`
            : '';
        el.innerHTML = `
            <div class="progress-bar" style="margin-bottom: 6px;">
                <div class="progress-fill" style="width: ${pct}%;"></div>
                <span class="progress-text">${esc(job.state)} · ${p.objects_done || 0}/${p.objects_total || 0} objects · ${esc(humanBytes(p.bytes_done || 0))} / ${esc(humanBytes(total))}</span>
            </div>
            ${p.current_key && job.state === 'running' ? `<p style="font-size: 0.85em; color: var(--text-muted);">${esc(p.phase || '')}: <code>${esc(p.current_key)}</code></p>` : ''}
            ${job.state === 'running' ? `<button type="button" class="action-button" style="max-width: 160px;" data-action="cancel-sync" data-id="${esc(job.id)}">Cancel</button>` : ''}
            ${job.error ? `<p style="color: var(--danger-dark);">${esc(job.error)}</p>` : ''}
            ${summary}`;
    }

    async cancelSync(id) {
        try {
            await MerlinaAPI.storageSyncCancel(id);
            this.toast.show('Cancelling after the current object…', 'info');
        } catch (err) {
            this.toast.show(`Cancel failed: ${err.message}`, 'error');
        }
    }

    /** Inline confirmation: preview the deletion, require typing the prefix. */
    async _openDelete() {
        const el = document.getElementById('storage-delete-confirm');
        if (!el || !this.selected) return;
        el.innerHTML = '<p style="color: var(--text-muted);">Counting…</p>';
        try {
            const preview = await MerlinaAPI.storageDelete({
                tier: this.tier, bucket: this.bucket, prefix: this.selected, dry_run: true,
            });
            el.innerHTML = `
                <div class="storage-danger-box">
                    <p style="margin-top: 0;">Permanently delete <strong>${preview.objects} object(s), ${esc(preview.human)}</strong>
                        under <code>${esc(this.tier)}:${esc(this.bucket)}/${esc(preview.prefix)}</code>? This cannot be undone.</p>
                    <label for="storage-delete-input" style="font-size: 0.9em;">Type <code>${esc(this.selected)}</code> to confirm</label>
                    <input type="text" id="storage-delete-input" autocomplete="off" spellcheck="false" style="max-width: 420px;">
                    <div style="display: flex; gap: 8px; margin-top: 8px;">
                        <button type="button" id="storage-delete-go" class="action-button danger" style="max-width: 200px;" data-action="delete-go" disabled>🗑️ Delete</button>
                        <button type="button" class="action-button" style="max-width: 140px;" data-action="delete-cancel">Cancel</button>
                    </div>
                </div>`;
        } catch (err) {
            el.innerHTML = `<p style="color: var(--danger-dark);">${esc(err.message)}</p>`;
        }
    }

    async deleteSelected() {
        const input = document.getElementById('storage-delete-input');
        if (!input || input.value !== this.selected) return;
        try {
            const r = await MerlinaAPI.storageDelete({
                tier: this.tier, bucket: this.bucket, prefix: this.selected, confirm: true,
            });
            this.toast.show(`Deleted ${r.objects} object(s), ${r.human}`, 'success');
            // Re-open the deleted prefix's parent folder (and refresh sizes).
            const parent = breadcrumbs(this.selected.replace(/[^/]+\/?$/, ''));
            this.prefix = parent[parent.length - 1].prefix;
            await this.load(true);
        } catch (err) {
            this.toast.show(`Delete failed: ${err.message}`, 'error');
        }
    }
}

/** HTML for a /storage/preview response (everything escaped). */
export function renderPreview(p) {
    if (!p) return '';
    if (p.kind === 'text') {
        const note = p.truncated
            ? `<p class="storage-hint">Showing the first ${esc(humanBytes(p.bytes_shown))} — truncated. Download for the full file.</p>`
            : '';
        return `${note}<pre class="storage-pre">${esc(p.text)}</pre>`;
    }
    if (p.kind === 'safetensors') {
        const dtypes = Object.entries(p.dtypes || {}).map(([k, v]) => `${esc(k)}: ${esc(humanCount(v))}`).join(' · ');
        const rows = (p.tensors || []).slice(0, MAX_TENSOR_ROWS).map(t =>
            `<tr><td><code>${esc(t.name)}</code></td><td>${esc(t.dtype)}</td><td>[${esc((t.shape || []).join(', '))}]</td><td style="text-align: right;">${esc(humanCount(t.params))}</td></tr>`).join('');
        const more = p.tensor_count > MAX_TENSOR_ROWS
            ? `<p class="storage-hint">Showing ${MAX_TENSOR_ROWS} of ${esc(p.tensor_count)} tensors.</p>` : '';
        const meta = p.metadata ? `<pre class="storage-pre" style="max-height: 120px;">${esc(JSON.stringify(p.metadata, null, 2))}</pre>` : '';
        return `<p><strong>${esc(p.tensor_count)}</strong> tensors · <strong>${esc(humanCount(p.total_params))}</strong> parameters
                (${esc(p.total_params)}) · ${dtypes} · header ${esc(humanBytes(p.header_bytes))}</p>
            ${meta}
            <div class="storage-scroll"><table class="storage-table"><thead><tr><th>Tensor</th><th>dtype</th><th>shape</th><th style="text-align: right;">params</th></tr></thead>
            <tbody>${rows}</tbody></table></div>${more}`;
    }
    if (p.kind === 'parquet') {
        if (!p.schema) return `<p class="storage-hint">${esc(p.message || 'No preview available.')}</p>`;
        const cols = p.schema.map(c => c.name);
        const schema = p.schema.map(c => `<li><code>${esc(c.name)}</code>: ${esc(c.type)}</li>`).join('');
        const head = cols.map(c => `<th>${esc(c)}</th>`).join('');
        const rows = (p.rows || []).map(r => `<tr>${cols.map(c => {
            const v = r[c];
            return `<td>${esc(typeof v === 'string' ? v : JSON.stringify(v))}</td>`;
        }).join('')}</tr>`).join('');
        return `<p><strong>${esc(p.num_rows)}</strong> rows · ${esc(p.num_columns)} columns · ${esc(p.num_row_groups)} row group(s)</p>
            <details><summary>Schema</summary><ul style="font-size: 0.85em;">${schema}</ul></details>
            <p style="font-size: 0.85em; color: var(--text-muted);">First ${(p.rows || []).length} row(s):</p>
            <div class="storage-scroll"><table class="storage-table"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
    }
    return `<p class="storage-hint">${esc(p.message || 'No inline preview for this file type; download it instead.')}</p>`;
}
