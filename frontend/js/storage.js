// Object Storage Section Manager
//
// Drives the "Object storage" section: tier cards (bucket, privacy, usage), a
// per-tier prefix browser with breadcrumbs, and sync / move / delete actions on
// the selected prefix. Every sync or move is previewed (dry run) first; Apply
// only appears on a fresh plan, so a private -> shared action is never offered
// without the server's guard result (refused objects) on screen. Deletion uses
// an inline type-the-prefix confirmation instead of window.confirm.

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

/** True when copying from `src` to `dst` crosses from private to shared. */
export function isGuarded(src, dst) {
    return !!(src && dst && src.private && !dst.private);
}

const POLL_MS = 1000;

export class StorageManager {
    constructor() {
        this.toast = new Toast();
        this.tiers = [];
        this.tier = null;        // name of the tier being browsed
        this.prefix = '';        // prefix being browsed
        this.selected = '';      // prefix the actions apply to
        this.plan = null;        // last dry-run plan {request, plan}
        this.pollTimer = null;
        this.bound = false;
        this.loaded = false;
    }

    init() {
        if (this.bound) return;
        const refresh = document.getElementById('storage-refresh-btn');
        if (!refresh) return;  // section not on this page
        refresh.addEventListener('click', () => this.load(true));
        // Event delegation keeps the dynamically-rendered markup simple.
        document.getElementById('storage-section')?.addEventListener('click', e => this._onClick(e));
        document.getElementById('storage-section')?.addEventListener('input', e => this._onInput(e));
        this.bound = true;
    }

    _tier(name) {
        return this.tiers.find(t => t.name === name) || null;
    }

    /** Load tier cards (with usage) and open the first configured tier. */
    async load(force = false) {
        if (this.loaded && !force) return;
        this.loaded = true;
        const cards = document.getElementById('storage-tiers');
        if (cards) cards.innerHTML = '<p style="color: var(--text-muted);">Loading tiers…</p>';
        try {
            const data = await MerlinaAPI.getStorageTiers(true);
            this.tiers = data.tiers || [];
            this._renderTiers();
            const first = this.tiers.find(t => t.configured);
            if (!first) {
                this._setBrowser('<p style="color: var(--text-muted);">No storage tier is configured. See docs/storage-tiers.md.</p>');
                return;
            }
            if (!this.tier || !this._tier(this.tier)?.configured) {
                await this.browse(first.name, first.hint_prefix || '');
            } else {
                await this.browse(this.tier, this.prefix);
            }
        } catch (err) {
            if (cards) cards.innerHTML = `<p style="color: var(--danger-dark);">Failed to load tiers: ${esc(err.message)}</p>`;
        }
    }

    _renderTiers() {
        const el = document.getElementById('storage-tiers');
        if (!el) return;
        el.innerHTML = this.tiers.map(t => {
            const badge = t.private
                ? '<span class="storage-badge private" title="Private data never leaves this tier">🔒 private</span>'
                : '<span class="storage-badge shared">🌐 shared</span>';
            if (!t.configured) {
                return `<div class="metric-card storage-tier-card" style="opacity: 0.6;">
                    <span class="metric-label">${esc(t.name)}</span>
                    <span class="metric-value" style="font-size: 1em;">not configured</span>
                </div>`;
            }
            const usage = t.usage
                ? `${esc(t.usage.human)} · ${t.usage.objects} object(s)`
                : (t.usage_error ? 'usage unavailable' : '—');
            const active = t.name === this.tier ? ' active' : '';
            return `<button type="button" class="metric-card storage-tier-card${active}" data-action="tier"
                        data-tier="${esc(t.name)}" style="text-align: left; cursor: pointer;">
                <span class="metric-label">${esc(t.name)} ${badge}</span>
                <span class="metric-value" style="font-size: 1.05em;">${usage}</span>
                <span style="font-size: 0.8em; color: var(--text-muted);">${esc(t.bucket)} @ ${esc(t.endpoint_host || 'default')}</span>
            </button>`;
        }).join('');
    }

    _setBrowser(html) {
        const el = document.getElementById('storage-browser');
        if (el) el.innerHTML = html;
    }

    /** Browse one level of a tier. */
    async browse(tier, prefix = '') {
        this.tier = tier;
        this.prefix = prefix;
        this.selected = prefix;
        this.plan = null;
        this._renderTiers();
        this._renderCrumbs();
        this._renderActions();
        this._setBrowser('<p style="color: var(--text-muted);">Loading…</p>');
        try {
            const data = await MerlinaAPI.storageLs(tier, prefix);
            this._renderListing(data);
        } catch (err) {
            this._setBrowser(`<p style="color: var(--danger-dark);">Failed to list: ${esc(err.message)}</p>`);
        }
    }

    _renderCrumbs() {
        const el = document.getElementById('storage-crumbs');
        if (!el) return;
        const crumbs = breadcrumbs(this.prefix).map((c, i, all) => {
            const last = i === all.length - 1;
            const label = i === 0 ? `${esc(this.tier)}:/` : esc(c.label);
            return last
                ? `<strong>${label}</strong>`
                : `<button type="button" class="link-button" data-action="crumb" data-prefix="${esc(c.prefix)}">${label}</button>`;
        });
        el.innerHTML = crumbs.join(' <span aria-hidden="true">/</span> ');
    }

    _renderListing(data) {
        const dirs = data.dirs || [];
        const objects = data.objects || [];
        if (!dirs.length && !objects.length) {
            this._setBrowser('<p style="color: var(--text-muted);">Empty.</p>');
            return;
        }
        const dirRows = dirs.map(d => `<tr>
            <td style="padding: 6px 10px;">
                <button type="button" class="link-button" data-action="open" data-prefix="${esc(d.prefix)}">📁 ${esc(d.name)}</button>
            </td>
            <td style="padding: 6px 10px; text-align: right;">—</td>
            <td style="padding: 6px 10px;"></td>
            <td style="padding: 6px 10px; text-align: right;">
                <button type="button" class="link-button" data-action="select" data-prefix="${esc(d.prefix)}">Select</button>
            </td>
        </tr>`).join('');
        const objRows = objects.map(o => `<tr>
            <td style="padding: 6px 10px;">📄 ${esc(o.name)}</td>
            <td style="padding: 6px 10px; text-align: right;">${esc(humanBytes(o.size))}</td>
            <td style="padding: 6px 10px; color: var(--text-muted);">${esc((o.mtime || '').slice(0, 19).replace('T', ' '))}</td>
            <td style="padding: 6px 10px; text-align: right;">
                <button type="button" class="link-button" data-action="select" data-prefix="${esc(o.key)}">Select</button>
            </td>
        </tr>`).join('');
        const more = data.truncated
            ? `<p style="color: var(--text-muted); font-size: 0.85em;">Showing ${objects.length} of ${data.object_count} objects.</p>`
            : '';
        this._setBrowser(`<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
            <thead><tr style="text-align: left; border-bottom: 2px solid var(--border-color);">
                <th style="padding: 6px 10px;">Name</th>
                <th style="padding: 6px 10px; text-align: right;">Size</th>
                <th style="padding: 6px 10px;">Modified</th>
                <th style="padding: 6px 10px;"></th>
            </tr></thead>
            <tbody>${dirRows}${objRows}</tbody>
        </table>${more}`);
    }

    /** Action bar for the selected prefix: sync/move to every other tier + delete. */
    _renderActions() {
        const el = document.getElementById('storage-actions');
        if (!el) return;
        const src = this._tier(this.tier);
        if (!src) { el.innerHTML = ''; return; }
        const target = this.selected || '';
        const others = this.tiers.filter(t => t.configured && t.name !== src.name);
        const buttons = others.map(dst => {
            const guarded = isGuarded(src, dst);
            const lock = guarded ? ' 🛡️' : '';
            const title = guarded
                ? 'Private -> shared: the preview lists anything the privacy guard refuses'
                : '';
            return `<button type="button" class="action-button" style="max-width: 230px;" title="${esc(title)}"
                        data-action="preview" data-dst="${esc(dst.name)}" data-move="0">🔍 Sync to ${esc(dst.name)}${lock}</button>
                    <button type="button" class="action-button" style="max-width: 230px;" title="${esc(title)}"
                        data-action="preview" data-dst="${esc(dst.name)}" data-move="1">🚚 Move to ${esc(dst.name)}${lock}</button>`;
        }).join('');
        const del = target
            ? `<button type="button" class="action-button danger" style="max-width: 200px;" data-action="delete-open">🗑️ Delete…</button>`
            : '';
        el.innerHTML = `
            <p style="margin: 0 0 8px 0;">Selected: <code>${esc(src.name)}:${esc(target || '/')}</code>
                ${target !== this.prefix ? '<button type="button" class="link-button" data-action="select" data-prefix="' + esc(this.prefix) + '">(use current folder)</button>' : ''}</p>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">${buttons || '<span style="color: var(--text-muted);">No other tier configured.</span>'} ${del}</div>
            <label style="display: inline-flex; gap: 6px; align-items: center; margin-top: 8px; font-size: 0.9em;">
                <input type="checkbox" id="storage-verify" style="width: auto;"> Verify by hashing (slower; always on for Move)
            </label>
            <div id="storage-delete-confirm" style="margin-top: 12px;"></div>
            <div id="storage-plan" style="margin-top: 12px;"></div>
            <div id="storage-progress" style="margin-top: 12px;"></div>`;
    }

    _verify() {
        return !!document.getElementById('storage-verify')?.checked;
    }

    async _onClick(e) {
        const btn = e.target.closest('[data-action]');
        if (!btn) return;
        const action = btn.dataset.action;
        if (action === 'tier') return this.browse(btn.dataset.tier, this._tier(btn.dataset.tier)?.hint_prefix || '');
        if (action === 'open' || action === 'crumb') return this.browse(this.tier, btn.dataset.prefix || '');
        if (action === 'select') { this.selected = btn.dataset.prefix || ''; this.plan = null; return this._renderActions(); }
        if (action === 'preview') return this.preview(btn.dataset.dst, btn.dataset.move === '1');
        if (action === 'apply') return this.apply();
        if (action === 'cancel-sync') return this.cancelSync(btn.dataset.id);
        if (action === 'delete-open') return this._openDelete();
        if (action === 'delete-cancel') { document.getElementById('storage-delete-confirm').innerHTML = ''; return; }
        if (action === 'delete-go') return this.deleteSelected();
        return undefined;
    }

    _onInput(e) {
        if (e.target?.id !== 'storage-delete-input') return;
        const go = document.getElementById('storage-delete-go');
        if (go) go.disabled = e.target.value !== this.selected;
    }

    /** Dry run: fetch and render the plan, then offer Apply. */
    async preview(dst, move) {
        const planEl = document.getElementById('storage-plan');
        const request = {
            src: this.tier, dst, prefix: this.selected, move,
            verify: move || this._verify(), overwrite: false, dry_run: true,
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
            ? `<div style="border-left: 3px solid #b45309; padding: 6px 10px; margin: 8px 0;">
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
                <p style="font-weight: 600; margin-top: 0;">${verb} <code>${esc(request.src)}:${esc(request.prefix || '/')}</code> → ${esc(request.dst)}</p>
                <div class="metrics-grid" style="margin-bottom: 8px;">
                    <div class="metric-card"><span class="metric-label">To copy</span><span class="metric-value">${c.to_copy} · ${esc(b.to_copy_human)}</span></div>
                    <div class="metric-card"><span class="metric-label">Already there</span><span class="metric-value">${c.skipped} · ${esc(b.skipped_human)}</span></div>
                    <div class="metric-card"><span class="metric-label">Refused</span><span class="metric-value">${c.refused}</span></div>
                </div>
                ${refused}
                ${list ? `<ul style="max-height: 200px; overflow: auto; font-size: 0.85em;">${list}</ul>` : ''}
                ${c.to_copy > 50 ? `<p style="font-size: 0.85em; color: var(--text-muted);">…and ${c.to_copy - 50} more.</p>` : ''}
                ${request.move ? `<p style="font-size: 0.9em;">Source objects are deleted only after every copy is hash-verified.</p>` : ''}
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
            const preview = await MerlinaAPI.storageDelete({ tier: this.tier, prefix: this.selected, dry_run: true });
            el.innerHTML = `
                <div style="border-left: 3px solid var(--danger-dark, #dc2626); padding: 8px 12px;">
                    <p style="margin-top: 0;">Permanently delete <strong>${preview.objects} object(s), ${esc(preview.human)}</strong>
                        under <code>${esc(this.tier)}:${esc(preview.prefix)}/</code>? This cannot be undone.</p>
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
            const r = await MerlinaAPI.storageDelete({ tier: this.tier, prefix: this.selected, confirm: true });
            this.toast.show(`Deleted ${r.objects} object(s), ${r.human}`, 'success');
            // Re-open the deleted prefix's parent folder (and refresh usage).
            const parent = breadcrumbs(this.selected.replace(/[^/]+\/?$/, ''));
            this.prefix = parent[parent.length - 1].prefix;
            await this.load(true);
        } catch (err) {
            this.toast.show(`Delete failed: ${err.message}`, 'error');
        }
    }
}
