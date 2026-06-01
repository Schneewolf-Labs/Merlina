// Cleanup & Analysis Section Manager
//
// Drives the "Cleanup" section (step 7): shows disk usage, a per-job
// checkpoint breakdown, and lets the user preview (dry-run) then apply a
// checkpoint prune. The backend (/disk/*) never touches active jobs, so the
// UI mirrors that — active jobs render as locked.

import { MerlinaAPI } from './api.js';
import { Toast } from './ui.js';

function esc(s) {
    return String(s).replace(/[&<>"]/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
    ));
}

export class DiskManager {
    constructor() {
        this.toast = new Toast();
        this.lastPreview = null;   // result of the most recent dry run
        this.bound = false;
    }

    /** Wire up the controls. Safe to call once; no-op on missing DOM. */
    init() {
        if (this.bound) return;
        const refresh = document.getElementById('cleanup-refresh-btn');
        const preview = document.getElementById('cleanup-preview-btn');
        const apply = document.getElementById('cleanup-apply-btn');
        const keep = document.getElementById('cleanup-keep');
        if (!refresh && !preview) return;  // section not on this page

        refresh?.addEventListener('click', () => this.loadAnalysis());
        preview?.addEventListener('click', () => this.preview());
        apply?.addEventListener('click', () => this.apply());
        // Changing keep/purge invalidates a stale preview.
        keep?.addEventListener('change', () => this._invalidatePreview());
        document.getElementById('cleanup-purge-failed')
            ?.addEventListener('change', () => this._invalidatePreview());

        // Tabs: checkpoints vs HuggingFace cache.
        document.querySelectorAll('.cleanup-tab-btn').forEach(btn => {
            btn.addEventListener('click', () => this._switchTab(btn.dataset.ctab));
        });
        // HuggingFace cache controls.
        document.getElementById('hf-refresh-btn')?.addEventListener('click', () => this.loadHfCache());
        document.getElementById('hf-preselect-btn')?.addEventListener('click', () => this.preselectStale());
        document.getElementById('hf-delete-btn')?.addEventListener('click', () => this.deleteHfSelected());
        // Saved models controls.
        document.getElementById('models-delete-btn')?.addEventListener('click', () => this.deleteModelsSelected());
        // Other artifacts controls.
        document.getElementById('gguf-delete-btn')?.addEventListener('click', () => this.deleteGgufSelected());
        document.getElementById('wandb-clear-btn')?.addEventListener('click', () => this.clearWandb());

        this.bound = true;
    }

    _switchTab(name) {
        const panels = {
            checkpoints: 'cleanup-panel-checkpoints',
            hfcache: 'cleanup-panel-hfcache',
            models: 'cleanup-panel-models',
            artifacts: 'cleanup-panel-artifacts',
        };
        document.querySelectorAll('.cleanup-tab-btn').forEach(btn => {
            const on = btn.dataset.ctab === name;
            btn.classList.toggle('active', on);
            btn.setAttribute('aria-selected', on ? 'true' : 'false');
            btn.style.borderBottomColor = on ? 'var(--color-primary, #7c3aed)' : 'transparent';
        });
        Object.entries(panels).forEach(([tab, id]) => {
            const el = document.getElementById(id);
            if (el) el.style.display = tab === name ? 'block' : 'none';
        });
        // Scan the HF cache on first open (it's an expensive walk — do it lazily).
        if (name === 'hfcache' && !this._hfLoaded) {
            this._hfLoaded = true;
            this.loadHfCache();
        }
        // The models tab is populated by loadAnalysis; ensure it has run.
        if (name === 'models' && !this._modelsRendered) {
            this.loadAnalysis();
        }
        if (name === 'artifacts' && !this._artifactsLoaded) {
            this._artifactsLoaded = true;
            this.loadArtifacts();
        }
    }

    _keep() {
        return Math.max(1, parseInt(document.getElementById('cleanup-keep')?.value, 10) || 1);
    }

    _purgeFailed() {
        return !!document.getElementById('cleanup-purge-failed')?.checked;
    }

    _invalidatePreview() {
        this.lastPreview = null;
        const apply = document.getElementById('cleanup-apply-btn');
        if (apply) apply.disabled = true;
        const result = document.getElementById('cleanup-result');
        if (result) result.innerHTML = '';
    }

    /** Fetch and render the read-only disk analysis. */
    async loadAnalysis() {
        const jobsEl = document.getElementById('cleanup-jobs');
        if (jobsEl) jobsEl.innerHTML = '<p style="color: #888;">Loading…</p>';
        try {
            const data = await MerlinaAPI.getDiskAnalysis(this._keep());
            this._renderFilesystem(data.filesystem);
            this._renderStats(data.results, data.models);
            this._renderJobs(data.results.jobs || []);
            this._renderModels(data.models || {});
        } catch (err) {
            if (jobsEl) jobsEl.innerHTML = `<p style="color: #dc2626;">Failed to load: ${esc(err.message)}</p>`;
            this.toast.show(`Disk analysis failed: ${err.message}`, 'error');
        }
    }

    _renderFilesystem(fs) {
        const fill = document.getElementById('cleanup-fs-fill');
        const text = document.getElementById('cleanup-fs-text');
        if (!fs) {
            if (text) text.textContent = 'unavailable';
            return;
        }
        const pct = fs.percent ?? 0;
        if (fill) {
            fill.style.width = `${pct}%`;
            // Redden as the disk fills up.
            fill.style.background = pct >= 90 ? '#dc2626' : pct >= 75 ? '#f59e0b' : '';
        }
        if (text) text.textContent = `${fs.used_human} / ${fs.total_human} (${pct}%) — ${fs.free_human} free`;
    }

    _renderStats(results, models) {
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('cleanup-results-total', results?.total_human ?? '—');
        set('cleanup-reclaimable', results?.reclaimable_human ?? '—');
        set('cleanup-models-total', models?.total_human ?? '—');
    }

    _renderJobs(jobs) {
        const el = document.getElementById('cleanup-jobs');
        if (!el) return;
        if (!jobs.length) {
            el.innerHTML = '<p style="color: #888;">No training runs found in results/.</p>';
            return;
        }
        const rows = jobs.map(j => {
            const cks = j.checkpoints.map(c => esc(c.name)).join(', ') || '—';
            const lock = j.active ? ' 🔒' : '';
            const reclaim = j.reclaimable_bytes > 0
                ? `<span style="color: #16a34a;">${esc(j.reclaimable_human)}</span>`
                : '<span style="color: #888;">—</span>';
            return `<tr${j.active ? ' style="opacity: 0.7;"' : ''}>
                <td style="padding: 6px 10px;">${esc(j.job_id)}${lock}</td>
                <td style="padding: 6px 10px;">${esc(j.status)}</td>
                <td style="padding: 6px 10px; text-align: right;">${esc(j.total_human)}</td>
                <td style="padding: 6px 10px;">${cks}</td>
                <td style="padding: 6px 10px; text-align: right;">${reclaim}</td>
            </tr>`;
        }).join('');
        el.innerHTML = `<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
            <thead><tr style="text-align: left; border-bottom: 2px solid #e5e7eb;">
                <th style="padding: 6px 10px;">Job</th>
                <th style="padding: 6px 10px;">Status</th>
                <th style="padding: 6px 10px; text-align: right;">Size</th>
                <th style="padding: 6px 10px;">Checkpoints</th>
                <th style="padding: 6px 10px; text-align: right;">Reclaimable</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
    }

    /** Dry-run: show exactly what would be deleted and enable Apply. */
    async preview() {
        const btn = document.getElementById('cleanup-preview-btn');
        const result = document.getElementById('cleanup-result');
        if (btn) { btn.disabled = true; btn.textContent = '⏳ Previewing…'; }
        try {
            const summary = await MerlinaAPI.runDiskCleanup({
                keep: this._keep(), purge_failed: this._purgeFailed(), apply: false,
            });
            this.lastPreview = summary;
            this._renderResult(summary, false);
            const apply = document.getElementById('cleanup-apply-btn');
            if (apply) apply.disabled = summary.count === 0;
        } catch (err) {
            if (result) result.innerHTML = `<p style="color: #dc2626;">Preview failed: ${esc(err.message)}</p>`;
            this.toast.show(`Preview failed: ${err.message}`, 'error');
        } finally {
            if (btn) { btn.disabled = false; btn.textContent = '🔍 Preview (dry run)'; }
        }
    }

    /** Apply the previewed cleanup after explicit confirmation. */
    async apply() {
        if (!this.lastPreview || this.lastPreview.count === 0) {
            this.toast.show('Nothing to delete — run a preview first.', 'warning');
            return;
        }
        const { count, freed_human } = this.lastPreview;
        if (!window.confirm(`Permanently delete ${count} checkpoint item(s), freeing ${freed_human}? This cannot be undone.`)) {
            return;
        }
        const btn = document.getElementById('cleanup-apply-btn');
        if (btn) { btn.disabled = true; btn.textContent = '⏳ Deleting…'; }
        try {
            const summary = await MerlinaAPI.runDiskCleanup({
                keep: this._keep(), purge_failed: this._purgeFailed(), apply: true,
            });
            this._renderResult(summary, true);
            this.toast.show(`Freed ${summary.freed_human} across ${summary.count} item(s) ✨`, 'success');
            this.lastPreview = null;
            await this.loadAnalysis();  // refresh sizes
        } catch (err) {
            this.toast.show(`Cleanup failed: ${err.message}`, 'error');
        } finally {
            if (btn) { btn.textContent = '🗑️ Delete previewed items'; btn.disabled = true; }
        }
    }

    _renderResult(summary, applied) {
        const el = document.getElementById('cleanup-result');
        if (!el) return;
        if (summary.count === 0) {
            el.innerHTML = '<p style="color: #16a34a;">Nothing to prune — everything\'s already tidy. ✨</p>';
            return;
        }
        const verb = applied ? 'Deleted' : 'Would delete';
        const items = summary.deletions.map(d => {
            const name = esc(d.path.split('/').slice(-2).join('/'));
            return `<li><code>${name}</code> — ${esc(d.reason)} <strong>(${esc(d.human ?? '')})</strong></li>`;
        }).join('');
        const human = summary.freed_human;
        el.innerHTML = `
            <p style="font-weight: 600;">${verb} ${summary.count} item(s), ${applied ? 'freed' : 'freeing'} ${esc(human)}:</p>
            <ul style="max-height: 260px; overflow: auto; font-size: 0.85em; line-height: 1.6;">${items}</ul>`;
    }

    // ── HuggingFace cache ────────────────────────────────────────────────────

    _staleDays() {
        return Math.max(1, parseInt(document.getElementById('hf-stale-days')?.value, 10) || 90);
    }

    /** Scan the HF cache and render the repo table. */
    async loadHfCache() {
        const table = document.getElementById('hf-cache-table');
        if (table) table.innerHTML = '<p style="color: #888;">Scanning cache…</p>';
        try {
            const data = await MerlinaAPI.getHfCache(this._staleDays());
            this._renderHfCache(data);
        } catch (err) {
            if (table) table.innerHTML = `<p style="color: #dc2626;">Scan failed: ${esc(err.message)}</p>`;
            this.toast.show(`HF cache scan failed: ${err.message}`, 'error');
        }
    }

    _renderHfCache(data) {
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        const table = document.getElementById('hf-cache-table');
        const result = document.getElementById('hf-result');
        if (result) result.innerHTML = '';

        if (!data.available) {
            set('hf-cache-total', 'unavailable');
            set('hf-cache-stale', '—');
            if (table) table.innerHTML = `<p style="color: #888;">No HuggingFace cache found${data.error ? ` (${esc(data.error)})` : ''}.</p>`;
            return;
        }
        set('hf-cache-total', data.total_human);
        set('hf-cache-stale', data.stale_reclaimable_human);
        this._updateHfSelection();

        if (!data.repos.length) {
            if (table) table.innerHTML = '<p style="color: #888;">Cache is empty.</p>';
            return;
        }
        const rows = data.repos.map(r => {
            const stale = r.stale
                ? '<span style="color: #b45309; font-size: 0.8em; font-weight: 600;">STALE</span>'
                : '';
            return `<tr>
                <td style="padding: 6px 10px;">
                    <input type="checkbox" class="hf-repo-check"
                        data-repo-id="${esc(r.repo_id)}" data-repo-type="${esc(r.repo_type)}" data-bytes="${r.bytes}">
                </td>
                <td style="padding: 6px 10px;">${esc(r.repo_id)}</td>
                <td style="padding: 6px 10px;">${esc(r.repo_type)}</td>
                <td style="padding: 6px 10px; text-align: right;">${esc(r.human)}</td>
                <td style="padding: 6px 10px;">${esc(r.last_accessed_date || '—')} ${stale}</td>
            </tr>`;
        }).join('');
        if (table) {
            table.innerHTML = `<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                <thead><tr style="text-align: left; border-bottom: 2px solid #e5e7eb;">
                    <th style="padding: 6px 10px;"></th>
                    <th style="padding: 6px 10px;">Repo</th>
                    <th style="padding: 6px 10px;">Type</th>
                    <th style="padding: 6px 10px; text-align: right;">Size</th>
                    <th style="padding: 6px 10px;">Last used</th>
                </tr></thead>
                <tbody>${rows}</tbody>
            </table>`;
            table.querySelectorAll('.hf-repo-check').forEach(cb =>
                cb.addEventListener('change', () => this._updateHfSelection()));
        }
    }

    _checkedRepos() {
        return Array.from(document.querySelectorAll('.hf-repo-check:checked'));
    }

    _updateHfSelection() {
        const checked = this._checkedRepos();
        const bytes = checked.reduce((sum, cb) => sum + (parseInt(cb.dataset.bytes, 10) || 0), 0);
        const el = document.getElementById('hf-selected-size');
        if (el) el.textContent = this._humanBytes(bytes);
        const btn = document.getElementById('hf-delete-btn');
        if (btn) btn.disabled = checked.length === 0;
    }

    _humanBytes(n) {
        const units = ['B', 'KB', 'MB', 'GB', 'TB'];
        let v = n, i = 0;
        while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
        return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
    }

    preselectStale() {
        document.querySelectorAll('.hf-repo-check').forEach(cb => {
            // Stale rows carry the STALE badge in their last cell.
            const row = cb.closest('tr');
            cb.checked = !!row && /STALE/.test(row.textContent);
        });
        this._updateHfSelection();
    }

    async deleteHfSelected() {
        const checked = this._checkedRepos();
        if (!checked.length) return;
        const repos = checked.map(cb => ({ repo_id: cb.dataset.repoId, repo_type: cb.dataset.repoType }));
        const bytes = checked.reduce((s, cb) => s + (parseInt(cb.dataset.bytes, 10) || 0), 0);
        if (!window.confirm(
            `Delete ${repos.length} cached repo(s) (~${this._humanBytes(bytes)})?\n\n` +
            `This cache is shared with other tools on this machine. Deleted models ` +
            `re-download from the Hub when next needed. Continue?`)) {
            return;
        }
        const btn = document.getElementById('hf-delete-btn');
        if (btn) { btn.disabled = true; btn.textContent = '⏳ Deleting…'; }
        try {
            const res = await MerlinaAPI.deleteHfCache({ repos, apply: true });
            const result = document.getElementById('hf-result');
            if (result) {
                result.innerHTML = `<p style="color: #16a34a; font-weight: 600;">Freed ${esc(res.freed_human)} across ${res.count} repo(s).</p>`;
            }
            this.toast.show(`Freed ${res.freed_human} from HF cache ✨`, 'success');
            await this.loadHfCache();  // rescan
        } catch (err) {
            this.toast.show(`HF cache deletion failed: ${err.message}`, 'error');
        } finally {
            if (btn) { btn.textContent = '🗑️ Delete selected repos'; }
            this._updateHfSelection();
        }
    }

    // ── Saved models ─────────────────────────────────────────────────────────

    _renderModels(models) {
        this._modelsRendered = true;
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('models-total', models.total_human ?? '—');
        const table = document.getElementById('models-table');
        if (!table) return;
        const items = models.items || [];
        if (!items.length) {
            table.innerHTML = '<p style="color: #888;">No saved models in ./models/.</p>';
            this._updateModelsSelection();
            return;
        }
        const rows = items.map(m => {
            const lock = m.protected
                ? '<input type="checkbox" disabled title="In use — locked"> 🔒'
                : `<input type="checkbox" class="model-check" data-name="${esc(m.name)}" data-bytes="${m.bytes}">`;
            return `<tr${m.protected ? ' style="opacity: 0.6;"' : ''}>
                <td style="padding: 6px 10px;">${lock}</td>
                <td style="padding: 6px 10px;">${esc(m.name)}</td>
                <td style="padding: 6px 10px; text-align: right;">${esc(m.human)}</td>
                <td style="padding: 6px 10px;">${esc(m.modified_date || '—')}</td>
            </tr>`;
        }).join('');
        table.innerHTML = `<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
            <thead><tr style="text-align: left; border-bottom: 2px solid #e5e7eb;">
                <th style="padding: 6px 10px;"></th>
                <th style="padding: 6px 10px;">Model</th>
                <th style="padding: 6px 10px; text-align: right;">Size</th>
                <th style="padding: 6px 10px;">Modified</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
        table.querySelectorAll('.model-check').forEach(cb =>
            cb.addEventListener('change', () => this._updateModelsSelection()));
        this._updateModelsSelection();
    }

    _checkedModels() {
        return Array.from(document.querySelectorAll('.model-check:checked'));
    }

    _updateModelsSelection() {
        const checked = this._checkedModels();
        const bytes = checked.reduce((s, cb) => s + (parseInt(cb.dataset.bytes, 10) || 0), 0);
        const el = document.getElementById('models-selected-size');
        if (el) el.textContent = this._humanBytes(bytes);
        const btn = document.getElementById('models-delete-btn');
        if (btn) btn.disabled = checked.length === 0;
    }

    async deleteModelsSelected() {
        const checked = this._checkedModels();
        if (!checked.length) return;
        const names = checked.map(cb => cb.dataset.name);
        const bytes = checked.reduce((s, cb) => s + (parseInt(cb.dataset.bytes, 10) || 0), 0);
        if (!window.confirm(
            `Permanently delete ${names.length} saved model(s) (~${this._humanBytes(bytes)})?\n\n` +
            `${names.join(', ')}\n\n` +
            `Models not pushed to the Hub CANNOT be recovered. Continue?`)) {
            return;
        }
        const btn = document.getElementById('models-delete-btn');
        if (btn) { btn.disabled = true; btn.textContent = '⏳ Deleting…'; }
        try {
            const res = await MerlinaAPI.deleteModels({ names, apply: true });
            const result = document.getElementById('models-result');
            if (result) {
                let html = `<p style="color: #16a34a; font-weight: 600;">Freed ${esc(res.freed_human)} across ${res.count} model(s).</p>`;
                if (res.skipped && res.skipped.length) {
                    const sk = res.skipped.map(s => `<li>${esc(s.name)} — ${esc(s.reason)}</li>`).join('');
                    html += `<p style="color: #b45309;">Skipped:</p><ul style="font-size: 0.85em;">${sk}</ul>`;
                }
                result.innerHTML = html;
            }
            this.toast.show(`Freed ${res.freed_human} from saved models ✨`, 'success');
            await this.loadAnalysis();  // refresh
        } catch (err) {
            this.toast.show(`Model deletion failed: ${err.message}`, 'error');
        } finally {
            if (btn) { btn.textContent = '🗑️ Delete selected models'; }
            this._updateModelsSelection();
        }
    }

    // ── Other artifacts (GGUF exports + W&B logs) ─────────────────────────────

    async loadArtifacts() {
        const gt = document.getElementById('gguf-table');
        if (gt) gt.innerHTML = '<p style="color: #888;">Loading…</p>';
        try {
            const data = await MerlinaAPI.getDiskArtifacts();
            this._renderGguf(data.gguf || {});
            this._renderWandb(data.wandb || {});
        } catch (err) {
            if (gt) gt.innerHTML = `<p style="color: #dc2626;">Failed to load: ${esc(err.message)}</p>`;
            this.toast.show(`Artifact scan failed: ${err.message}`, 'error');
        }
    }

    _renderGguf(gguf) {
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('gguf-total', gguf.total_human ?? '—');
        const table = document.getElementById('gguf-table');
        if (!table) return;
        const files = gguf.files || [];
        if (!files.length) {
            table.innerHTML = '<p style="color: #888;">No GGUF exports found.</p>';
            this._updateGgufSelection();
            return;
        }
        const rows = files.map(f => {
            const cb = f.loaded
                ? '<input type="checkbox" disabled title="Loaded for inference"> 🔒'
                : `<input type="checkbox" class="gguf-check" data-model="${esc(f.model)}" data-file="${esc(f.file)}" data-bytes="${f.bytes}">`;
            return `<tr${f.loaded ? ' style="opacity: 0.6;"' : ''}>
                <td style="padding: 6px 10px;">${cb}</td>
                <td style="padding: 6px 10px;">${esc(f.model)}</td>
                <td style="padding: 6px 10px;">${esc(f.quant || '—')}</td>
                <td style="padding: 6px 10px; text-align: right;">${esc(f.human)}</td>
                <td style="padding: 6px 10px;">${esc(f.modified_date || '—')}</td>
            </tr>`;
        }).join('');
        table.innerHTML = `<table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
            <thead><tr style="text-align: left; border-bottom: 2px solid #e5e7eb;">
                <th style="padding: 6px 10px;"></th>
                <th style="padding: 6px 10px;">Model</th>
                <th style="padding: 6px 10px;">Quant</th>
                <th style="padding: 6px 10px; text-align: right;">Size</th>
                <th style="padding: 6px 10px;">Modified</th>
            </tr></thead>
            <tbody>${rows}</tbody>
        </table>`;
        table.querySelectorAll('.gguf-check').forEach(cb =>
            cb.addEventListener('change', () => this._updateGgufSelection()));
        this._updateGgufSelection();
    }

    _checkedGguf() {
        return Array.from(document.querySelectorAll('.gguf-check:checked'));
    }

    _updateGgufSelection() {
        const checked = this._checkedGguf();
        const bytes = checked.reduce((s, cb) => s + (parseInt(cb.dataset.bytes, 10) || 0), 0);
        const el = document.getElementById('gguf-selected-size');
        if (el) el.textContent = this._humanBytes(bytes);
        const btn = document.getElementById('gguf-delete-btn');
        if (btn) btn.disabled = checked.length === 0;
    }

    async deleteGgufSelected() {
        const checked = this._checkedGguf();
        if (!checked.length) return;
        const files = checked.map(cb => ({ model: cb.dataset.model, file: cb.dataset.file }));
        const bytes = checked.reduce((s, cb) => s + (parseInt(cb.dataset.bytes, 10) || 0), 0);
        if (!window.confirm(
            `Delete ${files.length} GGUF file(s) (~${this._humanBytes(bytes)})?\n\n` +
            `These re-export from the model anytime. Continue?`)) {
            return;
        }
        const btn = document.getElementById('gguf-delete-btn');
        if (btn) { btn.disabled = true; btn.textContent = '⏳ Deleting…'; }
        try {
            const res = await MerlinaAPI.deleteGguf({ files, apply: true });
            const result = document.getElementById('gguf-result');
            if (result) result.innerHTML = `<p style="color: #16a34a; font-weight: 600;">Freed ${esc(res.freed_human)} across ${res.count} file(s).</p>`;
            this.toast.show(`Freed ${res.freed_human} of GGUF ✨`, 'success');
            await this.loadArtifacts();
        } catch (err) {
            this.toast.show(`GGUF deletion failed: ${err.message}`, 'error');
        } finally {
            if (btn) { btn.textContent = '🗑️ Delete selected GGUF'; }
            this._updateGgufSelection();
        }
    }

    _renderWandb(wandb) {
        const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        set('wandb-total', wandb.total_human ?? '—');
        set('wandb-runs', wandb.run_count != null ? String(wandb.run_count) : '—');
        set('wandb-reclaimable', wandb.reclaimable_human ?? '—');
        const btn = document.getElementById('wandb-clear-btn');
        if (btn) btn.disabled = !(wandb.reclaimable_bytes > 0);
    }

    async clearWandb() {
        const btn = document.getElementById('wandb-clear-btn');
        if (!window.confirm('Delete all local W&B run logs except the active run?\n\nMetrics remain on the W&B server.')) {
            return;
        }
        if (btn) { btn.disabled = true; btn.textContent = '⏳ Clearing…'; }
        try {
            const res = await MerlinaAPI.clearWandb({ apply: true });
            const result = document.getElementById('wandb-result');
            if (result) result.innerHTML = `<p style="color: #16a34a; font-weight: 600;">Cleared ${res.count} run(s), freed ${esc(res.freed_human)}.</p>`;
            this.toast.show(`Freed ${res.freed_human} of W&B logs ✨`, 'success');
            await this.loadArtifacts();
        } catch (err) {
            this.toast.show(`W&B clear failed: ${err.message}`, 'error');
        } finally {
            if (btn) { btn.textContent = '🗑️ Clear W&B logs'; }
        }
    }
}
