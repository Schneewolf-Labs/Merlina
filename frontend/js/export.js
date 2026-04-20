// Export Section Manager
//
// Drives the "Export" tab: GGUF quantization, HF Hub upload, and the
// artifacts inventory for a selected local model. Shares the
// /inference/models response with InferenceManager so GGUF artifacts
// appear in both places without a second endpoint.

import { MerlinaAPI, WebSocketManager } from './api.js';
import { Toast } from './ui.js';

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'];

function formatBytes(n) {
    if (!Number.isFinite(n) || n <= 0) return '0 B';
    let i = 0;
    let val = n;
    while (val >= 1024 && i < BYTE_UNITS.length - 1) {
        val /= 1024;
        i += 1;
    }
    return `${val.toFixed(val >= 100 || i === 0 ? 0 : 1)} ${BYTE_UNITS[i]}`;
}

function formatRelative(epochSeconds) {
    if (!epochSeconds) return '';
    const diff = Date.now() / 1000 - epochSeconds;
    if (diff < 60) return 'just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
}

export class ExportManager {
    constructor() {
        this.models = [];
        this.selectedName = null;
        this.wsManager = null;
        this.activeJobId = null;
        this.toast = new Toast();
        this.llamaCppAvailable = null;
    }

    init() {
        const section = document.getElementById('export-section');
        if (!section) return;

        this._bindTabs();
        this._bindModelSelect();
        this._bindGgufStart();
        this._bindHubStart();

        // Refresh when the Export tab is opened.
        const navBtn = document.querySelector('[data-section="export-section"]');
        if (navBtn) {
            navBtn.addEventListener('click', () => this.refresh());
        }
    }

    async refresh() {
        await Promise.all([
            this._loadModels(),
            this._probeLlamaCpp(),
        ]);
    }

    // --------------------------- Tab switching ---------------------------

    _bindTabs() {
        document.querySelectorAll('.export-tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                document.querySelectorAll('.export-tab-btn').forEach(b => {
                    b.classList.toggle('active', b === btn);
                    b.setAttribute('aria-selected', b === btn ? 'true' : 'false');
                });
                document.querySelectorAll('.export-panel').forEach(p => {
                    p.style.display = p.id === `export-panel-${tab}` ? '' : 'none';
                });
                // Refresh panel contents on switch — artifacts and hub
                // history both depend on the selected model.
                if (tab === 'artifacts') this._renderArtifacts();
                if (tab === 'hub') this._renderHubHistory();
            });
        });
    }

    // --------------------------- Model select ---------------------------

    _bindModelSelect() {
        const select = document.getElementById('export-model-select');
        if (!select) return;
        select.addEventListener('change', () => {
            this.selectedName = select.value || null;
            this._renderModelInfo();
            this._renderHubHistory();
            this._renderArtifacts();
            this._updateHubRepoIdDefault();
        });
    }

    async _loadModels() {
        try {
            const data = await MerlinaAPI.listInferenceModels();
            this.models = data.models || [];
        } catch (err) {
            console.error('Failed to list models:', err);
            this.models = [];
        }

        const select = document.getElementById('export-model-select');
        if (!select) return;
        select.innerHTML = '';

        if (this.models.length === 0) {
            select.innerHTML = '<option value="">No trained models yet — train something first!</option>';
            this.selectedName = null;
            this._renderModelInfo();
            return;
        }

        select.appendChild(new Option('Choose a model…', ''));
        for (const model of this.models) {
            const suffix = model.is_lora ? ` (LoRA on ${model.base_model})` : '';
            const gguf = (model.gguf || []).length;
            const ggufHint = gguf > 0 ? `  [${gguf} GGUF]` : '';
            select.appendChild(new Option(`${model.name}${suffix}${ggufHint}`, model.name));
        }

        // Restore prior selection if still present.
        if (this.selectedName && this.models.find(m => m.name === this.selectedName)) {
            select.value = this.selectedName;
        } else {
            select.value = '';
            this.selectedName = null;
        }
        this._renderModelInfo();
        this._updateHubRepoIdDefault();
    }

    _currentModel() {
        if (!this.selectedName) return null;
        return this.models.find(m => m.name === this.selectedName) || null;
    }

    _renderModelInfo() {
        const el = document.getElementById('export-model-info');
        if (!el) return;
        const model = this._currentModel();
        if (!model) {
            el.textContent = '';
            return;
        }
        const parts = [];
        if (model.is_lora) parts.push(`LoRA on ${model.base_model}`);
        else parts.push('full model');
        if ((model.gguf || []).length) parts.push(`${model.gguf.length} GGUF file(s)`);
        el.textContent = parts.join(' · ');
    }

    _updateHubRepoIdDefault() {
        const input = document.getElementById('export-hub-repo-id');
        if (!input) return;
        if (!input.dataset.userEdited && this.selectedName) {
            input.value = this.selectedName;
        }
        input.addEventListener('input', () => { input.dataset.userEdited = 'true'; }, { once: true });
    }

    // --------------------------- llama.cpp probe ---------------------------

    async _probeLlamaCpp() {
        const availEl = document.getElementById('export-gguf-availability');
        const startBtn = document.getElementById('export-gguf-start');
        try {
            const resp = await fetch('/llama-cpp/status');
            if (!resp.ok) throw new Error(`status ${resp.status}`);
            const data = await resp.json();
            this.llamaCppAvailable = !!data.available;
            if (data.available) {
                if (availEl) {
                    availEl.innerHTML = `<span style="color: #10b981;">✓ llama.cpp ready</span> <span style="color:#888;">(${data.source || 'resolver'})</span>`;
                }
                if (startBtn) startBtn.disabled = false;
            } else {
                if (availEl) {
                    availEl.innerHTML = `<span style="color: #ef4444;">⚠️ llama.cpp not detected.</span> <span style="color:#888;">${(data.warnings || []).join(' ')}</span>`;
                }
                if (startBtn) startBtn.disabled = true;
            }
        } catch (err) {
            if (availEl) availEl.textContent = 'Could not probe llama.cpp availability.';
            this.llamaCppAvailable = false;
        }
    }

    // --------------------------- GGUF export ---------------------------

    _bindGgufStart() {
        const btn = document.getElementById('export-gguf-start');
        if (!btn) return;
        btn.addEventListener('click', () => this._startGgufExport());
    }

    async _startGgufExport() {
        if (!this.selectedName) {
            this.toast.error('Pick a model first');
            return;
        }
        if (this.llamaCppAvailable === false) {
            this.toast.error('llama.cpp is not available on the server');
            return;
        }
        const quants = Array.from(document.querySelectorAll('.export-gguf-quant:checked')).map(cb => cb.value);
        if (quants.length === 0) {
            this.toast.error('Pick at least one quant type');
            return;
        }

        const statusEl = document.getElementById('export-gguf-status');
        const btn = document.getElementById('export-gguf-start');
        btn.disabled = true;
        if (statusEl) statusEl.textContent = 'Kicking off…';

        try {
            const resp = await fetch(`/models/${encodeURIComponent(this.selectedName)}/export-gguf`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    quant_types: quants,
                    keep_fp16: document.getElementById('export-keep-fp16').checked,
                }),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(err.detail || resp.statusText);
            }
            const data = await resp.json();
            this.toast.success(`GGUF export started (${data.job_id})`);
            if (statusEl) statusEl.textContent = `Job ${data.job_id} running…`;
            this._subscribeToJob(data.job_id, () => {
                btn.disabled = false;
                this._loadModels();   // refresh manifest-driven entries
                this._renderArtifacts();
            });
        } catch (err) {
            this.toast.error(`Could not start GGUF export: ${err.message}`);
            if (statusEl) statusEl.textContent = `Failed: ${err.message}`;
            btn.disabled = false;
        }
    }

    _subscribeToJob(jobId, onDone) {
        if (this.wsManager) {
            try { this.wsManager.disconnect(); } catch {}
        }
        this.activeJobId = jobId;
        this.wsManager = new WebSocketManager();
        this.wsManager.connect(jobId, {
            onGgufProgress: (data) => this._renderGgufPill(data),
            onStatus: () => {},
            onMetrics: () => {},
            onCompleted: () => {
                onDone && onDone();
            },
            onError: (msg) => {
                this.toast.error(`Job ${jobId}: ${msg}`);
                onDone && onDone();
            },
        });
    }

    _renderGgufPill(data) {
        const pill = document.getElementById('export-gguf-progress-pill');
        if (!pill) return;
        pill.style.display = 'inline-flex';
        const { stage, quant_type, current, total, error, message } = data;
        if (stage === 'merging') {
            pill.className = 'status-badge status-training';
            pill.textContent = '🔀 merging for GGUF';
        } else if (stage === 'converting') {
            pill.className = 'status-badge status-training';
            pill.textContent = '🔮 converting → fp16 GGUF';
        } else if (stage === 'quantizing') {
            pill.className = 'status-badge status-training';
            const progress = (current && total) ? ` (${current}/${total})` : '';
            pill.textContent = `⚗️ quantizing ${quant_type || ''}${progress}`;
        } else if (stage === 'complete') {
            pill.className = 'status-badge status-completed';
            pill.textContent = '✨ GGUF export complete';
            setTimeout(() => { pill.style.display = 'none'; }, 6000);
        } else if (stage === 'error') {
            pill.className = 'status-badge status-failed';
            pill.textContent = `⚠️ ${error || message || 'export failed'}`;
        }
    }

    // --------------------------- HF Hub upload ---------------------------

    _bindHubStart() {
        const btn = document.getElementById('export-hub-start');
        if (!btn) return;
        btn.addEventListener('click', () => this._startHubUpload());
    }

    async _startHubUpload() {
        if (!this.selectedName) {
            this.toast.error('Pick a model first');
            return;
        }

        const payload = {
            repo_id: document.getElementById('export-hub-repo-id').value.trim() || null,
            private: document.getElementById('export-hub-private').checked,
            commit_message: document.getElementById('export-hub-commit-message').value.trim() || null,
            include_adapter: document.getElementById('export-hub-incl-adapter').checked,
            include_merged: document.getElementById('export-hub-incl-merged').checked,
            include_gguf: document.getElementById('export-hub-incl-gguf').checked,
            include_readme: document.getElementById('export-hub-incl-readme').checked,
            hf_token: document.getElementById('export-hub-token').value || null,
            license: document.getElementById('export-hub-license').value.trim() || null,
            description: document.getElementById('export-hub-description').value.trim() || null,
        };
        const tagsStr = document.getElementById('export-hub-tags').value.trim();
        if (tagsStr) payload.tags = tagsStr.split(',').map(t => t.trim()).filter(Boolean);

        const statusEl = document.getElementById('export-hub-status');
        const btn = document.getElementById('export-hub-start');
        btn.disabled = true;
        if (statusEl) statusEl.textContent = 'Uploading…';

        try {
            const resp = await fetch(`/models/${encodeURIComponent(this.selectedName)}/upload`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(err.detail || resp.statusText);
            }
            const data = await resp.json();
            this.toast.success(`Upload started to ${data.repo_id}`);
            if (statusEl) statusEl.textContent = `Uploading as ${data.repo_id}…`;
            this._subscribeToJob(data.job_id, () => {
                btn.disabled = false;
                this._renderHubHistory();
            });
        } catch (err) {
            this.toast.error(`Upload failed to start: ${err.message}`);
            if (statusEl) statusEl.textContent = `Failed: ${err.message}`;
            btn.disabled = false;
        }
    }

    async _renderHubHistory() {
        const banner = document.getElementById('export-hub-status-banner');
        const historyEl = document.getElementById('export-hub-history');
        if (!banner || !historyEl) return;
        if (!this.selectedName) {
            banner.style.display = 'none';
            historyEl.innerHTML = '';
            return;
        }

        try {
            const resp = await fetch(`/models/${encodeURIComponent(this.selectedName)}/upload-state`);
            if (!resp.ok) throw new Error(`status ${resp.status}`);
            const state = await resp.json();

            const last = state.last_upload;
            if (last) {
                const url = last.repo_url
                    ? `<a href="${last.repo_url}" target="_blank" style="color: #f97316;">${last.repo_id}</a>`
                    : `<code>${last.repo_id}</code>`;
                const staleness = state.local_is_newer
                    ? '<span style="color: #b45309;">⚠ local files modified since last upload</span>'
                    : '<span style="color: #10b981;">✓ in sync with last upload</span>';
                banner.innerHTML = `
                    <div style="padding: 10px 14px; background: #fff; border: 1px solid #fed7aa; border-radius: 8px;">
                        <strong>Last uploaded:</strong> ${url} · <span style="color:#888;">${staleness}</span>
                    </div>`;
                banner.style.display = 'block';
            } else {
                banner.innerHTML = '<div style="padding: 10px 14px; background: #fff; border: 1px dashed #ccc; border-radius: 8px; color: #888;">No uploads yet.</div>';
                banner.style.display = 'block';
            }

            if (state.history && state.history.length > 0) {
                const rows = state.history.slice().reverse().map(evt => {
                    const icon = evt.status === 'success' ? '✅' : '❌';
                    const privacy = evt.private ? '🔒' : '🌍';
                    const art = (evt.artifacts || []).join(', ') || '—';
                    return `<tr>
                        <td>${icon}</td>
                        <td><code>${evt.repo_id || '—'}</code> ${privacy}</td>
                        <td style="color:#888;">${evt.timestamp || ''}</td>
                        <td style="color:#888;">${art}</td>
                    </tr>`;
                }).join('');
                historyEl.innerHTML = `
                    <details>
                        <summary style="cursor: pointer; color: #f97316;">History (${state.history.length})</summary>
                        <table style="width: 100%; margin-top: 8px; border-collapse: collapse;">
                            <thead>
                                <tr style="text-align: left; color: #888; font-size: 0.85em;">
                                    <th></th><th>Repo</th><th>When</th><th>Artifacts</th>
                                </tr>
                            </thead>
                            <tbody>${rows}</tbody>
                        </table>
                    </details>`;
            } else {
                historyEl.innerHTML = '';
            }
        } catch (err) {
            console.debug('Could not fetch upload state:', err);
        }
    }

    // --------------------------- Artifacts ---------------------------

    async _renderArtifacts() {
        const listEl = document.getElementById('export-artifacts-list');
        const sumEl = document.getElementById('export-artifacts-summary');
        if (!listEl || !sumEl) return;
        if (!this.selectedName) {
            listEl.innerHTML = '';
            sumEl.textContent = '';
            return;
        }

        try {
            const resp = await fetch(`/models/${encodeURIComponent(this.selectedName)}/artifacts`);
            if (!resp.ok) throw new Error(`status ${resp.status}`);
            const inv = await resp.json();
            sumEl.innerHTML = `<strong>${formatBytes(inv.total_bytes)}</strong> total${inv.is_lora ? ' · LoRA' : ''}${inv.has_merged ? ' · merged weights' : ''}${inv.has_gguf ? ' · GGUF present' : ''}`;

            const order = ['adapter', 'merged', 'gguf', 'tokenizer', 'processor', 'config', 'readme', 'other'];
            const labels = {
                adapter: '🧩 Adapter',
                merged: '🧱 Merged weights',
                gguf: '🪄 GGUF',
                tokenizer: '🔤 Tokenizer',
                processor: '🖼 Processor',
                config: '⚙️ Config',
                readme: '📝 Readme / metadata',
                other: '📦 Other',
            };

            const pieces = [];
            for (const cat of order) {
                const files = inv.categories[cat];
                if (!files || !files.length) continue;
                pieces.push(`<div style="margin-bottom: 14px;">
                    <h5 style="margin: 0 0 6px 0; color: #0369a1;">${labels[cat] || cat}</h5>
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                        <tbody>
                        ${files.map(f => `
                            <tr>
                                <td style="padding: 4px 0;"><code>${f.path}</code>${f.quant_type ? ` <span style="color:#8b5cf6;">${f.quant_type}</span>` : ''}</td>
                                <td style="padding: 4px 0; text-align: right; color: #888; white-space: nowrap;">${formatBytes(f.size_bytes)}</td>
                                <td style="padding: 4px 0 4px 10px; text-align: right;">
                                    ${f.protected
                                        ? '<span title="Protected — cannot delete via API" style="color:#888;">🔒</span>'
                                        : `<button type="button" class="action-button" data-artifact-delete="${encodeURIComponent(f.path)}" style="padding: 2px 8px; font-size: 0.85em; background: #fee2e2; color: #b91c1c;">delete</button>`
                                    }
                                </td>
                            </tr>
                        `).join('')}
                        </tbody>
                    </table>
                </div>`);
            }
            listEl.innerHTML = pieces.join('') || '<em>No artifacts found.</em>';

            listEl.querySelectorAll('[data-artifact-delete]').forEach(btn => {
                btn.addEventListener('click', () => this._deleteArtifact(decodeURIComponent(btn.dataset.artifactDelete)));
            });
        } catch (err) {
            listEl.innerHTML = `<span style="color:#ef4444;">Could not load artifacts: ${err.message}</span>`;
        }
    }

    async _deleteArtifact(relpath) {
        if (!confirm(`Permanently delete ${relpath}?`)) return;
        try {
            const resp = await fetch(
                `/models/${encodeURIComponent(this.selectedName)}/artifacts?path=${encodeURIComponent(relpath)}`,
                { method: 'DELETE' },
            );
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({ detail: resp.statusText }));
                throw new Error(err.detail || resp.statusText);
            }
            this.toast.success(`Deleted ${relpath}`);
            this._renderArtifacts();
            this._loadModels();
        } catch (err) {
            this.toast.error(`Delete failed: ${err.message}`);
        }
    }
}
