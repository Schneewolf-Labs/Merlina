// Remote GPUs Section Manager
//
// Drives the "Remote GPUs" section: rent a GPU, serve a model with vLLM
// behind an OpenAI-compatible endpoint, try it, and stop it.
//
//   * a session form: what to serve (HF repo, a model pushed to object
//     storage with `merlina models push`, or a finished remote training job)
//     plus the hardware, the caps and the vLLM settings;
//   * Plan (a free dry run) -> Start. Start only appears for a plan that
//     matches the form as it is now, and it takes a second, explicit click
//     that names the GPU and the maximum spend;
//   * a session list with live status, elapsed time and spend so far;
//   * a details panel for one session: endpoint, API key, curl example,
//     worker phase, request counters, errors and the vLLM log tail, plus a
//     small chat box that goes through the control plane.

import { MerlinaAPI } from './api.js';
import { Toast } from './ui.js';
import { esc } from './storage.js';

const ACTIVE = ['queued', 'provisioning', 'starting', 'ready', 'stopping'];
const POLL_ACTIVE_MS = 5000;
const POLL_IDLE_MS = 30000;

const PHASES = {
    preparing_model: 'Downloading the model',
    starting_vllm: 'Starting vLLM',
    serving: 'Serving',
};

/** Split a vLLM args string into argv, honouring "double" and 'single' quotes. */
export function splitArgs(text) {
    const out = [];
    const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
    let m;
    while ((m = re.exec(String(text || ''))) !== null) {
        out.push(m[1] ?? m[2] ?? m[3]);
    }
    return out;
}

function num(value) {
    if (value === '' || value === null || value === undefined) return null;
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

/**
 * Turn the form's raw values into a RemoteInferenceRequest body.
 * Returns {body, errors}; empty optional fields are left out so the server's
 * defaults apply.
 */
export function buildRemoteRequest(v) {
    const errors = [];
    const model = {};
    if (v.source === 'hf') {
        const repo = String(v.hf_repo || '').trim();
        if (!repo) errors.push('Enter a HuggingFace repo, e.g. Qwen/Qwen3-8B.');
        else if (!/^[A-Za-z0-9][A-Za-z0-9._-]*(\/[A-Za-z0-9][A-Za-z0-9._-]*)?$/.test(repo)) {
            errors.push('The HF repo must look like org/name (local paths don\'t reach a rented GPU).');
        }
        model.hf_repo = repo;
        if (String(v.revision || '').trim()) model.revision = String(v.revision).trim();
    } else if (v.source === 'store') {
        if (!v.store_model) errors.push('Pick a model from object storage.');
        model.store_model = v.store_model;
    } else if (v.source === 'job') {
        if (!v.job_id) errors.push('Pick a finished remote training job.');
        model.job_id = v.job_id;
        model.artifact = v.artifact || 'adapter';
    } else {
        errors.push('Choose what to serve.');
    }

    const body = { model };
    const served = String(v.served_model_name || '').trim();
    if (served) body.served_model_name = served;
    if (String(v.gpu_type_id || '').trim()) body.gpu_type_id = String(v.gpu_type_id).trim();
    const optional = {
        gpu_count: num(v.gpu_count),
        max_total_cost_usd: num(v.max_total_cost_usd),
        idle_timeout_minutes: num(v.idle_timeout_minutes),
        max_model_len: num(v.max_model_len),
        max_cost_per_hr: num(v.max_cost_per_hr),
    };
    for (const [k, val] of Object.entries(optional)) {
        if (val !== null) body[k] = val;
    }
    const hours = num(v.max_runtime_hours);
    if (hours === null || hours <= 0 || hours > 24) {
        errors.push('Max runtime must be between 0 and 24 hours.');
    } else {
        body.max_runtime_hours = hours;
    }
    const seqs = num(v.max_num_seqs);
    if (seqs !== null) body.max_num_seqs = seqs;
    body.cloud_type = v.cloud_type === 'community' ? 'community' : 'secure';
    const args = splitArgs(v.vllm_args);
    if (args.length) body.vllm_args = args;
    if (v.keep_instance_on_failure) body.keep_instance_on_failure = true;
    if (String(v.hf_token || '').trim()) body.hf_token = String(v.hf_token).trim();
    return { body, errors };
}

/** The part of a request a plan depends on; the token is never part of it. */
export function planKey(body) {
    const { hf_token, ...rest } = body || {};
    return JSON.stringify(rest);
}

export function money(n) {
    if (n === null || n === undefined || Number.isNaN(Number(n))) return '—';
    return `$${Number(n).toFixed(2)}`;
}

export function duration(seconds) {
    if (seconds === null || seconds === undefined || seconds < 0) return '—';
    const s = Math.floor(seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
    if (m) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
    return `${s}s`;
}

/**
 * Spend so far. Finished sessions carry the controller's figure; for live
 * ones it is estimated from the instance's $/hr since the session was
 * created (billing starts at provisioning, so this errs high).
 */
export function sessionSpend(session, now = Date.now() / 1000) {
    const metrics = session.metrics || {};
    if (metrics.est_cost_usd !== undefined && metrics.est_cost_usd !== null) {
        return { usd: metrics.est_cost_usd, estimate: false };
    }
    const rate = metrics.instance?.cost_per_hr;
    if (!rate || !ACTIVE.includes(session.status) || session.status === 'queued') {
        return { usd: null, estimate: true };
    }
    const end = session.ended_at || now;
    return { usd: Math.max(0, end - session.created_at) / 3600 * rate, estimate: true };
}

/** Human label for where a session is in its life. */
export function sessionPhase(session) {
    if (session.status === 'starting') {
        return PHASES[session.metrics?.phase] || 'Booting the worker';
    }
    if (session.status === 'provisioning') return 'Renting the GPU';
    if (session.status === 'queued') return 'Queued';
    if (session.status === 'ready') return 'Serving';
    if (session.status === 'stopping') return 'Stopping';
    const reasons = {
        user: 'stopped by you', idle_timeout: 'idle timeout', max_runtime_hours: 'runtime cap',
        max_total_cost_usd: 'cost cap', worker_stopped: 'worker stopped', error: 'error',
    };
    const why = reasons[session.stop_reason] || session.stop_reason;
    return why ? `${session.status} (${why})` : session.status;
}

export function sessionModelLabel(session) {
    const m = session.request?.model || {};
    if (m.hf_repo) return m.revision ? `${m.hf_repo}@${m.revision}` : m.hf_repo;
    if (m.store_model) return `store: ${m.store_model}`;
    if (m.job_id) return `${m.job_id} (${m.artifact || 'adapter'})`;
    return session.served_model_name || '—';
}

export class RemoteManager {
    constructor() {
        this.toast = new Toast();
        this.sources = null;
        this.sessions = [];
        this.selected = null;      // session id shown in the details panel
        this.detail = null;        // GET /remote/inference/{id} (has the key)
        this.plan = null;          // {key, plan}
        this.armed = false;        // Start clicked once, waiting for the confirm
        this.revealKey = false;
        this.chat = {};            // session id -> [{role, content}]
        this.chatBusy = false;
        this.pollTimer = null;
        this.bound = false;
        this.visible = false;
    }

    init() {
        if (this.bound) return;
        const section = document.getElementById('remote-section');
        if (!section) return;
        section.addEventListener('click', e => this._onClick(e));
        section.addEventListener('input', e => this._onFormChange(e));
        section.addEventListener('change', e => this._onFormChange(e));
        this.bound = true;
    }

    async load() {
        this.init();
        this.visible = true;
        await Promise.all([this.loadSources(), this.loadSessions()]);
        this._schedule();
    }

    hide() {
        this.visible = false;
        clearTimeout(this.pollTimer);
        this.pollTimer = null;
    }

    _schedule() {
        clearTimeout(this.pollTimer);
        if (!this.visible) return;
        const live = this.sessions.some(s => ACTIVE.includes(s.status));
        this.pollTimer = setTimeout(async () => {
            if (!this.visible || !document.getElementById('remote-section')
                || document.getElementById('remote-section').style.display === 'none') {
                this.visible = false;
                return;
            }
            await this.loadSessions();
            this._schedule();
        }, live ? POLL_ACTIVE_MS : POLL_IDLE_MS);
    }

    // ── data ────────────────────────────────────────────────────────────

    async loadSources() {
        try {
            this.sources = await MerlinaAPI.getRemoteInferenceSources();
        } catch (e) {
            this.sources = { store_models: [], jobs: [], store_error: e.message };
        }
        this._renderSources();
    }

    async loadSessions() {
        try {
            const res = await MerlinaAPI.listRemoteInferenceSessions();
            this.sessions = res.sessions || [];
        } catch (e) {
            this._set('remote-sessions', `<p class="remote-error">Could not load sessions: ${esc(e.message)}</p>`);
            return;
        }
        if (this.selected) {
            try {
                this.detail = await MerlinaAPI.getRemoteInferenceSession(this.selected);
            } catch {
                this.detail = null;
            }
        }
        this._renderSessions();
        this._renderDetail();
    }

    // ── form ────────────────────────────────────────────────────────────

    _values() {
        const val = id => document.getElementById(id)?.value ?? '';
        return {
            source: document.querySelector('input[name="remote-source"]:checked')?.value || 'hf',
            hf_repo: val('remote-hf-repo'),
            revision: val('remote-revision'),
            store_model: val('remote-store-model'),
            job_id: val('remote-job'),
            artifact: val('remote-artifact'),
            served_model_name: val('remote-served-name'),
            gpu_type_id: val('remote-gpu-type'),
            gpu_count: val('remote-gpu-count'),
            cloud_type: val('remote-cloud-type'),
            max_cost_per_hr: val('remote-max-cost-hr'),
            max_runtime_hours: val('remote-max-hours'),
            max_total_cost_usd: val('remote-max-total'),
            idle_timeout_minutes: val('remote-idle'),
            max_model_len: val('remote-max-len'),
            max_num_seqs: val('remote-max-seqs'),
            vllm_args: val('remote-vllm-args'),
            keep_instance_on_failure: document.getElementById('remote-keep')?.checked,
            hf_token: val('remote-hf-token'),
        };
    }

    _onFormChange(e) {
        if (!e.target.closest('#remote-form')) return;
        if (e.target.name === 'remote-source') this._showSourceFields();
        if (e.target.id === 'remote-job') this._renderArtifactChoices();
        // Any edit invalidates the shown plan's Start button.
        if (this.plan && this.plan.key !== planKey(buildRemoteRequest(this._values()).body)) {
            this.armed = false;
            this._renderPlan();
        }
    }

    _showSourceFields() {
        const source = this._values().source;
        for (const kind of ['hf', 'store', 'job']) {
            const el = document.getElementById(`remote-source-${kind}`);
            if (el) el.style.display = kind === source ? '' : 'none';
        }
    }

    _renderSources() {
        const s = this.sources || {};
        const store = document.getElementById('remote-store-model');
        if (store) {
            const models = s.store_models || [];
            store.innerHTML = models.length
                ? models.map(m => `<option value="${esc(m.name)}">${esc(m.name)} — ${(m.size / 1024 ** 3).toFixed(1)} GiB, ${m.files} files</option>`).join('')
                : '<option value="">No models pushed yet</option>';
        }
        const note = document.getElementById('remote-store-note');
        if (note) {
            note.innerHTML = s.store_error
                ? `<span class="remote-error">${esc(s.store_error)}</span>`
                : `Push a model directory with <code>merlina models push &lt;dir&gt; &lt;name&gt;</code>; it lands under <code>models/</code> in the ${esc(s.store_tier || 'artifact')} store.`;
        }
        const job = document.getElementById('remote-job');
        if (job) {
            const jobs = s.jobs || [];
            job.innerHTML = jobs.length
                ? jobs.map(j => `<option value="${esc(j.job_id)}">${esc(j.output_name || j.job_id)} — ${esc(j.base_model || '')}</option>`).join('')
                : '<option value="">No finished remote training jobs</option>';
        }
        this._renderArtifactChoices();
        this._showSourceFields();
    }

    _renderArtifactChoices() {
        const sel = document.getElementById('remote-artifact');
        if (!sel) return;
        const jobId = document.getElementById('remote-job')?.value;
        const job = (this.sources?.jobs || []).find(j => j.job_id === jobId);
        const artifacts = job?.artifacts || ['adapter'];
        const labels = { adapter: 'Adapter (base + vLLM LoRA)', merged: 'Merged weights' };
        sel.innerHTML = artifacts.map(a => `<option value="${a}">${labels[a] || a}</option>`).join('');
    }

    // ── plan / start ────────────────────────────────────────────────────

    async _plan() {
        const { body, errors } = buildRemoteRequest(this._values());
        if (errors.length) {
            this._set('remote-plan', `<div class="storage-warn-box">${errors.map(esc).join('<br>')}</div>`);
            return;
        }
        this.armed = false;
        this._set('remote-plan', '<p class="remote-muted">Planning… (nothing is rented)</p>');
        try {
            const plan = await MerlinaAPI.planRemoteInference(body);
            this.plan = { key: planKey(body), plan };
        } catch (e) {
            this.plan = null;
            this._set('remote-plan', `<div class="storage-warn-box">${esc(e.message)}</div>`);
            return;
        }
        this._renderPlan();
    }

    _planIsFresh() {
        return !!this.plan && this.plan.key === planKey(buildRemoteRequest(this._values()).body);
    }

    _renderPlan() {
        if (!this.plan) return;
        const p = this.plan.plan;
        const z = p.sizing || {};
        const v = p.vram_breakdown || {};
        const fresh = this._planIsFresh();
        const gpu = `${z.gpu_count}× ${z.gpu_type_id}`;
        const cap = p.est_max_total_cost_usd !== null && p.est_max_total_cost_usd !== undefined
            ? `${money(p.est_max_total_cost_usd)} max` : 'no price known';
        const warnings = (p.warnings || []).map(w => `<li>${esc(w)}</li>`).join('');
        const vram = Object.entries(v).filter(([k]) => k !== 'total_gb')
            .map(([k, val]) => `${esc(k.replace(/_gb$/, '').replace(/_/g, ' '))} ${Number(val).toFixed(1)}`).join(' · ');
        let start;
        if (!fresh) {
            start = '<p class="remote-muted">The form changed since this plan. Plan again to start.</p>';
        } else if (this.armed) {
            start = `<div class="storage-danger-box remote-confirm">
                <p><strong>Rent ${esc(gpu)}</strong> at ${money(z.est_cost_per_hr)}/hr, for up to
                ${Number(p.effective_max_hours).toFixed(2)} h (${esc(cap)}).
                It's terminated on stop, idle timeout or a cap.</p>
                <button type="button" class="action-button danger" data-act="start-confirm">Rent and start</button>
                <button type="button" class="action-button" data-act="start-cancel">Cancel</button>
            </div>`;
        } else {
            start = `<button type="button" class="action-button primary-button" data-act="start">Start session…</button>`;
        }
        this._set('remote-plan', `
            <div class="remote-plan">
                <div class="remote-plan-grid">
                    <div><span class="remote-label">GPU</span>${esc(gpu)} <span class="remote-muted">(${esc(z.cloud_type || '')})</span></div>
                    <div><span class="remote-label">Price</span>${money(z.est_cost_per_hr)}/hr</div>
                    <div><span class="remote-label">Ends within</span>${Number(p.effective_max_hours).toFixed(2)} h <span class="remote-muted">(${esc(p.deadline_reason)})</span></div>
                    <div><span class="remote-label">Max spend</span>${esc(cap)}</div>
                    <div><span class="remote-label">VRAM needed</span>${Number(v.total_gb || 0).toFixed(1)} GB <span class="remote-muted">${vram}</span></div>
                    <div><span class="remote-label">Disk</span>${Number(z.disk_required_gb || 0).toFixed(0)} GB</div>
                    <div><span class="remote-label">Serves as</span><code>${esc(p.served_model_name)}</code>${p.lora ? ` + base <code>${esc(p.base_served_model_name)}</code>` : ''}</div>
                    <div><span class="remote-label">Context</span>${esc(p.max_model_len)} tokens × ${esc(p.max_num_seqs)} seqs, TP ${esc(p.tensor_parallel_size)}</div>
                </div>
                ${warnings ? `<ul class="remote-warnings">${warnings}</ul>` : ''}
                <details><summary>vLLM command</summary><pre class="remote-pre">${esc((p.vllm_command || []).join(' '))}</pre></details>
                <div class="remote-start">${start}</div>
            </div>`);
    }

    async _start() {
        if (!this._planIsFresh()) return this._renderPlan();
        const { body } = buildRemoteRequest(this._values());
        this._set('remote-plan', '<p class="remote-muted">Starting…</p>');
        try {
            const res = await MerlinaAPI.startRemoteInference(body);
            this.toast.success(res.message || 'Session started');
            this.plan = null;
            this.armed = false;
            this._set('remote-plan', '');
            this.selected = res.session_id;
            await this.loadSessions();
            this._schedule();
        } catch (e) {
            this.armed = false;
            this._set('remote-plan', `<div class="storage-warn-box">${esc(e.message)}</div>`);
        }
    }

    // ── sessions ────────────────────────────────────────────────────────

    _renderSessions() {
        if (!this.sessions.length) {
            this._set('remote-sessions', '<p class="remote-muted">No sessions yet.</p>');
            return;
        }
        const now = Date.now() / 1000;
        const rows = this.sessions.map(s => {
            const spend = sessionSpend(s, now);
            const gpu = s.metrics?.instance
                ? `${s.metrics.instance.gpu_count}× ${s.metrics.instance.gpu_type_id}`
                : (s.plan?.sizing ? `${s.plan.sizing.gpu_count}× ${s.plan.sizing.gpu_type_id}` : '—');
            const end = s.ended_at || now;
            const live = ACTIVE.includes(s.status);
            return `<tr class="${s.session_id === this.selected ? 'remote-row-selected' : ''}">
                <td><button type="button" class="link-button" data-act="select" data-id="${esc(s.session_id)}">${esc(s.session_id)}</button></td>
                <td><span class="remote-status remote-status-${esc(s.status)}">${esc(sessionPhase(s))}</span></td>
                <td>${esc(sessionModelLabel(s))}</td>
                <td>${esc(gpu)}</td>
                <td>${duration(end - s.created_at)}</td>
                <td>${spend.usd === null ? '—' : `${spend.estimate ? '≈' : ''}${money(spend.usd)}`}</td>
                <td>${live && s.status !== 'stopping'
                    ? `<button type="button" class="action-button danger remote-small" data-act="stop" data-id="${esc(s.session_id)}">Stop</button>`
                    : ''}</td>
            </tr>`;
        }).join('');
        this._set('remote-sessions', `<div class="remote-table-wrap"><table class="storage-table">
            <thead><tr><th>Session</th><th>Status</th><th>Model</th><th>GPU</th><th>Time</th><th>Spend</th><th></th></tr></thead>
            <tbody>${rows}</tbody></table></div>`);
    }

    _renderDetail() {
        const d = this.detail;
        if (!this.selected || !d) {
            this._set('remote-detail', '');
            return;
        }
        // Polling re-renders this panel; keep a half-typed chat message.
        const input = document.getElementById('remote-chat-input');
        const draft = input?.value || '';
        const hadFocus = input && document.activeElement === input;
        const now = Date.now() / 1000;
        const m = d.metrics || {};
        const serve = m.serve || {};
        const key = d.api_key;
        const keyShown = key ? (this.revealKey ? esc(key) : '•'.repeat(24)) : '';
        const left = d.deadline_at ? d.deadline_at - now : null;
        const spend = sessionSpend(d, now);
        const lines = [
            ['Status', esc(sessionPhase(d))],
            ['Spend', spend.usd === null ? '—' : `${spend.estimate ? '≈' : ''}${money(spend.usd)}${m.instance?.cost_per_hr ? ` at ${money(m.instance.cost_per_hr)}/hr` : ''}`],
            ['Deadline', ACTIVE.includes(d.status) && left !== null ? `in ${duration(left)}` : '—'],
        ];
        if (d.base_url) lines.push(['Base URL', `<code>${esc(d.base_url)}</code> <button type="button" class="link-button" data-act="copy" data-text="${esc(d.base_url)}">copy</button>`]);
        if (d.served_model_name) lines.push(['Model name', `<code>${esc(d.served_model_name)}</code>`]);
        if (key) {
            lines.push(['API key', `<code>${keyShown}</code>
                <button type="button" class="link-button" data-act="reveal">${this.revealKey ? 'hide' : 'reveal'}</button>
                <button type="button" class="link-button" data-act="copy-key">copy</button>`]);
        }
        if (d.status === 'ready') {
            lines.push(['Traffic', `${serve.requests ?? 0} requests, ${serve.inflight ?? 0} in flight, idle ${duration(serve.idle_seconds ?? 0)}${serve.rejected_requests ? `, ${serve.rejected_requests} rejected` : ''}`]);
        }
        const example = d.openai?.example;
        const tail = (m.log_tail || []).join('\n');
        const transcript = (this.chat[d.session_id] || []).map(t =>
            `<div class="remote-chat-turn remote-chat-${esc(t.role)}"><strong>${esc(t.role)}</strong><div>${esc(t.content)}</div></div>`).join('');
        this._set('remote-detail', `
            <div class="spell-section">
                <h3 style="margin-top: 0;">${esc(d.session_id)}</h3>
                <dl class="remote-dl">${lines.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>
                ${d.error ? `<div class="storage-warn-box">${esc(d.error)}</div>` : ''}
                ${example ? `<details><summary>curl example</summary><pre class="remote-pre">${esc(example)}</pre>
                    <p class="remote-muted">Set <code>API_KEY</code> to the key above.</p></details>` : ''}
                ${tail ? `<details ${d.status === 'failed' ? 'open' : ''}><summary>vLLM log tail</summary><pre class="remote-pre">${esc(tail)}</pre></details>` : ''}
                ${d.status === 'ready' ? `
                    <h4>Try it</h4>
                    <div class="remote-chat">${transcript || '<p class="remote-muted">Messages go through Merlina to the endpoint; nothing streams.</p>'}</div>
                    <textarea id="remote-chat-input" class="magic-input" rows="3" placeholder="Say hello…"></textarea>
                    <div class="remote-chat-actions">
                        <button type="button" class="action-button primary-button" data-act="chat" ${this.chatBusy ? 'disabled' : ''}>${this.chatBusy ? 'Waiting…' : 'Send'}</button>
                        <button type="button" class="action-button" data-act="chat-clear">Clear</button>
                    </div>` : ''}
            </div>`);
        const next = document.getElementById('remote-chat-input');
        if (next && draft) next.value = draft;
        if (next && hadFocus) next.focus();
    }

    async _sendChat() {
        const input = document.getElementById('remote-chat-input');
        const text = input?.value.trim();
        const id = this.selected;
        if (!text || !id || this.chatBusy) return;
        const history = this.chat[id] = this.chat[id] || [];
        history.push({ role: 'user', content: text });
        input.value = '';
        this.chatBusy = true;
        this._renderDetail();
        try {
            const res = await MerlinaAPI.chatRemoteInference(id, { messages: history });
            const msg = res.choices?.[0]?.message || {};
            history.push({ role: 'assistant', content: msg.content ?? JSON.stringify(res).slice(0, 2000) });
        } catch (e) {
            history.pop();
            this.toast.error(e.message);
            if (input) input.value = text;
        }
        this.chatBusy = false;
        this._renderDetail();
    }

    async _copy(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.toast.success('Copied');
        } catch {
            this.toast.error('Could not copy — select it by hand');
        }
    }

    // ── events ──────────────────────────────────────────────────────────

    async _onClick(e) {
        const el = e.target.closest('[data-act]');
        if (!el) return;
        const act = el.dataset.act;
        const id = el.dataset.id;
        if (act === 'plan') return this._plan();
        if (act === 'start') { this.armed = true; return this._renderPlan(); }
        if (act === 'start-cancel') { this.armed = false; return this._renderPlan(); }
        if (act === 'start-confirm') return this._start();
        if (act === 'refresh') return this.load();
        if (act === 'select') {
            this.selected = this.selected === id ? null : id;
            this.revealKey = false;
            this.detail = null;
            if (this.selected) {
                try { this.detail = await MerlinaAPI.getRemoteInferenceSession(id); } catch { /* shown as empty */ }
            }
            this._renderSessions();
            return this._renderDetail();
        }
        if (act === 'stop') {
            try {
                await MerlinaAPI.stopRemoteInference(id);
                this.toast.info('Stop requested; the instance is terminated within a poll interval');
            } catch (err) {
                this.toast.error(err.message);
            }
            return this.loadSessions();
        }
        if (act === 'reveal') { this.revealKey = !this.revealKey; return this._renderDetail(); }
        if (act === 'copy-key' && this.detail?.api_key) return this._copy(this.detail.api_key);
        if (act === 'copy') return this._copy(el.dataset.text);
        if (act === 'chat') return this._sendChat();
        if (act === 'chat-clear') { delete this.chat[this.selected]; return this._renderDetail(); }
    }

    _set(id, html) {
        const el = document.getElementById(id);
        if (el) el.innerHTML = html;
    }
}
