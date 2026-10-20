// Merge Section Manager
//
// Drives the "Merge" page: builds a mergekit-style recipe (method, base
// model, models with per-model parameters, global parameters, dtype) and
// submits it as a Kessel merge job. The form is generated from
// /merge/methods, so a method Kessel adds shows up here with its
// parameters. The YAML tab edits the same recipe as text; every recipe is
// checked by /merge/preview, which parses it exactly as the job will.

import { MerlinaAPI } from './api.js';
import { Toast } from './ui.js';
import { debounce } from './validation.js';

// Starting values for required parameters, so a fresh row is already valid
const REQUIRED_STARTERS = { weight: 1.0, t: 0.5 };

/** '' -> undefined; JSON for lists/objects/bools (gradients like [0, 0.5, 1]); else a number, else the text. */
function parseParam(text) {
    const v = text.trim();
    if (v === '') return undefined;
    if (/^[[{]|^(true|false|null)$/.test(v)) {
        try { return JSON.parse(v); } catch { /* fall through */ }
    }
    const n = Number(v);
    return Number.isFinite(n) ? n : v;
}

function formatParam(v) {
    if (v === undefined || v === null) return '';
    return typeof v === 'object' ? JSON.stringify(v) : String(v);
}

function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = v;
        else if (v !== undefined && v !== null && v !== false) node.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children) if (c) node.append(c);
    return node;
}

export class MergeManager {
    constructor(app) {
        this.app = app;  // for job tracking and navigation after a submit
        this.methods = {};
        this.mode = 'form';
        this.rowSeq = 0;
        this.toast = new Toast();
        this.checkSeq = 0;
        this.lastCheck = null;  // {ok, yaml|error} for the current input
        this.check = debounce(() => this.runCheck(), 400);
    }

    init() {
        this.section = document.getElementById('merge-section');
        if (!this.section) return;
        this.methodSelect = document.getElementById('merge-method');
        this.baseInput = document.getElementById('merge-base-model');
        this.modelsList = document.getElementById('merge-models');
        this.globalsBox = document.getElementById('merge-global-params');
        this.yamlInput = document.getElementById('merge-yaml');
        this.status = document.getElementById('merge-check');
        this.submitBtn = document.getElementById('merge-submit');

        this.methodSelect.addEventListener('change', () => this.onMethodChange());
        document.getElementById('merge-add-model').addEventListener('click', () => {
            this.addModelRow();
            this.check();
        });
        this.section.querySelectorAll('.merge-mode-btn').forEach(btn =>
            btn.addEventListener('click', () => this.setMode(btn.dataset.mode)));
        // Any edit inside the recipe re-checks it (output name and device aren't part of it)
        this.section.querySelector('.merge-recipe').addEventListener('input', () => this.check());
        this.submitBtn.addEventListener('click', () => this.submit());
    }

    async refresh() {
        await Promise.all([this.loadMethods(), this.loadLocalModels()]);
    }

    async loadMethods() {
        if (Object.keys(this.methods).length) return;
        try {
            const { methods } = await MerlinaAPI.getMergeMethods();
            this.methods = Object.fromEntries(methods.map(m => [m.name, m]));
        } catch (error) {
            this.setStatus('error', error.statusCode === 501
                ? 'Merging needs Kessel on the server: pip install git+https://github.com/Schneewolf-Labs/kessel'
                : `Could not load merge methods: ${error.message}`);
            this.submitBtn.disabled = true;
            return;
        }
        this.methodSelect.replaceChildren(...Object.keys(this.methods).map(name =>
            el('option', { value: name, text: name, selected: name === 'model_stock' })));
        this.onMethodChange();
        if (!this.modelsList.children.length) {
            this.addModelRow();
            this.addModelRow();
        }
        this.check();
    }

    async loadLocalModels() {
        try {
            const { models } = await MerlinaAPI.listInferenceModels();
            const list = document.getElementById('merge-local-models');
            // Full models only: Kessel takes a LoRA as `base+adapter`, typed by hand
            list.replaceChildren(...(models || []).filter(m => !m.is_lora).map(m =>
                el('option', { value: m.path, label: m.name })));
        } catch {
            // Suggestions only; Hub ids and paths still work
        }
    }

    get method() {
        return this.methods[this.methodSelect.value];
    }

    onMethodChange() {
        const m = this.method;
        if (!m) return;
        const hint = document.getElementById('merge-method-hint');
        hint.textContent = m.needs_base
            ? 'Needs a base model: each model is merged as its difference from it.'
            : 'No base model needed (an optional one only sets which checkpoint provides config and tokenizer).';
        this.baseInput.required = m.needs_base;
        document.getElementById('merge-base-required').hidden = !m.needs_base;
        this.renderParams(this.globalsBox, m.global_params, 'merge-global', this.readParams(this.globalsBox));
        for (const row of this.modelsList.children) {
            const box = row.querySelector('.merge-model-params');
            this.renderParams(box, m.model_params, `merge-p${row.dataset.row}`, this.readParams(box));
        }
        this.check();
    }

    /** Fill `box` with one input per parameter; keeps values already typed for parameters that still exist. */
    renderParams(box, params, idPrefix, keep = {}) {
        const fields = Object.entries(params).map(([name, spec]) => {
            const id = `${idPrefix}-${name}`;
            const value = name in keep ? keep[name]
                : spec.required ? REQUIRED_STARTERS[name] : spec.default;
            if (typeof spec.default === 'boolean') {
                return el('label', { class: 'magic-checkbox merge-param' },
                    el('input', { type: 'checkbox', id, 'data-param': name, 'data-default': String(spec.default),
                                  checked: value === true }),
                    el('span', { text: name }));
            }
            return el('div', { class: 'merge-param' },
                el('label', { for: id, text: spec.required ? `${name} *` : name }),
                el('input', { type: 'text', id, class: 'magic-input', inputmode: 'decimal', 'data-param': name,
                              'data-default': formatParam(spec.default), value: formatParam(value),
                              required: spec.required,
                              placeholder: spec.required ? 'required' : `default ${formatParam(spec.default) || 'none'}` }));
        });
        box.replaceChildren(...fields);
        box.hidden = !fields.length;
    }

    /** Parameter values that matter for the recipe: required ones and any changed from the default. */
    readParams(box) {
        const out = {};
        box.querySelectorAll('[data-param]').forEach(input => {
            const name = input.dataset.param;
            if (input.type === 'checkbox') {
                if (String(input.checked) !== input.dataset.default) out[name] = input.checked;
                return;
            }
            const v = parseParam(input.value);
            if (v !== undefined && (input.required || formatParam(v) !== input.dataset.default)) out[name] = v;
        });
        return out;
    }

    addModelRow(model = '', params = {}) {
        const n = ++this.rowSeq;
        const id = `merge-model-${n}`;
        const paramsBox = el('div', { class: 'merge-model-params' });
        const remove = el('button', { type: 'button', class: 'action-button merge-remove',
                                      'aria-label': 'Remove model', text: '✕' });
        const row = el('div', { class: 'merge-model-row', 'data-row': n },
            el('div', { class: 'merge-model-main' },
                el('label', { for: id, class: 'sr-only', text: `Model ${n}` }),
                el('input', { type: 'text', id, class: 'magic-input merge-model-input', list: 'merge-local-models',
                              placeholder: 'org/model, a local path, or model+lora', value: model, required: true }),
                remove),
            paramsBox);
        remove.addEventListener('click', () => {
            row.remove();
            this.check();
        });
        this.modelsList.append(row);
        if (this.method) this.renderParams(paramsBox, this.method.model_params, `merge-p${n}`, params);
        return row;
    }

    /** The recipe the form describes (mergekit's YAML schema, as an object). */
    formRecipe() {
        const recipe = { merge_method: this.methodSelect.value };
        const base = this.baseInput.value.trim();
        if (base) recipe.base_model = base;
        recipe.models = [...this.modelsList.children].map(row => {
            const src = { model: row.querySelector('.merge-model-input').value.trim() };
            const params = this.readParams(row.querySelector('.merge-model-params'));
            if (Object.keys(params).length) src.parameters = params;
            return src;
        }).filter(s => s.model);
        const globals = this.readParams(this.globalsBox);
        if (Object.keys(globals).length) recipe.parameters = globals;
        const dtype = document.getElementById('merge-dtype').value;
        if (dtype) recipe.dtype = dtype;
        return recipe;
    }

    /** Load a recipe into the form; false when it uses something the form can't show (slices). */
    fillForm(recipe) {
        if (!recipe.models || !this.methods[recipe.merge_method]) return false;
        this.methodSelect.value = recipe.merge_method;
        this.baseInput.value = recipe.base_model || '';
        this.modelsList.replaceChildren();
        recipe.models.forEach(s => this.addModelRow(s.model, s.parameters || {}));
        const dtype = document.getElementById('merge-dtype');
        dtype.value = [...dtype.options].some(o => o.value === recipe.dtype) ? recipe.dtype : 'bfloat16';
        const m = this.method;
        this.renderParams(this.globalsBox, m.global_params, 'merge-global', recipe.parameters || {});
        this.onMethodChange();
        return true;
    }

    requestBody() {
        return this.mode === 'yaml' ? { yaml: this.yamlInput.value } : { recipe: this.formRecipe() };
    }

    async setMode(mode) {
        if (mode === this.mode) return;
        if (mode === 'yaml') {
            // Show what Kessel will run; if the form doesn't parse yet, still show the raw recipe
            try {
                const { yaml } = await MerlinaAPI.previewMerge({ recipe: this.formRecipe() });
                this.yamlInput.value = yaml;
            } catch {
                this.yamlInput.value = JSON.stringify(this.formRecipe(), null, 2);
            }
        } else if (this.yamlInput.value.trim()) {
            try {
                const { recipe } = await MerlinaAPI.previewMerge({ yaml: this.yamlInput.value });
                if (!this.fillForm(recipe)) {
                    this.toast.info('This recipe uses slices: keep editing it as YAML');
                    return;
                }
            } catch (error) {
                this.toast.error(`Fix the YAML first: ${error.message}`);
                return;
            }
        }
        this.mode = mode;
        this.section.querySelectorAll('.merge-mode-btn').forEach(btn => {
            const on = btn.dataset.mode === mode;
            btn.classList.toggle('active', on);
            btn.setAttribute('aria-selected', String(on));
        });
        document.getElementById('merge-panel-form').hidden = mode !== 'form';
        document.getElementById('merge-panel-yaml').hidden = mode !== 'yaml';
        this.check();
    }

    async runCheck() {
        const seq = ++this.checkSeq;
        let result;
        try {
            const { yaml } = await MerlinaAPI.previewMerge(this.requestBody());
            result = { ok: true, yaml };
        } catch (error) {
            result = { ok: false, error: error.message };
        }
        if (seq !== this.checkSeq) return;  // a newer edit is being checked
        this.lastCheck = result;
        if (result.ok) {
            this.setStatus('ok', 'Recipe looks good.');
            document.getElementById('merge-yaml-preview').textContent = result.yaml;
        } else {
            this.setStatus('error', result.error);
        }
    }

    setStatus(kind, text) {
        this.status.textContent = text;
        this.status.className = `merge-check merge-check-${kind}`;
    }

    async submit() {
        const nameInput = document.getElementById('merge-output-name');
        const output_name = nameInput.value.trim();
        if (!output_name) {
            nameInput.focus();
            this.toast.error('Give the merged model a name');
            return;
        }
        this.submitBtn.disabled = true;
        try {
            const res = await MerlinaAPI.submitMerge({
                ...this.requestBody(),
                output_name,
                device: document.getElementById('merge-device').value,
            });
            this.toast.success(`Merge queued: ${output_name}`);
            nameInput.value = '';
            await this.app.jobManager._trackSubmittedJob(res.job_id, { output_name });
            this.app.showSection('jobs-section');
        } catch (error) {
            this.toast.error(`Could not start the merge: ${error.message}`);
        } finally {
            this.submitBtn.disabled = false;
        }
    }
}
