// Diffusion Module — image dataset drag-drop, thumbnail grid, caption editor,
// and (later) sample-image gallery + LoRA test playground for the 2.0 wizardry.
//
// Wire-up: app.js imports { initDiffusionDropzone } and calls it on DOMReady.

import { MerlinaAPI } from './api.js';
import { Toast } from './ui.js';

const toast = new Toast();

// In-memory list of File objects the user dropped. We hold them client-side
// (not auto-uploaded) so the user can review/edit captions before committing
// the dataset to disk on the server.
let _stagedFiles = [];     // [{ file, caption, previewUrl }]


function fmtSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}


function defaultCaptionFromName(name) {
    return name
        .replace(/\.[^.]+$/, '')        // drop extension
        .replace(/[_\-]+/g, ' ')        // _ and - → spaces
        .trim();
}


function renderThumbnails() {
    const grid = document.getElementById('diffusion-thumbnails');
    const count = document.getElementById('diffusion-thumbnail-count');
    const wrapper = document.getElementById('diffusion-thumbnail-grid');
    if (!grid || !count || !wrapper) return;

    if (_stagedFiles.length === 0) {
        wrapper.style.display = 'none';
        grid.innerHTML = '';
        return;
    }

    wrapper.style.display = 'block';
    const totalBytes = _stagedFiles.reduce((s, e) => s + e.file.size, 0);
    count.textContent = `${_stagedFiles.length} image${_stagedFiles.length === 1 ? '' : 's'} selected · ${fmtSize(totalBytes)}`;

    grid.innerHTML = _stagedFiles.map((entry, idx) => `
        <div class="diffusion-thumb" style="
            background: white; border-radius: 8px; padding: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            display: flex; flex-direction: column; gap: 4px;
        ">
            <div style="position: relative; aspect-ratio: 1; overflow: hidden; border-radius: 6px; background: #f0e6ff;">
                <img src="${entry.previewUrl}" alt="${entry.file.name}"
                     style="width: 100%; height: 100%; object-fit: cover;">
                <button type="button" data-thumb-remove="${idx}" title="Remove from dataset" style="
                    position: absolute; top: 4px; right: 4px;
                    background: rgba(0,0,0,0.6); color: white;
                    border: none; border-radius: 50%; width: 22px; height: 22px;
                    cursor: pointer; font-size: 0.8em; line-height: 22px;
                ">✕</button>
            </div>
            <div style="font-size: 0.7em; color: #888; word-break: break-all;">
                ${entry.file.name}
            </div>
            <textarea data-thumb-caption="${idx}" rows="2" placeholder="caption…" style="
                width: 100%; border: 1px solid #e0d0f0; border-radius: 4px;
                padding: 4px 6px; font-size: 0.75em; resize: vertical;
                font-family: inherit; min-height: 38px;
            ">${entry.caption || ''}</textarea>
        </div>
    `).join('');

    // Wire per-thumbnail handlers (event delegation done once per render —
    // re-render is rare enough that this is fine.)
    grid.querySelectorAll('[data-thumb-remove]').forEach(btn => {
        btn.addEventListener('click', () => {
            const i = parseInt(btn.dataset.thumbRemove, 10);
            URL.revokeObjectURL(_stagedFiles[i].previewUrl);
            _stagedFiles.splice(i, 1);
            renderThumbnails();
        });
    });
    grid.querySelectorAll('[data-thumb-caption]').forEach(ta => {
        ta.addEventListener('input', () => {
            const i = parseInt(ta.dataset.thumbCaption, 10);
            _stagedFiles[i].caption = ta.value;
        });
    });
}


function addFiles(fileList) {
    const accepted = Array.from(fileList).filter(f => /^image\//.test(f.type));
    if (accepted.length === 0) {
        toast.error('No images found in the drop. Pick PNG / JPG / WebP files.');
        return;
    }
    for (const f of accepted) {
        _stagedFiles.push({
            file: f,
            caption: defaultCaptionFromName(f.name),
            previewUrl: URL.createObjectURL(f),
        });
    }
    if (accepted.length < fileList.length) {
        toast.info(`Added ${accepted.length} images (skipped ${fileList.length - accepted.length} non-images)`);
    } else {
        toast.success(`Added ${accepted.length} image${accepted.length === 1 ? '' : 's'}`);
    }
    renderThumbnails();
}


async function uploadStagedDataset() {
    if (_stagedFiles.length === 0) {
        toast.error('No images to upload — drop some first.');
        return;
    }

    const status = document.getElementById('diffusion-dataset-status');
    const saveBtn = document.getElementById('diffusion-save-dataset');
    if (saveBtn) saveBtn.disabled = true;
    if (status) status.textContent = `Uploading ${_stagedFiles.length} images…`;

    const fd = new FormData();
    const captions = {};
    for (const entry of _stagedFiles) {
        fd.append('files', entry.file, entry.file.name);
        captions[entry.file.name] = entry.caption || '';
    }
    fd.append('captions', JSON.stringify(captions));

    try {
        const resp = await fetch('/dataset/upload-images', { method: 'POST', body: fd });
        if (!resp.ok) {
            const err = await resp.text();
            throw new Error(`HTTP ${resp.status}: ${err.slice(0, 200)}`);
        }
        const data = await resp.json();

        // Pipe the result into the diffusion JSONL path field so the
        // training submit picks it up via the existing config builder.
        const jsonlInput = document.getElementById('diffusion-dataset-jsonl');
        if (jsonlInput) jsonlInput.value = data.jsonl_path;

        if (status) {
            status.innerHTML = `
                <div style="padding: 10px; background: #e6f4e6; border-left: 3px solid #4caf50; border-radius: 6px;">
                    ✨ <strong>${data.image_count} images saved</strong> as
                    <code style="font-size: 0.85em;">${data.jsonl_path}</code>.
                    Ready to train — scroll down and hit Start.
                </div>
            `;
        }
        toast.success(`Dataset saved: ${data.image_count} images. Training is ready.`);
    } catch (e) {
        console.error('Image dataset upload failed:', e);
        if (status) {
            status.innerHTML = `
                <div style="padding: 10px; background: #fde8e8; border-left: 3px solid #d32f2f; border-radius: 6px;">
                    ❌ Upload failed: ${e.message}
                </div>
            `;
        }
        toast.error(`Upload failed: ${e.message}`);
    } finally {
        if (saveBtn) saveBtn.disabled = false;
    }
}


// ---------------------------------------------------------------------
// JSONL image-dataset preview + inline editor
// ---------------------------------------------------------------------
//
// Pulls rows from an existing image-dataset JSONL (server reads it +
// returns thumbnails + captions), renders an editable grid, lets the
// user fix captions / drop bad rows, and POSTs the diff back to
// /dataset/save-jsonl. First save creates a .bak next to the file so
// misclicks are recoverable.

const JSONL_PAGE_SIZE = 24;
let _jsonlState = {
    path: null,
    offset: 0,
    total: 0,
    rows: [],                 // last fetched page, with caption-edit overlays
    deletes: new Set(),       // row_indices marked for deletion (global)
    pendingEdits: new Map(),  // row_index -> new prompt (global)
};


function jsonlPreviewEl(id) { return document.getElementById(`diffusion-jsonl-${id}`); }


function renderJsonlPreview() {
    const grid    = jsonlPreviewEl('thumbnails');
    const summary = jsonlPreviewEl('preview-summary');
    if (!grid || !summary) return;
    const s = _jsonlState;
    const pageEnd = Math.min(s.offset + JSONL_PAGE_SIZE, s.total);
    const dirty = s.pendingEdits.size + s.deletes.size;
    summary.innerHTML = `
        <code style="font-size: 0.85em; color: #444;">${s.path || ''}</code><br>
        <span style="color: #666; font-size: 0.9em;">
            showing ${s.offset + 1}–${pageEnd} of ${s.total}
            ${dirty ? `· <span style="color: var(--primary-purple);">${dirty} unsaved change${dirty === 1 ? '' : 's'}</span>` : ''}
        </span>
    `;

    grid.innerHTML = s.rows.map(row => {
        const isDeleted = s.deletes.has(row.row_index);
        const overlay = s.pendingEdits.get(row.row_index);
        const caption = overlay !== undefined ? overlay : row.prompt;
        return `
            <div style="
                background: ${isDeleted ? '#fde4e4' : 'white'};
                border-radius: 8px; padding: 6px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                opacity: ${isDeleted ? 0.5 : 1};
                display: flex; flex-direction: column; gap: 4px;
            ">
                <div style="position: relative; aspect-ratio: 1; overflow: hidden; border-radius: 6px; background: #f5edff;">
                    <img src="${row.image_url}" alt="row ${row.row_index}" loading="lazy"
                         style="width: 100%; height: 100%; object-fit: cover;">
                    <button type="button" data-jdrop="${row.row_index}" title="${isDeleted ? 'Restore' : 'Mark for deletion'}" style="
                        position: absolute; top: 4px; right: 4px;
                        background: ${isDeleted ? '#4caf50' : 'rgba(0,0,0,0.6)'}; color: white;
                        border: none; border-radius: 50%; width: 24px; height: 24px;
                        cursor: pointer; font-size: 0.8em; line-height: 24px;
                    ">${isDeleted ? '↺' : '✕'}</button>
                </div>
                <div style="font-size: 0.7em; color: #888;">row ${row.row_index}</div>
                <textarea data-jedit="${row.row_index}" rows="3" ${isDeleted ? 'disabled' : ''} style="
                    width: 100%; border: 1px solid ${overlay !== undefined ? 'var(--primary-purple)' : '#e0d0f0'};
                    border-radius: 4px; padding: 4px 6px; font-size: 0.75em;
                    resize: vertical; font-family: inherit; min-height: 50px;
                    ${isDeleted ? 'background: #fafafa;' : ''}
                ">${caption || ''}</textarea>
            </div>
        `;
    }).join('');

    grid.querySelectorAll('[data-jdrop]').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.dataset.jdrop, 10);
            if (s.deletes.has(idx)) s.deletes.delete(idx);
            else                    s.deletes.add(idx);
            renderJsonlPreview();
        });
    });
    grid.querySelectorAll('[data-jedit]').forEach(ta => {
        ta.addEventListener('input', () => {
            const idx = parseInt(ta.dataset.jedit, 10);
            const original = s.rows.find(r => r.row_index === idx)?.prompt || '';
            if (ta.value === original) s.pendingEdits.delete(idx);
            else                        s.pendingEdits.set(idx, ta.value);
            // Don't re-render — just update the summary dirty count
            const summary = jsonlPreviewEl('preview-summary');
            const dirty = s.pendingEdits.size + s.deletes.size;
            const counter = summary.querySelector('span span');
            // Cheap: just trigger a header re-render
            const pageEnd = Math.min(s.offset + JSONL_PAGE_SIZE, s.total);
            summary.innerHTML = `
                <code style="font-size: 0.85em; color: #444;">${s.path || ''}</code><br>
                <span style="color: #666; font-size: 0.9em;">
                    showing ${s.offset + 1}–${pageEnd} of ${s.total}
                    ${dirty ? `· <span style="color: var(--primary-purple);">${dirty} unsaved change${dirty === 1 ? '' : 's'}</span>` : ''}
                </span>
            `;
        });
    });
}


async function loadJsonlPage() {
    const section = jsonlPreviewEl('preview-section');
    const status  = jsonlPreviewEl('status');
    section.style.display = 'block';
    status.textContent = 'loading…';
    const s = _jsonlState;
    try {
        const url = `/dataset/preview-images?jsonl_path=${encodeURIComponent(s.path)}&limit=${JSONL_PAGE_SIZE}&offset=${s.offset}`;
        const r = await fetch(url);
        if (!r.ok) {
            const err = await r.text();
            throw new Error(`HTTP ${r.status}: ${err.slice(0, 200)}`);
        }
        const data = await r.json();
        s.rows = data.rows;
        s.total = data.total;
        renderJsonlPreview();
        status.textContent = `loaded ${data.returned} rows`;
    } catch (e) {
        console.error('JSONL preview failed:', e);
        status.innerHTML = `<span style="color: #d32f2f;">load failed: ${e.message}</span>`;
    }
}


async function saveJsonlEdits() {
    const s = _jsonlState;
    const status = jsonlPreviewEl('status');
    if (s.pendingEdits.size === 0 && s.deletes.size === 0) {
        status.textContent = 'nothing to save';
        return;
    }
    status.textContent = 'saving…';
    const body = {
        jsonl_path: s.path,
        edits: Array.from(s.pendingEdits.entries()).map(([idx, p]) => ({ row_index: idx, prompt: p })),
        deletes: Array.from(s.deletes),
    };
    try {
        const r = await fetch('/dataset/save-jsonl', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!r.ok) {
            const err = await r.text();
            throw new Error(`HTTP ${r.status}: ${err.slice(0, 200)}`);
        }
        const data = await r.json();
        status.innerHTML = `
            <span style="color: #2e7d32;">
                ✨ saved — edited ${data.edited}, deleted ${data.deleted},
                ${data.total_after} rows remain.
                backup at <code>${data.backup}</code>.
            </span>
        `;
        toast.success(`Dataset updated: ${data.edited} captions + ${data.deleted} removals`);
        // Clear staged edits + reload current page (indices may have shifted only if deletes;
        // we keep the offset for predictability, the user can navigate).
        s.pendingEdits.clear();
        s.deletes.clear();
        await loadJsonlPage();
    } catch (e) {
        status.innerHTML = `<span style="color: #d32f2f;">save failed: ${e.message}</span>`;
        toast.error(`Save failed: ${e.message}`);
    }
}


function initJsonlPreview() {
    const btn = document.getElementById('diffusion-jsonl-preview');
    const input = document.getElementById('diffusion-dataset-jsonl');
    if (!btn || !input) return;

    btn.addEventListener('click', () => {
        const p = input.value.trim();
        if (!p) {
            toast.error('Paste a JSONL path first.');
            return;
        }
        _jsonlState = { path: p, offset: 0, total: 0, rows: [], deletes: new Set(), pendingEdits: new Map() };
        loadJsonlPage();
    });

    jsonlPreviewEl('next-page')?.addEventListener('click', () => {
        const s = _jsonlState;
        if (s.offset + JSONL_PAGE_SIZE >= s.total) return;
        s.offset += JSONL_PAGE_SIZE;
        loadJsonlPage();
    });
    jsonlPreviewEl('prev-page')?.addEventListener('click', () => {
        const s = _jsonlState;
        if (s.offset === 0) return;
        s.offset = Math.max(0, s.offset - JSONL_PAGE_SIZE);
        loadJsonlPage();
    });
    jsonlPreviewEl('save-edits')?.addEventListener('click', saveJsonlEdits);
}


// ---------------------------------------------------------------------
// Sample-image gallery (post-training previews)
// ---------------------------------------------------------------------

/**
 * Fetch and render the sample-image gallery for a job into the
 * #job-samples-gallery-section block on the job detail modal. No-op when
 * the job didn't produce samples (text/VLM runs, in-progress runs, etc).
 */
export async function renderJobSamples(jobId) {
    const section = document.getElementById('job-samples-gallery-section');
    const grid = document.getElementById('job-samples-grid');
    const empty = document.getElementById('job-samples-empty');
    if (!section || !grid) return;

    section.style.display = 'none';
    grid.innerHTML = '';
    if (empty) empty.style.display = 'none';

    let data;
    try {
        const resp = await fetch(`/jobs/${encodeURIComponent(jobId)}/samples`);
        if (!resp.ok) return;
        data = await resp.json();
    } catch (e) {
        console.debug('Sample gallery fetch failed (non-fatal):', e);
        return;
    }

    if (!data || !Array.isArray(data.samples) || data.samples.length === 0) {
        // Show the section only if we got back a structured response (meaning
        // the job exists) so users get the "no samples yet" hint instead of
        // a blank space mid-modal.
        if (data && typeof data === 'object') {
            section.style.display = 'block';
            if (empty) empty.style.display = 'block';
        }
        return;
    }

    section.style.display = 'block';
    grid.innerHTML = data.samples.map((s, i) => `
        <figure style="
            margin: 0; background: white; border-radius: 10px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1); overflow: hidden;
            display: flex; flex-direction: column;
        ">
            <img src="${s.url}" alt="${s.prompt.replace(/"/g, '&quot;')}"
                 loading="lazy"
                 style="width: 100%; aspect-ratio: 1; object-fit: cover; background: #f5f0ff;">
            <figcaption style="padding: 8px 10px; font-size: 0.78em; color: #444;
                              border-top: 1px solid #f0e6ff;">
                ${s.prompt}
            </figcaption>
        </figure>
    `).join('');
}

// Expose globally so jobs.js (which doesn't currently use ES module imports)
// can call the renderer when a job detail modal opens, without us having
// to refactor its imports in this PR.
window.merlinaRenderJobSamples = renderJobSamples;


// ---------------------------------------------------------------------
// Diffusion Playground — pick base + LoRA, prompt → image
// ---------------------------------------------------------------------

async function refreshPlaygroundLoraList() {
    const select = document.getElementById('dplay-lora');
    if (!select) return;
    select.innerHTML = '<option value="">Loading…</option>';
    try {
        const r = await fetch('/diffusion/loras');
        const data = await r.json();
        if (!data.loras || data.loras.length === 0) {
            select.innerHTML = '<option value="">— no trained LoRAs found in ./models/ —</option>';
            return;
        }
        select.innerHTML = data.loras.map(l =>
            `<option value="${l.name}">${l.name} (${(l.size_bytes / 1024 / 1024).toFixed(1)} MB)</option>`
        ).join('');
    } catch (e) {
        select.innerHTML = `<option value="">error loading: ${e.message}</option>`;
    }
}


async function conjureImage() {
    const baseModel = document.getElementById('dplay-base-model')?.value.trim();
    const loraName  = document.getElementById('dplay-lora')?.value;
    const adapter   = document.getElementById('dplay-adapter')?.value;
    const prompt    = document.getElementById('dplay-prompt')?.value.trim();
    const width     = parseInt(document.getElementById('dplay-width')?.value || 1024);
    const height    = parseInt(document.getElementById('dplay-height')?.value || 1024);
    const numSteps  = parseInt(document.getElementById('dplay-steps')?.value || 25);

    if (!baseModel || !loraName || !prompt) {
        toast.error('Need a base model, a LoRA, and a prompt.');
        return;
    }

    const status   = document.getElementById('dplay-status');
    const conjure  = document.getElementById('dplay-conjure');
    const resultEl = document.getElementById('dplay-result');
    const imgEl    = document.getElementById('dplay-result-image');
    const promptEl = document.getElementById('dplay-result-prompt');

    if (conjure) {
        conjure.disabled = true;
        conjure.textContent = '✨ Conjuring…';
    }
    if (status) {
        status.innerHTML = `
            <div style="padding: 10px; background: #f0e6ff; border-left: 3px solid var(--primary-purple); border-radius: 6px;">
                🧙 Loading pipeline + LoRA, denoising ${numSteps} steps at ${width}×${height}… this can take ~30–60 seconds.
            </div>
        `;
    }
    if (resultEl) resultEl.style.display = 'none';

    try {
        const r = await fetch('/diffusion/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                base_model: baseModel,
                lora_name: loraName,
                adapter,
                prompt,
                width, height,
                num_steps: numSteps,
            }),
        });
        if (!r.ok) {
            const err = await r.text();
            throw new Error(`HTTP ${r.status}: ${err.slice(0, 400)}`);
        }
        const data = await r.json();

        // Bust browser cache — playground reuses subdir-per-request ids
        // anyway, but be paranoid.
        if (imgEl) imgEl.src = data.image_url + '?_=' + Date.now();
        if (promptEl) promptEl.textContent = `Prompt: ${data.prompt}`;
        if (resultEl) resultEl.style.display = 'block';
        if (status) {
            status.innerHTML = `
                <div style="padding: 10px; background: #e6f4e6; border-left: 3px solid #4caf50; border-radius: 6px;">
                    ✨ Done!
                </div>
            `;
        }
        toast.success('Image conjured!');
    } catch (e) {
        console.error('Playground conjure failed:', e);
        if (status) {
            status.innerHTML = `
                <div style="padding: 10px; background: #fde8e8; border-left: 3px solid #d32f2f; border-radius: 6px;">
                    ❌ ${e.message}
                </div>
            `;
        }
        toast.error(`Conjure failed: ${e.message}`);
    } finally {
        if (conjure) {
            conjure.disabled = false;
            conjure.textContent = '✨ Conjure Image';
        }
    }
}


export function initDiffusionPlayground() {
    const conjure = document.getElementById('dplay-conjure');
    const refresh = document.getElementById('dplay-refresh-loras');
    if (!conjure) return;

    conjure.addEventListener('click', conjureImage);
    if (refresh) refresh.addEventListener('click', refreshPlaygroundLoraList);
    refreshPlaygroundLoraList();  // populate initial list

    // Inference mode toggle — text-chat vs diffusion-image. Both live
    // under #inference-section in 2.0+ (was a separate Playground nav
    // entry pre-merge). The mode select swaps which sub-pane is shown.
    const modeSelect = document.getElementById('inference-mode');
    const llmPane = document.getElementById('inference-llm-mode');
    const diffPane = document.getElementById('inference-diffusion-mode');
    if (modeSelect && llmPane && diffPane) {
        const apply = (mode) => {
            const isDiff = mode === 'diffusion';
            llmPane.style.display  = isDiff ? 'none' : '';
            diffPane.style.display = isDiff ? '' : 'none';
            if (isDiff) refreshPlaygroundLoraList();  // re-pull on each visit
        };
        modeSelect.addEventListener('change', e => apply(e.target.value));
        apply(modeSelect.value);
    }
}


export function initDiffusionDropzone() {
    const dropzone = document.getElementById('diffusion-dropzone');
    const fileInput = document.getElementById('diffusion-image-files');
    const clearBtn = document.getElementById('diffusion-clear-images');
    const saveBtn = document.getElementById('diffusion-save-dataset');
    if (!dropzone || !fileInput) return;  // diffusion fields not on this page

    // Click-to-pick (the <input> already overlays the dropzone with cursor:pointer)
    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length) {
            addFiles(fileInput.files);
            fileInput.value = '';  // reset so the same selection re-triggers
        }
    });

    // Drag-drop
    const onDragOver = (e) => {
        e.preventDefault();
        dropzone.style.background = 'rgba(124, 77, 255, 0.12)';
        dropzone.style.borderColor = '#7c4dff';
    };
    const onDragLeave = () => {
        dropzone.style.background = 'rgba(255,255,255,0.55)';
        dropzone.style.borderColor = '';
    };
    const onDrop = (e) => {
        e.preventDefault();
        onDragLeave();
        if (e.dataTransfer && e.dataTransfer.files) {
            addFiles(e.dataTransfer.files);
        }
    };
    dropzone.addEventListener('dragover', onDragOver);
    dropzone.addEventListener('dragleave', onDragLeave);
    dropzone.addEventListener('drop', onDrop);

    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            _stagedFiles.forEach(e => URL.revokeObjectURL(e.previewUrl));
            _stagedFiles = [];
            const status = document.getElementById('diffusion-dataset-status');
            if (status) status.textContent = '';
            const jsonlInput = document.getElementById('diffusion-dataset-jsonl');
            if (jsonlInput) jsonlInput.value = '';
            renderThumbnails();
        });
    }
    if (saveBtn) {
        saveBtn.addEventListener('click', uploadStagedDataset);
    }

    // Wire the JSONL preview + inline editor (separate from drag-drop).
    initJsonlPreview();
}
