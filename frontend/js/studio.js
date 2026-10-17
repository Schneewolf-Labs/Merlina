// Shared navigation context and live recipe summary. Feature managers retain
// ownership of their forms; this module only presents their current values.
const setupSections = ['model-section', 'dataset-section', 'config-section'];

export class Studio {
    constructor(navigate) {
        this.navigate = navigate;
        document.addEventListener('click', event => {
            const action = event.target.closest('[data-go-section]');
            if (action) this.navigate(action.dataset.goSection);
        });
        document.querySelector('.brand')?.addEventListener('click', event => {
            event.preventDefault();
            this.navigate('model-section');
        });
        document.addEventListener('input', () => this.updateSummary());
        // Capture also catches non-bubbling changes from saved configurations.
        // Wait until the form's own change handlers finish synchronizing fields.
        document.addEventListener('change', () => queueMicrotask(() => this.updateSummary()), true);
        document.addEventListener('invalid', event => {
            let disclosure = event.target.closest('details');
            while (disclosure) {
                disclosure.open = true;
                disclosure = disclosure.parentElement.closest('details');
            }
        }, true);
        // Dataset cards can be added, removed, or restored from a saved recipe.
        const datasets = document.getElementById('datasets-list');
        if (datasets) new MutationObserver(() => this.updateSummary()).observe(datasets, { childList: true });
        this.updateSummary();
    }

    updateSummary() {
        const setText = (id, value) => {
            const element = document.getElementById(id);
            if (element && element.textContent !== value) element.textContent = value;
        };
        setText('summary-model', document.getElementById('base-model')?.value.trim() || 'Not set');
        const count = document.querySelectorAll('#datasets-list .dataset-card').length;
        setText('summary-datasets', `${count} ${count === 1 ? 'dataset' : 'datasets'}`);
        const mode = document.getElementById('training-mode');
        const name = mode?.selectedOptions[0]?.textContent.split(' - ')[0] || 'ORPO';
        setText('summary-method', name);
        const isDiffusion = mode?.value.startsWith('diffusion_');
        setText('summary-adaptation', isDiffusion ? 'Image LoRA' :
            document.getElementById('use-lora')?.checked ? 'LoRA fine-tuning' : 'Full fine-tuning');
        const strategy = document.getElementById('multi-gpu-strategy');
        const gpus = Array.from(document.getElementById('gpu-selection')?.selectedOptions || [])
            .filter(option => option.value !== '');
        const hardware = gpus.length ? `${gpus.length} GPU${gpus.length > 1 ? 's' : ''} selected` : 'Auto-detect GPUs';
        const strategyNames = { auto: 'customize hardware', ddp: 'distributed training', single: 'single process' };
        setText('compute-summary', `${hardware} · ${strategyNames[strategy?.value] || 'customize hardware'}`);
    }

    showSection(sectionId, { focus = true } = {}) {
        const target = document.getElementById(sectionId);
        const button = document.querySelector(`.section-nav-btn[data-section="${sectionId}"]`);
        document.getElementById('current-page-label').textContent = button?.querySelector('.nav-label')?.textContent || 'Model';
        // The group is the nearest nav heading above the button (Create / Explore / Workspace).
        let group = button?.previousElementSibling;
        while (group && !group.classList.contains('nav-group-label')) group = group.previousElementSibling;
        const groupLabel = document.getElementById('current-group-label');
        if (groupLabel && group) {
            const text = group.textContent.trim().toLowerCase();
            groupLabel.textContent = text.charAt(0).toUpperCase() + text.slice(1);
        }
        const isSetup = setupSections.includes(sectionId);
        document.getElementById('setup-summary').hidden = !isSetup;
        document.getElementById('main-content').classList.toggle('workspace-wide', !isSetup);
        document.querySelectorAll('.section-nav-btn').forEach(item => {
            if (item === button) item.setAttribute('aria-current', 'page');
            else item.removeAttribute('aria-current');
        });
        this.updateSummary();
        if (focus) target?.querySelector('h2')?.focus({ preventScroll: true });
        // Keep the selected destination in view in the horizontal mobile nav.
        if (window.matchMedia('(max-width: 760px)').matches && button) {
            const nav = document.getElementById('section-nav');
            nav.scrollLeft = button.offsetLeft - nav.offsetLeft - 12;
        }
    }
}
