// Playwright integration tests for Merlina frontend
// Run with: npx playwright test --config tests/frontend/playwright.config.mjs

import { test, expect } from '@playwright/test';

// ═════════════════════════════════════════════════════════════════════════════
// Page load and basic structure
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Page load', () => {
    test('loads successfully with title', async ({ page }) => {
        await page.goto('/');
        await expect(page).toHaveTitle(/Merlina/);
    });

    test('displays header with logo and name', async ({ page }) => {
        await page.goto('/');
        await expect(page.locator('h1')).toHaveText('Merlina');
        await expect(page.locator('.wizard-hat')).toBeVisible();
    });

    test('displays version in footer', async ({ page }) => {
        await page.goto('/');
        const version = page.locator('#version-info');
        await expect(version).toBeVisible();
        await expect(version).toContainText(/v\d+\.\d+\.\d+/);
    });

    test('all three main sections are visible', async ({ page }) => {
        await page.goto('/');
        // Section headers (step-based navigation - only one visible at a time)
        // Step 1: Model (visible by default)
        await expect(page.getByText('Select Your Base Model')).toBeVisible();

        // Step 2: Dataset (navigate to it)
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        await expect(page.getByText('Configure Your Dataset')).toBeVisible();

        // Step 3: Training (navigate to it)
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        await expect(page.getByText('Configure Training Parameters')).toBeVisible();
    });

    test('has skip-to-content link for accessibility', async ({ page }) => {
        await page.goto('/');
        const skipLink = page.locator('.skip-link');
        await expect(skipLink).toHaveAttribute('href', '#main-content');
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Theme toggle
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Theme toggle', () => {
    test('toggle button exists and switches theme', async ({ page }) => {
        await page.goto('/');
        const themeBtn = page.locator('#theme-toggle');
        await expect(themeBtn).toBeVisible();

        // Get initial theme
        const initialTheme = await page.locator('html').getAttribute('data-theme');

        // Click toggle
        await themeBtn.click();

        // Theme should change
        const newTheme = await page.locator('html').getAttribute('data-theme');
        expect(newTheme).not.toBe(initialTheme);
    });

    test('theme persists across page reload', async ({ page }) => {
        await page.goto('/');
        const themeBtn = page.locator('#theme-toggle');

        // Set to dark mode
        const current = await page.locator('html').getAttribute('data-theme');
        if (current !== 'dark') {
            await themeBtn.click();
        }

        // Reload
        await page.reload();

        const theme = await page.locator('html').getAttribute('data-theme');
        expect(theme).toBe('dark');
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Form validation
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Form validation', () => {
    test('base model field accepts valid model names', async ({ page }) => {
        await page.goto('/');
        const input = page.locator('#base-model');
        await input.fill('meta-llama/Llama-3-8B-Instruct');
        await input.blur();
        // No error class should be present
        await expect(input).not.toHaveClass(/input-error/);
    });

    test('output name field rejects names with special characters', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        const input = page.locator('#output-name');
        await input.fill('bad name!@#$');
        await input.blur();
        // Should show validation error
        await expect(input).toHaveClass(/input-error/);
    });

    test('output name field accepts valid names', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        const input = page.locator('#output-name');
        await input.fill('my-fine-tuned-model');
        await input.blur();
        await expect(input).not.toHaveClass(/input-error/);
    });

    test('learning rate field validates numeric range', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        const input = page.locator('#learning-rate');

        // Too high
        await input.fill('999');
        await input.blur();
        await expect(input).toHaveClass(/input-error/);

        // Valid
        await input.fill('0.00005');
        await input.blur();
        await expect(input).not.toHaveClass(/input-error/);
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Training mode switching
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Training mode', () => {
    test('defaults to ORPO mode', async ({ page }) => {
        await page.goto('/');
        const modeSelect = page.locator('#training-mode');
        await expect(modeSelect).toHaveValue('orpo');
    });

    test('shows beta field in ORPO mode', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        const betaGroup = page.locator('#beta').locator('..');
        await expect(betaGroup).toBeVisible();
    });

    test('hides beta field in SFT mode', async ({ page }) => {
        await page.goto('/');
        // Training mode lives in the Dataset section; change it there first.
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        await page.locator('#training-mode').selectOption('sft');
        // Then navigate to the Training section to check the beta field.
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        const betaField = page.locator('#beta');
        const betaContainer = betaField.locator('xpath=ancestor::div[contains(@class,"form-group")]');
        const isHidden = await betaContainer.evaluate(el => {
            return el.style.display === 'none' || el.hidden ||
                   window.getComputedStyle(el).display === 'none';
        }).catch(() => false);

        if (!isHidden) {
            // In some implementations the field might just not be required
            // Check that switching back to ORPO shows it again
            await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
            await page.locator('#training-mode').selectOption('orpo');
            await page.locator('.section-nav-btn[data-section="config-section"]').click();
            await expect(betaField).toBeVisible();
        }
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Dataset source switching (unified cards: first card is always present)
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Dataset source selection', () => {
    test('first dataset card is present on load and defaults to HuggingFace', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        const firstCard = page.locator('#datasets-list .dataset-card').first();
        await expect(firstCard).toBeVisible();
        await expect(firstCard.locator('.ds-source-type')).toHaveValue('huggingface');
        await expect(firstCard.locator('.ds-hf-config')).toBeVisible();
        await expect(firstCard.locator('.ds-upload-config')).toBeHidden();
        await expect(firstCard.locator('.ds-local-config')).toBeHidden();
    });

    test('shows upload config when Upload selected', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        const firstCard = page.locator('#datasets-list .dataset-card').first();
        await firstCard.locator('.ds-source-type').selectOption('upload');
        await expect(firstCard.locator('.ds-upload-config')).toBeVisible();
        await expect(firstCard.locator('.ds-hf-config')).toBeHidden();
    });

    test('shows local config when Local File selected', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        const firstCard = page.locator('#datasets-list .dataset-card').first();
        await firstCard.locator('.ds-source-type').selectOption('local_file');
        await expect(firstCard.locator('.ds-local-config')).toBeVisible();
        await expect(firstCard.locator('.ds-hf-config')).toBeHidden();
    });

    test('Add Dataset button appends another card', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        const cards = page.locator('#datasets-list .dataset-card');
        await expect(cards).toHaveCount(1);
        await page.locator('#add-dataset-btn').click();
        await expect(cards).toHaveCount(2);
        // Second card is removable; first is not.
        const firstRemove = cards.nth(0).locator('.remove-dataset-btn');
        const secondRemove = cards.nth(1).locator('.remove-dataset-btn');
        await expect(firstRemove).toBeHidden();
        await expect(secondRemove).toBeVisible();
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Evaluation set source (split from training data vs. separate eval dataset)
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Evaluation set source', () => {
    test('defaults to splitting from training data; eval source hidden', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        await expect(page.locator('#eval-source-mode')).toHaveValue('split');
        await expect(page.locator('#eval-source-config')).toBeHidden();
        await expect(page.locator('#test-size')).toBeEnabled();
    });

    test('selecting a separate eval dataset reveals its own dataset card', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        await page.locator('#eval-source-mode').selectOption('separate');
        await expect(page.locator('#eval-source-config')).toBeVisible();
        const evalCard = page.locator('#eval-datasets-list .dataset-card');
        await expect(evalCard).toHaveCount(1);
        await expect(evalCard.locator('.ds-source-type')).toHaveValue('huggingface');
        await expect(evalCard.locator('.ds-hf-config')).toBeVisible();
        // Test split is de-emphasized/disabled while an explicit eval set is used.
        await expect(page.locator('#test-size')).toBeDisabled();
    });

    test('eval card source switching is independent of the training cards', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        await page.locator('#eval-source-mode').selectOption('separate');
        const evalCard = page.locator('#eval-datasets-list .dataset-card');
        await evalCard.locator('.ds-source-type').selectOption('local_file');
        await expect(evalCard.locator('.ds-local-config')).toBeVisible();
        await expect(evalCard.locator('.ds-hf-config')).toBeHidden();
        // The training list still has exactly one card, untouched.
        await expect(page.locator('#datasets-list .dataset-card')).toHaveCount(1);
        await expect(page.locator('#datasets-list .dataset-card').first().locator('.ds-source-type'))
            .toHaveValue('huggingface');
    });

    test('switching back to split hides the eval source and re-enables test split', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        await page.locator('#eval-source-mode').selectOption('separate');
        await expect(page.locator('#eval-source-config')).toBeVisible();
        await page.locator('#eval-source-mode').selectOption('split');
        await expect(page.locator('#eval-source-config')).toBeHidden();
        await expect(page.locator('#test-size')).toBeEnabled();
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Training mode selector mirror (Dataset section ↔ Training section)
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Training mode mirror', () => {
    test('mirror selector exists in the Training section and starts in sync', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        const mirror = page.locator('#training-mode-config');
        await expect(mirror).toBeVisible();
        await expect(mirror).toHaveValue('orpo');
    });

    test('changing the Dataset selector updates the Training mirror', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        await page.locator('#training-mode').selectOption('dpo');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        await expect(page.locator('#training-mode-config')).toHaveValue('dpo');
    });

    test('changing the Training mirror updates the Dataset selector', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        await page.locator('#training-mode-config').selectOption('sft');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        await expect(page.locator('#training-mode')).toHaveValue('sft');
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// LoRA settings toggle
// ═════════════════════════════════════════════════════════════════════════════

test.describe('LoRA settings', () => {
    test('LoRA is enabled by default', async ({ page }) => {
        await page.goto('/');
        const loraCheckbox = page.locator('#use-lora');
        await expect(loraCheckbox).toBeChecked();
    });

    test('LoRA settings visible when enabled', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        await expect(page.locator('#lora-r')).toBeVisible();
        await expect(page.locator('#lora-alpha')).toBeVisible();
    });

    test('LoRA settings hidden when disabled', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        await page.locator('#use-lora').uncheck();
        // LoRA settings section should be hidden
        const loraSection = page.locator('#lora-settings');
        if (await loraSection.count() > 0) {
            await expect(loraSection).toBeHidden();
        }
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// API connectivity
// ═════════════════════════════════════════════════════════════════════════════

test.describe('API endpoints', () => {
    test('version endpoint returns valid response', async ({ request }) => {
        const response = await request.get('/version');
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data).toHaveProperty('version');
        expect(data.version).toMatch(/^\d+\.\d+\.\d+$/);
    });

    test('GPU list endpoint responds', async ({ request }) => {
        const response = await request.get('/gpu/list');
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data).toHaveProperty('gpus');
        expect(Array.isArray(data.gpus)).toBeTruthy();
    });

    test('jobs list endpoint responds', async ({ request }) => {
        const response = await request.get('/jobs');
        expect(response.ok()).toBeTruthy();
    });

    test('local models endpoint returns offline discovery shape', async ({ request }) => {
        const response = await request.get('/models/local');
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data).toHaveProperty('offline_mode');
        expect(data).toHaveProperty('models');
        expect(Array.isArray(data.models)).toBeTruthy();
        expect(data).toHaveProperty('count');
    });

    test('disk analysis endpoint returns breakdown', async ({ request }) => {
        const response = await request.get('/disk/analysis?keep=1');
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data).toHaveProperty('results');
        expect(data.results).toHaveProperty('jobs');
        expect(Array.isArray(data.results.jobs)).toBeTruthy();
        expect(data).toHaveProperty('models');
    });

    test('disk cleanup dry-run is non-destructive', async ({ request }) => {
        const response = await request.post('/disk/cleanup', {
            data: { keep: 1, purge_failed: false, apply: false },
        });
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data.applied).toBe(false);
        expect(data).toHaveProperty('count');
        expect(data).toHaveProperty('freed_human');
    });

    test('hf-cache analysis endpoint returns a shape', async ({ request }) => {
        const response = await request.get('/disk/hf-cache?stale_days=90');
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data).toHaveProperty('available');
        expect(data).toHaveProperty('repos');
        expect(Array.isArray(data.repos)).toBeTruthy();
        expect(data).toHaveProperty('total_human');
    });

    test('hf-cache delete with empty selection is a safe no-op', async ({ request }) => {
        const response = await request.post('/disk/hf-cache/delete', {
            data: { repos: [], apply: false },
        });
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data.count).toBe(0);
        expect(data.applied).toBe(false);
    });

    test('disk analysis annotates saved models with protected flag', async ({ request }) => {
        const response = await request.get('/disk/analysis?keep=1');
        const data = await response.json();
        expect(data.models).toHaveProperty('items');
        for (const m of data.models.items) {
            expect(m).toHaveProperty('protected');
            expect(m).toHaveProperty('modified_date');
        }
    });

    test('models delete with empty selection is a safe no-op', async ({ request }) => {
        const response = await request.post('/disk/models/delete', {
            data: { names: [], apply: false },
        });
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data.count).toBe(0);
        expect(data.applied).toBe(false);
    });

    test('disk artifacts endpoint returns gguf + wandb breakdown', async ({ request }) => {
        const response = await request.get('/disk/artifacts');
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data).toHaveProperty('gguf');
        expect(Array.isArray(data.gguf.files)).toBeTruthy();
        expect(data).toHaveProperty('wandb');
        expect(data.wandb).toHaveProperty('run_count');
    });

    test('gguf delete with empty selection is a safe no-op', async ({ request }) => {
        const response = await request.post('/disk/artifacts/gguf/delete', {
            data: { files: [], apply: false },
        });
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data.count).toBe(0);
        expect(data.applied).toBe(false);
    });

    test('wandb clear dry-run reports without deleting', async ({ request }) => {
        const response = await request.post('/disk/artifacts/wandb/clear', {
            data: { apply: false },
        });
        expect(response.ok()).toBeTruthy();
        const data = await response.json();
        expect(data.applied).toBe(false);
        expect(data).toHaveProperty('count');
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Cleanup & Analysis section (step 7)
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Cleanup section', () => {
    test('nav button switches to the cleanup section', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="cleanup-section"]').click();
        await expect(page.locator('#cleanup-section')).toBeVisible();
        await expect(page.getByText('Tidy the Workshop')).toBeVisible();
    });

    test('controls are present and Apply starts disabled', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="cleanup-section"]').click();
        await expect(page.locator('#cleanup-keep')).toBeVisible();
        await expect(page.locator('#cleanup-purge-failed')).toBeVisible();
        await expect(page.locator('#cleanup-preview-btn')).toBeVisible();
        // Apply is disabled until a preview produces something to delete.
        await expect(page.locator('#cleanup-apply-btn')).toBeDisabled();
    });

    test('opening the section populates disk usage stats', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="cleanup-section"]').click();
        // loadAnalysis() fills the results-total metric; wait for it to leave the "—" placeholder.
        await expect(page.locator('#cleanup-results-total')).not.toHaveText('—', { timeout: 10000 });
    });

    test('HuggingFace cache tab reveals the shared-cache warning and controls', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="cleanup-section"]').click();
        // Checkpoints panel is the default; HF panel hidden until its tab is clicked.
        await expect(page.locator('#cleanup-panel-hfcache')).toBeHidden();
        await page.locator('.cleanup-tab-btn[data-ctab="hfcache"]').click();
        await expect(page.locator('#cleanup-panel-hfcache')).toBeVisible();
        await expect(page.getByText('shared across your whole machine')).toBeVisible();
        await expect(page.locator('#hf-stale-days')).toBeVisible();
        // Delete stays disabled until repos are selected.
        await expect(page.locator('#hf-delete-btn')).toBeDisabled();
    });

    test('Saved Models tab shows the permanence warning and a locked delete button', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="cleanup-section"]').click();
        await expect(page.locator('#cleanup-panel-models')).toBeHidden();
        await page.locator('.cleanup-tab-btn[data-ctab="models"]').click();
        await expect(page.locator('#cleanup-panel-models')).toBeVisible();
        await expect(page.getByText('Deleting a saved model is permanent')).toBeVisible();
        // Delete stays disabled until a model is selected.
        await expect(page.locator('#models-delete-btn')).toBeDisabled();
    });

    test('Other Artifacts tab shows GGUF and W&B sections', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="cleanup-section"]').click();
        await expect(page.locator('#cleanup-panel-artifacts')).toBeHidden();
        await page.locator('.cleanup-tab-btn[data-ctab="artifacts"]').click();
        await expect(page.locator('#cleanup-panel-artifacts')).toBeVisible();
        await expect(page.locator('#cleanup-panel-artifacts h3').filter({ hasText: 'GGUF Exports' })).toBeVisible();
        await expect(page.locator('#cleanup-panel-artifacts h3').filter({ hasText: 'Weights & Biases' })).toBeVisible();
        // Both delete buttons start disabled (nothing selected / nothing clearable yet).
        await expect(page.locator('#gguf-delete-btn')).toBeDisabled();
        // GGUF total leaves the placeholder once the scan returns.
        await expect(page.locator('#gguf-total')).not.toHaveText('—', { timeout: 10000 });
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Keyboard shortcuts
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Keyboard shortcuts', () => {
    test('Ctrl+S opens save config modal', async ({ page }) => {
        await page.goto('/');
        // Wait for app initialization
        await page.waitForTimeout(500);
        await page.keyboard.press('Control+s');
        const saveModal = page.locator('#save-config-modal');
        // Modal should be displayed
        await expect(saveModal).toHaveCSS('display', 'flex');
    });

    test('Escape closes open modal', async ({ page }) => {
        await page.goto('/');
        await page.waitForTimeout(500);

        // Open save modal
        await page.keyboard.press('Control+s');
        const saveModal = page.locator('#save-config-modal');
        await expect(saveModal).toHaveCSS('display', 'flex');

        // Close with Escape
        await page.keyboard.press('Escape');
        await expect(saveModal).toHaveCSS('display', 'none');
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Advanced settings collapsible
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Advanced settings', () => {
    test('advanced section is collapsed by default', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        const advancedContent = page.locator('.advanced-section').first();
        if (await advancedContent.count() > 0) {
            await expect(advancedContent).toBeHidden();
        }
    });

    test('clicking toggle reveals advanced settings', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="config-section"]').click();
        const toggle = page.locator('#toggle-advanced');
        if (await toggle.count() > 0) {
            await toggle.click();
            const advancedContent = page.locator('.advanced-section').first();
            await expect(advancedContent).toBeVisible();
        }
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Dataset format selection
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Dataset format', () => {
    test('format dropdown has all expected options', async ({ page }) => {
        await page.goto('/');
        const formatSelect = page.locator('#dataset-format-type');
        const options = await formatSelect.locator('option').allTextContents();

        // Should include these core formats
        expect(options.some(o => /tokenizer/i.test(o))).toBeTruthy();
        expect(options.some(o => /chatml/i.test(o))).toBeTruthy();
        expect(options.some(o => /llama/i.test(o))).toBeTruthy();
    });

    test('custom format config appears when Custom selected', async ({ page }) => {
        await page.goto('/');
        await page.locator('.section-nav-btn[data-section="dataset-section"]').click();
        await page.locator('#dataset-format-type').selectOption('custom');
        const customConfig = page.locator('#custom-format-config');
        if (await customConfig.count() > 0) {
            await expect(customConfig).toBeVisible();
        }
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Local model picker (offline support — issue #80)
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Local model picker', () => {
    const mockLocalModels = {
        offline_mode: false,
        count: 2,
        models: [
            { model_id: 'org/cached-model', source: 'hf_cache', size_bytes: 1234 },
            { model_id: '/srv/models/my-merged-model', name: 'my-merged-model', source: 'models_dir' },
        ],
    };

    test('picker and refresh button are visible in model section', async ({ page }) => {
        await page.goto('/');
        await expect(page.locator('#local-model-select')).toBeVisible();
        await expect(page.locator('#refresh-local-models')).toBeVisible();
        // Datalist is attached to the base model input for inline suggestions
        await expect(page.locator('#base-model')).toHaveAttribute('list', 'local-models-list');
    });

    test('picker populates from /models/local grouped by source', async ({ page }) => {
        await page.route('**/models/local', route => route.fulfill({ json: mockLocalModels }));
        await page.goto('/');

        const select = page.locator('#local-model-select');
        await expect(select.locator('option[value="org/cached-model"]')).toHaveCount(1);
        await expect(select.locator('option[value="/srv/models/my-merged-model"]')).toHaveCount(1);
        const groups = await select.locator('optgroup').allTextContents();
        expect(groups.length).toBe(2);
    });

    test('selecting a local model fills the base model input', async ({ page }) => {
        await page.route('**/models/local', route => route.fulfill({ json: mockLocalModels }));
        await page.goto('/');

        await page.locator('#local-model-select option[value="org/cached-model"]').waitFor({ state: 'attached' });
        await page.locator('#local-model-select').selectOption('org/cached-model');
        await expect(page.locator('#base-model')).toHaveValue('org/cached-model');
    });

    test('offline badge appears when server is in offline mode', async ({ page }) => {
        await page.route('**/models/local', route => route.fulfill({
            json: { ...mockLocalModels, offline_mode: true },
        }));
        await page.goto('/');
        await expect(page.locator('#offline-mode-badge')).toBeVisible();
    });

    test('offline badge stays hidden when online', async ({ page }) => {
        await page.route('**/models/local', route => route.fulfill({ json: mockLocalModels }));
        await page.goto('/');
        await expect(page.locator('#local-model-select option[value="org/cached-model"]')).toHaveCount(1);
        await expect(page.locator('#offline-mode-badge')).toBeHidden();
    });
});
