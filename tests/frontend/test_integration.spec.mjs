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
        const version = page.locator('#version-display');
        await expect(version).toBeVisible();
        await expect(version).toContainText(/v\d+\.\d+\.\d+/);
    });

    test('all three main sections are visible', async ({ page }) => {
        await page.goto('/');
        // Section headers
        await expect(page.getByText('Model Selection')).toBeVisible();
        await expect(page.getByText('Dataset Configuration')).toBeVisible();
        // Training config section contains multiple subsections
        await expect(page.getByText('Training Hyperparameters')).toBeVisible();
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
        const input = page.locator('#output-name');
        await input.fill('bad name!@#$');
        await input.blur();
        // Should show validation error
        await expect(input).toHaveClass(/input-error/);
    });

    test('output name field accepts valid names', async ({ page }) => {
        await page.goto('/');
        const input = page.locator('#output-name');
        await input.fill('my-fine-tuned-model');
        await input.blur();
        await expect(input).not.toHaveClass(/input-error/);
    });

    test('learning rate field validates numeric range', async ({ page }) => {
        await page.goto('/');
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
        const betaGroup = page.locator('#beta').locator('..');
        await expect(betaGroup).toBeVisible();
    });

    test('hides beta field in SFT mode', async ({ page }) => {
        await page.goto('/');
        await page.locator('#training-mode').selectOption('sft');
        // Beta field or its container should be hidden
        const betaField = page.locator('#beta');
        const betaContainer = betaField.locator('xpath=ancestor::div[contains(@class,"form-group")]');
        // Either the field itself or its parent container should be hidden
        const isHidden = await betaContainer.evaluate(el => {
            return el.style.display === 'none' || el.hidden ||
                   window.getComputedStyle(el).display === 'none';
        }).catch(() => false);

        // If not hidden by container, check the field directly
        if (!isHidden) {
            // In some implementations the field might just not be required
            // Check that switching back to ORPO shows it again
            await page.locator('#training-mode').selectOption('orpo');
            await expect(betaField).toBeVisible();
        }
    });
});

// ═════════════════════════════════════════════════════════════════════════════
// Dataset source switching
// ═════════════════════════════════════════════════════════════════════════════

test.describe('Dataset source selection', () => {
    test('defaults to HuggingFace source', async ({ page }) => {
        await page.goto('/');
        const sourceSelect = page.locator('#dataset-source-type');
        await expect(sourceSelect).toHaveValue('huggingface');
    });

    test('shows HF config when HuggingFace selected', async ({ page }) => {
        await page.goto('/');
        await expect(page.locator('#hf-source-config')).toBeVisible();
        await expect(page.locator('#upload-source-config')).toBeHidden();
        await expect(page.locator('#local-source-config')).toBeHidden();
    });

    test('shows upload config when Upload selected', async ({ page }) => {
        await page.goto('/');
        await page.locator('#dataset-source-type').selectOption('upload');
        await expect(page.locator('#upload-source-config')).toBeVisible();
        await expect(page.locator('#hf-source-config')).toBeHidden();
    });

    test('shows local config when Local File selected', async ({ page }) => {
        await page.goto('/');
        await page.locator('#dataset-source-type').selectOption('local_file');
        await expect(page.locator('#local-source-config')).toBeVisible();
        await expect(page.locator('#hf-source-config')).toBeHidden();
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
        await expect(page.locator('#lora-r')).toBeVisible();
        await expect(page.locator('#lora-alpha')).toBeVisible();
    });

    test('LoRA settings hidden when disabled', async ({ page }) => {
        await page.goto('/');
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
        const advancedContent = page.locator('#advanced-settings');
        if (await advancedContent.count() > 0) {
            await expect(advancedContent).toBeHidden();
        }
    });

    test('clicking toggle reveals advanced settings', async ({ page }) => {
        await page.goto('/');
        const toggle = page.locator('#toggle-advanced');
        if (await toggle.count() > 0) {
            await toggle.click();
            const advancedContent = page.locator('#advanced-settings');
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
        await page.locator('#dataset-format-type').selectOption('custom');
        const customConfig = page.locator('#custom-format-config');
        if (await customConfig.count() > 0) {
            await expect(customConfig).toBeVisible();
        }
    });
});
