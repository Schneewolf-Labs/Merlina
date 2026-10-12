import { test, expect } from '@playwright/test';

test.describe('Studio workflow', () => {
    test('studio typography is served locally and loads successfully', async ({ page }) => {
        await page.goto('/');
        const fonts = await page.evaluate(async () => {
            await document.fonts.ready;
            return [...document.fonts].map(font => ({ name: font.family, status: font.status }));
        });
        expect(fonts).toEqual(expect.arrayContaining([
            { name: 'Nunito', status: 'loaded' },
            { name: 'Nunito Sans', status: 'loaded' },
        ]));
    });

    test('optional model access and compute controls expand on demand', async ({ page }) => {
        await page.goto('/');
        await expect(page.locator('#hf-token-preload')).toBeHidden();
        await page.locator('.model-access summary').click();
        await expect(page.getByLabel('HuggingFace Token', { exact: true }).first()).toBeVisible();
        await expect(page.locator('#gpu-selection')).toBeHidden();
        await page.locator('#compute-settings summary').click();
        await expect(page.locator('#gpu-selection')).toBeVisible();
        await page.locator('#refresh-gpu-button').click();
        await expect(page.locator('#gpu-list-container')).toContainText('No CUDA Available');
        await page.locator('#multi-gpu-strategy').selectOption('single');
        await page.locator('#compute-settings summary').click();
        await page.locator('#compute-settings summary').click();
        await expect(page.locator('#multi-gpu-strategy')).toHaveValue('single');
    });

    test('recipe summary responds to programmatic configuration changes', async ({ page }) => {
        await page.goto('/');
        await page.evaluate(() => {
            const base = document.getElementById('base-model');
            base.value = 'saved/model';
            base.dispatchEvent(new Event('change'));
            const lora = document.getElementById('use-lora');
            lora.checked = false;
            lora.dispatchEvent(new Event('change'));
        });
        await expect(page.locator('#summary-model')).toHaveText('saved/model');
        await expect(page.locator('#summary-adaptation')).toHaveText('Full fine-tuning');
    });

    test('setup actions keep form values and the live recipe in sync', async ({ page }) => {
        await page.goto('/');
        await page.locator('#base-model').fill('my-org/my-model');
        await expect(page.locator('#summary-model')).toHaveText('my-org/my-model');
        await page.getByRole('button', { name: 'Continue to dataset' }).click();
        await expect(page.locator('#dataset-section')).toBeVisible();
        await expect(page.locator('#dataset-section h2')).toBeFocused();
        await page.locator('#training-mode').selectOption('sft');
        await page.locator('#add-dataset-btn').click();
        await expect(page.locator('#summary-datasets')).toHaveText('2 datasets');
        await expect(page.locator('#summary-method')).toHaveText('SFT');
        await page.getByRole('button', { name: 'Continue to training' }).click();
        await expect(page.locator('#training-mode-config')).toHaveValue('sft');
        await page.locator('#use-lora').uncheck();
        await expect(page.locator('#summary-adaptation')).toHaveText('Full fine-tuning');
        await page.locator('[data-section="model-section"]').click();
        await expect(page.locator('#base-model')).toHaveValue('my-org/my-model');
    });

    test('deep links and browser history restore the active workspace', async ({ page }) => {
        await page.goto('/#export-section');
        await expect(page.locator('#export-section')).toBeVisible();
        await expect(page.locator('#setup-summary')).toBeHidden();
        await expect(page.locator('[data-section="export-section"]')).toHaveAttribute('aria-current', 'page');
        await page.locator('[data-section="inference-section"]').click();
        await expect(page.locator('#current-page-label')).toHaveText('Playground');
        await page.goBack();
        await expect(page.locator('#export-section')).toBeVisible();
        await page.reload();
        await expect(page.locator('#export-section')).toBeVisible();
    });

    test('jobs empty state leads back into setup', async ({ page }) => {
        await page.route('**/jobs', route => route.fulfill({ json: {} }));
        await page.goto('/#jobs-section');
        await expect(page.getByRole('heading', { name: 'Your next creation starts here' })).toBeVisible();
        await page.getByRole('button', { name: 'Create a training run' }).click();
        await expect(page.locator('#model-section')).toBeVisible();
    });

    for (const theme of ['light', 'dark']) {
        test(`all workspaces fit a phone in ${theme} mode`, async ({ page }) => {
            await page.setViewportSize({ width: 390, height: 844 });
            await page.addInitScript(value => localStorage.setItem('merlina-theme', value), theme);
            await page.goto('/');
            for (const section of ['model', 'dataset', 'config', 'jobs', 'inference', 'export', 'cleanup']) {
                const nav = page.locator(`[data-section="${section}-section"]`);
                await nav.click();
                await expect(nav).toHaveAttribute('aria-current', 'page');
                await expect(page.locator(`#${section}-section`)).toBeVisible();
                expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
            }
        });
    }

    test('keyboard navigation moves focus to the selected workspace heading', async ({ page }) => {
        await page.goto('/');
        await page.locator('[data-section="dataset-section"]').focus();
        await page.keyboard.press('Enter');
        await expect(page.locator('#dataset-section h2')).toBeFocused();
        await expect(page.locator('#model-section')).toBeHidden();
    });

    test('reduced motion disables decorative animation', async ({ page }) => {
        await page.emulateMedia({ reducedMotion: 'reduce' });
        await page.goto('/');
        await page.locator('.wizard-hat').hover();
        expect(await page.locator('.wizard-hat').evaluate(el => getComputedStyle(el).animationName)).toBe('none');
    });
});
