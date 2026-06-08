// Playwright configuration for Merlina frontend integration tests
import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..', '..');

export default defineConfig({
    testDir: '.',
    testMatch: 'test_integration*.spec.mjs',
    timeout: 15000,
    retries: process.env.CI ? 0 : 1,
    workers: 1,  // Sequential — single FastAPI server

    use: {
        baseURL: 'http://localhost:8000',
        navigationTimeout: 10000,
        screenshot: 'only-on-failure',
        trace: 'retain-on-failure',
        launchOptions: {
            executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
            args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--headless=new'],
        },
    },

    webServer: {
        command: `python ${path.join(rootDir, 'tests', 'frontend', 'serve_for_tests.py')}`,
        cwd: rootDir,
        url: 'http://localhost:8000/version',
        reuseExistingServer: !process.env.CI,
        timeout: 30000,
        stdout: 'pipe',
        stderr: 'pipe',
    },

    reporter: process.env.CI
        ? [['github'], ['html', { open: 'never' }]]
        : [['list']],
});
