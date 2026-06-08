// Playwright configuration for Merlina frontend integration tests
import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '..', '..');

// Resolve a browser executable only when the env var or well-known fallback
// path actually exists — otherwise leave executablePath unset so playwright
// uses its own bundled browser.
function resolveExecutablePath() {
    const candidates = [
        process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
        '/opt/pw-browsers/chromium-1194/chrome-linux/chrome',
    ].filter(Boolean);
    for (const p of candidates) {
        if (fs.existsSync(p)) return p;
    }
    return undefined;
}

const executablePath = resolveExecutablePath();

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
            ...(executablePath ? { executablePath } : {}),
            args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
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
