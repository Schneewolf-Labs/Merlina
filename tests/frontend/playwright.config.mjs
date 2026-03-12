// Playwright configuration for Merlina frontend integration tests
import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: '.',
    testMatch: 'test_integration*.spec.mjs',
    timeout: 30000,
    retries: 1,
    workers: 1,  // Sequential — single FastAPI server

    use: {
        baseURL: 'http://localhost:8000',
        screenshot: 'only-on-failure',
        trace: 'retain-on-failure',
    },

    webServer: {
        command: 'python ../../merlina.py',
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
