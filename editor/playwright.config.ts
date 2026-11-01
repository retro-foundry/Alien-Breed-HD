import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: false,
  workers: 1,
  timeout: 45000,
  expect: { timeout: 10000 },
  reporter: 'list',
  use: { baseURL: 'http://127.0.0.1:4174', viewport: { width: 1440, height: 950 }, trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'], launchOptions: {
      // Permit the real ANGLE software renderer on GPU-less Windows runners.
      // https://bugzilla.mozilla.org/show_bug.cgi?id=1968347
      firefoxUserPrefs: { 'webgl.force-enabled': true },
    } } },
  ],
  webServer: { command: 'npm run build && npm run preview -- --port 4174 --strictPort', url: 'http://127.0.0.1:4174', reuseExistingServer: false },
});
