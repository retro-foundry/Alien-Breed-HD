import { expect, test } from '@playwright/test';

test('tooltips explain actions on hover and keyboard focus; help works using the keyboard', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Pan tool', exact: true }).hover();
  await expect(page.getByRole('tooltip')).toContainText('hold Space');
  const grid = page.getByRole('combobox', { name: 'Grid spacing' });
  await grid.focus();
  await expect(page.getByRole('tooltip')).toContainText('Hold Alt');
  await expect(grid).toHaveAttribute('aria-describedby', /.+/);
  await grid.press('Escape');
  await expect(page.getByRole('tooltip')).toHaveCount(0);
  const help = page.getByRole('button', { name: 'Help and shortcuts' });
  await help.focus(); await help.press('Enter');
  const dialog = page.getByRole('dialog', { name: 'Editor help' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText('Crouch', { exact: true })).toBeVisible();
  await expect(dialog.getByText('Use doors and switches', { exact: true })).toBeVisible();
  const close = dialog.getByRole('button', { name: 'Close dialog' });
  await expect(close).toBeFocused();
  await close.press('Tab'); await expect(close).toBeFocused();
  await close.press('Space');
  await expect(dialog).toHaveCount(0);
  await expect(help).toBeFocused();
});

test('3D controls explain the current mode and tooltips stay inside a narrow viewport', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 760, height: 900 });
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Help and shortcuts' }).hover();
  const tooltip = page.getByRole('tooltip');
  await expect(tooltip).toBeVisible();
  const bounds = await tooltip.boundingBox();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(760);
  await page.screenshot({ path: testInfo.outputPath('control-tooltip.png'), fullPage: true });
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const details = page.locator('.engine-key-help');
  await details.locator('summary').click();
  await expect(details.getByText('3D inspection controls', { exact: true })).toBeVisible();
  await expect(details.getByText('Turn the inspection camera', { exact: true })).toBeVisible();
  const play = page.getByRole('button', { name: 'Build & play', exact: true });
  await expect(play).toBeEnabled({ timeout: 60000 }); await play.click();
  await expect(details.getByText('Playtest controls', { exact: true })).toBeVisible();
  await expect(details.getByText('Crouch', { exact: true })).toBeVisible();
  await expect(page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas')).toHaveAttribute('title', /C: Crouch/);
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
});

test('starting play before the initial preview loads survives delayed engine replies', async ({ page }) => {
  await page.addInitScript(() => {
    if (window !== window.top) return;
    const earliestPlay = new MutationObserver(() => {
      const button = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Build & play');
      if (button && !button.disabled) { earliestPlay.disconnect(); button.click(); }
    });
    earliestPlay.observe(document, { childList: true, subtree: true, attributes: true, attributeFilter: ['disabled'] });
    const delayed = new WeakSet<Event>();
    window.addEventListener('message', event => {
      if (delayed.has(event) || event.data?.source !== 'ab3d-engine' || event.data.type !== 'loaded') return;
      // Keep the actual engine load/renderer. Delay delivery of its completed
      // reply past the automatic-preview debounce to exercise the load race.
      event.stopImmediatePropagation();
      const reply = new MessageEvent('message', { data: event.data, origin: event.origin, source: event.source });
      delayed.add(reply); setTimeout(() => window.dispatchEvent(reply), 600);
    }, true);
  });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toHaveText('Playtest running', { timeout: 60000 });
  await expect(preview.getByRole('button', { name: 'Stop', exact: true })).toBeEnabled();
  await preview.locator('.engine-key-help summary').click();
  await expect(preview.getByText('Playtest controls', { exact: true })).toBeVisible();
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
});

test('a rejected play load remains actionable and engine reload can start a fresh playtest', async ({ page }) => {
  await page.addInitScript(() => {
    if (window === window.top) {
      const earliestPlay = new MutationObserver(() => {
        const button = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Build & play');
        if (button && !button.disabled) { earliestPlay.disconnect(); button.click(); }
      });
      earliestPlay.observe(document, { childList: true, subtree: true, attributes: true, attributeFilter: ['disabled'] });
    } else {
      window.addEventListener('message', event => {
        if (event.source !== parent || event.data?.source !== 'ab3d-editor' || event.data.type !== 'load'
          || sessionStorage.getItem('ab3d-editor-test-rejected-load')) return;
        sessionStorage.setItem('ab3d-editor-test-rejected-load', '1');
        // Exercise the real iframe's rejection path, without corrupting level
        // bytes or replacing the native game/renderer with a mock.
        event.data.slot = 16;
      }, true);
    }
  });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('alert')).toContainText('Invalid level slot', { timeout: 60000 });
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await preview.getByRole('button', { name: 'Reload engine', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('138 zones loaded · Game engine', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await expect(preview.getByRole('alert')).toHaveCount(0);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
});

test('a delayed engine error from an older load cannot invalidate a newer preview', async ({ page }) => {
  await page.addInitScript(() => {
    if (window !== window.top) return;
    const released = new WeakSet<Event>();
    window.addEventListener('message', event => {
      const host = window as Window & { releaseOldEngineError?: () => void };
      if (released.has(event) || host.releaseOldEngineError || event.data?.source !== 'ab3d-engine' || event.data.type !== 'error') return;
      event.stopImmediatePropagation();
      const reply = new MessageEvent('message', { data: event.data, origin: event.origin, source: event.source });
      released.add(reply); host.releaseOldEngineError = () => window.dispatchEvent(reply);
    }, true);
  });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toHaveText('138 zones loaded · Game engine', { timeout: 60000 });
  const revision = Number(await preview.getAttribute('data-revision'));
  await page.evaluate(revision => {
    document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!.contentWindow!
      .postMessage({ source: 'ab3d-editor', type: 'load', slot: 16, revision }, location.origin);
  }, revision);
  await page.waitForFunction(() => typeof (window as Window & { releaseOldEngineError?: () => void }).releaseOldEngineError === 'function');
  await page.getByRole('combobox', { name: 'Target game slot' }).selectOption('8');
  await expect(preview).toHaveAttribute('data-level-slot', '8');
  await expect(preview.getByRole('status')).toHaveText('138 zones loaded · Game engine');
  await page.evaluate(() => (window as Window & { releaseOldEngineError?: () => void }).releaseOldEngineError!());
  await expect(preview.getByRole('alert')).toHaveCount(0);
  await expect(preview).toHaveAttribute('data-level-slot', '8');
  await expect(page.frameLocator('iframe[title="Alien Breed game engine"]').locator('#message')).toHaveText('');
});
