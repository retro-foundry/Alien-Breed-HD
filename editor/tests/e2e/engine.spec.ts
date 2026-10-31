import { expect, test } from '@playwright/test';
import oracles from '../fixtures/original-levels.json' with { type: 'json' };

test('the actual engine reloads edits, visits zones and playtests without touching game saves', async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('dialog', dialog => void dialog.accept());
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.evaluate(() => {
    localStorage.setItem('ab3d1.autosave.v1.slot0.data', 'normal-game-save');
    localStorage.setItem('ab3d1.settings.mouse_look', '0');
  });
  await page.getByRole('button', { name: 'Split view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toHaveText('138 zones loaded · Game engine', { timeout: 60000 });
  await expect.poll(async () => Number(await preview.getAttribute('data-rendered-pixels'))).toBeGreaterThan(1000);
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  const brightness = page.getByRole('spinbutton', { name: 'Brightness', exact: true });
  const previousRevision = await preview.getAttribute('data-revision');
  await brightness.fill('10'); await brightness.press('Tab');
  await expect(preview).not.toHaveAttribute('data-revision', previousRevision!);
  await expect(preview.getByRole('status')).toHaveText('138 zones loaded · Game engine');
  await page.getByRole('button', { name: 'Turn preview right' }).click();
  await expect(preview.getByRole('alert')).toHaveCount(0);
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.locator('canvas[aria-label="Level map"]')).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath('engine-preview.png'), fullPage: true });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  // Real keyboard input goes to the actual engine canvas, inside its iframe.
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  await canvas.focus(); await page.keyboard.press('w');
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('138 zones loaded · Game engine');
  await expect(brightness).toHaveValue('10');
  await preview.getByRole('combobox', { name: 'Playtest level slot' }).selectOption({ value: '8' });
  await expect(preview).toHaveAttribute('data-level-slot', '8');
  await expect(preview.getByRole('status')).toHaveText('138 zones loaded · Game engine');
  expect(await page.evaluate(() => localStorage.getItem('ab3d1.autosave.v1.slot0.data'))).toBe('normal-game-save');
  expect(await page.evaluate(() => localStorage.getItem('ab3d1.settings.mouse_look'))).toBe('0');
  expect(errors).toEqual([]);
});

test('all sixteen untouched original maps load and draw in the actual game engine', async ({ page }) => {
  test.setTimeout(150000);
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  for (let level = 1; level <= 16; level++) {
    if (level > 1) await page.getByRole('combobox', { name: 'Original level' }).selectOption({ value: String(level) });
    await expect(page.locator('.project-picker strong')).toHaveText(`Level ${level}`);
    const preview = page.getByRole('region', { name: '3D game engine preview' });
    const zones = oracles.levels[level - 1]!.files['twolev.bin'].header[8]! + 1;
    await expect(preview.getByRole('status')).toHaveText(`${zones} zones loaded · Game engine`, { timeout: 45000 });
    await expect(preview).toHaveAttribute('data-level-slot', String(level - 1));
    await expect.poll(async () => Number(await preview.getAttribute('data-rendered-pixels'))).toBeGreaterThan(1000);
    await expect(preview.getByRole('alert')).toHaveCount(0);
  }
  expect(errors).toEqual([]);
});
