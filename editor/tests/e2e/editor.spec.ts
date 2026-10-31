import { expect, test } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { openArchive } from '../../src/model/project';
import { openLevel } from '../../src/model/level';
import oracles from '../fixtures/original-levels.json' with { type: 'json' };

test('a cancelled backup cannot mark edits saved when recovery storage is unavailable', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(indexedDB, 'open', { value: () => { throw new DOMException('Recovery storage is unavailable', 'QuotaExceededError'); } });
  });
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  const brightness = page.getByRole('spinbutton', { name: 'Brightness', exact: true });
  await brightness.fill('11'); await brightness.press('Tab');
  await expect(page.locator('.dirty-dot')).toBeVisible();
  const backup = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save project', exact: true }).click();
  await (await backup).cancel();
  await expect(page.locator('.dirty-dot')).toBeVisible();
  await expect(page.getByText('Autosave unavailable', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => {
    const event = new Event('beforeunload', { cancelable: true });
    window.dispatchEvent(event); return event.defaultPrevented;
  })).toBe(true);
  await expect(brightness).toHaveValue('11');
});

test('saving a focused numeric field includes the latest edit', async ({ page }, testInfo) => {
  page.on('dialog', dialog => void dialog.accept());
  await page.goto('/');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  const brightness = page.getByRole('spinbutton', { name: 'Brightness', exact: true });
  await brightness.fill('42');
  const saved = page.waitForEvent('download');
  await brightness.press('Control+s');
  const path = testInfo.outputPath('focused-field.ab3d');
  await (await saved).saveAs(path);
  const project = openArchive(new Uint8Array(await readFile(path)), 'focused-field.ab3d')[0]!;
  expect(openLevel(project.name, project.files).zones[3]?.brightness).toBe(42);
  await brightness.fill('1000'); await brightness.press('Control+s');
  await expect(brightness).toHaveValue('1000');
  expect(await brightness.evaluate((input: HTMLInputElement) => input.checkValidity())).toBe(false);
  await brightness.press('Escape');
  await expect(brightness).toHaveValue('42');
});

test('the project navigator and search shortcut remain available in a narrow window', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 760, height: 900 });
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Toggle project navigator' }).click();
  await page.getByRole('combobox', { name: 'Original level' }).selectOption('7');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 7');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  await expect(page.locator('.navigator')).toBeHidden();
  await page.keyboard.press('/');
  await expect(page.getByRole('textbox', { name: 'Search map contents' })).toBeFocused();
  await page.keyboard.type('12');
  await expect(page.getByRole('button', { name: /^Zone 012/ })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('narrow-editor.png'), fullPage: true });
});

test('real maps, editing, undo, portable saves, recovery and rejected imports', async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('dialog', dialog => void dialog.accept());
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await expect(page.locator('canvas[aria-label="Level map"]')).toBeVisible();
  await expect(page.locator('.status-size')).toHaveText('304 points · 138 zones');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  const brightness = page.getByRole('spinbutton', { name: 'Brightness', exact: true });
  await expect(brightness).toHaveValue('-15');
  await brightness.fill('6'); await brightness.press('Tab');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(brightness).toHaveValue('-15');
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await expect(brightness).toHaveValue('6');
  const saved = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const savedPath = testInfo.outputPath('edited.ab3d');
  await (await saved).saveAs(savedPath);
  await page.getByRole('combobox', { name: 'Original level' }).selectOption('2');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 2');
  await page.locator('input[type=file]').setInputFiles(savedPath);
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  await expect(brightness).toHaveValue('6');

  await page.locator('input[type=file]').setInputFiles(['twolev.bin', 'twolev.graph.bin', 'twolev.clips'].map(name => ({ name, mimeType: 'application/octet-stream', buffer: Buffer.from([0, 1]) })));
  await expect(page.getByRole('alert')).toContainText('record extends past');
  await expect(brightness).toHaveValue('6');
  await page.getByRole('button', { name: 'Dismiss error' }).click();
  await brightness.fill('7'); await brightness.press('Tab');
  await expect(page.getByText('Workspace saved', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  await expect(brightness).toHaveValue('7');

  const exported = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const exportedPath = testInfo.outputPath('edited-level.zip');
  await (await exported).saveAs(exportedPath);
  const project = openArchive(new Uint8Array(await readFile(exportedPath)), 'edited-level.zip')[0]!;
  expect(openLevel(project.name, project.files).zones[3]?.brightness).toBe(7);
  expect(errors).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath('editor.png'), fullPage: true });
});

test('an invalid player position reports an error without changing the map or crashing', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('tab', { name: 'Starts', exact: true }).click();
  await page.getByRole('button', { name: 'Player 1', exact: false }).click();
  await page.getByRole('spinbutton', { name: 'Position X' }).fill('32000');
  await page.getByRole('spinbutton', { name: 'Position Z', exact: true }).fill('32000');
  await page.getByRole('button', { name: 'Apply position' }).click();
  await expect(page.getByRole('alert')).toContainText('inside');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  expect(errors).toEqual([]);
});

test('all sixteen shipped maps render through the real browser interface', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  for (const oracle of oracles.levels) {
    await page.getByRole('combobox', { name: 'Original level' }).selectOption(String(oracle.level));
    await expect(page.locator('.project-picker strong')).toHaveText(`Level ${oracle.level}`);
    const points = (oracle.files['twolev.bin'].header[7] ?? 0) + 1;
    const zones = ((oracle.files['twolev.graph.bin'].header[3] ?? 0) - 16) / 4;
    await expect(page.locator('.status-size')).toHaveText(`${points} points · ${zones} zones`);
    await expect(page.getByRole('alert')).toHaveCount(0);
  }
  expect(errors).toEqual([]);
});

test('an older pending reference load cannot overwrite a newer imported document', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  let complete: (() => void) | undefined;
  const gate = new Promise<void>(resolve => { complete = resolve; });
  await page.route('**/reference/level_b/*', async route => { await gate; await route.continue(); });
  await page.getByRole('combobox', { name: 'Original level' }).selectOption('2');
  const folder = new URL('../../../amiga/data/levels/level_c/', import.meta.url);
  await page.locator('input[type=file]').setInputFiles(['twolev.bin', 'twolev.graph.bin', 'twolev.clips'].map(name => fileURLToPath(new URL(name, folder))));
  await expect(page.locator('.project-picker strong')).toHaveText('Imported level');
  complete?.();
  await page.waitForLoadState('networkidle');
  await expect(page.locator('.project-picker strong')).toHaveText('Imported level');
  await expect(page.locator('.status-size')).toHaveText('346 points · 160 zones');
});
