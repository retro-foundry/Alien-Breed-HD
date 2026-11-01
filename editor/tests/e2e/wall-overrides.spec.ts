import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { decodeProject, encodeProject, openArchive, openProjectCandidate } from '../../src/model/project';
import { wallOverrideFixture } from '../wallOverrideFixture';
import { captureEngine } from '../engineCapture';

test('native material replacements survive save/export/recovery and restore the actual game texture through undo and redo', async ({ page }, testInfo) => {
  test.setTimeout(120000); page.on('dialog', dialog => void dialog.accept());
  const fixture = wallOverrideFixture(), preview = page.getByRole('region', { name: '3D game engine preview' });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').first().setInputFiles({ name: 'wall-assets.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  const selectRoom = () => page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  await selectRoom(); await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const camera = { zone: Number(await preview.getAttribute('data-spawn-zone')), x: 20128, z: 20128, upper: 0, angle: 0 };
  const original = await captureEngine(page, camera);
  const tools = async () => {
    const browse = page.getByText('Browse lower wall materials', { exact: true });
    if (!await browse.locator('..').evaluate(node => (node as HTMLDetailsElement).open)) await browse.click();
    const native = page.getByText('Native sheet tools', { exact: true }); await expect(native).toBeVisible();
    if (!await native.locator('..').evaluate(node => (node as HTMLDetailsElement).open)) await native.click();
  };
  await tools();
  const input = page.getByLabel('Import lower wall native sheet', { exact: true });
  await input.setInputFiles({ name: 'broken.wad', mimeType: 'application/octet-stream', buffer: Buffer.from([1, 2, 3]) });
  await expect(page.getByRole('alert')).toContainText('damaged');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await input.setInputFiles({ name: 'replacement.wad', mimeType: 'application/octet-stream', buffer: Buffer.from(fixture.replacement) });
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Restore original sheet', exact: true })).toBeEnabled();
  await expect(preview).toHaveAttribute('data-level-slot', '0');
  const changed = await captureEngine(page, camera); expect(changed).not.toBe(original);
  const save = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const sourcePath = testInfo.outputPath('wall-assets.ab3d'); await (await save).saveAs(sourcePath);
  const saved = decodeProject(new Uint8Array(await readFile(sourcePath)));
  expect(saved.files).toEqual(fixture.document.files);
  expect(saved.wallOverrides).toEqual([{ texture: fixture.sheet.id, bytes: fixture.replacement }]);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const gamePath = testInfo.outputPath('wall-assets-game.zip'); await (await download).saveAs(gamePath);
  const game = openProjectCandidate(openArchive(new Uint8Array(await readFile(gamePath)), 'game.zip')[0]!);
  expect(game.wallOverrides).toEqual(saved.wallOverrides);
  for (let iteration = 0; iteration < 2; iteration++) {
    await page.getByRole('button', { name: 'Restore original sheet', exact: true }).click();
    await expect(preview).toHaveAttribute('data-level-slot', '0'); expect(await captureEngine(page, camera)).toBe(original);
    await page.getByRole('button', { name: 'Undo', exact: true }).click();
    await expect(preview).toHaveAttribute('data-level-slot', '0'); expect(await captureEngine(page, camera)).toBe(changed);
    await page.getByRole('button', { name: 'Redo', exact: true }).click();
    await expect(preview).toHaveAttribute('data-level-slot', '0'); expect(await captureEngine(page, camera)).toBe(original);
    await page.getByRole('button', { name: 'Undo', exact: true }).click();
    await expect(preview).toHaveAttribute('data-level-slot', '0'); expect(await captureEngine(page, camera)).toBe(changed);
  }
  await page.mouse.move(0, 0); await page.screenshot({ path: testInfo.outputPath('project-wall-override.png') });
  await expect(page.getByText('Workspace saved', { exact: true })).toBeVisible();
  await page.reload(); await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await selectRoom(); await page.getByRole('button', { name: 'Split view', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await captureEngine(page, camera)).toBe(changed);
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  // Opening a project without overrides must restore packaged assets too.
  await page.locator('input[type=file]').first().setInputFiles({ name: 'original.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject({ ...fixture.document, name: 'Original material project' })) });
  await expect(page.locator('.project-picker strong')).toHaveText('Original material project');
  await selectRoom(); await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await captureEngine(page, camera)).toBe(original);
});
