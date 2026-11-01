import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { switchCopyPlayerFixture } from '../switchCopyFixture';
import { decodeProject, encodeProject } from '../../src/model/project';
import { enginePlayerInventory, engineSwitchState } from '../engineCapture';

test('switch inspectors explain condition numbers and edit independent reset modes with native timer activity', async ({ page }, testInfo) => {
  const { document, switchCopy: copy, source } = switchCopyPlayerFixture();
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'switch-settings.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(source.zone).padStart(3, '0')}(?:\\s|$)`) }).click();
  const originalMode = page.getByRole('combobox', { name: `switch ${source.id + 1} reset mode`, exact: true });
  const initialMode = source.type ? 'automatic' : 'manual';
  await expect(originalMode).toHaveValue(initialMode);
  await originalMode.selectOption(source.type ? 'manual' : 'automatic');
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(originalMode).toHaveValue(initialMode);
  await page.getByRole('button', { name: new RegExp(`^Room ${copy.room.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  const section = page.getByRole('region', { name: `Copied switch ${copy.slot + 1}`, exact: true });
  await expect(section).toContainText(`Uses Switch ${copy.slot + 1} in this level's requirements`);
  await section.getByRole('button', { name: 'Show switch graphics room', exact: true }).click();
  await expect(section).toBeVisible(); await expect(page.locator('.dirty-dot')).toHaveCount(0);
  const mode = section.getByRole('combobox', { name: `copied switch ${copy.slot + 1} reset mode`, exact: true });
  await mode.selectOption('automatic');
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(mode).toHaveValue(initialMode);
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(mode).toHaveValue('automatic');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('switch-reset-settings.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.files).toEqual(document.files);
  expect(saved.roomSource.switchCopies![0]!.bytes).toEqual(document.roomSource.switchCopies![0]!.bytes.map((byte, index) => index === 2 ? 1 : byte));
  await page.screenshot({ path: testInfo.outputPath('switch-reset-inspector.png'), fullPage: true });
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const initial = await engineSwitchState(page, source.id);
  await expect.poll(async () => (await engineSwitchState(page, source.id)).phase).toBe(initial.phase & 0x7c);
  const original = await engineSwitchState(page, source.id);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus();
  await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(1 << (copy.slot + 4)); }
  finally { await page.keyboard.up('Space'); }
  await expect.poll(async () => { const state = await engineSwitchState(page, copy.slot); return state.active && state.countdown !== 0; }).toBe(true);
  expect(await engineSwitchState(page, source.id)).toEqual(original);
});
