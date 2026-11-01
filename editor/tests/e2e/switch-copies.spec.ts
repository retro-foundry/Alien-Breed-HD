import { expect, test } from '@playwright/test';
import { switchCopyPlayerFixture } from '../switchCopyFixture';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { decodeProject, encodeProject } from '../../src/model/project';
import { readFile } from 'node:fs/promises';
import { enginePlayerInventory, engineSwitchState } from '../engineCapture';

test('using a copied native switch changes its own condition bit and wall without toggling the original', async ({ page }, testInfo) => {
  const { document, switchCopy: copy, source } = switchCopyPlayerFixture();
  const built = buildPlayableLevel(document, -1);
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'switch-copy.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const initial = await engineSwitchState(page, source.id);
  await expect.poll(async () => (await engineSwitchState(page, source.id)).phase).toBe(initial.phase & 0x7c);
  const original = await engineSwitchState(page, source.id), copied = await engineSwitchState(page, copy.slot);
  expect(original.active).toBe(false); expect(copied.active).toBe(false);
  expect((await enginePlayerInventory(page)).conditions).toBe(0);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus();
  await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(1 << (copy.slot + 4)); }
  finally { await page.keyboard.up('Space'); }
  expect(await engineSwitchState(page, copy.slot)).toMatchObject({ active: true, phase: (copied.phase & 0x7c) | 2 });
  expect(await engineSwitchState(page, source.id)).toEqual(original);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('switch-source-after-play.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.files).toEqual(document.files); expect(saved.sourceStarts).toEqual(document.sourceStarts);
  expect(saved.roomSource.switchCopies).toEqual(document.roomSource.switchCopies);
  const oldFrame = await page.locator('iframe[title="Alien Breed game engine"]').elementHandle();
  await page.locator('input[type=file]').setInputFiles(path);
  await page.waitForFunction(element => !element!.isConnected, oldFrame);
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await engineSwitchState(page, copy.slot)).toMatchObject({ active: false, zone: built.document.switches[copy.slot]!.zone });
});
