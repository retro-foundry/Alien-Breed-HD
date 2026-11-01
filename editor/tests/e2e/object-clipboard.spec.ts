import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { objectStoreyFixture } from '../objectStoreyFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { engineObjectState } from '../engineCapture';

for (const count of [1, 2]) test(`copy/paste captures ${count} objects independently of deleted originals and undoes atomically`, async ({ page }, testInfo) => {
  const { document, ids } = objectStoreyFixture();
  const selected = ids.slice(0, count), firstNew = document.objects.length;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'clipboard.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('tab', { name: 'Objects', exact: true }).click();
  for (const [index, id] of selected.entries()) await page.getByRole('button', { name: `Health ${id}`, exact: true }).click(index ? { modifiers: ['Shift'] } : {});
  await page.keyboard.press('Control+c');
  await expect(page.getByRole('button', { name: 'Paste objects', exact: true })).toBeEnabled();
  await page.keyboard.press('Delete');
  await page.keyboard.press('Control+v');
  const pasted = selected.map((_, index) => firstNew + index);
  for (const id of pasted) await expect(page.getByRole('button', { name: `Health ${id}`, exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  for (const id of pasted) await expect(page.getByRole('button', { name: `Health ${id}`, exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('pasted.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  for (const [index, id] of pasted.entries()) {
    const original = document.objects[selected[index]!]!, copy = saved.objects[id]!;
    expect(copy).toMatchObject({ type: original.type, y: original.y, upper: original.upper, health: original.health });
    expect(copy.cid).not.toBe(original.cid);
    expect(saved.sourceObjects!.find(object => object.id === selected[index])!.room).toBeNull();
    expect(saved.sourceObjects!.find(object => object.id === id)!.room).not.toBeNull();
  }
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  for (const [index, id] of pasted.entries()) expect(await engineObjectState(page, id)).toMatchObject({ type: document.objects[selected[index]!]!.type, yFixed: document.objects[selected[index]!]!.y * 256 });
});
