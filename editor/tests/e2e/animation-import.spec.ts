import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { openLevel, parseLevel } from '../../src/model/level';
import { encodeProject, decodeProject } from '../../src/model/project';
import { originalFiles } from '../helpers';

test('unsafe animation imports preserve the current map, recovery and live engine', async ({ page }, testInfo) => {
  page.on('dialog', dialog => void dialog.accept());
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  const original = openLevel('Level 1', originalFiles(1)), graphics = original.files.graphics.slice();
  const active = original.switches.find(item => item.zone >= 0)!;
  new DataView(graphics.buffer).setInt32(active.offset + 6, graphics.length + 1024);
  // The low-level parser is also used to inspect damaged files; openLevel is
  // the validated user import boundary and must reject this archive.
  const unsafe = parseLevel('Unsafe animation', { ...original.files, graphics });
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toHaveText('138 zones loaded · Game engine', { timeout: 60000 });
  const revision = await preview.getAttribute('data-revision');
  const inputs = [
    [{ name: 'unsafe-animation.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(unsafe)) }],
    [{ name: 'twolev.bin', mimeType: 'application/octet-stream', buffer: Buffer.from(original.files.data) },
      { name: 'twolev.graph.bin', mimeType: 'application/octet-stream', buffer: Buffer.from(graphics) },
      { name: 'twolev.clips', mimeType: 'application/octet-stream', buffer: Buffer.from(original.files.clips) }],
  ];
  for (const files of inputs) {
    await page.locator('input[type=file]').setInputFiles(files);
    await expect(page.getByRole('alert')).toContainText('invalid animation graphics target');
    await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
    await expect(preview).toHaveAttribute('data-revision', revision!);
    await expect(preview.getByRole('alert')).toHaveCount(0);
  }
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('retained-level.ab3d'); await (await download).saveAs(path);
  expect(decodeProject(new Uint8Array(await readFile(path))).files).toEqual(original.files);
  await expect(page.getByText('Workspace saved', { exact: true })).toBeVisible();
  await page.reload(); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  expect(errors).toEqual([]);
});
