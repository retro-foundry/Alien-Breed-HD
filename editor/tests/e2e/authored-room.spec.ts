import { expect, test } from '@playwright/test';
import { openLevel } from '../../src/model/level';
import { encodeProject } from '../../src/model/project';
import { originalFiles } from '../helpers';
import { buildAuthoredRoom } from '../authoredRoom';

test('a newly assembled room renders and starts actual gameplay with a remapped spawn', async ({ page }, testInfo) => {
  test.setTimeout(90000);
  page.on('dialog', dialog => void dialog.accept());
  const original = openLevel('original', originalFiles(1)), compiled = buildAuthoredRoom(original), id = original.zones.length;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'generated-room.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(compiled)) });
  await expect(page.locator('.project-picker strong')).toHaveText('Generated room');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-spawn-zone', String(id), { timeout: 60000 });
  await expect.poll(async () => Number(await preview.getAttribute('data-rendered-pixels'))).toBeGreaterThan(1000);
  await expect(preview.getByRole('alert')).toHaveCount(0);
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  await expect(async () => {
    await canvas.click();
    await expect(page.getByRole('heading', { name: `Zone ${id}`, exact: true })).toBeVisible();
  }).toPass({ timeout: 10000 });
  await preview.getByRole('button', { name: 'Turn preview right' }).click();
  await page.screenshot({ path: testInfo.outputPath('generated-room.png'), fullPage: true });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await expect(preview).toHaveAttribute('data-spawn-zone', String(id));
  await canvas.focus(); await page.keyboard.press('w');
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('alert')).toHaveCount(0);
});
