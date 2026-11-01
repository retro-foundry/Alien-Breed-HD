import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { framePlacementRoom, clickMap } from '../mapPlacement';
import { roomHeightFixture } from '../roomHeightFixture';
import { decodeProject, encodeProject } from '../../src/model/project';

test('named palette searches, repeated clicks ignore existing markers, invalid placement stays atomic and Escape cancels', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(), count = fixture.document.objects.length;
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'click-placement.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await framePlacementRoom(page, fixture.from.id);
  await page.getByRole('button', { name: 'Add object', exact: true }).click();
  const palette = page.getByRole('dialog', { name: 'Choose object to place' });
  for (const colour of ['Green', 'Red', 'Yellow', 'Blue']) await expect(palette.getByRole('button', { name: `${colour} key`, exact: true })).toHaveCount(1);
  await palette.getByRole('searchbox').fill('shotgun');
  await expect(palette.getByRole('button', { name: 'Blue key', exact: true })).toHaveCount(0);
  await palette.getByRole('searchbox').fill('');
  await page.screenshot({ path: testInfo.outputPath('object-library.png') });
  await palette.getByRole('button', { name: 'Barrel', exact: true }).click();
  await clickMap(page, 20128, 20128); // Player marker must not intercept placement.
  await clickMap(page, 20128, 20128); // Existing object must not intercept either.
  await clickMap(page, 19984, 20128);
  await expect(page.getByRole('alert')).toContainText('Click inside a room');
  await page.getByRole('button', { name: 'Dismiss error' }).click();
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await page.locator('canvas[aria-label="Level map"]').focus(); await page.keyboard.press('Escape');
  await expect(page.locator('.object-placement-bar')).toHaveCount(0);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('placed.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.objects).toHaveLength(count + 2);
  expect(saved.objects[count]).toMatchObject({ type: 10, x: 20128, z: 20128 });
  expect(saved.objects[count + 1]!.cid).not.toBe(saved.objects[count]!.cid);
  expect(errors).toEqual([]);
});
