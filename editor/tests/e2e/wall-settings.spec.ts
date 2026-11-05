import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { nativeRoomCopyFixture } from '../nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from '../roomHeightFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { wallMaterialChoices, wallMaterialTarget, validateWallMaterialSampling } from '../../src/model/wallMaterials';
import { captureEngine, enginePickDetails } from '../engineCapture';

for (const upper of [false, true]) test(`edits ${upper ? 'upper' : 'lower'} wall brightness and alignment with cancel, atomic Undo and native rendering`, async ({ page }, info) => {
  test.setTimeout(90000);
  const { document, copy } = nativeRoomCopyFixture(upper ? twoStoreyHeightTemplate() : 0), built = buildPlayableLevel(document, -1);
  const cell = built.topology!.cells.find(cell => cell.sourceId === copy.id)!;
  const camera = { zone: cell.id, x: Math.round(copy.vertices.reduce((sum, p) => sum + p.x, 0) / copy.vertices.length),
    z: Math.round(copy.vertices.reduce((sum, p) => sum + p.z, 0) / copy.vertices.length), upper: upper ? 1 : 0, angle: 0 };
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'wall-settings.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${copy.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' }); await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  if (upper) await page.getByRole('button', { name: 'Upper storey', exact: true }).click();
  const hash = await captureEngine(page, camera), revision = await preview.getAttribute('data-revision');
  const samples = Array.from({ length: 361 }, (_, index) => ({ x: (index % 19 + 1) / 20, y: (Math.floor(index / 19) + 1) / 20 }));
  const picks = await enginePickDetails(page, samples), index = picks.findIndex(pick => pick.surface?.upper === upper && wallMaterialTarget(built, pick.zone, pick.surface)?.room === copy.id);
  expect(index).toBeGreaterThanOrEqual(0);
  const target = wallMaterialTarget(built, picks[index]!.zone, picks[index]!.surface!)!, point = samples[index]!;
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'), rect = (await canvas.boundingBox())!;
  const open = async () => { await canvas.click({ position: { x: point.x * rect.width, y: point.y * rect.height } }); await page.getByRole('button', { name: 'Paint this wall', exact: true }).click(); };
  await open();
  const dialog = page.getByRole('dialog', { name: 'Paint wall material' });
  await dialog.locator('summary').filter({ hasText: 'Alignment and brightness' }).click();
  await dialog.getByRole('spinbutton', { name: 'Wall brightness offset', exact: true }).fill('24'); await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible(); await expect(preview).toHaveAttribute('data-revision', revision!);
  await open(); await dialog.locator('summary').filter({ hasText: 'Alignment and brightness' }).click();
  await expect(dialog.getByRole('spinbutton', { name: 'Wall brightness offset', exact: true })).toHaveValue('');
  const style = wallMaterialChoices(document).find(item => { try { validateWallMaterialSampling(item.wall!, item.wall!, 0); return true; } catch { return false; } })!;
  await dialog.getByRole('combobox', { name: 'Picked wall material style', exact: true }).selectOption(String(style.offset));
  await dialog.getByRole('spinbutton', { name: 'Wall brightness offset', exact: true }).fill('24');
  await dialog.getByRole('spinbutton', { name: 'Wall horizontal tile offset', exact: true }).fill('0');
  await dialog.getByRole('spinbutton', { name: 'Wall vertical row offset', exact: true }).fill('9');
  await dialog.getByRole('button', { name: 'Apply wall material', exact: true }).click();
  await expect(dialog).not.toBeVisible(); await expect(preview).not.toHaveAttribute('data-revision', revision!);
  await expect(preview.getByRole('status')).toContainText('zones loaded'); const combinedHash = await captureEngine(page, camera); expect(combinedHash).not.toBe(hash);
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('wall-alignment-brightness.ab3d'); await (await pending).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path))), rebuilt = buildPlayableLevel(saved, -1);
  expect(saved.files).toEqual(document.files);
  expect(saved.roomSource.rooms.find(room => room.id === target.room)!.wallSettings).toEqual([{ record: target.record, upper, brightness: 24, horizontalOffset: 0, verticalOffset: 9 }]);
  const output = rebuilt.surfaceSources!.find(item => item.room === cell.id && item.upper === upper && item.sourceOffset === target.record)!;
  expect((upper ? rebuilt.document.zones[cell.id]!.upper : rebuilt.document.zones[cell.id]!.lower).find(item => item.offset === output.offset)!.wall).toMatchObject({ brightness: 24, horizontalOffset: 0, verticalOffset: 9 });
  // Isolate both renderer effects: material/geometry stay fixed for these edits.
  await open(); await dialog.locator('summary').filter({ hasText: 'Alignment and brightness' }).click();
  await dialog.getByRole('spinbutton', { name: 'Wall brightness offset', exact: true }).fill('-16');
  await dialog.getByRole('button', { name: 'Apply wall material', exact: true }).click(); await expect(dialog).not.toBeVisible();
  await expect(preview.getByRole('status')).toContainText('zones loaded'); const brighterHash = await captureEngine(page, camera); expect(brighterHash).not.toBe(combinedHash);
  await open(); await dialog.locator('summary').filter({ hasText: 'Alignment and brightness' }).click();
  await dialog.getByRole('spinbutton', { name: 'Wall vertical row offset', exact: true }).fill('23');
  await dialog.getByRole('button', { name: 'Apply wall material', exact: true }).click(); await expect(dialog).not.toBeVisible();
  await expect(preview.getByRole('status')).toContainText('zones loaded'); expect(await captureEngine(page, camera)).not.toBe(brighterHash);
  for (const expected of [brighterHash, combinedHash]) {
    await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(preview.getByRole('status')).toContainText('zones loaded');
    expect(await captureEngine(page, camera)).toBe(expected);
  }
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect(await captureEngine(page, camera)).toBe(hash);
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(preview.getByRole('status')).toContainText('zones loaded');
  await page.screenshot({ path: info.outputPath('wall-alignment-brightness.png') });
});
