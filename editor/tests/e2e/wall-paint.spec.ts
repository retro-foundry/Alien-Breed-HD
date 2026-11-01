import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { nativeRoomCopyFixture } from '../nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from '../roomHeightFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { roomSourceSnapshot } from '../../src/model/roomSource';
import { wallMaterialChoices, wallMaterialTarget } from '../../src/model/wallMaterials';
import { captureEngine, enginePickDetails, enginePlayerState } from '../engineCapture';

for (const upper of [false, true]) test(`3D ${upper ? 'upper' : 'lower'} wall paint stays independent through save, history and modal cancellation`, async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const { document, copy, original, dx, dz } = nativeRoomCopyFixture(upper ? twoStoreyHeightTemplate() : 0), built = buildPlayableLevel(document, -1);
  const cell = built.topology!.cells.find(cell => cell.sourceId === copy.id)!;
  const camera = { zone: cell.id, x: Math.round(copy.vertices.reduce((sum, p) => sum + p.x, 0) / copy.vertices.length),
    z: Math.round(copy.vertices.reduce((sum, p) => sum + p.z, 0) / copy.vertices.length), upper: upper ? 1 : 0, angle: 0 };
  const originalCamera = { ...camera, zone: built.topology!.cells.find(cell => cell.sourceId === original.id)!.id, x: camera.x - dx, z: camera.z - dz };
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'wall-paint.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  if (upper) await page.getByRole('button', { name: 'Upper', exact: true }).click();
  const originalHash = await captureEngine(page, originalCamera), beforeHash = await captureEngine(page, camera);
  const points = Array.from({ length: 361 }, (_, index) => ({ x: (index % 19 + 1) / 20, y: (Math.floor(index / 19) + 1) / 20 }));
  const before = await enginePickDetails(page, points), index = before.findIndex(pick => pick.surface && pick.surface.upper === upper && wallMaterialTarget(built, pick.zone, pick.surface)?.room === copy.id);
  expect(index).toBeGreaterThanOrEqual(0);
  const picked = before[index]!, target = wallMaterialTarget(built, picked.zone, picked.surface!)!;
  const style = wallMaterialChoices(document).find(item => item.texture !== (upper ? built.document.zones[picked.zone]!.upper : built.document.zones[picked.zone]!.lower).find(record => record.offset === picked.surface!.offset)!.texture)!;
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'), rect = (await canvas.boundingBox())!, point = points[index]!;
  const position = await enginePlayerState(page);
  await canvas.click({ position: { x: point.x * rect.width, y: point.y * rect.height } });
  await page.getByRole('button', { name: 'Paint this wall', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Paint wall material' });
  await dialog.getByRole('combobox', { name: 'Picked wall material style' }).selectOption(String(style.offset));
  await dialog.locator('summary').click();
  await expect(dialog.getByRole('group', { name: 'picked wall material choices' })).toBeVisible();
  await expect(dialog.getByRole('button', { name: 'Apply wall material', exact: true })).toBeInViewport();
  await page.screenshot({ path: testInfo.outputPath('wall-material-draft.png'), fullPage: true });
  await page.keyboard.press('Escape'); await expect(dialog).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  expect(await captureEngine(page, camera)).toBe(beforeHash);
  await page.getByRole('button', { name: 'Paint this wall', exact: true }).click();
  await dialog.getByRole('combobox', { name: 'Picked wall material style' }).selectOption(String(style.offset));
  const revision = (await preview.getAttribute('data-revision'))!;
  await dialog.getByRole('button', { name: 'Apply wall material', exact: true }).click();
  await expect(dialog).toHaveCount(0); await expect(preview).not.toHaveAttribute('data-revision', revision);
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  const paintedHash = await captureEngine(page, camera); expect(paintedHash).not.toBe(beforeHash);
  const after = await enginePickDetails(page, points);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('painted-source.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path))), rebuilt = buildPlayableLevel(saved, -1);
  expect(saved.files).toEqual(document.files); expect(roomSourceSnapshot(saved.roomSource).version).toBe(7);
  expect(saved.roomSource.rooms.find(room => room.id === copy.id)!.wallMaterials).toHaveLength(1);
  const changed = after.flatMap((pick, index) => pick.pixel !== before[index]!.pixel ? [pick] : []);
  expect(changed.length).toBeGreaterThan(0);
  for (const pick of changed) expect(wallMaterialTarget(rebuilt, pick.zone, pick.surface!)).toEqual(target);
  expect(await captureEngine(page, originalCamera)).toBe(originalHash);
  for (const [name, checksum] of [['Undo', beforeHash], ['Redo', paintedHash]] as const) {
    const old = (await preview.getAttribute('data-revision'))!;
    await page.getByRole('button', { name, exact: true }).click();
    await expect(preview).not.toHaveAttribute('data-revision', old); await expect(preview.getByRole('status')).toContainText('zones loaded');
    expect(await captureEngine(page, camera)).toBe(checksum);
  }
  expect(await enginePlayerState(page)).toEqual(position);
  await canvas.click({ position: { x: point.x * rect.width, y: point.y * rect.height } });
  await page.getByRole('button', { name: 'Paint this wall', exact: true }).click();
  const version = Number(await preview.getAttribute('data-revision'));
  await page.evaluate(() => {
    const host = window as Window & { wallPaintKey?: Promise<void> };
    host.wallPaintKey = new Promise(resolve => {
      const receive = (event: MessageEvent) => {
        if (event.data?.source !== 'ab3d-engine' || event.data.type !== 'shortcut') return;
        window.removeEventListener('message', receive); resolve();
      };
      window.addEventListener('message', receive);
    });
  });
  await canvas.evaluate((_, revision) => parent.postMessage({ source: 'ab3d-engine', type: 'shortcut', revision,
    shortcut: { key: 'z', ctrlKey: true, metaKey: false, altKey: false, shiftKey: false, repeat: false } }, location.origin), version);
  await page.evaluate(() => (window as Window & { wallPaintKey?: Promise<void> }).wallPaintKey);
  await expect(dialog).toBeVisible(); await expect(preview).toHaveAttribute('data-revision', String(version));
  await dialog.getByRole('button', { name: 'Reset wall material', exact: true }).click();
  await expect(preview).not.toHaveAttribute('data-revision', String(version)); await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect(await captureEngine(page, camera)).toBe(beforeHash);
  await expect(preview.getByRole('alert')).toHaveCount(0);
});
