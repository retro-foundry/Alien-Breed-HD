import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture, twoStoreyHeightTemplate } from '../roomHeightFixture';
import { nativeRoomCopyFixture } from '../nativeRoomCopyFixture';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { encodeProject, decodeProject } from '../../src/model/project';
import { roomSourceSnapshot } from '../../src/model/roomSource';
import { surfaceMaterialTarget, wallMaterialChoices } from '../../src/model/wallMaterials';
import { captureEngine, enginePickDetails, enginePlayerState } from '../engineCapture';

for (const mode of ['authored lower', 'authored upper', 'repaired doorway'] as const) test(`3D paints one ${mode} boundary with portable source and reversible native pixels`, async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const upper = mode === 'authored upper';
  const input = mode === 'repaired doorway' ? nativeRoomCopyFixture() : roomHeightFixture(upper ? twoStoreyHeightTemplate() : 0);
  const document = input.document, room = 'copy' in input ? input.copy : input.from, built = buildPlayableLevel(document, -1);
  const cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  const camera = { zone: cell.id, x: Math.round(room.vertices.reduce((sum, p) => sum + p.x, 0) / room.vertices.length),
    z: Math.round(room.vertices.reduce((sum, p) => sum + p.z, 0) / room.vertices.length), upper: upper ? 1 : 0, angle: 0 };
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'boundary.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  if (upper) await page.getByRole('button', { name: 'Upper', exact: true }).click();
  const points = Array.from({ length: 361 }, (_, index) => ({ x: (index % 19 + 1) / 20, y: (Math.floor(index / 19) + 1) / 20 }));
  let beforeHash = 0, before: Awaited<ReturnType<typeof enginePickDetails>> = [], pickedIndex = -1;
  // A repaired doorway can face any cardinal direction. Inspect actual native
  // coverage rather than choosing a fabricated graphics record or screen spot.
  for (const angle of [0, 2048, 4096, 6144]) {
    camera.angle = angle; beforeHash = await captureEngine(page, camera); before = await enginePickDetails(page, points);
    pickedIndex = before.findIndex(pick => {
      const target = pick.surface && surfaceMaterialTarget(built, pick.zone, pick.surface);
      return target && 'index' in target && target.room === room.id && target.upper === upper;
    });
    if (pickedIndex >= 0) break;
  }
  expect(pickedIndex).toBeGreaterThanOrEqual(0);
  const picked = before[pickedIndex]!, target = surfaceMaterialTarget(built, picked.zone, picked.surface!)!;
  const record = (upper ? built.document.zones[picked.zone]!.upper : built.document.zones[picked.zone]!.lower).find(item => item.offset === picked.surface!.offset)!;
  const style = wallMaterialChoices(document).find(item => item.texture !== record.texture)!;
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'), rect = (await canvas.boundingBox())!, point = points[pickedIndex]!;
  const position = await enginePlayerState(page);
  await canvas.click({ position: { x: point.x * rect.width, y: point.y * rect.height } });
  await page.getByRole('button', { name: 'Paint this wall', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Paint wall material' });
  await expect(dialog).toContainText('authored boundary');
  await dialog.getByRole('combobox', { name: 'Picked wall material style' }).selectOption(String(style.offset));
  const revision = (await preview.getAttribute('data-revision'))!;
  await dialog.getByRole('button', { name: 'Apply wall material', exact: true }).click();
  await expect(dialog).toHaveCount(0); await expect(preview).not.toHaveAttribute('data-revision', revision);
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  const paintedHash = await captureEngine(page, camera); expect(paintedHash).not.toBe(beforeHash);
  const after = await enginePickDetails(page, points);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const file = testInfo.outputPath('painted-boundary.ab3d'); await (await download).saveAs(file);
  const saved = decodeProject(new Uint8Array(await readFile(file))), rebuilt = buildPlayableLevel(saved, -1);
  expect(saved.files).toEqual(document.files); expect(roomSourceSnapshot(saved.roomSource).version).toBe(8);
  expect(saved.roomSource.rooms.find(item => item.id === room.id)!.boundaryMaterials!.overrides).toHaveLength(1);
  const changed = after.filter((pick, index) => pick.pixel !== before[index]!.pixel);
  expect(changed.length).toBeGreaterThan(0);
  for (const pick of changed) expect(surfaceMaterialTarget(rebuilt, pick.zone, pick.surface!)).toEqual(target);
  for (const [name, hash] of [['Undo', beforeHash], ['Redo', paintedHash]] as const) {
    const old = (await preview.getAttribute('data-revision'))!;
    await page.getByRole('button', { name, exact: true }).click();
    await expect(preview).not.toHaveAttribute('data-revision', old); await expect(preview.getByRole('status')).toContainText('zones loaded');
    expect(await captureEngine(page, camera)).toBe(hash);
  }
  expect(await enginePlayerState(page)).toEqual(position);
  await page.screenshot({ path: testInfo.outputPath('painted-boundary.png'), fullPage: true });
  const oldFrame = await page.locator('iframe[title="Alien Breed game engine"]').elementHandle();
  await page.locator('input[type=file]').setInputFiles(file);
  await expect.poll(() => oldFrame!.evaluate(node => node.isConnected)).toBe(false);
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect(await captureEngine(page, camera)).toBe(paintedHash);
});
