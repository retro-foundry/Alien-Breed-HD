import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture } from '../roomHeightFixture';
import { framePlacementRoom } from '../mapPlacement';
import { encodeProject, decodeProject } from '../../src/model/project';
import { applyCommand, setSourceStart } from '../../src/model/commands';
import { moveRoomGroup } from '../../src/model/roomGroups';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { captureEngine, enginePlayerState } from '../engineCapture';
async function open(page: Page, document: ReturnType<typeof roomHeightFixture>['document']) {
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'corners.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
}
async function pixel(page: Page, x: number, z: number) {
  const canvas = page.locator('canvas[aria-label="Level map"]'), bounds = (await canvas.boundingBox())!;
  const view = await canvas.evaluate(c => ({ x: Number(c.getAttribute('data-world-x')), z: Number(c.getAttribute('data-world-z')), scale: Number(c.getAttribute('data-world-scale')) }));
  return { x: bounds.x + bounds.width / 2 + (x - view.x) * view.scale, y: bounds.y + bounds.height / 2 - (z - view.z) * view.scale };
}

test('map edge insertion preserves native texture pixels and supports direct removal, saves and atomic history', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(), built = buildPlayableLevel(fixture.document, -1), zone = built.topology!.cells.find(c => c.sourceId === fixture.from.id)!.id;
  await open(page, fixture.document); await framePlacementRoom(page, fixture.from.id);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const camera = { zone, x: 20128, z: 20128, angle: 0, upper: 0 }, before = await captureEngine(page, camera);
  await page.getByRole('button', { name: 'Editor view', exact: true }).click(); await framePlacementRoom(page, fixture.from.id);
  await page.locator('canvas[aria-label="Level map"]').focus(); await page.keyboard.down('Space');
  await expect(page.getByRole('button', { name: 'Add corner on edge 1', exact: true })).toHaveCount(0);
  await page.keyboard.up('Space');
  await page.getByRole('button', { name: 'Add corner on edge 1', exact: true }).click();
  await expect(page.locator('.inspector')).toContainText('5 corners');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('inserted.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.roomSource.rooms.find(r => r.id === fixture.from.id)!.vertices).toContainEqual({ x: 20000, z: 20128 });
  await page.getByRole('button', { name: 'Engine view', exact: true }).click(); await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await captureEngine(page, camera)).toBe(before);
  await page.getByRole('button', { name: 'Editor view', exact: true }).click();
  await page.getByText('Edit room corners', { exact: true }).click();
  await page.getByRole('button', { name: 'Remove corner 2', exact: true }).click(); await expect(page.locator('.inspector')).toContainText('4 corners');
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(page.locator('.inspector')).toContainText('5 corners');
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(page.locator('.inspector')).toContainText('4 corners');
});

test('splitting and dragging a shared edge keeps both rooms joined and traversable in native play', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(), document = applyCommand(fixture.document, moveRoomGroup(fixture.document, [fixture.to.id], -744, 0));
  const room = document.roomSource.rooms.find(r => r.id === fixture.from.id)!, edge = room.vertices.findIndex((p, i) => p.x === 20256 && room.vertices[(i + 1) % room.vertices.length]!.x === 20256);
  await open(page, document); await framePlacementRoom(page, fixture.from.id);
  await page.getByRole('button', { name: `Add corner on edge ${edge + 1}`, exact: true }).click();
  const a = await pixel(page, 20256, 20128), b = await pixel(page, 20240, 20128);
  await page.mouse.move(a.x, a.y); await page.mouse.down(); await page.mouse.move(b.x, b.y, { steps: 5 }); await page.mouse.up();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('shared.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  for (const id of [fixture.from.id, fixture.to.id]) expect(saved.roomSource.rooms.find(r => r.id === id)!.vertices).toContainEqual({ x: 20240, z: 20128 });
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' }); await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus(); await page.keyboard.down('d');
  try { await expect.poll(async () => (await enginePlayerState(page)).x).toBeGreaterThan(20272); } finally { await page.keyboard.up('d'); }
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
});

test('unsafe corner removal is rejected with the player and history intact', async ({ page }) => {
  const fixture = roomHeightFixture(), document = applyCommand(fixture.document, setSourceStart(fixture.document, 1, fixture.from.id, 20008, 20008));
  await open(page, document); await framePlacementRoom(page, fixture.from.id); await page.getByText('Edit room corners', { exact: true }).click();
  await page.getByRole('button', { name: 'Remove corner 1', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Player 1 start'); await expect(page.locator('.inspector')).toContainText('4 corners');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
});

test('double-clicking an edge inserts at the pointer rather than forcing its midpoint', async ({ page }) => {
  const fixture = roomHeightFixture(); await open(page, fixture.document); await framePlacementRoom(page, fixture.from.id);
  const p = await pixel(page, 20000, 20064); await page.mouse.dblclick(p.x, p.y);
  await expect(page.locator('.inspector')).toContainText('5 corners');
  await page.getByText('Edit room corners', { exact: true }).click();
  await expect(page.getByRole('spinbutton', { name: 'Corner 2 X', exact: true })).toHaveValue('20000');
  await expect(page.getByRole('spinbutton', { name: 'Corner 2 Z', exact: true })).toHaveValue('20064');
});
