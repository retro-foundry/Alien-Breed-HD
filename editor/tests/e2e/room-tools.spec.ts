import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { decodeProject } from '../../src/model/project';
import { encodeProject } from '../../src/model/project';
import { openLevel } from '../../src/model/level';
import { removeSourceRooms } from '../../src/model/roomSource';
import { originalFiles } from '../helpers';

async function prepare(page: Page) {
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: /^Zone 000/ }).click();
  await page.getByRole('button', { name: 'Frame map', exact: true }).click();
  await page.getByRole('combobox', { name: 'Grid spacing' }).selectOption('0');
  return page.getByRole('button', { name: 'Draw room', exact: true });
}
async function clickCorners(page: Page, reverse = false) {
  const box = (await page.locator('canvas[aria-label="Level map"]').boundingBox())!;
  const corners = [[5, 5], [35, 5], [35, 35], [5, 35]];
  for (const [x, y] of reverse ? corners.toReversed() : corners) await page.mouse.click(box.x + x!, box.y + y!);
}
async function backup(page: Page, path: string) {
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  await (await download).saveAs(path); return decodeProject(new Uint8Array(await readFile(path)));
}
async function mapPoint(page: Page, point: { x: number; z: number }) {
  const map = page.locator('canvas[aria-label="Level map"]'), box = (await map.boundingBox())!;
  const view = await map.evaluate(node => ({ x: Number(node.dataset.worldX), z: Number(node.dataset.worldZ), scale: Number(node.dataset.worldScale) }));
  return { x: box.x + box.width / 2 + (point.x - view.x) * view.scale, y: box.y + box.height / 2 - (point.z - view.z) * view.scale };
}

test('drawing cancels on Escape and tool changes; either direction commits one undoable room', async ({ page }, testInfo) => {
  const draw = await prepare(page);
  await draw.click(); await clickCorners(page);
  await page.getByRole('button', { name: 'Pan tool', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Finish room', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await draw.click(); await clickCorners(page); await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  // Escape clears selection, so explicitly choose the settings template again.
  await page.getByRole('button', { name: /^Zone 000/ }).click(); await page.getByRole('button', { name: 'Frame map', exact: true }).click();
  await draw.click(); await clickCorners(page); await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Room 138', exact: true })).toBeVisible();
  const first = await backup(page, testInfo.outputPath('forward.ab3d'));
  expect(first.roomSource.rooms).toHaveLength(139);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('button', { name: /^Room 138/ })).toHaveCount(0);
  await page.getByRole('button', { name: /^Zone 000/ }).click(); await page.getByRole('button', { name: 'Frame map', exact: true }).click();
  await draw.click(); await clickCorners(page, true); await page.keyboard.press('Enter');
  const reversed = await backup(page, testInfo.outputPath('reverse.ab3d'));
  expect(reversed.roomSource).toEqual(first.roomSource);
  await expect(page.getByText(/This project has room outlines/)).toBeVisible();
  await page.getByRole('combobox', { name: 'Original level' }).selectOption('2');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 2');
  const next = openLevel('Level 2', originalFiles(2));
  const centre = (Math.min(...next.points.map(point => point.x)) + Math.max(...next.points.map(point => point.x))) / 2;
  await expect(page.locator('canvas[aria-label="Level map"]')).toHaveAttribute('data-world-x', String(centre));
});

test('room corners edit atomically, drag cancellation leaves source unchanged, and deletion is undoable', async ({ page }, testInfo) => {
  const draw = await prepare(page); await draw.click(); await clickCorners(page); await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Room 138', exact: true })).toBeVisible();
  const before = await backup(page, testInfo.outputPath('before.ab3d'));
  const corner = await mapPoint(page, before.roomSource.rooms.at(-1)!.vertices[0]!);
  await page.mouse.move(corner.x, corner.y); await page.mouse.down(); await page.mouse.move(corner.x - 3, corner.y + 3); await page.mouse.up();
  const dragged = await backup(page, testInfo.outputPath('dragged.ab3d')); expect(dragged.roomSource).not.toEqual(before.roomSource);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  expect((await backup(page, testInfo.outputPath('undone-drag.ab3d'))).roomSource).toEqual(before.roomSource);
  await page.mouse.move(corner.x, corner.y); await page.mouse.down(); await page.mouse.move(corner.x - 3, corner.y + 3);
  await page.keyboard.press('Escape'); await page.mouse.up();
  const cancelled = await backup(page, testInfo.outputPath('cancelled-drag.ab3d')); expect(cancelled.roomSource).toEqual(before.roomSource);
  await page.getByRole('button', { name: /^Room 138/ }).click();
  await page.locator('.inspector summary').filter({ hasText: 'Edit room corners' }).click();
  const x = page.getByRole('spinbutton', { name: 'Corner 1 X', exact: true }), oldX = before.roomSource.rooms.at(-1)!.vertices[0]!.x;
  await x.fill(String(oldX - 1)); await x.press('Enter');
  const edited = await backup(page, testInfo.outputPath('edited.ab3d')); expect(edited.roomSource.rooms.at(-1)!.vertices[0]!.x).toBe(oldX - 1);
  await page.screenshot({ path: testInfo.outputPath('room-outline-editor.png'), fullPage: true });
  await page.getByRole('button', { name: 'Delete room', exact: true }).click();
  await expect(page.getByRole('button', { name: /^Room 138/ })).toHaveCount(0);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('button', { name: /^Room 138/ })).toBeVisible();
  const restored = await backup(page, testInfo.outputPath('restored.ab3d')); expect(restored.roomSource).toEqual(edited.roomSource);
});

test('imported wall identities follow corner edits through downloads and browser recovery', async ({ page }, testInfo) => {
  page.on('dialog', dialog => void dialog.accept());
  const original = openLevel('Wall source recovery', originalFiles(1)), room = original.roomSource.rooms.find(room => room.vertices.length === 4)!;
  const isolated = { ...original, roomSource: removeSourceRooms(original.roomSource, original.roomSource.rooms.filter(other => other.id !== room.id).map(other => other.id)).after };
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'wall-sources.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(isolated)) });
  await expect(page.locator('.project-picker strong')).toHaveText('Wall source recovery');
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(room.recordZone).padStart(3, '0')}`) }).click();
  await page.locator('.inspector summary').filter({ hasText: 'Edit room corners' }).click();
  const x = page.getByRole('spinbutton', { name: 'Corner 1 X', exact: true });
  await x.fill(String(room.vertices[0]!.x - 1)); await x.press('Enter');
  const edited = await backup(page, testInfo.outputPath('wall-corner-edit.ab3d'));
  expect(edited.roomSource.rooms[0]!.vertices).not.toEqual(room.vertices);
  expect(edited.roomSource.rooms[0]!.wallSources).toEqual(room.wallSources);
  await expect(page.getByText('Workspace saved', { exact: true })).toBeVisible();
  await page.reload(); await expect(page.locator('.project-picker strong')).toHaveText('Wall source recovery');
  expect((await backup(page, testInfo.outputPath('recovered-wall-corner.ab3d'))).roomSource).toEqual(edited.roomSource);
});

test('crossing outlines fail without committing a partial room', async ({ page }, testInfo) => {
  const draw = await prepare(page); await draw.click();
  const box = (await page.locator('canvas[aria-label="Level map"]').boundingBox())!;
  for (const [x, y] of [[5, 5], [35, 35], [35, 5], [5, 35]]) await page.mouse.click(box.x + x!, box.y + y!);
  await page.keyboard.press('Enter');
  await expect(page.getByRole('alert')).toContainText('crosses itself');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  const saved = await backup(page, testInfo.outputPath('invalid-draft.ab3d')); expect(saved.roomSource.rooms).toHaveLength(138);
  await page.getByRole('button', { name: 'Cancel room', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Finish room', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: /^Zone 000/ }).click(); await page.getByRole('button', { name: 'Frame map', exact: true }).click();
  await draw.click(); await clickCorners(page); await page.keyboard.press('Enter');
  const firstRoom = await backup(page, testInfo.outputPath('first-good-room.ab3d'));
  await draw.click();
  for (const point of firstRoom.roomSource.rooms.at(-1)!.vertices) { const pixel = await mapPoint(page, point); await page.mouse.click(pixel.x, pixel.y); }
  await page.keyboard.press('Enter');
  await expect(page.getByRole('alert')).toContainText('overlap');
  const overlap = await backup(page, testInfo.outputPath('overlap-draft.ab3d')); expect(overlap.roomSource.rooms).toHaveLength(139);
});

test('an empty saved source can create a new room using its retained settings templates', async ({ page }, testInfo) => {
  const original = openLevel('Empty rooms', originalFiles(1));
  const empty = { ...original, roomSource: removeSourceRooms(original.roomSource, original.roomSource.rooms.map(room => room.id)).after };
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'empty.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(empty)) });
  await expect(page.locator('.project-picker strong')).toHaveText('Empty rooms');
  await page.getByRole('combobox', { name: 'Grid spacing' }).selectOption('0');
  await page.getByRole('combobox', { name: 'Room settings template' }).selectOption('3');
  await page.getByRole('button', { name: 'Draw room', exact: true }).click(); await clickCorners(page); await page.keyboard.press('Enter');
  const saved = await backup(page, testInfo.outputPath('reborn-source.ab3d'));
  expect(saved.roomSource.rooms).toHaveLength(1); expect(saved.roomSource.rooms[0]!.recordZone).toBe(3);
});
