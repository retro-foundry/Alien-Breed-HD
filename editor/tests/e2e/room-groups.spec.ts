import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture } from '../roomHeightFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { enginePlayerState, engineObjectState } from '../engineCapture';
import { applyCommand, setSourceTeleport, editRoomSource, editRoomAppearance } from '../../src/model/commands';
import { createObject, objectTemplates } from '../../src/model/objectAuthoring';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../../src/model/roomSource';

test('imported room groups toggle conventionally and Escape clears only selection', async ({ page }) => {
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  const first = page.getByRole('button', { name: /^Zone 000(?:\s|$)/ }), second = page.getByRole('button', { name: /^Zone 001(?:\s|$)/ });
  await first.click(); await second.click({ modifiers: ['Shift'] });
  const group = page.getByRole('complementary', { name: 'Room group inspector' });
  await expect(group.getByRole('heading', { name: '2 selected rooms' })).toBeVisible();
  await expect(first).toHaveAttribute('aria-pressed', 'true'); await expect(second).toHaveAttribute('aria-pressed', 'true');
  await first.click({ modifiers: ['Control'] }); await expect(group).toHaveCount(0);
  await expect(first).toHaveAttribute('aria-pressed', 'false'); await expect(second).toHaveAttribute('aria-pressed', 'true');
  await first.click({ modifiers: ['Shift'] }); await expect(group).toBeVisible();
  await page.keyboard.press('Escape'); await expect(group).toHaveCount(0);
  await expect(first).toHaveAttribute('aria-pressed', 'false'); await expect(second).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});

test('shared-boundary room moves protect a neighbouring start and compile a valid deformation in the native engine', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(); let document = fixture.document;
  document = applyCommand(document, editRoomSource(commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, 0, fixture.to.id),
    [{ x: 20256, z: 20064 }, { x: 20512, z: 20064 }, { x: 20512, z: 20192 }, { x: 20256, z: 20192 }]))));
  document = applyCommand(document, editRoomSource(commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, 0),
    [{ x: 22000, z: 20000 }, { x: 22256, z: 20000 }, { x: 22256, z: 20256 }, { x: 22000, z: 20256 }]))));
  const third = document.roomSource.rooms.at(-1)!;
  document = applyCommand(document, editRoomAppearance(document, { ...document.roomAppearances!.find(item => item.room === fixture.from.id)!, room: third.id }));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'neighbour-move.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.to.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  await page.getByRole('button', { name: new RegExp(`^Room ${third.id.replace(/^room-/, '')}(?:\\s|$)`) }).click({ modifiers: ['Shift'] });
  const group = page.getByRole('complementary', { name: 'Room group inspector' });
  const x = group.getByRole('spinbutton', { name: 'Room group X offset' });
  await x.fill('-160'); await group.getByRole('button', { name: 'Move selected rooms', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('Player 1 start would be outside neighbouring room');
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
  await page.getByRole('button', { name: 'Dismiss error', exact: true }).click();
  await x.fill('-16'); await group.getByRole('button', { name: 'Move selected rooms', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('neighbour-move.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.sourceStarts).toEqual(document.sourceStarts);
  expect(saved.roomSource.rooms.find(room => room.id === fixture.from.id)!.vertices).toEqual(expect.arrayContaining([{ x: 20240, z: 20064 }, { x: 20240, z: 20192 }, { x: 20000, z: 20000 }]));
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await enginePlayerState(page)).toMatchObject({ x: 20128, z: 20128, playing: false });
});

test('room translation moves linked starts, objects and inbound teleport destinations with one undo and native positions', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(), template = objectTemplates(fixture.document).find(object => object.type === 0)!;
  const object = createObject(fixture.document, { template: template.id, room: fixture.from.id, x: 20160.25, z: 20128.5, y: -7.5 });
  let document = applyCommand(fixture.document, object.command);
  const incoming = document.roomSource.rooms.find(room => room.originZone === 0)!;
  document = applyCommand(document, setSourceTeleport(document, incoming.id, fixture.from.id, 20128, 20128));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'move-rooms.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.to.id.replace(/^room-/, '')}(?:\\s|$)`) }).click({ modifiers: ['Shift'] });
  const group = page.getByRole('complementary', { name: 'Room group inspector' });
  const x = group.getByRole('spinbutton', { name: 'Room group X offset' });
  await x.fill('20000'); await group.getByRole('button', { name: 'Move selected rooms', exact: true }).click();
  await expect(page.getByRole('alert')).toBeVisible(); await expect(page.locator('.dirty-dot')).toHaveCount(0);
  await page.getByRole('button', { name: 'Dismiss error', exact: true }).click();
  await x.fill('64'); await group.getByRole('spinbutton', { name: 'Room group Z offset' }).fill('-32');
  await group.getByRole('button', { name: 'Move selected rooms', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('moved-rooms.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.sourceStarts![0]).toMatchObject({ x: 20192, z: 20096 });
  expect(saved.sourceObjects!.find(item => item.id === object.id)).toMatchObject({ x: 20224.25, z: 20096.5 });
  expect(saved.sourceTeleports!.find(link => link.room === incoming.id)).toMatchObject({ destination: fixture.from.id, x: 20192, z: 20096 });
  expect(saved.files.data).toEqual(document.files.data);
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await enginePlayerState(page)).toMatchObject({ x: 20192, z: 20096, playing: false });
  expect(await engineObjectState(page, object.id)).toMatchObject({ x: 20224.25, z: 20096.5, yFixed: -7.5 * 256 });
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect.poll(async () => (await enginePlayerState(page)).x).toBe(20128);
  expect(await engineObjectState(page, object.id)).toMatchObject({ x: 20160.25, z: 20128.5, yFixed: -7.5 * 256 });
});

test('room groups frame, toggle on the map and delete as one source edit without discarding linked starts', async ({ page }, testInfo) => {
  const { document, from, to } = roomHeightFixture();
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'room-group.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  const first = page.getByRole('button', { name: new RegExp(`^Room ${from.id.replace(/^room-/, '')}(?:\\s|$)`) });
  const second = page.getByRole('button', { name: new RegExp(`^Room ${to.id.replace(/^room-/, '')}(?:\\s|$)`) });
  await first.click(); await second.click({ modifiers: ['Shift'] });
  const group = page.getByRole('complementary', { name: 'Room group inspector' });
  await group.getByRole('button', { name: 'Frame selected rooms', exact: true }).click();
  const map = page.locator('canvas[aria-label="Level map"]');
  await expect(map).toHaveAttribute('data-world-x', '20628');
  await expect(map).toHaveAttribute('data-world-z', '20128');
  const point = await map.evaluate(node => {
    const canvas = node as HTMLCanvasElement, rect = canvas.getBoundingClientRect(), scale = Number(canvas.dataset.worldScale);
    return { x: rect.x + rect.width / 2 + (20160 - Number(canvas.dataset.worldX)) * scale,
      y: rect.y + rect.height / 2 - (20160 - Number(canvas.dataset.worldZ)) * scale };
  });
  await page.keyboard.down('Shift'); await page.mouse.click(point.x, point.y); await page.keyboard.up('Shift');
  await expect(group).toHaveCount(0); await expect(first).toHaveAttribute('aria-pressed', 'false');
  await page.keyboard.down('Shift'); await page.mouse.click(point.x, point.y); await page.keyboard.up('Shift');
  await expect(group).toBeVisible();
  await page.keyboard.press('Delete'); await expect(first).toHaveCount(0); await expect(second).toHaveCount(0);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('deleted-rooms.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.roomSource.rooms).toHaveLength(document.roomSource.rooms.length - 2);
  expect(saved.sourceStarts).toEqual(document.sourceStarts); expect(saved.files.data).toEqual(document.files.data);
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(first).toBeVisible(); await expect(second).toBeVisible();
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await enginePlayerState(page)).toMatchObject({ x: 20128, z: 20128, playing: false });
});
