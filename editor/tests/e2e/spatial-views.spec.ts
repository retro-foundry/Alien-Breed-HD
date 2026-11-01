import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture, twoStoreyHeightTemplate } from '../roomHeightFixture';
import { originalFiles } from '../helpers';
import { originalObjectPalette, placePaletteObject } from '../../src/model/objectPalette';
import { applyCommand } from '../../src/model/commands';
import { encodeProject, decodeProject } from '../../src/model/project';
import { editorScene } from '../../src/model/editorScene';
import { projectPoint, type EditorCamera, type Vector3 } from '../../src/model/editorCamera';
import { roomSettings } from '../../src/model/roomHeights';
import { engineObjectState } from '../engineCapture';

async function openFixture(page: Page, document: ReturnType<typeof roomHeightFixture>['document']) {
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'spatial.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
}
async function project(page: Page, position: Vector3) {
  const map = page.locator('canvas[aria-label="Editing viewport"]'), bounds = (await map.boundingBox())!;
  const camera = JSON.parse((await map.getAttribute('data-camera'))!) as EditorCamera;
  const size = JSON.parse((await map.getAttribute('data-size'))!) as { width: number; height: number };
  const point = projectPoint(position, camera, size.width, size.height);
  return { x: bounds.x + point.x, y: bounds.y + point.y };
}
async function selectRoom(page: Page, id: string) {
  await page.getByRole('tab', { name: 'Zones', exact: true }).click();
  await page.getByRole('button', { name: new RegExp(`^Room ${id.replace(/^room-/, '')}`) }).click();
}

test('all six axis directions and both editable 3D projections share selection and keep Engine view separate', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(); await openFixture(page, fixture.document); await selectRoom(page, fixture.from.id);
  const selector = page.getByRole('combobox', { name: 'Editing view', exact: true });
  await expect(selector.locator('option')).toHaveCount(8);
  for (const view of ['bottom', 'front', 'back', 'left', 'right', 'orthographic', 'perspective']) {
    await selector.selectOption(view);
    await expect(page.locator('canvas[aria-label="Editing viewport"]')).toHaveAttribute('data-view', view);
    await expect(page.locator('.inspector-heading h2')).toContainText(`Room ${fixture.from.id.replace(/^room-/, '')}`);
  }
  const canvas = page.locator('canvas[aria-label="Editing viewport"]'), before = await canvas.getAttribute('data-camera'), bounds = (await canvas.boundingBox())!;
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2); await page.mouse.down({ button: 'right' }); await page.mouse.move(bounds.x + bounds.width / 2 + 60, bounds.y + bounds.height / 2 + 20, { steps: 5 }); await page.mouse.up({ button: 'right' });
  expect(await canvas.getAttribute('data-camera')).not.toBe(before);
  await page.screenshot({ path: testInfo.outputPath('perspective-editor.png') });
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await expect(canvas).toHaveCount(0); await expect(selector).toHaveCount(0);
  await page.getByRole('button', { name: 'Editor view', exact: true }).click();
  await expect(selector).toHaveValue('perspective'); await expect(canvas).toBeVisible();
});

test('front height dragging previews without saving, cancels, commits once, and survives Undo and reopening', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(), baseline = roomSettings(fixture.document, fixture.from).floor;
  await openFixture(page, fixture.document); await selectRoom(page, fixture.from.id);
  await page.getByRole('combobox', { name: 'Editing view', exact: true }).selectOption('front');
  const handle = page.getByRole('button', { name: 'Drag room floor height', exact: true });
  const start = (await handle.boundingBox())!;
  const drag = async () => { await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2); await page.mouse.down(); await page.mouse.move(start.x + start.width / 2, start.y + start.height / 2 - 20, { steps: 4 }); };
  await drag();
  const download = page.waitForEvent('download'); await page.keyboard.press('Control+s');
  const path = testInfo.outputPath('draft.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(roomSettings(saved, saved.roomSource.rooms.find(r => r.id === fixture.from.id)!).floor).toBe(baseline);
  await page.keyboard.press('Escape'); await page.mouse.up();
  await expect(page.getByRole('spinbutton', { name: 'Lower floor height', exact: true })).toHaveValue(String(baseline));
  await drag(); await page.mouse.up();
  const value = Number(await page.getByRole('spinbutton', { name: 'Lower floor height', exact: true }).inputValue()); expect(value).toBeLessThan(baseline);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('spinbutton', { name: 'Lower floor height', exact: true })).toHaveValue(String(baseline));
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  const currentHandle = (await handle.boundingBox())!;
  await page.mouse.move(currentHandle.x + currentHandle.width / 2, currentHandle.y + currentHandle.height / 2); await page.mouse.down();
  await page.mouse.move(currentHandle.x + currentHandle.width / 2, currentHandle.y - 500, { steps: 5 });
  await expect(page.locator('.spatial-validation')).toContainText('ceiling'); await page.mouse.up();
  await expect(page.getByRole('alert')).toContainText('ceiling'); await page.getByRole('button', { name: 'Dismiss error' }).click();
  await expect(page.getByRole('spinbutton', { name: 'Lower floor height', exact: true })).toHaveValue(String(value));
  const finalDownload = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const finalPath = testInfo.outputPath('height.ab3d'); await (await finalDownload).saveAs(finalPath);
  await page.locator('input[type=file]').setInputFiles(finalPath);
  await selectRoom(page, fixture.from.id); await expect(page.getByRole('spinbutton', { name: 'Lower floor height', exact: true })).toHaveValue(String(value));
});

test('perspective picking selects the room under the pointer; object axis gestures edit real saved and native positions', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(), files = originalFiles(1), choice = originalObjectPalette([files.data], [files.graphics]).find(c => c.name === 'Barrel')!;
  const proposal = placePaletteObject(fixture.document, choice, fixture.from.id, 20160, 20192, false), document = applyCommand(fixture.document, proposal.command);
  await openFixture(page, document); await selectRoom(page, fixture.from.id);
  await page.getByRole('combobox', { name: 'Editing view', exact: true }).selectOption('perspective');
  await page.locator('canvas[aria-label="Editing viewport"]').focus(); await page.keyboard.press('Escape');
  const floor = roomSettings(document, fixture.from).floor, p = await project(page, { x: 20064, y: floor, z: 20128 });
  await page.mouse.click(p.x, p.y);
  await expect(page.locator('.inspector-heading h2')).toHaveText(`Room ${fixture.from.id.replace(/^room-/, '')}`);
  await page.getByRole('tab', { name: 'Objects', exact: true }).click();
  await page.getByRole('button', { name: `Barrel ${proposal.id}`, exact: true }).click();
  const initial = editorScene(document, false).markers.find(m => m.kind === 'object' && m.id === proposal.id)!.position;
  const y = page.getByRole('button', { name: 'Move object Y', exact: true });
  await y.focus(); await y.press('ArrowUp');
  await expect(page.getByRole('spinbutton', { name: 'Height', exact: true })).toHaveValue(String(initial.y - .5));
  await page.getByRole('combobox', { name: 'Editing view', exact: true }).selectOption('front');
  await page.getByRole('combobox', { name: 'Grid spacing' }).selectOption('8');
  const x = page.getByRole('button', { name: 'Move object X', exact: true }), box = (await x.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.down(); await page.mouse.move(box.x + box.width / 2 + 30, box.y + box.height / 2, { steps: 5 }); await page.mouse.up();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('object.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path))), marker = editorScene(saved, false).markers.find(m => m.kind === 'object' && m.id === proposal.id)!;
  expect(marker.position.x).toBeGreaterThan(initial.x); expect(marker.position.z).toBe(initial.z); expect(marker.position.y).toBe(initial.y - .5);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await engineObjectState(page, proposal.id)).toMatchObject({ x: marker.position.x, z: initial.z, yFixed: (initial.y - .5) * 256 });
});

test('orthographic upper-floor handles, source corners and perspective click placement save editable results', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(twoStoreyHeightTemplate()), baseline = roomSettings(fixture.document, fixture.from);
  await openFixture(page, fixture.document); await selectRoom(page, fixture.from.id);
  await page.getByRole('combobox', { name: 'Editing view', exact: true }).selectOption('orthographic');
  const lowerCamera = await page.locator('canvas[aria-label="Editing viewport"]').getAttribute('data-camera');
  await page.getByRole('button', { name: 'Upper', exact: true }).click();
  await expect.poll(async () => page.locator('canvas[aria-label="Editing viewport"]').getAttribute('data-camera')).not.toBe(lowerCamera);
  await page.getByRole('button', { name: 'Drag room floor height', exact: true }).focus(); await page.keyboard.press('ArrowUp');
  await expect(page.getByRole('spinbutton', { name: 'Upper floor height', exact: true })).toHaveValue(String(baseline.upperFloor - .25));
  await page.getByRole('combobox', { name: 'Grid spacing' }).selectOption('8');
  await page.getByRole('button', { name: 'Move room corner 1', exact: true }).focus(); await page.keyboard.press('ArrowRight');
  await page.getByRole('combobox', { name: 'Editing view', exact: true }).selectOption('perspective');
  await page.getByRole('button', { name: 'Add object', exact: true }).click();
  await page.getByRole('dialog', { name: 'Choose object to place' }).getByRole('button', { name: 'Blue key', exact: true }).click();
  const p = await project(page, { x: 20128, y: baseline.upperFloor - .25, z: 20192 }); await page.mouse.click(p.x, p.y); await page.keyboard.press('Escape');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('upper-edit.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path))), room = saved.roomSource.rooms.find(r => r.id === fixture.from.id)!;
  expect(roomSettings(saved, room).floor).toBe(baseline.floor); expect(roomSettings(saved, room).upperFloor).toBe(baseline.upperFloor - .25);
  expect(room.vertices).toContainEqual({ x: fixture.from.vertices[0]!.x + 8, z: fixture.from.vertices[0]!.z });
  expect(saved.objects).toHaveLength(fixture.document.objects.length + 1);
  expect(saved.objects.at(-1)).toMatchObject({ type: 4, upper: true });
  expect(saved.sourceObjects?.at(-1)).toMatchObject({ room: fixture.from.id, x: 20128, z: 20192 });
  await page.screenshot({ path: testInfo.outputPath('orthographic-upper-edit.png') });
});

test('side-view corner keyboard edits move Z and preserve the hidden X coordinate', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture(), corner = fixture.from.vertices[0]!;
  await openFixture(page, fixture.document); await selectRoom(page, fixture.from.id);
  await page.getByRole('combobox', { name: 'Editing view', exact: true }).selectOption('left');
  await page.getByRole('combobox', { name: 'Grid spacing' }).selectOption('8');
  await page.getByRole('button', { name: 'Move room corner 1', exact: true }).focus(); await page.keyboard.press('ArrowRight');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('side-corner.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path))), room = saved.roomSource.rooms.find(r => r.id === fixture.from.id)!;
  expect(room.vertices).toContainEqual({ x: corner.x, z: corner.z + 8 });
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Redo', exact: true })).toBeEnabled();
});
