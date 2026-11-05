import { expect, test, type Page, type TestInfo } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture } from '../roomHeightFixture';
import { decodeProject, encodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { roomSettings } from '../../src/model/roomHeights';
import { setUpperStorey, storeyMaterials } from '../../src/model/roomStoreys';
import { captureEngine, enginePlayerFeetFixed, enginePlayerState } from '../engineCapture';
import { applyCommand, editRoomAppearance, editRoomSource, setSourceStart } from '../../src/model/commands';
import { beginRoomDraft, commitRoomDraft, setDraftCorners, setRoomHeights } from '../../src/model/roomSource';
import { roomAppearanceChoices } from '../../src/model/roomAppearance';
import { createRoomOpening } from '../../src/model/roomOpening';
import { zoneContainsPoint } from '../../src/model/level';
import { setSkyCeiling } from '../../src/model/roomStoreys';

async function save(page: Page, info: TestInfo, name: string) {
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath(name); await (await pending).saveAs(path);
  return decodeProject(new Uint8Array(await readFile(path)));
}

test('floor and ceiling textures are directly editable from the inspector with cancel, Undo and portable saves', async ({ page }, info) => {
  const fixture = roomHeightFixture();
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'authoring.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  await expect(page.getByRole('button', { name: 'Edit floor texture', exact: true })).toBeInViewport();
  await expect(page.getByRole('button', { name: 'Edit ceiling texture', exact: true })).toBeInViewport();
  await page.locator('.inspector').screenshot({ path: info.outputPath('direct-floor-and-ceiling-controls.png') });
  for (const plane of ['floor', 'ceiling']) {
    await page.getByRole('button', { name: `Edit ${plane} texture`, exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Paint floor or ceiling material' });
    await expect(dialog.getByRole('heading')).toHaveText(`Lower ${plane} texture`);
    await dialog.getByRole('button', { name: 'Choose surface pattern 6', exact: true }).click();
    await dialog.getByRole('combobox', { name: 'Surface texture repeat', exact: true }).selectOption('-2');
    await dialog.getByRole('button', { name: 'Cancel surface paint', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: `Edit ${plane} texture`, exact: true }).click();
    await dialog.getByRole('button', { name: 'Choose surface pattern 6', exact: true }).click();
    await dialog.getByRole('combobox', { name: 'Surface texture repeat', exact: true }).selectOption('-2');
    await expect(dialog.locator('option[value=bump]')).toHaveJSProperty('disabled', true);
    await dialog.getByRole('button', { name: 'Apply surface material', exact: true }).click();
    const saved = await save(page, info, `${plane}.ab3d`), room = saved.roomSource.rooms.find(room => room.id === fixture.from.id)!;
    expect(room.polygonMaterials).toHaveLength(1); expect(room.polygonMaterials![0]).toMatchObject({ texture: 257, scale: -2, upper: false, mode: 'normal' });
    const built = buildPlayableLevel(saved, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
    expect(built.document.zones[cell.id]!.lower.find(record => record.type === (plane === 'floor' ? 1 : 2))!.polygon).toMatchObject({ texture: 257, scale: -2 });
    await page.getByRole('button', { name: 'Undo', exact: true }).click();
    expect((await save(page, info, `undo-${plane}.ab3d`)).roomSource).toEqual(fixture.document.roomSource);
  }
});

test('authors a rectangular skylight and can create a textured ceiling in an imported outdoor room', async ({ page }, info) => {
  const fixture = roomHeightFixture(), materials = storeyMaterials(fixture.document, fixture.from);
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'skylight.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  await page.getByText('Create sky opening', { exact: true }).click();
  await page.getByRole('combobox', { name: 'Storey wall style' }).selectOption(String(materials.walls[0]!.offset));
  await page.getByRole('combobox', { name: 'Storey corner lighting' }).selectOption(String(materials.walls[0]!.points[0]!));
  await page.getByRole('button', { name: 'Create opening', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  const skylight = await save(page, info, 'skylight.ab3d');
  expect(skylight.roomSource.rooms).toHaveLength(fixture.document.roomSource.rooms.length + 4);
  await expect(page.getByRole('checkbox', { name: 'Ceiling open to sky', exact: true })).toBeChecked();
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  expect((await save(page, info, 'undo-skylight.ab3d')).roomSource).toEqual(fixture.document.roomSource);
  const sky = applyCommand(fixture.document, editRoomSource(setSkyCeiling(fixture.document, fixture.from.id, false, true))), imported = buildPlayableLevel(sky, -1);
  const zone = imported.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!.id;
  await page.locator('input[type=file]').setInputFiles({ name: 'outdoor.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(imported.document)) });
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(zone).padStart(3, '0')}(?:\\s|$)`) }).click();
  const create = page.getByRole('combobox', { name: 'Create ceiling', exact: true });
  await create.selectOption((await create.locator('option').nth(1).getAttribute('value'))!);
  await page.getByRole('dialog').getByRole('button', { name: 'Choose surface pattern 6', exact: true }).click();
  await page.getByRole('button', { name: 'Apply surface material', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Edit ceiling texture', exact: true })).toBeEnabled();
  const saved = await save(page, info, 'created-ceiling.ab3d');
  expect(buildPlayableLevel(saved, -1).document.zones[zone]!.lower.some(record => record.type === 2 && record.texture === 257)).toBe(true);
  await expect(page.getByRole('alert')).toHaveCount(0);
});

test('native play walks from an upper landing onto the new upper floor and falls through its opening', async ({ page }, info) => {
  test.setTimeout(90000);
  const fixture = roomHeightFixture(), materials = storeyMaterials(fixture.document, fixture.from), settings = roomSettings(fixture.document, fixture.from);
  let document = applyCommand(fixture.document, editRoomSource(setUpperStorey(fixture.document, fixture.from.id,
    { floor: settings.ceiling, ceiling: settings.ceiling - (settings.floor - settings.ceiling), wall: materials.walls[0]!.offset, lightingPoint: materials.walls[0]!.points[0]! })));
  document = applyCommand(document, createRoomOpening(document, fixture.from.id, { left: 20128, right: 20224, top: 20064, bottom: 20192 },
    { lowerWall: materials.walls[0]!.offset, upperWall: null, lightingPoint: materials.walls[0]!.points[0]! }).command);
  document = applyCommand(document, editRoomSource(commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, fixture.from.recordZone),
    [{ x: 19744, z: 20000 }, { x: 20000, z: 20000 }, { x: 20000, z: 20256 }, { x: 19744, z: 20256 }]))));
  const landing = document.roomSource.rooms.at(-1)!, choices = roomAppearanceChoices(document, landing);
  document = applyCommand(document, editRoomAppearance(document, { room: landing.id, lowerWall: choices.lower[0]!.offset, upperWall: null, lightingPoint: choices.points[0]! }));
  document = applyCommand(document, editRoomSource(setRoomHeights(document, document.roomSource, landing.id,
    { floor: settings.ceiling, ceiling: settings.ceiling - (settings.floor - settings.ceiling), upperFloor: settings.upperFloor, upperCeiling: settings.upperCeiling })));
  document = applyCommand(document, setSourceStart(document, 1, landing.id, 19968, 20128));
  const built = buildPlayableLevel(document, -1), game = built.document;
  const west = game.zones.find(zone => zoneContainsPoint(game, zone, 20032, 20128))!;
  const shaft = game.zones.find(zone => zoneContainsPoint(game, zone, 20176, 20128))!;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'playable-opening.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await captureEngine(page, { zone: game.starts[0]!.zone, x: 19968, z: 20128, angle: 0, upper: 0 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  await canvas.focus(); await expect.poll(() => enginePlayerFeetFixed(page)).toBe(settings.ceiling * 256);
  await page.keyboard.down('d');
  try { await expect.poll(async () => (await enginePlayerState(page)).zone, { intervals: [10] }).toBe(west.id); } finally { await page.keyboard.up('d'); }
  await expect.poll(() => enginePlayerFeetFixed(page)).toBe(settings.ceiling * 256);
  await page.keyboard.down('d');
  try { await expect.poll(async () => (await enginePlayerState(page)).zone, { intervals: [10] }).toBe(shaft.id); } finally { await page.keyboard.up('d'); }
  await expect.poll(() => enginePlayerFeetFixed(page)).toBe(settings.floor * 256);
  await canvas.screenshot({ path: info.outputPath('native-opening-after-fall.png') });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
});

test('adds and textures an upper storey, opens its sky, and creates a saved native shaft through both storeys', async ({ page }, info) => {
  test.setTimeout(90000);
  const fixture = roomHeightFixture(), materials = storeyMaterials(fixture.document, fixture.from), baseline = roomSettings(fixture.document, fixture.from);
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'storey-authoring.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  await page.getByText('Add upper storey', { exact: true }).click();
  await page.getByRole('combobox', { name: 'Storey wall style' }).selectOption(String(materials.walls[0]!.offset));
  await page.getByRole('combobox', { name: 'Storey corner lighting' }).selectOption(String(materials.walls[0]!.points[0]!));
  await page.getByRole('button', { name: 'Create upper storey', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  const added = await save(page, info, 'upper.ab3d');
  expect(roomSettings(added, added.roomSource.rooms.find(room => room.id === fixture.from.id)!).upperFloor).toBe(baseline.ceiling);
  for (const plane of ['floor', 'ceiling']) {
    await page.getByRole('button', { name: `Edit upper ${plane} texture`, exact: true }).click();
    await page.getByRole('dialog').getByRole('button', { name: 'Choose surface pattern 6', exact: true }).click();
    await page.getByRole('button', { name: 'Apply surface material', exact: true }).click();
  }
  const painted = await save(page, info, 'upper-painted.ab3d');
  expect(painted.roomSource.rooms.find(room => room.id === fixture.from.id)!.polygonMaterials!.every(item => item.upper)).toBe(true);
  await page.getByRole('checkbox', { name: 'Upper ceiling open to sky', exact: true }).check();
  const sky = await save(page, info, 'upper-sky.ab3d'), builtSky = buildPlayableLevel(sky, -1), skyCell = builtSky.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!;
  expect(builtSky.document.zones[skyCell.id]!.upper.some(record => record.type === 12)).toBe(true);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  const camera = { zone: skyCell.id, x: 20128, z: 20128, upper: 1, angle: 0, height: baseline.ceiling - (baseline.floor - baseline.ceiling) + 4 };
  const skyFrame = await captureEngine(page, camera);
  const skyRevision = (await preview.getAttribute('data-revision'))!;
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(preview).not.toHaveAttribute('data-revision', skyRevision);
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect(await captureEngine(page, camera)).not.toBe(skyFrame);
  await page.getByText('Create opening between storeys', { exact: true }).click();
  for (const [field, value] of Object.entries({ left: 20064, right: 20192, top: 20064, bottom: 20192 })) await page.getByRole('spinbutton', { name: `Opening ${field}`, exact: true }).fill(String(value));
  await page.getByRole('combobox', { name: 'Storey wall style' }).selectOption(String(materials.walls[0]!.offset));
  await page.getByRole('combobox', { name: 'Storey corner lighting' }).selectOption(String(materials.walls[0]!.points[0]!));
  await page.getByRole('button', { name: 'Create opening', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  const opened = await save(page, info, 'opening.ab3d'); expect(opened.roomSource.rooms).toHaveLength(painted.roomSource.rooms.length + 4);
  const built = buildPlayableLevel(opened, -1), cell = built.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!;
  expect(built.document.zones[cell.id]!.upper).toHaveLength(0);
  expect(built.document.zones[cell.id]!.lower.find(record => record.type === 2)!.texture).toBe(257);
  expect(built.document.zones[cell.id]!.ceiling).toBe(baseline.ceiling - (baseline.floor - baseline.ceiling));
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect(await captureEngine(page, { ...camera, zone: cell.id, upper: 0 })).not.toBe(skyFrame);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  expect((await save(page, info, 'undo-opening.ab3d')).roomSource).toEqual(painted.roomSource);
  await page.locator('input[type=file]').setInputFiles({ name: 'reopened.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(opened)) });
  await expect(page.getByRole('alert')).toHaveCount(0);
});
