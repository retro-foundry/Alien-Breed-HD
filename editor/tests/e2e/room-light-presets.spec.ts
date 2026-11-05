import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { nativeRoomCopyFixture } from '../nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from '../roomHeightFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { captureEngine, enginePickDetails, engineCornerLights, type EnginePickSample } from '../engineCapture';
import { applyCommand, editRoomSource, setSourceStart } from '../../src/model/commands';
import { setRoomProperties } from '../../src/model/roomSource';
import { captureRoomProperties } from '../../src/model/roomProperties';
import { roomSettings } from '../../src/model/roomHeights';

test('whole-room presets shade the chosen storey and save as one Undo', async ({ page }, info) => {
  test.setTimeout(120000);
  const { document, copy: room } = nativeRoomCopyFixture(twoStoreyHeightTemplate()), built = buildPlayableLevel(document, -1);
  const cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!, output = built.document.zones[cell.id]!;
  const x = Math.round(room.vertices.reduce((sum, point) => sum + point.x, 0) / room.vertices.length), z = Math.round(room.vertices.reduce((sum, point) => sum + point.z, 0) / room.vertices.length);
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'room-presets.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${room.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await page.locator('summary').filter({ hasText: /^Corner lighting$/ }).click();
  const samples = Array.from({ length: 361 }, (_, index) => ({ x: (index % 19 + 1) / 20, y: (Math.floor(index / 19) + 1) / 20 }));
  for (const upper of [false, true]) {
    const camera = { zone: cell.id, x, z, upper: upper ? 1 : 0, angle: 0, height: (upper ? output.upperFloor : output.floor) - 8 };
    await captureEngine(page, camera); const before = await enginePickDetails(page, samples);
    await page.getByRole('combobox', { name: 'Lighting preset storey', exact: true }).selectOption(upper ? 'upper' : 'lower');
    await page.getByRole('combobox', { name: 'Room lighting preset', exact: true }).selectOption('5');
    const revision = await preview.getAttribute('data-revision');
    await page.getByRole('button', { name: 'Apply room lighting preset', exact: true }).click();
    await expect(preview).not.toHaveAttribute('data-revision', revision!); await expect(preview.getByRole('status')).toContainText('zones loaded');
    await captureEngine(page, camera); const after = await enginePickDetails(page, samples);
    const changed = after.filter((pick, index) => pick.pixel !== before[index]!.pixel);
    expect(changed.length).toBeGreaterThan(0);
    for (const pick of changed) expect([1, 2]).toContain(pick.surface?.type);
    const undoRevision = await preview.getAttribute('data-revision');
    await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(preview).not.toHaveAttribute('data-revision', undoRevision!); await expect(preview.getByRole('status')).toContainText('zones loaded');
    await captureEngine(page, camera); expect((await enginePickDetails(page, samples)).map(pick => pick.pixel)).toEqual(before.map(pick => pick.pixel));
    const redoRevision = await preview.getAttribute('data-revision');
    await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(preview).not.toHaveAttribute('data-revision', redoRevision!); await expect(preview.getByRole('status')).toContainText('zones loaded');
  }
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('room-light-presets.ab3d'); await (await pending).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  const lights = saved.roomSource.rooms.find(item => item.id === room.id)!.cornerLights!;
  expect(lights).toHaveLength(room.vertices.length * 4);
  expect(saved.files).toEqual(document.files);
  await expect(preview.getByRole('alert')).toHaveCount(0);
});

test('all glowing presets save their original selectors and animate in the native engine', async ({ page }, info) => {
  test.setTimeout(180000);
  const fixture = nativeRoomCopyFixture(twoStoreyHeightTemplate()), room = fixture.copy;
  const x = Math.round(room.vertices.reduce((sum, point) => sum + point.x, 0) / room.vertices.length), z = Math.round(room.vertices.reduce((sum, point) => sum + point.z, 0) / room.vertices.length);
  const positioned = applyCommand(fixture.document, setSourceStart(fixture.document, 1, room.id, x, z));
  // Use a midrange room shade so the original negative glow samples are
  // visible instead of being clipped at the palette's brightest endpoint.
  const document = applyCommand(positioned, editRoomSource(setRoomProperties(positioned.roomSource, room.id, {
    ...captureRoomProperties(roomSettings(positioned, room)), brightness: 8, upperBrightness: 8,
  })));
  const built = buildPlayableLevel(document, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!, output = built.document.zones[cell.id]!;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'glowing-room.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${room.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await page.locator('summary').filter({ hasText: /^Corner lighting$/ }).click();
  for (const upper of [false, true]) for (const preset of [6, 7, 8, 9]) {
    await page.getByRole('combobox', { name: 'Lighting preset storey', exact: true }).selectOption(upper ? 'upper' : 'lower');
    await page.getByRole('combobox', { name: 'Room lighting preset', exact: true }).selectOption(String(preset));
    const revision = await preview.getAttribute('data-revision');
    await page.getByRole('button', { name: 'Apply room lighting preset', exact: true }).click();
    await expect(preview).not.toHaveAttribute('data-revision', revision!); await expect(preview.getByRole('status')).toContainText('zones loaded');
    expect(await captureEngine(page, { zone: cell.id, x, z, upper: upper ? 1 : 0, angle: 0, height: (upper ? output.upperFloor : output.floor) - 8 })).toBeGreaterThan(0);
    await expect(preview.getByRole('alert')).toHaveCount(0);
  }
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('glowing-room.ab3d'); await (await pending).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  const lights = saved.roomSource.rooms.find(item => item.id === room.id)!.cornerLights!;
  expect(lights).toHaveLength(room.vertices.length * 4);
  expect(lights.some(light => ((light.lowerBrightness >>> 8) & 15) >= 4 || ((light.upperBrightness >>> 8) & 15) >= 4)).toBe(true);
  const samples = Array.from({ length: 361 }, (_, index) => ({ x: (index % 19 + 1) / 20, y: (Math.floor(index / 19) + 1) / 20 }));
  const cameras = [false, true].flatMap(upper => [(upper ? output.upperFloor : output.floor) - 8, (upper ? output.upperCeiling : output.ceiling) + 8].flatMap(height => [0, 2048, 4096, 6144].map(angle => ({ zone: cell.id, x, z, upper: upper ? 1 : 0, angle, height }))));
  const lowerFloor = document.zones[room.recordZone]!.lower.find(face => face.type === 1)!;
  await page.getByRole('combobox', { name: 'Corner lighting surfaces', exact: true }).selectOption(String(lowerFloor.offset));
  const commit = async (button: string) => {
    const revision = await preview.getAttribute('data-revision');
    await page.getByRole('button', { name: button, exact: true }).click();
    await expect(preview).not.toHaveAttribute('data-revision', revision!); await expect(preview.getByRole('status')).toContainText('zones loaded');
  };
  await page.getByRole('spinbutton', { name: 'Lower corner brightness', exact: true }).fill('0');
  await page.getByRole('combobox', { name: 'Lower corner animation', exact: true }).selectOption('0');
  await commit('Apply corner lighting');
  await captureEngine(page, cameras[0]!); const steady = await enginePickDetails(page, samples);
  await page.getByRole('spinbutton', { name: 'Lower corner brightness', exact: true }).fill('96');
  await page.getByRole('group', { name: 'Lower storey', exact: true }).locator('summary').filter({ hasText: /^Original animation selector$/ }).click();
  await page.getByRole('spinbutton', { name: 'Lower original corner selector', exact: true }).fill('15');
  await page.getByRole('combobox', { name: 'Lower corner animation strength', exact: true }).selectOption('16');
  await commit('Apply corner lighting');
  await captureEngine(page, cameras[0]!);
  // Full-strength selector 15 must target the original table's zero entry,
  // exactly matching Static 0, rather than retaining the base 96.
  expect((await enginePickDetails(page, samples)).map(pick => pick.pixel)).toEqual(steady.map(pick => pick.pixel));
  await commit('Undo'); await commit('Undo');
  const baseline: EnginePickSample[][] = [];
  for (const camera of cameras) { await captureEngine(page, camera); baseline.push(await enginePickDetails(page, samples)); }
  const savedBuild = buildPlayableLevel(saved, -1), savedZone = savedBuild.document.zones[cell.id]!;
  const pointIds = [...new Set([...savedZone.lower, ...savedZone.upper].filter(face => [1, 2].includes(face.type)).flatMap(face => face.points))];
  const beforeNative = await engineCornerLights(page, pointIds);
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  // Observe real frames rather than injecting animation values or stepping a
  // separate test simulation.
  await expect.poll(async () => {
    const values = await engineCornerLights(page, pointIds);
    return values.some((point, index) => point.lowerValue !== beforeNative[index]!.lowerValue || point.upperValue !== beforeNative[index]!.upperValue);
  }).toBe(true);
  // Toolbar Stop reloads the authored artifact. Freeze through the existing
  // play bridge first, keeping the React playtest mounted for frame inspection.
  await page.evaluate(() => {
    window.document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!.contentWindow!.postMessage({ source: 'ab3d-editor', type: 'play', playing: false }, location.origin);
  });
  const pausedNative = await engineCornerLights(page, pointIds);
  expect(pausedNative.some((point, index) => point.lowerValue !== beforeNative[index]!.lowerValue || point.upperValue !== beforeNative[index]!.upperValue)).toBe(true);
  const changedStoreys = [false, false];
  const changedPixels: number[] = [];
  // enginePickDetails reads renderer_get_cw_buffer before SDL's hit-flash
  // overlay; gameplay damage tint cannot satisfy these lighting comparisons.
  for (const [index, camera] of cameras.entries()) {
    await captureEngine(page, camera); const animated = await enginePickDetails(page, samples);
    const changed = animated.filter((pick, at) => [1, 2].includes(pick.surface?.type ?? -1) && pick.pixel !== baseline[index]![at]!.pixel).length;
    changedPixels.push(changed); changedStoreys[camera.upper] ||= changed > 0;
  }
  await info.attach('native-glow-state', { contentType: 'application/json', body: JSON.stringify({ beforeNative, afterNative: await engineCornerLights(page, pointIds), cameras, changedPixels }) });
  expect(changedStoreys).toEqual([true, true]);
  await page.screenshot({ path: info.outputPath('glowing-room.png') });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('status')).toContainText('zones loaded');
});
