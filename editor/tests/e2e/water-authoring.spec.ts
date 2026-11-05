import { expect, test, type Page, type TestInfo } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture } from '../roomHeightFixture';
import { roomSettings } from '../../src/model/roomHeights';
import { waterMaterialChoices } from '../../src/model/sourceWater';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { importedAnimationTables } from '../../src/model/animationBuild';
import { decodeProject, encodeProject } from '../../src/model/project';
import { captureEngine, enginePickDetails, engineWaterState } from '../engineCapture';
import { sourceWaterFixture } from '../sourceWaterFixture';
import { openLevel } from '../../src/model/level';
import { originalFiles } from '../helpers';
import { buildGraphicsStreams } from '../../src/model/graphicsBuild';

async function save(page: Page, info: TestInfo, name: string) {
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath(name); await (await pending).saveAs(path);
  return { path, document: decodeProject(new Uint8Array(await readFile(path))) };
}
async function openWater(page: Page) {
  const details = page.locator('details').filter({ has: page.getByText('Water and animation groups', { exact: true }) });
  if (await details.getAttribute('open') === null) await page.getByText('Water and animation groups', { exact: true }).click();
}
async function openGroups(page: Page) {
  const details = page.locator('details').filter({ has: page.getByText('Create or edit water animation group', { exact: true }) }).last();
  if (await details.getAttribute('open') === null) await page.getByText('Create or edit water animation group', { exact: true }).click();
}

test('authors and saves shared water motion, validates groups, and natively pauses all assigned planes', async ({ page }, info) => {
  test.setTimeout(120000);
  page.on('dialog', dialog => void dialog.accept());
  const fixture = roomHeightFixture(), floor = roomSettings(fixture.document, fixture.from).floor, group = 5;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'water-authoring.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  await openWater(page); await openGroups(page);
  await page.getByRole('combobox', { name: 'Water animation group to edit' }).selectOption(String(group));
  for (const [key, value] of [['top', floor - 16], ['bottom', floor - 40], ['starting', floor - 32]] as const) await page.getByRole('spinbutton', { name: `Water group ${key} height` }).fill(String(value));
  await page.getByRole('button', { name: 'Apply water group', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('top water height');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  for (const [key, value] of [['top', floor - 40], ['bottom', floor - 16], ['starting', floor - 32]] as const) await page.getByRole('spinbutton', { name: `Water group ${key} height` }).fill(String(value));
  await page.getByRole('button', { name: 'Apply water group', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  for (const room of [fixture.from, fixture.to]) {
    await page.getByRole('button', { name: new RegExp(`^Room ${room.id.replace(/^room-/, '')}`) }).click(); await openWater(page);
    await page.getByRole('combobox', { name: 'Water coverage and pattern' }).selectOption(String(waterMaterialChoices(fixture.document, room)[0]!.offset));
    await page.getByRole('combobox', { name: 'Room water animation' }).selectOption(String(group));
    await page.getByRole('button', { name: 'Apply room water', exact: true }).click(); await expect(page.getByRole('alert')).toHaveCount(0);
  }
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Inherit template water', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  const saved = await save(page, info, 'shared-water.ab3d'), built = buildPlayableLevel(saved.document, -1);
  expect(saved.document.roomSource.waterGroups).toMatchObject([{ id: group, top: floor - 40, bottom: floor - 16, current: floor - 32, enabled: true }]);
  const entries = importedAnimationTables(built.document).tables.water[group]!.entries;
  expect(entries).toHaveLength(2);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', String(saved.document.playtestSlot), { timeout: 60000 });
  const from = built.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!;
  const eye = Math.max(built.document.zones[from.id]!.ceiling + 1, floor - 40);
  expect(await captureEngine(page, { zone: from.id, x: 20128, z: 20128, upper: 0, angle: 0, height: eye })).toBeGreaterThan(0);
  const points = Array.from({ length: 361 }, (_, index) => ({ x: (index % 19 + 1) / 20, y: (Math.floor(index / 19) + 1) / 20 }));
  const picks = await enginePickDetails(page, points), waterPick = picks.findIndex(pick => pick.surface?.type === 7 && !pick.object);
  await info.attach('water-render-state', { body: JSON.stringify({ eye, floor, ceiling: built.document.zones[from.id]!.ceiling, water: built.document.zones[from.id]!.water, surfaces: [...new Set(picks.map(pick => pick.surface?.type))] }), contentType: 'application/json' });
  expect(waterPick).toBeGreaterThanOrEqual(0);
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'), rectangle = (await canvas.boundingBox())!, point = points[waterPick]!;
  await page.mouse.click(rectangle.x + point.x * rectangle.width, rectangle.y + point.y * rectangle.height);
  await expect(preview.getByRole('note', { name: 'Picked surface' })).toContainText('Water');
  await expect(preview.getByRole('note', { name: 'Picked surface' })).toContainText('Water and animation groups');
  await preview.screenshot({ path: info.outputPath('authored-native-water.png') });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const heights = new Set<number>();
  await expect.poll(async () => {
    const states = await engineWaterState(page, entries);
    for (const state of states) { expect(state.polygon).toBe(state.water); expect(state.water).toBe(states[0]!.water); expect(state.water).toBeGreaterThanOrEqual(floor - 40); expect(state.water).toBeLessThanOrEqual(floor - 16); }
    heights.add(states[0]!.water); return heights.size;
  }, { intervals: [30, 30, 30] }).toBeGreaterThan(2);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await page.locator('input[type=file]').setInputFiles(saved.path);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click(); await openWater(page); await openGroups(page);
  await expect(page.getByRole('combobox', { name: 'Room water animation' })).toHaveValue(String(group));
  await page.getByRole('combobox', { name: 'Water animation group to edit' }).selectOption(String(group));
  await expect(page.getByRole('button', { name: `Show water in Room ${fixture.to.id.replace(/^room-/, '')}`, exact: true })).toBeVisible();
  await page.getByRole('checkbox', { name: 'Animate water group', exact: true }).uncheck();
  await page.getByRole('button', { name: 'Apply water group', exact: true }).click();
  const paused = await save(page, info, 'paused-water.ab3d'), pausedBuilt = buildPlayableLevel(paused.document, -1);
  expect(importedAnimationTables(pausedBuilt.document).tables.water[group]!.entries).toHaveLength(0);
  const pausedEntries = entries.map(entry => ({ zone: entry.zone, graphic: pausedBuilt.document.zones[entry.zone]!.lower.find(record => record.type === 7)!.offset }));
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  for (let i = 0; i < 8; i++) {
    await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    for (const state of await engineWaterState(page, pausedEntries)) { expect(state.water).toBe(floor - 32); expect(state.polygon).toBe(state.water); }
  }
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await page.getByRole('button', { name: 'Remove room water', exact: true }).click();
  const removed = await save(page, info, 'removed-water.ab3d'), dryBuilt = buildPlayableLevel(removed.document, -1), dry = dryBuilt.document.zones[dryBuilt.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!.id]!;
  expect(dry.lower.some(record => record.type === 7)).toBe(false); expect(dry.water).toBe(dry.floor + 4);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Room water animation' })).toHaveValue(String(group));
});

test('native water starts in an authored level whose imported baseline has no lift or animation table', async ({ page }) => {
  const original = openLevel('No native animation tables', originalFiles(1)), tables = importedAnimationTables(original).tables;
  const empty = openLevel(original.name, { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined, { animations: { ...tables, doors: null, lifts: null, switches: null, water: [] } }) });
  const fixture = sourceWaterFixture(empty), built = buildPlayableLevel(fixture.document, -1), native = importedAnimationTables(built.document).tables;
  expect(native.lifts).toEqual([]); expect(native.water).toHaveLength(21);
  const entries = native.water[fixture.group.id]!.entries; expect(entries).toHaveLength(2);
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'new-water-no-lifts.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', String(fixture.document.playtestSlot), { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const seen = new Set<number>();
  await expect.poll(async () => {
    const states = await engineWaterState(page, entries);
    expect(states[0]!.water).toBe(states[1]!.water);
    for (const state of states) expect(state.polygon).toBe(state.water);
    seen.add(states[0]!.water); return seen.size;
  }, { intervals: [30, 30, 30] }).toBeGreaterThan(2);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
});
