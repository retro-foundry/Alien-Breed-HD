import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { openLevel } from '../../src/model/level';
import { originalFiles } from '../helpers';
import { movingRoomOwnership } from '../../src/model/movingRoomEligibility';
import { importedMovingFixture } from '../importedMovingFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { compileRoomTopology } from '../../src/model/roomTopology';
import { applyCommand, setSourceStart } from '../../src/model/commands';
import { engineMechanismState, engineMechanismTrace, captureEngine, enginePickDetails } from '../engineCapture';

for (const kind of ['door', 'lift'] as const) test(`convert an original room to a saved native ${kind} without replacing its identity`, async ({ page }, info) => {
  test.setTimeout(90000);
  const fixture = importedMovingFixture(1, kind), topology = compileRoomTopology(fixture.document, fixture.document.roomSource);
  const cell = topology.cells.find(cell => cell.sourceId === fixture.room.id)!;
  const neighbour = topology.cells.find(target => cell.neighbours.includes(target.id) && target.sourceId !== fixture.room.id)!;
  const points = neighbour.vertices.map(id => topology.points[id]!);
  const document = applyCommand(fixture.document, setSourceStart(fixture.document, 1, neighbour.sourceId,
    Math.round(points.reduce((sum, point) => sum + point.x, 0) / points.length), Math.round(points.reduce((sum, point) => sum + point.z, 0) / points.length)));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'convert.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(fixture.room.originZone).padStart(3, '0')}(?:\\s|$)`) }).click();
  const section = page.getByRole('region', { name: 'Create door or lift', exact: true });
  await section.getByRole('button', { name: `Make ${kind}`, exact: true }).click();
  await section.getByRole('spinbutton', { name: 'New mechanism top', exact: true }).fill(String(fixture.mechanism.top));
  await section.getByRole('spinbutton', { name: 'New mechanism bottom', exact: true }).fill(String(fixture.mechanism.bottom));
  await section.getByRole('combobox', { name: 'New mechanism opening rule', exact: true }).selectOption(kind === 'door' ? '4' : '2');
  await section.getByRole('combobox', { name: 'New mechanism closing rule', exact: true }).selectOption(kind === 'door' ? '0' : '2');
  await section.getByRole('button', { name: 'Apply mechanism', exact: true }).click();
  await expect(section.getByRole('heading', { level: 3 })).toHaveText(`Authored ${kind}`);
  await expect(section.getByRole('alert')).toHaveCount(0);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const file = info.outputPath('converted.ab3d'); await (await download).saveAs(file);
  const saved = decodeProject(new Uint8Array(await readFile(file))), built = buildPlayableLevel(saved, -1);
  expect(saved.roomSource.rooms.find(room => room.id === fixture.room.id)!.originZone).toBe(fixture.room.originZone);
  const mechanism = built.document.mechanisms.filter(item => item.kind === kind).at(-1)!;
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(section.getByRole('button', { name: `Make ${kind}`, exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  expect(await engineMechanismState(page, kind, mechanism.id)).toMatchObject({ current: mechanism.bottom });
  if (kind === 'door') {
    const room = built.topology!.cells.find(cell => cell.id === mechanism.zone)!;
    const outside = built.topology!.cells.find(cell => cell.sourceId === neighbour.sourceId)!;
    const center = (cell: typeof room) => { const points = cell.vertices.map(id => built.topology!.points[id]!); return { x: Math.round(points.reduce((sum, point) => sum + point.x, 0) / points.length), z: Math.round(points.reduce((sum, point) => sum + point.z, 0) / points.length) }; };
    const a = center(outside), b = center(room), dimensions = built.document.zones[outside.id]!;
    await captureEngine(page, { zone: outside.id, ...a, upper: 0, angle: Math.round(Math.atan2(b.x - a.x, b.z - a.z) * 8192 / (Math.PI * 2)) & 8191, height: (dimensions.floor + dimensions.ceiling) / 2 });
    const samples = Array.from({ length: 361 }, (_, index) => ({ x: (index % 19 + 1) / 20, y: (Math.floor(index / 19) + 1) / 20 }));
    const pixels = await enginePickDetails(page, samples), index = pixels.findIndex(pixel => mechanism.walls.some(wall => wall.graphic === pixel.surface?.offset));
    expect(index).toBeGreaterThanOrEqual(0);
    const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'), rect = (await canvas.boundingBox())!;
    await canvas.click({ position: { x: samples[index]!.x * rect.width, y: samples[index]!.y * rect.height } });
    await expect(page.getByRole('button', { name: 'Paint this wall', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Edit door', exact: true }).click();
    await expect(section.getByRole('heading', { level: 3 })).toHaveText('Authored door');
  }

  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const trace = await engineMechanismTrace(page, kind, mechanism.id, true);
  expect(trace.some(state => state.current > mechanism.top && state.current < mechanism.bottom)).toBe(true);
  for (const state of trace) expect(state).toMatchObject({ plane: state.current, unsynchronizedWalls: 0 });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await section.getByRole('button', { name: 'Remove mechanism', exact: true }).click();
  await expect(section.getByRole('button', { name: `Make ${kind}`, exact: true })).toBeVisible();
});


test('a room already owned by native movement links to its existing controller without edits', async ({ page }) => {
  const document = openLevel('Native ownership', originalFiles(1));
  const room = document.roomSource.rooms.find(room => { const owner = movingRoomOwnership(document, room); return owner?.room && owner.room !== room.id; })!;
  expect(room).toBeDefined(); const owner = movingRoomOwnership(document, room)!;
  const target = document.roomSource.rooms.find(room => room.id === owner.room)!;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(room.originZone).padStart(3, '0')}(?:\\s|$)`) }).click();
  await expect(page.getByRole('button', { name: 'Make door', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: `Go to existing ${owner.kind}`, exact: true }).click();
  await expect(page.locator('.inspector-heading h2')).toHaveText(`Zone ${target.originZone}`);
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
