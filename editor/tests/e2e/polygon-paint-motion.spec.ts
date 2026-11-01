import { expect, test } from '@playwright/test';
import { movingControllerCopyFixture } from '../movingControllerCopyFixture';
import { applyCommand, editRoomSource } from '../../src/model/commands';
import { setRoomControllerCopies } from '../../src/model/movingControllerCopies';
import { mechanismDependencies } from '../../src/model/mechanismDependencies';
import { setPolygonMaterial } from '../../src/model/polygonMaterials';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { encodeProject } from '../../src/model/project';
import { engineMechanismState, engineMechanismTrace } from '../engineCapture';

for (const kind of ['door', 'lift'] as const) test(`painted copied ${kind} floor-bank surfaces retain synchronized native motion independently`, async ({ page }) => {
  test.setTimeout(60000);
  const input = movingControllerCopyFixture(kind);
  const base = applyCommand(input.document, editRoomSource(setRoomControllerCopies(input.document, [input.copy])));
  const dependency = mechanismDependencies(base, kind, input.mechanism.id).surface;
  const template = base.zones[dependency.room]!;
  const upper = !template.lower.some(record => record.offset === dependency.graphic);
  const record = (upper ? template.upper : template.lower).find(record => record.offset === dependency.graphic)!;
  expect([1, 2]).toContain(record.type);
  const source = applyCommand(base, editRoomSource(setPolygonMaterial(base,
    { room: input.copy.graphicsRooms.get(dependency.room)!, record: record.offset, upper },
    { texture: record.texture === 771 ? 0 : 771, scale: record.polygon!.scale })));
  const built = buildPlayableLevel(source, -1), copied = built.document.mechanisms.filter(item => item.kind === kind)[1]!;
  const zone = built.topology!.cells.find(cell => cell.sourceId === input.copy.room)!.id;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'painted-motion.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(source)) });
  await expect(page.locator('.project-picker strong')).toHaveText(source.name);
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await expect.poll(() => engineMechanismState(page, kind, 0)).toMatchObject({ current: input.mechanism.current, unsynchronizedWalls: 0 });
  const original = await engineMechanismState(page, kind, 0), trace = await engineMechanismTrace(page, kind, 1, false);
  expect(trace.at(-1)!.current).toBe(copied.top);
  expect(trace.some(state => state.current > copied.top && state.current < copied.bottom)).toBe(true);
  for (const state of trace) expect(state).toMatchObject({ zone, plane: state.current, unsynchronizedWalls: 0, wallCount: copied.walls.length });
  expect(await engineMechanismState(page, kind, 0)).toEqual(original);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('alert')).toHaveCount(0);
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
