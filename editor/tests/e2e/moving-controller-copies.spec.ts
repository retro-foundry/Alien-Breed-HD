import { expect, test } from '@playwright/test';
import { movingControllerCopyFixture } from '../movingControllerCopyFixture';
import { buildSourceLevel } from '../../src/model/sourceBuild';
import { roomAppearanceBuildOptions } from '../../src/model/roomAppearance';
import { encodeProject } from '../../src/model/project';
import { engineMechanismState, engineMechanismTrace } from '../engineCapture';

for (const kind of ['door', 'lift'] as const) test(`a copied ${kind} moves its own walls and plane while the original remains still`, async ({ page }) => {
  test.setTimeout(90000);
  const { document, copy, mechanism } = movingControllerCopyFixture(kind);
  const built = buildSourceLevel(document, { resolvedExit: -1, ...roomAppearanceBuildOptions(document), controllerCopies: [copy] });
  const controllers = built.document.mechanisms.filter(item => item.kind === kind), copied = controllers[1]!;
  const copyZone = built.topology.cells.find(cell => cell.sourceId === copy.room)!.id;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'independent-motion.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(built.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(built.document.name);
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await engineMechanismState(page, kind, 0)).toMatchObject({ current: mechanism.current });
  expect(await engineMechanismState(page, kind, 1)).toMatchObject({ zone: copyZone, current: mechanism.current, top: copied.top });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  // Native controller routines synchronize authored plane/wall records on the
  // first logic update; paused imported geometry can precede that initialization.
  await expect.poll(() => engineMechanismState(page, kind, 0), { intervals: [10, 10, 10] })
    .toMatchObject({ current: mechanism.current, plane: mechanism.current, unsynchronizedWalls: 0 });
  const original = await engineMechanismState(page, kind, 0);
  const trace = await engineMechanismTrace(page, kind, 1, false);
  expect(trace.at(-1)!.current).toBe(copied.top);
  expect(trace.some(state => state.current > copied.top && state.current < copied.bottom)).toBe(true);
  for (const state of trace) {
    expect(state.current).toBeGreaterThanOrEqual(copied.top); expect(state.current).toBeLessThanOrEqual(copied.bottom);
    expect(state).toMatchObject({ plane: state.current, unsynchronizedWalls: 0, wallCount: copied.walls.length, zone: copyZone });
  }
  expect(await engineMechanismState(page, kind, 0)).toEqual(original);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
