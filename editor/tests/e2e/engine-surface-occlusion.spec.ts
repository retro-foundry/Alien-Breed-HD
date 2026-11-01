import { expect, test } from '@playwright/test';
import { roomHeightFixture } from '../roomHeightFixture';
import { originalFiles } from '../helpers';
import { openLevel } from '../../src/model/level';
import { encodeProject } from '../../src/model/project';
import { createObject, floorMatchedObjectHeight, objectTemplates } from '../../src/model/objectAuthoring';
import { applyCommand } from '../../src/model/commands';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { captureEngine, enginePickDetails } from '../engineCapture';

for (const polygon of [false, true]) test(`${polygon ? 'polygon objects' : 'opaque sprite texels'} hide surfaces without covering transparent pixels`, async ({ page }, testInfo) => {
  const original = openLevel('Surface occlusion', originalFiles(polygon ? 8 : 1), 0, -1);
  const fixture = roomHeightFixture(0, original), baseline = fixture.document;
  const template = objectTemplates(baseline).find(object => !object.upper && (polygon
    ? baseline.files.data[object.offset + 6] === 255 : object.type === 0 && baseline.files.data[object.offset + 6] !== 255))!;
  expect(template).toBeDefined();
  const created = createObject(baseline, { template: template.id, room: fixture.from.id, x: 20128, z: 20180,
    y: floorMatchedObjectHeight(baseline, template, fixture.from.id, false), upper: false });
  const source = { ...applyCommand(baseline, created.command), name: `${baseline.name} with actor` };
  const built = buildPlayableLevel(source, -1);
  const camera = { zone: built.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!.id, x: 20128, z: 20032, upper: 0, angle: 0 };
  const points = Array.from({ length: 441 }, (_, index) => ({ x: (index % 21 + 1) / 22, y: (Math.floor(index / 21) + 1) / 22 }));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  const captures: Awaited<ReturnType<typeof enginePickDetails>>[] = [], hashes: number[] = [];
  for (const document of [baseline, source]) {
    await page.locator('input[type=file]').setInputFiles({ name: 'occlusion.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
    await expect(page.locator('.project-picker strong')).toHaveText(document.name);
    if (captures.length === 0) await page.getByRole('button', { name: '3D view', exact: true }).click();
    await expect(page.getByRole('region', { name: '3D game engine preview' }).getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
    hashes.push(await captureEngine(page, camera)); captures.push(await enginePickDetails(page, points));
  }
  expect(hashes[0]).not.toBe(hashes[1]);
  const before = captures[0]!, after = captures[1]!;
  const changed = after.flatMap((pick, index) => before[index]!.surface && before[index]!.pixel !== pick.pixel ? [index] : []);
  expect(changed.length).toBeGreaterThan(0);
  for (const index of changed) expect(after[index]!.surface).toBeNull();
  const bounds = { minX: Math.min(...changed.map(index => points[index]!.x)), maxX: Math.max(...changed.map(index => points[index]!.x)),
    minY: Math.min(...changed.map(index => points[index]!.y)), maxY: Math.max(...changed.map(index => points[index]!.y)) };
  // Pixels inside the object's observed silhouette bounds can still be holes.
  // The actual colour raster, not a projected bounding box, identifies coverage.
  expect(after.filter((pick, index) => points[index]!.x > bounds.minX && points[index]!.x < bounds.maxX
    && points[index]!.y > bounds.minY && points[index]!.y < bounds.maxY
    && pick.pixel === before[index]!.pixel && pick.surface?.offset === before[index]!.surface?.offset && pick.surface).length).toBeGreaterThan(0);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot({ path: testInfo.outputPath('object-occludes-surface.png') });
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
