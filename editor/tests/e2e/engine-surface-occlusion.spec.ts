import { expect, test } from '@playwright/test';
import { roomHeightFixture, twoStoreyHeightTemplate } from '../roomHeightFixture';
import { originalFiles } from '../helpers';
import { openLevel } from '../../src/model/level';
import { encodeProject } from '../../src/model/project';
import { createObject, floorMatchedObjectHeight, objectTemplates } from '../../src/model/objectAuthoring';
import { applyCommand } from '../../src/model/commands';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { captureEngine, engineObjectState, enginePickDetails, enginePlayerState } from '../engineCapture';

for (const { polygon, upper } of [{ polygon: false, upper: false }, { polygon: true, upper: false }, { polygon: false, upper: true }])
test(`${upper ? 'upper-floor ' : ''}${polygon ? 'polygon objects' : 'opaque sprite texels'} select visible objects and preserve picks through transparent pixels`, async ({ page }, testInfo) => {
  const original = openLevel('Surface occlusion', originalFiles(polygon ? 8 : 1), 0, -1);
  const fixture = roomHeightFixture(upper ? twoStoreyHeightTemplate() : 0, original), baseline = fixture.document;
  const template = objectTemplates(baseline).find(object => !object.upper && (polygon
    ? baseline.files.data[object.offset + 6] === 255 : object.type === 0 && baseline.files.data[object.offset + 6] !== 255))!;
  expect(template).toBeDefined();
  const created = createObject(baseline, { template: template.id, room: fixture.from.id, x: 20128, z: 20180,
    y: floorMatchedObjectHeight(baseline, template, fixture.from.id, upper), upper });
  const source = { ...applyCommand(baseline, created.command), name: `${baseline.name} with actor` };
  const built = buildPlayableLevel(source, -1);
  const camera = { zone: built.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!.id, x: 20128, z: 20032, upper: upper ? 1 : 0, angle: 0 };
  const points = Array.from({ length: 441 }, (_, index) => ({ x: (index % 21 + 1) / 22, y: (Math.floor(index / 21) + 1) / 22 }));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  const captures: Awaited<ReturnType<typeof enginePickDetails>>[] = [], hashes: number[] = [];
  for (const document of [baseline, source]) {
    await page.locator('input[type=file]').setInputFiles({ name: 'occlusion.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
    await expect(page.locator('.project-picker strong')).toHaveText(document.name);
    if (captures.length === 0) await page.getByRole('button', { name: 'Engine view', exact: true }).click();
    await expect(page.getByRole('region', { name: '3D game engine preview' }).getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
    hashes.push(await captureEngine(page, camera)); captures.push(await enginePickDetails(page, points));
  }
  expect(hashes[0]).not.toBe(hashes[1]);
  const before = captures[0]!, after = captures[1]!;
  const changed = after.flatMap((pick, index) => before[index]!.surface && before[index]!.pixel !== pick.pixel ? [index] : []);
  expect(changed.length).toBeGreaterThan(0);
  for (const index of changed) {
    expect(after[index]!.surface).toBeNull();
    expect(after[index]!.object).toEqual({ id: created.id, cid: source.objects[created.id]!.cid });
  }
  const bounds = { minX: Math.min(...changed.map(index => points[index]!.x)), maxX: Math.max(...changed.map(index => points[index]!.x)),
    minY: Math.min(...changed.map(index => points[index]!.y)), maxY: Math.max(...changed.map(index => points[index]!.y)) };
  // Pixels inside the object's observed silhouette bounds can still be holes.
  // The actual colour raster, not a projected bounding box, identifies coverage.
  const holes = after.flatMap((pick, index) => points[index]!.x > bounds.minX && points[index]!.x < bounds.maxX
    && points[index]!.y > bounds.minY && points[index]!.y < bounds.maxY
    && pick.pixel === before[index]!.pixel && pick.surface?.offset === before[index]!.surface?.offset && pick.surface ? [index] : []);
  expect(holes.length).toBeGreaterThan(0);
  for (const index of holes) expect(after[index]!.object).toBeNull();
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'), rect = (await canvas.boundingBox())!;
  const point = points[changed[Math.floor(changed.length / 2)]!]!, position = await enginePlayerState(page);
  await page.getByRole('textbox', { name: 'Search map contents' }).fill('no-matching-record');
  await canvas.click({ position: { x: point.x * rect.width, y: point.y * rect.height } });
  await expect(page.locator('.inspector-heading')).toContainText(`Object ${created.id}`);
  await expect(page.getByRole('textbox', { name: 'Search map contents' })).toHaveValue('');
  await expect(page.getByRole('tab', { name: 'Objects', exact: true })).toHaveAttribute('aria-selected', 'true');
  await expect(page.locator('.content-row[aria-pressed="true"]')).toContainText(String(created.id));
  await expect(page.locator('.content-row[aria-pressed="true"]')).toBeInViewport();
  await expect(canvas).toBeFocused();
  expect(await page.evaluate(() => window.scrollY)).toBe(0);
  await expect(page.locator('.inspector').getByText(upper ? 'Upper' : 'Lower', { exact: true })).toBeVisible();
  expect(await enginePlayerState(page)).toEqual(position);
  await expect(page.getByRole('note', { name: 'Picked surface' })).toHaveCount(0);
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  const oldHeight = source.objects[created.id]!.y, revision = await preview.getAttribute('data-revision');
  const height = page.getByRole('spinbutton', { name: 'Height', exact: true });
  await height.fill(String(oldHeight + .5)); await height.press('Tab');
  await expect(preview).not.toHaveAttribute('data-revision', revision!);
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect((await engineObjectState(page, created.id)).yFixed).toBe((oldHeight + .5) * 256);
  expect((await engineObjectState(page, template.id)).yFixed).toBe(template.y * 256);
  const changedRevision = await preview.getAttribute('data-revision');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(preview).not.toHaveAttribute('data-revision', changedRevision!);
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect((await engineObjectState(page, created.id)).yFixed).toBe(oldHeight * 256);
  await captureEngine(page, camera);
  const hole = points[holes[0]!]!;
  await canvas.click({ position: { x: hole.x * rect.width, y: hole.y * rect.height } });
  await expect(page.getByRole('note', { name: 'Picked surface' })).toBeVisible();
  await expect(page.locator('.inspector-heading')).not.toContainText(`Object ${created.id}`);
  expect(await enginePlayerState(page)).toEqual(position);
  await canvas.screenshot({ path: testInfo.outputPath('object-occludes-surface.png') });
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
