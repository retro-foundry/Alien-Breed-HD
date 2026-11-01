import { expect, test } from '@playwright/test';
import { stairFixture } from '../stairFixture';
import { createStairFlight } from '../../src/model/stairs';
import { applyCommand } from '../../src/model/commands';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { encodeProject } from '../../src/model/project';
import { captureEngine, enginePickDetails, enginePickSamples, enginePlayerState } from '../engineCapture';
import { resolveEngineSurface } from '../../src/model/engineSurfacePick';
import { openLevel } from '../../src/model/level';
import { originalFiles } from '../helpers';

test('clicked native pixels select authored rooms without moving the inspection camera', async ({ page }, testInfo) => {
  const fixture = stairFixture(), source = applyCommand(fixture.document, createStairFlight(fixture.document, fixture.flight).command);
  const built = buildPlayableLevel(source, -1);
  const camera = { zone: built.document.starts[0]!.zone, x: 20064, z: 20064, angle: 0, upper: 0 };
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'pick-stairs.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(source)) });
  await expect(page.locator('.project-picker strong')).toHaveText(source.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await captureEngine(page, camera);
  const position = await enginePlayerState(page);
  const points = Array.from({ length: 81 }, (_, index) => ({ x: (index % 9 + 1) / 10, y: (Math.floor(index / 9) + 1) / 10 }));
  const details = await enginePickDetails(page, points), zones = details.map(pick => pick.zone), center = (await enginePickSamples(page, [{ x: .5, y: .5 }]))[0]!;
  const types = new Set<number>();
  for (const pick of details) if (pick.zone >= 0) {
    expect(pick.surface).not.toBeNull();
    const resolved = resolveEngineSurface(built.document, pick.zone, pick.surface);
    expect(resolved).toBeDefined(); expect(resolved!.upper).toBe(false); types.add(resolved!.type);
  }
  expect([...types].sort()).toEqual([0, 1]);
  const visible = [...new Set(zones.filter(zone => zone >= 0))];
  expect(visible.length).toBeGreaterThan(1);
  const authored = new Set(built.topology!.cells.filter(cell => cell.originZone === null).map(cell => cell.id));
  for (const zone of visible) expect(authored.has(zone)).toBe(true);
  const index = zones.findIndex(zone => zone >= 0 && zone !== center); expect(index).toBeGreaterThanOrEqual(0);
  const point = points[index]!, target = built.topology!.cells.find(cell => cell.id === zones[index])!.sourceId;
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  const rect = (await canvas.boundingBox())!;
  await canvas.click({ position: { x: point.x * rect.width, y: point.y * rect.height } });
  await expect(page.getByRole('button', { name: new RegExp(`^Room ${target.replace(/^room-/, '')}(?:\\s|$)`) })).toHaveAttribute('aria-pressed', 'true');
  const surface = page.getByRole('note', { name: 'Picked surface' });
  await expect(surface).toHaveAttribute('data-graphic', String(details[index]!.surface!.offset));
  await expect(surface).toContainText(resolveEngineSurface(built.document, zones[index]!, details[index]!.surface)!.kind);
  expect(await enginePlayerState(page)).toEqual(position);
  await canvas.press('Enter');
  const centered = built.topology!.cells.find(cell => cell.id === center)!.sourceId;
  await expect(page.getByRole('button', { name: new RegExp(`^Room ${centered.replace(/^room-/, '')}(?:\\s|$)`) })).toHaveAttribute('aria-pressed', 'true');
  expect(await enginePlayerState(page)).toEqual(position);
  expect(await enginePickSamples(page, [{ x: -1, y: .5 }, { x: 1, y: .5 }, { x: .5, y: -1 }, { x: .5, y: 1 }])).toEqual([-1, -1, -1, -1]);
  await canvas.screenshot({ path: testInfo.outputPath('native-stair-picking.png') });
  await canvas.press('ArrowLeft'); await expect(surface).toHaveCount(0);
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});

test('native ceiling and water pixels identify their actual graphics records', async ({ page }) => {
  const document = openLevel('Native surface inspection', originalFiles(1), 0, -1);
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' }).getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  const points = Array.from({ length: 361 }, (_, index) => ({ x: (index % 19 + 1) / 20, y: (Math.floor(index / 19) + 1) / 20 }));
  for (const type of [2, 7]) {
    const zone = document.zones.find(zone => zone.floor - zone.ceiling >= 64
      && !document.mechanisms.some(item => item.zone === zone.id)
      && zone.lower.some(record => record.type === type && record.points.length >= 3))!;
    const record = zone.lower.find(record => record.type === type)!;
    const corners = record.points.map(id => document.points[id]!);
    const height = type === 7 ? Math.max(zone.ceiling + 1, record.height! - 16) : zone.ceiling + 1;
    const checksum = await captureEngine(page, { zone: zone.id, x: Math.round(corners.reduce((sum, p) => sum + p.x, 0) / corners.length),
      z: Math.round(corners.reduce((sum, p) => sum + p.z, 0) / corners.length), angle: 0, upper: 0, height });
    expect(checksum, `camera in zone ${zone.id} at height ${height}`).not.toBe(0);
    expect((await enginePlayerState(page)).zone).toBe(zone.id);
    const picks = await enginePickDetails(page, points), index = picks.findIndex(pick => pick.surface?.type === type);
    expect(index, `visible opcode ${type}`).toBeGreaterThanOrEqual(0);
    const pick = picks[index]!, resolved = resolveEngineSurface(document, pick.zone, pick.surface)!;
    expect(resolved).toBeDefined();
    const point = points[index]!, rect = (await canvas.boundingBox())!;
    await canvas.click({ position: { x: point.x * rect.width, y: point.y * rect.height } });
    const info = page.getByRole('note', { name: 'Picked surface' });
    await expect(info).toContainText(type === 2 ? 'Ceiling' : 'Water');
    await expect(info).toHaveAttribute('data-graphic', String(pick.surface!.offset));
    await expect(info).toContainText(`Authored height ${resolved.height}`);
  }
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});

test('native upper-room picks resolve generated cell IDs after viewport resizing', async ({ page }) => {
  const fixture = stairFixture(), placed = applyCommand(fixture.document, createStairFlight(fixture.document, fixture.flight).command);
  // Put authored rooms first so generated IDs differ from original zone IDs.
  const source = { ...placed, roomSource: { ...placed.roomSource, rooms: [
    ...placed.roomSource.rooms.filter(room => room.originZone === null),
    ...placed.roomSource.rooms.filter(room => room.originZone !== null),
  ] } };
  const built = buildPlayableLevel(source, -1), topology = built.topology!;
  const cell = topology.cells.find(cell => cell.originZone !== null && cell.id !== cell.originZone
    && built.document.zones[cell.id]!.upper.length > 0)!;
  expect(cell).toBeDefined();
  const corners = cell.vertices.map(id => topology.points[id]!);
  const camera = { zone: cell.id, x: Math.round(corners.reduce((sum, point) => sum + point.x, 0) / corners.length),
    z: Math.round(corners.reduce((sum, point) => sum + point.z, 0) / corners.length), angle: 0, upper: 1 };
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'pick-upper.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(source)) });
  await expect(page.locator('.project-picker strong')).toHaveText(source.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' }).getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await page.getByRole('button', { name: 'Upper', exact: true }).click();
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  for (const viewport of [{ width: 1440, height: 950 }, { width: 1100, height: 720 }]) {
    await page.setViewportSize(viewport); await captureEngine(page, camera);
    const before = await enginePlayerState(page);
    const points = Array.from({ length: 49 }, (_, index) => ({ x: (index % 7 + 1) / 8, y: (Math.floor(index / 7) + 1) / 8 }));
    const details = await enginePickDetails(page, points), zones = details.map(pick => pick.zone);
    const index = zones.findIndex((id, index) => {
      const cell = topology.cells.find(cell => cell.id === id); return cell?.originZone !== undefined && cell.originZone !== null && cell.originZone !== id
        && details[index]!.surface?.upper;
    });
    expect(index).toBeGreaterThanOrEqual(0);
    const native = topology.cells.find(cell => cell.id === zones[index])!.originZone!, point = points[index]!, rect = (await canvas.boundingBox())!;
    await canvas.click({ position: { x: point.x * rect.width, y: point.y * rect.height } });
    await expect(page.getByRole('button', { name: new RegExp(`^Zone ${String(native).padStart(3, '0')}(?:\\s|$)`) })).toHaveAttribute('aria-pressed', 'true');
    await expect(page.getByRole('note', { name: 'Picked surface' })).toHaveAttribute('data-storey', 'upper');
    expect(await enginePlayerState(page)).toEqual(before);
  }
  // A deliberate list selection still visits its room after native picking.
  const approach = page.getByRole('button', { name: new RegExp(`^Room ${fixture.approach.id.replace(/^room-/, '')}(?:\\s|$)`) });
  await approach.click();
  await expect.poll(async () => (await enginePlayerState(page)).x).toBe(20064);
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
