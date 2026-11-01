import { expect, test } from '@playwright/test';
import { openLevel, zoneContainsPoint } from '../../src/model/level';
import { originalFiles } from '../helpers';
import { roomHeightFixture } from '../roomHeightFixture';
import { encodeProject } from '../../src/model/project';
import { objectIsActive } from '../../src/model/sourceObjects';

test('invisible start-marker margins do not capture clicks in neighbouring zones', async ({ page }) => {
  const document = openLevel('Level 1', originalFiles(1));
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Show objects', exact: true }).click();
  const canvas = page.locator('canvas[aria-label="Level map"]'), bounds = (await canvas.boundingBox())!;
  const scale = Number(await canvas.getAttribute('data-world-scale')), x = Number(await canvas.getAttribute('data-world-x')), z = Number(await canvas.getAttribute('data-world-z'));
  const point = document.starts.flatMap(start => [-11, -9, 9, 11].flatMap(dx => [-8, -6].flatMap(dy => {
    if (Math.hypot(dx, dy) >= 13) return [];
    const p = { x: start.x + dx / scale, z: start.z - dy / scale }, owners = document.zones.filter(zone => zoneContainsPoint(document, zone, p.x, p.z));
    return owners.length === 1 ? [{ x: bounds.width / 2 + (p.x - x) * scale, y: bounds.height / 2 - (p.z - z) * scale, zone: owners[0]!.id }] : [];
  })))[0]!;
  expect(point).toBeDefined();
  await canvas.click({ position: point });
  await expect(page.getByRole('heading', { name: `Zone ${point.zone}`, exact: true })).toBeVisible();
});

test('invisible marker margins do not select objects instead of the zone under the pointer', async ({ page }) => {
  const document = openLevel('Level 1', originalFiles(1));
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  const canvas = page.locator('canvas[aria-label="Level map"]'), bounds = (await canvas.boundingBox())!;
  const scale = Number(await canvas.getAttribute('data-world-scale')), x = Number(await canvas.getAttribute('data-world-x')), z = Number(await canvas.getAttribute('data-world-z'));
  const objects = document.objects.filter(object => objectIsActive(object) && !object.upper);
  const point = objects.flatMap(object => [1, -1].flatMap(direction => {
    const p = { x: object.x + direction * 7 / scale, z: object.z };
    if (objects.some(other => Math.hypot(other.x - p.x, other.z - p.z) * scale < 5)
      || document.starts.some(start => Math.hypot(start.x - p.x, start.z - p.z) * scale < 15)) return [];
    const owners = document.zones.filter(zone => zoneContainsPoint(document, zone, p.x, p.z));
    const screen = { x: bounds.width / 2 + (p.x - x) * scale, y: bounds.height / 2 - (p.z - z) * scale };
    return owners.length === 1 && screen.x > 20 && screen.y > 20 && screen.x < bounds.width - 20 && screen.y < bounds.height - 20
      ? [{ ...screen, zone: owners[0]!.id }] : [];
  }))[0]!;
  expect(point).toBeDefined();
  await canvas.click({ position: point });
  await expect(page.getByRole('heading', { name: `Zone ${point.zone}`, exact: true })).toBeVisible();
});

test('a selected corner cannot capture a click in the neighbouring zone outside its visible handle', async ({ page }) => {
  const document = openLevel('Level 1', originalFiles(1));
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Show objects', exact: true }).click();
  const canvas = page.locator('canvas[aria-label="Level map"]');
  const fitX = (Math.min(...document.points.map(p => p.x)) + Math.max(...document.points.map(p => p.x))) / 2;
  const fitZ = (Math.min(...document.points.map(p => p.z)) + Math.max(...document.points.map(p => p.z))) / 2;
  await expect(canvas).toHaveAttribute('data-world-x', String(fitX));
  await expect(canvas).toHaveAttribute('data-world-z', String(fitZ));
  const scale = Number(await canvas.getAttribute('data-world-scale'));
  const candidates = document.zones.flatMap(zone => {
    const lines = zone.boundaryIds.map(id => document.lines[id]!), xs = lines.map(line => line.x), zs = lines.map(line => line.z);
    return [.23, .43, .67, .81].flatMap(u => [.19, .39, .61, .79].flatMap(v => {
      const p = { x: Math.min(...xs) + u * (Math.max(...xs) - Math.min(...xs)), z: Math.min(...zs) + v * (Math.max(...zs) - Math.min(...zs)) };
      const owners = document.zones.filter(zone => zoneContainsPoint(document, zone, p.x, p.z));
      return owners.length === 1 && owners[0]!.id === zone.id && document.starts.every(start => Math.hypot(p.x - start.x, p.z - start.z) * scale > 15) ? [{ ...p, zone: zone.id }] : [];
    }));
  });
  const example = document.zones.flatMap(zone => {
    const distances = (p: { x: number; z: number }) => zone.boundaryIds.map(id => Math.hypot(p.x - document.lines[id]!.x, p.z - document.lines[id]!.z) * scale);
    const point = candidates.find(p => p.zone !== zone.id && Math.min(...distances(p)) > 5 && Math.min(...distances(p)) < 8);
    return point ? [{ selected: zone.id, point }] : [];
  })[0]!;
  expect(example).toBeDefined();
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(example.selected).padStart(3, '0')}\\b`) }).click();
  await page.getByRole('button', { name: 'Frame map', exact: true }).click();
  await expect(canvas).toHaveAttribute('data-world-x', String(fitX));
  await expect(canvas).toHaveAttribute('data-world-z', String(fitZ));
  const x = (await canvas.getAttribute('data-world-x'))!, z = (await canvas.getAttribute('data-world-z'))!;
  const bounds = (await canvas.boundingBox())!, currentScale = Number(await canvas.getAttribute('data-world-scale'));
  await canvas.click({ position: { x: bounds.width / 2 + (example.point.x - Number(x)) * currentScale,
    y: bounds.height / 2 - (example.point.z - Number(z)) * currentScale } });
  await expect(page.getByRole('heading', { name: `Zone ${example.point.zone}`, exact: true })).toBeVisible();
  await expect(canvas).toHaveAttribute('data-world-x', x); await expect(canvas).toHaveAttribute('data-world-z', z);
});

test('clicking an authored room selects under the pointer without recentering the map', async ({ page }) => {
  const { document, from, to } = roomHeightFixture();
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'rooms.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${from.id.replace('room-', '')}\\b`) }).click();
  await page.getByRole('button', { name: 'Frame map', exact: true }).click();
  const canvas = page.locator('canvas[aria-label="Level map"]'), bounds = (await canvas.boundingBox())!;
  const point = { x: to.vertices.reduce((sum, p) => sum + p.x, 0) / to.vertices.length, z: to.vertices.reduce((sum, p) => sum + p.z, 0) / to.vertices.length };
  const beforeZoom = await canvas.evaluate(node => ({ x: Number(node.dataset.worldX), z: Number(node.dataset.worldZ), scale: Number(node.dataset.worldScale) }));
  await page.mouse.move(bounds.x + bounds.width / 2 + (point.x - beforeZoom.x) * beforeZoom.scale,
    bounds.y + bounds.height / 2 - (point.z - beforeZoom.z) * beforeZoom.scale);
  await page.mouse.wheel(0, -1800);
  await expect(canvas).not.toHaveAttribute('data-world-scale', String(beforeZoom.scale));
  const x = (await canvas.getAttribute('data-world-x'))!, z = (await canvas.getAttribute('data-world-z'))!, scale = (await canvas.getAttribute('data-world-scale'))!;
  await canvas.click({ position: { x: bounds.width / 2 + (point.x - Number(x)) * Number(scale), y: bounds.height / 2 - (point.z - Number(z)) * Number(scale) } });
  await expect(page.getByRole('heading', { name: `Room ${to.id.replace('room-', '')}`, exact: true })).toBeVisible();
  await expect(canvas).toHaveAttribute('data-world-x', x); await expect(canvas).toHaveAttribute('data-world-z', z);
  await expect(canvas).toHaveAttribute('data-world-scale', scale);
});

test('secondary fences do not turn a Zone 1 click into Zone 0', async ({ page }) => {
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Show objects', exact: true }).click();
  await page.getByRole('button', { name: /^Zone 001/ }).click();
  const canvas = page.locator('canvas[aria-label="Level map"]'), bounds = (await canvas.boundingBox())!;
  const old = (await canvas.getAttribute('data-world-scale'))!;
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2); await page.mouse.wheel(0, -1200);
  await expect(canvas).not.toHaveAttribute('data-world-scale', old);
  const view = await canvas.evaluate(node => ({ x: Number(node.dataset.worldX), z: Number(node.dataset.worldZ), scale: Number(node.dataset.worldScale) }));
  await canvas.click({ position: { x: bounds.width / 2 + (-482.56 - view.x) * view.scale, y: bounds.height / 2 - (278.08 - view.z) * view.scale } });
  await expect(page.getByRole('heading', { name: 'Zone 1', exact: true })).toBeVisible();
});
