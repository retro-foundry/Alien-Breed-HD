import {expect, test} from '@playwright/test';
import {openLevel, zoneContainsPoint} from '../../src/model/level';
import {originalFiles} from '../helpers';
test('invisible start-marker margins do not capture clicks in neighbouring zones', async ({ page }) => {
  const document = openLevel('Level 1', originalFiles(1));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
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
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  const canvas = page.locator('canvas[aria-label="Level map"]'), bounds = (await canvas.boundingBox())!;
  const scale = Number(await canvas.getAttribute('data-world-scale')), x = Number(await canvas.getAttribute('data-world-x')), z = Number(await canvas.getAttribute('data-world-z'));
  const objects = document.objects.filter(object => object.zone >= 0 && !object.upper);
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


test('secondary fences do not turn a Zone 1 click into Zone 0', async ({ page }) => {
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
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
