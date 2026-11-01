import { expect, test } from '@playwright/test';
import { roomHeightFixture } from '../roomHeightFixture';
import { encodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { visibilityInspection } from '../../src/model/visibilityInspection';
import { originalFiles } from '../helpers';
import { openLevel } from '../../src/model/level';

test('visibility inspection shows current compiled cells and exact native table entries without editing source', async ({ page }, testInfo) => {
  const fixture = roomHeightFixture();
  fixture.document = { ...fixture.document, roomSource: { ...fixture.document.roomSource, rooms: fixture.document.roomSource.rooms.map(room => room.id === fixture.from.id ? {
    ...room, wallSources: [null, null, null, null, null, null], vertices: [{ x: 20000, z: 20000 }, { x: 20256, z: 20000 }, { x: 20256, z: 20128 }, { x: 20128, z: 20128 }, { x: 20128, z: 20256 }, { x: 20000, z: 20256 }],
  } : room) } };
  const built = buildPlayableLevel(fixture.document, -1);
  const expected = visibilityInspection(built, fixture.from.id, undefined);
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'visibility.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  const panel = page.getByRole('region', { name: 'Visibility inspection' });
  const toggle = panel.getByRole('button', { name: 'Visibility overlay', exact: true });
  await toggle.focus(); await page.keyboard.press('Enter');
  const cell = panel.getByRole('combobox', { name: 'Visibility runtime cell' });
  await expect(cell).toHaveValue(String(expected.selected));
  await expect(panel.getByRole('status')).toHaveText(`${expected.rows.length} visibility entries · ${expected.portals.length} portals`);
  await panel.getByText('Runtime visibility table', { exact: true }).click();
  await expect(panel.locator('tbody tr')).toHaveCount(expected.rows.length);
  for (let index = 0; index < expected.rows.length; index++) {
    const row = expected.rows[index]!;
    await expect(panel.locator('tbody tr').nth(index).locator('td')).toHaveText([String(row.zone), String(row.clip), String(row.bounds?.wordOffset ?? '—'), `0x${(row.mask >>> 0).toString(16)}`]);
  }
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
  expect(expected.cells.length).toBeGreaterThan(1);
  await cell.selectOption(String(expected.cells.at(-1)!));
  const other = visibilityInspection(built, fixture.from.id, undefined, expected.cells.at(-1));
  await expect(panel.getByRole('status')).toHaveText(`${other.rows.length} visibility entries · ${other.portals.length} portals`);
  await expect(panel.locator('tbody tr td:first-child')).toHaveText(other.rows.map(row => String(row.zone)));
  await expect(panel.getByRole('combobox', { name: 'Visibility clip target' })).toHaveValue(String(other.target));
  const map = page.locator('canvas[aria-label="Level map"]'), beforeScale = Number(await map.getAttribute('data-world-scale'));
  await panel.getByRole('button', { name: 'Frame visibility', exact: true }).click();
  await expect.poll(async () => Number(await map.getAttribute('data-world-scale'))).toBeGreaterThan(beforeScale);
  const colours = await map.evaluate((node, polygons) => {
    const canvas = node as HTMLCanvasElement, context = canvas.getContext('2d')!;
    const scale = Number(canvas.dataset.worldScale), x = Number(canvas.dataset.worldX), z = Number(canvas.dataset.worldZ);
    return polygons.map(polygon => {
      const centre = polygon.vertices.reduce((sum, point) => ({ x: sum.x + point.x / polygon.vertices.length, z: sum.z + point.z / polygon.vertices.length }), { x: 0, z: 0 });
      const px = (canvas.clientWidth / 2 + (centre.x - x) * scale) * canvas.width / canvas.clientWidth;
      const py = (canvas.clientHeight / 2 - (centre.z - z) * scale) * canvas.height / canvas.clientHeight;
      return { origin: polygon.origin, rgb: Array.from(context.getImageData(Math.floor(px), Math.floor(py), 1, 1).data) };
    });
  }, other.polygons);
  const amber = colours.find(colour => colour.origin)!.rgb, cyan = colours.find(colour => !colour.origin)!.rgb;
  expect(amber[0]!).toBeGreaterThan(amber[1]!);
  expect(cyan[1]! - cyan[0]!).toBeGreaterThan(20);
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
  await page.mouse.move(0, 0);
  await page.screenshot({ path: testInfo.outputPath('visibility-overlay.png'), fullPage: true });
  await toggle.click(); await expect(cell).toHaveCount(0);
});

test('failed source builds remove old visibility and undo restores a valid inspection', async ({ page }) => {
  const document = openLevel('Visibility failure recovery', originalFiles(1));
  const moving = document.mechanisms.find(item => item.kind === 'door')!, room = document.roomSource.rooms.find(room => room.originZone === moving.zone)!;
  const zone = document.zones[room.recordZone]!;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'door.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(zone.id).padStart(3, '0')}(?:\\s|$)`) }).click();
  const panel = page.getByRole('region', { name: 'Visibility inspection' });
  await panel.getByRole('button', { name: 'Visibility overlay', exact: true }).click();
  await expect(panel.getByRole('combobox', { name: 'Visibility runtime cell' })).toBeVisible();
  await page.getByRole('spinbutton', { name: 'Lower floor height', exact: true }).fill(String(zone.floor + 32));
  await page.getByRole('spinbutton', { name: 'Lower ceiling height', exact: true }).fill(String(zone.ceiling - 32));
  await page.getByRole('button', { name: 'Apply heights', exact: true }).click();
  await expect(panel.getByRole('alert')).toContainText('moving door or lift height');
  await expect(panel.getByRole('combobox', { name: 'Visibility runtime cell' })).toHaveCount(0);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(panel.getByRole('combobox', { name: 'Visibility runtime cell' })).toBeVisible();
  await expect(panel.getByRole('alert')).toHaveCount(0);
});
