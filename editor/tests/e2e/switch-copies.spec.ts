import { expect, test } from '@playwright/test';
import { switchCopyFixture } from '../switchCopyFixture';
import { applyCommand, setSourceStart } from '../../src/model/commands';
import { pointInRoomOutline } from '../../src/model/geometry';
import { buildSourceLevel } from '../../src/model/sourceBuild';
import { roomAppearanceBuildOptions } from '../../src/model/roomAppearance';
import { encodeProject } from '../../src/model/project';
import { enginePlayerInventory, engineSwitchState } from '../engineCapture';

test('using a copied native switch changes its own condition bit and wall without toggling the original', async ({ page }) => {
  const fixture = switchCopyFixture(), { switchCopy: copy, source, dx, dz } = fixture;
  const room = fixture.document.roomSource.rooms.find(room => room.id === copy.room)!;
  const a = fixture.document.points[source.point]!, b = fixture.document.points[source.point + 1]!;
  const centre = { x: Math.floor((a.x + b.x) / 2) + dx, z: Math.floor((a.z + b.z) / 2) + dz };
  const positions = Array.from({ length: 13 }, (_, x) => Array.from({ length: 13 }, (_, z) => ({ x: centre.x + (x - 6) * 8, z: centre.z + (z - 6) * 8 }))).flat()
    .filter(point => Math.hypot(point.x - centre.x, point.z - centre.z) < 55 && pointInRoomOutline(room.vertices, point));
  const clearance = (point: typeof centre) => Math.min(...room.vertices.map((a, index) => {
    const b = room.vertices[(index + 1) % room.vertices.length]!, x = b.x - a.x, z = b.z - a.z;
    const t = Math.max(0, Math.min(1, ((point.x - a.x) * x + (point.z - a.z) * z) / (x * x + z * z)));
    return Math.hypot(point.x - a.x - t * x, point.z - a.z - t * z);
  }));
  const position = positions.sort((a, b) => clearance(b) - clearance(a))[0];
  expect(position).toBeDefined();
  const document = applyCommand(fixture.document, setSourceStart(fixture.document, 1, copy.room, position!.x, position!.z));
  const built = buildSourceLevel(document, { resolvedExit: -1, ...roomAppearanceBuildOptions(document), switchCopies: [copy] });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'switch-copy.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(built.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(built.document.name);
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const initial = await engineSwitchState(page, source.id);
  await expect.poll(async () => (await engineSwitchState(page, source.id)).phase).toBe(initial.phase & 0x7c);
  const original = await engineSwitchState(page, source.id), copied = await engineSwitchState(page, copy.slot);
  expect(original.active).toBe(false); expect(copied.active).toBe(false);
  expect((await enginePlayerInventory(page)).conditions).toBe(0);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus();
  await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(1 << (copy.slot + 4)); }
  finally { await page.keyboard.up('Space'); }
  expect(await engineSwitchState(page, copy.slot)).toMatchObject({ active: true, phase: (copied.phase & 0x7c) | 2 });
  expect(await engineSwitchState(page, source.id)).toEqual(original);
});
