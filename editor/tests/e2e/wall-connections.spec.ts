import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { closedPortalFixture } from '../portalFixture';
import { roomPortalTargets } from '../../src/model/roomPortals';
import { roomAppearanceChoices } from '../../src/model/roomAppearance';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { captureEngine, enginePlayerState } from '../engineCapture';

test('closes an existing shared wall, blocks native movement, saves and reopens it as one edit', async ({ page }, info) => {
  test.setTimeout(90000);
  const fixture = closedPortalFixture(), document = fixture.baseline, owner = fixture.owner, other = fixture.removed;
  const target = roomPortalTargets(document, owner.id).find(target => target.neighbour === other.id)!, label = target.edges.map(edge => edge + 1).join(', '), name = `Room ${other.id.replace(/^room-/, '')}`;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'shared-wall.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(owner.originZone).padStart(3, '0')}(?:\\s|$)`) }).click();
  await page.getByText('Wall connections (solid / dotted)', { exact: true }).click();
  await page.getByRole('button', { name: `Close edge ${label} to ${name}`, exact: true }).click();
  await page.getByRole('button', { name: 'Cancel wall closure', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: `Close edge ${label} to ${name}`, exact: true }).click();
  for (const id of [owner.id, other.id]) {
    const room = document.roomSource.rooms.find(room => room.id === id)!, choices = roomAppearanceChoices(document, room), title = `Room ${id.replace(/^room-/, '')}`;
    await page.getByRole('combobox', { name: `Wall style for ${title}`, exact: true }).selectOption(String(choices.lower[0]!.offset));
    await page.getByRole('combobox', { name: `Corner lighting for ${title}`, exact: true }).selectOption(String(choices.points[0]!));
  }
  await page.getByRole('button', { name: 'Apply wall closure', exact: true }).click();
  await expect(page.getByRole('alert')).toHaveCount(0);
  await expect(page.getByRole('button', { name: `Close edge ${label} to ${name}`, exact: true })).toBeDisabled();
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('closed-connection.ab3d'); await (await pending).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(roomPortalTargets(saved, owner.id).find(target => target.neighbour === other.id)!.open).toBe(false);
  const built = buildPlayableLevel(saved, -1), own = built.topology!.cells.find(cell => cell.sourceId === owner.id)!, destination = built.topology!.cells.find(cell => cell.sourceId === other.id)!;
  expect(own.neighbours).not.toContain(destination.id);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  const angle = fixture.normal.x > 0 ? 2048 : fixture.normal.x < 0 ? 6144 : fixture.normal.z > 0 ? 0 : 4096;
  const camera = { ...fixture.start, zone: own.id, upper: 0, angle }, closedFrame = await captureEngine(page, camera);
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'); await canvas.focus();
  const key = fixture.normal.x > 0 ? 'd' : fixture.normal.x < 0 ? 'a' : fixture.normal.z > 0 ? 'w' : 's';
  const before = await enginePlayerState(page); await page.keyboard.down(key);
  try {
    await expect.poll(async () => { const p = await enginePlayerState(page); return (p.x - before.x) * fixture.normal.x + (p.z - before.z) * fixture.normal.z; }).toBeGreaterThan(8);
    for (let i = 0; i < 20; i++) {
      await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      const p = await enginePlayerState(page); expect(p.zone).toBe(own.id);
      expect((p.x - fixture.x) * fixture.normal.x + (p.z - fixture.z) * fixture.normal.z).toBeLessThan(0);
    }
  } finally { await page.keyboard.up(key); }
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  // Re-import the closed game artifact: opening must cut actual native solid
  // graphics/collision, rather than simply restoring an earlier source state.
  await page.locator('input[type=file]').setInputFiles({ name: 'native-solid-wall.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(built.document)) });
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(owner.originZone).padStart(3, '0')}(?:\\s|$)`) }).click();
  await page.getByText('Wall connections (solid / dotted)', { exact: true }).click();
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  const revision = (await preview.getAttribute('data-revision'))!;
  await page.getByRole('button', { name: `Open edge ${label} to ${name}`, exact: true }).click();
  await expect(preview).not.toHaveAttribute('data-revision', revision); await expect(preview.getByRole('status')).toContainText('zones loaded');
  expect(await captureEngine(page, camera)).not.toBe(closedFrame);
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await canvas.focus(); await page.keyboard.down(key);
  try { await expect.poll(async () => (await enginePlayerState(page)).zone, { intervals: [10] }).toBe(destination.id); } finally { await page.keyboard.up(key); }
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('button', { name: `Close edge ${label} to ${name}`, exact: true })).toBeDisabled();
  await page.locator('input[type=file]').setInputFiles({ name: 'reopened-closed.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(saved)) });
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(owner.originZone).padStart(3, '0')}(?:\\s|$)`) }).click();
  await page.getByText('Wall connections (solid / dotted)', { exact: true }).click();
  await expect(page.getByRole('button', { name: `Close edge ${label} to ${name}`, exact: true })).toBeDisabled();
});
