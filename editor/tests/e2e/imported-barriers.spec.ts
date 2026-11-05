import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { importedBarrierFixture } from '../importedBarrierFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { roomPortalTargets } from '../../src/model/roomPortals';
import { captureEngine, enginePlayerState, enginePickDetails } from '../engineCapture';

test('recognises a raw imported barrier, preserves it through a height edit, and explicitly opens it for native walking', async ({ page }, info) => {
  test.setTimeout(120000);
  const fixture = importedBarrierFixture(), { document, own, other } = fixture;
  const target = roomPortalTargets(document, own.id).find(target => target.neighbour === other.id)!, label = target.edges.map(edge => edge + 1).join(', '), name = `Room ${other.id.replace(/^room-/, '')}`;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'raw-native-barrier.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(own.originZone).padStart(3, '0')}(?:\\s|$)`) }).click();
  await page.getByText('Wall connections (solid / dotted)', { exact: true }).click();
  await expect(page.getByRole('button', { name: `See-through barrier on edge ${label} to ${name}`, exact: true })).toBeDisabled();
  await expect(page.getByRole('button', { name: `Open edge ${label} to ${name}`, exact: true })).toBeEnabled();
  await expect(page.getByRole('note', { name: 'Map line legend' })).toContainText('see-through barrier');
  const headroom = page.getByRole('spinbutton', { name: 'Room headroom', exact: true });
  await headroom.fill(String(document.zones[own.recordZone]!.floor - document.zones[own.recordZone]!.ceiling + 1)); await headroom.press('Tab');
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('edited-native-barrier.ab3d'); await (await pending).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.roomSource.rooms.find(room => room.id === own.id)!.heights).toBeDefined(); expect(saved.roomAppearances).toBeUndefined();
  const built = buildPlayableLevel(saved, -1), ownCell = built.topology!.cells.find(cell => cell.sourceId === own.id)!, destination = built.topology!.cells.find(cell => cell.sourceId === other.id)!;
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' }); await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  const angle = fixture.normal.x > 0 ? 2048 : fixture.normal.x < 0 ? 6144 : fixture.normal.z > 0 ? 0 : 4096;
  await captureEngine(page, { ...fixture.start, zone: ownCell.id, upper: 0, angle });
  const samples = Array.from({ length: 961 }, (_, index) => ({ x: (index % 31 + 1) / 32, y: (Math.floor(index / 31) + 1) / 32 }));
  const picks = await enginePickDetails(page, samples);
  expect(picks.some(pick => pick.surface?.type === 13)).toBe(true);
  expect(picks.some(pick => pick.zone === destination.id && pick.surface?.type !== 13)).toBe(true);
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  const key = fixture.normal.x > 0 ? 'd' : fixture.normal.x < 0 ? 'a' : fixture.normal.z > 0 ? 'w' : 's';
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await canvas.focus(); const before = await enginePlayerState(page); await page.keyboard.down(key);
  try {
    await expect.poll(async () => { const p = await enginePlayerState(page); return (p.x - before.x) * fixture.normal.x + (p.z - before.z) * fixture.normal.z; }).toBeGreaterThan(8);
    for (let i = 0; i < 20; i++) {
      await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      expect((await enginePlayerState(page)).zone).toBe(ownCell.id);
    }
  } finally { await page.keyboard.up(key); }
  await preview.getByRole('button', { name: 'Stop', exact: true }).click(); await expect(preview.getByRole('status')).toContainText('zones loaded');
  const revision = await preview.getAttribute('data-revision');
  await page.getByRole('button', { name: `Open edge ${label} to ${name}`, exact: true }).click();
  await expect(preview).not.toHaveAttribute('data-revision', revision!); await expect(preview.getByRole('status')).toContainText('zones loaded');
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await canvas.focus(); await page.keyboard.down(key);
  try { await expect.poll(async () => (await enginePlayerState(page)).zone, { intervals: [10] }).toBe(destination.id); }
  finally { await page.keyboard.up(key); }
  await preview.getByRole('button', { name: 'Stop', exact: true }).click(); await expect(preview.getByRole('status')).toContainText('zones loaded');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('button', { name: `See-through barrier on edge ${label} to ${name}`, exact: true })).toBeDisabled();
  await page.getByRole('button', { name: `Close edge ${label} to ${name}`, exact: true }).click();
  for (const room of [own, other]) {
    const face = document.zones[room.recordZone]!.lower.find(face => face.type === 13)!, title = `Room ${room.id.replace(/^room-/, '')}`;
    await page.getByRole('combobox', { name: `Wall style for ${title}`, exact: true }).selectOption(String(face.offset));
    await page.getByRole('combobox', { name: `Corner lighting for ${title}`, exact: true }).selectOption(String(face.points[0]!));
  }
  const closeRevision = await preview.getAttribute('data-revision');
  await page.getByRole('button', { name: 'Apply wall closure', exact: true }).click();
  await expect(preview).not.toHaveAttribute('data-revision', closeRevision!); await expect(preview.getByRole('status')).toContainText('zones loaded');
  await captureEngine(page, { ...fixture.start, zone: ownCell.id, upper: 0, angle });
  const closed = await enginePickDetails(page, samples);
  expect(closed.some(pick => pick.surface?.type === 13 || pick.zone === destination.id)).toBe(false);
  expect(closed.some(pick => pick.surface?.type === 0)).toBe(true);
  await expect(preview.getByRole('alert')).toHaveCount(0);
});
