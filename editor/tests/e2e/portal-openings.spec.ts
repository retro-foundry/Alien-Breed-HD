import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { encodeProject, openArchive, openProjectCandidate } from '../../src/model/project';
import { zoneContainsPoint } from '../../src/model/level';
import { importedPortalFixture } from '../portalFixture';
import { captureEngine, enginePlayerState } from '../engineCapture';

test('an authored doorway cuts a shipped solid wall and the player walks through it in the actual game', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const fixture = importedPortalFixture(), preview = page.getByRole('region', { name: '3D game engine preview' });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'closed-reference.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.baseline)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.baseline.name);
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const oldZone = Number(await preview.getAttribute('data-spawn-zone')), closed: number[] = [];
  for (const angle of [0, 2048, 4096, 6144]) closed.push(await captureEngine(page, { ...fixture.start, zone: oldZone, upper: 0, angle }));
  await page.locator('input[type=file]').setInputFiles({ name: 'open-doorway.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const newZone = Number(await preview.getAttribute('data-spawn-zone')), opened: number[] = [];
  for (const angle of [0, 2048, 4096, 6144]) {
    opened.push(await captureEngine(page, { ...fixture.start, zone: newZone, upper: 0, angle }));
    await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas')
      .screenshot({ path: testInfo.outputPath(`opened-wall-angle-${angle}.png`) });
  }
  expect(opened.some((hash, index) => hash !== closed[index])).toBe(true);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const path = testInfo.outputPath('new-doorway-game.zip'); await (await download).saveAs(path);
  const compiled = openProjectCandidate(openArchive(new Uint8Array(await readFile(path)), 'doorway.zip')[0]!);
  const inside = { x: fixture.x + fixture.normal.x * 128, z: fixture.z + fixture.normal.z * 128 };
  const destination = compiled.zones.find(zone => zoneContainsPoint(compiled, zone, inside.x, inside.z))!;
  expect(destination).toBeDefined(); expect(destination.id).not.toBe(newZone);
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'); await canvas.focus();
  const key = fixture.normal.x > 0 ? 'd' : fixture.normal.x < 0 ? 'a' : fixture.normal.z > 0 ? 'w' : 's';
  await page.keyboard.down(key);
  try { await expect.poll(async () => (await enginePlayerState(page)).zone).toBe(destination.id); }
  finally { await page.keyboard.up(key); }
  const state = await enginePlayerState(page);
  expect((state.x - fixture.x) * fixture.normal.x + (state.z - fixture.z) * fixture.normal.z).toBeGreaterThan(0);
  await testInfo.attach('doorway-gameplay', { body: JSON.stringify({ source: fixture.room.id, added: fixture.added.id,
    start: fixture.start, boundary: { x: fixture.x, z: fixture.z }, normal: fixture.normal, closed, opened, arrived: state }), contentType: 'application/json' });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('alert')).toHaveCount(0);
});
