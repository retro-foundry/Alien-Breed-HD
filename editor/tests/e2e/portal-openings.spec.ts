import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { decodeProject, encodeProject, openArchive, openProjectCandidate } from '../../src/model/project';
import { zoneContainsPoint } from '../../src/model/level';
import { closedPortalFixture, importedPortalFixture, replacementPortalFixture } from '../portalFixture';
import { captureEngine, enginePlayerState } from '../engineCapture';

test('an authored doorway cuts a shipped solid wall and the player walks through it in the actual game', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const fixture = importedPortalFixture(), preview = page.getByRole('region', { name: '3D game engine preview' });
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'closed-reference.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.baseline)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.baseline.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
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

test('a replacement room reconnects an imported doorway and is reachable in the actual game', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const fixture = replacementPortalFixture(), preview = page.getByRole('region', { name: '3D game engine preview' });
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'replacement-room.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(preview.getByRole('alert')).toContainText('changed doorway needing a wall style', { timeout: 60000 });
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(fixture.owner.originZone).padStart(3, '0')}(?:\\s|$)`) }).click();
  await page.getByRole('combobox', { name: 'Lower wall style' }).selectOption(String(fixture.appearance.lowerWall));
  await page.getByRole('combobox', { name: 'Room corner lighting' }).selectOption(String(fixture.appearance.lightingPoint));
  await page.getByRole('button', { name: 'Apply appearance', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const path = testInfo.outputPath('replacement-game.zip'); await (await download).saveAs(path);
  const compiled = openProjectCandidate(openArchive(new Uint8Array(await readFile(path)), 'replacement-game.zip')[0]!);
  const inside = { x: fixture.x + fixture.normal.x * 128, z: fixture.z + fixture.normal.z * 128 };
  const destination = compiled.zones.find(zone => zoneContainsPoint(compiled, zone, inside.x, inside.z))!;
  expect(destination).toBeDefined(); expect(destination.id).not.toBe(compiled.starts[0]!.zone);
  // renderer.c's player arrow uses (sin_lookup(angle), cos_lookup(angle)).
  const angle = fixture.normal.x > 0 ? 2048 : fixture.normal.x < 0 ? 6144 : fixture.normal.z > 0 ? 0 : 4096;
  await captureEngine(page, { ...fixture.start, zone: compiled.starts[0]!.zone, upper: 0, angle });
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot({ path: testInfo.outputPath('replacement-doorway.png') });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus();
  const key = fixture.normal.x > 0 ? 'd' : fixture.normal.x < 0 ? 'a' : fixture.normal.z > 0 ? 'w' : 's';
  await page.keyboard.down(key);
  try { await expect.poll(async () => (await enginePlayerState(page)).zone).toBe(destination.id); }
  finally { await page.keyboard.up(key); }
  const state = await enginePlayerState(page);
  expect((state.x - fixture.x) * fixture.normal.x + (state.z - fixture.z) * fixture.normal.z).toBeGreaterThan(0);
  await testInfo.attach('replacement-gameplay', { body: JSON.stringify({ old: fixture.removed.id, replacement: fixture.replacement.id, arrived: state }), contentType: 'application/json' });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('alert')).toHaveCount(0);
});

test('deleting an adjoining room closes its doorway with an undoable saved appearance and solid native collision', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const fixture = closedPortalFixture(), preview = page.getByRole('region', { name: '3D game engine preview' });
  const selectZone = (id: number) => page.getByRole('button', { name: new RegExp(`^Zone ${String(id).padStart(3, '0')}(?:\\s|$)`) }).click();
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'open-reference.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.baseline)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.baseline.name);
  await selectZone(fixture.removed.originZone!);
  await page.getByRole('button', { name: 'Delete room', exact: true }).click();
  await expect(page.getByRole('button', { name: new RegExp(`^Zone ${String(fixture.removed.originZone).padStart(3, '0')}(?:\\s|$)`) })).toHaveCount(0);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(preview.getByRole('alert')).toContainText('changed doorway needing a wall style', { timeout: 60000 });
  await selectZone(fixture.owner.originZone!);
  await expect(page.getByRole('heading', { name: 'Wall repair appearance' })).toBeVisible();
  await page.getByRole('combobox', { name: 'Lower wall style' }).selectOption(String(fixture.appearance.lowerWall));
  await page.getByRole('combobox', { name: 'Room corner lighting' }).selectOption(String(fixture.appearance.lightingPoint));
  await page.getByRole('button', { name: 'Apply appearance', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await expect(preview.getByRole('alert')).toHaveCount(0);
  const angle = fixture.normal.x > 0 ? 2048 : fixture.normal.x < 0 ? 6144 : fixture.normal.z > 0 ? 0 : 4096;
  await captureEngine(page, { ...fixture.start, zone: Number(await preview.getAttribute('data-spawn-zone')), upper: 0, angle });
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot({ path: testInfo.outputPath('closed-doorway.png') });
  const savedDownload = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const savedPath = testInfo.outputPath('closed-doorway-source.ab3d'); await (await savedDownload).saveAs(savedPath);
  const saved = decodeProject(new Uint8Array(await readFile(savedPath)));
  expect(saved.roomAppearances).toEqual([fixture.appearance]); expect(saved.roomSource).toEqual(fixture.document.roomSource);
  expect(saved.files).toEqual(fixture.baseline.files);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(preview.getByRole('alert')).toContainText('changed doorway needing a wall style');
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0');
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus();
  const key = fixture.normal.x > 0 ? 'd' : fixture.normal.x < 0 ? 'a' : fixture.normal.z > 0 ? 'w' : 's';
  const before = await enginePlayerState(page);
  await page.keyboard.down(key);
  try {
    // Wait for real simulation movement toward the wall, then observe further
    // frames under held input. Crossing would enter the removed room's space.
    await expect.poll(async () => { const p = await enginePlayerState(page); return (p.x - before.x) * fixture.normal.x + (p.z - before.z) * fixture.normal.z; }).toBeGreaterThan(8);
    const samples = [];
    for (let i = 0; i < 20; i++) {
      await page.evaluate(() => new Promise<void>(resolve => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
      const p = await enginePlayerState(page); samples.push(p);
      expect((p.x - fixture.x) * fixture.normal.x + (p.z - fixture.z) * fixture.normal.z).toBeLessThan(0);
      expect(p.zone).toBe(before.zone);
    }
    const final = samples.at(-1)!;
    expect(samples.slice(-6)).toEqual(Array(6).fill(final));
    const distanceBefore = (fixture.x - before.x) * fixture.normal.x + (fixture.z - before.z) * fixture.normal.z;
    expect((final.x - before.x) * fixture.normal.x + (final.z - before.z) * fixture.normal.z).toBeGreaterThan(distanceBefore / 2);
    await testInfo.attach('closed-doorway-movement', { body: JSON.stringify(samples), contentType: 'application/json' });
  } finally { await page.keyboard.up(key); }
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
});
