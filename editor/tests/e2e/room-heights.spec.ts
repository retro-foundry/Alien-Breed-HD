import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture, twoStoreyHeightTemplate } from '../roomHeightFixture';
import { decodeProject, encodeProject, openArchive, openProjectCandidate } from '../../src/model/project';
import { captureEngine, enginePlayerFeetFixed, enginePlayerState } from '../engineCapture';
import { importedPortalFixture } from '../portalFixture';
import { zoneContainsPoint } from '../../src/model/level';
import { buildPlayableLevel } from '../../src/model/playableBuild';

test('room heights apply atomically, copy, undo and save independently of their template and drive native player floor height', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const authored = roomHeightFixture(), built = buildPlayableLevel(authored.document, -1);
  const fixture = { ...authored, document: built.document,
    from: built.document.roomSource.rooms.find(room => room.originZone === built.topology!.cells.find(cell => cell.sourceId === authored.from.id)!.id)!,
    to: built.document.roomSource.rooms.find(room => room.originZone === built.topology!.cells.find(cell => cell.sourceId === authored.to.id)!.id)! };
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'room-heights.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(fixture.from.originZone).padStart(3, '0')}(?:\\s|$)`) }).click();
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const before = await captureEngine(page, { zone: Number(await preview.getAttribute('data-spawn-zone')), x: 20128, z: 20128, upper: 0, angle: 0 });
  const floor = page.getByRole('spinbutton', { name: 'Lower floor height', exact: true }), ceiling = page.getByRole('spinbutton', { name: 'Lower ceiling height', exact: true });
  await floor.fill('32.25'); await ceiling.fill('-160.5'); await page.getByRole('button', { name: 'Apply heights', exact: true }).click();
  await expect(floor).toHaveValue('32.25');
  await expect(page.locator('.info-row').filter({ hasText: 'Floor height' })).toContainText('32.25');
  const gameDownload = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const gamePath = testInfo.outputPath('height-game.zip'); await (await gameDownload).saveAs(gamePath);
  const game = openProjectCandidate(openArchive(new Uint8Array(await readFile(gamePath)), 'height-game.zip')[0]!);
  expect(game.zones.find(zone => zone.id === game.starts[0]!.zone)).toMatchObject({ floor: 32.25, ceiling: -160.5 });
  await expect(preview).toHaveAttribute('data-level-slot', '0');
  const after = await captureEngine(page, { zone: game.starts[0]!.zone, x: 20128, z: 20128, upper: 0, angle: 0 });
  expect(after).not.toBe(before);
  await page.mouse.move(0, 0); await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot({ path: testInfo.outputPath('authored-heights.png') });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await expect.poll(() => enginePlayerFeetFixed(page)).toBe(32.25 * 256);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  // Invalid pairs cannot overwrite the last complete set; Escape restores a
  // field without clearing the room selection or applying the other draft.
  await floor.fill('-200'); await page.getByRole('button', { name: 'Apply heights', exact: true }).click();
  await expect(page.getByText(/lower ceiling must be above the lower floor/)).toBeVisible();
  await floor.press('Escape'); await expect(floor).toHaveValue('32.25');
  await page.getByRole('combobox', { name: 'Copy heights from room' }).selectOption(fixture.to.id);
  await page.getByRole('button', { name: 'Load heights', exact: true }).click();
  await expect(floor).toHaveValue(String(fixture.original.zones[0]!.floor));
  await page.getByRole('button', { name: 'Apply heights', exact: true }).click();
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(floor).toHaveValue('32.25');
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(floor).toHaveValue(String(fixture.original.zones[0]!.floor));
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  const savedDownload = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('height-source.ab3d'); await (await savedDownload).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.files).toEqual(fixture.document.files);
  expect(saved.roomSource.rooms.find(room => room.id === fixture.from.id)!.heights).toMatchObject({ floor: 32.25, ceiling: -160.5 });
  expect(saved.roomSource.rooms.find(room => room.id === fixture.to.id)!.heights).toBeUndefined();
  await page.getByRole('button', { name: 'Restore template heights', exact: true }).click();
  await expect(floor).toHaveValue(String(fixture.original.zones[0]!.floor));
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(floor).toHaveValue('32.25');
  await expect(page.getByText('Workspace saved', { exact: true })).toBeVisible();
  await page.reload(); await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(fixture.from.originZone).padStart(3, '0')}(?:\\s|$)`) }).click();
  await expect(page.getByRole('spinbutton', { name: 'Lower floor height', exact: true })).toHaveValue('32.25');
});

test('an upper ceiling edit changes the actual upper-storey view without changing the lower floor', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const fixture = roomHeightFixture(twoStoreyHeightTemplate()), template = fixture.original.zones[fixture.from.recordZone]!;
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'upper-heights.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const camera = { zone: Number(await preview.getAttribute('data-spawn-zone')), x: 20128, z: 20128, upper: 1, angle: 0 };
  const before = await captureEngine(page, camera);
  await page.getByRole('spinbutton', { name: 'Upper ceiling height', exact: true }).fill(String(template.upperCeiling - 16));
  await page.getByRole('button', { name: 'Apply heights', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const path = testInfo.outputPath('upper-game.zip'); await (await download).saveAs(path);
  const game = openProjectCandidate(openArchive(new Uint8Array(await readFile(path)), 'upper-game.zip')[0]!);
  expect(game.zones.find(zone => zone.id === game.starts[0]!.zone)).toMatchObject({ floor: template.floor, ceiling: template.ceiling,
    upperFloor: template.upperFloor, upperCeiling: template.upperCeiling - 16 });
  await expect(preview).toHaveAttribute('data-level-slot', '0');
  expect(await captureEngine(page, { ...camera, zone: game.starts[0]!.zone })).not.toBe(before);
  await page.mouse.move(0, 0); await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot({ path: testInfo.outputPath('upper-height.png') });
});

test('different authored heights generate a visible step and the native player walks up onto its floor', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const fixture = importedPortalFixture(), preview = page.getByRole('region', { name: '3D game engine preview' });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'step-heights.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.added.id.replace(/^room-/, '')}`) }).click();
  await page.getByRole('spinbutton', { name: 'Lower floor height', exact: true }).fill('-32');
  await page.getByRole('spinbutton', { name: 'Lower ceiling height', exact: true }).fill('-160');
  await page.getByRole('button', { name: 'Apply heights', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const path = testInfo.outputPath('step-game.zip'); await (await download).saveAs(path);
  const game = openProjectCandidate(openArchive(new Uint8Array(await readFile(path)), 'step-game.zip')[0]!);
  const inside = { x: fixture.x + fixture.normal.x * 128, z: fixture.z + fixture.normal.z * 128 };
  const destination = game.zones.find(zone => zoneContainsPoint(game, zone, inside.x, inside.z))!;
  expect(destination).toMatchObject({ floor: -32, ceiling: -160 });
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const angle = fixture.normal.x > 0 ? 2048 : fixture.normal.x < 0 ? 6144 : fixture.normal.z > 0 ? 0 : 4096;
  await captureEngine(page, { ...fixture.start, zone: game.starts[0]!.zone, upper: 0, angle });
  await page.mouse.move(0, 0); await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot({ path: testInfo.outputPath('authored-step.png') });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus();
  const key = fixture.normal.x > 0 ? 'd' : fixture.normal.x < 0 ? 'a' : fixture.normal.z > 0 ? 'w' : 's';
  await page.keyboard.down(key);
  try { await expect.poll(async () => (await enginePlayerState(page)).zone).toBe(destination.id); }
  finally { await page.keyboard.up(key); }
  await expect.poll(() => enginePlayerFeetFixed(page)).toBe(-32 * 256);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
});
