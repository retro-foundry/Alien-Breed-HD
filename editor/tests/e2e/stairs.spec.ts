import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { stairFixture } from '../stairFixture';
import { decodeProject, encodeProject, openArchive, openProjectCandidate } from '../../src/model/project';
import { zoneContainsPoint } from '../../src/model/level';
import { captureEngine, enginePlayerFeetFixed, enginePlayerState } from '../engineCapture';

test('stair draft cancellation, one-step undo, source backup and native walking in both directions', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const fixture = stairFixture(), preview = page.getByRole('region', { name: '3D game engine preview' });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'stairs.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.approach.id.replace(/^room-/, '')}`) }).click();
  const dialog = page.getByRole('dialog', { name: 'Create stairs', exact: true });
  const fillFlight = async () => {
    await page.getByRole('button', { name: 'Build stairs', exact: true }).click();
    await expect(dialog).toBeVisible();
    for (const [name, value] of Object.entries({ 'Stair start X': 20064, 'Stair start Z': 20128, 'Stair end X': 20064,
      'Stair end Z': 20640, 'Stair width': 128, 'Number of steps': 4, 'Floor before first step': 0, 'Final step floor': -64, 'Stair clearance': 128 })) {
      await dialog.getByRole('spinbutton', { name, exact: true }).fill(String(value));
    }
    await dialog.getByRole('combobox', { name: 'Stair lower wall style', exact: true }).selectOption(String(fixture.flight.appearance.lowerWall));
    await dialog.getByRole('combobox', { name: 'Stair corner lighting', exact: true }).selectOption(String(fixture.flight.appearance.lightingPoint));
    await expect(dialog.getByRole('status')).toContainText('4 steps ready');
  };
  await fillFlight();
  await dialog.getByRole('spinbutton', { name: 'Stair clearance', exact: true }).focus();
  await expect(dialog.getByRole('tooltip')).toContainText('standing/crouched clearance');
  const draftSave = page.waitForEvent('download'); await page.keyboard.press('Control+s');
  const draftPath = testInfo.outputPath('uncommitted-stairs.ab3d'); await (await draftSave).saveAs(draftPath);
  expect(decodeProject(new Uint8Array(await readFile(draftPath))).roomSource).toEqual(fixture.document.roomSource);
  await expect(dialog).toBeVisible();
  await dialog.getByRole('spinbutton', { name: 'Final step floor', exact: true }).fill('-160');
  await expect(dialog.getByRole('alert')).toContainText('risers below 40');
  await expect(dialog.getByRole('button', { name: 'Create stairs', exact: true })).toBeDisabled();
  await page.keyboard.press('Escape'); await expect(dialog).not.toBeVisible();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await fillFlight();
  await dialog.getByRole('button', { name: 'Cancel stairs', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await fillFlight();
  await dialog.getByRole('button', { name: 'Create stairs', exact: true }).click();
  await expect(dialog).not.toBeVisible();
  const floor = page.getByRole('spinbutton', { name: 'Lower floor height', exact: true });
  await expect(floor).toHaveValue('-16');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  const save = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const sourcePath = testInfo.outputPath('stairs.ab3d'); await (await save).saveAs(sourcePath);
  const saved = decodeProject(new Uint8Array(await readFile(sourcePath)));
  expect(saved.files).toEqual(fixture.document.files);
  expect(saved.roomSource.rooms.slice(-4).map(room => room.heights?.floor)).toEqual([-16, -32, -48, -64]);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const gamePath = testInfo.outputPath('stairs-game.zip'); await (await download).saveAs(gamePath);
  const game = openProjectCandidate(openArchive(new Uint8Array(await readFile(gamePath)), 'stairs-game.zip')[0]!);
  const destination = game.zones.find(zone => zoneContainsPoint(game, zone, 20064, 20576))!;
  expect(destination).toMatchObject({ floor: -64, ceiling: -192 });
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await captureEngine(page, { zone: game.starts[0]!.zone, x: 20064, z: 20064, upper: 0, angle: 0 });
  await page.mouse.move(0, 0);
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  await canvas.screenshot({ path: testInfo.outputPath('authored-stair-flight.png') });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await canvas.focus(); await page.keyboard.down('w');
  try { await expect.poll(async () => (await enginePlayerState(page)).zone, { timeout: 15000 }).toBe(destination.id); }
  finally { await page.keyboard.up('w'); }
  await expect.poll(() => enginePlayerFeetFixed(page)).toBe(-64 * 256);
  await page.keyboard.down('s');
  try { await expect.poll(async () => (await enginePlayerState(page)).zone, { timeout: 15000 }).toBe(game.starts[0]!.zone); }
  finally { await page.keyboard.up('s'); }
  await expect.poll(() => enginePlayerFeetFixed(page)).toBe(0);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
});
