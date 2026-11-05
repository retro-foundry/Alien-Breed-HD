import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { authoredMovingFixture } from '../authoredMovingFixture';
import { switchCopyPlayerFixture } from '../switchCopyFixture';
import { applyCommand, editRoomSource, setSourceStart } from '../../src/model/commands';
import { setSwitchAutomaticReset } from '../../src/model/switchSettings';
import { setSourceMovingRoom } from '../../src/model/sourceMovingRoom';
import { roomAppearanceChoices } from '../../src/model/roomAppearance';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { encodeProject, decodeProject } from '../../src/model/project';
import { engineSwitchState, engineMechanismState, enginePlayerInventory, captureEngine, enginePickDetails } from '../engineCapture';

test('moves an original switch to a new room, saves and undoes it, and releases a native door with the same condition number', async ({ page }, info) => {
  test.setTimeout(90000);
  const fixture = authoredMovingFixture('door'), original = fixture.document.switches.find(item => item.zone >= 0)!, slot = original.id;
  let document = applyCommand(fixture.document, setSwitchAutomaticReset(fixture.document, slot, false));
  const header = fixture.input.header.slice(), view = new DataView(header.buffer);
  view.setInt16(4, view.getInt16(0)); view.setUint16(14, 1 << (slot + 4)); view.setUint8(16, 4); view.setUint8(17, 1);
  document = applyCommand(document, editRoomSource(setSourceMovingRoom(document, fixture.room.id, { kind: 'door', header: [...header], wall: roomAppearanceChoices(document, fixture.room).lower[0]!.offset })));
  document = applyCommand(document, setSourceStart(document, 1, fixture.neighbour.id, 20480, 20128));
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'original-switch.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(original.zone).padStart(3, '0')}(?:\\s|$)`) }).click();
  const section = page.getByRole('region', { name: `Switch ${slot + 1}`, exact: true });
  await expect(section.getByRole('button', { name: `Remove switch ${slot + 1}`, exact: true })).toBeDisabled();
  await section.locator('summary').click();
  const destination = section.getByRole('combobox', { name: `Switch ${slot + 1} destination room`, exact: true });
  const wall = section.getByRole('combobox', { name: `Switch ${slot + 1} destination wall`, exact: true });
  await destination.selectOption(fixture.room.id); await wall.selectOption('0');
  await section.getByRole('button', { name: `Move switch ${slot + 1}`, exact: true }).click();
  await expect(section.getByRole('alert')).toContainText('Choose a static room');
  await destination.selectOption(fixture.neighbour.id); await wall.selectOption('2');
  await section.getByRole('button', { name: `Move switch ${slot + 1}`, exact: true }).click();
  await expect(section).toHaveCount(0);
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(section).toBeVisible();
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(section).toHaveCount(0);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.neighbour.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  const authored = page.getByRole('region', { name: 'Wall switches', exact: true }); await authored.locator('summary').first().click();
  const group = authored.getByRole('group', { name: `Authored switch ${slot + 1}`, exact: true });
  await expect(group).toBeVisible(); await expect(group.getByRole('combobox', { name: `Switch ${slot + 1} wall`, exact: true })).toHaveValue('2');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('reassigned-native-switch.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.files).toEqual(document.files); expect(saved.roomSource.removedNativeSwitches).toEqual([slot]);
  const built = buildPlayableLevel(saved, -1).document, native = built.switches[slot]!, door = built.mechanisms.filter(item => item.kind === 'door').at(-1)!;
  expect(native.zone).not.toBe(original.zone);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  let visible = false;
  for (const angle of [0, 2048, 4096, 6144]) {
    await captureEngine(page, { zone: native.zone, x: 20480, z: 20128, upper: 0, angle, height: built.zones[native.zone]!.floor - 48 });
    const picks = await enginePickDetails(page, Array.from({ length: 81 }, (_, index) => ({ x: (index % 9 + 1) / 10, y: (Math.floor(index / 9) + 1) / 10 })));
    if (picks.some(pick => pick.surface?.offset === native.graphic - 10)) { visible = true; break; }
  }
  expect(visible).toBe(true);
  await page.screenshot({ path: info.outputPath('native-reassigned-panel.png') });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  expect(await engineMechanismState(page, 'door', door.id)).toMatchObject({ current: door.bottom, unsynchronizedWalls: 0 });
  expect((await enginePlayerInventory(page)).conditions).toBe(0);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus(); await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(1 << (slot + 4)); }
  finally { await page.keyboard.up('Space'); }
  expect(await engineSwitchState(page, slot)).toMatchObject({ active: true, phase: 2 });
  await expect.poll(async () => (await engineMechanismState(page, 'door', door.id)).current).toBe(door.top);
  expect(await engineMechanismState(page, 'door', door.id)).toMatchObject({ unsynchronizedWalls: 0 });
  await expect(preview.getByRole('alert')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('removes an unrequired original switch without removing its independently copied controller', async ({ page }, info) => {
  const { document, source: original, switchCopy: copy } = switchCopyPlayerFixture();
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'remove-native-switch.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(original.zone).padStart(3, '0')}(?:\\s|$)`) }).click();
  const section = page.getByRole('region', { name: `Switch ${original.id + 1}`, exact: true });
  await section.getByRole('button', { name: `Remove switch ${original.id + 1}`, exact: true }).click(); await expect(section).toHaveCount(0);
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(section).toBeVisible();
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(section).toHaveCount(0);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('removed-native-switch.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.files).toEqual(document.files); expect(saved.roomSource.switchCopies).toEqual(document.roomSource.switchCopies);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' }); await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  expect(await engineSwitchState(page, original.id)).toMatchObject({ zone: -1, active: false, phase: null, countdown: 0 });
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus(); await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(1 << (copy.slot + 4)); }
  finally { await page.keyboard.up('Space'); }
  expect((await engineSwitchState(page, copy.slot)).zone).toBeGreaterThanOrEqual(0);
  expect(await engineSwitchState(page, original.id)).toMatchObject({ zone: -1, active: false, countdown: 0 });
  await expect(preview.getByRole('alert')).toHaveCount(0);
});
