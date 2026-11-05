import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { copiedSwitchAuthoringFixture } from '../copiedSwitchAuthoringFixture';
import { switchCopyPlayerFixture } from '../switchCopyFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { engineSwitchState, enginePlayerInventory, engineMechanismState, captureEngine, enginePickDetails } from '../engineCapture';

test('moves a required copied switch, saves and undoes its attachments, and opens its copied native door with the same condition bit', async ({ page }, info) => {
  test.setTimeout(90000);
  const fixture = copiedSwitchAuthoringFixture(), document = fixture.document, slot = fixture.switchCopy.slot;
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message));
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'copied-switch-move.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.switchCopy.room.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  const section = page.getByRole('region', { name: `Copied switch ${slot + 1}`, exact: true });
  await expect(section.getByRole('button', { name: `Remove switch ${slot + 1}`, exact: true })).toBeDisabled();
  await section.getByRole('combobox', { name: `copied switch ${slot + 1} reset mode`, exact: true }).selectOption('manual');
  await section.locator('summary').click();
  const destination = section.getByRole('combobox', { name: `Switch ${slot + 1} destination room`, exact: true });
  const wall = section.getByRole('combobox', { name: `Switch ${slot + 1} destination wall`, exact: true });
  await destination.selectOption(fixture.motion.room); await wall.selectOption('0');
  await section.getByRole('button', { name: `Move switch ${slot + 1}`, exact: true }).click();
  await expect(section.getByRole('alert')).toContainText('Choose a static room');
  await destination.selectOption(fixture.room.id); await wall.selectOption('0');
  await section.getByRole('button', { name: `Move switch ${slot + 1}`, exact: true }).click(); await expect(section).toHaveCount(0);
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(section).toBeVisible();
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(section).toHaveCount(0);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.room.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  const switches = page.getByRole('region', { name: 'Wall switches', exact: true }); await switches.locator('summary').first().click();
  const group = switches.getByRole('group', { name: `Authored switch ${slot + 1}`, exact: true }); await expect(group).toBeVisible();
  await expect(group.getByRole('button', { name: `Remove switch ${slot + 1}`, exact: true })).toBeDisabled();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('moved-copied-switch.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.files).toEqual(document.files); expect(saved.roomSource.switchCopies).toBeUndefined();
  expect(saved.roomSource.controllerCopies).toEqual(document.roomSource.controllerCopies);
  expect(saved.roomSource.rooms.at(-1)!.switches![0]!.settings?.[2]).toBe(0);
  const built = buildPlayableLevel(saved, -1).document, native = built.switches[slot]!, door = built.mechanisms.filter(item => item.kind === 'door').at(-1)!;
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' }); await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  let panelVisible = false;
  for (const angle of [0, 2048, 4096, 6144]) {
    await captureEngine(page, { zone: native.zone, x: 20032, z: 20128, upper: 0, angle, height: built.zones[native.zone]!.floor - 48 });
    const picks = await enginePickDetails(page, Array.from({ length: 81 }, (_, index) => ({ x: (index % 9 + 1) / 10, y: (Math.floor(index / 9) + 1) / 10 })));
    if (picks.some(pick => pick.surface?.offset === native.graphic - 10)) { panelVisible = true; break; }
  }
  expect(panelVisible).toBe(true); await page.screenshot({ path: info.outputPath('moved-copied-switch-panel.png') });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const originalState = await engineSwitchState(page, fixture.switchTemplate.id);
  expect((await enginePlayerInventory(page)).conditions).toBe(0);
  expect(await engineMechanismState(page, 'door', door.id)).toMatchObject({ current: door.bottom, unsynchronizedWalls: 0 });
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus(); await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(1 << (slot + 4)); }
  finally { await page.keyboard.up('Space'); }
  expect(await engineSwitchState(page, slot)).toMatchObject({ active: true, phase: 2 });
  await expect.poll(async () => (await engineMechanismState(page, 'door', door.id)).current).toBe(door.top);
  expect(await engineMechanismState(page, 'door', door.id)).toMatchObject({ unsynchronizedWalls: 0 });
  expect(await engineSwitchState(page, fixture.switchTemplate.id)).toEqual(originalState);
  await expect(preview.getByRole('alert')).toHaveCount(0); expect(errors).toEqual([]);
});

test('removes only an unrequired copied switch, saves its stripped room and leaves its original native controller active', async ({ page }, info) => {
  const fixture = switchCopyPlayerFixture(), document = fixture.document, slot = fixture.switchCopy.slot;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'copied-switch-remove.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.switchCopy.room.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  const section = page.getByRole('region', { name: `Copied switch ${slot + 1}`, exact: true });
  await section.getByRole('button', { name: `Remove switch ${slot + 1}`, exact: true }).click(); await expect(section).toHaveCount(0);
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(section).toBeVisible();
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(section).toHaveCount(0);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('removed-copied-switch.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.files).toEqual(document.files); expect(saved.roomSource.clearedSwitchSlots).toEqual([slot]);
  expect(saved.roomSource.rooms.find(room => room.id === fixture.switchCopy.graphicsRoom)!.retiredCopiedSwitches).toEqual([fixture.source.id]);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' }); await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  expect(await engineSwitchState(page, slot)).toEqual({ zone: -1, active: false, phase: null, countdown: 0 });
  expect((await engineSwitchState(page, fixture.source.id)).zone).toBeGreaterThanOrEqual(0);
  await expect(preview.getByRole('alert')).toHaveCount(0);
});
