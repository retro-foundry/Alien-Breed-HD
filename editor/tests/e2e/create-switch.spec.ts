import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { switchRoomFixture, switchDoorFixture } from '../sourceSwitchFixture';
import { applyCommand, setSourceStart } from '../../src/model/commands';
import { encodeProject, decodeProject } from '../../src/model/project';
import { enginePlayerInventory, engineSwitchState, engineMechanismState, engineMechanismTrace, captureEngine, enginePickDetails } from '../engineCapture';
import { buildPlayableLevel } from '../../src/model/playableBuild';

test('creates, reassigns and saves a new wall switch and operates its own native condition bit', async ({ page }, info) => {
  test.setTimeout(90000);
  const fixture = switchRoomFixture(), slot = fixture.slot;
  const document = applyCommand(fixture.document, setSourceStart(fixture.document, 1, fixture.room.id, 20032, 20128));
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'new-switch.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.room.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  const section = page.getByRole('region', { name: 'Wall switches', exact: true });
  await section.locator('summary').click();
  await section.getByRole('combobox', { name: 'New switch number', exact: true }).selectOption(String(slot));
  await section.getByRole('button', { name: 'Choose edge 1 for new switch', exact: true }).click();
  await expect(section.getByRole('combobox', { name: 'New switch wall', exact: true })).toHaveValue('0');
  await section.getByRole('button', { name: 'Add wall switch', exact: true }).click();
  const group = section.getByRole('group', { name: `Authored switch ${slot + 1}`, exact: true });
  await expect(group).toBeVisible(); await expect(section.getByRole('alert')).toHaveCount(0);
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(group).toHaveCount(0);
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(group).toBeVisible();
  const wall = group.getByRole('combobox', { name: `Switch ${slot + 1} wall`, exact: true });
  await wall.selectOption('1'); await expect(wall).toHaveValue('1'); await wall.selectOption('0');
  const mode = group.getByRole('combobox', { name: `authored switch ${slot + 1} reset mode`, exact: true });
  await mode.selectOption('automatic'); await expect(mode).toHaveValue('automatic');
  await mode.selectOption('manual');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('new-switch.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.files).toEqual(document.files);
  expect(saved.roomSource.rooms.at(-1)!.switches).toEqual([{ slot, boundary: 0, automatic: false }]);
  await page.screenshot({ path: info.outputPath('wall-switch-authoring.png') });
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const built = buildPlayableLevel(saved, -1).document, native = built.switches[slot]!;
  let visible = false;
  for (const angle of [0, 2048, 4096, 6144]) {
    await captureEngine(page, { zone: native.zone, x: 20064, z: 20128, upper: 0, angle, height: built.zones[native.zone]!.floor - 48 });
    const picks = await enginePickDetails(page, Array.from({ length: 81 }, (_, index) => ({ x: (index % 9 + 1) / 10, y: (Math.floor(index / 9) + 1) / 10 })));
    visible = picks.some(pick => pick.surface?.offset === native.graphic - 10);
    if (visible) break;
  }
  expect(visible).toBe(true);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus(); await page.keyboard.press('Enter');
  await expect(preview.getByRole('note', { name: 'Picked surface' })).toContainText(`Switch ${slot + 1}: its settings are in the room inspector`);
  await preview.getByRole('button', { name: `Edit switch ${slot + 1}`, exact: true }).click();
  await expect(group).toBeVisible();
  await page.screenshot({ path: info.outputPath('native-new-wall-switch.png') });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const original = document.switches.find(item => item.zone >= 0)!, before = await engineSwitchState(page, original.id);
  expect((await enginePlayerInventory(page)).conditions).toBe(0);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus();
  await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(1 << (slot + 4)); }
  finally { await page.keyboard.up('Space'); }
  expect(await engineSwitchState(page, slot)).toMatchObject({ active: true, phase: 2 });
  expect(await engineSwitchState(page, original.id)).toEqual(before);
  await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(0); }
  finally { await page.keyboard.up('Space'); }
  expect(await engineSwitchState(page, slot)).toMatchObject({ active: false, phase: 0 });
});

test('a new switch releases a gated native door and keeps its moving walls synchronized', async ({ page }) => {
  test.setTimeout(90000);
  const fixture = switchDoorFixture(), document = fixture.document;
  const built = buildPlayableLevel(document, -1).document, door = built.mechanisms.filter(item => item.kind === 'door').at(-1)!;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'switch-gated-door.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  expect(await engineMechanismState(page, 'door', door.id)).toMatchObject({ current: door.bottom, plane: door.bottom, unsynchronizedWalls: 0 });
  expect((await enginePlayerInventory(page)).conditions).toBe(0);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus();
  await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(1 << (fixture.slot + 4)); }
  finally { await page.keyboard.up('Space'); }
  const trace = await engineMechanismTrace(page, 'door', door.id, false);
  expect(trace.some(state => state.current < door.bottom)).toBe(true);
  for (const state of trace) expect(state).toMatchObject({ plane: state.current, unsynchronizedWalls: 0 });
  await expect.poll(async () => (await engineMechanismState(page, 'door', door.id)).current).toBe(door.top);
});
