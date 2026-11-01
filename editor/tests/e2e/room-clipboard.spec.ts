import { expect, test } from '@playwright/test';
import { linkedRoomCopyFixture } from '../linkedRoomCopyFixture';
import { switchCopyPlayerFixture } from '../switchCopyFixture';
import { copyRooms, pasteRooms } from '../../src/model/roomClipboard';
import { applyCommand, setSourceStart } from '../../src/model/commands';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { decodeProject, encodeProject } from '../../src/model/project';
import { engineMechanismState, engineMechanismTrace, enginePlayerInventory, engineSwitchState } from '../engineCapture';
import { readFile } from 'node:fs/promises';

test('a pasted room group activates its copied door through its copied switch without activating the source circuit', async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const fixture = linkedRoomCopyFixture(), result = pasteRooms(fixture.document, copyRooms(fixture.document, fixture.ids), -8000, 0);
  const pasted = applyCommand(fixture.document, result.command), copiedSwitch = pasted.roomSource.switchCopies!.at(-1)!;
  const oldSwitch = pasted.roomSource.switchCopies![0]!;
  const start = switchCopyPlayerFixture().document.sourceStarts!.find(start => start.player === 1)!;
  const document = applyCommand(pasted, setSourceStart(pasted, 1, copiedSwitch.room, start.x - 8000, start.z));
  const built = buildPlayableLevel(document, -1), copiedDoor = built.document.mechanisms.filter(item => item.kind === 'door').at(-1)!;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'room-clipboard.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  expect(await engineMechanismState(page, 'door', 1)).toMatchObject({ current: copiedDoor.current });
  expect(await engineMechanismState(page, 'door', 2)).toMatchObject({ current: copiedDoor.current });
  expect((await enginePlayerInventory(page)).conditions).toBe(0);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').focus();
  await page.keyboard.down('Space');
  try { await expect.poll(async () => (await enginePlayerInventory(page)).conditions).toBe(1 << (copiedSwitch.slot + 4)); }
  finally { await page.keyboard.up('Space'); }
  const trace = await engineMechanismTrace(page, 'door', 2, false);
  expect(trace.at(-1)!.current).toBe(copiedDoor.top);
  for (const state of trace) expect(state).toMatchObject({ plane: state.current, unsynchronizedWalls: 0 });
  expect(await engineMechanismState(page, 'door', 1)).toMatchObject({ current: copiedDoor.current, plane: copiedDoor.current, unsynchronizedWalls: 0 });
  expect(await engineSwitchState(page, oldSwitch.slot)).toMatchObject({ active: false });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('pasted-circuit-source.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.roomSource).toEqual(document.roomSource); expect(saved.files).toEqual(document.files);
});
