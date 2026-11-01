import { expect, test } from '@playwright/test';
import { applyCommand, editRoomSource, setSourceStart } from '../../src/model/commands';
import { openLevel } from '../../src/model/level';
import { decodeProject, encodeProject } from '../../src/model/project';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../../src/model/roomSource';
import { buildSourceLevel } from '../../src/model/sourceBuild';
import { originalFiles } from '../helpers';
import { readFile } from 'node:fs/promises';

test('the editor places starts in authored rooms, undoes them and recovers their saved positions', async ({ page }, testInfo) => {
  page.on('dialog', dialog => void dialog.accept());
  const original = openLevel('Start authoring', originalFiles(1));
  const change = commitRoomDraft(original.roomSource, setDraftCorners(beginRoomDraft(original, original.roomSource, 0),
    [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }]));
  const drawn = applyCommand(original, editRoomSource(change)), room = change.after.rooms.at(-1)!;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'drawn.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(drawn)) });
  await expect(page.locator('.project-picker strong')).toHaveText(drawn.name);
  await page.getByRole('tab', { name: 'Starts', exact: true }).click();
  await page.getByRole('button', { name: /^Player 1/ }).click();
  const roomField = page.getByRole('combobox', { name: 'Player start room' });
  await roomField.selectOption(room.id);
  await page.getByRole('spinbutton', { name: 'Position X' }).fill('20064');
  await page.getByRole('spinbutton', { name: 'Position Z' }).fill('20032');
  await page.getByRole('button', { name: 'Apply position', exact: true }).click();
  await expect(page.locator('.dirty-dot')).toBeVisible();
  await expect(roomField).toHaveValue(room.id);
  await expect(page.getByRole('button', { name: /^Player 1/ })).toContainText(room.id);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('spinbutton', { name: 'Position X' })).toHaveValue(String(original.starts[0]!.x));
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await expect(page.getByRole('spinbutton', { name: 'Position X' })).toHaveValue('20064');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('authored-start.ab3d'); await (await download).saveAs(path);
  const restored = decodeProject(new Uint8Array(await readFile(path)));
  expect(restored.sourceStarts).toEqual([{ player: 1, room: room.id, x: 20064, z: 20032 }]);
  expect(restored.files).toEqual(original.files);
  await expect(page.getByText('Workspace saved', { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.locator('.project-picker strong')).toHaveText('Start authoring');
  await page.getByRole('tab', { name: 'Starts', exact: true }).click();
  await page.getByRole('button', { name: /^Player 1/ }).click();
  await expect(page.getByRole('combobox', { name: 'Player start room' })).toHaveValue(room.id);
  await expect(page.getByRole('spinbutton', { name: 'Position X' })).toHaveValue('20064');
  await expect(page.getByRole('spinbutton', { name: 'Position Z' })).toHaveValue('20032');
});

test('a saved authored start spawns the real game inside a newly drawn room', async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const original = openLevel('Authored spawn', originalFiles(1));
  const change = commitRoomDraft(original.roomSource, setDraftCorners(beginRoomDraft(original, original.roomSource, 0),
    [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }]));
  const drawn = applyCommand(original, editRoomSource(change)), room = change.after.rooms.at(-1)!;
  const saved = decodeProject(encodeProject(applyCommand(drawn, setSourceStart(drawn, 1, room.id, 20064, 20064))));
  const wall = original.zones[0]!.lower.find(record => record.wall)!.wall!;
  const built = buildSourceLevel(saved, { resolvedExit: -1,
    newRoomGraphics: new Map([[room.id, { walls: { lower: [wall] }, wallInsertion: { lower: 0 } }]]),
    newRoomLighting: new Map([[room.id, { kind: 'copy', source: 0 }]]) });
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('dialog', dialog => void dialog.accept());
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'compiled.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(built.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(built.document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toHaveText(`${built.document.zones.length} zones loaded · Game engine`, { timeout: 60000 });
  const spawn = built.document.starts[0]!;
  expect(built.topology.cells[spawn.zone]!.sourceId).toBe(room.id);
  await expect(preview).toHaveAttribute('data-spawn-zone', String(spawn.zone));
  await expect.poll(async () => Number(await preview.getAttribute('data-rendered-pixels'))).toBeGreaterThan(1000);
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await expect(preview).toHaveAttribute('data-spawn-zone', String(spawn.zone));
  await page.screenshot({ path: testInfo.outputPath('authored-room-spawn.png'), fullPage: true });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText(`${built.document.zones.length} zones loaded · Game engine`);
  await expect(preview.getByRole('alert')).toHaveCount(0);
  expect(errors).toEqual([]);
});
