import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { applyCommand, editRoomSource, setSourceStart, setSourceTeleport } from '../../src/model/commands';
import { openLevel } from '../../src/model/level';
import { decodeProject, encodeProject } from '../../src/model/project';
import { beginRoomDraft, commitRoomDraft, removeSourceRooms, setDraftCorners } from '../../src/model/roomSource';
import { buildSourceLevel } from '../../src/model/sourceBuild';
import { originalFiles } from '../helpers';
import { enginePlayerState } from '../engineCapture';

function fixture() {
  const original = openLevel('Teleporter authoring', originalFiles(1));
  let document = original;
  for (const x of [20000, 21000]) document = applyCommand(document, editRoomSource(commitRoomDraft(document.roomSource,
    setDraftCorners(beginRoomDraft(document, document.roomSource, 0), [{ x, z: 20000 }, { x: x + 128, z: 20000 }, { x: x + 128, z: 20128 }, { x, z: 20128 }]))));
  return { original, document, from: document.roomSource.rooms.at(-2)!, to: document.roomSource.rooms.at(-1)! };
}

test('teleporter links can be authored, shown on the map, disabled, undone and recovered', async ({ page }, testInfo) => {
  page.on('dialog', dialog => void dialog.accept());
  const { original, document, from, to } = fixture();
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'drawn.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${from.id.replace(/^room-/, '')}`) }).click();
  const destination = page.getByRole('combobox', { name: 'Teleport destination room' });
  await destination.selectOption(to.id);
  await page.getByRole('spinbutton', { name: 'Position X' }).fill('21064');
  await page.getByRole('spinbutton', { name: 'Position Z' }).fill('20064');
  await page.getByRole('button', { name: 'Apply position', exact: true }).click();
  await expect(page.locator('.dirty-dot')).toBeVisible();
  await expect(page.getByText('Enabled. Entering this room teleports to the landing point below.')).toBeVisible();
  await page.getByRole('button', { name: 'Show destination', exact: true }).click();
  await expect(page.locator('canvas[aria-label="Level map"]')).toHaveAttribute('data-world-x', '21064');
  await expect.poll(async () => Number(await page.locator('canvas[aria-label="Level map"]').getAttribute('data-world-scale'))).toBeGreaterThan(0.5);
  await expect(page.getByRole('heading', { name: `Room ${from.id.replace(/^room-/, '')}`, exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('teleport-destination-overlay.png'), fullPage: true });
  await page.getByRole('button', { name: 'Disable teleporter', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Show destination', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(destination).toHaveValue(to.id);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(destination).toHaveValue('');
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await expect(destination).toHaveValue(to.id);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('teleport-source.ab3d'); await (await download).saveAs(path);
  const restored = decodeProject(new Uint8Array(await readFile(path)));
  expect(restored.sourceTeleports).toEqual([{ room: from.id, destination: to.id, x: 21064, z: 20064 }]);
  expect(restored.files).toEqual(original.files);
  await expect(page.getByText('Workspace saved', { exact: true })).toBeVisible();
  await page.reload(); await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${from.id.replace(/^room-/, '')}`) }).click();
  await expect(destination).toHaveValue(to.id);
  await expect(page.getByRole('spinbutton', { name: 'Position X' })).toHaveValue('21064');
});

test('the actual game teleports from an authored room to its saved destination', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const { original, document, from, to } = fixture();
  let source = applyCommand(document, setSourceStart(document, 1, from.id, 20064, 20064));
  source = applyCommand(source, setSourceTeleport(source, from.id, to.id, 21064, 20064));
  source = decodeProject(encodeProject(source));
  const wall = original.zones[0]!.lower.find(record => record.wall)!.wall!;
  const built = buildSourceLevel(source, { resolvedExit: -1,
    newRoomGraphics: new Map([from, to].map(room => [room.id, { walls: { lower: [wall] }, wallInsertion: { lower: 0 } }])),
    newRoomLighting: new Map([from, to].map(room => [room.id, { kind: 'copy', source: 0 }])) });
  const fromId = built.topology.sourceCells.get(from.id)![0]!, toId = built.topology.sourceCells.get(to.id)![0]!;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'compiled.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(built.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(built.document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toHaveText(`${built.document.zones.length} zones loaded · Game engine`, { timeout: 60000 });
  expect(await enginePlayerState(page)).toEqual({ zone: fromId, x: 20064, z: 20064, playing: false });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await expect.poll(async () => (await enginePlayerState(page)).zone).toBe(toId);
  expect(await enginePlayerState(page)).toEqual({ zone: toId, x: 21064, z: 20064, playing: true });
  await page.screenshot({ path: testInfo.outputPath('authored-teleport-arrival.png'), fullPage: true });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText(`${built.document.zones.length} zones loaded · Game engine`);
  await expect(preview.getByRole('alert')).toHaveCount(0);
});

test('missing destination rooms are visible and repairable without losing the teleporter', async ({ page }) => {
  page.on('dialog', dialog => void dialog.accept());
  const { document, from, to } = fixture();
  let broken = applyCommand(document, setSourceTeleport(document, from.id, to.id, 21064, 20064));
  broken = applyCommand(broken, editRoomSource(removeSourceRooms(broken.roomSource, [to.id])));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'broken-link.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(broken)) });
  await expect(page.locator('.project-picker strong')).toHaveText(broken.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${from.id.replace(/^room-/, '')}`) }).click();
  await expect(page.getByText('Destination room is missing. Choose another room, or disable this teleporter.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Show destination', exact: true })).toBeDisabled();
  const start = document.starts[0]!, room = document.roomSource.rooms.find(room => room.originZone === start.zone)!;
  await page.getByRole('combobox', { name: 'Teleport destination room' }).selectOption(room.id);
  await page.getByRole('spinbutton', { name: 'Position X' }).fill(String(start.x));
  await page.getByRole('spinbutton', { name: 'Position Z' }).fill(String(start.z));
  await page.getByRole('button', { name: 'Apply position', exact: true }).click();
  await expect(page.getByText('Enabled. Entering this room teleports to the landing point below.')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Show destination', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByText('Destination room is missing. Choose another room, or disable this teleporter.')).toBeVisible();
});
