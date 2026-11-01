import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture } from '../roomHeightFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { captureEngine, engineRoomProperties } from '../engineCapture';
import { openLevel } from '../../src/model/level';
import { originalFiles } from '../helpers';
import { buildGraphicsStreams, importedGraphicsStreams } from '../../src/model/graphicsBuild';
import { encodePolygonSurface } from '../../src/model/surfaces';

test('new room lighting is independent source data, survives saves and changes native pixels with one undo', async ({ page }, testInfo) => {
  const original = openLevel('Native bump surface room lighting', originalFiles(1)), stream = importedGraphicsStreams(original).get(0)!;
  const template = original.zones[0]!, floor = template.lower.find(record => record.type === 1)!;
  const baseline = openLevel(original.name, { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined, {
    streams: new Map([[0, { ...stream, lower: stream.lower.map(item => item.sourceOffset === floor.offset
      ? { bytes: encodePolygonSurface({ ...floor.polygon!, type: 8, texture: 0, heightQuarter: template.floor * 4 }) } : item) }]]),
  }) });
  const { document, from, to } = roomHeightFixture(0, baseline);
  const built = buildPlayableLevel(document, -1), cell = built.topology!.cells.find(cell => cell.sourceId === from.id)!;
  const camera = { zone: cell.id, x: 20128, z: 20128, upper: 0, angle: 0, height: template.floor - 16 };
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'room-properties.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${from.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  const lighting = page.getByRole('region', { name: 'Room lighting and properties' });
  const brightness = lighting.getByRole('spinbutton', { name: 'Brightness', exact: true });
  await brightness.fill('127'); await brightness.press('Tab');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('room-properties.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.files.data).toEqual(document.files.data);
  expect(saved.roomSource.rooms.find(room => room.id === from.id)!.properties).toMatchObject({ brightness: 127 });
  expect(saved.roomSource.rooms.find(room => room.id === to.id)!.properties).toBeUndefined();
  const generated = buildPlayableLevel(saved, -1);
  expect(generated.document.zones[cell.id]!.brightness).toBe(127);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', String(document.playtestSlot), { timeout: 60000 });
  const surfaceFrame = () => captureEngine(page, camera);
  const edited = await surfaceFrame();
  expect(edited).not.toBe(0);
  expect(await captureEngine(page, { ...camera, height: template.ceiling - 1 })).toBe(0);
  expect((await engineRoomProperties(page, cell.id)).brightness).toBe(127);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect.poll(async () => (await engineRoomProperties(page, cell.id)).brightness).toBe(document.zones[from.recordZone]!.brightness);
  await expect.poll(surfaceFrame).not.toBe(edited);
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await expect.poll(surfaceFrame).toBe(edited);
  await lighting.getByRole('button', { name: 'Inherit template properties', exact: true }).click();
  await expect(brightness).toHaveValue(String(document.zones[from.recordZone]!.brightness));
  await expect.poll(surfaceFrame).not.toBe(edited);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect.poll(surfaceFrame).toBe(edited);
});
