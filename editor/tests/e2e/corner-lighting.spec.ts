import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { nativeRoomCopyFixture } from '../nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from '../roomHeightFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { captureEngine, enginePickDetails, type EnginePickSample } from '../engineCapture';

test('authors and natively shades each of the four independent floor/ceiling corner channels', async ({ page }, info) => {
  test.setTimeout(180000);
  const { document, copy: room } = nativeRoomCopyFixture(twoStoreyHeightTemplate()), built = buildPlayableLevel(document, -1);
  const zone = document.zones[room.recordZone]!, planes = [
    { upper: false, record: zone.lower.find(item => item.type === 1)! }, { upper: false, record: zone.lower.find(item => item.type === 2)! },
    { upper: true, record: zone.upper.find(item => item.type === 1)! }, { upper: true, record: zone.upper.find(item => item.type === 2)! },
  ];
  const cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!, output = built.document.zones[cell.id]!;
  const x = Math.round(room.vertices.reduce((sum, p) => sum + p.x, 0) / room.vertices.length), z = Math.round(room.vertices.reduce((sum, p) => sum + p.z, 0) / room.vertices.length);
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'corner-lighting.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${room.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' }); await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await page.locator('summary').filter({ hasText: /^Corner lighting$/ }).click();
  const lowerDraft = page.getByRole('spinbutton', { name: 'Lower corner brightness', exact: true });
  await lowerDraft.fill('17'); await lowerDraft.press('Escape'); await expect(lowerDraft).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  const samples = Array.from({ length: 361 }, (_, index) => ({ x: (index % 19 + 1) / 20, y: (Math.floor(index / 19) + 1) / 20 }));
  for (let index = 0; index < planes.length; index++) {
    const { upper, record } = planes[index]!, floor = upper ? output.upperFloor : output.floor, ceiling = upper ? output.upperCeiling : output.ceiling;
    const camera = { zone: cell.id, x, z, upper: upper ? 1 : 0, angle: 0, height: record.type === 1 ? floor - 8 : ceiling + 8 };
    const before: EnginePickSample[][] = [];
    for (const angle of [0, 2048, 4096, 6144]) {
      await captureEngine(page, { ...camera, angle }); before.push(await enginePickDetails(page, samples));
    }
    const identity = built.surfaceSources!.find(item => item.room === cell.id && item.upper === upper && item.sourceOffset === record.offset)!;
    expect(before.some(view => view.some(pick => pick.surface?.offset === identity.offset))).toBe(true);
    await page.getByRole('combobox', { name: 'Corner lighting surfaces', exact: true }).selectOption(String(record.offset));
    await page.getByRole('spinbutton', { name: 'Lower corner brightness', exact: true }).fill(String(upper ? 0 : 63 - index * 8));
    await page.getByRole('spinbutton', { name: 'Upper corner brightness', exact: true }).fill(String(upper ? 63 - index * 8 : 0));
    const revision = await preview.getAttribute('data-revision');
    await page.getByRole('button', { name: 'Apply corner lighting', exact: true }).click();
    await expect(preview).not.toHaveAttribute('data-revision', revision!); await expect(preview.getByRole('status')).toContainText('zones loaded');
    let changedCount = 0;
    for (const [view, angle] of [0, 2048, 4096, 6144].entries()) {
      await captureEngine(page, { ...camera, angle }); const after = await enginePickDetails(page, samples);
      const changed = after.filter((pick, at) => pick.pixel !== before[view]![at]!.pixel); changedCount += changed.length;
      for (const pick of changed) expect(pick.surface?.offset).toBe(identity.offset);
    }
    expect(changedCount, `${upper ? 'upper' : 'lower'} ${record.type === 1 ? 'floor' : 'ceiling'} native lighting`).toBeGreaterThan(0);
  }
  await page.getByRole('combobox', { name: 'Corner lighting surfaces', exact: true }).selectOption('all');
  await page.getByRole('combobox', { name: 'Copy original corner lighting', exact: true }).selectOption('0');
  await page.getByRole('button', { name: 'Apply corner lighting', exact: true }).click(); await expect(preview.getByRole('status')).toContainText('zones loaded');
  await page.getByRole('spinbutton', { name: 'Lower corner brightness', exact: true }).fill('18');
  await page.getByRole('combobox', { name: 'Lower corner animation', exact: true }).selectOption('1');
  await page.getByRole('combobox', { name: 'Lower corner animation strength', exact: true }).selectOption('8');
  await page.getByRole('spinbutton', { name: 'Upper corner brightness', exact: true }).fill('6');
  await page.getByRole('combobox', { name: 'Upper corner animation', exact: true }).selectOption('3');
  await page.getByRole('combobox', { name: 'Upper corner animation strength', exact: true }).selectOption('16');
  await page.getByRole('button', { name: 'Apply corner lighting', exact: true }).click(); await expect(preview.getByRole('status')).toContainText('zones loaded');
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('four-corner-channels.ab3d'); await (await pending).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path))); expect(saved.files).toEqual(document.files);
  const lights = saved.roomSource.rooms.find(item => item.id === room.id)!.cornerLights!;
  expect(lights).toHaveLength(5); expect(lights.find(item => item.record === undefined)).toMatchObject({ lowerBrightness: 0x7112, upperBrightness: -3322 });
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(preview.getByRole('status')).toContainText('zones loaded');
  await page.getByRole('button', { name: 'Redo', exact: true }).click(); await expect(preview.getByRole('status')).toContainText('zones loaded');
  await expect(preview.getByRole('alert')).toHaveCount(0);
  await page.screenshot({ path: info.outputPath('corner-lighting-inspector.png') });
});

test('explicit corner lighting lets an inserted native corner bend and remains attached to its new position', async ({ page }, info) => {
  test.setTimeout(90000);
  const { document, copy: room } = nativeRoomCopyFixture(), a = room.vertices[0]!, b = room.vertices[1]!;
  const midpoint = { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 };
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'bent-corner-lighting.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Room ${room.id.replace(/^room-/, '')}(?:\\s|$)`) }).click();
  await page.locator('summary').filter({ hasText: /^Edit room corners$/ }).click();
  await page.getByRole('button', { name: 'Add corner after corner 1', exact: true }).click();
  await page.locator('summary').filter({ hasText: /^Corner lighting$/ }).click();
  await page.getByRole('combobox', { name: 'Lighting corner', exact: true }).selectOption('1');
  await page.getByRole('spinbutton', { name: 'Lower corner brightness', exact: true }).fill('0');
  await page.getByRole('spinbutton', { name: 'Upper corner brightness', exact: true }).fill('0');
  await page.getByRole('button', { name: 'Apply corner lighting', exact: true }).click();
  const field = page.getByRole('spinbutton', { name: 'Corner 2 X', exact: true });
  await field.fill(String(midpoint.x - 32)); await field.press('Enter');
  const pending = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = info.outputPath('bent-lit-corner.ab3d'); await (await pending).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path))), changed = saved.roomSource.rooms.find(item => item.id === room.id)!;
  expect(changed.cornerLights![0]).toEqual({ x: midpoint.x - 32, z: midpoint.z, lowerBrightness: 0, upperBrightness: 0 });
  const built = buildPlayableLevel(saved, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' }); await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  expect(await captureEngine(page, { zone: cell.id, x: Math.round(changed.vertices.reduce((sum, p) => sum + p.x, 0) / changed.vertices.length), z: Math.round(changed.vertices.reduce((sum, p) => sum + p.z, 0) / changed.vertices.length), upper: 0, angle: 0 })).toBeGreaterThan(0);
  await expect(preview.getByRole('alert')).toHaveCount(0);
});
