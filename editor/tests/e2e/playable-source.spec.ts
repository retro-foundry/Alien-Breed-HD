import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { applyCommand, editRoomAppearance, editRoomSource, setSourceObject, setSourceStart, setSourceTeleport } from '../../src/model/commands';
import { openLevel } from '../../src/model/level';
import { decodeProject, encodeProject, openArchive, openProjectCandidate } from '../../src/model/project';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../../src/model/roomSource';
import { enginePlayerState } from '../engineCapture';
import { originalFiles } from '../helpers';
import { roomAppearanceChoices } from '../../src/model/roomAppearance';

function fixture() {
  const original = openLevel('Interactive source build', originalFiles(1));
  let document = original;
  for (const x of [20000, 21000]) document = applyCommand(document, editRoomSource(commitRoomDraft(document.roomSource,
    setDraftCorners(beginRoomDraft(document, document.roomSource, 0), [{ x, z: 20000 }, { x: x + 128, z: 20000 }, { x: x + 128, z: 20128 }, { x, z: 20128 }]))));
  const from = document.roomSource.rooms.at(-2)!, to = document.roomSource.rooms.at(-1)!;
  document = applyCommand(document, setSourceStart(document, 1, from.id, 20064, 20064));
  document = applyCommand(document, setSourceTeleport(document, from.id, to.id, 21064, 20064));
  const object = original.objects.find(object => object.type === 1 && object.zone >= 0 && !object.upper)!;
  document = applyCommand(document, setSourceObject(document, object.id, to.id, 21100, 20096));
  // Deliberately change runtime numbering: the UI must map compiled IDs back
  // to authored rooms for visits/picking instead of selecting unrelated zones.
  document = { ...document, roomSource: { ...document.roomSource, rooms: document.roomSource.rooms.toReversed() } };
  return { original, document, from, to, object };
}

test('source appearance, export, viewport visits/picking and one-click play use the same compiled level', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const { original, document, from, to, object } = fixture();
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'source.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  for (const room of [from, to]) {
    await page.getByRole('button', { name: new RegExp(`^Room ${room.id.replace(/^room-/, '')}`) }).click();
    await page.getByRole('combobox', { name: 'Lower wall style' }).selectOption({ index: 1 });
    await page.getByRole('combobox', { name: 'Room corner lighting' }).selectOption({ index: 1 });
    await page.getByRole('button', { name: 'Apply appearance', exact: true }).click();
  }
  const gameDownload = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const gamePath = testInfo.outputPath('compiled-game.zip'); await (await gameDownload).saveAs(gamePath);
  const compiled = openProjectCandidate(openArchive(new Uint8Array(await readFile(gamePath)), 'compiled-game.zip')[0]!);
  const spawn = compiled.starts[0]!, arrival = compiled.zones.find(zone => zone.id === spawn.zone)!.teleport;
  expect(spawn).toMatchObject({ x: 20064, z: 20064 }); expect(arrival).toMatchObject({ x: 21064, z: 20064 });
  expect(compiled.objects.find(item => item.id === object.id)).toMatchObject({ zone: arrival.zone, x: 21100, z: 20096 });
  expect(spawn.zone).not.toBe(Number(from.id.replace(/^room-/, '')));
  expect(arrival.zone).not.toBe(Number(to.id.replace(/^room-/, '')));
  expect(compiled.exitZone).not.toBeNull();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toHaveText(`${compiled.zones.length} zones loaded · Game engine`, { timeout: 60000 });
  await expect(preview).toHaveAttribute('data-exit-zone', String(compiled.exitZone));
  // The most recently selected authored room is the arrival room. Native
  // inspection visits its generated cell, including the reversed numbering.
  await expect.poll(async () => (await enginePlayerState(page)).zone).toBe(arrival.zone);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').click();
  await expect(page.getByRole('heading', { name: `Room ${to.id.replace(/^room-/, '')}`, exact: true })).toBeVisible();
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await expect.poll(async () => (await enginePlayerState(page)).zone).toBe(arrival.zone);
  expect(await enginePlayerState(page)).toEqual({ zone: arrival.zone, x: 21064, z: 20064, playing: true });
  await page.screenshot({ path: testInfo.outputPath('source-build-play.png'), fullPage: true });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  const sourceDownload = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const sourcePath = testInfo.outputPath('editable-source.ab3d'); await (await sourceDownload).saveAs(sourcePath);
  const saved = decodeProject(new Uint8Array(await readFile(sourcePath)));
  expect(saved.files).toEqual(original.files);
  expect(saved.roomSource).toEqual(document.roomSource);
  expect(saved.roomAppearances).toHaveLength(2);
  expect(saved.sourceStarts).toEqual(document.sourceStarts); expect(saved.sourceObjects).toEqual(document.sourceObjects);
  expect(saved.sourceTeleports).toEqual(document.sourceTeleports);
  await expect(preview.getByRole('alert')).toHaveCount(0);
});

test('opening a newer project during an export build cannot download stale source files', async ({ page }) => {
  page.on('dialog', dialog => void dialog.accept());
  const fixtureData = fixture(); let source = fixtureData.document;
  for (const room of [fixtureData.from, fixtureData.to]) {
    const choices = roomAppearanceChoices(source, room);
    source = applyCommand(source, editRoomAppearance(source, {
      room: room.id, lowerWall: choices.lower[0]!.offset, upperWall: null, lightingPoint: choices.points[0]!,
    }));
  }
  let release!: () => void, requested!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; }), started = new Promise<void>(resolve => { requested = resolve; });
  await page.route('**/reference/game-config.json', async route => {
    const response = await route.fetch(); requested(); await gate; await route.fulfill({ response });
  });
  const downloads: string[] = []; page.on('download', item => downloads.push(item.suggestedFilename()));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'source.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(source)) });
  await expect(page.locator('.project-picker strong')).toHaveText(source.name);
  await page.getByRole('button', { name: 'Export level', exact: true }).click(); await started;
  const next = openLevel('Newer project', originalFiles(2), 1);
  await page.locator('input[type=file]').setInputFiles({ name: 'newer.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(next)) });
  await expect(page.locator('.project-picker strong')).toHaveText(next.name);
  release();
  await expect(page.getByText('The level changed during the build. Export again to use the latest edits.', { exact: true })).toBeVisible();
  expect(downloads).toEqual([]);
  await expect(page.locator('.project-picker strong')).toHaveText(next.name);
});
