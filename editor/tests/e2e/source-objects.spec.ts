import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { applyCommand, editRoomSource, setSourceObject } from '../../src/model/commands';
import { openLevel } from '../../src/model/level';
import { decodeProject, encodeProject } from '../../src/model/project';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../../src/model/roomSource';
import { buildSourceLevel } from '../../src/model/sourceBuild';
import { originalFiles } from '../helpers';
import { captureEngine } from '../engineCapture';

function fixture() {
  const original = openLevel('Object authoring', originalFiles(1));
  const object = original.objects.find(object => object.type === 1 && object.zone >= 0 && !object.upper)!;
  const change = commitRoomDraft(original.roomSource, setDraftCorners(beginRoomDraft(original, original.roomSource, 0),
    [{ x: 20000, z: 20000 }, { x: 20256, z: 20000 }, { x: 20256, z: 20256 }, { x: 20000, z: 20256 }]));
  return { original, object, drawn: applyCommand(original, editRoomSource(change)), room: change.after.rooms.at(-1)! };
}

test('objects move into authored rooms, drag, delete, undo and recover through the browser', async ({ page }, testInfo) => {
  page.on('dialog', dialog => void dialog.accept());
  const { original, object, drawn, room } = fixture();
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'drawn.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(drawn)) });
  await expect(page.locator('.project-picker strong')).toHaveText(drawn.name);
  await page.getByRole('tab', { name: 'Objects', exact: true }).click();
  const row = page.getByRole('button', { name: `Health ${object.id}`, exact: true });
  await row.click();
  const roomField = page.getByRole('combobox', { name: 'Object room' });
  await roomField.selectOption(room.id);
  await page.getByRole('spinbutton', { name: 'Position X' }).fill('20128');
  await page.getByRole('spinbutton', { name: 'Position Z' }).fill('20128');
  await page.getByRole('button', { name: 'Apply position', exact: true }).click();
  await expect(page.locator('.dirty-dot')).toBeVisible();
  await expect(roomField).toHaveValue(room.id);
  // Selecting the row focuses its authored room; its map marker is hit-tested
  // at the authored position, not the imported object's old coordinates.
  await row.click();
  const canvas = page.locator('canvas[aria-label="Level map"]');
  await expect(canvas).toHaveAttribute('data-world-x', '20128');
  const box = (await canvas.boundingBox())!;
  // Integer wheel coordinates exercise the rounding used by Linux browsers.
  // At overview zoom, a fractional-pixel anchor can move the world centre.
  await page.mouse.move(Math.floor(box.x + box.width / 2), Math.floor(box.y + box.height / 2));
  await page.mouse.wheel(0, -6000);
  await expect(canvas).toHaveAttribute('data-world-scale', '8');
  await page.getByRole('combobox', { name: 'Grid spacing' }).selectOption('0');
  const currentBox = (await canvas.boundingBox())!;
  const view = await canvas.evaluate(node => ({ x: Number(node.dataset.worldX), z: Number(node.dataset.worldZ), scale: Number(node.dataset.worldScale) }));
  const marker = { x: currentBox.x + currentBox.width / 2 + (20128 - view.x) * view.scale,
    y: currentBox.y + currentBox.height / 2 - (20128 - view.z) * view.scale };
  await page.mouse.move(marker.x, marker.y);
  await page.mouse.down(); await page.mouse.move(marker.x + 16 * view.scale, marker.y, { steps: 5 }); await page.mouse.up();
  await expect(page.getByRole('spinbutton', { name: 'Position X' })).toHaveValue('20144');
  await page.getByRole('button', { name: 'Delete object', exact: true }).click();
  await expect(row).toHaveCount(0);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(row).toHaveCount(1);
  await expect(page.getByRole('spinbutton', { name: 'Position X' })).toHaveValue('20144');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('spinbutton', { name: 'Position X' })).toHaveValue('20128');
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await expect(page.getByRole('spinbutton', { name: 'Position X' })).toHaveValue('20144');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('object-source.ab3d'); await (await download).saveAs(path);
  const restored = decodeProject(new Uint8Array(await readFile(path)));
  expect(restored.sourceObjects).toEqual([{ id: object.id, room: room.id, x: 20144, z: 20128 }]);
  expect(restored.files).toEqual(original.files);
  await expect(page.getByText('Workspace saved', { exact: true })).toBeVisible();
  await page.reload(); await expect(page.locator('.project-picker strong')).toHaveText(drawn.name);
  await page.getByRole('tab', { name: 'Objects', exact: true }).click(); await row.click();
  await expect(roomField).toHaveValue(room.id);
  await expect(page.getByRole('spinbutton', { name: 'Position X' })).toHaveValue('20144');
  await page.screenshot({ path: testInfo.outputPath('object-source-editor.png'), fullPage: true });
});

test('a moved object renders in its authored room and disappears when deleted in the actual engine', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const { original, object, drawn, room } = fixture();
  const wall = original.zones[0]!.lower.find(record => record.wall)!.wall!;
  const options = { resolvedExit: -1, newRoomGraphics: new Map([[room.id, { walls: { lower: [wall] }, wallInsertion: { lower: 0 } }]]),
    newRoomLighting: new Map([[room.id, { kind: 'copy' as const, source: 0 }]]) };
  const presentSource = decodeProject(encodeProject(applyCommand(drawn, setSourceObject(drawn, object.id, room.id, 20128, 20128))));
  const hiddenSource = applyCommand(presentSource, setSourceObject(presentSource, object.id, null, 20128, 20128));
  const present = buildSourceLevel(presentSource, options), hidden = buildSourceLevel({ ...hiddenSource, name: 'Removed object' }, options);
  const zone = present.topology.sourceCells.get(room.id)![0]!;
  expect(present.document.objects[object.id]!.zone).toBe(zone);
  expect(hidden.document.objects[object.id]!.zone).toBe(-1);
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  const frames: number[][] = [];
  for (const [index, built] of [present, hidden].entries()) {
    const previousRevision = await preview.getAttribute('data-revision');
    await page.locator('input[type=file]').setInputFiles({ name: `compiled-${index}.ab3d`, mimeType: 'application/zip', buffer: Buffer.from(encodeProject(built.document)) });
    await expect(page.locator('.project-picker strong')).toHaveText(built.document.name);
    await expect(preview).not.toHaveAttribute('data-revision', previousRevision!);
    await expect(preview.getByRole('status')).toHaveText(`${built.document.zones.length} zones loaded · Game engine`, { timeout: 60000 });
    const hashes: number[] = [];
    for (const angle of [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168]) {
      hashes.push(await captureEngine(page, { zone, x: 20128, z: 20032, upper: 0, angle }));
      if (index === 0) await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas')
        .screenshot({ path: testInfo.outputPath(`moved-object-angle-${angle}.png`) });
    }
    frames.push(hashes); await expect(preview.getByRole('alert')).toHaveCount(0);
  }
  expect(frames[0]!.some((hash, index) => hash !== frames[1]![index])).toBe(true);
  await testInfo.attach('authored-object-rendering', { body: JSON.stringify({ room: room.id, zone, object: object.id, present: frames[0], removed: frames[1] }), contentType: 'application/json' });
});
