import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { decodeProject, encodeProject, openArchive, openProjectCandidate } from '../../src/model/project';
import { redrawnRoomFixture } from '../portalFixture';
import { captureEngine, enginePlayerState } from '../engineCapture';

async function save(page: Page, path: string) {
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  await (await download).saveAs(path); return decodeProject(new Uint8Array(await readFile(path)));
}
async function clickPoint(page: Page, point: { x: number; z: number }) {
  const canvas = page.locator('canvas[aria-label="Level map"]'), box = (await canvas.boundingBox())!;
  const view = await canvas.evaluate(node => ({ x: Number(node.dataset.worldX), z: Number(node.dataset.worldZ), scale: Number(node.dataset.worldScale) }));
  await page.mouse.click(box.x + box.width / 2 + (point.x - view.x) * view.scale, box.y + box.height / 2 - (point.z - view.z) * view.scale);
}

test('redrawing an imported room cancels safely, commits one stable outline, and builds new walls with selected lighting', async ({ page }, testInfo) => {
  test.setTimeout(90000); page.on('dialog', dialog => void dialog.accept());
  const fixture = redrawnRoomFixture(), zone = fixture.room.originZone!;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'redraw.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.baseline)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.baseline.name);
  const selectRoom = () => page.getByRole('button', { name: new RegExp(`^Zone ${String(zone).padStart(3, '0')}(?:\\s|$)`) }).click();
  // The fixture lies on a 16-unit grid. Preserve those exact shared edges
  // across browsers that round fractional CSS-pixel clicks differently.
  await selectRoom(); await page.getByRole('combobox', { name: 'Grid spacing' }).selectOption('16');
  const canvas = page.locator('canvas[aria-label="Level map"]'), box = (await canvas.boundingBox())!;
  await expect(canvas).toHaveAttribute('data-world-x', String(fixture.center.x));
  const scale = Number(await canvas.getAttribute('data-world-scale'));
  const outline = fixture.room.vertices;
  const width = Math.max(...outline.map(p => p.x)) - Math.min(...outline.map(p => p.x));
  const depth = Math.max(...outline.map(p => p.z)) - Math.min(...outline.map(p => p.z));
  // Fit the complete drawing into clear canvas space beside the map guide.
  // A fixed 3x zoom put a corner outside a 1280px browser window.
  const drawingScale = Math.min(3, (box.width - 400) / width, (box.height - 80) / depth);
  await page.mouse.move(Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2));
  await page.mouse.wheel(0, -Math.log(drawingScale / scale) / 0.0015);
  await expect.poll(async () => Number(await canvas.getAttribute('data-world-scale'))).toBeCloseTo(drawingScale, 3);
  const clearGuide = async () => {
    await page.mouse.move(Math.round(box.x + box.width / 2), Math.round(box.y + box.height / 2));
    await page.mouse.down({ button: 'middle' });
    await page.mouse.move(Math.round(box.x + box.width / 2) - 160, Math.round(box.y + box.height / 2), { steps: 4 });
    await page.mouse.up({ button: 'middle' });
  };
  await clearGuide();
  await page.getByRole('button', { name: 'Redraw outline', exact: true }).click();
  for (const p of fixture.vertices.slice(0, 2)) await clickPoint(page, p);
  expect((await save(page, testInfo.outputPath('unfinished.ab3d'))).roomSource).toEqual(fixture.baseline.roomSource);
  await page.keyboard.press('Escape'); await selectRoom(); await clearGuide();
  await page.getByRole('button', { name: 'Redraw outline', exact: true }).click();
  for (const p of fixture.vertices.slice(0, 2)) await clickPoint(page, p);
  await page.getByRole('button', { name: 'Pan tool', exact: true }).click();
  expect((await save(page, testInfo.outputPath('cancelled.ab3d'))).roomSource).toEqual(fixture.baseline.roomSource);
  await page.getByRole('button', { name: 'Redraw outline', exact: true }).click();
  for (const [index, p] of fixture.vertices.toReversed().entries()) {
    await clickPoint(page, p);
    await expect(page.locator('.selection-description')).toContainText(`${index + 1} ${index ? 'corners' : 'corner'}`);
  }
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', { name: 'Finish room', exact: true })).toHaveCount(0);
  const drawn = await save(page, testInfo.outputPath('redrawn.ab3d'));
  expect(drawn.roomSource).toEqual(fixture.document.roomSource); expect(drawn.files).toEqual(fixture.baseline.files);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  expect((await save(page, testInfo.outputPath('undo.ab3d'))).roomSource).toEqual(fixture.baseline.roomSource);
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('alert')).toContainText('new boundary walls needing a wall style', { timeout: 60000 });
  await page.getByRole('combobox', { name: 'Lower wall style' }).selectOption(String(fixture.appearance.lowerWall));
  await page.getByRole('combobox', { name: 'Room corner lighting' }).selectOption(String(fixture.appearance.lightingPoint));
  await page.getByRole('button', { name: 'Apply appearance', exact: true }).click();
  await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const path = testInfo.outputPath('redrawn-game.zip'); await (await download).saveAs(path);
  const compiled = openProjectCandidate(openArchive(new Uint8Array(await readFile(path)), 'redrawn-game.zip')[0]!);
  expect(compiled.zones.find(item => item.id === compiled.starts[0]!.zone)!.boundaryIds).toHaveLength(5);
  const angle = Math.round(Math.atan2(fixture.corner.x - fixture.center.x, fixture.corner.z - fixture.center.z) * 8192 / (Math.PI * 2)) & 8191;
  await captureEngine(page, { ...fixture.center, zone: compiled.starts[0]!.zone, upper: 0, angle });
  await page.mouse.move(0, 0);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot({ path: testInfo.outputPath('redrawn-wall.png') });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  expect(await enginePlayerState(page)).toEqual({ ...fixture.center, zone: compiled.starts[0]!.zone, playing: true });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('alert')).toHaveCount(0);
});
