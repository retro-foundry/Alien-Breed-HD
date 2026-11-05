import { expect, test, type Locator, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture, twoStoreyHeightTemplate } from '../roomHeightFixture';
import { decodeProject, encodeProject, openArchive } from '../../src/model/project';
import { openLevel } from '../../src/model/level';
import { enginePlayerState } from '../engineCapture';

async function coordinate(diagram: Locator, x: number, z: number) {
  await diagram.scrollIntoViewIfNeeded();
  return diagram.evaluate((node, { x, z }) => {
    const svg = node as SVGSVGElement, point = svg.createSVGPoint(); point.x = x; point.y = z;
    const screen = point.matrixTransform(svg.getScreenCTM()!); return { x: screen.x, y: screen.y };
  }, { x, z });
}
async function place(page: Page, diagram: Locator, x: number, z: number) {
  await page.getByRole('button', { name: 'Place control point in room', exact: true }).click();
  await diagram.scrollIntoViewIfNeeded(); const screen = await coordinate(diagram, x, z); await page.mouse.click(screen.x, screen.y);
  await precisePosition(page, x, z);
}
async function precisePosition(page: Page, x: number, z: number) {
  // A screen pixel can cover multiple world units. Use the numerical controls
  // for exact placement after verifying the graphical gesture created a point.
  const panel = page.locator('details').filter({ has: page.getByText('Control points and links', { exact: true }) });
  await panel.getByRole('spinbutton', { name: 'Position X', exact: true }).fill(String(x));
  await panel.getByRole('spinbutton', { name: 'Position Z', exact: true }).fill(String(z));
  const apply = panel.getByRole('button', { name: 'Apply position', exact: true }); if (await apply.isEnabled()) await apply.click();
}

test('authors lower/upper controls, graphical moves, assignments and links with portable native exports', async ({ page }, info) => {
  test.setTimeout(120000);
  const fixture = roomHeightFixture(twoStoreyHeightTemplate()), first = fixture.document.controlPoints.length;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'controls.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  await page.getByText('Control points and links', { exact: true }).click();
  const diagram = page.getByRole('img', { name: 'Control-point room diagram' }), storey = page.getByRole('combobox', { name: 'Control-point storey' });
  await page.getByRole('button', { name: 'Place control point in room', exact: true }).click(); await diagram.focus(); await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await place(page, diagram, 20064, 20064);
  await page.getByRole('combobox', { name: 'Lower room control point' }).selectOption(String(first));
  const screen = await coordinate(diagram, 20064, 20064), moved = await coordinate(diagram, 20096, 20096);
  await page.mouse.move(screen.x, screen.y); await page.mouse.down(); await page.mouse.move(moved.x, moved.y); await page.keyboard.press('Escape'); await page.mouse.up();
  await expect(diagram.locator(`[data-control-id="${first}"]`)).toHaveAttribute('cx', '20064');
  await page.mouse.move(screen.x, screen.y); await page.mouse.down(); await page.mouse.move(moved.x, moved.y); await page.mouse.up();
  await expect.poll(async () => Number(await diagram.locator(`[data-control-id="${first}"]`).getAttribute('cx'))).toBeGreaterThan(20064);
  const movedX = await diagram.locator(`[data-control-id="${first}"]`).getAttribute('cx');
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(diagram.locator(`[data-control-id="${first}"]`)).toHaveAttribute('cx', '20064');
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await expect(diagram.locator(`[data-control-id="${first}"]`)).toHaveAttribute('cx', movedX!);
  await precisePosition(page, 20096, 20096);
  await storey.selectOption('upper'); await place(page, diagram, 20192, 20192);
  await page.getByRole('combobox', { name: 'Upper room control point' }).selectOption(String(first + 1));
  await expect(page.getByText('Runtime level files do not contain the editable link graph.', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: 'Import twolev.links', exact: true }).click();
  await page.getByLabel('Import control-point links', { exact: true }).setInputFiles({ name: 'broken.links', mimeType: 'application/octet-stream', buffer: Buffer.from([0]) });
  await expect(page.getByRole('alert')).toContainText('10000 bytes');
  await page.getByRole('button', { name: 'Cancel link graph import', exact: true }).click();
  await page.getByRole('button', { name: 'Create new empty link graph', exact: true }).click();
  await page.getByRole('combobox', { name: 'Control-point link partner' }).selectOption(String(first));
  await page.getByRole('combobox', { name: 'Control-point link direction' }).selectOption('2/1');
  await page.getByRole('button', { name: 'Rebuild visual links automatically' }).click();
  await expect(page.getByRole('combobox', { name: 'Control-point link direction' })).toHaveValue('2/1');
  await page.getByRole('button', { name: 'Assign nearest visible lower point', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Lower room control point' })).toHaveValue(String(first));
  const graphDownload = page.waitForEvent('download'); await page.getByRole('button', { name: 'Download twolev.links' }).click();
  const graphPath = info.outputPath('twolev.links'); await (await graphDownload).saveAs(graphPath);
  const graph = await readFile(graphPath); expect(graph.length).toBe(10000); expect(graph[first * 100 + first + 1]).toBe(1); expect(graph[(first + 1) * 100 + first]).toBe(2);
  const saving = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const savedPath = info.outputPath('controls.ab3d'); await (await saving).saveAs(savedPath);
  const saved = decodeProject(new Uint8Array(await readFile(savedPath)));
  expect(saved.roomSource.navigation!.points.slice(-2)).toMatchObject([{ id: first, x: 20096, z: 20096, upper: false }, { id: first + 1, x: 20192, z: 20192, upper: true }]);
  await page.getByRole('checkbox', { name: 'Show control points and links' }).check();
  await page.getByRole('combobox', { name: 'Control-point link overlay' }).selectOption('physical');
  await page.locator('canvas[aria-label="Level map"]').focus(); await page.keyboard.press('Shift+F');
  await page.screenshot({ path: info.outputPath('control-point-authoring.png') });
  const exporting = page.waitForEvent('download'); await page.getByRole('button', { name: 'Export level', exact: true }).click();
  const gamePath = info.outputPath('controls-game.zip'); await (await exporting).saveAs(gamePath);
  const candidate = openArchive(new Uint8Array(await readFile(gamePath)), 'controls.zip')[0]!, native = openLevel(candidate.name, candidate.files);
  expect(native.controlPoints.slice(-2)).toMatchObject([{ x: 20096, z: 20096, upper: 0 }, { x: 20192, z: 20192, upper: 1 }]);
  expect(native.zones.find(zone => zone.control === first && zone.upperControl === first + 1)).toBeDefined();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect.poll(async () => Number(await preview.getAttribute('data-rendered-pixels')), { timeout: 60000 }).toBeGreaterThan(0);
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'); await canvas.focus();
  const before = await enginePlayerState(page); await page.keyboard.down('w');
  try { await expect.poll(async () => { const p = await enginePlayerState(page); return Math.hypot(p.x - before.x, p.z - before.z); }).toBeGreaterThan(4); }
  finally { await page.keyboard.up('w'); }
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await page.locator('input[type=file]').setInputFiles(savedPath);
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  const details = page.locator('details').filter({ has: page.getByText('Control points and links', { exact: true }) });
  if (await details.getAttribute('open') === null) await page.getByText('Control points and links', { exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Lower room control point' })).toHaveValue(String(first));
  await expect(page.getByRole('combobox', { name: 'Upper room control point' })).toHaveValue(String(first + 1));
  await page.getByRole('combobox', { name: 'Selected control point' }).selectOption(String(first));
  await page.getByRole('button', { name: 'Delete control point and links', exact: true }).click();
  await expect(page.getByRole('combobox', { name: 'Lower room control point' })).toHaveValue('');
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(page.getByRole('combobox', { name: 'Lower room control point' })).toHaveValue(String(first));
});
