import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture } from '../roomHeightFixture';
import { createObject, objectTemplates } from '../../src/model/objectAuthoring';
import { applyCommand } from '../../src/model/commands';
import { encodeProject, decodeProject } from '../../src/model/project';
import { engineObjectState } from '../engineCapture';

test('multi-object selection moves, aligns, rejects partial changes and deletes atomically in the native game', async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const fixture = roomHeightFixture(); let document = fixture.document;
  const template = objectTemplates(document).find(object => object.type === 0)!.id, ids: number[] = [];
  for (const [x, z] of [[20128, 20128], [20240, 20192]]) {
    const proposal = createObject(document, { template, room: fixture.from.id, x: x!, z: z!, y: -8 });
    document = applyCommand(document, proposal.command); ids.push(proposal.id);
  }
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'group.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('tab', { name: 'Objects', exact: true }).click();
  const first = page.getByRole('button', { name: `Alien ${ids[0]}`, exact: true }), second = page.getByRole('button', { name: `Alien ${ids[1]}`, exact: true });
  await first.click(); await second.click({ modifiers: ['Shift'] });
  const inspector = page.getByRole('complementary', { name: 'Object group inspector' });
  await expect(inspector.getByRole('heading', { name: '2 selected objects' })).toBeVisible();
  await expect(first).toHaveAttribute('aria-pressed', 'true'); await expect(second).toHaveAttribute('aria-pressed', 'true');
  await second.click({ modifiers: ['Control'] }); await expect(inspector).toHaveCount(0);
  const map = page.locator('canvas[aria-label="Level map"]'), box = (await map.boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2); await page.mouse.wheel(0, -3000);
  await expect.poll(async () => Number(await map.getAttribute('data-world-scale'))).toBeGreaterThan(1);
  const view = await map.evaluate(node => ({ x: Number((node as HTMLElement).dataset.worldX), z: Number((node as HTMLElement).dataset.worldZ), scale: Number((node as HTMLElement).dataset.worldScale) }));
  await map.click({ position: { x: box.width / 2 + (20240 - view.x) * view.scale, y: box.height / 2 - (20192 - view.z) * view.scale }, modifiers: ['Shift'] });
  await expect(inspector).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('object-group.png'), fullPage: true });
  await inspector.getByRole('spinbutton', { name: 'Group X offset' }).fill('32');
  await inspector.getByRole('button', { name: 'Move selected objects' }).click();
  await expect(page.getByRole('alert')).toContainText('outside the editable rooms');
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
  await inspector.getByRole('spinbutton', { name: 'Group X offset' }).fill('-8.5');
  await inspector.getByRole('spinbutton', { name: 'Group Z offset' }).fill('16.25');
  await inspector.getByRole('button', { name: 'Move selected objects' }).click();
  await inspector.getByRole('button', { name: 'Align X', exact: true }).click();
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('group.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(ids.map(id => saved.sourceObjects!.find(object => object.id === id))).toMatchObject([
    { x: 20119.5, z: 20144.25 }, { x: 20119.5, z: 20208.25 },
  ]);
  await inspector.getByRole('button', { name: 'Delete selected objects', exact: true }).click();
  await expect(first).toHaveCount(0); await expect(second).toHaveCount(0);
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(inspector).toBeVisible();
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await engineObjectState(page, ids[0]!)).toMatchObject({ x: 20119.5, z: 20144.25 });
  expect(await engineObjectState(page, ids[1]!)).toMatchObject({ x: 20119.5, z: 20208.25 });
  await first.click({ modifiers: ['Control'] });
  await expect(inspector).toHaveCount(0);
});
