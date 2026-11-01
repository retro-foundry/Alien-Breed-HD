import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture } from '../roomHeightFixture';
import { createObject, objectTemplates } from '../../src/model/objectAuthoring';
import { applyCommand } from '../../src/model/commands';
import { encodeProject, decodeProject } from '../../src/model/project';
import { engineObjectState } from '../engineCapture';

test('group duplication copies a batch with independent positions, one undo and guarded keyboard repeat', async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const fixture = roomHeightFixture(); let document = fixture.document;
  const template = objectTemplates(document).find(object => object.type === 0)!.id, ids: number[] = [];
  for (const [x, z, y] of [[20128, 20128, -7.5], [20240, 20192, -10]]) {
    const proposal = createObject(document, { template, room: fixture.from.id, x: x!, z: z!, y: y! });
    document = applyCommand(document, proposal.command); ids.push(proposal.id);
  }
  const firstCopy = document.objects.length;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'group-copy.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('tab', { name: 'Objects', exact: true }).click();
  await page.getByRole('button', { name: `Alien ${ids[0]}`, exact: true }).click();
  await page.getByRole('button', { name: `Alien ${ids[1]}`, exact: true }).click({ modifiers: ['Shift'] });
  const inspector = page.getByRole('complementary', { name: 'Object group inspector' });
  await inspector.getByRole('spinbutton', { name: 'Group X offset' }).fill('32');
  await inspector.getByRole('button', { name: 'Duplicate selected objects', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('No objects were duplicated');
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
  await inspector.getByRole('spinbutton', { name: 'Group X offset' }).fill('-8.5');
  await inspector.getByRole('spinbutton', { name: 'Group Z offset' }).fill('16.25');
  await inspector.getByRole('button', { name: 'Duplicate selected objects', exact: true }).click();
  await expect(page.getByRole('button', { name: `Alien ${firstCopy}`, exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(page.getByRole('button', { name: `Alien ${firstCopy}`, exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  const map = page.locator('canvas[aria-label="Level map"]');
  await map.focus(); await page.keyboard.press('Control+d');
  await map.dispatchEvent('keydown', { key: 'd', ctrlKey: true, repeat: true });
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('group-copies.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.objects).toHaveLength(firstCopy + 4);
  expect(new Set([...ids, firstCopy, firstCopy + 1, firstCopy + 2, firstCopy + 3].map(id => saved.objects[id]!.cid)).size).toBe(6);
  for (let copy = firstCopy; copy < firstCopy + 4; copy++) {
    const original = document.objects[ids[(copy - firstCopy) % 2]!]!;
    expect(saved.sourceObjects!.find(object => object.id === copy)).toMatchObject({ room: fixture.from.id, x: original.x - 8.5, z: original.z + 16.25 });
    expect(saved.objects[copy]!.y).toBe(original.y);
  }
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  for (let copy = firstCopy; copy < firstCopy + 4; copy++) {
    const original = document.objects[ids[(copy - firstCopy) % 2]!]!;
    expect(await engineObjectState(page, copy)).toMatchObject({ x: original.x - 8.5, z: original.z + 16.25, yFixed: original.y * 256, health: original.health });
  }
});
