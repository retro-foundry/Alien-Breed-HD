import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture } from '../roomHeightFixture';
import { encodeProject, decodeProject } from '../../src/model/project';
import { enginePlayerState } from '../engineCapture';

test('imported room groups toggle conventionally and Escape clears only selection', async ({ page }) => {
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  const first = page.getByRole('button', { name: /^Zone 000(?:\s|$)/ }), second = page.getByRole('button', { name: /^Zone 001(?:\s|$)/ });
  await first.click(); await second.click({ modifiers: ['Shift'] });
  const group = page.getByRole('complementary', { name: 'Room group inspector' });
  await expect(group.getByRole('heading', { name: '2 selected rooms' })).toBeVisible();
  await expect(first).toHaveAttribute('aria-pressed', 'true'); await expect(second).toHaveAttribute('aria-pressed', 'true');
  await first.click({ modifiers: ['Control'] }); await expect(group).toHaveCount(0);
  await expect(first).toHaveAttribute('aria-pressed', 'false'); await expect(second).toHaveAttribute('aria-pressed', 'true');
  await first.click({ modifiers: ['Shift'] }); await expect(group).toBeVisible();
  await page.keyboard.press('Escape'); await expect(group).toHaveCount(0);
  await expect(first).toHaveAttribute('aria-pressed', 'false'); await expect(second).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});

test('room groups frame, toggle on the map and delete as one source edit without discarding linked starts', async ({ page }, testInfo) => {
  const { document, from, to } = roomHeightFixture();
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'room-group.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  const first = page.getByRole('button', { name: new RegExp(`^Room ${from.id.replace(/^room-/, '')}(?:\\s|$)`) });
  const second = page.getByRole('button', { name: new RegExp(`^Room ${to.id.replace(/^room-/, '')}(?:\\s|$)`) });
  await first.click(); await second.click({ modifiers: ['Shift'] });
  const group = page.getByRole('complementary', { name: 'Room group inspector' });
  await group.getByRole('button', { name: 'Frame selected rooms', exact: true }).click();
  const map = page.locator('canvas[aria-label="Level map"]');
  await expect(map).toHaveAttribute('data-world-x', '20628');
  await expect(map).toHaveAttribute('data-world-z', '20128');
  const point = await map.evaluate(node => {
    const canvas = node as HTMLCanvasElement, rect = canvas.getBoundingClientRect(), scale = Number(canvas.dataset.worldScale);
    return { x: rect.x + rect.width / 2 + (20160 - Number(canvas.dataset.worldX)) * scale,
      y: rect.y + rect.height / 2 - (20160 - Number(canvas.dataset.worldZ)) * scale };
  });
  await page.keyboard.down('Shift'); await page.mouse.click(point.x, point.y); await page.keyboard.up('Shift');
  await expect(group).toHaveCount(0); await expect(first).toHaveAttribute('aria-pressed', 'false');
  await page.keyboard.down('Shift'); await page.mouse.click(point.x, point.y); await page.keyboard.up('Shift');
  await expect(group).toBeVisible();
  await page.keyboard.press('Delete'); await expect(first).toHaveCount(0); await expect(second).toHaveCount(0);
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath('deleted-rooms.ab3d'); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path)));
  expect(saved.roomSource.rooms).toHaveLength(document.roomSource.rooms.length - 2);
  expect(saved.sourceStarts).toEqual(document.sourceStarts); expect(saved.files.data).toEqual(document.files.data);
  await page.getByRole('button', { name: 'Undo', exact: true }).click(); await expect(first).toBeVisible(); await expect(second).toBeVisible();
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await enginePlayerState(page)).toMatchObject({ x: 20128, z: 20128, playing: false });
});
