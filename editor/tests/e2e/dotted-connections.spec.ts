import { expect, test } from '@playwright/test';
import { roomHeightFixture } from '../roomHeightFixture';
import { translateSourceRooms } from '../../src/model/roomSource';
import { encodeProject } from '../../src/model/project';

test('explains dotted connections and draws a selected newly authored shared edge with visible gaps', async ({ page }, info) => {
  const fixture = roomHeightFixture(), document = { ...fixture.document,
    roomSource: translateSourceRooms(fixture.document.roomSource, [fixture.to.id], -744, 0).after };
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'adjoining-rooms.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.from.id.replace(/^room-/, '')}`) }).click();
  const map = page.locator('canvas[aria-label="Level map"]'); await map.focus(); await page.keyboard.press('Shift+F');
  const legend = page.getByRole('note', { name: 'Map line legend' });
  await expect(legend).toContainText('Dotted: zone connection');
  await legend.getByText('How to make dotted lines', { exact: true }).click();
  await expect(legend).toContainText('There is no separate dotted-line zone type');
  await expect(legend).toContainText('Wall connections (solid / dotted)');
  await expect(legend).toContainText('choose Open edge');
  await expect(legend).toContainText('Put one edge exactly on the other room');
  await expect(legend.getByRole('img')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await page.screenshot({ path: info.outputPath('dotted-zone-instructions.png') });
  await legend.getByText('How to make dotted lines', { exact: true }).click();
  const gaps = await map.evaluate(node => {
    const canvas = node as HTMLCanvasElement, context = canvas.getContext('2d')!, ratio = canvas.width / canvas.clientWidth;
    const scale = Number(node.dataset.worldScale), x = Number(node.dataset.worldX), z = Number(node.dataset.worldZ);
    const px = Math.round((canvas.clientWidth / 2 + (20256 - x) * scale) * ratio);
    const top = Math.round((canvas.clientHeight / 2 - (20224 - z) * scale) * ratio);
    const bottom = Math.round((canvas.clientHeight / 2 - (20032 - z) * scale) * ratio);
    let ink = 0, gap = 0;
    for (let y = Math.max(0, top); y < Math.min(canvas.height, bottom); y++) {
      const pixels = context.getImageData(px - 2, y, 5, 1).data;
      const green = Array.from({ length: 5 }, (_, i) => pixels[i * 4]! > 150 && pixels[i * 4 + 1]! > 200 && pixels[i * 4 + 2]! < 220).some(Boolean);
      if (green) ink++; else gap++;
    }
    return { ink, gap };
  });
  expect(gaps.ink).toBeGreaterThan(5); expect(gaps.gap).toBeGreaterThan(5);
  await page.getByRole('button', { name: 'Draw room', exact: true }).click();
  await expect(page.getByText(/Draw beside another room with an exactly shared edge/)).toBeVisible();
  await map.focus(); await page.keyboard.press('Escape');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
});
