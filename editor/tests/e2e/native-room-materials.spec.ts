import { expect, test } from '@playwright/test';
import { nativeRoomCopyFixture } from '../nativeRoomCopyFixture';
import { encodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { captureEngine } from '../engineCapture';

test('a fresh native-material room identity loads ordered graphics and point lighting in the actual engine', async ({ page }, testInfo) => {
  const { document, copy } = nativeRoomCopyFixture();
  const built = buildPlayableLevel(document, -1), cell = built.topology!.cells.find(cell => cell.sourceId === copy.id)!;
  const centre = cell.vertices.reduce((sum, id) => ({ x: sum.x + built.topology!.points[id]!.x / cell.vertices.length,
    z: sum.z + built.topology!.points[id]!.z / cell.vertices.length }), { x: 0, z: 0 });
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'native-material-copy.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  const frames: number[] = [];
  for (const angle of [0, 2048, 4096, 6144]) {
    const camera = { zone: cell.id, x: Math.round(centre.x), z: Math.round(centre.z), upper: 0, angle };
    const frame = await captureEngine(page, camera);
    expect(await captureEngine(page, camera)).toBe(frame);
    frames.push(frame);
  }
  expect(new Set(frames).size).toBeGreaterThan(1);
  await page.screenshot({ path: testInfo.outputPath('native-material-copy.png'), fullPage: true });
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
