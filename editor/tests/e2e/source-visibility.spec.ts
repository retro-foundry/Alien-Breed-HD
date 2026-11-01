import { expect, test } from '@playwright/test';
import { openLevel } from '../../src/model/level';
import { buildSourceLevel } from '../../src/model/sourceBuild';
import { buildVisibilityFiles } from '../../src/model/visibilityBuild';
import { encodeProject } from '../../src/model/project';
import { originalFiles } from '../helpers';
import { captureEngine } from '../engineCapture';

test('rebuilding Level 13 visibility repairs the shipped gap without changing geometry', async ({ page }) => {
  page.on('dialog', dialog => void dialog.accept()); await page.goto('/?view=top');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const original = openLevel('Original 13', originalFiles(13), 12, -1);
  const built = buildSourceLevel(original, { resolvedExit: -1 });
  const reference = openLevel('Visibility reference', buildVisibilityFiles(original), 12, -1);
  const cell = built.topology.cells.filter(cell => cell.originZone === 41).at(-1)!;
  const vertices = cell.vertices.map(id => built.topology.points[id]!);
  const camera = { x: Math.round(vertices.reduce((sum, point) => sum + point.x, 0) / vertices.length),
    z: Math.round(vertices.reduce((sum, point) => sum + point.z, 0) / vertices.length), angle: 7168, upper: 0 };
  const hashes: number[] = [];
  for (const document of [original, reference, built.document]) {
    await page.locator('input[type=file]').setInputFiles({ name: 'audit.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
    await expect(page.locator('.project-picker strong')).toHaveText(document.name);
    await expect(page.getByRole('region', { name: '3D game engine preview' }).getByRole('status')).toHaveText(`${document.zones.length} zones loaded · Game engine`, { timeout: 60000 });
    hashes.push(await captureEngine(page, { ...camera, zone: document === built.document ? cell.id : 41 }));
  }
  expect(hashes[0]).not.toBe(hashes[1]); expect(hashes[2]).toBe(hashes[1]);
});
