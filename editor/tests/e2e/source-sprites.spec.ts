import { expect, test } from '@playwright/test';
import { openLevel } from '../../src/model/level';
import { buildSourceLevel } from '../../src/model/sourceBuild';
import { encodeProject } from '../../src/model/project';
import { cellContainsPoint } from '../../src/model/roomReferences';
import { originalFiles } from '../helpers';
import { captureEngine } from '../engineCapture';

test('authored-room sprites survive collision decomposition with their original appearance', async ({ page }, testInfo) => {
  page.on('dialog', dialog => void dialog.accept()); await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  const results: object[] = [];
  for (const hidden of [false, true]) {
    const files = originalFiles(2), baseline = openLevel('source', files, 1);
    if (hidden) {
      const object = baseline.objects.find(object => object.id === 25)!;
      files.data = baseline.files.data.slice(); new DataView(files.data.buffer).setInt16(object.offset + 12, -1);
    }
    const original = openLevel(hidden ? 'Marine hidden' : 'Marine present', files, 1);
    const built = buildSourceLevel(original, { resolvedExit: -1 });
    const compiled = built.document;
    expect(compiled.drawGroups).toBeDefined();
    const preview = page.getByRole('region', { name: '3D game engine preview' });
    await page.locator('input[type=file]').setInputFiles({ name: 'original.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(original)) });
    await expect(page.locator('.project-picker strong')).toHaveText(original.name);
    await expect(preview.getByRole('status')).toHaveText(`${original.zones.length} zones loaded · Game engine`, { timeout: 60000 });
    const before: number[] = [];
    for (const angle of [1024, 2048, 3072, 6144]) before.push(await captureEngine(page, { zone: 67, x: -2448, z: 1067, angle, upper: 0 }));
    await page.locator('input[type=file]').setInputFiles({ name: 'compiled.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(compiled)) });
    await expect(page.locator('.project-picker strong')).toHaveText(compiled.name);
    await expect(preview.getByRole('status')).toHaveText(`${compiled.zones.length} zones loaded · Game engine`, { timeout: 60000 });
    const cell = built.topology.cells.find(cell => cell.originZone === 67 && cellContainsPoint(built.topology, cell, { x: -2448, z: 1067 }))!;
    for (const [index, angle] of [1024, 2048, 3072, 6144].entries()) {
      const after = await captureEngine(page, { zone: cell.id, x: -2448, z: 1067, angle, upper: 0 });
      results.push({ hidden, angle, before: before[index], after, matches: after === before[index] });
      expect(after, `Marine ${hidden ? 'hidden' : 'present'}, angle ${angle}`).toBe(before[index]);
    }
  }
  await testInfo.attach('actor-isolation', { body: JSON.stringify(results), contentType: 'application/json' });
});
