import { expect, test } from '@playwright/test';
import { openLevel } from '../../src/model/level';
import { encodeProject } from '../../src/model/project';
import { buildSourceLevel } from '../../src/model/sourceBuild';
import { cellContainsPoint } from '../../src/model/roomReferences';
import { originalFiles } from '../helpers';
import { captureEngine, type EngineCamera } from '../engineCapture';
import { writeFile } from 'node:fs/promises';
import { buildVisibilityFiles } from '../../src/model/visibilityBuild';

for (let level = 1; level <= 16; level++) test(`complete source build ${level} retains original starts and matches visibility-only reference views`, async ({ page }, testInfo) => {
  test.setTimeout(120000); page.on('dialog', dialog => void dialog.accept());
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const comparisons: object[] = [];
  const original = openLevel(`Level ${level}`, originalFiles(level), level - 1);
  if (level > 1) await page.getByRole('combobox', { name: 'Original level' }).selectOption(String(level));
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toHaveText(`${original.zones.length} zones loaded · Game engine`, { timeout: 60000 });
  await expect(preview).toHaveAttribute('data-exit-zone', /^-?\d+$/);
  const built = buildSourceLevel(original, { resolvedExit: Number(await preview.getAttribute('data-exit-zone')) });
  const scenes: EngineCamera[] = original.starts.map(start => ({ zone: start.zone, x: start.x, z: start.z, upper: 0, angle: 0 }));
  for (const room of original.roomSource.rooms) {
    const cells = built.topology.cells.filter(cell => cell.sourceId === room.id);
    if (cells.length < 2) continue;
    for (const cell of cells) {
      const vertices = cell.vertices.map(id => built.topology.points[id]!);
      const x = Math.round(vertices.reduce((sum, point) => sum + point.x, 0) / vertices.length);
      const z = Math.round(vertices.reduce((sum, point) => sum + point.z, 0) / vertices.length);
      for (const angle of [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168]) scenes.push({ zone: room.originZone!, x, z, upper: 0, angle });
    }
  }
  const before: number[] = [], beforeImages: Buffer[] = []; for (let index = 0; index < scenes.length; index++) {
    const scene = scenes[index]!; before.push(await captureEngine(page, scene));
    beforeImages.push(await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot());
  }
  const rawBefore = [...before];
  if (scenes.length > original.starts.length) {
    // The shipped Level 13 tables contain a visible white gap. Compare
    // decomposition to an independent visibility-only build that retains
    // original zone geometry, graphics and object ownership. Original spawn
    // views below still compare directly with the shipped files.
    const reference = openLevel(`Visibility reference ${level}`, buildVisibilityFiles(original), level - 1);
    await page.locator('input[type=file]').setInputFiles({ name: 'visibility-reference.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(reference)) });
    await expect(page.locator('.project-picker strong')).toHaveText(reference.name);
    await expect(preview.getByRole('status')).toHaveText(`${reference.zones.length} zones loaded · Game engine`, { timeout: 60000 });
    for (let index = original.starts.length; index < scenes.length; index++) {
      before[index] = await captureEngine(page, scenes[index]!);
      beforeImages[index] = await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot();
    }
  }
  await page.locator('input[type=file]').setInputFiles({ name: `compiled-${level}.ab3d`, mimeType: 'application/zip', buffer: Buffer.from(encodeProject(built.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(built.document.name);
  await expect(preview.getByRole('status')).toHaveText(`${built.document.zones.length} zones loaded · Game engine`, { timeout: 60000 });
  await expect(preview).toHaveAttribute('data-level-slot', String(level - 1));
  await expect(preview).toHaveAttribute('data-spawn-zone', String(built.document.starts[0]!.zone));
  await expect(preview).toHaveAttribute('data-exit-zone', String(built.document.exitZone));
  for (let index = 0; index < scenes.length; index++) {
    const scene = scenes[index]!, cell = built.topology.cells.find(cell => cell.originZone === scene.zone && cellContainsPoint(built.topology, cell, scene));
    expect(cell, `Camera has a generated cell in level ${level}, source room ${scene.zone}`).toBeDefined();
    const after = await captureEngine(page, { ...scene, zone: cell!.id });
    comparisons.push({ level, scene, runtimeRoom: cell!.id, rawBefore: rawBefore[index],
      reference: index < original.starts.length ? 'shipped' : 'visibility-only', before: before[index], after });
    if (after !== before[index]) {
      await writeFile(testInfo.outputPath(`original-${level}-${index}.png`), beforeImages[index]!);
      await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot({ path: testInfo.outputPath(`compiled-${level}-${index}.png`) });
    }
    expect(after, `Level ${level}, source room ${scene.zone}, angle ${scene.angle}`).toBe(before[index]);
  }
  await testInfo.attach('source-build-frame-comparisons', { body: JSON.stringify(comparisons, null, 2), contentType: 'application/json' });
});
