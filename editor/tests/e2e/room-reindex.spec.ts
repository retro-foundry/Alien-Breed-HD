import { expect, test } from '@playwright/test';
import { openLevel } from '../../src/model/level';
import { encodeProject } from '../../src/model/project';
import { originalFiles } from '../helpers';
import { captureEngine, type EngineCamera } from '../engineCapture';
import { reindexRooms } from '../reindexedRooms';

test('renumbered rooms retain engine spawn, exit setup and original views across all maps', async ({ page }, testInfo) => {
  test.setTimeout(360000); page.on('dialog', dialog => void dialog.accept());
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  const comparisons: object[] = [];
  for (let level = 1; level <= 16; level++) {
    const original = openLevel(`Level ${level}`, originalFiles(level), level - 1);
    if (level > 1) await page.getByRole('combobox', { name: 'Original level' }).selectOption(String(level));
    const preview = page.getByRole('region', { name: '3D game engine preview' });
    await expect(preview.getByRole('status')).toHaveText(`${original.zones.length} zones loaded · Game engine`, { timeout: 60000 });
    await expect(preview).toHaveAttribute('data-level-slot', String(level - 1));
    await expect(preview).toHaveAttribute('data-exit-zone', /^-?\d+$/);
    const exit = Number(await preview.getAttribute('data-exit-zone'));
    const ids = new Map(original.zones.map(zone => [zone.id, original.zones.length - 1 - zone.id]));
    const start = original.starts[0]!, scenes: EngineCamera[] = original.starts.map(start => ({ zone: start.zone, x: start.x, z: start.z, upper: 0, angle: 0 }));
    // Include the renderer's existing special-case viewpoint while changing
    // its numeric identity, so a passing spawn view cannot hide ID coupling.
    const inverseSpecial = [...ids].find(([, target]) => target === 53)?.[0];
    for (const zone of new Set([53, inverseSpecial])) {
      const special = original.zones.find(room => room.id === zone);
      if (!special) continue;
      const lines = special.boundaryIds.map(id => original.lines[id]!);
      for (const angle of [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168]) scenes.push({ zone: special.id,
        x: Math.round(lines.reduce((sum, line) => sum + line.x, 0) / lines.length),
        z: Math.round(lines.reduce((sum, line) => sum + line.z, 0) / lines.length), upper: 0, angle });
    }
    const before: number[] = []; for (const scene of scenes) before.push(await captureEngine(page, scene));
    const rebuilt = reindexRooms(original, ids, exit);
    await page.locator('input[type=file]').setInputFiles({ name: `renumbered-${level}.ab3d`, mimeType: 'application/zip', buffer: Buffer.from(encodeProject(rebuilt)) });
    await expect(page.locator('.project-picker strong')).toHaveText(rebuilt.name);
    await expect(preview.getByRole('status')).toHaveText(`${original.zones.length} zones loaded · Game engine`, { timeout: 60000 });
    await expect(preview).toHaveAttribute('data-level-slot', String(level - 1));
    await expect(preview).toHaveAttribute('data-spawn-zone', String(ids.get(start.zone)));
    await expect(preview).toHaveAttribute('data-exit-zone', String(rebuilt.exitZone));
    for (let i = 0; i < scenes.length; i++) {
      const scene = scenes[i]!, after = await captureEngine(page, { ...scene, zone: ids.get(scene.zone)! });
      comparisons.push({ level, scene, before: before[i], after });
      if (after !== before[i]) await page.screenshot({ path: testInfo.outputPath(`renumbered-${level}-${i}.png`), fullPage: true });
      expect(after, `Level ${level}, original zone ${scene.zone}, angle ${scene.angle}`).toBe(before[i]);
    }
  }
  await testInfo.attach('renumbered-frame-comparisons', { body: JSON.stringify(comparisons, null, 2), contentType: 'application/json' });
});
