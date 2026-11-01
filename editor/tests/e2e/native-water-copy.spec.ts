import { expect, test } from '@playwright/test';
import { nativeWaterRoomCopyFixture } from '../nativeRoomCopyFixture';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { importedAnimationTables } from '../../src/model/animationBuild';
import { encodeProject } from '../../src/model/project';
import { engineWaterState } from '../engineCapture';

test('native material copies retain shared water motion in both the polygon and collision plane', async ({ page }) => {
  const { document, copy, zone, group } = nativeWaterRoomCopyFixture();
  const built = buildPlayableLevel(document, -1), cells = built.topology!.cells;
  const copied = cells.find(cell => cell.sourceId === copy.id)!, original = cells.find(cell => cell.originZone === zone)!;
  const animation = importedAnimationTables(built.document).tables.water[group]!;
  const entries = animation.entries.filter(entry => entry.zone === copied.id || entry.zone === original.id);
  expect(new Set(entries.map(entry => entry.zone))).toEqual(new Set([copied.id, original.id]));
  expect(new Set(entries.map(entry => entry.graphic)).size).toBe(entries.length);
  const header = new DataView(animation.header.buffer, animation.header.byteOffset, animation.header.byteLength);
  const top = header.getInt32(0) / 256, bottom = header.getInt32(4) / 256;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'water-copy.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', String(document.playtestSlot), { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const heights = new Set<number>();
  const samples: Awaited<ReturnType<typeof engineWaterState>>[] = [];
  await expect.poll(async () => {
    const states = await engineWaterState(page, entries);
    samples.push(states);
    heights.add(states[0]!.water);
    return heights.size;
  }, { intervals: [30, 30, 30] }).toBeGreaterThan(2);
  for (const states of samples) {
    for (const state of states) {
      expect(state.water).toBeGreaterThanOrEqual(top); expect(state.water).toBeLessThanOrEqual(bottom);
      expect(state.polygon).toBe(state.water);
      expect(state.water).toBe(states[0]!.water);
    }
  }
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('status')).not.toHaveText('Playtest running');
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
