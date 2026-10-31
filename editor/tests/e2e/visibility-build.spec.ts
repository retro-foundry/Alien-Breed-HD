import { expect, test, type Page } from '@playwright/test';
import { openLevel } from '../../src/model/level';
import { buildVisibilityFiles } from '../../src/model/visibilityBuild';
import { buildGraphicsStreams } from '../../src/model/graphicsBuild';
import { encodeProject } from '../../src/model/project';
import { originalFiles } from '../helpers';

async function capture(page: Page, camera: { zone: number; x: number; z: number; upper: number; angle: number }) {
  return page.evaluate(camera => new Promise<number>((resolve, reject) => {
    const frame = document.querySelector<HTMLIFrameElement>('iframe[title="Alien Breed game engine"]')!;
    const id = crypto.randomUUID();
    const timer = setTimeout(() => { window.removeEventListener('message', receive); reject(new Error('Engine frame capture timed out')); }, 10000);
    const receive = (event: MessageEvent) => {
      if (event.source !== frame.contentWindow || event.origin !== location.origin || event.data?.type !== 'captured' || event.data.id !== id) return;
      clearTimeout(timer); window.removeEventListener('message', receive); resolve(event.data.checksum);
    };
    window.addEventListener('message', receive);
    frame.contentWindow!.postMessage({ source: 'ab3d-editor', type: 'capture', id, camera }, location.origin);
  }), camera);
}

test('rebuilt visibility and relocated graphics load in the real engine and preserve original spawn views', async ({ page }, testInfo) => {
  test.setTimeout(240000);
  page.on('dialog', dialog => void dialog.accept());
  const errors: string[] = [], comparisons: object[] = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: '3D view', exact: true }).click();
  for (let level = 1; level <= 16; level++) {
    const original = openLevel(`Level ${level}`, originalFiles(level));
    if (level > 1) await page.getByRole('combobox', { name: 'Original level' }).selectOption({ value: String(level) });
    const preview = page.getByRole('region', { name: '3D game engine preview' });
    await expect(preview.getByRole('status')).toHaveText(`${original.zones.length} zones loaded · Game engine`, { timeout: 60000 });
    const start = original.starts[0]!, camera = { zone: start.zone, x: start.x, z: start.z, upper: 0, angle: 0 };
    const before = await capture(page, camera);
    expect(before, `Level ${level} original capture`).not.toBe(0);
    const visibility = openLevel(`${original.name} visibility`, buildVisibilityFiles(original));
    const graphics = buildGraphicsStreams(visibility, visibility.zones.map(zone => zone.id).toReversed(), visibility.zones.length + 2);
    const rebuilt = openLevel(`${original.name} rebuilt`, { ...visibility.files, graphics });
    await page.locator('input[type=file]').setInputFiles({ name: `rebuilt-${level}.ab3d`, mimeType: 'application/zip', buffer: Buffer.from(encodeProject(rebuilt)) });
    await expect(page.locator('.project-picker strong')).toHaveText(rebuilt.name);
    await preview.getByRole('combobox', { name: 'Playtest level slot' }).selectOption({ value: String(level - 1) });
    // The host reports runtime zone-slot capacity, including the two empty
    // slots deliberately added to exercise prologue and animation relocation.
    await expect(preview.getByRole('status')).toHaveText(`${original.zones.length + 2} zones loaded · Game engine`, { timeout: 60000 });
    await expect(preview).toHaveAttribute('data-level-slot', String(level - 1));
    const after = await capture(page, camera);
    comparisons.push({ level, before, after, equal: before === after });
    if (before !== after) await page.screenshot({ path: testInfo.outputPath(`rebuilt-${level}.png`), fullPage: true });
    expect(after, `Level ${level} start view after rebuilding`).toBe(before);
  }
  await testInfo.attach('engine-frame-comparisons', { body: JSON.stringify(comparisons, null, 2), contentType: 'application/json' });
  expect(errors).toEqual([]);
});
