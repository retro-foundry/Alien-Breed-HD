import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { decodeProject } from '../../src/model/project';
import { originalFiles } from '../helpers';
import { openLevel } from '../../src/model/level';
import { enginePickDetails } from '../engineCapture';

test('unavailable WebGL reports a startup error, preserves source and recovers on engine reload', async ({ page }, testInfo) => {
  test.setTimeout(90000);
  await page.addInitScript(() => {
    if (window === window.top || sessionStorage.getItem('editor-test-webgl-denied')) return;
    sessionStorage.setItem('editor-test-webgl-denied', '1');
    // Reject actual graphics context creation for the first iframe lifetime.
    // The unchanged native initialization must detect SDL's failed backend.
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, type: string, ...args: unknown[]) {
      if (['webgl', 'webgl2', 'experimental-webgl'].includes(type)) return null;
      return Reflect.apply(getContext, this, [type, ...args]);
    } as typeof getContext;
  });
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('alert')).toContainText('The 3D renderer could not start', { timeout: 60000 });
  await expect(preview.getByRole('alert')).toContainText('WebGL2');
  await expect(preview.getByRole('alert')).toContainText('Your map is still editable and can be saved');
  await expect(preview.getByRole('button', { name: 'Build & play', exact: true })).toBeDisabled();
  await expect(preview.getByRole('status')).toHaveText('Engine preview unavailable');
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const file = testInfo.outputPath('source-after-renderer-failure.ab3d'); await (await download).saveAs(file);
  const saved = decodeProject(new Uint8Array(await readFile(file))), original = openLevel('Level 1', originalFiles(1));
  for (const key of ['data', 'graphics', 'clips'] as const)
    expect(Buffer.from(saved.files[key]).equals(Buffer.from(original.files[key]))).toBe(true);
  await page.screenshot({ path: testInfo.outputPath('renderer-startup-error.png'), fullPage: true });
  await preview.getByRole('button', { name: 'Reload engine', exact: true }).click();
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await expect(preview.getByRole('alert')).toHaveCount(0);
  const samples = await enginePickDetails(page, [{ x: .5, y: .5 }]);
  expect(samples[0]!.zone).toBeGreaterThanOrEqual(0);
  expect(samples[0]!.surface).not.toBeNull();
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
});
