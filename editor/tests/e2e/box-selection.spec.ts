import { expect, test } from '@playwright/test';
import { objectStoreyFixture } from '../objectStoreyFixture';
import { encodeProject } from '../../src/model/project';

test('rectangle selection supports every direction, additive selection and cancellation without edits', async ({ page }, testInfo) => {
  const fixture = objectStoreyFixture();
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'box.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(fixture.document.name);
  await page.getByRole('button', { name: 'Upper', exact: true }).click();
  await page.getByRole('tab', { name: 'Objects', exact: true }).click();
  const first = page.getByRole('button', { name: `Health ${fixture.ids[0]}`, exact: true });
  await first.click();
  const map = page.locator('canvas[aria-label="Level map"]');
  await map.focus(); await page.keyboard.press('Shift+f');
  const box = (await map.boundingBox())!, view = await map.evaluate(node => ({ x: Number((node as HTMLElement).dataset.worldX), z: Number((node as HTMLElement).dataset.worldZ), scale: Number((node as HTMLElement).dataset.worldScale) }));
  const pixel = (x: number, z: number) => ({ x: box.x + box.width / 2 + (x - view.x) * view.scale, y: box.y + box.height / 2 - (z - view.z) * view.scale });
  const inspector = page.getByRole('complementary', { name: 'Object group inspector' });
  const tool = page.getByRole('button', { name: 'Box select objects', exact: true });
  for (const [index, rectangle] of [[20100, 20100, 20160, 20220], [20160, 20220, 20100, 20100], [20100, 20220, 20160, 20100], [20160, 20100, 20100, 20220]].entries()) {
    await map.focus(); await page.keyboard.press('b'); await expect(tool).toHaveAttribute('aria-pressed', 'true');
    const a = pixel(rectangle[0]!, rectangle[1]!), b = pixel(rectangle[2]!, rectangle[3]!);
    await page.mouse.move(a.x, a.y); await page.mouse.down(); await page.mouse.move(b.x, b.y, { steps: 5 });
    if (index === 0) await page.screenshot({ path: testInfo.outputPath('box-selection.png'), fullPage: true });
    await page.mouse.up();
    await expect(inspector.getByRole('heading', { name: '2 selected objects', exact: true })).toBeVisible();
    await expect(tool).toHaveAttribute('aria-pressed', 'false');
  }
  await first.click(); await map.focus(); await page.keyboard.press('b');
  const a = pixel(20100, 20170), b = pixel(20160, 20210);
  await page.keyboard.down('Shift'); await page.mouse.move(a.x, a.y); await page.mouse.down(); await page.mouse.move(b.x, b.y, { steps: 5 }); await page.mouse.up(); await page.keyboard.up('Shift');
  await expect(inspector).toBeVisible();
  await tool.click(); await page.mouse.move(a.x, a.y); await page.mouse.down(); await page.mouse.move(b.x, b.y, { steps: 5 });
  await page.keyboard.press('Escape'); await page.mouse.up();
  await expect(inspector.getByRole('heading', { name: '2 selected objects', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Show objects', exact: true }).click();
  await tool.click(); await page.mouse.move(a.x, a.y); await page.mouse.down(); await page.mouse.move(b.x, b.y, { steps: 5 }); await page.mouse.up();
  await expect(inspector).toHaveCount(0);
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
