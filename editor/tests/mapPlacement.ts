import { expect, type Page } from '@playwright/test';
export async function framePlacementRoom(page: Page, room: string) {
  await page.getByRole('tab', { name: 'Zones', exact: true }).click();
  await page.getByRole('button', { name: new RegExp(`^Room ${room.replace(/^room-/, '')}`) }).click();
  await page.locator('canvas[aria-label="Level map"]').focus();
  await page.keyboard.press('Shift+f');
  await expect.poll(async () => Number(await page.locator('canvas[aria-label="Level map"]').getAttribute('data-world-scale'))).toBeGreaterThan(0.5);
  await page.getByRole('combobox', { name: 'Grid spacing' }).selectOption('8');
}
export async function clickMap(page: Page, x: number, z: number) {
  const map = page.locator('canvas[aria-label="Level map"]'), bounds = (await map.boundingBox())!;
  const view = await map.evaluate(node => ({ x: Number(node.getAttribute('data-world-x')), z: Number(node.getAttribute('data-world-z')), scale: Number(node.getAttribute('data-world-scale')) }));
  await map.click({ position: { x: bounds.width / 2 + (x - view.x) * view.scale, y: bounds.height / 2 - (z - view.z) * view.scale } });
}
