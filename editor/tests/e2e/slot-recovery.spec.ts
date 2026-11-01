import { expect, test } from '@playwright/test';

test('slot-only changes retain unload protection and the open project when recovery storage fails', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(indexedDB, 'open', { value: () => { throw new DOMException('Recovery storage is unavailable', 'QuotaExceededError'); } });
  });
  await page.goto('/?view=top');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  const target = page.getByRole('combobox', { name: 'Target game slot', exact: true });
  await target.selectOption('8');
  await expect(page.locator('.dirty-dot')).toBeVisible();
  await expect(page.getByText('Autosave unavailable', { exact: true })).toBeVisible();
  const backup = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save project', exact: true }).click();
  await (await backup).cancel();
  await expect(page.locator('.dirty-dot')).toBeVisible();
  expect(await page.evaluate(() => {
    const event = new Event('beforeunload', { cancelable: true }); window.dispatchEvent(event); return event.defaultPrevented;
  })).toBe(true);
  await page.getByRole('combobox', { name: 'Original level' }).selectOption('2');
  await expect(page.getByRole('alert')).toContainText('Recovery storage is unavailable');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await expect(target).toHaveValue('8');
});
