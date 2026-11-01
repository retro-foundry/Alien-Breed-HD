import { expect, test } from '@playwright/test';

for (const panel of ['Editor help', 'Recent workspaces']) test(`${panel} contains shortcuts and Escape preserves the editing context`, async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  const brightness = page.getByRole('spinbutton', { name: 'Brightness', exact: true });
  await brightness.fill('6'); await brightness.press('Tab');
  await page.getByRole('button', { name: 'Pan tool', exact: true }).click();
  const trigger = panel === 'Editor help' ? page.getByRole('button', { name: 'Help and shortcuts' }) : page.getByRole('button', { name: /Recent workspaces/ });
  await trigger.click();
  const dialog = page.getByRole('dialog', { name: panel, exact: true });
  await expect(dialog).toBeVisible();
  for (const shortcut of ['Control+z', 'Control+Shift+z', 'Control+y', 'v', 'b', '/', 'Delete']) {
    await page.keyboard.press(shortcut);
    await expect(brightness).toHaveValue('6');
    await expect(page.getByRole('button', { name: 'Pan tool', exact: true })).toHaveAttribute('aria-pressed', 'true');
  }
  await expect(dialog).toBeVisible();
  await expect(brightness).toHaveValue('6');
  await expect(page.getByRole('button', { name: 'Pan tool', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await expect(dialog.getByRole('button', { name: 'Close dialog' })).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await expect(brightness).toHaveValue('6');
  await expect(page.getByRole('button', { name: 'Pan tool', exact: true })).toHaveAttribute('aria-pressed', 'true');
  // The same shortcut must still work after leaving the modal.
  await page.keyboard.press('Control+z');
  await expect(brightness).toHaveValue('-15');
});
