import { expect, test } from '@playwright/test';
import { engineRoomProperties } from '../engineCapture';

test('getting started leads to a searchable room, with clear selection actions', async ({ page }) => {
  await page.goto('/?view=top');
  await page.getByRole('button', { name: 'Find a room', exact: true }).click();
  const search = page.getByRole('textbox', { name: 'Search map contents' });
  await expect(search).toBeFocused();
  await search.fill('Zone 003');
  await expect(page.locator('.content-row')).toHaveCount(1);
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  await expect(page.locator('.selection-description')).toHaveText('Zone 3');
  await page.getByRole('button', { name: 'Frame selection', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Copy rooms', exact: true })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Copy objects', exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: 'Clear contents search' }).click();
  await expect(search).toBeFocused();
  await expect(page.locator('.content-row')).toHaveCount(138);
  await page.getByRole('button', { name: 'Clear selection', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Find a room', exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
});

test('find settings reveals expandable fields, retains invalid drafts, and resets for another selection', async ({ page }) => {
  await page.goto('/?view=top');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  const search = page.getByRole('textbox', { name: 'Find a setting', exact: true });
  await search.fill('Corner 1 X');
  const corner = page.getByRole('spinbutton', { name: 'Corner 1 X', exact: true });
  // Search must also discover labels on fields whose visual labels are abbreviated.
  await expect(corner).toBeVisible();
  const original = await corner.inputValue();
  await corner.fill('100000');
  await search.fill('brightness');
  await expect(page.getByRole('spinbutton', { name: 'Brightness', exact: true })).toBeVisible();
  await expect(corner).toBeHidden();
  await page.getByRole('button', { name: 'Save project', exact: true }).click();
  await expect(search).toHaveValue('');
  await expect(corner).toBeVisible();
  await expect(corner).toBeFocused();
  await search.fill('Corner 1 X');
  await expect(corner).toHaveValue('100000');
  await corner.press('Escape');
  await expect(corner).toHaveValue(original);
  await search.fill('no-such-setting');
  await expect(page.locator('.setting-results')).toContainText('No settings match');
  await search.press('Escape');
  await expect(search).toHaveValue('');
  await expect(page.locator('.inspector-heading h2')).toHaveText('Zone 3');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await page.getByRole('combobox', { name: 'Jump to properties' }).selectOption({ label: 'Room heights' });
  await expect(page.getByRole('heading', { name: 'Room heights', exact: true })).toBeInViewport();
  await search.fill('water');
  await page.getByRole('button', { name: /^Zone 004/ }).click();
  await expect(search).toHaveValue('');
});

test('filtered numeric edits keep Undo and project saving', async ({ page }, testInfo) => {
  await page.goto('/?view=top');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  await page.getByRole('textbox', { name: 'Find a setting', exact: true }).fill('brightness');
  const brightness = page.getByRole('spinbutton', { name: 'Brightness', exact: true });
  await brightness.fill('21'); await brightness.press('Enter');
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(brightness).toHaveValue('-15');
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  await expect(brightness).toHaveValue('21');
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const saved = testInfo.outputPath('settings-search.ab3d');
  await (await download).saveAs(saved);
  await page.locator('input[type=file]').setInputFiles(saved);
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  await expect(brightness).toHaveValue('21');
});

test('help finds control tables and keeps keyboard focus inside the guide', async ({ page }) => {
  await page.goto('/?view=top');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  const help = page.getByRole('button', { name: 'Help and shortcuts' });
  await help.click();
  const dialog = page.getByRole('dialog', { name: 'Editor help' });
  const close = dialog.getByRole('button', { name: 'Close dialog' });
  await expect(close).toBeFocused();
  await close.press('Tab');
  const search = dialog.getByRole('textbox', { name: 'Search editor help' });
  await expect(search).toBeFocused();
  await search.fill('crouch');
  await expect(dialog.getByText('Crouch', { exact: true })).toBeVisible();
  await expect(dialog.getByRole('heading', { name: 'Wall alignment and brightness', exact: true })).toHaveCount(0);
  await search.press('Escape');
  await expect(dialog).toBeVisible();
  await expect(search).toHaveValue('');
  await search.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(help).toBeFocused();
  await expect(page.locator('.inspector-heading h2')).toHaveText('Zone 3');
});

test('main tools fit smaller windows and player starts can be searched', async ({ page }) => {
  for (const width of [760, 480]) {
    await page.setViewportSize({ width, height: 950 });
    await page.goto('/?view=top');
    await page.getByRole('button', { name: 'Toggle project navigator' }).click();
    await page.getByRole('tab', { name: 'Starts', exact: true }).click();
    const search = page.getByRole('textbox', { name: 'Search map contents' });
    await expect(search).toHaveAttribute('placeholder', 'Find a player start…');
    await search.fill('player 2');
    await expect(page.locator('.content-row')).toHaveCount(1);
    await page.getByRole('button', { name: /^Player 2/ }).click();
    await expect(page.locator('.navigator')).toBeHidden();
    for (const name of ['Select tool', 'Pan tool', 'Draw room', 'Add object', 'Build stairs', 'Undo', 'Redo']) {
      await expect(page.getByRole('button', { name, exact: true })).toBeInViewport();
    }
    expect(await page.locator('.map-toolbar').evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    expect(await page.locator('.editing-options').evaluate(element => element.scrollWidth <= element.clientWidth)).toBe(true);
    for (const name of ['Find action', 'Build and play level', 'Save project', 'Export level', 'Help and shortcuts']) {
      await expect(page.getByRole('button', { name, exact: true })).toBeInViewport();
    }
  }
});

test('drawing has visible progress and changing tools cancels the unfinished room', async ({ page }) => {
  await page.goto('/?view=top');
  await page.getByRole('button', { name: 'Draw room', exact: true }).click();
  await expect(page.locator('.selection-description')).toContainText('Drawing room');
  const canvas = page.locator('canvas[aria-label="Level map"]');
  await canvas.click({ position: { x: 240, y: 180 } });
  await expect(page.locator('.selection-description')).toContainText('1 corner');
  await expect(page.getByRole('button', { name: 'Finish room', exact: true })).toBeDisabled();
  await page.getByRole('button', { name: 'Undo corner', exact: true }).click();
  await expect(page.locator('.selection-description')).toContainText('0 corners');
  await page.getByRole('button', { name: 'Pan tool', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Finish room', exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
});

test('action search finds creation tools with keyboard navigation and preserves focus on cancel', async ({ page }) => {
  await page.goto('/?view=top');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  const trigger = page.getByRole('button', { name: 'Find action', exact: true });
  await trigger.click();
  const dialog = page.getByRole('dialog', { name: 'Find an editor action' });
  const search = dialog.getByRole('textbox', { name: 'Search editor actions' });
  await expect(search).toBeFocused();
  await search.fill('scenery');
  await search.press('ArrowDown');
  await expect(dialog.getByRole('button', { name: /^Add object/ })).toBeFocused();
  await page.keyboard.press('Enter');
  const palette = page.getByRole('dialog', { name: 'Choose object to place' });
  await expect(palette).toBeVisible();
  await page.keyboard.press('Escape');
  await trigger.focus();
  await page.keyboard.press('Control+k');
  await expect(search).toBeFocused();
  await search.fill('no-such-action');
  await expect(dialog.getByRole('status')).toContainText('No actions match');
  await search.press('Escape');
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
});

test('one-click play builds current edits and does not restart when returning to inspection', async ({ page }) => {
  await page.goto('/?view=top');
  await page.getByRole('button', { name: /^Zone 003/ }).click();
  const brightness = page.getByRole('spinbutton', { name: 'Brightness', exact: true });
  await brightness.fill('12');
  await brightness.press('Control+Enter');
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toHaveText('Playtest running', { timeout: 60000 });
  await expect(brightness).toHaveValue('12');
  await expect(preview).toHaveAttribute('data-rendered-pixels', /[1-9]/);
  expect((await engineRoomProperties(page, 3)).brightness).toBe(12);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await page.getByRole('button', { name: 'Editor view', exact: true }).click();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await expect(preview.getByRole('button', { name: 'Build & play', exact: true })).toBeEnabled();
  await expect(preview.getByRole('button', { name: 'Stop', exact: true })).toHaveCount(0);
});
