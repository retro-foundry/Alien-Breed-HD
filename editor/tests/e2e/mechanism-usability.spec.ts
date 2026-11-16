import { expect, test } from '@playwright/test';
import { authoredMovingFixture } from '../authoredMovingFixture';
import { switchRoomFixture } from '../sourceSwitchFixture';
import { encodeProject } from '../../src/model/project';

test('room setup reveals filtered door and lift forms, focuses heights and offers actual landing floors', async ({ page }) => {
  const fixture = authoredMovingFixture('lift');
  await page.goto('/?view=top');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'setup.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.room.id.replace('room-', '')}(?:\\s|$)`) }).click();
  const search = page.getByRole('textbox', { name: 'Find a setting', exact: true });
  await search.fill('water');
  await page.getByRole('button', { name: 'Set up door', exact: true }).click();
  const form = page.getByRole('form', { name: 'Authored mechanism settings' });
  await expect(search).toHaveValue('');
  await expect(form.getByRole('spinbutton', { name: 'New mechanism top', exact: true })).toBeFocused();
  await expect(form.getByRole('combobox', { name: 'New mechanism initial position' })).toHaveValue('bottom');
  await expect(form).toContainText('The ceiling rises');
  await form.getByRole('button', { name: 'Cancel mechanism draft', exact: true }).click();
  await page.getByRole('button', { name: 'Find action', exact: true }).click();
  const actions = page.getByRole('dialog', { name: 'Find an editor action' });
  await actions.getByRole('textbox').fill('create lift');
  await actions.getByRole('button', { name: /^Create lift/ }).click();
  await expect(form.getByRole('status')).toContainText('Choose two different stops');
  await form.getByRole('combobox', { name: 'Choose upper landing' }).selectOption('0');
  await expect(form.getByRole('spinbutton', { name: 'New mechanism top', exact: true })).toHaveValue('0');
  await expect(form.getByRole('status')).toContainText('Travel: 32 units');
  await form.getByRole('button', { name: 'Cancel mechanism draft', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
});

test('switch setup opens from room actions with an unused number selected and wall choice focused', async ({ page }) => {
  const fixture = switchRoomFixture();
  await page.goto('/?view=top');
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'switch-setup.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(fixture.document)) });
  await page.getByRole('button', { name: new RegExp(`^Room ${fixture.room.id.replace('room-', '')}(?:\\s|$)`) }).click();
  await page.getByRole('textbox', { name: 'Find a setting', exact: true }).fill('lighting');
  await page.getByRole('button', { name: 'Set up switch', exact: true }).click();
  const section = page.getByRole('region', { name: 'Wall switches', exact: true });
  await expect(section.getByRole('combobox', { name: 'New switch wall', exact: true })).toBeFocused();
  await expect(section.getByRole('combobox', { name: 'New switch number', exact: true })).toHaveValue(String(fixture.slot));
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
});
