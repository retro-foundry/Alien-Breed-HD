import { expect, test } from '@playwright/test';
import { openLevel } from '../../src/model/level';
import { originalFiles } from '../helpers';
import { encodeProject } from '../../src/model/project';
import { mechanismDependencies } from '../../src/model/mechanismDependencies';

for (const kind of ['door', 'lift'] as const) test(`${kind} attachments show native owners and frame linked rooms without editing`, async ({ page }, testInfo) => {
  const document = openLevel('Attachment inspection', originalFiles(1), 0, -1);
  const mechanism = document.mechanisms.find(item => item.kind === kind && item.walls.length)!;
  const dependencies = mechanismDependencies(document, kind, mechanism.id);
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'attachments.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(dependencies.room).padStart(3, '0')}(?:\\s|$)`) }).click();
  const section = page.locator('.property-section').filter({ has: page.getByRole('heading', { name: `${kind === 'door' ? 'Door' : 'Lift'} ${mechanism.id}`, exact: true }) });
  await section.locator('summary').click();
  const attachments = section.locator('.mechanism-attachments');
  const table = attachments.getByRole('table', { name: 'Moving wall attachments' });
  await expect(table.getByRole('row')).toHaveCount(dependencies.walls.length + 1);
  for (const [index, wall] of dependencies.walls.entries()) {
    const row = table.getByRole('row').nth(index + 1);
    await expect(row.getByRole('cell').nth(0).getByRole('button')).toHaveText(`Zone ${wall.room}`);
    await expect(row.getByRole('cell').nth(1).getByRole('button')).toHaveText(wall.collisionRooms.map(room => `Zone ${room}`));
  }
  const link = table.getByRole('row').nth(1).getByRole('cell').nth(0).getByRole('button');
  await link.focus(); await expect(link).toBeFocused(); await link.press('Enter');
  await expect(section.getByRole('heading', { name: `${kind === 'door' ? 'Door' : 'Lift'} ${mechanism.id}`, exact: true })).toBeVisible();
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
  await page.screenshot({ path: testInfo.outputPath(`${kind}-attachments.png`), fullPage: true });
});
