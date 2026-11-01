import { expect, test } from '@playwright/test';
import { roomHeightFixture } from '../roomHeightFixture';
import { applyCommand, setSourceObject } from '../../src/model/commands';
import { createObject, objectTemplates } from '../../src/model/objectAuthoring';
import { encodeProject } from '../../src/model/project';

test('frame selection fits authored group positions and remains an explicit view-only action', async ({ page }) => {
  const fixture = roomHeightFixture(); let document = fixture.document;
  const template = objectTemplates(document).find(object => object.type === 0)!.id, ids: number[] = [];
  for (const [room, x, z] of [[fixture.from.id, 20128, 20128], [fixture.to.id, 21128, 20192]] as const) {
    const proposal = createObject(document, { template, room, x, z, y: -8 });
    document = applyCommand(document, proposal.command); ids.push(proposal.id);
  }
  document = applyCommand(document, setSourceObject(document, ids[0]!, fixture.from.id, 20140.5, 20128.75));
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'framing.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('tab', { name: 'Objects', exact: true }).click();
  await page.getByRole('button', { name: `Alien ${ids[0]}`, exact: true }).click();
  await page.getByRole('button', { name: `Alien ${ids[1]}`, exact: true }).click({ modifiers: ['Shift'] });
  const frame = page.getByRole('button', { name: 'Frame selected objects', exact: true });
  await frame.click();
  const map = page.locator('canvas[aria-label="Level map"]');
  await expect(map).toHaveAttribute('data-world-x', '20634.25');
  await expect(map).toHaveAttribute('data-world-z', '20160.375');
  const visible = await map.evaluate(node => {
    const canvas = node as HTMLCanvasElement, scale = Number(canvas.dataset.worldScale), x = Number(canvas.dataset.worldX), z = Number(canvas.dataset.worldZ);
    return [[20140.5, 20128.75], [21128, 20192]].every(point => Math.abs(point[0]! - x) * scale <= canvas.clientWidth / 2 - 40 && Math.abs(point[1]! - z) * scale <= canvas.clientHeight / 2 - 40);
  });
  expect(visible).toBe(true); await expect(page.locator('.dirty-dot')).toHaveCount(0);
  const box = (await map.boundingBox())!;
  await page.mouse.move(box.x + box.width * 0.75, box.y + box.height / 2); await page.mouse.down({ button: 'middle' });
  await page.mouse.move(box.x + box.width * 0.75 + 40, box.y + box.height / 2, { steps: 4 }); await page.mouse.up({ button: 'middle' });
  const panned = await map.getAttribute('data-world-x'); expect(panned).not.toBe('20634.25');
  await page.getByRole('spinbutton', { name: 'Group X offset' }).fill('8');
  await page.getByRole('button', { name: 'Move selected objects', exact: true }).click();
  await expect(map).toHaveAttribute('data-world-x', panned!);
  await map.focus(); await page.keyboard.press('Shift+f');
  await expect(map).toHaveAttribute('data-world-x', '20642.25');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(map).toHaveCount(0);
  await frame.click();
  await expect(map).toHaveAttribute('data-world-x', '20642.25');
  await expect(map).toHaveAttribute('data-world-z', '20160.375');
  await page.getByRole('button', { name: `Alien ${ids[0]}`, exact: true }).click();
  await map.focus(); await page.keyboard.press('Shift+f');
  await expect(map).toHaveAttribute('data-world-x', '20148.5');
  await expect(map).toHaveAttribute('data-world-z', '20128.75');
  await page.getByRole('tab', { name: 'Starts', exact: true }).click();
  await page.getByRole('button', { name: /^Player 1(?:\s|$)/ }).click();
  await map.focus(); await page.keyboard.press('Shift+f');
  await expect(map).toHaveAttribute('data-world-x', '20128');
  await expect(map).toHaveAttribute('data-world-z', '20128');
});
