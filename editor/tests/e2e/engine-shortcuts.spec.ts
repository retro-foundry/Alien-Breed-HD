import { expect, test, type Page, type Locator } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture } from '../roomHeightFixture';
import { createObject, floorMatchedObjectHeight, objectTemplates } from '../../src/model/objectAuthoring';
import { applyCommand } from '../../src/model/commands';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { decodeProject, encodeProject } from '../../src/model/project';
import { captureEngine, engineObjectState, enginePickDetails } from '../engineCapture';
import { objectSelectionIds } from '../../src/model/objectGroups';

async function pickActor(page: Page) {
  const fixture = roomHeightFixture(), template = objectTemplates(fixture.document).find(object => object.type === 0)!;
  const made = createObject(fixture.document, { template: template.id, room: fixture.from.id, x: 20128, z: 20180,
    y: floorMatchedObjectHeight(fixture.document, template, fixture.from.id, false), upper: false });
  const source = applyCommand(fixture.document, made.command), built = buildPlayableLevel(source, -1);
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'shortcut-actor.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(source)) });
  await expect(page.locator('.project-picker strong')).toHaveText(source.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await captureEngine(page, { zone: built.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!.id, x: 20128, z: 20032, upper: 0, angle: 0 });
  const points = Array.from({ length: 441 }, (_, index) => ({ x: (index % 21 + 1) / 22, y: (Math.floor(index / 21) + 1) / 22 }));
  const samples = await enginePickDetails(page, points), index = samples.findIndex(pick => pick.object?.id === made.id); expect(index).toBeGreaterThanOrEqual(0);
  const canvas = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas'), rect = (await canvas.boundingBox())!, point = points[index]!;
  await canvas.click({ position: { x: point.x * rect.width, y: point.y * rect.height } });
  await expect(page.locator('.inspector-heading')).toContainText(`Object ${made.id}`);
  await expect(canvas).toBeFocused();
  return { source, id: made.id, preview, canvas };
}
async function rebuilt(preview: Locator, revision: string) {
  await expect(preview).not.toHaveAttribute('data-revision', revision);
  await expect(preview.getByRole('status')).toContainText('zones loaded');
}
async function sendKey(page: Page, revision: number, key: string, modifiers: object = {}) {
  const frame = page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  await frame.evaluate((_, { revision, key, modifiers }) => parent.postMessage({ source: 'ab3d-engine', type: 'shortcut', revision,
    shortcut: { key, ctrlKey: false, metaKey: false, altKey: false, shiftKey: false, repeat: false, ...modifiers } }, location.origin), { revision, key, modifiers });
}

test('inspection clipboard, save, Undo/Redo and Delete use normal source history', async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const { source, id, preview, canvas } = await pickActor(page), copy = source.objects.length;
  const chooser = page.waitForEvent('filechooser'); await page.keyboard.press('Control+o'); await (await chooser).setFiles([]);
  await expect(page.locator('.inspector-heading')).toContainText(`Object ${id}`); await canvas.focus();
  await page.keyboard.press('Control+c'); await expect(page.getByRole('button', { name: 'Paste objects', exact: true })).toBeEnabled();
  const revision = (await preview.getAttribute('data-revision'))!;
  await page.keyboard.press('Control+v'); await rebuilt(preview, revision);
  await expect(page.locator('.inspector-heading')).toContainText(`Object ${copy}`);
  expect((await engineObjectState(page, copy)).cid).not.toBe((await engineObjectState(page, id)).cid);
  await canvas.focus(); const download = page.waitForEvent('download'); await page.keyboard.press('Control+s');
  const path = testInfo.outputPath('inspection-shortcuts.ab3d'); await (await download).saveAs(path);
  expect(decodeProject(new Uint8Array(await readFile(path))).objects).toHaveLength(copy + 1);
  for (const shortcut of ['Control+z', 'Control+Shift+z', 'Delete', 'Meta+z']) {
    await canvas.focus(); const before = (await preview.getAttribute('data-revision'))!;
    await page.keyboard.press(shortcut); await rebuilt(preview, before);
  }
  expect((await engineObjectState(page, copy)).zone).toBeGreaterThanOrEqual(0);
  expect((await engineObjectState(page, id)).zone).toBeGreaterThanOrEqual(0);
  await page.getByRole('button', { name: 'Show objects', exact: true }).click();
  await canvas.focus(); await page.keyboard.press('Control+a');
  await expect(page.getByRole('heading', { name: `${objectSelectionIds(source, '', false).length + 1} selected objects`, exact: true })).toBeVisible();
  await expect(preview.getByRole('alert')).toHaveCount(0);
});

test('inspection dialogs, text fields, stale messages and playtests isolate editor commands', async ({ page }) => {
  test.setTimeout(90000);
  const { id, preview, canvas } = await pickActor(page), revision = Number(await preview.getAttribute('data-revision'));
  await page.getByRole('button', { name: 'Add object', exact: true }).click(); const dialog = page.getByRole('dialog', { name: 'Choose object to place', exact: true });
  await expect(dialog).toBeVisible();
  await sendKey(page, revision, 'Delete'); await sendKey(page, revision, 'z', { ctrlKey: true });
  const x = dialog.getByRole('searchbox', { name: 'Find object to place', exact: true });
  await x.fill('20128'); await x.press('Control+a'); await x.press('Backspace'); await expect(x).toHaveValue('');
  await page.keyboard.press('Escape'); await expect(dialog).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await sendKey(page, revision - 1, 'Delete'); await sendKey(page, revision, 'Delete', { ctrlKey: 'yes' });
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await canvas.focus(); await page.keyboard.press('?'); await expect(page.getByRole('dialog', { name: 'Editor help' })).toBeVisible();
  await sendKey(page, revision, 'Delete'); await sendKey(page, revision, 'z', { ctrlKey: true });
  await page.keyboard.press('Escape'); await expect(page.getByRole('dialog', { name: 'Editor help' })).toHaveCount(0);
  await expect(page.locator('.inspector-heading')).toContainText(`Object ${id}`);
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await canvas.focus(); await page.keyboard.press('Delete'); await page.keyboard.press('Control+z');
  await sendKey(page, Number(await preview.getAttribute('data-revision')), 'Delete');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await expect(page.getByRole('dialog')).toHaveCount(0); await expect(page.locator('.dirty-dot')).toHaveCount(0);
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  await expect(page.locator('.inspector-heading')).toContainText(`Object ${id}`);
});

test('a missing shortcut module gives a recoverable engine error without losing source', async ({ page }) => {
  await page.route('**/inspection-shortcuts.mjs', route => route.abort());
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview.getByRole('alert')).toContainText(/import|module|fetch|load/i, { timeout: 60000 });
  await page.unroute('**/inspection-shortcuts.mjs');
  await preview.getByRole('button', { name: 'Reload engine' }).click();
  await expect(preview.getByRole('status')).toContainText('zones loaded', { timeout: 60000 });
  await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});

test('successive Undo commands and save remain available during a delayed inspection rebuild', async ({ page }, testInfo) => {
  await page.addInitScript(() => {
    if (window !== window.top) return;
    const host = window as Window & { holdEditorLoads?: boolean; heldEditorLoads?: MessageEvent[] };
    host.heldEditorLoads = [];
    window.addEventListener('message', event => {
      if (!host.holdEditorLoads || event.data?.source !== 'ab3d-engine' || event.data.type !== 'loaded') return;
      event.stopImmediatePropagation();
      host.heldEditorLoads!.push(new MessageEvent('message', { data: event.data, origin: event.origin, source: event.source }));
    }, true);
  });
  const { source, id, preview, canvas } = await pickActor(page), original = source.objects[id]!.y;
  for (const offset of [.5, 1]) {
    const revision = (await preview.getAttribute('data-revision'))!;
    const height = page.getByRole('spinbutton', { name: 'Height', exact: true });
    await height.fill(String(original + offset)); await height.press('Tab'); await rebuilt(preview, revision);
  }
  await page.evaluate(() => { (window as Window & { holdEditorLoads?: boolean }).holdEditorLoads = true; });
  await canvas.focus(); await page.keyboard.press('Control+z');
  await page.waitForFunction(() => ((window as Window & { heldEditorLoads?: MessageEvent[] }).heldEditorLoads?.length ?? 0) > 0);
  await page.keyboard.press('Control+z');
  await expect(page.getByRole('button', { name: 'Undo', exact: true })).toBeDisabled();
  await expect(page.getByRole('spinbutton', { name: 'Height', exact: true })).toHaveValue(String(original));
  const download = page.waitForEvent('download'); await page.keyboard.press('Control+s');
  const path = testInfo.outputPath('rapid-undo.ab3d'); await (await download).saveAs(path);
  expect(decodeProject(new Uint8Array(await readFile(path))).objects[id]!.y).toBe(original);
  await page.evaluate(() => {
    const host = window as Window & { holdEditorLoads?: boolean; heldEditorLoads?: MessageEvent[] };
    host.holdEditorLoads = false;
    for (const event of host.heldEditorLoads!) window.dispatchEvent(event);
    host.heldEditorLoads = [];
  });
  await expect(preview.getByRole('status')).toContainText('zones loaded');
  await expect.poll(async () => (await engineObjectState(page, id)).yFixed).toBe(original * 256);
});
