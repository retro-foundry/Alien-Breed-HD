import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { openLevel } from '../../src/model/level';
import { originalFiles } from '../helpers';
import { encodeProject, decodeProject } from '../../src/model/project';
import { resolveAnimationRoom } from '../../src/model/animationRooms';
import { engineMechanismState, engineMechanismTrace } from '../engineCapture';
import { mechanismNeedsEndpoint } from '../../src/model/mechanismRanges';

for (const kind of ['door', 'lift'] as const) test(`${kind} movement range validates, applies atomically and loads native quarter-unit destinations`, async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const document = Array.from({ length: 16 }, (_, index) => openLevel('Movement range', originalFiles(index + 1))).find(document => document.mechanisms.some(item => item.kind === kind && item.top <= item.current && item.bottom >= item.current && item.top < item.bottom))!;
  const mechanism = document.mechanisms.find(item => item.kind === kind && item.top <= item.current && item.bottom >= item.current && item.top < item.bottom)!;
  const zone = resolveAnimationRoom(document, mechanism.zone, kind === 'lift')!;
  const field = mechanism.top < mechanism.current ? 'top' : 'bottom', value = field === 'top' ? mechanism.top + 0.25 : mechanism.bottom - 0.25;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'ranges.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(zone).padStart(3, '0')}(?:\\s|$)`) }).click();
  const form = page.getByRole('form', { name: `${kind} ${mechanism.id} movement range`, exact: true });
  const top = form.getByRole('spinbutton', { name: `${kind} ${mechanism.id} top destination` });
  await top.fill(String(mechanism.bottom + 1));
  await expect(form.getByRole('alert')).toBeVisible(); await expect(form.getByRole('button', { name: 'Apply range', exact: true })).toBeDisabled();
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
  await top.press('Escape'); await expect(top).toHaveValue(String(mechanism.top));
  if (kind === 'door' && mechanism.direction === 0 && mechanism.conditions === 0) {
    await top.fill(String(mechanism.current - 1));
    await form.getByRole('spinbutton', { name: `${kind} ${mechanism.id} bottom destination` }).fill(String(mechanism.current + 1));
    await form.getByRole('button', { name: 'Apply range', exact: true }).click();
    const key = page.locator('.property-section').filter({ has: form }).getByRole('checkbox', { name: 'Key 1', exact: true });
    await key.click();
    await expect(page.getByRole('alert')).toContainText('Keep one destination');
    await expect(key).not.toBeChecked();
    await page.getByRole('button', { name: 'Dismiss error', exact: true }).click();
    await page.getByRole('button', { name: 'Undo', exact: true }).click();
    await expect(page.locator('.dirty-dot')).toHaveCount(0);
  }
  if (mechanismNeedsEndpoint(mechanism)) {
    await top.fill(String(mechanism.current - 1));
    await form.getByRole('spinbutton', { name: `${kind} ${mechanism.id} bottom destination` }).fill(String(mechanism.current + 1));
    await expect(form.getByRole('alert')).toContainText('Keep one destination');
    await expect(form.getByRole('button', { name: 'Apply range', exact: true })).toBeDisabled();
    await form.getByRole('button', { name: 'Reset range draft', exact: true }).click();
    await expect(page.locator('.dirty-dot')).toHaveCount(0);
  }
  await form.getByRole('spinbutton', { name: `${kind} ${mechanism.id} ${field} destination` }).fill(String(value));
  await form.getByRole('button', { name: 'Apply range', exact: true }).click();
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  await expect(form.getByRole('spinbutton', { name: `${kind} ${mechanism.id} ${field} destination` })).toHaveValue(String(mechanism[field]));
  await page.getByRole('button', { name: 'Redo', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath(`${kind}-range.ab3d`); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path))), changed = saved.mechanisms.find(item => item.kind === kind && item.id === mechanism.id)!;
  expect(changed[field]).toBe(value); expect(changed.current).toBe(mechanism.current); expect(changed.walls).toEqual(mechanism.walls);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await engineMechanismState(page, kind, mechanism.id)).toMatchObject({ zone, current: mechanism.current, top: changed.top, bottom: changed.bottom });
});

for (const kind of ['door', 'lift'] as const) for (const cycle of [false, true]) test(`${kind} ${cycle ? 'cycles through an edited range' : 'animates to an edited destination'} with synchronized collision and wall geometry`, async ({ page }, testInfo) => {
  test.setTimeout(90000);
  const eligible = (item: ReturnType<typeof openLevel>['mechanisms'][number]) => item.kind === kind && item.current === item.bottom && item.top + 8.25 < item.bottom && item.walls.length > 0 && item.conditions < 4096;
  const document = Array.from({ length: 16 }, (_, index) => openLevel('Animated range', originalFiles(index + 1), index, -1)).find(document => document.mechanisms.some(eligible))!;
  const mechanism = document.mechanisms.find(eligible)!, zone = resolveAnimationRoom(document, mechanism.zone, kind === 'lift')!;
  const top = mechanism.top + 8.25;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'motion.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: new RegExp(`^Zone ${String(zone).padStart(3, '0')}(?:\\s|$)`) }).click();
  const section = page.locator('.property-section').filter({ has: page.getByRole('heading', { name: `${kind === 'door' ? 'Door' : 'Lift'} ${mechanism.id}`, exact: true }) });
  while (await section.getByRole('checkbox', { checked: true }).count()) await section.getByRole('checkbox', { checked: true }).first().uncheck();
  await section.getByRole('combobox', { name: `${kind} ${mechanism.id} opening rule` }).selectOption({ label: 'Timer expires' });
  await section.getByRole('combobox', { name: `${kind} ${mechanism.id} closing rule` }).selectOption({ label: cycle ? 'Timer expires' : 'Never' });
  const form = section.getByRole('form', { name: `${kind} ${mechanism.id} movement range` });
  await form.getByRole('spinbutton', { name: `${kind} ${mechanism.id} top destination` }).fill(String(top));
  await form.getByRole('button', { name: 'Apply range', exact: true }).click();
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' });
  await expect(preview).toHaveAttribute('data-level-slot', String(document.playtestSlot), { timeout: 60000 });
  expect(await engineMechanismState(page, kind, mechanism.id)).toMatchObject({ current: mechanism.current, top, bottom: mechanism.bottom, wallCount: mechanism.walls.length });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click();
  await expect(preview.getByRole('status')).toHaveText('Playtest running');
  await expect.poll(async () => {
    const state = await engineMechanismState(page, kind, mechanism.id);
    return state.current < mechanism.bottom && state.current > top && state.plane === state.current && state.unsynchronizedWalls === 0;
  }, { intervals: [20, 20, 20], timeout: 10000 }).toBe(true);
  // Sample beside the native display loop: cross-frame postMessage polling can
  // miss the single frame at which an automatic controller reverses direction.
  const trace = await engineMechanismTrace(page, kind, mechanism.id, cycle);
  const topAt = trace.findIndex(state => state.current === top);
  expect(topAt).toBeGreaterThanOrEqual(0);
  for (const state of trace) {
    expect(state.current).toBeGreaterThanOrEqual(top); expect(state.current).toBeLessThanOrEqual(mechanism.bottom);
    expect(state).toMatchObject({ plane: state.current, unsynchronizedWalls: 0, wallCount: mechanism.walls.length });
  }
  if (cycle) expect(trace.slice(topAt + 1).some(state => state.current === mechanism.bottom)).toBe(true);
  await page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas').screenshot({ path: testInfo.outputPath(`${kind}-animated.png`) });
  await preview.getByRole('button', { name: 'Stop', exact: true }).click();
  const download = page.waitForEvent('download'); await page.getByRole('button', { name: 'Save project', exact: true }).click();
  const path = testInfo.outputPath(`${kind}-after-motion.ab3d`); await (await download).saveAs(path);
  const saved = decodeProject(new Uint8Array(await readFile(path))), savedMechanism = saved.mechanisms.find(item => item.kind === kind && item.id === mechanism.id)!;
  expect(savedMechanism).toMatchObject({ current: mechanism.current, top, bottom: mechanism.bottom });
  expect(Buffer.from(saved.files.data).equals(Buffer.from(document.files.data))).toBe(true);
  for (const wall of mechanism.walls) expect(saved.files.graphics.slice(wall.graphic, wall.graphic + 30)).toEqual(document.files.graphics.slice(wall.graphic, wall.graphic + 30));
});
