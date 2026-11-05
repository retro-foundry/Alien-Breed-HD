import { expect, test } from '@playwright/test';
import { movingControllerCopyFixture } from '../movingControllerCopyFixture';
import { applyCommand, editRoomSource } from '../../src/model/commands';
import { setRoomControllerCopies } from '../../src/model/movingControllerCopies';
import { setWallSettings } from '../../src/model/wallSettings';
import { importedAnimationTables } from '../../src/model/animationBuild';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { encodeProject } from '../../src/model/project';
import { engineMechanismTrace, engineMechanismWallState } from '../engineCapture';
import { setWallMaterial, validateWallMaterialSampling, wallMaterialChoices } from '../../src/model/wallMaterials';

for (const kind of ['door', 'lift'] as const) test(`native ${kind} updates retain edited wall brightness and use the edited texture phase base`, async ({ page }) => {
  test.setTimeout(90000);
  const fixture = movingControllerCopyFixture(kind);
  let document = applyCommand(fixture.document, editRoomSource(setRoomControllerCopies(fixture.document, [fixture.copy])));
  const controller = importedAnimationTables(document).tables[kind === 'door' ? 'doors' : 'lifts']![fixture.copy.id]!, graphic = controller.walls[0]!.graphic;
  const owner = document.zones.find(zone => [...zone.lower, ...zone.upper].some(item => item.offset === graphic))!;
  const target = { room: fixture.copy.graphicsRooms.get(owner.id)!, record: graphic, upper: owner.upper.some(item => item.offset === graphic) };
  const style = wallMaterialChoices(document).find(item => {
    try { validateWallMaterialSampling(item.wall!, { ...item.wall!, horizontalMask: 63 }, 1); return true; } catch { return false; }
  })!;
  expect(style).toBeDefined();
  document = applyCommand(document, editRoomSource(setWallSettings(document, target, { horizontalOffset: 1, verticalOffset: 9, brightness: -4, repeat: { width: 64, fit: 'up' } })));
  document = applyCommand(document, editRoomSource(setWallMaterial(document, target, style.offset)));
  const built = buildPlayableLevel(document, -1).document, mechanism = built.mechanisms.filter(item => item.kind === kind).at(-1)!;
  const compiled = importedAnimationTables(built).tables[kind === 'door' ? 'doors' : 'lifts']!.at(-1)!, base = compiled.walls[0]!.base;
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'moving-wall-settings.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  const preview = page.getByRole('region', { name: '3D game engine preview' }); await expect(preview).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  await preview.getByRole('button', { name: 'Build & play', exact: true }).click(); await expect(preview.getByRole('status')).toHaveText('Playtest running');
  const trace = await engineMechanismTrace(page, kind, mechanism.id, false);
  expect(trace.some(state => state.current > mechanism.top && state.current < mechanism.bottom)).toBe(true);
  for (const state of trace) expect(state).toMatchObject({ plane: state.current, unsynchronizedWalls: 0 });
  const wall = await engineMechanismWallState(page, kind, mechanism.id), scroll = (-Math.floor(wall.current)) & 255;
  expect(wall.base).toBe(base); expect(wall.brightness).toBe(-4);
  expect(wall.horizontal).toBe(1);
  expect(wall.horizontal).toBe(kind === 'door' ? base >>> 12 : base >> 16);
  expect(wall.vertical & 65535).toBe(kind === 'door' ? (base & 4095) + scroll : base & 65535);
  await expect(preview.getByRole('alert')).toHaveCount(0);
});
