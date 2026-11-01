import { expect, test } from '@playwright/test';
import { roomHeightFixture } from '../roomHeightFixture';
import { captureRoomProperties } from '../../src/model/roomProperties';
import { roomSettings } from '../../src/model/roomHeights';
import { setRoomProperties } from '../../src/model/roomSource';
import { applyCommand, editRoomSource, numberPatch } from '../../src/model/commands';
import { encodeProject } from '../../src/model/project';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { engineRoomProperties } from '../engineCapture';

for (const brightness of [17, 2 << 8]) test(`captured room properties with brightness ${brightness} reach the native engine independently of later template edits`, async ({ page }) => {
  const fixture = roomHeightFixture(); let document = fixture.document;
  const properties = { ...captureRoomProperties(roomSettings(document, fixture.from)), brightness, upperBrightness: -12,
    control: -3, upperControl: 7, floorSound: 2, upperFloorSound: 3, backdrop: true };
  document = applyCommand(document, editRoomSource(setRoomProperties(document.roomSource, fixture.from.id, properties)));
  const template = document.zones[fixture.from.recordZone]!;
  document = applyCommand(document, { label: 'Change template', patches: [numberPatch(document, 'data', template.offset + 22, 6, 2)] });
  const built = buildPlayableLevel(document, -1), cell = built.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!;
  await page.goto('/'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({ name: 'native-room-properties.ab3d', mimeType: 'application/zip', buffer: Buffer.from(encodeProject(document)) });
  await expect(page.locator('.project-picker strong')).toHaveText(document.name);
  await page.getByRole('button', { name: 'Engine view', exact: true }).click();
  await expect(page.getByRole('region', { name: '3D game engine preview' })).toHaveAttribute('data-level-slot', '0', { timeout: 60000 });
  expect(await engineRoomProperties(page, cell.id)).toEqual(properties);
  expect((await engineRoomProperties(page, 0)).brightness).toBe(6);
  await expect(page.locator('.dirty-dot')).toHaveCount(0);
});
