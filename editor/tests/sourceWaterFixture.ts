import { applyCommand } from '../src/model/commands';
import { setRoomWater, setWaterGroup, waterMaterialChoices } from '../src/model/sourceWater';
import { roomHeightFixture } from './roomHeightFixture';
import { roomSettings } from '../src/model/roomHeights';
import type { LevelDocument } from '../src/model/level';

export function sourceWaterFixture(original?: LevelDocument) {
  const fixture = roomHeightFixture(0, original), settings = roomSettings(fixture.document, fixture.from);
  const group = { id: 5, top: settings.floor - 40, bottom: settings.floor - 16, current: settings.floor - 32, velocity: 128, enabled: true };
  let document = applyCommand(fixture.document, setWaterGroup(fixture.document, group));
  for (const room of [fixture.from, fixture.to]) document = applyCommand(document, setRoomWater(document, room.id, {
    material: waterMaterialChoices(document, room)[0]!.offset, height: group.current, group: group.id,
  }));
  return { ...fixture, document, group };
}
