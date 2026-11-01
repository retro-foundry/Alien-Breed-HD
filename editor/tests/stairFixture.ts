import { applyCommand, editRoomAppearance, editRoomSource, setSourceStart } from '../src/model/commands';
import { openLevel } from '../src/model/level';
import { beginRoomDraft, commitRoomDraft, setDraftCorners, setRoomHeights } from '../src/model/roomSource';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import type { StairFlight } from '../src/model/stairs';
import { originalFiles } from './helpers';

export function stairFixture() {
  const original = openLevel('Stair authoring', originalFiles(1));
  let document = applyCommand(original, editRoomSource(commitRoomDraft(original.roomSource, setDraftCorners(
    beginRoomDraft(original, original.roomSource, 0), [{ x: 20000, z: 20000 }, { x: 20128, z: 20000 }, { x: 20128, z: 20128 }, { x: 20000, z: 20128 }]))));
  const approach = document.roomSource.rooms.at(-1)!, choices = roomAppearanceChoices(document, approach);
  const appearance = { lowerWall: choices.lower[0]!.offset, upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! };
  document = applyCommand(document, editRoomAppearance(document, { ...appearance, room: approach.id }));
  document = applyCommand(document, editRoomSource(setRoomHeights(document, document.roomSource, approach.id,
    { floor: 0, ceiling: -128, upperFloor: choices.template.upperFloor, upperCeiling: choices.template.upperCeiling })));
  document = applyCommand(document, setSourceStart(document, 1, approach.id, 20064, 20064));
  const flight: StairFlight = { start: { x: 20064, z: 20128 }, end: { x: 20064, z: 20640 }, width: 128,
    steps: 4, startFloor: 0, endFloor: -64, clearance: 128, recordZone: 0, appearance };
  return { original, document, approach, flight };
}
