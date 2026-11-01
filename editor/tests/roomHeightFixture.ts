import { applyCommand, editRoomAppearance, editRoomSource, setSourceStart } from '../src/model/commands';
import { openLevel } from '../src/model/level';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../src/model/roomSource';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { originalFiles } from './helpers';

export function roomHeightFixture(recordZone = 0) {
  const original = openLevel('Room height authoring', originalFiles(1)); let document = original;
  for (const x of [20000, 21000]) {
    document = applyCommand(document, editRoomSource(commitRoomDraft(document.roomSource, setDraftCorners(
      beginRoomDraft(document, document.roomSource, recordZone), [{ x, z: 20000 }, { x: x + 256, z: 20000 }, { x: x + 256, z: 20256 }, { x, z: 20256 }]))));
    const room = document.roomSource.rooms.at(-1)!, choices = roomAppearanceChoices(document, room);
    document = applyCommand(document, editRoomAppearance(document, { room: room.id, lowerWall: choices.lower[0]!.offset,
      upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! }));
  }
  const from = document.roomSource.rooms.at(-2)!, to = document.roomSource.rooms.at(-1)!;
  document = applyCommand(document, setSourceStart(document, 1, from.id, 20128, 20128));
  return { original, document, from, to };
}

export function twoStoreyHeightTemplate() {
  const original = openLevel('upper template', originalFiles(1));
  const key = (ids: readonly number[]) => ids.map(id => `${original.points[id]!.x},${original.points[id]!.z}`).sort().join(';');
  const zone = original.zones.find(zone => {
    const outline = zone.boundaryIds.map(id => { const line = original.lines[id]!; return `${line.x},${line.z}`; }).sort().join(';');
    return zone.upperFloor > zone.upperCeiling && zone.upper.some(record => record.wall)
      && [...zone.lower, ...zone.upper].every(record => record.wall || record.points.length === 0 || (record.polygon && key(record.points) === outline));
  });
  if (!zone) throw new Error('No complete two-storey height template was found in the shipped level.');
  return zone.id;
}
