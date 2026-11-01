import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { importedAnimationTables } from '../src/model/animationBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import type { SwitchCopy } from '../src/model/switchCopies';

export function switchCopyFixture() {
  const original = openLevel('Independent switch copy', originalFiles(1), 0, -1), tables = importedAnimationTables(original).tables;
  const source = original.switches.find(item => item.zone >= 0 && original.zones[item.zone]!.lower.concat(original.zones[item.zone]!.upper)
    .some(record => record.offset === item.graphic - 10))!;
  const slot = original.switches.find(item => item.zone < 0)!.id;
  const baseline = openLevel(original.name, { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined,
    { animations: { ...tables, doors: null, lifts: [], switches: tables.switches!.map((record, id) => id === source.id ? record : { ...record, zone: -1 }) } }) }, 0, -1);
  const templates = new Set([source.zone, ...baseline.zones[source.zone]!.boundaryIds.map(id => baseline.lines[id]!.connect).filter(id => id >= 0)]);
  // A neighbouring corridor with no authored wall material cannot acquire a
  // fabricated closure. Retain its adjoining native rooms instead.
  for (const zone of templates) if (!baseline.zones[zone]!.lower.some(record => record.wall))
    baseline.zones[zone]!.boundaryIds.forEach(id => { const target = baseline.lines[id]!.connect; if (target >= 0) templates.add(target); });
  const mapping = new Map<number, string>(), dx = 16000, dz = 16000;
  const added = [...templates].map((zone, index) => {
    const room = baseline.roomSource.rooms.find(room => room.originZone === zone)!;
    const id = `room-${baseline.roomSource.nextId + index}`; mapping.set(zone, id);
    return { ...room, id, originZone: null, vertices: room.vertices.map(point => ({ x: point.x + dx, z: point.z + dz })) };
  });
  const document = { ...baseline, roomSource: { rooms: [...baseline.roomSource.rooms, ...added], nextId: baseline.roomSource.nextId + added.length },
    roomAppearances: added.flatMap(room => {
      const choices = roomAppearanceChoices(baseline, room);
      return choices.lower.length ? [{ room: room.id, lowerWall: choices.lower[0]!.offset, upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! }] : [];
    }) };
  const copy: SwitchCopy = { id: source.id, slot, room: mapping.get(source.zone)!, graphicsRoom: mapping.get(source.zone)! };
  return { document, switchCopy: copy, source: baseline.switches[source.id]!, dx, dz };
}
