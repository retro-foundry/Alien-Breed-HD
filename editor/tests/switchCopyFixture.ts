import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { importedAnimationTables } from '../src/model/animationBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { setRoomSwitchCopies, type SwitchCopy } from '../src/model/switchCopies';
import { applyCommand, editRoomSource, setSourceStart } from '../src/model/commands';
import { pointInRoomOutline } from '../src/model/geometry';

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

export function switchCopyPlayerFixture() {
  const fixture = switchCopyFixture(), { switchCopy: copy, source, dx, dz } = fixture;
  const room = fixture.document.roomSource.rooms.find(room => room.id === copy.room)!;
  const a = fixture.document.points[source.point]!, b = fixture.document.points[source.point + 1]!;
  const centre = { x: Math.floor((a.x + b.x) / 2) + dx, z: Math.floor((a.z + b.z) / 2) + dz };
  const positions = Array.from({ length: 13 }, (_, x) => Array.from({ length: 13 }, (_, z) => ({ x: centre.x + (x - 6) * 8, z: centre.z + (z - 6) * 8 }))).flat()
    .filter(point => Math.hypot(point.x - centre.x, point.z - centre.z) < 55 && pointInRoomOutline(room.vertices, point));
  const clearance = (point: typeof centre) => Math.min(...room.vertices.map((a, index) => {
    const b = room.vertices[(index + 1) % room.vertices.length]!, x = b.x - a.x, z = b.z - a.z;
    const t = Math.max(0, Math.min(1, ((point.x - a.x) * x + (point.z - a.z) * z) / (x * x + z * z)));
    return Math.hypot(point.x - a.x - t * x, point.z - a.z - t * z);
  }));
  const position = positions.sort((a, b) => clearance(b) - clearance(a))[0];
  if (!position) throw new Error('The original switch fixture has no valid nearby player start.');
  const captured = applyCommand(fixture.document, editRoomSource(setRoomSwitchCopies(fixture.document, [copy])));
  return { ...fixture, document: applyCommand(captured, setSourceStart(captured, 1, copy.room, position.x, position.z)) };
}
