import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { importedAnimationTables } from '../src/model/animationBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { mechanismDependencies } from '../src/model/mechanismDependencies';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import type { MovingControllerCopy } from '../src/model/movingControllerCopies';

export function movingControllerCopyFixture(kind: 'door' | 'lift') {
  const original = openLevel('Independent moving copy', originalFiles(1), 0, -1), tables = importedAnimationTables(original).tables;
  const selected = original.mechanisms.find(item => item.kind === kind && item.current === item.bottom && item.top + 8.25 < item.bottom && item.walls.length)!;
  const record = tables[kind === 'door' ? 'doors' : 'lifts']![selected.id]!;
  const header = record.header.slice(), view = new DataView(header.buffer);
  view.setInt16(6, 0); view.setUint16(14, 0); view.setUint8(16, kind === 'door' ? 5 : 3); view.setUint8(17, kind === 'door' ? 1 : 3);
  const baseline = openLevel(original.name, { ...original.files, graphics: buildGraphicsStreams(original, undefined, undefined,
    { animations: { ...tables, doors: kind === 'door' ? [{ ...record, header }] : null,
      lifts: kind === 'lift' ? [{ ...record, header }] : [], switches: null } }) }, 0, -1);
  const mechanism = baseline.mechanisms.find(item => item.kind === kind)!, dependencies = mechanismDependencies(baseline, kind, mechanism.id);
  const templates = new Set([...dependencies.graphicsRooms, ...dependencies.walls.flatMap(wall => wall.collisionRooms),
    ...baseline.zones[dependencies.room]!.boundaryIds.map(id => baseline.lines[id]!.connect).filter(id => id >= 0)]);
  const mapping = new Map<number, string>();
  const added = [...templates].map((zone, index) => {
    const room = baseline.roomSource.rooms.find(room => room.originZone === zone)!;
    const id = `room-${baseline.roomSource.nextId + index}`; mapping.set(zone, id);
    return { ...room, id, originZone: null, vertices: room.vertices.map(point => ({ x: point.x + 16000, z: point.z + 16000 })) };
  });
  const document = { ...baseline, roomSource: { rooms: [...baseline.roomSource.rooms, ...added], nextId: baseline.roomSource.nextId + added.length },
    roomAppearances: added.flatMap(room => {
      const choices = roomAppearanceChoices(baseline, room);
      return choices.lower.length ? [{ room: room.id, lowerWall: choices.lower[0]!.offset, upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! }] : [];
    }) };
  const copiedHeader = header.slice(), copiedView = new DataView(copiedHeader.buffer);
  copiedView.setInt16(2, (mechanism.top + 8.25) * 4); copiedView.setUint8(16, kind === 'door' ? 4 : 2);
  const copy: MovingControllerCopy = { kind, id: mechanism.id, room: mapping.get(dependencies.room)!, graphicsRooms: mapping,
    collisionRooms: new Map(dependencies.walls.map(wall => [wall.line, mapping.get(wall.collisionRooms[0]!)!])), header: copiedHeader };
  return { document, copy, mechanism, added };
}
