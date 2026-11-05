import { roomHeightFixture } from './roomHeightFixture';
import { applyCommand, editRoomSource, setSourceStart } from '../src/model/commands';
import { translateSourceRooms, setRoomHeights } from '../src/model/roomSource';
import { roomSettings } from '../src/model/roomHeights';
import { roomAppearanceChoices, roomAppearanceBuildOptions } from '../src/model/roomAppearance';
import { importedAnimationTables } from '../src/model/animationBuild';
import { buildSourceLevel } from '../src/model/sourceBuild';
import type { AuthoredMovingRoom } from '../src/model/authoredMovingRoom';
import type { LevelDocument } from '../src/model/level';

export function authoredMovingFixture(kind: 'door' | 'lift', original?: LevelDocument) {
  const fixture = roomHeightFixture(0, original);
  let document = applyCommand(fixture.document, editRoomSource(translateSourceRooms(fixture.document.roomSource, [fixture.to.id], -744, 0)));
  const settings = roomSettings(document, fixture.from);
  if (kind === 'lift') document = applyCommand(document, editRoomSource(setRoomHeights(document, document.roomSource, fixture.from.id,
    { floor: settings.floor + 32, ceiling: settings.ceiling, upperFloor: settings.upperFloor, upperCeiling: settings.upperCeiling })));
  document = applyCommand(document, setSourceStart(document, 1, fixture.to.id, 20384, 20128));
  const room = document.roomSource.rooms.find(room => room.id === fixture.from.id)!;
  const currentSettings = roomSettings(document, room), current = currentSettings[kind === 'door' ? 'ceiling' : 'floor'];
  const tables = importedAnimationTables(document).tables, native = (kind === 'door' ? tables.doors : tables.lifts)![0]!;
  const header = native.header.slice(), view = new DataView(header.buffer);
  view.setInt16(0, (kind === 'door' ? settings.floor : current) * 4);
  view.setInt16(2, (kind === 'door' ? current : settings.floor - 32) * 4);
  view.setInt16(4, current * 4); view.setInt16(6, 0); view.setUint16(14, 0);
  view.setUint8(16, kind === 'door' ? 4 : 2); view.setUint8(17, kind === 'door' ? 0 : 2);
  const wall = roomAppearanceChoices(document, room).lower[0]!.wall!;
  const [a, b] = wall.points.map(point => document.points[point]!);
  const input: AuthoredMovingRoom = { room: room.id, kind, header, wall: { surface: wall, length: Math.hypot(b!.x - a!.x, b!.z - a!.z) } };
  const options = { resolvedExit: -1, ...roomAppearanceBuildOptions(document), authoredControllers: [input] };
  const built = buildSourceLevel(document, options), mechanism = built.document.mechanisms.filter(item => item.kind === kind).at(-1)!;
  return { document, room, neighbour: fixture.to, input, options, built, mechanism };
}
