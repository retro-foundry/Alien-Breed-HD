import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { switchCopyFixture } from './switchCopyFixture';
import { importedAnimationTables } from '../src/model/animationBuild';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { openLevel } from '../src/model/level';
import { applyCommand, editRoomSource } from '../src/model/commands';
import { setRoomControllerCopies } from '../src/model/movingControllerCopies';
import { setRoomSwitchCopies } from '../src/model/switchCopies';

/** Explicit source circuit: the copied door requires the copied switch.
 * Original controller/switch identities stay independent. */
export function linkedRoomCopyFixture() {
  const moving = movingControllerCopyFixture('door'), switched = switchCopyFixture();
  const tables = importedAnimationTables(moving.document).tables;
  const native = openLevel('Linked room group copy', { ...moving.document.files, graphics: buildGraphicsStreams(moving.document, undefined, undefined,
    { animations: { ...tables, switches: importedAnimationTables(switched.document).tables.switches } }) }, 0, -1);
  const rooms = [...moving.document.roomSource.rooms], appearances = [...moving.document.roomAppearances], mapping = new Map<string, string>();
  let nextId = moving.document.roomSource.nextId;
  for (const room of switched.document.roomSource.rooms.filter(room => room.originZone === null)) {
    let target = rooms.find(item => item.originZone === null && item.recordZone === room.recordZone);
    if (!target) {
      target = { ...room, id: `room-${nextId++}` }; rooms.push(target);
      const appearance = switched.document.roomAppearances.find(item => item.room === room.id);
      if (appearance) appearances.push({ ...appearance, room: target.id });
    }
    mapping.set(room.id, target.id);
  }
  const combined = { ...native, roomSource: { rooms, nextId }, roomAppearances: appearances };
  const switchCopy = { ...switched.switchCopy, room: mapping.get(switched.switchCopy.room)!, graphicsRoom: mapping.get(switched.switchCopy.graphicsRoom)! };
  const motion = { ...moving.copy, header: moving.copy.header!.slice() };
  new DataView(motion.header.buffer).setUint16(14, 1 << (switchCopy.slot + 4));
  const controlled = applyCommand(combined, editRoomSource(setRoomControllerCopies(combined, [motion])));
  const document = applyCommand(controlled, editRoomSource(setRoomSwitchCopies(controlled, [switchCopy])));
  return { document, motion, switchCopy, switchTemplate: document.switches[switchCopy.id]!, ids: rooms.filter(room => room.originZone === null).map(room => room.id) };
}
