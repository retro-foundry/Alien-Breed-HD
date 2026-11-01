import { checkedInteger } from './binary';
import type { LevelDocument } from './level';
import { roomSettings } from './roomHeights';
import { setRoomHeights, type RoomSource } from './roomSource';

/** Headroom is the existing native floor-minus-ceiling distance. A positive
 * user-facing size moves only the selected ceiling, using the same plane
 * validation as direct height editing; it adds no new stored fields. */
export function setRoomHeadroom(document: Pick<LevelDocument, 'zones'>, current: RoomSource, id: string, upper: boolean, height: number) {
  checkedInteger(height * 4, 1, 65535, 'Room headroom in quarter units');
  const room = current.rooms.find(room => room.id === id);
  if (!room) throw new Error('Select an existing room before changing its headroom.');
  const settings = roomSettings(document, room);
  if (upper && (!settings.upper.length || settings.upperFloor <= settings.upperCeiling)) throw new Error('This room has no upper storey. Edit its lower room height.');
  const { floor, ceiling, upperFloor, upperCeiling } = settings;
  const change = setRoomHeights(document, current, id, { floor, ceiling, upperFloor, upperCeiling,
    [upper ? 'upperCeiling' : 'ceiling']: settings[upper ? 'upperFloor' : 'floor'] - height });
  return { ...change, label: `Change ${upper ? 'upper' : 'lower'} room headroom` };
}
