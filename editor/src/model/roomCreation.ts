import type { LevelDocument } from './level';
import { commitRoomDraft, type RoomDraft } from './roomSource';
import { defaultRoomAppearance, setRoomAppearance } from './roomAppearance';
import { ceilingIsSky } from './roomStoreys';
import { roomSettings } from './roomHeights';
import type { EditCommand } from './commands';

/** User-facing room creation includes actual saved materials in the same
 * transaction as its outline. Redraws retain all authored material choices. */
export function finishRoomDraft(document: LevelDocument, draft: RoomDraft): EditCommand {
  const change = commitRoomDraft(document.roomSource, draft);
  if (draft.roomId !== null) return { label: change.label, patches: [], roomSource: { before: change.before, after: change.after } };
  const id = `room-${document.roomSource.nextId}`, created = change.after.rooms.find(room => room.id === id)!;
  const settings = roomSettings(document, created), upper = settings.upper.length > 0 && settings.upperFloor > settings.upperCeiling;
  const sky = [!upper && ceilingIsSky(document, created, false), upper && ceilingIsSky(document, created, true)];
  const room = sky.some(Boolean) ? Object.freeze({ ...created, sky: Object.freeze(sky) }) : created;
  const after = room === created ? change.after : Object.freeze({ ...change.after, rooms: Object.freeze(change.after.rooms.map(item => item.id === id ? room : item)) });
  const source = { ...document, roomSource: after };
  const appearances = setRoomAppearance(source, defaultRoomAppearance(source, room));
  return { label: 'Create textured room', patches: [], roomSource: { before: document.roomSource, after }, roomAppearances: { before: document.roomAppearances, after: appearances } };
}
