import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { applyCommand, editRoomSource, setSourceStart } from '../src/model/commands';
import { finishRoomDraft } from '../src/model/roomCreation';
import { beginRoomDraft, setDraftCorners } from '../src/model/roomSource';
import { availableSwitchSlots, setSourceSwitch } from '../src/model/sourceSwitches';
import { authoredMovingFixture } from './authoredMovingFixture';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { setSourceMovingRoom } from '../src/model/sourceMovingRoom';

export function switchRoomFixture(original = openLevel('New wall switch', originalFiles(1))) {
  const document = applyCommand(original, finishRoomDraft(original, setDraftCorners(beginRoomDraft(original, original.roomSource, 0), [
    { x: 20000, z: 20000 }, { x: 20256, z: 20000 }, { x: 20256, z: 20256 }, { x: 20000, z: 20256 },
  ])));
  const room = document.roomSource.rooms.at(-1)!;
  return { document, room, slot: availableSwitchSlots(document)[0]! };
}

export function switchDoorFixture() {
  const fixture = authoredMovingFixture('door'), slot = availableSwitchSlots(fixture.document)[0]!;
  let document = applyCommand(fixture.document, editRoomSource(setSourceSwitch(fixture.document, fixture.neighbour.id, slot, { edge: 2, automatic: false })));
  const header = fixture.input.header.slice(), view = new DataView(header.buffer);
  view.setInt16(4, view.getInt16(0)); view.setUint16(14, 1 << (slot + 4));
  view.setUint8(16, 4); view.setUint8(17, 1);
  document = applyCommand(document, editRoomSource(setSourceMovingRoom(document, fixture.room.id, { kind: 'door', header: [...header], wall: roomAppearanceChoices(document, fixture.room).lower[0]!.offset })));
  document = applyCommand(document, setSourceStart(document, 1, fixture.neighbour.id, 20480, 20128));
  return { ...fixture, document, slot };
}
