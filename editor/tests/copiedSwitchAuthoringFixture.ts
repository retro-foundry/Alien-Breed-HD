import { linkedRoomCopyFixture } from './linkedRoomCopyFixture';
import { switchRoomFixture } from './sourceSwitchFixture';
import { applyCommand, setSourceStart } from '../src/model/commands';

export function copiedSwitchAuthoringFixture() {
  const fixture = linkedRoomCopyFixture(), fresh = switchRoomFixture(fixture.document);
  return { ...fixture, room: fresh.room, document: applyCommand(fresh.document, setSourceStart(fresh.document, 1, fresh.room.id, 20032, 20128)) };
}
