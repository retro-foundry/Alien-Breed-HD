import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { authoredMovingFixture } from './authoredMovingFixture';
import { applyCommand, editRoomSource, setSourceStart } from '../src/model/commands';
import { setRoomControllerCopies } from '../src/model/movingControllerCopies';
import { setSourceMovingRoom } from '../src/model/sourceMovingRoom';

export function mechanismTransferFixture(kind: 'door' | 'lift') {
  const copied = movingControllerCopyFixture(kind);
  const baseline = applyCommand(copied.document, editRoomSource(setRoomControllerCopies(copied.document, [copied.copy])));
  const fixture = authoredMovingFixture(kind, baseline);
  let document = applyCommand(fixture.document, editRoomSource(setSourceMovingRoom(fixture.document, fixture.room.id, { kind, wall: fixture.document.roomAppearances!.find(item => item.room === fixture.room.id)!.lowerWall, header: [...fixture.input.header] })));
  document = applyCommand(document, setSourceStart(document, 1, fixture.room.id, 20128, 20128));
  return { ...fixture, document, copied };
}
