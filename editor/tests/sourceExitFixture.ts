import { applyCommand, editRoomSource } from '../src/model/commands';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../src/model/roomSource';
import { roomHeightFixture } from './roomHeightFixture';

export function sourceExitFixture() {
  const fixture = roomHeightFixture();
  const change = commitRoomDraft(fixture.document.roomSource, setDraftCorners(beginRoomDraft(fixture.document, fixture.document.roomSource, fixture.from.recordZone, fixture.from.id),
    [{ x: 20000, z: 20000 }, { x: 20400, z: 20000 }, { x: 20400, z: 20200 }, { x: 20200, z: 20200 }, { x: 20200, z: 20400 }, { x: 20000, z: 20400 }]));
  return { ...fixture, document: applyCommand(fixture.document, editRoomSource(change)) };
}
