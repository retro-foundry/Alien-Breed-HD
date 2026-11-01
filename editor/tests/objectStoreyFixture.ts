import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { applyCommand, editRoomSource, editRoomAppearance } from '../src/model/commands';
import { beginRoomDraft, commitRoomDraft, setDraftCorners } from '../src/model/roomSource';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { createObject, objectTemplates, floorMatchedObjectHeight } from '../src/model/objectAuthoring';

export function objectStoreyFixture() {
  const fixture = roomHeightFixture(twoStoreyHeightTemplate()); let document = fixture.document;
  document = applyCommand(document, editRoomSource(commitRoomDraft(document.roomSource, setDraftCorners(beginRoomDraft(document, document.roomSource, 0),
    [{ x: 20744, z: 20000 }, { x: 21000, z: 20000 }, { x: 21000, z: 20256 }, { x: 20744, z: 20256 }]))));
  const lower = document.roomSource.rooms.at(-1)!, choices = roomAppearanceChoices(document, lower);
  document = applyCommand(document, editRoomAppearance(document, { room: lower.id, lowerWall: choices.lower[0]!.offset, upperWall: null, lightingPoint: choices.points[0]! }));
  // The one-storey candidate occurs first in source order and shares an edge
  // with the two-storey destination. Its interior does not overlap that room.
  document = { ...document, roomSource: { ...document.roomSource, rooms: [...document.roomSource.rooms.filter(room => room.id !== lower.id && room.id !== fixture.to.id), lower, fixture.to] } };
  const template = objectTemplates(document).find(object => object.type === 1)!, ids: number[] = [];
  for (const z of [20128, 20192]) {
    const proposal = createObject(document, { template: template.id, room: fixture.from.id, x: 20128, z,
      upper: true, y: floorMatchedObjectHeight(document, template, fixture.from.id, true) });
    document = applyCommand(document, proposal.command); ids.push(proposal.id);
  }
  return { ...fixture, document, lower, ids };
}
