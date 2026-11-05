import { expect, it } from 'vitest';
import { roomHeightFixture } from './roomHeightFixture';
import { roomConnectionEdges } from '../src/model/roomConnections';
import { translateSourceRooms, beginRoomDraft, setDraftCorners, commitRoomDraft } from '../src/model/roomSource';

it('classifies a newly adjoining room and a partial shared edge as connections', () => {
  const fixture = roomHeightFixture(), roomSource = translateSourceRooms(fixture.document.roomSource, [fixture.to.id], -744, 0).after;
  const document = { ...fixture.document, roomSource };
  const whole = roomConnectionEdges(document).filter(edge => edge.room === fixture.from.id && edge.neighbour === fixture.to.id);
  expect(whole).toHaveLength(1);
  expect([whole[0]!.start.x, whole[0]!.end.x]).toEqual([20256, 20256]);
  const draft = setDraftCorners(beginRoomDraft(document, roomSource, fixture.to.recordZone, fixture.to.id),
    [{ x: 20256, z: 20064 }, { x: 20512, z: 20064 }, { x: 20512, z: 20192 }, { x: 20256, z: 20192 }]);
  const partial = roomConnectionEdges({ ...document, roomSource: commitRoomDraft(roomSource, draft).after })
    .filter(edge => edge.room === fixture.from.id && edge.neighbour === fixture.to.id);
  expect(partial).toHaveLength(1);
  expect([partial[0]!.start.z, partial[0]!.end.z].sort()).toEqual([20064, 20192]);
});

it('does not label corner contacts or internal convex seams as zone connections', () => {
  const fixture = roomHeightFixture(), roomSource = translateSourceRooms(fixture.document.roomSource, [fixture.to.id], -744, 256).after;
  expect(roomConnectionEdges({ ...fixture.document, roomSource }).filter(edge => edge.room === fixture.from.id && edge.neighbour === fixture.to.id)).toHaveLength(0);
  const draft = setDraftCorners(beginRoomDraft(fixture.document, roomSource, fixture.from.recordZone, fixture.from.id),
    [{ x: 20000, z: 20000 }, { x: 20512, z: 20000 }, { x: 20512, z: 20128 }, { x: 20128, z: 20128 }, { x: 20128, z: 20512 }, { x: 20000, z: 20512 }]);
  const document = { ...fixture.document, roomSource: commitRoomDraft(roomSource, draft).after };
  const edges = roomConnectionEdges(document).filter(edge => edge.room === fixture.from.id);
  expect(edges).toHaveLength(6); expect(edges.every(edge => edge.neighbour === null)).toBe(true);
});
