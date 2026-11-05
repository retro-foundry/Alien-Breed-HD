import { closedPortalFixture } from './portalFixture';
import { roomPortalTargets, setRoomPortal } from '../src/model/roomPortals';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { applyCommand } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';

/** Raw native export: no source portal overrides or appearance recipes. */
export function importedBarrierFixture() {
  const fixture = closedPortalFixture(), source = fixture.baseline, a = fixture.owner.id, b = fixture.removed.id;
  const target = roomPortalTargets(source, a).find(target => target.neighbour === b)!;
  const appearances = [a, b].map(id => {
    const room = source.roomSource.rooms.find(room => room.id === id)!, choices = roomAppearanceChoices(source, room);
    return { room: id, lowerWall: choices.lower[0]!.offset, upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! };
  });
  const barrier = applyCommand(source, setRoomPortal(source, a, target.edges[0]!, b, 'barrier', appearances));
  const built = buildPlayableLevel(barrier, -1), own = built.topology!.cells.find(cell => cell.sourceId === a)!, other = built.topology!.cells.find(cell => cell.sourceId === b)!;
  const document = built.document;
  return { ...fixture, source, document, own: document.roomSource.rooms.find(room => room.originZone === own.id)!, other: document.roomSource.rooms.find(room => room.originZone === other.id)! };
}
