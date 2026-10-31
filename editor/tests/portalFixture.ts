import { applyCommand, editRoomAppearance, editRoomSource, removeObject, setExitZone, setSourceStart } from '../src/model/commands';
import { pointInRoomOutline } from '../src/model/geometry';
import { openLevel } from '../src/model/level';
import { resolveAnimationRoom } from '../src/model/animationRooms';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { beginRoomDraft, commitRoomDraft, removeSourceRooms, setDraftCorners } from '../src/model/roomSource';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { originalFiles } from './helpers';

/** An actual shipped solid wall with empty authoring space outside it. Keep
 * the original level intact; only actors in the controlled test room are
 * removed so gameplay checks isolate the wall rather than enemy collisions. */
export function importedPortalFixture() {
  const original = openLevel('Imported doorway source', originalFiles(1)), template = original.zones[0]!;
  const moving = new Set(original.mechanisms.map(item => resolveAnimationRoom(original, item.zone, item.kind === 'lift')));
  const candidates = original.roomSource.rooms.flatMap(room => room.vertices.flatMap((a, edge) => {
    const b = room.vertices[(edge + 1) % room.vertices.length]!, zone = original.zones.find(zone => zone.id === room.originZone)!;
    const sourceLine = room.wallSources[edge];
    const dx = b.x - a.x, dz = b.z - a.z, length = Math.hypot(dx, dz);
    if (room.vertices.length > 8 || moving.has(zone.id) || zone.floor !== template.floor || zone.ceiling > -96
      || zone.upper.length || (dx !== 0 && dz !== 0) || length < 192 || sourceLine == null || original.lines[sourceLine]!.connect !== -1) return [];
    const x = Math.round((a.x + b.x) / 2), z = Math.round((a.z + b.z) / 2);
    const normal = { x: -Math.sign(dz), z: Math.sign(dx) }, tangent = { x: Math.sign(dx), z: Math.sign(dz) };
    const start = { x: x - normal.x * 64, z: z - normal.z * 64 }, axis: 'x' | 'z' = dx ? 'x' : 'z';
    if (!pointInRoomOutline(room.vertices, start)) return [];
    const material = zone.lower.find(record => record.wall && record.type === 0 && record.top! <= zone.ceiling && record.bottom! >= zone.floor
      && record.points.every(id => (original.points[id]!.x - a.x) * dz === (original.points[id]!.z - a.z) * dx)
      && Math.min(...record.points.map(id => original.points[id]![axis])) <= (axis === 'x' ? x : z) - 64
      && Math.max(...record.points.map(id => original.points[id]![axis])) >= (axis === 'x' ? x : z) + 64);
    return material ? [{ room, zone, edge, sourceLine, material, start, x, z, normal, tangent, axis }] : [];
  })).toSorted((a, b) => b.z - a.z);
  for (const candidate of candidates) {
    let change;
    try {
      const point = (along: number, outside: number) => ({ x: candidate.x + candidate.tangent.x * along + candidate.normal.x * outside,
        z: candidate.z + candidate.tangent.z * along + candidate.normal.z * outside });
      change = commitRoomDraft(original.roomSource, setDraftCorners(beginRoomDraft(original, original.roomSource, 0),
        [point(-64, 0), point(64, 0), point(64, 256), point(-64, 256)]));
    } catch { continue; } // Some shipped closed walls back onto another room.
    let baseline = applyCommand(original, setExitZone(original, -1));
    for (const object of baseline.objects.filter(object => object.zone === candidate.zone.id && ![5, 11].includes(object.type)))
      baseline = applyCommand(baseline, removeObject(baseline, object.id));
    baseline = applyCommand(baseline, setSourceStart(baseline, 1, candidate.room.id, candidate.start.x, candidate.start.z));
    const source = applyCommand(baseline, editRoomSource(change)), added = change.after.rooms.at(-1)!;
    const choices = roomAppearanceChoices(source, added);
    const document = applyCommand(source, editRoomAppearance(source, { room: added.id,
      lowerWall: choices.lower[0]!.offset, upperWall: null, lightingPoint: choices.points[0]! }));
    return { ...candidate, original, baseline: { ...baseline, name: 'Closed doorway reference' }, document, added };
  }
  throw new Error('No isolated static shipped-wall fixture was found for a partial doorway.');
}

/** Re-import a generated, playable portal as ordinary level data, then delete
 * its empty adjoining room. This exercises closing an imported doorway rather
 * than merely undoing construction of a source-only room. */
export function closedPortalFixture() {
  const fixture = importedPortalFixture(), built = buildPlayableLevel(fixture.document, -1);
  const baseline = built.document, destination = built.topology!.cells.find(cell => cell.sourceId === fixture.added.id)!;
  const removed = baseline.roomSource.rooms.find(room => room.originZone === destination.id)!;
  const owner = baseline.roomSource.rooms.find(room => room.originZone === baseline.starts[0]!.zone)!;
  const document = applyCommand(baseline, editRoomSource(removeSourceRooms(baseline.roomSource, [removed.id])));
  const choices = roomAppearanceChoices(document, owner);
  const appearance = { room: owner.id, lowerWall: choices.lower[0]!.offset, upperWall: null, lightingPoint: choices.points[0]! };
  return { ...fixture, baseline, removed, owner, document, appearance };
}

export function replacementPortalFixture() {
  const fixture = closedPortalFixture();
  const change = commitRoomDraft(fixture.document.roomSource, setDraftCorners(
    beginRoomDraft(fixture.document, fixture.document.roomSource, fixture.removed.recordZone), fixture.removed.vertices));
  let document = applyCommand(fixture.document, editRoomSource(change));
  const replacement = document.roomSource.rooms.at(-1)!, choices = roomAppearanceChoices(document, replacement);
  document = applyCommand(document, editRoomAppearance(document, { room: replacement.id, lowerWall: choices.lower[0]!.offset,
    upperWall: null, lightingPoint: choices.points[0]! }));
  return { ...fixture, document, replacement };
}
