import { expect, it } from 'vitest';
import { importedBarrierFixture } from './importedBarrierFixture';
import { roomPortalTargets, setRoomPortal } from '../src/model/roomPortals';
import { roomAppearanceChoices } from '../src/model/roomAppearance';
import { applyCommand, editRoomSource, execute, undo, redo } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { roomSourceMatchesLevel, setRoomProperties, insertSourceCorner } from '../src/model/roomSource';
import { captureRoomProperties } from '../src/model/roomProperties';
import { roomSettings } from '../src/model/roomHeights';
import { encodeProject, decodeProject } from '../src/model/project';
import { copyRooms, pasteRooms } from '../src/model/roomClipboard';
import { visibilityCells } from '../src/model/visibilityImport';
import { importedBarrierLines } from '../src/model/importedBarriers';
import { openLevel } from '../src/model/level';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';
import { encodeWallSurface } from '../src/model/surfaces';
import { removeSourceRooms } from '../src/model/roomSource';
import { editRoomAppearance } from '../src/model/commands';
import { roomHeightFixture, twoStoreyHeightTemplate } from './roomHeightFixture';
import { translateSourceRooms } from '../src/model/roomSource';

it('recognises a raw exported barrier without inventing source overrides and preserves it through unrelated edits', () => {
  const { document, own, other } = importedBarrierFixture();
  expect(roomSourceMatchesLevel(document)).toBe(true); expect(document.roomAppearances).toBeUndefined();
  expect(own.portals).toBeUndefined();
  const target = roomPortalTargets(document, own.id).find(target => target.neighbour === other.id)!;
  expect(target).toMatchObject({ open: true, barrier: true, authored: false });
  const legacy = visibilityCells(document), ownCells = [...legacy.owners].filter(([, id]) => id === own.originZone).map(([id]) => id), otherCells = [...legacy.owners].filter(([, id]) => id === other.originZone).map(([id]) => id);
  expect(legacy.rooms.filter(room => ownCells.includes(room.id)).some(room => room.neighbours.some(id => otherCells.includes(id)))).toBe(true);
  const properties = captureRoomProperties(roomSettings(document, own));
  const edited = applyCommand(document, editRoomSource(setRoomProperties(document.roomSource, own.id, { ...properties, floorSound: properties.floorSound + 1 })));
  const saved = decodeProject(encodeProject(edited)), built = buildPlayableLevel(saved, -1), cell = built.topology!.cells.find(cell => cell.sourceId === own.id)!, destination = built.topology!.cells.find(cell => cell.sourceId === other.id)!;
  expect(cell.neighbours).toContain(destination.id);
  for (const edge of cell.neighbours.flatMap((id, edge) => id === destination.id ? [edge] : [])) expect(built.document.lines[built.document.zones[cell.id]!.boundaryIds[edge]!]!.connect).toBe(-1);
  const original = document.zones[own.recordZone]!.lower.filter(face => face.type === 13);
  const result = built.document.zones[cell.id]!.lower.filter(face => face.type === 13);
  expect(result.map(face => face.wall)).toEqual(original.map(face => face.wall));
});

it('explicitly opens or closes an imported barrier with saved provenance and Undo', () => {
  const { document, own, other } = importedBarrierFixture(), target = roomPortalTargets(document, own.id).find(target => target.neighbour === other.id)!;
  for (const open of [true, false]) {
    const appearances = open ? [] : [own, other].map(room => {
      const choices = roomAppearanceChoices(document, room);
      return { room: room.id, lowerWall: choices.lower[0]!.offset, upperWall: choices.upper[0]?.offset ?? null, lightingPoint: choices.points[0]! };
    });
    const command = setRoomPortal(document, own.id, target.edges[0]!, other.id, open, appearances);
    const history = execute({ document, past: [], future: [] }, command), built = buildPlayableLevel(history.document, -1);
    const cell = built.topology!.cells.find(cell => cell.sourceId === own.id)!, destination = built.topology!.cells.find(cell => cell.sourceId === other.id)!;
    expect(cell.neighbours.includes(destination.id)).toBe(open);
    expect(built.document.zones[cell.id]!.lower.some(face => face.type === 13)).toBe(false);
    expect(undo(history).document).toEqual(document); expect(redo(undo(history)).document.roomSource).toEqual(history.document.roomSource);
    expect(buildPlayableLevel(decodeProject(encodeProject(history.document)), -1).document.files).toEqual(built.document.files);
  }
});

it('retains imported barrier mappings across inserted corners and independent group copies', () => {
  const { document, own, other } = importedBarrierFixture(), target = roomPortalTargets(document, own.id).find(target => target.neighbour === other.id)!;
  const index = target.edges[0]!, a = own.vertices[index]!, b = own.vertices[(index + 1) % own.vertices.length]!;
  const split = { ...document, roomSource: insertSourceCorner(document.roomSource, own.id, index, { x: (a.x + b.x) / 2, z: (a.z + b.z) / 2 }).after };
  expect(() => buildPlayableLevel(split, -1)).not.toThrow();
  const proposal = pasteRooms(document, copyRooms(document, [own.id, other.id]), 20000, 20000), copied = applyCommand(document, proposal.command);
  expect(roomPortalTargets(copied, proposal.ids[0]!).find(target => target.neighbour === proposal.ids[1])).toMatchObject({ barrier: true });
  expect(importedBarrierLines(document).size).toBeGreaterThan(0);
});

it('preserves the native one-sided collision contract without adding a reciprocal wall', () => {
  const { document, own, other } = importedBarrierFixture();
  const zone = document.zones[other.recordZone]!;
  const graphics = buildGraphicsStreams(document, undefined, undefined, { streams: new Map([[zone.id, {
    lower: zone.lower.filter(record => record.type !== 13).map(record => ({ sourceOffset: record.offset, bytes: document.files.graphics.slice(record.offset, record.offset + record.size) })),
    upper: zone.upper.map(record => ({ sourceOffset: record.offset, bytes: document.files.graphics.slice(record.offset, record.offset + record.size) })),
  }]]) });
  const data = document.files.data.slice(), view = new DataView(data.buffer);
  const target = roomPortalTargets(document, own.id).find(target => target.neighbour === other.id)!;
  for (const edge of target.otherEdges) view.setInt16(document.lines[other.wallSources[edge]!]!.offset + 8, own.originZone!);
  const raw = openLevel('One-sided native barrier', { ...document.files, data, graphics });
  const props = captureRoomProperties(roomSettings(raw, raw.roomSource.rooms[own.recordZone]!));
  const rebuilt = buildPlayableLevel({ ...raw, roomSource: setRoomProperties(raw.roomSource, own.id, { ...props, floorSound: props.floorSound + 1 }).after }, -1);
  const ownCell = rebuilt.topology!.cells.find(cell => cell.sourceId === own.id)!, otherCell = rebuilt.topology!.cells.find(cell => cell.sourceId === other.id)!;
  const ownZone = rebuilt.document.zones[ownCell.id]!, otherZone = rebuilt.document.zones[otherCell.id]!;
  expect(ownCell.neighbours).toContain(otherCell.id); expect(otherCell.neighbours).toContain(ownCell.id);
  expect(ownZone.boundaryIds.some(id => rebuilt.document.lines[id]!.connect === otherCell.id)).toBe(false);
  expect(otherZone.boundaryIds.some(id => rebuilt.document.lines[id]!.connect === ownCell.id)).toBe(true);
  expect(otherZone.lower.some(face => face.type === 13)).toBe(false);
});

it('does not turn partial decorative see-through strips into inferred full-room portals', () => {
  const { document, own, other } = importedBarrierFixture();
  const replacements = new Map(document.zones.flatMap(zone => zone.lower.filter(face => face.type === 13).map(face => [face.offset,
    encodeWallSurface({ ...face.wall!, topFixed: Math.trunc((face.wall!.topFixed + face.wall!.bottomFixed) / 2) })] as const)));
  const raw = openLevel('Partial decorative strips', { ...document.files, graphics: buildGraphicsStreams(document, undefined, undefined, { replacements }) });
  expect(importedBarrierLines(raw).size).toBe(0);
  expect(roomPortalTargets(raw, own.id).find(target => target.neighbour === other.id)).toMatchObject({ open: false, barrier: false });
});

it('repairs a retained native barrier to a solid wall after its adjoining room is deleted', () => {
  const { document, own, other } = importedBarrierFixture();
  const removed = { ...document, roomSource: removeSourceRooms(document.roomSource, [other.id]).after };
  expect(() => buildPlayableLevel(removed, -1)).toThrow(/wall style/);
  const choices = roomAppearanceChoices(removed, own);
  const fixed = applyCommand(removed, editRoomAppearance(removed, { room: own.id, lowerWall: choices.lower[0]!.offset, upperWall: null, lightingPoint: choices.points[0]! }));
  const built = buildPlayableLevel(fixed, -1), cell = built.topology!.cells.find(cell => cell.sourceId === own.id)!;
  expect(built.document.zones[cell.id]!.lower.some(face => face.type === 13)).toBe(false);
});

it('preserves independent lower and upper native barrier records through a raw import and rebuild', () => {
  const fixture = roomHeightFixture(twoStoreyHeightTemplate());
  const joined = { ...fixture.document, roomSource: translateSourceRooms(fixture.document.roomSource, [fixture.to.id], -744, 0).after };
  const target = roomPortalTargets(joined, fixture.from.id).find(target => target.neighbour === fixture.to.id)!;
  const barrier = applyCommand(joined, setRoomPortal(joined, fixture.from.id, target.edges[0]!, fixture.to.id, 'barrier'));
  const initial = buildPlayableLevel(barrier, -1), raw = initial.document, runtime = initial.topology!.cells.find(cell => cell.sourceId === fixture.from.id)!;
  const room = raw.roomSource.rooms.find(room => room.originZone === runtime.id)!, props = captureRoomProperties(roomSettings(raw, room));
  const edited = { ...raw, roomSource: setRoomProperties(raw.roomSource, room.id, { ...props, floorSound: props.floorSound + 1 }).after };
  const built = buildPlayableLevel(edited, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  for (const layer of ['lower', 'upper'] as const) {
    const original = raw.zones[runtime.id]![layer].filter(face => face.type === 13).map(face => face.wall);
    expect(original.length).toBeGreaterThan(0);
    expect(built.document.zones[cell.id]![layer].filter(face => face.type === 13).map(face => face.wall)).toEqual(original);
  }
});

it('closes a native barrier opaquely even when its chosen texture style comes from a see-through face', () => {
  const { document, own, other } = importedBarrierFixture(), target = roomPortalTargets(document, own.id).find(target => target.neighbour === other.id)!;
  const appearances = [own, other].map(room => {
    const face = document.zones[room.recordZone]!.lower.find(face => face.type === 13)!;
    return { room: room.id, lowerWall: face.offset, upperWall: null, lightingPoint: face.points[0]! };
  });
  const closed = applyCommand(document, setRoomPortal(document, own.id, target.edges[0]!, other.id, false, appearances));
  const built = buildPlayableLevel(closed, -1), cell = built.topology!.cells.find(cell => cell.sourceId === own.id)!;
  expect(built.document.zones[cell.id]!.lower.some(face => face.type === 13)).toBe(false);
});
