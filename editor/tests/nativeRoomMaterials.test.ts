import { expect, it } from 'vitest';
import { nativeRoomCopyFixture, nativePartialRoomCopyFixture } from './nativeRoomCopyFixture';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { resolveAnimationRoom } from '../src/model/animationRooms';
import { resolvePointLighting } from '../src/model/pointLighting';
import { openLevel } from '../src/model/level';
import { buildGraphicsStreams, importedGraphicsStreams } from '../src/model/graphicsBuild';
import { encodePolygonSurface } from '../src/model/surfaces';

it('compiles a new native-material room identity through saves without losing original walls or lighting provenance', () => {
  const { document, original, copy } = nativeRoomCopyFixture();
  const saved = decodeProject(encodeProject(document));
  const built = buildPlayableLevel(saved, -1);
  const cell = built.topology!.cells.find(cell => cell.sourceId === copy.id)!;
  expect(cell.originZone).toBeNull(); expect(cell.recordZone).toBe(original.recordZone);
  expect(saved.roomSource.rooms.find(room => room.id === copy.id)!.wallSources).toEqual(original.wallSources);
  expect(built.document.zones[cell.id]!.lower.filter(record => record.wall).length).toBeGreaterThanOrEqual(document.zones[original.recordZone]!.lower.filter(record => record.wall).length);
  expect(built.document.starts[0]).toEqual(document.starts[0]);
  expect(saved.files).toEqual(document.files);
});

it('translates a partial floor footprint and its exact native lighting roles without expanding it across the copied room', () => {
  const { document, copy, dx, dz, source, polygon } = nativePartialRoomCopyFixture(), points = polygon.points;
  const result = buildPlayableLevel(document, -1), cell = result.topology!.cells.find(cell => cell.sourceId === copy.id)!;
  const compiled = result.document.zones[cell.id]!.lower.find(record => record.type === 1)!.polygon!;
  expect(compiled.points.map(id => result.document.points[id])).toEqual(points.map(id => expect.objectContaining({ x: source.points[id]!.x + dx, z: source.points[id]!.z + dz })));
  expect(compiled).toMatchObject({ scale: polygon.scale, texture: polygon.texture, brightness: polygon.brightness, heightQuarter: polygon.heightQuarter });
  compiled.points.forEach((id, index) => expect(resolvePointLighting(result.document, { kind: 'copy', source: id })).toEqual(resolvePointLighting(source, { kind: 'copy', source: points[index]! })));
});

it('rejects native-material copies with moving controllers or switches instead of silently making inert copies', () => {
  const base = nativeRoomCopyFixture().document;
  const moving = base.mechanisms[0]!, zone = resolveAnimationRoom(base, moving.zone, moving.kind === 'lift')!;
  const animated = nativeRoomCopyFixture(zone);
  expect(() => buildPlayableLevel(animated.document, -1)).toThrow(/independent controller/);
  const activeSwitch = base.switches.find(item => item.zone >= 0)!;
  const switched = nativeRoomCopyFixture(activeSwitch.zone);
  expect(() => buildPlayableLevel(switched.document, -1)).toThrow(/independent (activation record|controller)/);
});

it('retains an explicit partial water plane, corner order and sampling through native room translation', () => {
  const partial = nativePartialRoomCopyFixture(), floor = partial.source.zones[0]!.lower.find(record => record.type === 1)!;
  const water = { ...floor.polygon!, type: 7 as const };
  const stream = importedGraphicsStreams(partial.source).get(0)!;
  const source = openLevel('Partial water copy', { ...partial.source.files, graphics: buildGraphicsStreams(partial.source, undefined, undefined,
    { streams: new Map([[0, { ...stream, lower: [...stream.lower, { bytes: encodePolygonSurface(water) }] }]]) }) });
  const { document, copy, dx, dz } = nativeRoomCopyFixture(0, source), result = buildPlayableLevel(document, -1);
  const cell = result.topology!.cells.find(cell => cell.sourceId === copy.id)!;
  const compiled = result.document.zones[cell.id]!.lower.find(record => record.type === 7)!.polygon!;
  expect(compiled).toMatchObject({ heightQuarter: water.heightQuarter, scale: water.scale, texture: water.texture, brightness: water.brightness });
  expect(compiled.points.map(id => result.document.points[id])).toEqual(water.points.map(id => expect.objectContaining({ x: source.points[id]!.x + dx, z: source.points[id]!.z + dz })));
  expect(compiled.closingPoint).toBe(compiled.points[0]);
});
