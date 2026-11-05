import { expect, it } from 'vitest';
import { applyCommand, editRoomAppearance, execute, redo, undo } from '../src/model/commands';
import { finishRoomDraft } from '../src/model/roomCreation';
import { beginRoomDraft, setDraftCorners, type SourceRoom } from '../src/model/roomSource';
import { defaultRoomAppearance, validateRoomAppearances } from '../src/model/roomAppearance';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { roomPlaneRecords } from '../src/model/roomPlanes';
import { setPolygonMaterial } from '../src/model/polygonMaterials';
import { decodeProject, encodeProject } from '../src/model/project';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { nativePartialRoomCopyFixture } from './nativeRoomCopyFixture';
import { twoStoreyHeightTemplate } from './roomHeightFixture';
import { createRoomOpening } from '../src/model/roomOpening';
import { buildGraphicsStreams } from '../src/model/graphicsBuild';

const rectangle = [{ x: 20000, z: 20000 }, { x: 20256, z: 20000 }, { x: 20256, z: 20256 }, { x: 20000, z: 20256 }];

it('creates and persists a textured room in one stale-checked Undo without an appearance setup step', () => {
  const original = openLevel('Textured room', originalFiles(1));
  const draft = setDraftCorners(beginRoomDraft(original, original.roomSource, 0), rectangle), command = finishRoomDraft(original, draft);
  const history = execute({ document: original, past: [], future: [] }, command), source = history.document, room = source.roomSource.rooms.at(-1)!;
  expect(source.roomAppearances).toHaveLength(1); expect(source.roomAppearances![0]!.surfaces).toBeDefined();
  const appearanceState = source.roomAppearances![0]!;
  expect(() => validateRoomAppearances(source, [{ ...appearanceState, surfaces: { ...appearanceState.surfaces!, lowerFloor: appearanceState.surfaces!.lowerCeiling } }])).toThrow(/missing default/);
  const built = buildPlayableLevel(source, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!, native = built.document.zones[cell.id]!;
  expect(native.lower.some(record => record.wall)).toBe(true); expect(native.lower.some(record => record.type === 1)).toBe(true); expect(native.lower.some(record => record.type === 2)).toBe(true);
  expect(decodeProject(encodeProject(source)).roomAppearances).toEqual(source.roomAppearances);
  expect(undo(history).document.roomAppearances).toBeUndefined(); expect(undo(history).document.roomSource).toBe(original.roomSource);
  expect(redo(undo(history)).document.roomAppearances).toBe(source.roomAppearances);
  expect(() => applyCommand(source, command)).toThrow(/room appearance changed|room source changed/);
  const appearance = source.roomAppearances![0]!, changed = applyCommand(source, editRoomAppearance(source, { room: room.id, lowerWall: appearance.lowerWall, upperWall: appearance.upperWall, lightingPoint: appearance.lightingPoint }));
  expect(changed.roomAppearances![0]!.surfaces).toEqual(appearance.surfaces);
  const target = { room: room.id, upper: false, record: appearance.surfaces!.lowerFloor };
  const painted = { ...source, roomSource: setPolygonMaterial(source, target, { texture: 0, scale: -1, mode: 'normal' }).after };
  expect(buildPlayableLevel(painted, -1).document.zones[cell.id]!.lower.find(record => record.type === 1)!.polygon).toMatchObject({ texture: 0, scale: -1 });
  const redrawn = applyCommand(source, finishRoomDraft(source, setDraftCorners(beginRoomDraft(source, source.roomSource, 0, room.id), rectangle.map(point => ({ x: point.x + 8, z: point.z })))));
  expect(redrawn.roomAppearances).toBe(source.roomAppearances);
});

it('gives both storeys editable normal textures immediately and retains the recipe through opening partitions', () => {
  const original = openLevel('Two-storey defaults', originalFiles(1)), template = twoStoreyHeightTemplate();
  const source = applyCommand(original, finishRoomDraft(original, setDraftCorners(beginRoomDraft(original, original.roomSource, template), rectangle)));
  const room = source.roomSource.rooms.at(-1)!, appearance = source.roomAppearances![0]!, built = buildPlayableLevel(source, -1), zone = built.document.zones[built.topology!.cells.find(cell => cell.sourceId === room.id)!.id]!;
  for (const stream of [zone.lower, zone.upper]) {
    expect(stream.some(record => record.wall)).toBe(true); expect(stream.some(record => record.type === 1)).toBe(true); expect(stream.some(record => record.type === 2)).toBe(true);
  }
  const proposal = createRoomOpening(source, room.id, { left: 20064, right: 20192, top: 20064, bottom: 20192 }, appearance, 'sky');
  const opened = applyCommand(source, proposal.command);
  for (const added of opened.roomSource.rooms.filter(item => item.originZone === null)) expect(opened.roomAppearances!.find(item => item.room === added.id)!.surfaces).toEqual(appearance.surfaces);
  expect(() => buildPlayableLevel(decodeProject(encodeProject(opened)), -1)).not.toThrow();
});

it('uses template textures without copying independent control geometry into a fresh room', () => {
  const baseline = openLevel('Control geometry template', originalFiles(1)), template = baseline.zones[0]!;
  const light = new Uint8Array(6), view = new DataView(light.buffer);
  view.setInt16(0, 6); view.setInt16(2, template.lower.find(record => record.wall)!.points[0]!); view.setInt16(4, template.lower.find(record => record.wall)!.points[1]!);
  const records = (stream: typeof template.lower) => stream.map(record => ({ sourceOffset: record.offset, bytes: baseline.files.graphics.slice(record.offset, record.offset + record.size) }));
  const graphics = buildGraphicsStreams(baseline, undefined, undefined, { streams: new Map([[template.id, { lower: [...records(template.lower), { bytes: light }], upper: records(template.upper) }]]) });
  const original = openLevel(baseline.name, { ...baseline.files, graphics });
  const source = applyCommand(original, finishRoomDraft(original, setDraftCorners(beginRoomDraft(original, original.roomSource, template.id), rectangle)));
  const room = source.roomSource.rooms.at(-1)!, built = buildPlayableLevel(source, -1), zone = built.document.zones[built.topology!.cells.find(cell => cell.sourceId === room.id)!.id]!;
  expect([...zone.lower, ...zone.upper].some(record => record.type === 5 || record.type === 6)).toBe(false);
  expect(zone.lower.some(record => record.wall)).toBe(true);
});

it('uses complete starter surfaces when the chosen template contains partial decorations', () => {
  const fixture = nativePartialRoomCopyFixture(), original = fixture.document;
  const source = applyCommand(original, finishRoomDraft(original, setDraftCorners(beginRoomDraft(original, original.roomSource, fixture.copy.recordZone), rectangle.map(point => ({ x: point.x + 2000, z: point.z })))));
  const room = source.roomSource.rooms.at(-1)!, built = buildPlayableLevel(source, -1), cell = built.topology!.cells.find(cell => cell.sourceId === room.id)!;
  expect(built.document.zones[cell.id]!.lower.filter(record => record.polygon).every(record => record.points.length === 4)).toBe(true);
  expect(roomPlaneRecords(source, room, false)).toHaveLength(2);
});

it.each(Array.from({ length: 16 }, (_, i) => i + 1))('finds real complete starter materials for every template in level %i', level => {
  const document = openLevel('Template defaults', originalFiles(level));
  for (const zone of document.zones) {
    const room: SourceRoom = { id: 'starter', recordZone: zone.id, originZone: null, vertices: rectangle, wallSources: rectangle.map(() => null) };
    const appearance = defaultRoomAppearance(document, room), source = { ...document, roomSource: { ...document.roomSource, rooms: [...document.roomSource.rooms, room] } };
    expect(() => validateRoomAppearances(source, [appearance])).not.toThrow();
    expect(appearance.surfaces).toBeDefined();
  }
});
