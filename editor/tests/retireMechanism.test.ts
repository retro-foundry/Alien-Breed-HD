import { roomTemplate } from '../src/model/staticMechanismPose';
import { setSourceSwitch } from '../src/model/sourceSwitches';
import { setWallSettings } from '../src/model/wallSettings';
import { mechanismDependencies } from '../src/model/mechanismDependencies';
import { copyRoomGroup, pasteRooms } from '../src/model/roomClipboard';
import { originalFiles } from './helpers';
import { openLevel } from '../src/model/level';
import { expect, it } from 'vitest';
import { mechanismTransferFixture } from './mechanismTransferFixture';
import { retireMechanism, retiringMechanismSettings, reassignImportedMechanism, type ImportedMechanismOwner } from '../src/model/retireMechanism';
import { activeNativeMechanisms } from '../src/model/nativeMechanisms';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { applyCommand, editRoomSource, execute, undo, redo, setMechanismRules } from '../src/model/commands';
import { encodeProject, decodeProject } from '../src/model/project';
import { roomSettings } from '../src/model/roomHeights';
import { roomSourceSnapshot, restoreRoomSource, setRoomHeights } from '../src/model/roomSource';
import { setSourceMovingRoom } from '../src/model/sourceMovingRoom';
import { mechanismSettingsOwners } from '../src/model/mechanismSettingsTransfer';

for (const kind of ['door', 'lift'] as const) for (const type of ['native', 'copy'] as const) {
  it(`retires an ${type} ${kind} and every attachment as one saved, reversible static pose`, () => {
    const { document } = mechanismTransferFixture(kind);
    const owner = mechanismSettingsOwners(document, kind).find(item => item.owner.type === type)!.owner as ImportedMechanismOwner;
    const settings = retiringMechanismSettings(document, owner), height = kind === 'door' ? settings.top : settings.current;
    const beforeBuild = buildPlayableLevel(document, -1);
    const command = editRoomSource(retireMechanism(document, owner, height));
    const history = execute({ document, past: [], future: [] }, command), result = history.document;
    expect(result.files).toEqual(document.files);
    expect(result.mechanisms).toEqual(document.mechanisms);
    expect(activeNativeMechanisms(result)).toHaveLength(type === 'native' ? 0 : 1);
    expect(result.roomSource.controllerCopies ?? []).toHaveLength(type === 'copy' ? 0 : 1);
    const room = result.roomSource.rooms.find(room => room.id === settings.room)!;
    expect(roomSettings(result, room)[kind === 'door' ? 'ceiling' : 'floor']).toBe(height);
    const built = buildPlayableLevel(result, -1);
    expect(built.document.mechanisms).toHaveLength(beforeBuild.document.mechanisms.length - 1);
    const zone = built.topology!.cells.find(cell => cell.sourceId === room.id)!.id;
    expect(built.document.zones[zone]![kind === 'door' ? 'ceiling' : 'floor']).toBe(height);
    const poses = result.roomSource.rooms.flatMap(room => (room.staticMechanisms ?? []).map(pose => ({ room, pose })));
    expect(poses.length).toBeGreaterThan(0);
    for (const { room: ownerRoom, pose } of poses) for (const wall of pose.walls) {
      const records = built.surfaceSources!.filter(record => record.sourceOffset === wall.record && built.topology!.cells[record.room]!.sourceId === ownerRoom.id);
      expect(records.length).toBeGreaterThan(0);
      for (const record of records) {
        const surface = built.document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(item => item.offset === record.offset)!.wall!;
        expect(surface[kind === 'door' ? 'bottomFixed' : 'topFixed']).toBe(height * 256);
        expect(surface).toMatchObject({ horizontalOffset: wall.horizontalOffset, verticalOffset: wall.verticalOffset });
      }
    }
    expect(roomSourceSnapshot(result.roomSource).version).toBe(24);
    expect(decodeProject(encodeProject(result)).roomSource).toEqual(result.roomSource);
    expect(undo(history).document.roomSource).toBe(document.roomSource);
    expect(redo(undo(history)).document.roomSource).toBe(result.roomSource);
    expect(() => retireMechanism(result, owner, height)).toThrow(/no longer exists/);
    // Paste a captured static group after Undo: the original native controller
    // is live again in the destination, but the clipboard must stay static.
    const clipboard = copyRoomGroup(result, [settings.room]);
    const pasted = applyCommand(document, pasteRooms(document, clipboard, -16000, 0).command);
    expect(buildPlayableLevel(pasted, -1).document.mechanisms).toHaveLength(beforeBuild.document.mechanisms.length);
    expect(pasted.roomSource.rooms.slice(document.roomSource.rooms.length).some(room => room.staticMechanisms?.length)).toBe(true);
    const current = roomSettings(result, room);
    const edited = applyCommand(result, editRoomSource(setRoomHeights(result, result.roomSource, room.id, { floor: current.floor + 4, ceiling: current.ceiling, upperFloor: current.upperFloor, upperCeiling: current.upperCeiling })));
    expect(edited.roomSource.removedNativeMechanisms).toEqual(result.roomSource.removedNativeMechanisms);
    expect(buildPlayableLevel(edited, -1).document.mechanisms).toHaveLength(beforeBuild.document.mechanisms.length - 1);

  });

  it(`captures effective ${type} ${kind} wall phase while keeping independent brightness`, () => {
    const fixture = mechanismTransferFixture(kind);
    let document = fixture.document;
    const owner = mechanismSettingsOwners(document, kind).find(item => item.owner.type === type)!.owner as ImportedMechanismOwner;
    const copy = document.roomSource.controllerCopies![0]!;
    const native = document.mechanisms[0]!, dependency = mechanismDependencies(document, kind, native.id), attachment = dependency.walls[0]!;
    const room = document.roomSource.rooms.find(room => type === 'native' ? room.originZone === attachment.room : room.id === copy.graphicsRooms.find(pair => pair[0] === attachment.room)![1])!;
    const upper = document.zones[room.recordZone]!.upper.some(record => record.offset === attachment.graphic);
    document = applyCommand(document, editRoomSource(setWallSettings(document, { room: room.id, record: attachment.graphic, upper }, { horizontalOffset: 0, verticalOffset: 17, brightness: 2 })));
    const settings = retiringMechanismSettings(document, owner), height = kind === 'door' ? settings.top : settings.current;
    const retired = applyCommand(document, editRoomSource(retireMechanism(document, owner, height)));
    const saved = retired.roomSource.rooms.find(item => item.id === room.id)!;
    const expectedRow = kind === 'door' ? ((17 + Math.floor(settings.current)) & 255) + ((-Math.floor(height)) & 255) : 17;
    expect(saved.staticMechanisms![0]!.walls.find(wall => wall.record === attachment.graphic)).toMatchObject({ horizontalOffset: 0, verticalOffset: expectedRow });
    expect(saved.wallSettings!.find(item => 'record' in item && item.record === attachment.graphic)).toEqual({ record: attachment.graphic, upper, brightness: 2 });
    const snapshot = roomSourceSnapshot(retired.roomSource);
    const bad = { ...snapshot, removedNativeMechanisms: [{ kind, id: 255 }] };
    expect(() => buildPlayableLevel({ ...retired, roomSource: restoreRoomSource(retired, bad) }, -1)).toThrow(/template/);
    expect(() => restoreRoomSource(retired, { ...snapshot, rooms: snapshot.rooms.map(item => item.id === room.id ? { ...item, staticMechanisms: [{ ...item.staticMechanisms![0], height: '0' }] } : item) })).toThrow(/numeric/);
    expect(decodeProject(encodeProject(retired)).roomSource).toEqual(retired.roomSource);
  });

  it(`reassigns an ${type} ${kind} to an empty room and rebuilds independent destination attachments`, () => {
    const fixture = mechanismTransferFixture(kind), { room } = fixture;
    let document = applyCommand(fixture.document, editRoomSource(setSourceMovingRoom(fixture.document, room.id, null)));
    const owner = mechanismSettingsOwners(document, kind).find(item => item.owner.type === type)!.owner as ImportedMechanismOwner;
    expect(() => reassignImportedMechanism(fixture.document, owner, room.id, 0, 0)).toThrow(/already/);
    const settings = retiringMechanismSettings(document, owner);
    const height = kind === 'door' ? settings.top : settings.current;
    if (kind === 'lift') for (const id of [room.id, fixture.neighbour.id]) {
      const target = document.roomSource.rooms.find(item => item.id === id)!, current = roomSettings(document, target);
      document = applyCommand(document, editRoomSource(setRoomHeights(document, document.roomSource, id, { floor: current.floor, ceiling: settings.top - 32, upperFloor: current.upperFloor, upperCeiling: current.upperCeiling })));
    }
    const result = applyCommand(document, editRoomSource(reassignImportedMechanism(document, owner, room.id, height, document.roomAppearances!.find(item => item.room === room.id)!.lowerWall)));
    expect(result.roomSource.rooms.find(item => item.id === room.id)!.moving?.kind).toBe(kind);
    expect(buildPlayableLevel(result, -1).document.mechanisms).toHaveLength(buildPlayableLevel(document, -1).document.mechanisms.length);
    expect(() => reassignImportedMechanism(document, owner, settings.room, height, 0)).toThrow(/another/);
    expect(() => retireMechanism(document, owner, settings.bottom + 0.25)).toThrow(/between/);
    const snapshot = roomSourceSnapshot(result.roomSource);
    expect(() => restoreRoomSource(result, { ...snapshot, version: 23 })).toThrow(/unsupported/);
  });
}

for (let level = 1; level <= 16; level++) it(`retires the independently owned original controllers of shipped level ${level}`, () => {
  const document = openLevel(`Original ${level}`, originalFiles(level), level - 1, -1);
  for (const mechanism of document.mechanisms) {
    const owner = { type: 'native' as const, kind: mechanism.kind, id: mechanism.id };
    const result = applyCommand(document, editRoomSource(retireMechanism(document, owner, mechanism.current)));
    expect(buildPlayableLevel(result, -1).document.mechanisms).toHaveLength(document.mechanisms.length - 1);
  }
}, 60000);

it('does not reserve a retired original controller switch requirements when allocating a pasted switch', () => {
  const fixture = mechanismTransferFixture('door');
  let document = applyCommand(fixture.document, setMechanismRules(fixture.document, 'door', 0, 16, 5, 1));
  document = applyCommand(document, editRoomSource(setSourceSwitch(document, fixture.neighbour.id, 1, { edge: 2, automatic: false })));
  const retired = applyCommand(document, editRoomSource(retireMechanism(document, { type: 'native', kind: 'door', id: 0 }, document.mechanisms[0]!.top)));
  const copy = pasteRooms(retired, copyRoomGroup(retired, [fixture.neighbour.id]), -16000, 0);
  const pasted = applyCommand(retired, copy.command);
  expect(pasted.roomSource.rooms.find(room => room.id === copy.ids[0])!.switches![0]!.slot).toBe(0);
  expect(() => buildPlayableLevel(pasted, -1)).not.toThrow();
});

it('keeps static copied-door lintels when lowering its ceiling while the original remains animated', () => {
  const fixture = mechanismTransferFixture('door'), copy = fixture.document.roomSource.controllerCopies![0]!;
  const owner = { type: 'copy' as const, kind: 'door' as const, room: copy.room };
  const retired = applyCommand(fixture.document, editRoomSource(retireMechanism(fixture.document, owner, -48)));
  const room = retired.roomSource.rooms.find(room => room.id === copy.room)!, settings = roomSettings(retired, room);
  const edited = applyCommand(retired, editRoomSource(setRoomHeights(retired, retired.roomSource, room.id, { floor: settings.floor, ceiling: -32, upperFloor: settings.upperFloor, upperCeiling: settings.upperCeiling })));
  const built = buildPlayableLevel(edited, -1), dependency = mechanismDependencies(edited, 'door', copy.id);
  const walls = dependency.walls.filter(wall => wall.room !== dependency.room);
  expect(walls.length).toBeGreaterThan(0);
  for (const wall of walls) {
    const owner = copy.graphicsRooms.find(pair => pair[0] === wall.room)![1];
    const generated = built.surfaceSources!.filter(record => record.sourceOffset === wall.graphic && built.topology!.cells[record.room]!.sourceId === owner);
    expect(generated.length).toBeGreaterThan(0);
    for (const identity of generated) expect(built.document.zones.flatMap(zone => [...zone.lower, ...zone.upper]).find(record => record.offset === identity.offset)!.wall!.bottomFixed).toBe(-32 * 256);
  }
});

it('can author a fresh controller in a retired copy without reserving the original controller walls', () => {
  const fixture = mechanismTransferFixture('door'), copy = fixture.document.roomSource.controllerCopies![0]!;
  const retired = applyCommand(fixture.document, editRoomSource(retireMechanism(fixture.document, { type: 'copy', kind: 'door', room: copy.room }, -48)));
  const dependency = mechanismDependencies(retired, 'door', copy.id);
  const recreated = applyCommand(retired, editRoomSource(setSourceMovingRoom(retired, copy.room, { kind: 'door', header: copy.header, wall: dependency.walls[0]!.graphic })));
  const built = buildPlayableLevel(recreated, -1);
  expect(built.document.mechanisms).toHaveLength(3);
  expect(activeNativeMechanisms(recreated)).toEqual(activeNativeMechanisms(retired));
  expect(built.document.mechanisms.at(-1)!.walls.length).toBeGreaterThan(0);
});

it('preserves unattached imported bump decoration heights when freezing a normal moving ceiling', () => {
  const fixture = mechanismTransferFixture('door'), document = fixture.document;
  const dependency = mechanismDependencies(document, 'door', 0), template = document.zones[dependency.room]!;
  const floor = template.lower.find(record => record.type === 1)!;
  const decoration = { ...floor, type: 9, polygon: { ...floor.polygon!, type: 9 as const, heightQuarter: -128 } };
  const original = { ...document, zones: document.zones.map(zone => zone.id === template.id ? { ...zone, lower: zone.lower.map(record => record === floor ? decoration : record) } : zone) };
  const projected = roomTemplate(original, { recordZone: template.id, staticMechanisms: [{ kind: 'door', id: 0, height: -48, plane: true, surface: dependency.surface.graphic, walls: [] }] });
  expect(projected.lower.find(record => record.offset === decoration.offset)).toBe(decoration);
  expect(projected.lower.find(record => record.offset === dependency.surface.graphic)!.polygon!.heightQuarter).toBe(-192);
});
