import { expect, it } from 'vitest';
import { mechanismTransferFixture } from './mechanismTransferFixture';
import { captureMechanismSettings, mechanismOwnerKey, mechanismSettingsOwners, transferMechanismSettings } from '../src/model/mechanismSettingsTransfer';
import { moveSourceMovingRoom } from '../src/model/sourceMovingRoom';
import { applyCommand, editRoomSource, execute, undo, redo, numberPatch } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { encodeProject, decodeProject } from '../src/model/project';
import { copiedMechanismSettings } from '../src/model/movingControllerCopies';
import { openLevel } from '../src/model/level';
import { originalFiles } from './helpers';
import { resolveAnimationRoom } from '../src/model/animationRooms';

for (const kind of ['door', 'lift'] as const) {
  it(`copies ${kind} settings across original, copied and authored targets while preserving their own attachments and state`, () => {
    const { document } = mechanismTransferFixture(kind), owners = mechanismSettingsOwners(document, kind);
    expect(owners.map(item => item.owner.type)).toEqual(['native', 'copy', 'authored']);
    for (const source of owners) for (const target of owners) {
      const before = captureMechanismSettings(document, target.owner), captured = captureMechanismSettings(document, source.owner);
      const command = transferMechanismSettings(document, target.owner, { ...captured, conditions: 1 }, { range: false, rules: true, requirements: true });
      const history = execute({ document, past: [], future: [] }, command), result = history.document;
      expect(captureMechanismSettings(result, target.owner)).toEqual({ ...before, conditions: 1, openRule: captured.openRule, closeRule: captured.closeRule });
      for (const unchanged of owners.filter(item => mechanismOwnerKey(item.owner) !== mechanismOwnerKey(target.owner)))
        expect(captureMechanismSettings(result, unchanged.owner)).toEqual(captureMechanismSettings(document, unchanged.owner));
      if (target.owner.type === 'native') {
        const native = document.mechanisms[0]!, after = result.mechanisms[0]!;
        expect(after).toMatchObject({ zone: native.zone, current: native.current, direction: native.direction, walls: native.walls });
        expect(result.files.graphics.slice(native.offset + 4, native.offset + 14)).toEqual(document.files.graphics.slice(native.offset + 4, native.offset + 14));
      } else if (target.owner.type === 'copy') {
        expect(result.roomSource.controllerCopies![0]).toMatchObject({ room: document.roomSource.controllerCopies![0]!.room, graphicsRooms: document.roomSource.controllerCopies![0]!.graphicsRooms, collisionRooms: document.roomSource.controllerCopies![0]!.collisionRooms });
      }
      expect(() => buildPlayableLevel(result, -1)).not.toThrow();
      expect(decodeProject(encodeProject(result)).roomSource).toEqual(result.roomSource);
      expect(undo(history).document.roomSource).toBe(document.roomSource);
      expect(redo(undo(history)).document.files).toEqual(result.files);
    }
  });

  it(`copies ${kind} destinations only when its initial position and native triggers remain valid`, () => {
    const { document } = mechanismTransferFixture(kind), owners = mechanismSettingsOwners(document, kind), native = owners[0]!.owner, copy = owners[1]!.owner;
    const source = captureMechanismSettings(document, copy);
    const result = applyCommand(document, transferMechanismSettings(document, native, source, { range: true, rules: false, requirements: false }));
    expect(captureMechanismSettings(result, native)).toMatchObject({ top: source.top, bottom: source.bottom });
    expect(result.mechanisms[0]!.current).toBe(document.mechanisms[0]!.current);
    const unsafe = { ...source, top: source.bottom + 1, bottom: source.bottom + 2 };
    expect(() => transferMechanismSettings(document, native, unsafe, { range: true, rules: true, requirements: false })).toThrow(/include the initial position/);
    expect(() => transferMechanismSettings(document, native, source, { range: false, rules: false, requirements: false })).toThrow(/at least one/);
    expect(() => transferMechanismSettings(document, native, { ...source, kind: kind === 'door' ? 'lift' : 'door' }, { range: false, rules: true, requirements: false })).toThrow(/two doors or two lifts/);
  });

  it(`moves an authored ${kind} as one reversible source edit and regenerates its native attachments`, () => {
    const fixture = mechanismTransferFixture(kind), { document, room, neighbour } = fixture;
    const history = execute({ document, past: [], future: [] }, editRoomSource(moveSourceMovingRoom(document, room.id, neighbour.id)));
    const result = history.document, built = buildPlayableLevel(result, -1), mechanism = built.document.mechanisms.filter(item => item.kind === kind).at(-1)!;
    expect(result.roomSource.rooms.find(item => item.id === room.id)!.moving).toBeUndefined();
    expect(result.roomSource.rooms.find(item => item.id === neighbour.id)!.moving).toEqual(document.roomSource.rooms.find(item => item.id === room.id)!.moving);
    expect(built.topology!.cells.find(item => item.id === mechanism.zone)!.sourceId).toBe(neighbour.id);
    expect(result.roomSource.controllerCopies).toBe(document.roomSource.controllerCopies);
    expect(undo(history).document.roomSource).toBe(document.roomSource);
    expect(redo(undo(history)).document.roomSource).toBe(result.roomSource);
    expect(decodeProject(encodeProject(result)).roomSource).toEqual(result.roomSource);
    expect(() => moveSourceMovingRoom(document, room.id, fixture.copied.copy.room)).toThrow(/copied controller/);
    expect(() => moveSourceMovingRoom(document, room.id, room.id)).toThrow(/another existing room/);
  });
}

it('captures independent settings that survive source edits and distinguishes preserved requirements from explicit replacement', () => {
  const { document } = mechanismTransferFixture('door'), owners = mechanismSettingsOwners(document, 'door'), source = owners[2]!.owner, target = owners[1]!.owner;
  const captured = captureMechanismSettings(document, source);
  const edited = applyCommand(document, transferMechanismSettings(document, source, { ...captured, openRule: 3, conditions: 2 }, { range: false, rules: true, requirements: true }));
  const result = applyCommand(edited, transferMechanismSettings(edited, target, captured, { range: false, rules: true, requirements: false }));
  expect(copiedMechanismSettings(result.roomSource.controllerCopies![0]!)).toMatchObject({ conditions: 0, openRule: captured.openRule });
  expect(captureMechanismSettings(result, source)).toMatchObject({ openRule: 3, conditions: 2 });
});

it('validates copied requirements against the final combined range rather than committing an invalid intermediate edit', () => {
  const fixture = mechanismTransferFixture('door'), native = fixture.document.mechanisms[0]!;
  const current = (native.top + native.bottom) / 2;
  const document = applyCommand(fixture.document, { label: 'Intermediate native door', patches: [numberPatch(fixture.document, 'graphics', native.offset + 4, current * 4, 2)] });
  const owner = { type: 'native' as const, kind: 'door' as const, id: native.id }, settings = captureMechanismSettings(document, owner);
  expect(() => transferMechanismSettings(document, owner, { ...settings, conditions: 1 }, { range: false, rules: false, requirements: true })).toThrow(/Keep one destination/);
  const command = transferMechanismSettings(document, owner, { ...settings, top: current, conditions: 1 }, { range: true, rules: false, requirements: true });
  const history = execute({ document, past: [], future: [] }, command);
  expect(history.document.mechanisms[0]).toMatchObject({ top: current, current, conditions: 1 });
  expect(undo(history).document.files).toEqual(document.files);
});

it.each(Array.from({ length: 16 }, (_, index) => index + 1))('lists and captures native controller settings in original level %i with resolved room names', level => {
  const document = openLevel(`Controller settings ${level}`, originalFiles(level));
  for (const kind of ['door', 'lift'] as const) {
    const owners = mechanismSettingsOwners(document, kind);
    expect(owners).toHaveLength(document.mechanisms.filter(item => item.kind === kind).length);
    for (const item of owners) {
      const native = document.mechanisms.find(control => control.kind === kind && item.owner.type === 'native' && control.id === item.owner.id)!;
      expect(item.label).toContain(`Zone ${resolveAnimationRoom(document, native.zone, kind === 'lift')}`);
      expect(captureMechanismSettings(document, item.owner)).toEqual({ kind, top: native.top, bottom: native.bottom, conditions: native.conditions, openRule: native.openRule, closeRule: native.closeRule });
    }
  }
});
