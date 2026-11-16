import { expect, it } from 'vitest';
import { mechanismTransferFixture } from './mechanismTransferFixture';
import { availableSwitchSlots } from '../src/model/sourceSwitches';
import { captureMechanismSettings, mechanismSettingsOwners, mechanismOwnerKey } from '../src/model/mechanismSettingsTransfer';
import { createLinkedSwitch, linkSwitchMechanism } from '../src/model/switchMechanismLinks';
import { execute, undo, redo } from '../src/model/commands';
import { decodeProject, encodeProject } from '../src/model/project';

for (const kind of ['door', 'lift'] as const) for (const type of ['native', 'copy', 'authored'] as const)
  it(`places and connects a switch to a ${type} ${kind} atomically without changing other controllers`, () => {
    const fixture = mechanismTransferFixture(kind), { document } = fixture;
    const owners = mechanismSettingsOwners(document, kind), owner = owners.find(item => item.owner.type === type)!.owner;
    const before = captureMechanismSettings(document, owner), slot = availableSwitchSlots(document)[0]!;
    const command = createLinkedSwitch(document, fixture.neighbour.id, slot, 2, owner, 'activate');
    const history = execute({ document, past: [], future: [] }, command), result = history.document;
    expect(history.past).toHaveLength(1);
    expect(result.roomSource.rooms.find(room => room.id === fixture.neighbour.id)!.switches).toContainEqual({ slot, boundary: 2, automatic: false });
    expect(captureMechanismSettings(result, owner)).toEqual({ ...before, conditions: before.conditions | (1 << (slot + 4)), openRule: kind === 'door' ? 4 : 2 });
    for (const other of owners.filter(item => mechanismOwnerKey(item.owner) !== mechanismOwnerKey(owner)))
      expect(captureMechanismSettings(result, other.owner)).toEqual(captureMechanismSettings(document, other.owner));
    expect(undo(history).document.roomSource).toBe(document.roomSource);
    expect(undo(history).document.files).toEqual(document.files);
    expect(redo(undo(history)).document.files).toEqual(result.files);
    const reopened = decodeProject(encodeProject(result));
    expect(captureMechanismSettings(reopened, owner)).toEqual(captureMechanismSettings(result, owner));
    const unlink = execute(history, linkSwitchMechanism(result, owner, slot, false)).document;
    expect(captureMechanismSettings(unlink, owner)).toEqual({ ...captureMechanismSettings(result, owner), conditions: before.conditions });
  });

it('keeps trigger rules when only unlocking an existing mechanism and rejects a missing target atomically', () => {
  const fixture = mechanismTransferFixture('door'), { document } = fixture;
  const owner = mechanismSettingsOwners(document, 'door').find(item => item.owner.type === 'authored')!.owner;
  const slot = availableSwitchSlots(document)[0]!, before = captureMechanismSettings(document, owner);
  const history = execute({ document, past: [], future: [] }, createLinkedSwitch(document, fixture.neighbour.id, slot, 2, owner));
  expect(captureMechanismSettings(history.document, owner)).toEqual({ ...before, conditions: before.conditions | (1 << (slot + 4)) });
  expect(() => linkSwitchMechanism(document, owner, slot, true)).toThrow(/Place this wall switch/);
  expect(() => createLinkedSwitch(document, fixture.neighbour.id, slot, 2, { type: 'authored', kind: 'door', room: 'missing' })).toThrow(/no longer exists/);
  expect(document.roomSource.rooms.find(room => room.id === fixture.neighbour.id)!.switches).toBeUndefined();
});
