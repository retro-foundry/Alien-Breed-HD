import { expect, it } from 'vitest';
import { movingControllerCopyFixture } from './movingControllerCopyFixture';
import { copiedMechanismSettings, setRoomControllerCopies, setCopiedMechanismRange, setCopiedMechanismRules } from '../src/model/movingControllerCopies';
import { applyCommand, editRoomSource } from '../src/model/commands';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';

it.each(['door', 'lift'] as const)('edits copied %s ranges and requirements independently with atomic validation and Undo', kind => {
  const fixture = movingControllerCopyFixture(kind), { copy } = fixture;
  const document = applyCommand(fixture.document, editRoomSource(setRoomControllerCopies(fixture.document, [copy])));
  const before = copiedMechanismSettings(document.roomSource.controllerCopies![0]!);
  expect(() => setCopiedMechanismRange(document, copy.room, kind, before.top + 0.125, before.bottom)).toThrow(/0.25-unit steps/);
  expect(() => setCopiedMechanismRange(document, copy.room, kind, before.top, 249.75)).toThrow(/cannot be stored/);
  expect(() => setCopiedMechanismRange(document, copy.room, kind, before.bottom + 1, before.bottom)).toThrow(/upper destination/);
  expect(() => setCopiedMechanismRules(document, copy.room, kind, 65536, before.openRule, before.closeRule)).toThrow(/requirements/);
  expect(() => setCopiedMechanismRules(document, copy.room, kind, 0, 256, before.closeRule)).toThrow(/Opening rule/);
  const range = editRoomSource(setCopiedMechanismRange(document, copy.room, kind, before.top + 0.25, before.bottom));
  const ranged = applyCommand(document, range);
  const rules = editRoomSource(setCopiedMechanismRules(ranged, copy.room, kind, 1 | (1 << 4), before.openRule, before.closeRule));
  const ruled = applyCommand(ranged, rules), saved = decodeProject(encodeProject(ruled));
  expect(saved.files).toEqual(document.files);
  expect(copiedMechanismSettings(saved.roomSource.controllerCopies![0]!)).toMatchObject({ top: before.top + 0.25, conditions: 17 });
  const controllers = buildPlayableLevel(saved, -1).document.mechanisms.filter(item => item.kind === kind);
  expect(controllers[0]).toEqual(buildPlayableLevel(document, -1).document.mechanisms.find(item => item.kind === kind));
  expect(controllers[1]).toMatchObject({ top: before.top + 0.25, conditions: 17, current: before.current });
  expect(applyCommand(ruled, rules, true).roomSource).toBe(ranged.roomSource);
  expect(applyCommand(ranged, range, true).roomSource).toBe(document.roomSource);
  expect(setCopiedMechanismRange(document, copy.room, kind, before.top, before.bottom).after).toBe(document.roomSource);
});

it('keeps the native occupied-door exception while rejecting a new gate that would strand a copied idle door', () => {
  const fixture = movingControllerCopyFixture('door'), { copy } = fixture;
  const captured = applyCommand(fixture.document, editRoomSource(setRoomControllerCopies(fixture.document, [copy])));
  const settings = copiedMechanismSettings(captured.roomSource.controllerCopies![0]!);
  const intermediate = applyCommand(captured, editRoomSource(setCopiedMechanismRange(captured, copy.room, 'door', settings.top, settings.bottom + 1)));
  expect(copiedMechanismSettings(intermediate.roomSource.controllerCopies![0]!)).toMatchObject({ current: settings.current, bottom: settings.bottom + 1, conditions: 0 });
  expect(() => setCopiedMechanismRules(intermediate, copy.room, 'door', 1, settings.openRule, settings.closeRule)).toThrow(/stationary door/);
  expect(copiedMechanismSettings(intermediate.roomSource.controllerCopies![0]!).conditions).toBe(0);
  expect(intermediate.files).toEqual(captured.files);
});
