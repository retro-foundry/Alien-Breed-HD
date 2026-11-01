import { expect, it } from 'vitest';
import { switchCopyFixture } from './switchCopyFixture';
import { applyCommand, editRoomSource } from '../src/model/commands';
import { setRoomSwitchCopies } from '../src/model/switchCopies';
import { setCopiedSwitchAutomaticReset, setSwitchAutomaticReset } from '../src/model/switchSettings';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';

it('changes original and copied reset flags independently while preserving countdown, state and attachments', () => {
  const fixture = switchCopyFixture(), { switchCopy: copy, source } = fixture;
  const document = applyCommand(fixture.document, editRoomSource(setRoomSwitchCopies(fixture.document, [copy])));
  const snapshot = document.roomSource.switchCopies![0]!;
  const originalChange = setSwitchAutomaticReset(document, source.id, source.type === 0), original = applyCommand(document, originalChange);
  expect(original.roomSource.switchCopies).toBe(document.roomSource.switchCopies);
  expect(original.switches[source.id]!.type).toBe(source.type ? 0 : 1);
  expect(applyCommand(original, originalChange, true).files).toEqual(document.files);
  const command = setCopiedSwitchAutomaticReset(document, copy.slot, snapshot.bytes[2] === 0), edited = applyCommand(document, command);
  const after = edited.roomSource.switchCopies![0]!;
  expect(after).toEqual({ ...snapshot, bytes: snapshot.bytes.map((byte, index) => index === 2 ? byte ? 0 : 1 : byte) });
  expect(edited.files).toEqual(document.files);
  const saved = decodeProject(encodeProject(edited)), built = buildPlayableLevel(saved, -1);
  expect(built.document.switches[source.id]!.type).toBe(source.type);
  expect(built.document.switches[copy.slot]!.type).toBe(after.bytes[2]);
  expect(built.document.switches[copy.slot]!.countdown).toBe(source.countdown);
  expect(applyCommand(edited, command, true).roomSource).toBe(document.roomSource);
  expect(setCopiedSwitchAutomaticReset(document, copy.slot, snapshot.bytes[2] !== 0).roomSource!.after).toBe(document.roomSource);
});

it('preserves a native nonzero reset flag and rejects invalid targets without changing source', () => {
  const fixture = switchCopyFixture(), { switchCopy: copy, source } = fixture;
  const bytes = fixture.document.files.graphics.slice(source.offset, source.offset + 14); bytes[2] = 255;
  const document = applyCommand(fixture.document, editRoomSource(setRoomSwitchCopies(fixture.document, [{ ...copy, bytes }])));
  expect(setCopiedSwitchAutomaticReset(document, copy.slot, true).roomSource!.after).toBe(document.roomSource);
  expect(() => setCopiedSwitchAutomaticReset(document, 8, true)).toThrow(/slot/);
  expect(() => setCopiedSwitchAutomaticReset(document, source.id, true)).toThrow(/no longer exists/);
  expect(() => setSwitchAutomaticReset(document, copy.slot, true)).toThrow(/no longer exists/);
  expect(() => setCopiedSwitchAutomaticReset(document, copy.slot, 'yes' as unknown as boolean)).toThrow(/enabled or disabled/);
  expect(document.roomSource.switchCopies![0]!.bytes[2]).toBe(255);
});
