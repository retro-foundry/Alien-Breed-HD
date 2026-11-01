import { expect, it } from 'vitest';
import { nativeRoomCopyFixture } from './nativeRoomCopyFixture';
import { buildPlayableLevel } from '../src/model/playableBuild';
import { decodeProject, encodeProject } from '../src/model/project';
import { resolveAnimationRoom } from '../src/model/animationRooms';

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

it('rejects native-material copies with moving controllers or switches instead of silently making inert copies', () => {
  const base = nativeRoomCopyFixture().document;
  const moving = base.mechanisms[0]!, zone = resolveAnimationRoom(base, moving.zone, moving.kind === 'lift')!;
  const animated = nativeRoomCopyFixture(zone);
  expect(() => buildPlayableLevel(animated.document, -1)).toThrow(/independent controller/);
  const activeSwitch = base.switches.find(item => item.zone >= 0)!;
  const switched = nativeRoomCopyFixture(activeSwitch.zone);
  expect(() => buildPlayableLevel(switched.document, -1)).toThrow(/independent (activation record|controller)/);
});
