import { checkedInteger } from './binary';
import type { LevelDocument } from './level';
import type { RoomSource, SourceRoom } from './roomSource';

/** Original records remain templates for independent copies after removal. */
export function activeNativeSwitches(document: LevelDocument, source: RoomSource = document.roomSource) {
  return document.switches.filter(item => item.zone >= 0 && !source.removedNativeSwitches?.includes(item.id));
}

export function restoreRemovedNativeSwitches(value: unknown): readonly number[] {
  if (!Array.isArray(value) || value.length > 8) throw new Error('Removed native switches must be a list of at most eight switch numbers.');
  const slots = value.map(slot => checkedInteger(slot, 0, 7, 'Removed native switch'));
  if (new Set(slots).size !== slots.length) throw new Error('Removed native switches repeat a switch number.');
  return Object.freeze(slots);
}

export function validateRemovedNativeSwitches(document: LevelDocument) {
  for (const slot of restoreRemovedNativeSwitches(document.roomSource.removedNativeSwitches ?? [])) {
    if (!document.switches.some(item => item.id === slot && item.zone >= 0)) throw new Error(`Removed switch ${slot + 1} has no active native template.`);
  }
}

export function removedNativeSwitchPanel(document: LevelDocument, room: SourceRoom, offset: number): boolean {
  // Independent copied controllers still own their original template panel.
  // Copying an already stripped room must not resurrect an orphan panel.
  return (document.roomSource.removedNativeSwitches ?? []).some(slot => document.switches[slot]!.graphic - 10 === offset
    && (room.originZone !== null || !document.roomSource.switchCopies?.some(copy => copy.id === slot && copy.graphicsRoom === room.id)));
}

export function switchSourceRoom(document: LevelDocument, slot: number): string | undefined {
  const authored = document.roomSource.rooms.find(room => room.switches?.some(item => item.slot === slot));
  if (authored) return authored.id;
  const copy = document.roomSource.switchCopies?.find(item => item.slot === slot);
  if (copy) return copy.room;
  const original = activeNativeSwitches(document).find(item => item.id === slot);
  return original && document.roomSource.rooms.find(room => room.originZone === original.zone)?.id;
}
