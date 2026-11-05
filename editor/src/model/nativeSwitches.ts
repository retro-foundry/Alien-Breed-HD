import { checkedInteger } from './binary';
import type { LevelDocument } from './level';
import type { RoomSource, SourceRoom } from './roomSource';

/** Original records remain templates for independent copies after removal. */
export function activeNativeSwitches(document: LevelDocument, source: RoomSource = document.roomSource) {
  return document.switches.filter(item => item.zone >= 0 && !source.removedNativeSwitches?.includes(item.id));
}

export function restoreRemovedNativeSwitches(value: unknown, label = 'Removed native switches'): readonly number[] {
  if (!Array.isArray(value) || value.length > 8) throw new Error(`${label} must be a list of at most eight switch numbers.`);
  const slots = value.map(slot => checkedInteger(slot, 0, 7, label));
  if (new Set(slots).size !== slots.length) throw new Error(`${label} repeat a switch number.`);
  return Object.freeze(slots);
}

export function restoreSwitchSettings(value: unknown): readonly number[] {
  if (!Array.isArray(value) || value.length !== 14) throw new Error('Switch settings need a complete 14-byte native record.');
  return Object.freeze(value.map(byte => checkedInteger(byte, 0, 255, 'Switch settings byte')));
}

export function validateRemovedNativeSwitches(document: LevelDocument) {
  for (const slot of restoreRemovedNativeSwitches(document.roomSource.removedNativeSwitches ?? [])) {
    if (!document.switches.some(item => item.id === slot && item.zone >= 0)) throw new Error(`Removed switch ${slot + 1} has no active native template.`);
  }
  for (const slot of restoreRemovedNativeSwitches(document.roomSource.clearedSwitchSlots ?? [], 'Retired copied switches')) {
    if (!document.switches[slot] || activeNativeSwitches(document).some(item => item.id === slot)) throw new Error(`Retired copied switch ${slot + 1} would clear an active original switch or has no native slot.`);
  }
  for (const room of document.roomSource.rooms) for (const id of restoreRemovedNativeSwitches(room.retiredCopiedSwitches ?? [], 'Retired copied switch attachments')) {
    const original = document.switches[id], owner = document.zones.find(zone => zone.id === room.recordZone);
    if (room.originZone !== null || !original || original.zone < 0 || !owner || original.zone !== owner.id && ![...owner.lower, ...owner.upper].some(record => record.offset === original.graphic - 10))
      throw new Error(`Room ${room.id} has a retired switch attachment without a matching copied native owner.`);
    if (document.roomSource.switchCopies?.some(copy => copy.id === id && (copy.graphicsRoom === room.id || copy.room === room.id))) throw new Error(`Room ${room.id} still has a controller for its retired switch attachment.`);
  }
}

export function removedNativeSwitchPanel(document: LevelDocument, room: SourceRoom, offset: number): boolean {
  // Independent copied controllers still own their original template panel.
  // Copying an already stripped room must not resurrect an orphan panel.
  return (room.retiredCopiedSwitches ?? []).some(slot => document.switches[slot]!.graphic - 10 === offset)
    || (document.roomSource.removedNativeSwitches ?? []).some(slot => document.switches[slot]!.graphic - 10 === offset
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
