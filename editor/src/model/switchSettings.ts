import type { LevelDocument } from './level';
import { checkedInteger } from './binary';
import { editRoomSource, numberPatch, type EditCommand } from './commands';
import { compileSwitchCopies, setRoomSwitchCopies } from './switchCopies';

/** Anims.s backtoend tests byte +2 for zero/nonzero. Preserve an existing
 * nonzero value and all countdown/state/attachment bytes when changing mode. */
function resetType(current: number, enabled: boolean): number {
  if (typeof enabled !== 'boolean') throw new Error('Automatic switch reset must be enabled or disabled.');
  return enabled ? current || 1 : 0;
}

export function setSwitchAutomaticReset(document: LevelDocument, id: number, enabled: boolean): EditCommand {
  checkedInteger(id, 0, 7, 'Switch slot');
  const item = document.switches.find(item => item.id === id && item.zone >= 0);
  if (!item) throw new Error('That active switch no longer exists.');
  return { label: `Change switch ${id + 1} reset mode`, patches: [numberPatch(document, 'graphics', item.offset + 2, resetType(item.type, enabled), 1, false)] };
}

export function setCopiedSwitchAutomaticReset(document: LevelDocument, slot: number, enabled: boolean): EditCommand {
  checkedInteger(slot, 0, 7, 'Copied switch slot');
  const copies = [...compileSwitchCopies(document.roomSource.switchCopies ?? [])], item = copies.find(item => item.slot === slot);
  if (!item) throw new Error('That copied switch no longer exists.');
  item.bytes![2] = resetType(item.bytes![2]!, enabled);
  return { ...editRoomSource(setRoomSwitchCopies(document, copies)), label: `Change copied switch ${slot + 1} reset mode` };
}
