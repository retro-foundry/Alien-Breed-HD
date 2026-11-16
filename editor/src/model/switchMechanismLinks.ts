import { checkedInteger } from './binary';
import { applyCommand, editRoomSource, type EditCommand } from './commands';
import type { LevelDocument } from './level';
import { availableSwitchSlots, setSourceSwitch } from './sourceSwitches';
import { captureMechanismSettings, transferMechanismSettings, type MechanismOwner } from './mechanismSettingsTransfer';

export type SwitchLinkMode = 'requirement' | 'activate';

/** Reuse the native requirement/rule editors; preserve ranges, live positions,
 * closing rules, attachments, keys and other switch requirements. The explicit
 * activate option selects the existing timer trigger: objects.c door_routine
 * mode 4 and lift_routine mode 2 (Amiga Anims.s DoorRoutine/LiftRoutine). */
export function linkSwitchMechanism(document: LevelDocument, owner: MechanismOwner, slot: number, required: boolean, mode: SwitchLinkMode = 'requirement'): EditCommand {
  checkedInteger(slot, 0, 7, 'Switch slot');
  if (mode !== 'requirement' && mode !== 'activate') throw new Error('Choose how the switch controls this mechanism.');
  if (required && availableSwitchSlots(document).includes(slot)) throw new Error('Place this wall switch before linking a mechanism.');
  const saved = captureMechanismSettings(document, owner), bit = 1 << (slot + 4);
  return { ...transferMechanismSettings(document, owner, {
    ...saved, conditions: required ? saved.conditions | bit : saved.conditions & ~bit,
    openRule: required && mode === 'activate' ? owner.kind === 'door' ? 4 : 2 : saved.openRule,
  }, { range: false, rules: required && mode === 'activate', requirements: true }),
    label: `${required ? 'Link' : 'Unlink'} switch ${slot + 1} ${required ? 'to' : 'from'} ${owner.kind}` };
}

/** Placement and its optional connection form one history entry. Compute both
 * with the existing commands before exposing any change to the workspace. */
export function createLinkedSwitch(document: LevelDocument, room: string, slot: number, edge: number, target?: MechanismOwner, mode: SwitchLinkMode = 'requirement'): EditCommand {
  const placement = editRoomSource(setSourceSwitch(document, room, slot, { edge, automatic: false }));
  if (!target) return placement;
  const placed = applyCommand(document, placement);
  const link = linkSwitchMechanism(placed, target, slot, true, mode);
  return { label: `Create switch ${slot + 1} for ${target.kind}`, patches: link.patches,
    roomSource: { before: document.roomSource, after: link.roomSource?.after ?? placed.roomSource } };
}
