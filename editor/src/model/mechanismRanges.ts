import { activeNativeMechanisms } from './nativeMechanisms';
import { numberPatch, type EditCommand } from './commands';
import type { LevelDocument } from './level';
import { assertMechanismRange } from './mechanismActivation';
export { mechanismNeedsEndpoint } from './mechanismActivation';

/** level.c expands 18-byte Amiga moving-room headers: top/bottom/current
 * words are quarter-world units, multiplied by 64 for runtime 24.8. Only
 * destinations change here; initial geometry and animation attachments stay valid. */
export function setMechanismRange(document: LevelDocument, kind: 'door' | 'lift', id: number, top: number, bottom: number): EditCommand {
  const mechanism = activeNativeMechanisms(document).find(item => item.kind === kind && item.id === id);
  if (!mechanism) throw new Error('That door or lift no longer exists.');
  assertMechanismRange(mechanism, top, bottom);
  const topWord = top * 4, bottomWord = bottom * 4;
  return { label: `Change ${kind} ${id} movement range`, patches: [
    numberPatch(document, 'graphics', mechanism.offset, bottomWord, 2),
    numberPatch(document, 'graphics', mechanism.offset + 2, topWord, 2),
  ] };
}
