import { numberPatch, type EditCommand } from './commands';
import type { LevelDocument } from './level';
import { assertMechanismActivation } from './mechanismActivation';
export { mechanismNeedsEndpoint } from './mechanismActivation';

/** level.c expands 18-byte Amiga moving-room headers: top/bottom/current
 * words are quarter-world units, multiplied by 64 for runtime 24.8. Only
 * destinations change here; initial geometry and animation attachments stay valid. */
export function setMechanismRange(document: LevelDocument, kind: 'door' | 'lift', id: number, top: number, bottom: number): EditCommand {
  const mechanism = document.mechanisms.find(item => item.kind === kind && item.id === id);
  if (!mechanism) throw new Error('That door or lift no longer exists.');
  for (const [name, value] of [['Upper', top], ['Lower', bottom]] as const)
    if (!Number.isFinite(value) || !Number.isInteger(value * 4) || value < -8192 || value > 8191.75)
      throw new Error(`${name} destination must be between -8192 and 8191.75 in 0.25-unit steps.`);
  const topWord = top * 4, bottomWord = bottom * 4;
  if (bottomWord === 999) throw new Error('The lower destination 249.75 cannot be stored in this level format. Choose 249.5 or 250 instead.');
  if (top > bottom) throw new Error('The upper destination must be above or equal to the lower destination. Smaller height values are higher.');
  if (mechanism.current < top || mechanism.current > bottom) throw new Error(`The range must include the initial position ${mechanism.current}. No animation ranges were changed.`);
  // door_routine/lift_routine normally evaluate trigger rules at endpoints.
  // Moving records may start in transit; an idle record must retain an endpoint
  // so a range edit cannot strand it between its own activation states.
  assertMechanismActivation({ ...mechanism, top, bottom });
  return { label: `Change ${kind} ${id} movement range`, patches: [
    numberPatch(document, 'graphics', mechanism.offset, bottomWord, 2),
    numberPatch(document, 'graphics', mechanism.offset + 2, topWord, 2),
  ] };
}
