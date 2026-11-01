import { BinaryReader, checkedInteger } from './binary';
import { numberPatch, type EditCommand } from './commands';
import type { LevelDocument } from './level';

// leved.amos KEYPUT / KEYSAVE: colour CO selects 5*65536+CO artwork
// and 2^CO in can_see. objects.c object_handle_key ORs that byte into conditions.
export const keyColours = ['Green', 'Red', 'Yellow', 'Blue'] as const;
export function keyColour(document: LevelDocument, id: number): number | undefined {
  const object = document.objects.find(object => object.id === id);
  if (!object || object.type !== 4) return undefined;
  const data = new BinaryReader(document.files.data, 'key settings');
  const colour = data.i32(object.offset + 8) - 5 * 65536;
  return colour >= 0 && colour < keyColours.length && data.u8(object.offset + 17) === 2 ** colour ? colour : undefined;
}
export function setKeyColour(document: LevelDocument, id: number, colour: number): EditCommand {
  checkedInteger(colour, 0, keyColours.length - 1, 'Key colour');
  const object = document.objects.find(object => object.id === id);
  if (!object || object.type !== 4) throw new Error('Select a key pickup to change its colour.');
  return { label: `Change key ${id} to ${keyColours[colour]}`, patches: [
    numberPatch(document, 'data', object.offset + 8, 5 * 65536 + colour, 4),
    numberPatch(document, 'data', object.offset + 17, 2 ** colour, 1),
  ] };
}
