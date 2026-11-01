import { BinaryReader, checkedInteger } from './binary';
import { numberPatch, type EditCommand } from './commands';
import type { LevelDocument } from './level';
import ammoFrames from './ammoFrames.json' with { type: 'json' };

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

// game_data.h AB3D_GUN_* and objects.c object_take_ammo_clip: original
// player weapons only; flamethrower ammo is explicitly rejected at runtime.
export const ammunitionChoices = [
  { id: 0, name: 'Pistol' }, { id: 1, name: 'Plasma gun' },
  { id: 2, name: 'Rocket launcher' }, { id: 4, name: 'Grenade launcher' },
  { id: 7, name: 'Shotgun' },
] as const;
export function ammunitionType(document: LevelDocument, id: number): number | undefined {
  const object = document.objects.find(object => object.id === id);
  if (!object || object.type !== 9) return undefined;
  const data = new BinaryReader(document.files.data, 'ammunition settings'), type = data.i16(object.offset + 18);
  return ammunitionChoices.some(choice => choice.id === type) && data.i32(object.offset + 8) === 65536 + ammoFrames[type]! ? type : undefined;
}
export function setAmmunitionType(document: LevelDocument, id: number, type: number): EditCommand {
  const choice = ammunitionChoices.find(choice => choice.id === type);
  if (!choice) throw new Error('Choose ammunition for an original player weapon. Flamethrower and unused slots do not support pickups.');
  const object = document.objects.find(object => object.id === id);
  if (!object || object.type !== 9) throw new Error('Select an ammunition pickup to change its weapon.');
  // object_handle_ammo reads AmmoType at +18 and resets the artwork from
  // ammo_graphic_table each tick. Match the initial inspection frame too.
  return { label: `Change ammunition ${id} to ${choice.name}`, patches: [
    numberPatch(document, 'data', object.offset + 18, type, 2),
    numberPatch(document, 'data', object.offset + 8, 65536 + ammoFrames[type]!, 4),
  ] };
}
