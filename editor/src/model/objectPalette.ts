import { BinaryReader } from './binary';
import { unpackSb } from './sb';
import { originalObjectRecords, loadOriginalObjectData } from './objectTypes';
import { ammunitionChoices, weaponPickupChoices, keyColours } from './objectSettings';
import { createObjectRecord, objectTemplates } from './objectAuthoring';
import type { LevelDocument } from './level';
import { objectName } from './level';
import { editableObjects } from './sourceObjects';
import { roomSettings } from './roomHeights';

export interface ObjectPaletteChoice { readonly id: string; readonly name: string; readonly group: 'Enemies' | 'Pickups' | 'Scenery'; readonly record: readonly number[]; readonly floorOffset: number }
function identity(type: number, record: readonly number[]) {
  const reader = new BinaryReader(new Uint8Array(record), 'object palette');
  if (type === 4) {
    const colour = reader.i32(8) - 5 * 65536;
    if (colour < 0 || colour > 3 || reader.u8(17) !== 2 ** colour) return;
    return { id: `key-${colour}`, name: `${keyColours[colour]} key` };
  }
  if (type === 9) {
    const weapon = ammunitionChoices.find(item => item.id === reader.i16(18));
    return weapon ? { id: `ammo-${weapon.id}`, name: `${weapon.name} ammo` } : undefined;
  }
  if (type === 3) {
    const weapon = weaponPickupChoices.find(item => item.id === reader.u8(17));
    return weapon ? { id: `weapon-${weapon.id}`, name: weapon.name } : undefined;
  }
  return { id: `type-${type}`, name: objectName(type) };
}
/** Native source Y relative to its actual floor, preserving authored sprite/
 * model anchors instead of inventing per-class placement heights. */
export function originalObjectPalette(dataFiles: readonly Uint8Array[], graphFiles: readonly Uint8Array[]): readonly ObjectPaletteChoice[] {
  if (dataFiles.length !== graphFiles.length) throw new Error('The object library needs matching map and graphics files.');
  const data = dataFiles.map(file => new BinaryReader(unpackSb(file), 'original object floor'));
  const graphs = graphFiles.map(file => new BinaryReader(unpackSb(file), 'original object rooms'));
  const choices = new Map<string, ObjectPaletteChoice>();
  for (const object of originalObjectRecords(dataFiles)) {
    const named = identity(object.type, object.record);
    if (!named || choices.has(named.id)) continue;
    const record = new BinaryReader(new Uint8Array(object.record), 'original object');
    const graph = graphs[object.level - 1]!, reader = data[object.level - 1]!;
    const zone = record.i16(12), count = (graph.i32(12) - 16) / 4;
    if (zone < 0 || zone >= count) throw new Error('An original object has no usable room for its placement height.');
    const floor = reader.i32(graph.i32(16 + zone * 4) + (record.u8(63) ? 10 : 2)) / 256;
    choices.set(named.id, Object.freeze({ ...named, group: object.group, record: object.record, floorOffset: record.i16(4) / 2 - floor }));
  }
  return Object.freeze([...choices.values()].sort((a, b) => a.name.localeCompare(b.name)));
}
export function placePaletteObject(document: LevelDocument, choice: ObjectPaletteChoice, room: string, x: number, z: number, upper: boolean) {
  const target = document.roomSource.rooms.find(item => item.id === room);
  if (!target) throw new Error('Click inside a room to place an object.');
  const settings = roomSettings(document, target);
  if (upper && (!settings.upper.length || settings.upperFloor <= settings.upperCeiling)) throw new Error('This room has no upper floor. Choose Lower floor or another room.');
  const y = Math.round((settings[upper ? 'upperFloor' : 'floor'] + choice.floorOffset) * 2) / 2;
  return createObjectRecord(document, { room, x, z, y, upper }, new Uint8Array(choice.record), `Place ${choice.name}`);
}
export function duplicatePaletteChoice(document: LevelDocument, id: number): ObjectPaletteChoice {
  const object = editableObjects(document).find(item => item.id === id)!;
  if (!object || !objectTemplates(document).some(item => item.id === id)) throw new Error('Choose an enemy, pickup or scenery object to duplicate.');
  const room = document.roomSource.rooms.find(item => object.room ? item.id === object.room : item.originZone === object.zone);
  if (!room) throw new Error('Restore the object room before duplicating it.');
  const record = Object.freeze([...document.files.data.slice(object.offset, object.offset + 64)]);
  return Object.freeze({ id: `copy-${id}`, name: identity(object.type, record)?.name ?? objectName(object.type), group: [1, 3, 4, 9].includes(object.type) ? 'Pickups' : [10, 16, 20].includes(object.type) ? 'Scenery' : 'Enemies', record,
    floorOffset: object.y - roomSettings(document, room)[object.upper ? 'upperFloor' : 'floor'] });
}
let pending: Promise<readonly ObjectPaletteChoice[]> | undefined;
export function loadObjectPalette(): Promise<readonly ObjectPaletteChoice[]> {
  pending ??= Promise.all([loadOriginalObjectData(), Promise.all(Array.from({ length: 16 }, async (_, index) => {
    const response = await fetch(`${import.meta.env.BASE_URL}reference/level_${String.fromCharCode(97 + index)}/twolev.graph.bin`);
    if (!response.ok) throw new Error('Object placement heights could not be loaded. Check the editor assets and retry.');
    return new Uint8Array(await response.arrayBuffer());
  }))]).then(([data, graphs]) => originalObjectPalette(data, graphs)).catch(error => { pending = undefined; throw error; });
  return pending;
}
