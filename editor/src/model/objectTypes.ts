import { BinaryReader } from './binary';
import { unpackSb } from './sb';
import limits from './objectLimits.json' with { type: 'json' };
import { objectName, type LevelDocument } from './level';
import { replaceObjectRecord } from './objectAuthoring';

export interface ObjectTypeChoice { readonly type: number; readonly name: string; readonly group: 'Enemies' | 'Pickups' | 'Scenery'; readonly record: readonly number[]; readonly level: number; readonly object: number }
const group = (type: number): ObjectTypeChoice['group'] => [1, 3, 4, 9].includes(type) ? 'Pickups' : [10, 16, 20].includes(type) ? 'Scenery' : 'Enemies';

/** Defs.i / level.c's 64-byte source records, excluding avatar and shot pools.
 * Use pristine released records, not a user's modified instance. All runtime
 * sprite/vector banks are shared across the original maps (io.c). */
export function originalObjectRecords(files: readonly Uint8Array[]): readonly ObjectTypeChoice[] {
  const choices: ObjectTypeChoice[] = [];
  files.forEach((file, index) => {
    const reader = new BinaryReader(unpackSb(file, `Level ${index + 1} objects`), 'original object library');
    const start = reader.i32(30), shotStarts = [reader.i32(34), reader.i32(38)], avatars = [reader.i32(46), reader.i32(50)];
    let terminated = false;
    for (let id = 0; id < limits.objects; id++) {
      const at = start + id * 64;
      if (reader.i16(at) < 0) { terminated = true; break; }
      reader.require(at, 64);
      const type = reader.i8(at + 16), zone = reader.i16(at + 12);
      if (type < 0 || type > 20 || [2, 5, 11].includes(type) || zone < 0 || avatars.includes(at)
        || at >= shotStarts[0]! && at < shotStarts[0]! + limits.sourceShotSlots * 64
        || at >= shotStarts[1]! && at < shotStarts[1]! + limits.sourceShotSlots * 128) continue;
      choices.push(Object.freeze({ type, name: objectName(type), group: group(type),
        record: Object.freeze([...reader.bytes.slice(at, at + 64)]), level: index + 1, object: id }));
    }
    if (!terminated) throw new Error('The original object library contains an unterminated object table.');
  });
  return Object.freeze(choices);
}
export function originalObjectTypes(files: readonly Uint8Array[]): readonly ObjectTypeChoice[] {
  const types = new Map<number, ObjectTypeChoice>();
  for (const choice of originalObjectRecords(files)) if (!types.has(choice.type)) types.set(choice.type, choice);
  return Object.freeze([...types.values()].sort((a, b) => a.name.localeCompare(b.name)));
}
export function changeObjectType(document: LevelDocument, id: number, choice: ObjectTypeChoice) {
  if (choice.record[16] !== choice.type) throw new Error('The selected object type does not match its source record. Reload the object library.');
  return replaceObjectRecord(document, id, new Uint8Array(choice.record), `Change object ${id} to ${choice.name}`);
}
let dataFiles: Promise<readonly Uint8Array[]> | undefined;
export function loadOriginalObjectData(): Promise<readonly Uint8Array[]> {
  dataFiles ??= Promise.all(Array.from({ length: 16 }, async (_, index) => {
    const response = await fetch(`${import.meta.env.BASE_URL}reference/level_${String.fromCharCode(97 + index)}/twolev.bin`);
    if (!response.ok) throw new Error('Object types could not be loaded. Check the editor asset connection and retry.');
    return new Uint8Array(await response.arrayBuffer());
  })).catch(error => { dataFiles = undefined; throw error; });
  return dataFiles;
}
let pending: Promise<readonly ObjectTypeChoice[]> | undefined;
export function loadObjectTypes(): Promise<readonly ObjectTypeChoice[]> {
  pending ??= loadOriginalObjectData().then(originalObjectTypes).catch(error => { pending = undefined; throw error; });
  return pending;
}
