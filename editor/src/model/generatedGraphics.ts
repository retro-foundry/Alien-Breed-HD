import { BinaryReader, checkedInteger } from './binary';
import { buildAnimationTables, type AnimationTables, type AnimationTargetContext } from './animationBuild';
import { readGraphicRecord, type Graphic } from './graphics';
import type { GraphicsStreamInput } from './graphicsBuild';
import type { LevelDocument } from './level';
import { BinaryWriter } from './writer';
import { relocateRecordPointer } from './graphicsRelocation';
import { assertGraphicsRecordBudget } from './renderLimits';

export interface GeneratedGraphicsRoom extends GraphicsStreamInput { id: number; dataOffset: number }
export interface GeneratedGraphicsTarget { room: number; upper: boolean; index: number }
export interface GeneratedGraphicIdentity extends GeneratedGraphicsTarget { readonly offset: number; readonly sourceOffset?: number; readonly boundaryIndex?: number }
export type SelectGraphicsTarget = (pointer: number, context: AnimationTargetContext,
  candidates: readonly GeneratedGraphicsTarget[]) => GeneratedGraphicsTarget;

/** Assemble the generated room set independently of the imported slot count.
 * Source record identities may occur in multiple cells. Live references must
 * select a concrete copy when ambiguous; never pick the first material copy.
 * Animation room/line/point relationships are supplied by the source compiler.
 */
export function buildGeneratedGraphics(document: LevelDocument, rooms: readonly GeneratedGraphicsRoom[],
  animations: AnimationTables, selectTarget?: SelectGraphicsTarget): Uint8Array {
  return compileGeneratedGraphics(document, rooms, animations, selectTarget).bytes;
}

export function compileGeneratedGraphics(document: LevelDocument, rooms: readonly GeneratedGraphicsRoom[],
  animations: AnimationTables, selectTarget?: SelectGraphicsTarget): { bytes: Uint8Array; records: readonly GeneratedGraphicIdentity[] } {
  checkedInteger(rooms.length, 1, 256, 'Generated graphics room count');
  const ids = new Set(rooms.map(room => room.id));
  if (ids.size !== rooms.length || rooms.some(room => !Number.isInteger(room.id) || room.id < 0 || room.id >= rooms.length))
    throw new Error('Generated graphics rooms need unique dense runtime identities.');
  const animationRooms = [...(animations.doors ?? []), ...(animations.lifts ?? []),
    ...animations.water.flatMap(group => group.entries), ...(animations.switches ?? []).filter(slot => slot.zone >= 0)];
  if (animationRooms.some(record => !ids.has(record.zone))) throw new Error('An animation refers to a missing generated room.');
  const source = new Map(document.zones.flatMap(zone => [...zone.lower, ...zone.upper].map(record => [record.offset, record] as const)));
  const output = new BinaryWriter(), table = 16 + rooms.length * 4;
  output.reserve(16 + rooms.length * 12); output.patchLong(12, table);
  const targets: { key: GeneratedGraphicsTarget; start: number; original: Graphic; generated: Graphic }[] = [];
  const records: GeneratedGraphicIdentity[] = [];
  for (const room of rooms) {
    output.patchLong(16 + room.id * 4, checkedInteger(room.dataOffset, 0, 16 * 1024 * 1024 - 1, 'Generated zone data offset'));
    for (const [upper, stream] of [[false, room.lower], [true, room.upper]] as const) {
      if (stream === undefined) continue;
      checkedInteger(stream.length, 0, 4095, 'Generated graphics record count');
      assertGraphicsRecordBudget(room.id, upper, stream.length);
      output.patchLong(table + room.id * 8 + (upper ? 4 : 0), output.offset); output.word(room.id);
      stream.forEach((item, index) => {
        const generated = readGraphicRecord(new BinaryReader(item.bytes, 'generated graphics record'), 0, 2048);
        if (generated.size !== item.bytes.length) throw new Error('A generated graphics record contains trailing or incomplete data.');
        if (item.sourceOffset !== undefined) {
          const original = source.get(item.sourceOffset);
          if (!original || original.type !== generated.type) throw new Error('A generated graphics identity must name a source record with the same opcode.');
          targets.push({ key: { room: room.id, upper, index }, start: output.offset, original, generated });
        }
        records.push(Object.freeze({ room: room.id, upper, index, offset: output.offset,
          ...(item.sourceOffset === undefined ? {} : { sourceOffset: item.sourceOffset }),
          ...(item.boundaryIndex === undefined ? {} : { boundaryIndex: item.boundaryIndex }) }));
        output.write(item.bytes);
      });
      output.long(-1);
    }
  }
  const resolve = (pointer: number, context: AnimationTargetContext): number => {
    const candidates = targets.filter(target => pointer >= target.original.offset && pointer < target.original.offset + target.original.size);
    if (!candidates.length) throw new Error(`Animation graphics target ${pointer} has no generated record.`);
    let target = candidates[0]!;
    if (candidates.length > 1) {
      if (!selectTarget) throw new Error(`Animation graphics target ${pointer} has multiple generated copies. Select its room and record explicitly.`);
      const selected = selectTarget(pointer, context, candidates.map(candidate => ({ ...candidate.key })));
      const match = candidates.find(candidate => candidate.key.room === selected.room && candidate.key.upper === selected.upper && candidate.key.index === selected.index);
      if (!match) throw new Error(`Animation graphics target ${pointer} selected a record outside its generated copies.`);
      target = match;
    }
    return relocateRecordPointer(target.original, target.generated, target.start, pointer);
  };
  const built = buildAnimationTables(animations, output.offset, resolve);
  output.write(built.bytes);
  output.patchLong(0, built.doors); output.patchLong(4, built.lifts); output.patchLong(8, built.switches);
  return { bytes: output.finish(), records: Object.freeze(records) };
}
