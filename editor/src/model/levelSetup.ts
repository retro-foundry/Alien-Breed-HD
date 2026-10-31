import { BinaryReader, checkedInteger } from './binary';
import { BinaryWriter } from './writer';

/** Port extension, not an Amiga level field. ABEX + version 1 + signed zone:
 * -2 retains the slot's original exit, -1 disables completion, 0..255 selects
 * a runtime zone. Always export it so replacing a custom map clears old setup.
 * Version 2 appends a count and representative draw-room IDs, one per runtime
 * slot. This separates authored rendering from convex collision decomposition.
 */
export function validateDrawGroups(groups: readonly number[]): void {
  checkedInteger(groups.length, 1, 256, 'Draw room count');
  groups.forEach(group => checkedInteger(group, 0, groups.length - 1, 'Draw room owner'));
  if (groups.some(group => groups[group] !== group)) throw new Error('Draw room owners must identify their own representative room.');
}
export function encodeLevelSetup(exitZone: number | null, drawGroups?: readonly number[]): Uint8Array {
  if (drawGroups) validateDrawGroups(drawGroups);
  const output = new BinaryWriter(); output.write(new Uint8Array([65, 66, 69, 88])); output.word(drawGroups ? 2 : 1);
  output.word(exitZone === null ? -2 : checkedInteger(exitZone, -1, 255, 'Exit zone'));
  if (drawGroups) { output.word(drawGroups.length); drawGroups.forEach(group => output.word(group)); }
  return output.finish();
}
export function decodeLevelSetupDetails(bytes: Uint8Array): { exitZone: number | null; drawGroups?: readonly number[] } {
  const reader = new BinaryReader(bytes, 'twolev.cfg');
  if (bytes.length < 8 || reader.u32(0) !== 0x41424558 || ![1, 2].includes(reader.u16(4))) throw new Error('Unsupported or damaged level exit setup.');
  const zone = checkedInteger(reader.i16(6), -2, 255, 'Exit zone'), exitZone = zone === -2 ? null : zone;
  if (reader.u16(4) === 1) {
    if (bytes.length !== 8) throw new Error('Unsupported or damaged level exit setup.'); return { exitZone };
  }
  if (bytes.length < 10) throw new Error('Unsupported or damaged draw room setup.');
  const count = checkedInteger(reader.u16(8), 1, 256, 'Draw room count');
  if (bytes.length !== 10 + count * 2) throw new Error('Unsupported or damaged draw room setup.');
  const drawGroups = Array.from({ length: count }, (_, index) => reader.u16(10 + index * 2)); validateDrawGroups(drawGroups);
  return { exitZone, drawGroups };
}
export function decodeLevelSetup(bytes: Uint8Array): number | null { return decodeLevelSetupDetails(bytes).exitZone; }
