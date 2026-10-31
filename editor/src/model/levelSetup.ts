import { BinaryReader, checkedInteger } from './binary';
import { BinaryWriter } from './writer';

/** Port extension, not an Amiga level field. ABEX + version 1 + signed zone:
 * -2 retains the slot's original exit, -1 disables completion, 0..255 selects
 * a runtime zone. Always export it so replacing a custom map clears old setup.
 */
export function encodeLevelSetup(exitZone: number | null): Uint8Array {
  const output = new BinaryWriter(); output.write(new Uint8Array([65, 66, 69, 88])); output.word(1);
  output.word(exitZone === null ? -2 : checkedInteger(exitZone, -1, 255, 'Exit zone')); return output.finish();
}
export function decodeLevelSetup(bytes: Uint8Array): number | null {
  const reader = new BinaryReader(bytes, 'twolev.cfg');
  if (bytes.length !== 8 || reader.u32(0) !== 0x41424558 || reader.u16(4) !== 1) throw new Error('Unsupported or damaged level exit setup.');
  const zone = checkedInteger(reader.i16(6), -2, 255, 'Exit zone'); return zone === -2 ? null : zone;
}
