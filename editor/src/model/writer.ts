import { checkedInteger } from './binary';

/** Checked big-endian output for original source and generated engine files. */
export class BinaryWriter {
  private bytes = new Uint8Array(4096);
  private used = 0;
  get offset(): number { return this.used; }

  reserve(count: number): number {
    checkedInteger(count, 0, 16 * 1024 * 1024, 'Record size');
    const end = this.used + count;
    if (end > 16 * 1024 * 1024) throw new Error('The generated level exceeds the 16 MiB file limit.');
    if (end > this.bytes.length) {
      const grown = new Uint8Array(Math.max(end, Math.min(this.bytes.length * 2, 16 * 1024 * 1024)));
      grown.set(this.bytes); this.bytes = grown;
    }
    const start = this.used; this.used = end; return start;
  }
  byte(value: number): void { checkedInteger(value, 0, 255, 'Byte'); const offset = this.reserve(1); this.bytes[offset] = value; }
  word(value: number): void { this.patchWord(this.reserve(2), value); }
  long(value: number): void { this.patchLong(this.reserve(4), value); }
  write(bytes: Uint8Array): void { const offset = this.reserve(bytes.length); this.bytes.set(bytes, offset); }
  patchWord(offset: number, value: number): void {
    checkedInteger(value, -32768, 65535, 'Word'); this.checkPatch(offset, 2);
    new DataView(this.bytes.buffer).setUint16(offset, value);
  }
  patchLong(offset: number, value: number): void {
    checkedInteger(value, -2147483648, 4294967295, 'Long word'); this.checkPatch(offset, 4);
    new DataView(this.bytes.buffer).setUint32(offset, value);
  }
  finish(): Uint8Array { return this.bytes.slice(0, this.used); }
  private checkPatch(offset: number, length: number) {
    if (!Number.isInteger(offset) || offset < 0 || offset + length > this.used)
      throw new Error('A compiler patch points outside the output written so far.');
  }
}
