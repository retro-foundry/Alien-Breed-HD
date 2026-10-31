export class LevelFormatError extends Error {
  constructor(message: string, file: string, offset?: number) {
    super(`${file}${offset === undefined ? '' : ` at byte ${offset}`}: ${message}`);
    this.name = 'LevelFormatError';
  }
}

export class BinaryReader {
  readonly bytes: Uint8Array;
  readonly name: string;
  private readonly view: DataView;

  constructor(bytes: Uint8Array, name: string) {
    this.bytes = bytes;
    this.name = name;
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }

  require(offset: number, length: number): void {
    if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0 || offset + length > this.bytes.length)
      throw new LevelFormatError(`record extends past the ${this.bytes.length}-byte file`, this.name, offset);
  }

  u8(offset: number): number { this.require(offset, 1); return this.view.getUint8(offset); }
  i8(offset: number): number { this.require(offset, 1); return this.view.getInt8(offset); }
  u16(offset: number): number { this.require(offset, 2); return this.view.getUint16(offset); }
  i16(offset: number): number { this.require(offset, 2); return this.view.getInt16(offset); }
  i32(offset: number): number { this.require(offset, 4); return this.view.getInt32(offset); }
  u32(offset: number): number { this.require(offset, 4); return this.view.getUint32(offset); }

  wordsUntil(offset: number, terminator: number, limit: number): number[] {
    const words: number[] = [];
    for (let i = 0; i < limit; i++) {
      const word = this.i16(offset + i * 2);
      if (word === terminator) return words;
      words.push(word);
    }
    throw new LevelFormatError(`list has no ${terminator} terminator within ${limit} entries`, this.name, offset);
  }
}

export function checkedInteger(value: number, minimum: number, maximum: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum)
    throw new Error(`${label} must be a whole number between ${minimum} and ${maximum}.`);
  return value;
}
