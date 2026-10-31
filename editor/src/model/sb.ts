/* =SB= LH6 decoder. Contracts and tables come from src/sb_decompress.c.
 * Canonical Huffman trees replace the C decoder's lookahead tables, so bounds
 * are checked at the actual consumed bit rather than by padding corrupt input. */
import { BinaryReader, LevelFormatError } from './binary';

const MAX_UNPACKED_BYTES = 16 * 1024 * 1024;
const DICTIONARY_SIZE = 32768;

class Bits {
  position = 96;
  readonly source: Uint8Array;
  readonly file: string;

  constructor(source: Uint8Array, file: string) { this.source = source; this.file = file; }

  read(count: number): number {
    if (count < 0 || count > 16 || this.position + count > this.source.length * 8)
      throw new LevelFormatError('compressed stream is truncated', this.file, Math.floor(this.position / 8));
    let value = 0;
    for (let bit = 0; bit < count; bit++, this.position++)
      value = value * 2 + (((this.source[this.position >>> 3] ?? 0) >>> (7 - (this.position & 7))) & 1);
    return value;
  }

  fail(message: string): never {
    throw new LevelFormatError(message, this.file, Math.floor(this.position / 8));
  }
}

class Huffman {
  readonly codes: Map<number, number>[];
  readonly single: number | undefined;

  constructor(lengths: number[], bits: Bits, single?: number) {
    this.single = single;
    this.codes = Array.from({ length: 17 }, () => new Map<number, number>());
    if (single !== undefined) return;
    const counts = new Uint16Array(17);
    for (const length of lengths) {
      if (length < 0 || length > 16) bits.fail('invalid Huffman code length');
      counts[length] = (counts[length] ?? 0) + 1;
    }
    const next = new Uint32Array(17);
    let code = 0;
    for (let length = 1; length <= 16; length++) {
      code = (code + (length === 1 ? 0 : (counts[length - 1] ?? 0))) * 2;
      next[length] = code;
      if (code + (counts[length] ?? 0) > 2 ** length) bits.fail('oversubscribed Huffman table');
    }
    if (code + (counts[16] ?? 0) !== 65536) bits.fail('incomplete Huffman table');
    lengths.forEach((length, symbol) => {
      if (length > 0) {
        const value = next[length] ?? 0;
        this.codes[length]?.set(value, symbol);
        next[length] = value + 1;
      }
    });
  }

  decode(bits: Bits): number {
    if (this.single !== undefined) return this.single;
    let code = 0;
    for (let length = 1; length <= 16; length++) {
      code = code * 2 + bits.read(1);
      const symbol = this.codes[length]?.get(code);
      if (symbol !== undefined) return symbol;
    }
    return bits.fail('invalid Huffman symbol');
  }
}

function readLengths(bits: Bits, count: number, width: number, zeroPosition: number): Huffman {
  const used = bits.read(width);
  if (used === 0) {
    const symbol = bits.read(width);
    if (symbol >= count) bits.fail('Huffman symbol is outside its alphabet');
    return new Huffman([], bits, symbol);
  }
  if (used > count) bits.fail('Huffman table is larger than its alphabet');
  const lengths = Array<number>(count).fill(0);
  let index = 0;
  while (index < used) {
    let length = bits.read(3);
    if (length === 7) while (bits.read(1) !== 0) {
      if (++length > 16) bits.fail('Huffman code is longer than 16 bits');
    }
    lengths[index++] = length;
    if (index === zeroPosition) {
      const zeros = bits.read(2);
      if (index + zeros > count) bits.fail('zero run extends beyond the Huffman table');
      index += zeros;
    }
  }
  return new Huffman(lengths, bits);
}

function readCharacters(bits: Bits, lengthsTree: Huffman): Huffman {
  const used = bits.read(9);
  if (used === 0) {
    const symbol = bits.read(9);
    if (symbol >= 510) bits.fail('literal symbol is outside its alphabet');
    return new Huffman([], bits, symbol);
  }
  if (used > 510) bits.fail('literal table is larger than its alphabet');
  const lengths = Array<number>(510).fill(0);
  let index = 0;
  while (index < used) {
    const symbol = lengthsTree.decode(bits);
    if (symbol <= 2) {
      const zeros = symbol === 0 ? 1 : symbol === 1 ? bits.read(4) + 3 : bits.read(9) + 20;
      if (index + zeros > used) bits.fail('zero run extends beyond the literal table');
      index += zeros;
    } else lengths[index++] = symbol - 2;
  }
  return new Huffman(lengths, bits);
}

export function isSb(bytes: Uint8Array): boolean {
  return bytes.length >= 4 && bytes[0] === 61 && bytes[1] === 83 && bytes[2] === 66 && bytes[3] === 61;
}

export function unpackSb(source: Uint8Array, file = 'level file'): Uint8Array {
  if (source.length > MAX_UNPACKED_BYTES) throw new LevelFormatError('file exceeds the 16 MiB level-file limit', file);
  if (!isSb(source)) return new Uint8Array(source);
  const header = new BinaryReader(source, file);
  header.require(0, 12);
  const size = header.u32(4);
  const packedSize = header.u32(8);
  if (size === 0 || size > MAX_UNPACKED_BYTES)
    throw new LevelFormatError('unpacked size is outside the 1 byte to 16 MiB limit', file, 4);
  if (packedSize === 0 || packedSize > source.length - 12)
    throw new LevelFormatError('packed length exceeds the available file data', file, 8);
  const bits = new Bits(source.subarray(0, 12 + packedSize), file);
  const output = new Uint8Array(size);
  const ring = new Uint8Array(DICTIONARY_SIZE).fill(32);
  let out = 0, position = 0, remaining = 0;
  let characters: Huffman | undefined, distances: Huffman | undefined;
  while (out < size) {
    if (remaining === 0) {
      remaining = bits.read(16);
      if (remaining === 0) bits.fail('empty compression block');
      const lengths = readLengths(bits, 19, 5, 3);
      characters = readCharacters(bits, lengths);
      distances = readLengths(bits, 16, 5, -1);
    }
    remaining--;
    if (!characters || !distances) return bits.fail('missing compression tables');
    const symbol = characters.decode(bits);
    if (symbol < 256) {
      output[out++] = symbol;
      ring[position] = symbol;
      position = (position + 1) & 32767;
    } else {
      const count = symbol - 253;
      const prefix = distances.decode(bits);
      const distance = prefix === 0 ? 0 : 2 ** (prefix - 1) + bits.read(prefix - 1);
      if (out + count > size) bits.fail('match extends beyond the declared unpacked size');
      let from = (position - distance - 1) & 32767;
      for (let i = 0; i < count; i++) {
        const value = ring[from] ?? 0;
        from = (from + 1) & 32767;
        output[out++] = value;
        ring[position] = value;
        position = (position + 1) & 32767;
      }
    }
  }
  return output;
}
