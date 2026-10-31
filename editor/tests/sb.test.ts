import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import oracles from './fixtures/original-levels.json';
import { unpackSb } from '../src/model/sb';

describe('original =SB= LH6 files', () => {
  for (const level of oracles.levels) for (const [name, oracle] of Object.entries(level.files)) {
    it(`matches the independent C decoder for level ${level.level} ${name}`, () => {
      const path = new URL(`../../amiga/data/levels/level_${String.fromCharCode(96 + level.level)}/${name}`, import.meta.url);
      const raw = unpackSb(readFileSync(path), name);
      expect(raw.length).toBe(oracle.size);
      expect(createHash('sha256').update(raw).digest('hex')).toBe(oracle.sha256);
    });
  }
  it('keeps uncompressed bytes unchanged without sharing the input buffer', () => {
    const source = new Uint8Array([1, 2, 3]);
    const raw = unpackSb(source);
    expect(raw).toEqual(source);
    expect(raw).not.toBe(source);
  });
  it('rejects truncated headers, truncated streams and excessive allocations', () => {
    expect(() => unpackSb(new TextEncoder().encode('=SB='))).toThrow(/past/);
    const header = new Uint8Array(12);
    header.set([61, 83, 66, 61]);
    new DataView(header.buffer).setUint32(4, 32 * 1024 * 1024);
    expect(() => unpackSb(header)).toThrow(/16 MiB/);
    const source = readFileSync(new URL('../../amiga/data/levels/level_a/twolev.bin', import.meta.url));
    for (const length of [12, 32, 1024, source.length - 8])
      expect(() => unpackSb(source.subarray(0, length))).toThrow();
  });
});
