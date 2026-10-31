import { describe, expect, it } from 'vitest';
import { readAmosProject, writeAmosProject, type AmosFiles } from '../src/model/amosProject';
import { BinaryWriter } from '../src/model/writer';

// Minimal blank AMOS source layout, not a shipped runtime map or a claim of
// live AMOS execution. Offsets are calculated independently from LEVELSAVE:
// 4 + 21*20 + 21*22 + 8*4 + 2 + 2 + 2 = 924; its extra empty zone is 490 bytes.
function blankSource(): AmosFiles {
  const data = new Uint8Array(3036), view = new DataView(data.buffer);
  view.setInt32(0, -1); view.setInt16(920, -1);
  view.setInt32(924, -1); view.setInt16(936, -15);
  view.setInt16(1024, -1); // Source-only zone's door assignment.
  for (let i = 0; i < 64; i++) { view.setInt16(1030 + i * 6, -1); view.setUint32(1032 + i * 6, 0xffffffff); }
  data[1434] = 0xe9; data[3033] = 255; // Preserve original single-byte text, including its final byte.
  view.setInt16(3034, -1);
  return { data, objects: Uint8Array.from({ length: 6400 }, (_, i) => i & 255), links: Uint8Array.from({ length: 10000 }, (_, i) => (i * 7) & 255) };
}

describe('original editable source-project format', () => {
  it('preserves the extra zone slot, original text bytes and entire object/link banks', () => {
    const files = blankSource(), source = readAmosProject(files);
    expect(source.zoneCount).toBe(0); expect(source.zones).toHaveLength(1);
    expect(source.points).toHaveLength(0); expect(source.controlPoints).toHaveLength(0);
    expect(source.zones[0]?.brightness).toBe(-15);
    expect(source.zones[0]?.door).toBe(-1);
    expect(source.zones[0]?.visibility[63]).toEqual({ zone: -1, mask: 0xffffffff });
    expect(source.introduction[0]).toBe(0xe9); expect(source.introduction[1599]).toBe(255);
    expect(writeAmosProject(source)).toEqual(files);
    files.objects.fill(0); files.links.fill(0);
    expect(source.objects[1]).toBe(1); expect(source.links[1]).toBe(7);
  });
  it('retains extension bytes instead of discarding unrecognised trailing data', () => {
    const files = blankSource(), extended = new Uint8Array(files.data.length + 6000);
    extended.set(files.data); extended.fill(0x91, files.data.length);
    const original = { ...files, data: extended };
    expect(writeAmosProject(readAmosProject(original))).toEqual(original);
  });
  it('rejects truncated or oversized record counts before unsafe allocation', () => {
    const files = blankSource();
    expect(() => readAmosProject({ ...files, data: files.data.subarray(0, 3000) })).toThrow(/past/);
    const broken = files.data.slice(); new DataView(broken.buffer).setInt16(944, 11);
    expect(() => readAmosProject({ ...files, data: broken })).toThrow(/corner/);
    new DataView(broken.buffer).setInt32(0, 100);
    expect(() => readAmosProject({ ...files, data: broken })).toThrow(/control point/);
    expect(() => readAmosProject({ ...files, links: new Uint8Array(1) })).toThrow(/100 × 100/);
  });
  it('bounds compiler output and rejects pointers outside written output', () => {
    const writer = new BinaryWriter();
    writer.reserve(4095); writer.byte(0x65); writer.byte(0xa5); writer.write(new Uint8Array(5000).fill(0x42));
    expect(writer.finish()[4095]).toBe(0x65); expect(writer.finish()[4096]).toBe(0xa5);
    expect(writer.finish()[9096]).toBe(0x42);
    expect(() => writer.patchLong(9097, 1)).toThrow(/outside/);
    expect(() => writer.reserve(16 * 1024 * 1024)).toThrow(/exceeds/);
  });
});
