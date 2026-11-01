// PNG signature/chunk layout and CRC: https://www.w3.org/TR/png/
const signature = [137, 80, 78, 71, 13, 10, 26, 10];
const crcTable = Array.from({ length: 256 }, (_, value) => {
  for (let bit = 0; bit < 8; bit++) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
  return value >>> 0;
});
export function pngDimensions(bytes: Uint8Array): { width: number; height: number } {
  if (bytes.length < 57 || bytes.length > 16 * 1024 * 1024 || signature.some((value, index) => bytes[index] !== value))
    throw new Error('Choose a valid PNG image smaller than 16 MiB.');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let width = 0, height = 0, data = false, idatSeen = false, endedData = false, palette = false, indexed = false, gray = false;
  for (let at = 8; at < bytes.length;) {
    if (at + 12 > bytes.length) throw new Error('The PNG has a truncated chunk.');
    const size = view.getUint32(at), end = at + 12 + size;
    if (end > bytes.length) throw new Error('The PNG has a truncated chunk.');
    const type = String.fromCharCode(...bytes.subarray(at + 4, at + 8));
    if (!/^[a-zA-Z]{4}$/.test(type) || (bytes[at + 6]! & 32)) throw new Error('The PNG chunk type is invalid.');
    let crc = 0xffffffff;
    for (let offset = at + 4; offset < end - 4; offset++) crc = crcTable[(crc ^ bytes[offset]!) & 255]! ^ (crc >>> 8);
    if (((crc ^ 0xffffffff) >>> 0) !== view.getUint32(end - 4)) throw new Error(`The PNG ${type} checksum is damaged.`);
    if (at === 8 && type !== 'IHDR') throw new Error('The PNG header must come first.');
    if (type === 'IHDR') {
      if (at !== 8 || size !== 13) throw new Error('The PNG header is invalid.');
      width = view.getUint32(at + 8); height = view.getUint32(at + 12);
      if (!width || !height || width > 4096 || height > 4096)
        throw new Error('PNG images must fit within 4096 × 4096.');
      const depth = bytes[at + 16]!, color = bytes[at + 17]!;
      const depths: Record<number, number[]> = { 0: [1, 2, 4, 8, 16], 2: [8, 16], 3: [1, 2, 4, 8], 4: [8, 16], 6: [8, 16] };
      if (!depths[color]?.includes(depth) || bytes[at + 18] !== 0 || bytes[at + 19] !== 0 || bytes[at + 20]! > 1)
        throw new Error('The PNG header has unsupported image settings.');
      indexed = color === 3;
      gray = color === 0 || color === 4;
    } else if (type === 'PLTE') {
      if (gray || palette || idatSeen || !size || size > 768 || size % 3) throw new Error('The PNG palette is invalid.');
      palette = true;
    } else if (type === 'IDAT') {
      if (endedData || (indexed && !palette)) throw new Error('The PNG image chunks are out of order.');
      idatSeen = true;
      data ||= size > 0;
    } else {
      if (idatSeen) endedData = true;
      if (type === 'IEND') {
        if (size || !data || end !== bytes.length) throw new Error('The PNG image ending is invalid.');
        return { width, height };
      }
      if (type === 'acTL') throw new Error('Choose a still PNG image; animated PNGs are not supported.');
      if (!(bytes[at + 4]! & 32)) throw new Error(`The PNG uses an unsupported ${type} chunk.`);
    }
    at = end;
  }
  throw new Error('The PNG image ending is missing.');
}
