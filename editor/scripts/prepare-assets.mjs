// Copy immutable, already-released reference maps. No local build output or saves.
import { mkdir, copyFile, writeFile, readFile, readdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = new URL('../', import.meta.url);
const manifest = [];
for (let level = 1; level <= 16; level++) {
  const folder = `level_${String.fromCharCode(96 + level)}`;
  const target = new URL(`public/reference/${folder}/`, root);
  await mkdir(target, { recursive: true });
  for (const file of ['twolev.bin', 'twolev.graph.bin', 'twolev.clips'])
    await copyFile(new URL(`../amiga/data/levels/${folder}/${file}`, root), new URL(file, target));
  manifest.push({ level, name: `Level ${level}`, folder });
}
await writeFile(new URL('public/reference/manifest.json', root), `${JSON.stringify(manifest, null, 2)}\n`);
// One source of truth: package the actual runtime ENDZONES configuration.
const gameData = await readFile(new URL('../src/game_data.c', root), 'utf8');
const endZoneBody = /const\s+int16_t\s+end_zones\[16\]\s*=\s*\{([^}]+)\}/.exec(gameData)?.[1];
const exitZones = endZoneBody?.trim().split(/\s*,\s*/).map(value => Number(value.trim()));
if (!exitZones || exitZones.length !== 16 || exitZones.some(value => !Number.isInteger(value) || value < -1 || value > 255))
  throw new Error('Cannot package the runtime end_zones configuration. Check src/game_data.c.');
await writeFile(new URL('public/reference/game-config.json', root), `${JSON.stringify({ format: 'alien-breed-game-config', version: 1, exitZones })}\n`);
console.log(`Prepared ${manifest.length} original map triplets in ${fileURLToPath(new URL('public/reference/', root)).split(/[\\/]/).slice(-2).join('/')}`);

// io.c is the runtime authority for wall indices and ambiguous sheet dimensions.
const io = await readFile(new URL('../src/io.c', root), 'utf8');
const table = /\}\s*wall_texture_table\[\]\s*=\s*\{([\s\S]*?)\n\};/.exec(io)?.[1];
const overrides = /wall_dump_dim_override\[\]\s*=\s*\{([\s\S]*?)\n\};/.exec(io)?.[1];
if (!table || !overrides) throw new Error('Cannot read the native wall texture declarations in src/io.c.');
const entries = [...table.matchAll(/\{\s*"([^"]+)"\s*,\s*(\d+)\s*\}/g)];
if (!entries.length) throw new Error('The native wall texture table is empty.');
const rowOverrides = new Map([...overrides.matchAll(/\{\s*(\d+)\s*,\s*\d+\s*,\s*(\d+)\s*\}/g)].map(match => [Number(match[1]), Number(match[2])]));
const wallSource = new URL('../amiga/data/includes/walls/', root), wallTarget = new URL('public/reference/walls/', root);
const names = await readdir(wallSource); await mkdir(wallTarget, { recursive: true });
const walls = [];
for (const [id, entry] of entries.entries()) {
  const name = names.find(name => name.toLowerCase() === entry[1].toLowerCase());
  if (!name) throw new Error(`Required native wall texture ${entry[1]} is missing.`);
  const bytes = await readFile(new URL(name, wallSource));
  const size = bytes.subarray(0, 4).toString() === '=SB=' ? bytes.readUInt32BE(4) : bytes.length;
  if (size !== Number(entry[2]) || size <= 2048) throw new Error(`Wall ${name} does not match its native declared size.`);
  const pixels = size - 2048;
  const rows = rowOverrides.get(id) ?? [64, 32, 16, 8].find(rows => pixels % (rows * 2) === 0);
  if (!rows || pixels % (rows * 2)) throw new Error(`Wall ${name} needs reviewed native sheet dimensions.`);
  const file = `${id}.wad`; await copyFile(new URL(name, wallSource), new URL(file, wallTarget));
  walls.push({ id, name: entry[1], file, size, rows, width: pixels / (rows * 2) * 3 });
}
await writeFile(new URL('catalog.json', wallTarget), `${JSON.stringify({ format: 'alien-breed-wall-catalog', version: 1, walls }, null, 2)}\n`);
await writeFile(new URL('src/model/nativeWalls.json', root), `${JSON.stringify({ format: 'alien-breed-wall-catalog', version: 1, walls }, null, 2)}\n`);
console.log(`Prepared ${walls.length} native wall sheets and their runtime catalog.`);
// io.c's floor bank and FloorPalScaled parser; renderer.c selects 15 palettes.
const floorTarget = new URL('public/reference/floors/', root);
await mkdir(floorTarget, { recursive: true });
const floorTile = await readFile(new URL('../amiga/data/includes/floortile', root));
const paletteSource = await readFile(new URL('../amiga/data/pal/FloorPalScaled.s', root), 'utf8');
const floorPalette = Buffer.from(paletteSource.split(/\r?\n/).filter(line => line.trim()).flatMap(line => {
  const match = /^\s*dc\.b\s+\$([0-9a-f]{2}),\$([0-9a-f]{2})\s*$/i.exec(line);
  if (!match) throw new Error('Cannot decode the native FloorPalScaled source.');
  return match.slice(1).map(value => Number.parseInt(value, 16));
}));
const renderer = await readFile(new URL('../src/renderer.c', root), 'utf8');
const paletteLevels = Number(/^#define FLOOR_PAL_LEVEL_COUNT\s+(\d+)/m.exec(renderer)?.[1]);
if (floorTile.length !== 65536 || !Number.isSafeInteger(paletteLevels) || floorPalette.length !== paletteLevels * 512)
  throw new Error('Native floor bank and palette sizes do not match the renderer.');
await writeFile(new URL('floortile', floorTarget), floorTile);
await writeFile(new URL('palette.bin', floorTarget), floorPalette);
await writeFile(new URL('src/model/nativeFloor.json', root), `${JSON.stringify({ format: 'alien-breed-floor-bank', version: 1, size: floorTile.length, paletteLevels })}\n`);
for (const [name, tile, palette] of [['bump', 'BumpTile', 'BumpPalScaled'], ['smooth', 'SmoothBumpTile', 'SmoothBumpPalScaled']]) {
  const pixels = await readFile(new URL(`../amiga/data/gfx/${tile}`, root)), colours = await readFile(new URL(`../amiga/data/pal/${palette}`, root));
  if (pixels.length !== floorTile.length || colours.length !== floorPalette.length) throw new Error(`Native ${name} material resources do not match renderer dimensions.`);
  await writeFile(new URL(`${name}-tile`, floorTarget), pixels); await writeFile(new URL(`${name}-palette.bin`, floorTarget), colours);
}
console.log('Prepared native floor-bank materials and original palettes.');
const gameState = await readFile(new URL('../src/game_state.h', root), 'utf8');
const objectLimits = {};
for (const [key, symbol] of [['objects', 'MAX_OBJECTS'], ['sourceShotSlots', 'PLAYER_SHOT_SLOT_COUNT']]) {
  const value = Number(new RegExp(`^#define ${symbol}\\s+(\\d+)`, 'm').exec(gameState)?.[1]);
  if (!Number.isSafeInteger(value) || value <= 0) throw new Error(`Cannot read native ${symbol}.`);
  objectLimits[key] = value;
}
await writeFile(new URL('src/model/objectLimits.json', root), `${JSON.stringify(objectLimits, null, 2)}\n`);
const ammoFrames = /const int8_t ammo_graphic_table\[8\]\s*=\s*\{([^}]+)\}/.exec(gameData)?.[1].split(',').map(value => Number(value.trim()));
if (!ammoFrames || ammoFrames.length !== 8 || !ammoFrames.every(Number.isInteger)) throw new Error('Cannot read native ammunition artwork table.');
await writeFile(new URL('src/model/ammoFrames.json', root), `${JSON.stringify(ammoFrames)}\n`);
