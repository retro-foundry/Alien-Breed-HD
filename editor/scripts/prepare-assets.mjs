// Copy immutable, already-released reference maps. No local build output or saves.
import { mkdir, copyFile, writeFile, readFile } from 'node:fs/promises';
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
