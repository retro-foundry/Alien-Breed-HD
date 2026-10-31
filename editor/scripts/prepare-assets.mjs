// Copy immutable, already-released reference maps. No local build output or saves.
import { mkdir, copyFile, writeFile } from 'node:fs/promises';
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
console.log(`Prepared ${manifest.length} original map triplets in ${fileURLToPath(new URL('public/reference/', root)).split(/[\\/]/).slice(-2).join('/')}`);
