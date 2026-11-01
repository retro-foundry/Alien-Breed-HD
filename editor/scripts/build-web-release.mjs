// Package one coherent native runtime under content-versioned URLs. Existing
// published pages can keep their runtime while a refreshed editor loads this one.
import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = new URL('../', import.meta.url), names = ['engine.js', 'engine.wasm', 'engine.data', 'frame.html', 'inspection-shortcuts.mjs'];
const hash = createHash('sha256'), files = [];
for (const name of names) {
  let bytes;
  try { bytes = await readFile(new URL(`public/runtime/${name}`, root)); }
  catch (cause) { throw new Error(`Cannot package runtime/${name}. Build the native editor runtime with tools/build_editor_runtime.py first.`, { cause }); }
  if (['frame.html', 'inspection-shortcuts.mjs'].includes(name) && !bytes.equals(await readFile(new URL(`runtime/${name}`, root))))
    throw new Error(`Runtime ${name} is stale. Rebuild with tools/build_editor_runtime.py before packaging.`);
  hash.update(name); hash.update('\0'); hash.update(bytes);
  files.push({ name, bytes, sha256: createHash('sha256').update(bytes).digest('hex') });
}
const revision = hash.digest('hex'), runtimePath = `runtime/r-${revision.slice(0, 16)}/`;
const env = { ...process.env, VITE_EDITOR_RUNTIME_PATH: runtimePath };
for (const args of [['scripts/prepare-assets.mjs'], ['node_modules/typescript/bin/tsc', '--noEmit'], ['node_modules/vite/bin/vite.js', 'build']]) {
  const result = spawnSync(process.execPath, args, { cwd: fileURLToPath(root), env, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
await mkdir(new URL(`dist/${runtimePath}`, root), { recursive: true });
for (const file of files) {
  await writeFile(new URL(`dist/${runtimePath}${file.name}`, root), file.bytes);
  // Remove only Vite's unversioned copies, never a source or published runtime.
  await rm(new URL(`dist/runtime/${file.name}`, root));
}
await writeFile(new URL('dist/build-info.json', root), JSON.stringify({ format: 'alien-breed-editor-web-build', version: 1,
  runtimeRevision: revision, runtimePath, files: files.map(({ name, sha256 }) => ({ name, sha256 })) }, null, 2) + '\n');
console.log(`Packaged editor with native runtime ${runtimePath}`);
