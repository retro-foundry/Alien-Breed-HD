import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { exportLevel, openLevel, type LevelDocument, type LevelFiles } from './level';

const MAX_ARCHIVE = 64 * 1024 * 1024;
const MAX_FILE = 16 * 1024 * 1024;
const LEVEL_NAMES = ['twolev.bin', 'twolev.graph.bin', 'twolev.clips'];
const MANIFEST = 'alien-breed-project.json';
export interface ProjectCandidate { name: string; folder: string; files: LevelFiles }

export function encodeProject(document: LevelDocument): Uint8Array {
  const files = exportLevel(document);
  return zipSync({
    [MANIFEST]: strToU8(JSON.stringify({ format: 'alien-breed-level-project', version: 1, name: document.name })),
    'level/twolev.bin': files.data, 'level/twolev.graph.bin': files.graphics, 'level/twolev.clips': files.clips,
  }, { level: 6 });
}

export function encodeGameLevel(document: LevelDocument): Uint8Array {
  const files = exportLevel(document);
  return zipSync({ 'twolev.bin': files.data, 'twolev.graph.bin': files.graphics, 'twolev.clips': files.clips }, { level: 6 });
}

function candidates(entries: Record<string, Uint8Array>, title: string): ProjectCandidate[] {
  let projectName: string | undefined;
  const manifest = entries[MANIFEST];
  if (manifest) {
    if (manifest.length > 16384) throw new Error('The project manifest is too large.');
    let metadata: unknown;
    try { metadata = JSON.parse(strFromU8(manifest)); }
    catch { throw new Error('The project manifest is not valid JSON.'); }
    if (!metadata || typeof metadata !== 'object' || !('format' in metadata) || metadata.format !== 'alien-breed-level-project' || !('version' in metadata) || metadata.version !== 1)
      throw new Error('This project format is not supported by this editor version.');
    if ('name' in metadata && typeof metadata.name === 'string') projectName = metadata.name.slice(0, 200);
  }
  const groups = new Map<string, Record<string, Uint8Array>>();
  for (const [name, bytes] of Object.entries(entries)) {
    const path = name.replaceAll('\\', '/');
    if (path.includes('\0') || path.split('/').includes('..')) throw new Error('The archive contains an unsafe file path.');
    const base = path.split('/').at(-1)?.toLowerCase() ?? '';
    if (!LEVEL_NAMES.includes(base)) continue;
    const folder = path.slice(0, path.length - base.length);
    const group = groups.get(folder) ?? {};
    if (group[base]) throw new Error(`The archive contains duplicate ${base} files in the same level.`);
    if (bytes.length > MAX_FILE) throw new Error(`${base} exceeds the 16 MiB level-file limit.`);
    group[base] = bytes;
    groups.set(folder, group);
  }
  const result: ProjectCandidate[] = [];
  for (const [folder, files] of groups) {
    const data = files['twolev.bin'], graphics = files['twolev.graph.bin'], clips = files['twolev.clips'];
    if (!data || !graphics || !clips) throw new Error(`The level in ${folder || 'this selection'} needs all three files: twolev.bin, twolev.graph.bin and twolev.clips.`);
    const letter = /level_([a-p])\/$/i.exec(folder)?.[1];
    result.push({ name: projectName ?? (letter ? `Level ${letter.toLowerCase().charCodeAt(0) - 96}` : title), folder, files: { data, graphics, clips } });
  }
  if (result.length === 0) throw new Error('No game level was found. Open an .ab3d project, a level ZIP, or select all three twolev files together.');
  return result.sort((a, b) => a.folder.localeCompare(b.folder, undefined, { numeric: true }));
}

export function openArchive(bytes: Uint8Array, name: string): ProjectCandidate[] {
  if (bytes.length > MAX_ARCHIVE) throw new Error('The archive exceeds the 64 MiB import limit.');
  let extracted = 0;
  let entries: Record<string, Uint8Array>;
  try {
    entries = unzipSync(bytes, { filter(file) {
      const base = file.name.replaceAll('\\', '/').split('/').at(-1)?.toLowerCase() ?? '';
      if (!LEVEL_NAMES.includes(base) && file.name !== MANIFEST) return false;
      if (file.originalSize > MAX_FILE || (extracted += file.originalSize) > MAX_ARCHIVE)
        throw new Error('The archive contains too much unpacked level data.');
      return true;
    } });
  } catch (error) {
    throw new Error(`Could not open ${name}: ${error instanceof Error ? error.message : 'invalid ZIP data'}`);
  }
  return candidates(entries, name.replace(/\.(ab3d|zip)$/i, ''));
}

export function decodeProject(bytes: Uint8Array): LevelDocument {
  const projects = openArchive(bytes, 'Recovered project');
  if (projects.length !== 1) throw new Error('A saved project must contain exactly one level.');
  const project = projects[0]!;
  return openLevel(project.name, project.files);
}

export async function readFiles(files: File[]): Promise<ProjectCandidate[]> {
  if (files.length === 1 && /\.(zip|ab3d)$/i.test(files[0]?.name ?? '')) {
    const file = files[0]!;
    if (file.size > MAX_ARCHIVE) throw new Error('The archive exceeds the 64 MiB import limit.');
    return openArchive(new Uint8Array(await file.arrayBuffer()), file.name);
  }
  const entries: Record<string, Uint8Array> = {};
  for (const file of files) {
    if (file.size > MAX_FILE) throw new Error(`${file.name} exceeds the 16 MiB level-file limit.`);
    const name = file.webkitRelativePath || file.name;
    if (entries[name]) throw new Error(`More than one file is named ${name}.`);
    entries[name] = new Uint8Array(await file.arrayBuffer());
  }
  return candidates(entries, 'Imported level');
}
