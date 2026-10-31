import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';
import { exportLevel, openLevel, type LevelDocument, type LevelFiles } from './level';
import { decodeLevelSetupDetails, encodeLevelSetup } from './levelSetup';
import { roomSourceSnapshot, restoreRoomSource, sourceMatchesLevel } from './roomSource';
import { restoreSourceStarts, type SourceStart } from './sourceStarts';
import { restoreSourceObjects, validateSourceObjects, type SourceObject } from './sourceObjects';

const MAX_ARCHIVE = 64 * 1024 * 1024;
const MAX_FILE = 16 * 1024 * 1024;
const LEVEL_NAMES = ['twolev.bin', 'twolev.graph.bin', 'twolev.clips', 'twolev.cfg'];
const MANIFEST = 'alien-breed-project.json';
const ROOM_SOURCE = 'room-source.json';
const PLAYER_STARTS = 'player-starts.json';
const OBJECT_PLACEMENTS = 'object-placements.json';
export interface ProjectCandidate { name: string; folder: string; files: LevelFiles; playtestSlot: number; exitZone: number | null; roomSourceSnapshot?: unknown; drawGroups?: readonly number[]; sourceStarts?: readonly SourceStart[]; sourceObjects?: readonly SourceObject[] }

export function encodeProject(document: LevelDocument): Uint8Array {
  const files = exportLevel(document);
  return zipSync({
    [MANIFEST]: strToU8(JSON.stringify({ format: 'alien-breed-level-project', version: document.sourceObjects ? 6 : document.sourceStarts ? 5 : 4, name: document.name, playtestSlot: document.playtestSlot, exitZone: document.exitZone })),
    [ROOM_SOURCE]: strToU8(JSON.stringify(roomSourceSnapshot(document.roomSource))),
    ...(document.sourceStarts ? { [PLAYER_STARTS]: strToU8(JSON.stringify(restoreSourceStarts(document.sourceStarts))) } : {}),
    ...(document.sourceObjects ? { [OBJECT_PLACEMENTS]: strToU8(JSON.stringify(validateSourceObjects(document, document.sourceObjects))) } : {}),
    'level/twolev.bin': files.data, 'level/twolev.graph.bin': files.graphics, 'level/twolev.clips': files.clips,
    ...(document.drawGroups ? { 'level/twolev.cfg': encodeLevelSetup(document.exitZone, document.drawGroups) } : {}),
  }, { level: 6 });
}

export function encodeGameLevel(document: LevelDocument): Uint8Array {
  if (!sourceMatchesLevel(document)) throw new Error('This editor version cannot yet compile the changed room outlines or source gameplay edits through Export level. Save project preserves all editable source; game export has been stopped to avoid exporting stale geometry or gameplay.');
  const files = exportLevel(document);
  const folder = `levels/level_${String.fromCharCode(97 + document.playtestSlot)}/`;
  return zipSync({ [`${folder}twolev.bin`]: files.data, [`${folder}twolev.graph.bin`]: files.graphics, [`${folder}twolev.clips`]: files.clips,
    [`${folder}twolev.cfg`]: encodeLevelSetup(document.exitZone, document.drawGroups),
    'INSTALL.txt': strToU8(`Alien Breed level slot ${document.playtestSlot + 1}\n\nExtract this ZIP into the game's data directory. It replaces ${folder}\nKeep a copy of the original level before replacing it.\n${document.exitZone === null ? '' : 'Custom exit setup requires the updated Alien Breed port; the original Amiga engine uses its built-in exit.\n'}`) }, { level: 6 });
}

function candidates(entries: Record<string, Uint8Array>, title: string): ProjectCandidate[] {
  let projectName: string | undefined;
  let projectSlot: number | undefined;
  let projectExit: number | null | undefined;
  let sourceSnapshot: unknown;
  let sourceStarts: readonly SourceStart[] | undefined;
  let sourceObjects: readonly SourceObject[] | undefined;
  const manifest = entries[MANIFEST];
  if (manifest) {
    if (manifest.length > 16384) throw new Error('The project manifest is too large.');
    let metadata: unknown;
    try { metadata = JSON.parse(strFromU8(manifest)); }
    catch { throw new Error('The project manifest is not valid JSON.'); }
    if (!metadata || typeof metadata !== 'object' || !('format' in metadata) || metadata.format !== 'alien-breed-level-project' || !('version' in metadata) || typeof metadata.version !== 'number' || ![1, 2, 3, 4, 5, 6].includes(metadata.version))
      throw new Error('This project format is not supported by this editor version.');
    if (metadata.version >= 2) {
      if (!('playtestSlot' in metadata) || typeof metadata.playtestSlot !== 'number' || !Number.isInteger(metadata.playtestSlot) || metadata.playtestSlot < 0 || metadata.playtestSlot > 15)
        throw new Error('The project has an invalid game level slot. Choose a slot from 1 to 16.');
      projectSlot = metadata.playtestSlot;
    }
    if (metadata.version >= 3) {
      if (!('exitZone' in metadata) || (metadata.exitZone !== null && (typeof metadata.exitZone !== 'number' || !Number.isInteger(metadata.exitZone) || metadata.exitZone < -1 || metadata.exitZone > 255)))
        throw new Error('The project has an invalid exit zone.');
      projectExit = metadata.exitZone as number | null;
    }
    if (metadata.version >= 4) {
      const roomBytes = entries[ROOM_SOURCE];
      if (!roomBytes || roomBytes.length > MAX_FILE) throw new Error('The editable room source is missing or too large.');
      try { sourceSnapshot = JSON.parse(strFromU8(roomBytes)); }
      catch { throw new Error('The editable room source is not valid JSON.'); }
    } else if (entries[ROOM_SOURCE]) throw new Error('Editable room source requires project format version 4.');
    if (metadata.version >= 5 && (metadata.version === 5 || entries[PLAYER_STARTS])) {
      const startBytes = entries[PLAYER_STARTS];
      if (!startBytes || startBytes.length > 4096) throw new Error('The editable player starts are missing or too large.');
      let value: unknown;
      try { value = JSON.parse(strFromU8(startBytes)); }
      catch { throw new Error('The editable player starts are not valid JSON.'); }
      sourceStarts = restoreSourceStarts(value);
    } else if (entries[PLAYER_STARTS]) throw new Error('Editable player starts require project format version 5.');
    if (metadata.version === 6) {
      const objectBytes = entries[OBJECT_PLACEMENTS];
      if (!objectBytes || objectBytes.length > 65536) throw new Error('The editable object placements are missing or too large.');
      let value: unknown;
      try { value = JSON.parse(strFromU8(objectBytes)); }
      catch { throw new Error('The editable object placements are not valid JSON.'); }
      sourceObjects = restoreSourceObjects(value);
    } else if (entries[OBJECT_PLACEMENTS]) throw new Error('Editable object placements require project format version 6.');
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
    const setup = files['twolev.cfg'] ? decodeLevelSetupDetails(files['twolev.cfg']) : undefined, setupExit = setup?.exitZone;
    if (projectExit !== undefined && setupExit !== undefined && projectExit !== setupExit) throw new Error('The project manifest and level setup disagree about the exit zone.');
    result.push({ name: projectName ?? (letter ? `Level ${letter.toLowerCase().charCodeAt(0) - 96}` : title), folder, files: { data, graphics, clips },
      playtestSlot: projectSlot ?? (letter ? letter.toLowerCase().charCodeAt(0) - 97 : 0), exitZone: projectExit !== undefined ? projectExit : setupExit ?? null,
      ...(setup?.drawGroups ? { drawGroups: setup.drawGroups } : {}), ...(sourceSnapshot === undefined ? {} : { roomSourceSnapshot: sourceSnapshot }),
      ...(sourceStarts === undefined ? {} : { sourceStarts }), ...(sourceObjects === undefined ? {} : { sourceObjects }) });
  }
  if (result.length === 0) throw new Error('No game level was found. Open an .ab3d project, a level ZIP, or select all three twolev files together.');
  if (entries[ROOM_SOURCE] && (!manifest || result.length !== 1)) throw new Error('Editable room source must belong to one saved project.');
  if (entries[PLAYER_STARTS] && (!manifest || result.length !== 1)) throw new Error('Editable player starts must belong to one saved project.');
  if (entries[OBJECT_PLACEMENTS] && (!manifest || result.length !== 1)) throw new Error('Editable object placements must belong to one saved project.');
  return result.sort((a, b) => a.folder.localeCompare(b.folder, undefined, { numeric: true }));
}

export function openArchive(bytes: Uint8Array, name: string): ProjectCandidate[] {
  if (bytes.length > MAX_ARCHIVE) throw new Error('The archive exceeds the 64 MiB import limit.');
  let extracted = 0;
  let entries: Record<string, Uint8Array>;
  try {
    entries = unzipSync(bytes, { filter(file) {
      const base = file.name.replaceAll('\\', '/').split('/').at(-1)?.toLowerCase() ?? '';
      if (!LEVEL_NAMES.includes(base) && file.name !== MANIFEST && file.name !== ROOM_SOURCE && file.name !== PLAYER_STARTS && file.name !== OBJECT_PLACEMENTS) return false;
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
  return openProjectCandidate(project);
}

export function openProjectCandidate(project: ProjectCandidate): LevelDocument {
  const document = openLevel(project.name, project.files, project.playtestSlot, project.exitZone, project.drawGroups);
  return { ...document, ...(project.roomSourceSnapshot === undefined ? {} : { roomSource: restoreRoomSource(document, project.roomSourceSnapshot) }),
    ...(project.sourceStarts === undefined ? {} : { sourceStarts: restoreSourceStarts(project.sourceStarts) }),
    ...(project.sourceObjects === undefined ? {} : { sourceObjects: validateSourceObjects(document, project.sourceObjects) }) };
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
