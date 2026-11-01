import { Component, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { flushSync } from 'react-dom';
import { openLevel, objectName, sameBytes, zoneContainsPoint, type LevelDocument, type LevelFiles } from './model/level';
import { execute, redo, undo, moveObject, removeObject, setStart, setSourceStart, setSourceObject, setPlaytestSlot, setExitZone, editRoomSource, type EditCommand, type History } from './model/commands';
import { editableStarts } from './model/sourceStarts';
import { editableObjects, objectIsActive, objectDestinationRoom } from './model/sourceObjects';
import { pointInRoomOutline } from './model/geometry';
import { decodeProject, encodeGameLevel, encodeProject, readFiles, openProjectCandidate, type ProjectCandidate } from './model/project';
import { sourceMatchesLevel, beginRoomDraft, setDraftCorners, commitRoomDraft, removeSourceRooms, moveSourceCorner, type RoomDraft } from './model/roomSource';
import { listWorkspaces, saveWorkspace, type SavedWorkspace } from './persistence';
import { MapCanvas, type Selection } from './components/MapCanvas';
import { Inspector } from './components/Inspector';
import { ObjectGroupInspector } from './components/ObjectGroupInspector';
import { RoomGroupInspector } from './components/RoomGroupInspector';
import { StairBuilder } from './components/StairBuilder';
import { ObjectBuilder } from './components/ObjectBuilder';
import { ObjectPaste } from './components/ObjectPaste';
import { RoomPaste } from './components/RoomPaste';
import { objectTemplates } from './model/objectAuthoring';
import { editObjectGroup, objectSelectionIds } from './model/objectGroups';
import { moveRoomGroup } from './model/roomGroups';
import { duplicateObjects } from './model/duplicateObjects';
import { copyObjects, pasteObjects, type ObjectClipboard } from './model/objectClipboard';
import { copyRoomGroup, type RoomClipboard } from './model/roomClipboard';
import { Icon } from './components/Icon';
import { EnginePreview } from './components/EnginePreview';
import { ControlTooltips } from './components/ControlTooltips';
import { ControlsHelp } from './components/ControlsHelp';
import { preparePlayableLevel } from './model/playableBuild';
import type { InspectionShortcut } from '../runtime/inspection-shortcuts.mjs';

interface EditorKeyInput extends InspectionShortcut {
  target: EventTarget | null; preventDefault: () => void; inspection?: boolean;
}

interface Workspace { id: string; history: History; saved: LevelFiles; savedSlot: number; savedExit: number | null; savedRoomSource: LevelDocument['roomSource']; savedSourceStarts: LevelDocument['sourceStarts']; savedSourceObjects: LevelDocument['sourceObjects']; savedSourceTeleports: LevelDocument['sourceTeleports']; savedAppearances: LevelDocument['roomAppearances']; savedWallOverrides: LevelDocument['wallOverrides'] }
function workspaceDirty(workspace: Workspace): boolean {
  return workspace.history.document.wallOverrides !== workspace.savedWallOverrides || workspace.history.document.roomAppearances !== workspace.savedAppearances || workspace.history.document.sourceTeleports !== workspace.savedSourceTeleports || workspace.history.document.sourceObjects !== workspace.savedSourceObjects || workspace.history.document.sourceStarts !== workspace.savedSourceStarts || workspace.history.document.roomSource !== workspace.savedRoomSource || workspace.history.document.playtestSlot !== workspace.savedSlot || workspace.history.document.exitZone !== workspace.savedExit || (['data', 'graphics', 'clips'] as const).some(file => !sameBytes(workspace.history.document.files[file], workspace.saved[file]));
}
class CanvasBoundary extends Component<{ children: ReactNode; onExport: () => void }, { error: string | null }> {
  state: { error: string | null } = { error: null };
  static getDerivedStateFromError(error: Error) { return { error: error.message }; }
  render() { return this.state.error ? <div className="canvas-error"><Icon name="alert" size={32} /><h2>The map view stopped unexpectedly</h2><p>Your document is still available. Save a copy before reloading.</p><p className="muted">{this.state.error}</p><button className="primary-button" onClick={this.props.onExport}>Save project copy</button></div> : this.props.children; }
}

function download(bytes: Uint8Array, name: string) {
  const blob = new Blob([new Uint8Array(bytes)], { type: 'application/zip' });
  const url = URL.createObjectURL(blob), anchor = document.createElement('a');
  anchor.href = url; anchor.download = name; anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 60000);
}

async function originalLevel(number: number, signal?: AbortSignal): Promise<LevelDocument> {
  const folder = `reference/level_${String.fromCharCode(96 + number)}/`;
  const load = async (name: string) => {
    const response = await fetch(`${import.meta.env.BASE_URL}${folder}${name}`, { signal });
    if (!response.ok) throw new Error(`Could not load ${name} for level ${number} (${response.status}).`);
    return new Uint8Array(await response.arrayBuffer());
  };
  const [data, graphics, clips] = await Promise.all(['twolev.bin', 'twolev.graph.bin', 'twolev.clips'].map(load));
  if (!data || !graphics || !clips) throw new Error('The reference level is incomplete.');
  return openLevel(`Level ${number}`, { data, graphics, clips }, number - 1);
}

function makeWorkspace(document: LevelDocument, id: string = crypto.randomUUID()): Workspace {
  return { id, history: { document, past: [], future: [] }, saved: document.files, savedSlot: document.playtestSlot, savedExit: document.exitZone, savedRoomSource: document.roomSource, savedSourceStarts: document.sourceStarts, savedSourceObjects: document.sourceObjects, savedSourceTeleports: document.sourceTeleports, savedAppearances: document.roomAppearances, savedWallOverrides: document.wallOverrides };
}

export function App() {
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const workspaceRef = useRef(workspace); workspaceRef.current = workspace;
  const [error, setError] = useState(''), [busy, setBusy] = useState('Opening workspace…');
  const [status, setStatus] = useState(''), [autosave, setAutosave] = useState('Local workspace');
  const [selection, setSelection] = useState<Selection>(null), [tab, setTab] = useState<'zones' | 'objects' | 'starts'>('zones');
  const [objectIds, setObjectIds] = useState<number[]>([]);
  const [roomIds, setRoomIds] = useState<string[]>([]);
  const [objectClipboard, setObjectClipboard] = useState<ObjectClipboard | null>(null);
  const [roomClipboard, setRoomClipboard] = useState<RoomClipboard | null>(null);
  const [roomPlacement, setRoomPlacement] = useState<RoomClipboard | null>(null);
  useEffect(() => setObjectIds([]), [workspace?.id]);
  const [search, setSearch] = useState(''), [tool, setTool] = useState<'select' | 'pan' | 'room' | 'box'>('select');
  const contentsList = useRef<HTMLDivElement>(null), revealEngineObject = useRef(false);
  const editorKeyHandler = useRef<((event: EditorKeyInput) => void) | null>(null);
  const [roomDraft, setRoomDraft] = useState<RoomDraft | null>(null), [selectedRoomId, setSelectedRoomId] = useState<string | null>(null);
  const [focusRoom, setFocusRoom] = useState<{ id: string; request: number; frame?: boolean }>();
  const [roomTemplate, setRoomTemplate] = useState(0);
  const [stairsOpen, setStairsOpen] = useState(false);
  const [objectOpen, setObjectOpen] = useState(false);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [objectSeed, setObjectSeed] = useState<number>();
  useEffect(() => { setStairsOpen(false); setObjectOpen(false); setPasteOpen(false); setRoomPlacement(null); setRoomClipboard(null); setObjectClipboard(null); }, [workspace?.id]);
  const [layer, setLayer] = useState<'lower' | 'upper'>('lower'), [grid, setGrid] = useState(64), [showObjects, setShowObjects] = useState(true);
  const [view, setView] = useState<'map' | 'split' | '3d'>('map');
  const [cursor, setCursor] = useState({ x: 0, z: 0 }), [fitKey, setFitKey] = useState(0), [focusZone, setFocusZone] = useState<{ id: number; request: number }>();
  const [frameSelection, setFrameSelection] = useState(0);
  const frameCurrent = useCallback(() => { setView(current => current === '3d' ? 'map' : current); setFrameSelection(value => value + 1); }, []);
  const [navigatorOpen, setNavigatorOpen] = useState(false);
  const [help, setHelp] = useState(false), [choices, setChoices] = useState<ProjectCandidate[]>([]), [recents, setRecents] = useState<SavedWorkspace[]>([]), [showRecents, setShowRecents] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null), searchInput = useRef<HTMLInputElement>(null), generation = useRef(0);
  const document = workspace?.history.document;
  const unbuiltRooms = useMemo(() => document ? !sourceMatchesLevel(document) : false, [document]);
  const selectedRooms = useMemo(() => selectedRoomId ? document?.roomSource.rooms.filter(room => roomIds.includes(room.id)) ?? [] : [], [document?.roomSource, roomIds, selectedRoomId]);
  const selectedRoom = selectedRooms.length > 1 ? undefined : document?.roomSource.rooms.find(room => selectedRoomId ? room.id === selectedRoomId : selection?.kind === 'zone' && room.originZone === selection.id);
  const template = selectedRoom?.recordZone ?? (document?.zones.some(zone => zone.id === roomTemplate) ? roomTemplate : document?.zones[0]?.id);
  useEffect(() => { if (selectedRoom) setRoomTemplate(selectedRoom.recordZone); }, [selectedRoom?.recordZone]);
  const select = useCallback((value: Selection) => { setObjectIds([]); setRoomIds([]); setSelection(value); setSelectedRoomId(null); }, []);
  useEffect(() => {
    if (!revealEngineObject.current) return;
    revealEngineObject.current = false;
    if (selection?.kind !== 'object' || tab !== 'objects') return;
    const list = contentsList.current, row = list?.querySelector<HTMLElement>(`[data-object-id="${selection.id}"]`);
    if (!list || !row) return;
    const bounds = list.getBoundingClientRect(), target = row.getBoundingClientRect();
    // Reveal only inside this list; keep page scroll, map view and keyboard focus.
    if (target.top < bounds.top) list.scrollTop += target.top - bounds.top;
    else if (target.bottom > bounds.bottom) list.scrollTop += target.bottom - bounds.bottom;
  }, [selection, tab, search]);
  const selectRoom = useCallback((id: string, add = false) => {
    const source = workspaceRef.current?.history.document;
    if (!source) return;
    if (add) {
      if (!source.roomSource.rooms.some(room => room.id === id)) return;
      const previous = selectedRooms.length ? selectedRooms.map(room => room.id) : selectedRoom ? [selectedRoom.id] : [];
      const next = previous.includes(id) ? previous.filter(value => value !== id) : [...previous, id];
      setRoomIds(next); setSelectedRoomId(next.at(-1) ?? null); setObjectIds([]); setSelection(null); return;
    }
    setRoomIds([]); setObjectIds([]); setSelection(null); setSelectedRoomId(id);
  }, [selectedRooms, selectedRoom]);
  const visitRoom = useCallback((id: string, add = false) => {
    selectRoom(id, add);
    if (!add) setFocusRoom(current => ({ id, request: (current?.request ?? 0) + 1 }));
  }, [selectRoom]);
  const toggleObject = useCallback((id: number) => {
    const previous = objectIds.length ? objectIds : selection?.kind === 'object' ? [selection.id] : [];
    const next = previous.includes(id) ? previous.filter(value => value !== id) : [...previous, id];
    setObjectIds(next); setSelection(next.length ? { kind: 'object', id: next.at(-1)! } : null); setSelectedRoomId(null);
  }, [objectIds, selection]);
  const selectBoxObjects = useCallback((ids: readonly number[], add: boolean) => {
    const source = workspaceRef.current?.history.document; if (!source) return;
    const eligible = new Set(objectSelectionIds(source));
    const previous = add ? (objectIds.length ? objectIds : selection?.kind === 'object' ? [selection.id] : []).filter(id => eligible.has(id)) : [];
    const next = [...new Set([...previous, ...ids])];
    setObjectIds(next); setSelection(next.length ? { kind: 'object', id: next.at(-1)! } : null); setSelectedRoomId(null); setTool('select');
  }, [objectIds, selection]);
  useEffect(() => { if (tool !== 'room' || view === '3d') setRoomDraft(null); }, [tool, view]);
  useEffect(() => {
    if (roomDraft && roomDraft.base !== document?.roomSource) { setRoomDraft(null); setTool('select'); }
    if (selectedRoomId && !document?.roomSource.rooms.some(room => room.id === selectedRoomId)) setSelectedRoomId(null);
  }, [document?.roomSource, selectedRoomId, roomDraft]);
  useEffect(() => {
    if (!help && !showRecents && choices.length === 0) return;
    const previous = window.document.activeElement;
    const dialog = window.document.querySelector<HTMLElement>('[role="dialog"]');
    dialog?.querySelector<HTMLButtonElement>('button')?.focus();
    const trap = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || !dialog) return;
      const controls = [...dialog.querySelectorAll<HTMLElement>('button:not(:disabled), input, select, summary, a[href], [tabindex="0"]')].filter(node => node.offsetParent !== null);
      const first = controls[0], last = controls.at(-1);
      if (event.shiftKey && window.document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && window.document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    window.document.addEventListener('keydown', trap);
    return () => { window.document.removeEventListener('keydown', trap); if (previous instanceof HTMLElement && previous.isConnected) previous.focus(); };
  }, [help, showRecents, choices.length]);
  const dirty = useMemo(() => !!workspace && workspaceDirty(workspace), [workspace]);

  useEffect(() => {
    const controller = new AbortController();
    const request = ++generation.current;
    let cancelled = false;
    (async () => {
      let saved: SavedWorkspace[] = [];
      try { saved = await listWorkspaces(); if (!cancelled) setRecents(saved); }
      catch (error) { if (!cancelled) setError(`Browser recovery storage is unavailable: ${error instanceof Error ? error.message : error}`); }
      if (cancelled || request !== generation.current) return;
      if (saved[0]) {
        try { const restored = decodeProject(saved[0].bytes); if (!cancelled && request === generation.current) { setWorkspace(makeWorkspace(restored, saved[0].id)); setStatus('Previous workspace restored'); setBusy(''); } return; }
        catch (error) { setError(`Could not restore the last workspace: ${error instanceof Error ? error.message : error}`); }
      }
      try { const opened = await originalLevel(1, controller.signal); if (!cancelled && request === generation.current) { setWorkspace(makeWorkspace(opened)); setBusy(''); } }
      catch (error) { if (!cancelled && request === generation.current) { setError(error instanceof Error ? error.message : String(error)); setBusy(''); } }
    })();
    return () => { cancelled = true; controller.abort(); };
  }, []);

  useEffect(() => {
    if (!workspace) return;
    const current = workspace;
    let cancelled = false;
    setAutosave('Saving workspace…');
    const timer = window.setTimeout(() => {
      saveWorkspace(current.id, current.history.document).then(() => { if (!cancelled) {
        setAutosave('Workspace saved');
        setWorkspace(latest => latest?.id === current.id && latest.history.document === current.history.document
          ? { ...latest, saved: current.history.document.files, savedSlot: current.history.document.playtestSlot, savedExit: current.history.document.exitZone, savedRoomSource: current.history.document.roomSource, savedSourceStarts: current.history.document.sourceStarts, savedSourceObjects: current.history.document.sourceObjects, savedSourceTeleports: current.history.document.sourceTeleports, savedAppearances: current.history.document.roomAppearances, savedWallOverrides: current.history.document.wallOverrides } : latest);
      } })
        .catch(error => { if (!cancelled) { setAutosave('Autosave unavailable'); setError(error instanceof Error ? error.message : String(error)); } });
    }, 600);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [workspace?.id, workspace?.history.document]);

  useEffect(() => {
    const unload = (event: BeforeUnloadEvent) => { if (dirty) event.preventDefault(); };
    window.addEventListener('beforeunload', unload);
    return () => window.removeEventListener('beforeunload', unload);
  }, [dirty]);

  const fail = useCallback((error: unknown) => setError(error instanceof Error ? error.message : String(error)), []);
  const edit = useCallback((request: EditCommand | (() => EditCommand)) => {
    try { const command = typeof request === 'function' ? request() : request; setWorkspace(current => { if (!current) return current; try { return { ...current, history: execute(current.history, command) }; } catch (error) { queueMicrotask(() => fail(error)); return current; } }); setStatus(command.label); }
    catch (error) { fail(error); }
  }, [fail]);

  const finishRoom = useCallback(() => {
    if (!document || !roomDraft) return;
    try {
      const change = commitRoomDraft(document.roomSource, roomDraft);
      edit(editRoomSource(change)); visitRoom(roomDraft.roomId ?? `room-${document.roomSource.nextId}`);
      setRoomDraft(null); setTool('select');
    } catch (error) { fail(error); }
  }, [document, roomDraft, edit, selectRoom, fail]);
  function beginRoom() {
    if (!document || template === undefined) return;
    setRoomDraft(beginRoomDraft(document, document.roomSource, template)); setTool('room'); setView('map');
    setStatus('Click corners in either direction. Enter finishes; Backspace removes the last corner; Escape cancels.');
  }
  function redrawRoom() {
    if (!document || !selectedRoom) return;
    setRoomDraft(setDraftCorners(beginRoomDraft(document, document.roomSource, selectedRoom.recordZone, selectedRoom.id), []));
    setTool('room'); setView('map');
    setStatus('Redraw the outline in either direction. Enter applies it as one edit; Escape or changing tools keeps the original room. New walls need a wall repair appearance.');
  }
  function addRoomPoint(x: number, z: number) {
    if (!roomDraft) return;
    try { setRoomDraft(setDraftCorners(roomDraft, [...roomDraft.vertices, { x, z }])); } catch (error) { fail(error); }
  }
  function editRoomCorner(index: number, axis: 'x' | 'z', value: number) {
    if (!document || !selectedRoom) return;
    edit(() => editRoomSource(moveSourceCorner(document.roomSource, selectedRoom.id, index, { ...selectedRoom.vertices[index]!, [axis]: value })));
  }
  function moveRoomCorner(index: number, x: number, z: number) {
    if (document && selectedRoom) edit(() => editRoomSource(moveSourceCorner(document.roomSource, selectedRoom.id, index, { x, z })));
  }
  function deleteRoom() {
    if (!document || !selectedRoom) return;
    edit(() => editRoomSource(removeSourceRooms(document.roomSource, [selectedRoom.id]))); select(null); setTool('select'); setRoomDraft(null);
  }

  const save = useCallback(() => {
    // Blur commits a pending numeric property before reading the document.
    // This also covers Ctrl+S, where clicking a toolbar button would not blur it.
    flushSync(() => { if (window.document.activeElement instanceof HTMLInputElement) window.document.activeElement.blur(); });
    const invalid = window.document.querySelector<HTMLInputElement>('input:invalid');
    if (invalid) { invalid.focus(); invalid.reportValidity(); return; }
    const current = workspaceRef.current;
    if (!current) return;
    try {
      download(encodeProject(current.history.document), `${current.history.document.name.replace(/[^\w -]/g, '').trim() || 'level'}.ab3d`);
      // Download requests have no completion acknowledgement in standard
      // browser APIs. Only the completed recovery transaction clears dirty.
      setStatus('Project copy download requested');
    } catch (error) { fail(error); }
  }, [fail]);

  const historyAction = useCallback((direction: 'undo' | 'redo') => {
    setRoomDraft(null); setTool('select');
    setWorkspace(current => {
      if (!current) return current;
      try { return { ...current, history: direction === 'undo' ? undo(current.history) : redo(current.history) }; }
      catch (error) { queueMicrotask(() => fail(error)); return current; }
    });
  }, [fail]);

  const duplicateGroup = useCallback((x: number, z: number) => {
    const source = workspaceRef.current?.history.document;
    if (!source || busy) return;
    try {
      const proposal = duplicateObjects(source, objectIds, x, z);
      edit(proposal.command); setObjectIds(proposal.ids); setSelection({ kind: 'object', id: proposal.ids.at(-1)! }); setSelectedRoomId(null); setShowObjects(true);
    } catch (error) { fail(error); }
  }, [objectIds, busy, edit, fail]);

  const duplicateObject = useCallback(() => {
    const source = workspaceRef.current?.history.document;
    if (objectIds.length > 1) { duplicateGroup(0, 0); return; }
    if (!source || busy || selection?.kind !== 'object' || !objectTemplates(source).some(object => object.id === selection.id)
      || !editableObjects(source).some(object => object.id === selection.id && objectIsActive(object))) {
      setStatus('Select an enemy, pickup or scenery object to duplicate. Player and projectile slots cannot be copied.'); return;
    }
    setRoomDraft(null); setTool('select'); setObjectSeed(selection.id); setObjectOpen(true);
  }, [selection, busy, objectIds.length, duplicateGroup]);

  const copySelectedObjects = useCallback(() => {
    const source = workspaceRef.current?.history.document;
    if (!source || busy || roomDraft) return;
    try {
      const copied = copyObjects(source, objectIds.length ? objectIds : selection?.kind === 'object' ? [selection.id] : []);
      setObjectClipboard(copied); setRoomClipboard(null); setStatus(`Copied ${copied.ids.length} objects. Paste keeps their captured settings and positions in this workspace.`);
    } catch (error) { fail(error); }
  }, [objectIds, selection, busy, roomDraft, fail]);

  const copySelectedRooms = useCallback((duplicate = false) => {
    const source = workspaceRef.current?.history.document;
    if (!source || busy || roomDraft) return;
    try {
      const ids = selectedRooms.length ? selectedRooms.map(room => room.id) : selectedRoom ? [selectedRoom.id] : [];
      const copied = copyRoomGroup(source, ids);
      if (duplicate) setRoomPlacement(copied);
      else { setRoomClipboard(copied); setObjectClipboard(null); }
      setStatus(`Captured ${copied.ids.length} rooms including required attachments. Paste rooms opens a placement preview; one Undo removes the whole paste.`);
    } catch (error) { fail(error); }
  }, [selectedRooms, selectedRoom, busy, roomDraft, fail]);

  const pasteCopiedObjects = useCallback(() => {
    const source = workspaceRef.current?.history.document;
    if (!source || !objectClipboard || busy || roomDraft) return;
    try {
      const proposal = pasteObjects(source, objectClipboard);
      edit(proposal.command); setObjectIds(proposal.ids); setSelection({ kind: 'object', id: proposal.ids.at(-1)! }); setSelectedRoomId(null); setShowObjects(true);
      setStatus(`Pasted ${proposal.ids.length} objects in place. Drag the selection or use its inspector to move it; Undo removes the whole paste.`);
    } catch (error) { fail(error); }
  }, [objectClipboard, busy, roomDraft, edit, fail]);

  const deleteSelectedObjects = useCallback(() => {
    const source = workspaceRef.current?.history.document;
    if (!source || busy || selection?.kind !== 'object') return;
    const objects = editableObjects(source);
    const ids = (objectIds.length ? objectIds : [selection.id]).filter(id => objects.some(object => object.id === id && objectIsActive(object)));
    if (ids.length > 1) edit(() => editObjectGroup(source, ids, { kind: 'delete' }));
    else if (ids.length === 1) edit(() => removeObject(source, ids[0]!));
  }, [objectIds, selection, busy, edit]);

  const deleteRoomGroup = useCallback(() => {
    const source = workspaceRef.current?.history.document;
    if (!source || busy || selectedRooms.length < 2) return;
    edit(() => editRoomSource(removeSourceRooms(source.roomSource, selectedRooms.map(room => room.id))));
    select(null); setTool('select'); setRoomDraft(null);
  }, [selectedRooms, busy, edit, select]);

  const selectAllObjects = useCallback((scope: 'list' | 'map' | 'inspection' = 'list') => {
    const source = workspaceRef.current?.history.document;
    if (!source || busy) return;
    const floor = scope !== 'list';
    const ids = scope === 'map' && !showObjects ? [] : objectSelectionIds(source, floor ? '' : search, floor ? layer === 'upper' : undefined);
    setObjectIds(ids); setSelection(ids.length ? { kind: 'object', id: ids.at(-1)! } : null); setSelectedRoomId(null);
    setStatus(ids.length ? `Selected ${ids.length} objects${floor ? ` on the ${layer} floor` : ' matching the current search'}` : 'No editable objects match this selection.');
  }, [busy, showObjects, search, layer]);
  const beginBoxSelection = useCallback(() => {
    if (!workspaceRef.current || busy) return;
    setRoomDraft(null); setTool('box'); setView(current => current === '3d' ? 'map' : current);
  }, [busy]);

  useEffect(() => {
    const key = (event: EditorKeyInput) => {
      const editing = event.target instanceof HTMLElement && (event.target.matches('input, textarea, select') || event.target.isContentEditable);
      const viewport = event.inspection || event.target instanceof HTMLCanvasElement;
      // Modal panels own keyboard input. In particular, Undo must not change
      // the document behind help, and closing help must retain map selection.
      if (help || showRecents || choices.length) {
        if (event.key === 'Escape') {
          event.preventDefault(); setHelp(false); setShowRecents(false); setChoices([]);
        } else if ((event.ctrlKey || event.metaKey) && ['s', 'o', 'z', 'y', 'd'].includes(event.key.toLowerCase())) {
          event.preventDefault();
        }
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save(); return; }
      if (stairsOpen || objectOpen || pasteOpen || roomPlacement || window.document.querySelector('dialog[open]')) { if ((event.ctrlKey || event.metaKey) && ['o', 'd'].includes(event.key.toLowerCase())) event.preventDefault(); return; }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'o') { event.preventDefault(); fileInput.current?.click(); return; }
      if (editing) return;
      if ((event.ctrlKey || event.metaKey) && ['c', 'v'].includes(event.key.toLowerCase())) {
        if (event.key.toLowerCase() === 'c' && selection?.kind !== 'object' && !selectedRoom && !selectedRooms.length) return;
        event.preventDefault();
        if (!event.repeat) {
          if (event.key.toLowerCase() === 'c') { if (selection?.kind === 'object') copySelectedObjects(); else copySelectedRooms(); }
          else if (roomClipboard) { if (!busy && !roomDraft) setRoomPlacement(roomClipboard); }
          else if (event.shiftKey) { if (objectClipboard && !busy && !roomDraft) setPasteOpen(true); }
          else pasteCopiedObjects();
        }
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a' && (tab === 'objects' || viewport)) {
        event.preventDefault(); if (!roomDraft && !help && !showRecents && !choices.length) selectAllObjects(event.inspection ? 'inspection' : viewport ? 'map' : 'list'); return;
      }
      if (event.key === 'Delete' && !event.ctrlKey && !event.metaKey && !event.altKey) {
        if (!event.repeat && !roomDraft && !help && !showRecents && !choices.length) { event.preventDefault(); if (selectedRooms.length > 1) deleteRoomGroup(); else deleteSelectedObjects(); }
        return;
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'd') { event.preventDefault(); if (!event.repeat && !help && !showRecents && !choices.length) { if (selection?.kind !== 'object' && (selectedRoom || selectedRooms.length)) copySelectedRooms(true); else duplicateObject(); } return; }
      if (roomDraft && viewport) {
        if (event.key === 'Enter') { event.preventDefault(); finishRoom(); return; }
        if (event.key === 'Backspace' || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z' && !event.shiftKey)) {
          event.preventDefault(); setRoomDraft(setDraftCorners(roomDraft, roomDraft.vertices.slice(0, -1))); return;
        }
      }
      if ((event.ctrlKey || event.metaKey) && ['z', 'y'].includes(event.key.toLowerCase())) { event.preventDefault(); historyAction(event.key.toLowerCase() === 'y' || event.shiftKey ? 'redo' : 'undo'); }
      if (event.key.toLowerCase() === 'f' && !event.ctrlKey && !event.metaKey) { if (event.shiftKey) frameCurrent(); else setFitKey(key => key + 1); }
      if (event.key.toLowerCase() === 'v') setTool('select');
      if (event.key.toLowerCase() === 'h') setTool('pan');
      if (event.key.toLowerCase() === 'b' && !event.ctrlKey && !event.metaKey && !help && !showRecents && !choices.length) beginBoxSelection();
      if (event.key === '/') { event.preventDefault(); flushSync(() => setNavigatorOpen(true)); searchInput.current?.focus(); }
      if (event.key === 'Escape') { select(null); setRoomDraft(null); setTool('select'); setHelp(false); setShowRecents(false); setChoices([]); setNavigatorOpen(false); }
      if (event.key === '?') setHelp(true);
    };
    editorKeyHandler.current = key;
    window.addEventListener('keydown', key);
    return () => { window.removeEventListener('keydown', key); if (editorKeyHandler.current === key) editorKeyHandler.current = null; };
  }, [save, historyAction, roomDraft, finishRoom, select, stairsOpen, objectOpen, pasteOpen, roomPlacement, duplicateObject, help, showRecents, choices.length, frameCurrent, deleteSelectedObjects, selectAllObjects, tab, beginBoxSelection, selection, copySelectedObjects, copySelectedRooms, pasteCopiedObjects, objectClipboard, roomClipboard, selectedRoom, busy, selectedRooms, deleteRoomGroup]);

  async function openDocument(next: LevelDocument, id?: string, request = ++generation.current) {
    const current = workspaceRef.current;
    if (current && workspaceDirty(current))
      await saveWorkspace(current.id, current.history.document);
    if (request !== generation.current) return;
    setWorkspace(makeWorkspace(next, id)); select(null); setRoomDraft(null); setRoomTemplate(next.zones[0]?.id ?? 0); setTool('select'); setFocusZone(undefined); setFocusRoom(undefined); setSearch(''); setChoices([]); setShowRecents(false); setStatus(`Opened ${next.name}`);
  }

  async function loadOriginal(level: number) {
    const request = ++generation.current;
    setBusy(`Opening level ${level}…`); setError('');
    try { const next = await originalLevel(level); if (request === generation.current) await openDocument(next, undefined, request); }
    catch (error) { if (request === generation.current) fail(error); }
    finally { if (request === generation.current) setBusy(''); }
  }

  async function importFiles(files: File[]) {
    const request = ++generation.current;
    setBusy('Opening files…'); setError('');
    try {
      const candidates = await readFiles(files);
      if (request !== generation.current) return;
      if (candidates.length === 1 && candidates[0]) await openDocument(openProjectCandidate(candidates[0]), undefined, request);
      else setChoices(candidates);
    } catch (error) { if (request === generation.current) fail(error); }
    finally { if (request === generation.current) setBusy(''); }
  }

  async function exportGame() {
    const source = workspaceRef.current?.history.document, request = generation.current;
    if (!source) return;
    setBusy('Building game level…');
    try {
      const artifact = await preparePlayableLevel(source);
      if (workspaceRef.current?.history.document !== source) { setStatus('The level changed during the build. Export again to use the latest edits.'); return; }
      download(encodeGameLevel(artifact.document), `${source.name.replace(/[^\w -]/g, '').trim() || 'level'}-game.zip`);
      setStatus('Game level ZIP downloaded');
    } catch (error) { if (workspaceRef.current?.history.document === source) fail(error); }
    finally { if (generation.current === request) setBusy(''); }
  }

  const move = (selected: Exclude<Selection, null>, x: number, z: number) => {
    if (!document) return;
    try {
      if ((selected.kind === 'start' || selected.kind === 'object') && !sourceMatchesLevel(document)) {
        const current = selected.kind === 'start' ? document.sourceStarts?.find(start => start.player === selected.id)?.room
          : document.sourceObjects?.find(object => object.id === selected.id)?.room;
        const rooms = document.roomSource.rooms;
        const object = selected.kind === 'object' ? editableObjects(document).find(object => object.id === selected.id) : undefined;
        const target = object ? objectDestinationRoom(document, object, { x, z })
          : rooms.find(room => room.id === current && pointInRoomOutline(room.vertices, { x, z })) ?? rooms.find(room => pointInRoomOutline(room.vertices, { x, z }));
        if (!target) throw new Error('Place the object or player start inside an editable room.');
        edit(selected.kind === 'start' ? setSourceStart(document, selected.id as 1 | 2, target.id, x, z)
          : setSourceObject(document, selected.id, target.id, x, z));
        return;
      }
      const oldZone = selected.kind === 'object' ? document.objects.find(object => object.id === selected.id)?.zone : document.starts.find(start => start.player === selected.id)?.zone;
      const upper = selected.kind === 'object' && !!document.objects.find(object => object.id === selected.id)?.upper;
      const compatible = (zone: LevelDocument['zones'][number]) => (!upper || zone.upper.length > 0) && zoneContainsPoint(document, zone, x, z);
      const zone = document.zones.find(zone => zone.id === oldZone && compatible(zone)) ?? document.zones.find(compatible);
      if (!zone) throw new Error('That point is outside the level. Place the object inside a zone.');
      if (selected.kind === 'object') edit(moveObject(document, selected.id, x, z, zone.id));
      if (selected.kind === 'start') edit(setStart(document, selected.id as 1 | 2, x, z, zone.id));
    } catch (error) { fail(error); }
  };

  const filteredZones = document?.zones.filter(zone => document.roomSource.rooms.some(room => room.originZone === zone.id) && String(zone.id).includes(search)).map(zone => {
    const room = document.roomSource.rooms.find(room => room.originZone === zone.id);
    return room?.heights ? { ...zone, ...room.heights } : zone;
  }) ?? [];
  const newRooms = document?.roomSource.rooms.filter(room => room.originZone === null && room.id.toLowerCase().includes(search.toLowerCase())) ?? [];
  const objects = useMemo(() => document ? editableObjects(document) : [], [document]);
  const groupIds = useMemo(() => objectIds.filter(id => objects.some(object => object.id === id && objectIsActive(object))), [objectIds, objects]);
  const filteredObjects = objects.filter(object => objectIsActive(object) && `${object.id} ${objectName(object.type)}`.toLowerCase().includes(search.toLowerCase()));
  const selectableIds = useMemo(() => document ? objectSelectionIds(document, search) : [], [document, search]);
  const activeObjects = objects.filter(objectIsActive).length;
  return <div className={`app ${navigatorOpen ? 'navigator-open' : ''}`} onDragOver={event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); }} onDrop={event => { event.preventDefault(); if (event.dataTransfer.files.length) void importFiles([...event.dataTransfer.files]); }}>
    <ControlTooltips />
    <header className="app-header"><a className="brand" href="." aria-label="Alien Breed editor home"><span className="brand-mark">AB</span><span>ALIEN BREED<span className="brand-sub">LEVEL EDITOR</span></span></a><span className="header-divider" />
      <div className="document-title">{dirty && <span className="dirty-dot" title="Changes waiting to be saved in this browser" />}{document?.name ?? 'Workspace'}<span className="document-type">PROJECT</span></div>
      <div className="header-actions"><button onClick={() => fileInput.current?.click()} title="Open an .ab3d project, a level ZIP or the three twolev files together. Ctrl / Command + O."><Icon name="open" />Open</button><button onClick={save} disabled={!document} title="Download a portable project backup. Recovery is saved separately in this browser. Ctrl / Command + S."><Icon name="save" />Save project</button><button className="export-button" title="Build the latest source and download game-level files for the selected slot. Your editable project is kept." disabled={!document || !!busy} onClick={() => void exportGame()}><Icon name="export" />Export level</button><button className="icon-button" aria-label="Help and shortcuts" title="Show map, 3D inspection and playtest controls. Shortcut: ?" onClick={() => setHelp(true)}><Icon name="help" /></button></div>
      <input ref={fileInput} type="file" hidden multiple aria-label="Open level files" accept=".ab3d,.zip,.bin,.clips,.cfg" onChange={event => { const files = [...(event.target.files ?? [])]; if (files.length) void importFiles(files); event.target.value = ''; }} />
    </header>
    {error && <div className="error-banner" role="alert"><Icon name="alert" /><span>{error}</span><button className="icon-button" aria-label="Dismiss error" title="Close this message. Your current document stays open." onClick={() => setError('')}><Icon name="close" size={16} /></button></div>}
    {unbuiltRooms && <div className="error-banner" role="status"><Icon name="layers" /><span>This project has editable source changes. Preview and Export level rebuild game files; Save project preserves the editable rooms and links.</span></div>}
    <div className="workspace">
      <aside className="navigator"><div className="panel-title">PROJECT <span className="eyebrow">LOCAL</span></div><div className="project-picker"><Icon name="layers" /><div><strong>{document?.name ?? 'Opening…'}</strong><span>{document?.roomSource.rooms.length ?? 0} rooms · {activeObjects} objects</span></div></div>
        <label className="reference-picker"><span>Target game slot</span><select aria-label="Target game slot" title="Saved with this project. Build & play uses this game slot; Export level writes its matching level directory. Undo restores the previous slot." disabled={!document} value={document?.playtestSlot ?? 0} onChange={event => { if (document) edit(() => setPlaytestSlot(document, Number(event.target.value))); }}>{Array.from({ length: 16 }, (_, i) => <option key={i} value={i}>Level slot {i + 1}</option>)}</select></label>
        <label className="reference-picker"><span>Level exit</span><select aria-label="Level exit" title="Finish the level when player one enters this zone. Original uses the chosen game slot's built-in exit. No exit is useful for testing. Custom exits are supported by the updated game port." disabled={!document} value={document?.exitZone ?? 'original'} onChange={event => { if (document) edit(() => setExitZone(document, event.target.value === 'original' ? null : Number(event.target.value))); }}><option value="original">Original slot exit</option><option value="-1">No exit (playtest freely)</option>{document?.zones.map(zone => <option key={zone.id} value={zone.id}>Zone {zone.id}</option>)}</select></label>
        <label className="reference-picker"><span>Original levels</span><select aria-label="Original level" title="Open one of the sixteen shipped maps. Your current edits are saved in this browser first." value={/^Level (\d+)$/.exec(document?.name ?? '')?.[1] ?? ''} disabled={!!busy} onChange={event => { if (event.target.value) void loadOriginal(Number(event.target.value)); }}><option value="">Choose a reference level</option>{Array.from({ length: 16 }, (_, i) => <option key={i} value={i + 1}>Level {String(i + 1).padStart(2, '0')}</option>)}</select></label>
        <div className="navigator-tabs" role="tablist" aria-label="Map contents">{(['zones', 'objects', 'starts'] as const).map(value => <button key={value} role="tab" title={value === 'starts' ? 'Find and select player starting positions.' : value === 'zones' ? 'Find and select rooms by zone number.' : 'Find enemies, pickups and scenery by name.'} aria-selected={tab === value} className={tab === value ? 'active' : ''} onClick={() => { setTab(value); setSearch(''); }}>{value === 'starts' ? 'Starts' : value[0]?.toUpperCase() + value.slice(1)}</button>)}</div>
        <label className="search-box"><Icon name="search" size={15} /><input ref={searchInput} aria-label="Search map contents" title="Find a zone number or object name. Press / from the map to focus search." value={search} onChange={event => setSearch(event.target.value)} placeholder={tab === 'zones' ? 'Find a zone…' : 'Find an object…'} /><kbd>/</kbd></label>
        <div className="contents-list" ref={contentsList}>
          {tab === 'objects' && <button className="small-button" disabled={!selectableIds.length || !!busy} title="Select all active objects matching this search, across both floors. Player avatars are excluded. Shortcut: Ctrl/Command+A outside text fields. On the map the shortcut selects only the displayed floor."
            onClick={() => selectAllObjects()}>Select all matching objects</button>}
          {tab === 'zones' && newRooms.map(room => <button key={room.id} className={`content-row ${selectedRoomId === room.id || selectedRooms.some(item => item.id === room.id) ? 'selected' : ''}`} aria-pressed={selectedRoomId === room.id || selectedRooms.some(item => item.id === room.id)} title="Select this room. Shift, Ctrl or Command click to add or remove it from a room group." onClick={event => { visitRoom(room.id, event.shiftKey || event.ctrlKey || event.metaKey); setNavigatorOpen(false); }}><Icon name="layers" size={14} /><span>Room {room.id.replace(/^room-/, '')}</span><span className="row-badge">OUTLINE</span></button>)}
          {tab === 'zones' && filteredZones.map(zone => <button key={zone.id} className={`content-row ${selection?.kind === 'zone' && selection.id === zone.id || selectedRooms.some(room => room.originZone === zone.id) ? 'selected' : ''}`} aria-pressed={selection?.kind === 'zone' && selection.id === zone.id || selectedRooms.some(room => room.originZone === zone.id)} title={`Select zone ${zone.id} and centre the map. Shift, Ctrl or Command click adds/removes its room from a group. In 3D inspection, visit this zone.`} onClick={event => { if (event.shiftKey || event.ctrlKey || event.metaKey) { const room = document?.roomSource.rooms.find(room => room.originZone === zone.id); if (room) selectRoom(room.id, true); return; } select({ kind: 'zone', id: zone.id }); setFocusZone(current => ({ id: zone.id, request: (current?.request ?? 0) + 1 })); setNavigatorOpen(false); }}><span className={`zone-dot ${document?.mechanisms.some(mechanism => mechanism.zone === zone.id && mechanism.kind === 'door') ? 'door-dot' : zone.water > zone.ceiling && zone.water < zone.floor ? 'water-dot' : ''}`} /><span>Zone {String(zone.id).padStart(3, '0')}</span>{zone.upper.length > 0 && <span className="row-badge">2F</span>}<span className="row-detail">{zone.floor}</span></button>)}
          {tab === 'objects' && filteredObjects.map(object => <button key={object.id} data-object-id={object.id} className={`content-row ${groupIds.includes(object.id) || selection?.kind === 'object' && selection.id === object.id ? 'selected' : ''}`} aria-pressed={groupIds.includes(object.id) || selection?.kind === 'object' && selection.id === object.id} title={`Select ${objectName(object.type)} ${object.id}. Shift, Ctrl or Command click to add or remove it from a group. Drag a single object on the map to move it.`} onClick={event => {
            if (event.shiftKey || event.ctrlKey || event.metaKey) { toggleObject(object.id); return; }
            select({ kind: 'object', id: object.id });
            if (object.room) setFocusRoom(current => ({ id: object.room!, request: (current?.request ?? 0) + 1 }));
            else setFocusZone(current => ({ id: object.zone, request: (current?.request ?? 0) + 1 }));
            setNavigatorOpen(false);
          }}><Icon name="object" size={14} /><span>{objectName(object.type)}</span><span className="row-detail">{object.id}</span></button>)}
          {tab === 'starts' && document && editableStarts(document).map(start => <button key={start.player} className={`content-row ${selection?.kind === 'start' && selection.id === start.player ? 'selected' : ''}`} title={`Select player ${start.player} start. Drag the triangle on the map to reposition it.`} onClick={() => {
            select({ kind: 'start', id: start.player });
            if (start.room) setFocusRoom(current => ({ id: start.room!, request: (current?.request ?? 0) + 1 }));
            else if (start.zone >= 0) setFocusZone(current => ({ id: start.zone, request: (current?.request ?? 0) + 1 }));
            setNavigatorOpen(false);
          }}><Icon name="pin" size={14} /><span>Player {start.player}</span><span className="row-detail">{start.zone >= 0 ? `Z${start.zone}` : start.room ?? 'Missing room'}</span></button>)}
          {tab !== 'starts' && (tab === 'zones' ? filteredZones.length + newRooms.length : filteredObjects.length) === 0 && <div className="list-empty">No matches</div>}
        </div>
        <button className="recent-button" title="Open a workspace previously saved in this browser." onClick={() => { listWorkspaces().then(setRecents).catch(fail); setShowRecents(true); }}><Icon name="open" size={15} />Recent workspaces<span>↗</span></button>
      </aside>
      <main className="map-main"><div className="map-toolbar"><div className="tool-group"><button className={`icon-button ${tool === 'select' ? 'active' : ''}`} aria-label="Select tool" aria-pressed={tool === 'select'} title="Click to select a zone, object or start. Drag objects and starts to move them. Shortcut: V." onClick={() => setTool('select')}><Icon name="select" /></button><button className={`icon-button ${tool === 'pan' ? 'active' : ''}`} aria-label="Pan tool" aria-pressed={tool === 'pan'} title="Drag to move around the map. Shortcut: H, or hold Space while dragging. Middle/right drag also pans." onClick={() => setTool('pan')}><Icon name="hand" /></button></div><span className="toolbar-divider" />
        <label className="grid-select"><select aria-label="Room settings template" title="Copy room settings from this original zone when drawing a new outline. Available even after deleting all outlines. Selecting a room also picks its settings." disabled={!document || !!roomDraft} value={template ?? 0} onChange={event => { setRoomTemplate(Number(event.target.value)); select(null); }}>{document?.zones.map(zone => <option key={zone.id} value={zone.id}>Template {zone.id}</option>)}</select></label>
        <button className={`small-button ${tool === 'room' ? 'active' : ''}`} aria-label="Draw room" aria-pressed={tool === 'room'} disabled={template === undefined} title="Draw an editable room outline using the chosen settings template, in either direction. Enter finishes; Escape cancels. Game geometry compilation is still in progress." onClick={beginRoom}>Draw room</button>
        <button className="small-button" disabled={!document || template === undefined || !!busy} title="Create adjoining steps from endpoints, width and floor heights. Preview the whole flight and apply it as one undoable edit." onClick={() => { setRoomDraft(null); setTool('select'); setStairsOpen(true); }}>Build stairs</button>
        <button className="small-button" disabled={!document || !!busy} aria-pressed={tool === 'box'} title="Drag a rectangle in either direction to select objects on the displayed floor. Hold Shift to add to the selection. Escape cancels the rectangle. Shortcut: B." onClick={beginBoxSelection}>Box select objects</button>
        <button className="small-button" disabled={!document || !!busy} title="Add an enemy, pickup or scenery object using a source template. Choose its room and position; Escape cancels the draft." onClick={() => { setRoomDraft(null); setTool('select'); setObjectSeed(undefined); setObjectOpen(true); }}>Add object</button>
        <button className="small-button" disabled={selection?.kind !== 'object' || !!busy || !!roomDraft} title="Capture selected objects and their current settings in the editor clipboard. Shortcut: Ctrl/Command+C. Text fields keep normal copy controls." onClick={copySelectedObjects}>Copy objects</button>
        <button className="small-button" disabled={(!selectedRoom && !selectedRooms.length) || !!busy || !!roomDraft} title="Capture selected rooms, objects, links and mechanism settings. Required attached rooms are included and listed before placement. Ctrl/Command+C copies the current room selection."
          onClick={() => copySelectedRooms()}>Copy rooms</button>
        <button className="small-button" disabled={!roomClipboard || !document || roomClipboard.source.original !== document.original || !!busy || !!roomDraft} title="Open a placement preview for captured rooms. Choose a destination, review included rooms and validation, then commit one undoable paste. Ctrl/Command+V. Escape cancels."
          onClick={() => setRoomPlacement(roomClipboard)}>Paste rooms</button>
        <button className="small-button" disabled={(!selectedRoom && !selectedRooms.length) || !!busy || !!roomDraft} title="Duplicate the selected room group in a placement preview without replacing the clipboard. Required attachments are included. Ctrl/Command+D."
          onClick={() => copySelectedRooms(true)}>Duplicate rooms</button>
        <button className="small-button" disabled={!objectClipboard || !document || objectClipboard.source.original !== document.original || !!busy || !!roomDraft} title="Paste captured objects in place as one undoable edit, then drag or use the inspector to move them. Same workspace only; clipboard lasts until reload. Shortcut: Ctrl/Command+V." onClick={pasteCopiedObjects}>Paste objects</button>
        <button className="small-button" disabled={!objectClipboard || !document || objectClipboard.source.original !== document.original || !!busy || !!roomDraft} title="Choose the first copied object's destination before pasting the whole selection. Spacing and heights are retained. Escape cancels. Shortcut: Ctrl/Command+Shift+V." onClick={() => setPasteOpen(true)}>Place pasted objects</button>
        {roomDraft && <><button className="small-button" disabled={roomDraft.vertices.length < 3} title="Validate and commit this whole outline as one undoable edit. Shortcut: Enter on the map." onClick={finishRoom}>Finish room</button><button className="small-button" title="Discard this unfinished outline. No project geometry changes. Shortcut: Escape." onClick={() => { setRoomDraft(null); setTool('select'); }}>Cancel room</button></>}
        <button className="icon-button" aria-label="Undo" title={`Undo ${workspace?.history.past.at(-1)?.label ?? ''} (Ctrl+Z)`} disabled={!workspace?.history.past.length} onClick={() => historyAction('undo')}><Icon name="undo" /></button><button className="icon-button" aria-label="Redo" title="Redo (Ctrl+Shift+Z)" disabled={!workspace?.history.future.length} onClick={() => historyAction('redo')}><Icon name="redo" /></button><span className="toolbar-divider" />
        <div className="layer-toggle"><button className={layer === 'lower' ? 'active' : ''} title="Inspect the lower floor of each zone." onClick={() => setLayer('lower')}>Lower</button><button className={layer === 'upper' ? 'active' : ''} title="Inspect upper-floor geometry where a zone has two floors." onClick={() => setLayer('upper')}>Upper</button></div><span className="toolbar-spacer" />
        <label className="grid-select"><Icon name="grid" size={15} /><select aria-label="Grid spacing" title="Set snapping for dragged objects and starts. Hold Alt to bypass snapping; Grid off uses whole coordinates." value={grid} onChange={event => setGrid(Number(event.target.value))}>{[0, 8, 16, 32, 64, 128].map(value => <option key={value} value={value}>{value === 0 ? 'Grid off' : `Grid ${value}`}</option>)}</select></label><button className={`icon-button ${showObjects ? 'active' : ''}`} aria-label="Show objects" title="Show or hide enemies, pickups and scenery on the map." aria-pressed={showObjects} onClick={() => setShowObjects(show => !show)}><Icon name="eye" /></button><button className="icon-button" title="Fit the whole level into the map view. Shortcut: F." aria-label="Frame map" onClick={() => setFitKey(key => key + 1)}><Icon name="fit" /></button>
      </div>
        <div className="view-tabs" role="group" aria-label="Viewport layout"><button className="project-toggle" aria-label="Toggle project navigator" title="Show or hide the level picker and map contents. Shortcut / opens search." aria-expanded={navigatorOpen} onClick={() => setNavigatorOpen(open => !open)}><Icon name="layers" size={14} />Project</button>{(['map', 'split', '3d'] as const).map(mode => <button key={mode} aria-pressed={view === mode} className={view === mode ? 'active' : ''} title={mode === 'map' ? 'Edit and inspect the overhead map.' : mode === 'split' ? 'Show the map and actual game renderer together.' : 'Inspect the level in the actual game renderer.'} onClick={() => setView(mode)}>{mode === 'map' ? '2D map' : mode === 'split' ? 'Split view' : '3D view'}</button>)}</div>
        {document ? <CanvasBoundary key={workspace?.id} onExport={save}><div className={`viewports viewports-${view}`}>
          {view !== '3d' && <MapCanvas document={document} selection={selection} objectIds={groupIds} frameSelection={frameSelection} onObjectToggle={toggleObject} onObjectBoxSelect={selectBoxObjects} onGroupMove={(ids, x, z) => edit(() => editObjectGroup(document, ids, { kind: 'move', x, z }))} onSelect={select} onMove={move} tool={tool} layer={layer} grid={grid} showObjects={showObjects} onCursor={(x, z) => setCursor({ x, z })} fitKey={fitKey} focusZone={focusZone} roomDraft={roomDraft} selectedRoom={selectedRoom} roomGroup={selectedRooms} onRoomSelect={selectRoom} onRoomPoint={addRoomPoint} onRoomFinish={finishRoom} onRoomCornerMove={moveRoomCorner} focusRoom={focusRoom} />}
          {view !== 'map' && <EnginePreview key={workspace?.id} document={document} selection={selection} sourceRoomId={selectedRoom?.id} layer={layer} onSelectRoom={selectRoom} onSelectZone={id => select({ kind: 'zone', id })} onSelectObject={id => { revealEngineObject.current = true; setTab('objects'); setSearch(''); select({ kind: 'object', id }); }} onShortcut={shortcut => editorKeyHandler.current?.({ ...shortcut, target: null, inspection: true, preventDefault: () => {} })} onEdit={edit} onSlotChange={slot => edit(() => setPlaytestSlot(document, slot))} />}
        </div></CanvasBoundary> : <div className="canvas-error"><Icon name="layers" size={36} /><h2>Open a level to begin</h2><p>Drop a project, level ZIP or the three twolev files here.</p><button className="primary-button" onClick={() => fileInput.current?.click()}>Open files</button></div>}
        {busy && <div className="busy-overlay" role="status"><span className="spinner" />{busy}</div>}
        {stairsOpen && document && template !== undefined && <StairBuilder document={document} template={template} selected={selectedRoom} cursor={cursor}
          onCancel={() => setStairsOpen(false)} onCreate={proposal => { edit(proposal.command); setStairsOpen(false); setTool('select'); setView('map'); visitRoom(proposal.rooms[0]!.id); }} />}
        {pasteOpen && document && objectClipboard && <ObjectPaste document={document} clipboard={objectClipboard} cursor={cursor} onCancel={() => setPasteOpen(false)}
          onPaste={proposal => { edit(proposal.command); setPasteOpen(false); setTool('select'); setTab('objects'); setShowObjects(true); setObjectIds(proposal.ids); setSelection({ kind: 'object', id: proposal.ids.at(-1)! }); setSelectedRoomId(null); }} />}
        {roomPlacement && document && <RoomPaste document={document} clipboard={roomPlacement} cursor={cursor} grid={grid} onCancel={() => setRoomPlacement(null)}
          onPaste={proposal => { edit(proposal.command); setRoomPlacement(null); setTool('select'); setTab('zones'); setView('map'); setObjectIds([]); setSelection(null);
            setRoomIds(proposal.ids); setSelectedRoomId(proposal.ids.at(-1)!); setFrameSelection(value => value + 1); }} />}
        {objectOpen && document && <ObjectBuilder document={document} initialObject={objectSeed} room={selectedRoom?.id} cursor={cursor} layer={layer} onCancel={() => setObjectOpen(false)}
          onCreate={proposal => { edit(proposal.command); setObjectOpen(false); setTab('objects'); setShowObjects(true); select({ kind: 'object', id: proposal.id }); }} />}
      </main>
      {document && (selectedRooms.length > 1 ? <RoomGroupInspector rooms={selectedRooms} frame={frameCurrent} clear={() => select(null)} remove={deleteRoomGroup} move={(x, z) => { if (x !== 0 || z !== 0) edit(() => moveRoomGroup(document, selectedRooms.map(room => room.id), x, z)); }} /> : groupIds.length > 1 ? <ObjectGroupInspector document={document} ids={groupIds} edit={edit} frame={frameCurrent} duplicate={duplicateGroup} clear={() => select(null)} /> : <Inspector document={document} selection={selection} edit={edit} sourceRoom={selectedRoom} onObjectDuplicate={duplicateObject} onRoomCorner={editRoomCorner} onRoomRedraw={redrawRoom} onRoomDelete={deleteRoom} onInspectRoom={id => { const target = document.roomSource.rooms.find(room => room.id === id); if (!target) return; if (target.originZone !== null) select({ kind: 'zone', id: target.originZone }); else selectRoom(id); setFocusRoom(current => ({ id, request: (current?.request ?? 0) + 1, frame: true })); setView('map'); }} onShowDestination={id => { setFocusRoom(current => ({ id, request: (current?.request ?? 0) + 1, frame: true })); setView('map'); }} />)}
    </div>
    <footer className="statusbar"><span className="status-dot" /><span>{autosave}</span><span className="status-message" aria-live="polite">{status || 'Choose a zone to start editing'}</span><span>X <strong>{Math.round(cursor.x)}</strong></span><span>Z <strong>{Math.round(cursor.z)}</strong></span><span className="status-size">{document?.points.length ?? 0} points · {document?.zones.length ?? 0} zones</span></footer>
    {(help || showRecents || choices.length > 0) && <div className="modal-backdrop" onPointerDown={event => { if (event.target === event.currentTarget) { setHelp(false); setShowRecents(false); setChoices([]); } }}><section className="modal" role="dialog" aria-modal="true" aria-label={help ? 'Editor help' : choices.length ? 'Choose a level' : 'Recent workspaces'}><button className="modal-close icon-button" aria-label="Close dialog" title="Close this panel. Shortcut: Esc." onClick={() => { setHelp(false); setShowRecents(false); setChoices([]); }}><Icon name="close" /></button>
      {help && <ControlsHelp />}
      {showRecents && <><span className="eyebrow">SAVED IN THIS BROWSER</span><h2>Recent workspaces</h2>{recents.length === 0 && <p>No workspaces have been saved yet.</p>}{recents.map(recent => <button className="recent-entry" key={recent.id} onClick={() => { try { void openDocument(decodeProject(recent.bytes), recent.id).catch(fail); } catch (error) { fail(error); } }}><Icon name="layers" /><span><strong>{recent.name}</strong><small>{new Date(recent.updatedAt).toLocaleString()}</small></span><Icon name="chevron" size={15} /></button>)}</>}
      {choices.length > 0 && <><span className="eyebrow">{choices.length} LEVELS FOUND</span><h2>Choose a level to open</h2>{choices.map(choice => <button className="recent-entry" key={choice.folder} onClick={() => { try { void openDocument(openProjectCandidate(choice)).catch(fail); } catch (error) { fail(error); } }}><Icon name="layers" /><span><strong>{choice.name}</strong><small>{choice.folder}</small></span><Icon name="chevron" size={15} /></button>)}</>}
    </section></div>}
  </div>;
}
