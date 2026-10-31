import { Component, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { flushSync } from 'react-dom';
import { openLevel, objectName, sameBytes, zoneContainsPoint, type LevelDocument, type LevelFiles } from './model/level';
import { execute, redo, undo, moveObject, setStart, type EditCommand, type History } from './model/commands';
import { decodeProject, encodeGameLevel, encodeProject, readFiles, type ProjectCandidate } from './model/project';
import { listWorkspaces, saveWorkspace, type SavedWorkspace } from './persistence';
import { MapCanvas, type Selection } from './components/MapCanvas';
import { Inspector } from './components/Inspector';
import { Icon } from './components/Icon';
import { EnginePreview } from './components/EnginePreview';
import { ControlTooltips } from './components/ControlTooltips';
import { ControlsHelp } from './components/ControlsHelp';

interface Workspace { id: string; history: History; saved: LevelFiles }
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
  return openLevel(`Level ${number}`, { data, graphics, clips });
}

function makeWorkspace(document: LevelDocument, id: string = crypto.randomUUID()): Workspace {
  return { id, history: { document, past: [], future: [] }, saved: document.files };
}

export function App() {
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const workspaceRef = useRef(workspace); workspaceRef.current = workspace;
  const [error, setError] = useState(''), [busy, setBusy] = useState('Opening workspace…');
  const [status, setStatus] = useState(''), [autosave, setAutosave] = useState('Local workspace');
  const [selection, setSelection] = useState<Selection>(null), [tab, setTab] = useState<'zones' | 'objects' | 'starts'>('zones');
  const [search, setSearch] = useState(''), [tool, setTool] = useState<'select' | 'pan'>('select');
  const [layer, setLayer] = useState<'lower' | 'upper'>('lower'), [grid, setGrid] = useState(64), [showObjects, setShowObjects] = useState(true);
  const [view, setView] = useState<'map' | 'split' | '3d'>('map');
  const [cursor, setCursor] = useState({ x: 0, z: 0 }), [fitKey, setFitKey] = useState(0), [focusZone, setFocusZone] = useState<{ id: number; request: number }>();
  const [navigatorOpen, setNavigatorOpen] = useState(false);
  const [help, setHelp] = useState(false), [choices, setChoices] = useState<ProjectCandidate[]>([]), [recents, setRecents] = useState<SavedWorkspace[]>([]), [showRecents, setShowRecents] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null), searchInput = useRef<HTMLInputElement>(null), generation = useRef(0);
  const document = workspace?.history.document;
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
  const dirty = useMemo(() => !!workspace && (['data', 'graphics', 'clips'] as const).some(file => !sameBytes(workspace.history.document.files[file], workspace.saved[file])), [workspace]);

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
          ? { ...latest, saved: current.history.document.files } : latest);
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
    setWorkspace(current => {
      if (!current) return current;
      try { return { ...current, history: direction === 'undo' ? undo(current.history) : redo(current.history) }; }
      catch (error) { queueMicrotask(() => fail(error)); return current; }
    });
  }, [fail]);

  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      const editing = event.target instanceof HTMLElement && (event.target.matches('input, textarea, select') || event.target.isContentEditable);
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') { event.preventDefault(); save(); return; }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'o') { event.preventDefault(); fileInput.current?.click(); return; }
      if (editing) return;
      if ((event.ctrlKey || event.metaKey) && ['z', 'y'].includes(event.key.toLowerCase())) { event.preventDefault(); historyAction(event.key.toLowerCase() === 'y' || event.shiftKey ? 'redo' : 'undo'); }
      if (event.key.toLowerCase() === 'f' && !event.ctrlKey && !event.metaKey) setFitKey(key => key + 1);
      if (event.key.toLowerCase() === 'v') setTool('select');
      if (event.key.toLowerCase() === 'h') setTool('pan');
      if (event.key === '/') { event.preventDefault(); flushSync(() => setNavigatorOpen(true)); searchInput.current?.focus(); }
      if (event.key === 'Escape') { setSelection(null); setHelp(false); setShowRecents(false); setChoices([]); setNavigatorOpen(false); }
      if (event.key === '?') setHelp(true);
    };
    window.addEventListener('keydown', key);
    return () => window.removeEventListener('keydown', key);
  }, [save, historyAction]);

  async function openDocument(next: LevelDocument, id?: string, request = ++generation.current) {
    const current = workspaceRef.current;
    if (current && (['data', 'graphics', 'clips'] as const).some(file => !sameBytes(current.history.document.files[file], current.saved[file])))
      await saveWorkspace(current.id, current.history.document);
    if (request !== generation.current) return;
    setWorkspace(makeWorkspace(next, id)); setSelection(null); setFocusZone(undefined); setSearch(''); setChoices([]); setShowRecents(false); setStatus(`Opened ${next.name}`);
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
      if (candidates.length === 1 && candidates[0]) await openDocument(openLevel(candidates[0].name, candidates[0].files), undefined, request);
      else setChoices(candidates);
    } catch (error) { if (request === generation.current) fail(error); }
    finally { if (request === generation.current) setBusy(''); }
  }

  const move = (selected: Exclude<Selection, null>, x: number, z: number) => {
    if (!document) return;
    try {
      const oldZone = selected.kind === 'object' ? document.objects.find(object => object.id === selected.id)?.zone : document.starts.find(start => start.player === selected.id)?.zone;
      const zone = document.zones.find(zone => zone.id === oldZone && zoneContainsPoint(document, zone, x, z)) ?? document.zones.find(zone => zoneContainsPoint(document, zone, x, z));
      if (!zone) throw new Error('That point is outside the level. Place the object inside a zone.');
      if (selected.kind === 'object') edit(moveObject(document, selected.id, x, z, zone.id));
      if (selected.kind === 'start') edit(setStart(document, selected.id as 1 | 2, x, z, zone.id));
    } catch (error) { fail(error); }
  };

  const filteredZones = document?.zones.filter(zone => String(zone.id).includes(search)) ?? [];
  const filteredObjects = document?.objects.filter(object => object.zone >= 0 && `${object.id} ${objectName(object.type)}`.toLowerCase().includes(search.toLowerCase())) ?? [];
  const activeObjects = document?.objects.filter(object => object.zone >= 0).length ?? 0;
  return <div className={`app ${navigatorOpen ? 'navigator-open' : ''}`} onDragOver={event => { if (event.dataTransfer.types.includes('Files')) event.preventDefault(); }} onDrop={event => { event.preventDefault(); if (event.dataTransfer.files.length) void importFiles([...event.dataTransfer.files]); }}>
    <ControlTooltips />
    <header className="app-header"><a className="brand" href="." aria-label="Alien Breed editor home"><span className="brand-mark">AB</span><span>ALIEN BREED<span className="brand-sub">LEVEL EDITOR</span></span></a><span className="header-divider" />
      <div className="document-title">{dirty && <span className="dirty-dot" title="Changes waiting to be saved in this browser" />}{document?.name ?? 'Workspace'}<span className="document-type">PROJECT</span></div>
      <div className="header-actions"><button onClick={() => fileInput.current?.click()} title="Open an .ab3d project, a level ZIP or the three twolev files together. Ctrl / Command + O."><Icon name="open" />Open</button><button onClick={save} disabled={!document} title="Download a portable project backup. Recovery is saved separately in this browser. Ctrl / Command + S."><Icon name="save" />Save project</button><button className="export-button" title="Download the three game-level files in a ZIP for the game." disabled={!document} onClick={() => { if (document) try { download(encodeGameLevel(document), `${document.name.replace(/[^\w -]/g, '').trim() || 'level'}-game.zip`); setStatus('Game level ZIP downloaded'); } catch (error) { fail(error); } }}><Icon name="export" />Export level</button><button className="icon-button" aria-label="Help and shortcuts" title="Show map, 3D inspection and playtest controls. Shortcut: ?" onClick={() => setHelp(true)}><Icon name="help" /></button></div>
      <input ref={fileInput} type="file" hidden multiple aria-label="Open level files" accept=".ab3d,.zip,.bin,.clips" onChange={event => { const files = [...(event.target.files ?? [])]; if (files.length) void importFiles(files); event.target.value = ''; }} />
    </header>
    {error && <div className="error-banner" role="alert"><Icon name="alert" /><span>{error}</span><button className="icon-button" aria-label="Dismiss error" title="Close this message. Your current document stays open." onClick={() => setError('')}><Icon name="close" size={16} /></button></div>}
    <div className="workspace">
      <aside className="navigator"><div className="panel-title">PROJECT <span className="eyebrow">LOCAL</span></div><div className="project-picker"><Icon name="layers" /><div><strong>{document?.name ?? 'Opening…'}</strong><span>{document?.zones.length ?? 0} zones · {activeObjects} objects</span></div></div>
        <label className="reference-picker"><span>Original levels</span><select aria-label="Original level" title="Open one of the sixteen shipped maps. Your current edits are saved in this browser first." value={/^Level (\d+)$/.exec(document?.name ?? '')?.[1] ?? ''} disabled={!!busy} onChange={event => { if (event.target.value) void loadOriginal(Number(event.target.value)); }}><option value="">Choose a reference level</option>{Array.from({ length: 16 }, (_, i) => <option key={i} value={i + 1}>Level {String(i + 1).padStart(2, '0')}</option>)}</select></label>
        <div className="navigator-tabs" role="tablist" aria-label="Map contents">{(['zones', 'objects', 'starts'] as const).map(value => <button key={value} role="tab" title={value === 'starts' ? 'Find and select player starting positions.' : value === 'zones' ? 'Find and select rooms by zone number.' : 'Find enemies, pickups and scenery by name.'} aria-selected={tab === value} className={tab === value ? 'active' : ''} onClick={() => { setTab(value); setSearch(''); }}>{value === 'starts' ? 'Starts' : value[0]?.toUpperCase() + value.slice(1)}</button>)}</div>
        <label className="search-box"><Icon name="search" size={15} /><input ref={searchInput} aria-label="Search map contents" title="Find a zone number or object name. Press / from the map to focus search." value={search} onChange={event => setSearch(event.target.value)} placeholder={tab === 'zones' ? 'Find a zone…' : 'Find an object…'} /><kbd>/</kbd></label>
        <div className="contents-list">
          {tab === 'zones' && filteredZones.map(zone => <button key={zone.id} className={`content-row ${selection?.kind === 'zone' && selection.id === zone.id ? 'selected' : ''}`} title={`Select zone ${zone.id} and centre the map. In 3D inspection, visit this zone.`} onClick={() => { setSelection({ kind: 'zone', id: zone.id }); setFocusZone(current => ({ id: zone.id, request: (current?.request ?? 0) + 1 })); setNavigatorOpen(false); }}><span className={`zone-dot ${document?.mechanisms.some(mechanism => mechanism.zone === zone.id && mechanism.kind === 'door') ? 'door-dot' : zone.water > zone.ceiling && zone.water < zone.floor ? 'water-dot' : ''}`} /><span>Zone {String(zone.id).padStart(3, '0')}</span>{zone.upper.length > 0 && <span className="row-badge">2F</span>}<span className="row-detail">{zone.floor}</span></button>)}
          {tab === 'objects' && filteredObjects.map(object => <button key={object.id} className={`content-row ${selection?.kind === 'object' && selection.id === object.id ? 'selected' : ''}`} title={`Select ${objectName(object.type)} ${object.id}. Drag it on the map to move it.`} onClick={() => { setSelection({ kind: 'object', id: object.id }); setFocusZone(current => ({ id: object.zone, request: (current?.request ?? 0) + 1 })); setNavigatorOpen(false); }}><Icon name="object" size={14} /><span>{objectName(object.type)}</span><span className="row-detail">{object.id}</span></button>)}
          {tab === 'starts' && document?.starts.map(start => <button key={start.player} className={`content-row ${selection?.kind === 'start' && selection.id === start.player ? 'selected' : ''}`} title={`Select player ${start.player} start. Drag the triangle on the map to reposition it.`} onClick={() => { setSelection({ kind: 'start', id: start.player }); setFocusZone(current => ({ id: start.zone, request: (current?.request ?? 0) + 1 })); setNavigatorOpen(false); }}><Icon name="pin" size={14} /><span>Player {start.player}</span><span className="row-detail">Z{start.zone}</span></button>)}
          {tab !== 'starts' && (tab === 'zones' ? filteredZones : filteredObjects).length === 0 && <div className="list-empty">No matches</div>}
        </div>
        <button className="recent-button" title="Open a workspace previously saved in this browser." onClick={() => { listWorkspaces().then(setRecents).catch(fail); setShowRecents(true); }}><Icon name="open" size={15} />Recent workspaces<span>↗</span></button>
      </aside>
      <main className="map-main"><div className="map-toolbar"><div className="tool-group"><button className={`icon-button ${tool === 'select' ? 'active' : ''}`} aria-label="Select tool" aria-pressed={tool === 'select'} title="Click to select a zone, object or start. Drag objects and starts to move them. Shortcut: V." onClick={() => setTool('select')}><Icon name="select" /></button><button className={`icon-button ${tool === 'pan' ? 'active' : ''}`} aria-label="Pan tool" aria-pressed={tool === 'pan'} title="Drag to move around the map. Shortcut: H, or hold Space while dragging. Middle/right drag also pans." onClick={() => setTool('pan')}><Icon name="hand" /></button></div><span className="toolbar-divider" />
        <button className="icon-button" aria-label="Undo" title={`Undo ${workspace?.history.past.at(-1)?.label ?? ''} (Ctrl+Z)`} disabled={!workspace?.history.past.length} onClick={() => historyAction('undo')}><Icon name="undo" /></button><button className="icon-button" aria-label="Redo" title="Redo (Ctrl+Shift+Z)" disabled={!workspace?.history.future.length} onClick={() => historyAction('redo')}><Icon name="redo" /></button><span className="toolbar-divider" />
        <div className="layer-toggle"><button className={layer === 'lower' ? 'active' : ''} title="Inspect the lower floor of each zone." onClick={() => setLayer('lower')}>Lower</button><button className={layer === 'upper' ? 'active' : ''} title="Inspect upper-floor geometry where a zone has two floors." onClick={() => setLayer('upper')}>Upper</button></div><span className="toolbar-spacer" />
        <label className="grid-select"><Icon name="grid" size={15} /><select aria-label="Grid spacing" title="Set snapping for dragged objects and starts. Hold Alt to bypass snapping; Grid off uses whole coordinates." value={grid} onChange={event => setGrid(Number(event.target.value))}>{[0, 8, 16, 32, 64, 128].map(value => <option key={value} value={value}>{value === 0 ? 'Grid off' : `Grid ${value}`}</option>)}</select></label><button className={`icon-button ${showObjects ? 'active' : ''}`} aria-label="Show objects" title="Show or hide enemies, pickups and scenery on the map." aria-pressed={showObjects} onClick={() => setShowObjects(show => !show)}><Icon name="eye" /></button><button className="icon-button" title="Fit the whole level into the map view. Shortcut: F." aria-label="Frame map" onClick={() => setFitKey(key => key + 1)}><Icon name="fit" /></button>
      </div>
        <div className="view-tabs" role="group" aria-label="Viewport layout"><button className="project-toggle" aria-label="Toggle project navigator" title="Show or hide the level picker and map contents. Shortcut / opens search." aria-expanded={navigatorOpen} onClick={() => setNavigatorOpen(open => !open)}><Icon name="layers" size={14} />Project</button>{(['map', 'split', '3d'] as const).map(mode => <button key={mode} aria-pressed={view === mode} className={view === mode ? 'active' : ''} title={mode === 'map' ? 'Edit and inspect the overhead map.' : mode === 'split' ? 'Show the map and actual game renderer together.' : 'Inspect the level in the actual game renderer.'} onClick={() => setView(mode)}>{mode === 'map' ? '2D map' : mode === 'split' ? 'Split view' : '3D view'}</button>)}</div>
        {document ? <CanvasBoundary key={workspace?.id} onExport={save}><div className={`viewports viewports-${view}`}>
          {view !== '3d' && <MapCanvas document={document} selection={selection} onSelect={setSelection} onMove={move} tool={tool} layer={layer} grid={grid} showObjects={showObjects} onCursor={(x, z) => setCursor({ x, z })} fitKey={fitKey} focusZone={focusZone} />}
          {view !== 'map' && <EnginePreview key={workspace?.id} document={document} selection={selection} layer={layer} onSelectZone={id => setSelection({ kind: 'zone', id })} />}
        </div></CanvasBoundary> : <div className="canvas-error"><Icon name="layers" size={36} /><h2>Open a level to begin</h2><p>Drop a project, level ZIP or the three twolev files here.</p><button className="primary-button" onClick={() => fileInput.current?.click()}>Open files</button></div>}
        {busy && <div className="busy-overlay" role="status"><span className="spinner" />{busy}</div>}
      </main>
      {document && <Inspector document={document} selection={selection} edit={edit} />}
    </div>
    <footer className="statusbar"><span className="status-dot" /><span>{autosave}</span><span className="status-message" aria-live="polite">{status || 'Choose a zone to start editing'}</span><span>X <strong>{Math.round(cursor.x)}</strong></span><span>Z <strong>{Math.round(cursor.z)}</strong></span><span className="status-size">{document?.points.length ?? 0} points · {document?.zones.length ?? 0} zones</span></footer>
    {(help || showRecents || choices.length > 0) && <div className="modal-backdrop" onPointerDown={event => { if (event.target === event.currentTarget) { setHelp(false); setShowRecents(false); setChoices([]); } }}><section className="modal" role="dialog" aria-modal="true" aria-label={help ? 'Editor help' : choices.length ? 'Choose a level' : 'Recent workspaces'}><button className="modal-close icon-button" aria-label="Close dialog" title="Close this panel. Shortcut: Esc." onClick={() => { setHelp(false); setShowRecents(false); setChoices([]); }}><Icon name="close" /></button>
      {help && <ControlsHelp />}
      {showRecents && <><span className="eyebrow">SAVED IN THIS BROWSER</span><h2>Recent workspaces</h2>{recents.length === 0 && <p>No workspaces have been saved yet.</p>}{recents.map(recent => <button className="recent-entry" key={recent.id} onClick={() => { try { void openDocument(decodeProject(recent.bytes), recent.id).catch(fail); } catch (error) { fail(error); } }}><Icon name="layers" /><span><strong>{recent.name}</strong><small>{new Date(recent.updatedAt).toLocaleString()}</small></span><Icon name="chevron" size={15} /></button>)}</>}
      {choices.length > 0 && <><span className="eyebrow">{choices.length} LEVELS FOUND</span><h2>Choose a level to open</h2>{choices.map(choice => <button className="recent-entry" key={choice.folder} onClick={() => { try { void openDocument(openLevel(choice.name, choice.files)).catch(fail); } catch (error) { fail(error); } }}><Icon name="layers" /><span><strong>{choice.name}</strong><small>{choice.folder}</small></span><Icon name="chevron" size={15} /></button>)}</>}
    </section></div>}
  </div>;
}
