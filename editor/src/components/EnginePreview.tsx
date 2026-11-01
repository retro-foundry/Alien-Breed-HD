import { pointInRoomOutline } from '../model/geometry';
import { authoredMovingSurfaceOwner } from '../model/authoredMovingRoom';
import { useEffect, useRef, useState } from 'react';
import { type LevelDocument } from '../model/level';
import { type Selection } from './MapCanvas';
import { Icon } from './Icon';
import { previewKeys, playKeys } from './ControlsHelp';
import { encodeLevelSetup } from '../model/levelSetup';
import { validateAnimationReferences } from '../model/animationBuild';
import { preparePlayableLevel, type PlayableLevel } from '../model/playableBuild';
import { wallOverrideFiles } from '../model/wallOverrides';
import { resolveEngineSurface, type EngineSurfacePick, type NativeSurfacePick } from '../model/engineSurfacePick';
import { resolveEngineObject, type NativeObjectPick } from '../model/engineObjectPick';
import { inspectionShortcut, inspectionShortcutScope, type InspectionShortcut } from '../../runtime/inspection-shortcuts.mjs';
import { surfaceMaterialTarget, type SurfaceMaterialTarget } from '../model/wallMaterials';
import type { EditCommand } from '../model/commands';
import { WallPaintDialog } from './WallPaintDialog';
import { PolygonPaintDialog } from './PolygonPaintDialog';
import { polygonMaterialTarget, type PolygonMaterialTarget } from '../model/polygonMaterials';

interface NativeEditorCamera { zone: number; x: number; z: number; y: number; angle: number; look: number; upper: boolean }
interface EngineMessage { camera?: NativeEditorCamera; world?: { x: number; z: number; upper: boolean } | null; source: 'ab3d-engine'; type: string; revision?: number; zone?: number; zones?: number; message?: string; nonblack?: number; slot?: number; exitZone?: number; surface?: NativeSurfacePick | null; object?: NativeObjectPick | null; shortcut?: unknown }
interface Props { editorView?: boolean; placement?: { name: string; place: (room: string | undefined, x: number, z: number, upper?: boolean) => void }; onEngineView?: () => void; document: LevelDocument; selection: Selection; sourceRoomId?: string; layer: 'lower' | 'upper'; onSelectZone: (zone: number) => void; onSelectRoom: (room: string) => void; onSelectObject: (id: number) => void; onShortcut: (shortcut: InspectionShortcut) => void; onEdit: (command: EditCommand) => void; onSlotChange: (slot: number) => void }

export function EnginePreview({ editorView = false, placement, onEngineView, document, selection, sourceRoomId, layer, onSelectZone, onSelectRoom, onSelectObject, onShortcut, onEdit, onSlotChange }: Props) {
  const frame = useRef<HTMLIFrameElement>(null);
  const editorCamera = useRef<{ camera: NativeEditorCamera; room?: string } | undefined>(undefined);
  const [ready, setReady] = useState(false), [loaded, setLoaded] = useState(false);
  const [playing, setPlaying] = useState(false), [starting, setStarting] = useState(false);
  const [error, setError] = useState(''), [notice, setNotice] = useState('Loading the game engine…');
  const [pickedSurface, setPickedSurface] = useState<EngineSurfacePick>();
  const [wallPaint, setWallPaint] = useState<{ source: LevelDocument; target: SurfaceMaterialTarget }>();
  const [polygonPaint, setPolygonPaint] = useState<{ source: LevelDocument; target: PolygonMaterialTarget }>();
  const slot = document.playtestSlot;
  const [revision, setRevision] = useState(0);
  const [nonblack, setNonblack] = useState(0);
  const [actualSlot, setActualSlot] = useState(-1);
  const [actualExit, setActualExit] = useState<number>();
  const [spawnZone, setSpawnZone] = useState<number>();
  const revisionRef = useRef(0), pendingPlay = useRef(false);
  const hasLoadedFrame = useRef(false);
  const pickedViewTarget = useRef<{ source: LevelDocument; zone?: number; room?: string; object?: number; layer: Props['layer'] } | undefined>(undefined);
  const reloadTimer = useRef<number | undefined>(undefined);
  const failedDocument = useRef<LevelDocument | undefined>(undefined);
  const artifact = useRef<PlayableLevel | undefined>(undefined), previousDocument = useRef(document), readyRef = useRef(ready);
  readyRef.current = ready;
  const current = useRef({ editorView, placement, document, selection, sourceRoomId, layer, onSelectZone, onSelectRoom, onSelectObject, onShortcut, inspection: loaded && !playing && !starting, paused: !playing && !starting });
  current.current = { editorView, placement, document, selection, sourceRoomId, layer, onSelectZone, onSelectRoom, onSelectObject, onShortcut, inspection: loaded && !playing && !starting, paused: !playing && !starting };
  const send = (type: string, detail: object = {}) => frame.current?.contentWindow?.postMessage({ source: 'ab3d-editor', type, ...detail }, location.origin);
  function stopInvalidPreview(message: string, notice: string) {
    failedDocument.current = current.current.document;
    send('play', { playing: false }); pendingPlay.current = false; setPlaying(false); setStarting(false); setLoaded(false);
    setNotice(notice); setError(message);
    setActualSlot(-1); setActualExit(undefined); setSpawnZone(undefined); setNonblack(0);
  }
  function positionCamera() {
    setPickedSurface(undefined);
    const { document: source, selection, sourceRoomId, layer } = current.current;
    const built = artifact.current;
    if (!built || built.source !== source) return;
    const document = built.document;
    const object = selection?.kind === 'object' ? document.objects.find(object => object.id === selection.id) : undefined;
    const start = selection?.kind === 'start' ? document.starts.find(start => start.player === selection.id) : undefined;
    if (built.topology && !object && !start) {
      const room = sourceRoomId ?? (selection?.kind === 'zone' ? source.roomSource.rooms.find(room => room.originZone === selection.id)?.id : undefined);
      const cell = built.topology.cells.find(cell => cell.sourceId === room);
      if (!cell) return;
      const center = (target: typeof cell) => {
        const points = target.vertices.map(id => built.topology!.points[id]!);
        return { x: Math.round(points.reduce((sum, point) => sum + point.x, 0) / points.length),
          z: Math.round(points.reduce((sum, point) => sum + point.z, 0) / points.length) };
      };
      let view = cell, angle = 0;
      const dimensions = document.zones.find(zone => zone.id === cell.id)!;
      // A closed door has no interior air for an inspection camera. Keep its
      // inspector selected while looking toward it from an adjoining room.
      if (layer === 'lower' && dimensions.floor === dimensions.ceiling && source.roomSource.rooms.find(item => item.id === room)?.moving?.kind === 'door') {
        const adjacent = cell.neighbours.map(id => built.topology!.cells.find(item => item.id === id)).find(item => item && document.zones[item.id]!.floor > document.zones[item.id]!.ceiling);
        if (adjacent) {
          view = adjacent;
          const target = center(cell), origin = center(view);
          angle = Math.round(Math.atan2(target.x - origin.x, target.z - origin.z) * 8192 / (Math.PI * 2)) & 8191;
        }
      }
      send('camera', { camera: { zone: view.id, ...center(view),
        upper: layer === 'upper' && document.zones.find(zone => zone.id === view.id)!.upper.length > 0 ? 1 : 0, angle } });
      return;
    }
    const zoneId = selection?.kind === 'zone' ? selection.id : object?.zone ?? start?.zone;
    const zone = document.zones.find(zone => zone.id === zoneId);
    if (!zone) return;
    const lines = zone.boundaryIds.map(id => document.lines[id]).filter(line => !!line);
    if (lines.length === 0) return;
    const x = Math.round(object?.x ?? start?.x ?? lines.reduce((sum, line) => sum + line.x, 0) / lines.length);
    const z = Math.round(object?.z ?? start?.z ?? lines.reduce((sum, line) => sum + line.z, 0) / lines.length);
    send('camera', { camera: { zone: zone.id, x, z, upper: layer === 'upper' && zone.upper.length > 0 ? 1 : 0, angle: 0 } });
  }

  async function load(play = false) {
    if (!ready || (!play && (pendingPlay.current || failedDocument.current === current.current.document))) return;
    if (play) failedDocument.current = undefined;
    // An explicit play request supersedes the debounced inspection load.
    // Otherwise a slow first load can receive a second, paused load request
    // before its reply arrives, clearing the user's pending play intent.
    if (play && reloadTimer.current !== undefined) {
      window.clearTimeout(reloadTimer.current); reloadTimer.current = undefined;
    }
    const source = current.current.document;
    setPickedSurface(undefined);
    pendingPlay.current = play;
    setStarting(play); setLoaded(false); setError(''); setNotice('Building current level…');
    const next = ++revisionRef.current; setRevision(next);
    try {
      const built = await preparePlayableLevel(source);
      if (source !== current.current.document || next !== revisionRef.current || !readyRef.current || !frame.current) return;
      validateAnimationReferences(built.document); artifact.current = built;
      setNotice('Loading current level…');
      send('load', { slot: source.playtestSlot, revision: next, files: built.document.files, setup: encodeLevelSetup(built.document.exitZone, built.document.drawGroups),
        wallAssets: Object.entries(wallOverrideFiles(source.wallOverrides)).map(([path, bytes]) => ({ name: path.split('/').at(-1), bytes })) });
    } catch (error) {
      if (source === current.current.document && next === revisionRef.current && frame.current)
        stopInvalidPreview(error instanceof Error ? error.message : String(error), 'Build needs attention');
    }
  }

  useEffect(() => {
    const receive = (event: MessageEvent<EngineMessage>) => {
      if (event.origin !== location.origin || event.source !== frame.current?.contentWindow || event.data?.source !== 'ab3d-engine') return;
      const message = event.data;
      if (message.type === 'shortcut') {
        const shortcut = inspectionShortcut(message.shortcut), scope = inspectionShortcutScope(shortcut);
        const currentFrame = message.revision === revisionRef.current && current.current.inspection && artifact.current?.source === current.current.document;
        if (shortcut && Number.isSafeInteger(message.revision) && message.revision! >= 0 && hasLoadedFrame.current
          && current.current.paused && (scope === 'workspace' || currentFrame)) current.current.onShortcut(shortcut);
        return;
      }
      if (message.revision !== undefined && message.revision >= 0 && message.revision !== revisionRef.current) return;
      if (message.type === 'ready') { setReady(true); setError(''); }
      if (message.type === 'loaded' && message.revision === revisionRef.current && artifact.current?.source === current.current.document) {
        hasLoadedFrame.current = true;
        setNonblack(message.nonblack ?? 0);
        setActualSlot(message.slot ?? -1);
        setActualExit(message.exitZone);
        setSpawnZone(message.zone);
        setLoaded(true); setNotice(`${message.zones} zones loaded · Game engine`);
        if (pendingPlay.current) { pendingPlay.current = false; setPlaying(true); setStarting(false); send('play', { playing: true }); }
        else if (current.current.editorView && editorCamera.current) {
          const saved = editorCamera.current, built = artifact.current!, candidates = built.topology?.cells.filter(cell => cell.sourceId === saved.room);
          const cell = candidates?.find(cell => pointInRoomOutline(cell.vertices.map(id => built.topology!.points[id]!), saved.camera));
          send('restore-camera', { camera: { ...saved.camera, zone: cell?.id ?? saved.camera.zone } });
        } else { positionCamera(); if (current.current.editorView) send('camera-state'); }
      }
      if (message.type === 'error') {
        failedDocument.current = current.current.document; pendingPlay.current = false;
        setNotice('Engine preview unavailable');
        setError(message.message ?? 'The engine preview stopped.'); setStarting(false); setPlaying(false); setLoaded(false);
      }
      if (message.type === 'stopped') { setPlaying(false); setStarting(false); setNotice('Playtest ended. Build and play again to restart.'); }
      if (message.type === 'view-changed' || message.type === 'camera-state') {
        const camera = message.camera, built = artifact.current, previous = editorCamera.current?.camera;
        if (message.type === 'view-changed' && (!camera || !previous || (['zone', 'x', 'z', 'y', 'angle', 'look', 'upper'] as const).some(key => camera[key] !== previous[key]))) setPickedSurface(undefined);
        if (camera && [camera.zone,camera.x,camera.z,camera.y,camera.angle,camera.look].every(Number.isFinite) && built?.source === current.current.document) editorCamera.current = { camera, room: built.topology?.cells.find(cell => cell.id === camera.zone)?.sourceId ?? built.source.roomSource.rooms.find(room => room.originZone === camera.zone)?.id };
      }
      if (message.type === 'camera-invalid') { positionCamera(); setNotice('Camera returned inside the edited room after its geometry changed.'); }
      if (message.type === 'pick') {
        if (current.current.placement && current.current.inspection && artifact.current?.source === current.current.document) {
          const built = artifact.current, world = message.world;
          const room = built.topology?.cells.find(cell => cell.id === message.zone)?.sourceId ?? built.source.roomSource.rooms.find(room => room.originZone === message.zone)?.id;
          current.current.placement.place(world && Number.isFinite(world.x) && Number.isFinite(world.z) ? room : undefined, world?.x ?? 0, world?.z ?? 0, world?.upper);
          return;
        }
        const built = artifact.current;
        setPickedSurface(built?.source === current.current.document && message.zone !== undefined
          ? resolveEngineSurface(built.document, message.zone, message.surface) : undefined);
        if (built?.source === current.current.document && message.object) {
          const id = resolveEngineObject(built, message.object);
          if (id !== undefined) {
            pickedViewTarget.current = { source: built.source, layer: current.current.layer, object: id };
            current.current.onSelectObject(id);
          }
          // Opaque unsupported/runtime records cannot select a room behind them.
          return;
        }
      }
      if (message.type === 'pick' && message.zone !== undefined && message.zone >= 0) {
        const built = artifact.current;
        if (!built || built.source !== current.current.document) return;
        const pick = { source: built.source, layer: current.current.layer };
        if (!built.topology) {
          pickedViewTarget.current = { ...pick, zone: message.zone }; current.current.onSelectZone(message.zone);
        }
        else {
          const id = built.topology.cells.find(cell => cell.id === message.zone)?.sourceId;
          const room = built.source.roomSource.rooms.find(room => room.id === id);
          if (room?.originZone !== null && room?.originZone !== undefined) {
            pickedViewTarget.current = { ...pick, zone: room.originZone }; current.current.onSelectZone(room.originZone);
          } else if (room) {
            pickedViewTarget.current = { ...pick, room: room.id }; current.current.onSelectRoom(room.id);
          }
        }
      }
    };
    window.addEventListener('message', receive);
    return () => window.removeEventListener('message', receive);
  }, []);

  useEffect(() => {
    if (!ready || playing || starting) return;
    const timer = window.setTimeout(() => { reloadTimer.current = undefined; void load(); }, 180);
    reloadTimer.current = timer;
    return () => { window.clearTimeout(timer); if (reloadTimer.current === timer) reloadTimer.current = undefined; };
  }, [ready, document, slot, playing, starting]);
  useEffect(() => {
    if (previousDocument.current === document) return;
    previousDocument.current = document; ++revisionRef.current; pendingPlay.current = false; artifact.current = undefined;
    setWallPaint(undefined);
    setPolygonPaint(undefined);
    setPickedSurface(undefined);
    send('play', { playing: false }); setPlaying(false); setStarting(false); setLoaded(false);
    setActualSlot(-1); setActualExit(undefined); setSpawnZone(undefined); setNonblack(0);
  }, [document]);

  useEffect(() => {
    const picked = pickedViewTarget.current; pickedViewTarget.current = undefined;
    // Picking changes the inspector, not the inspection camera. A later map /
    // list selection still visits its room normally, including the same room.
    if (picked?.source === document && picked.layer === layer && (
      picked.room !== undefined && picked.room === sourceRoomId
      || picked.object !== undefined && selection?.kind === 'object' && selection.id === picked.object
      || picked.zone !== undefined && selection?.kind === 'zone' && selection.id === picked.zone)) return;
    if (loaded && !playing) positionCamera();
  }, [selection, sourceRoomId, layer]);
  useEffect(() => {
    if (ready) send('instructions', { text: (playing ? playKeys : previewKeys).map(([key, action]) => `${key}: ${action}`).join(' · ') });
  }, [ready, playing]);

  useEffect(() => { if (editorView && (playing || starting)) stop(); }, [editorView, playing, starting]);
  function stop() { send('play', { playing: false }); setPlaying(false); setStarting(false); pendingPlay.current = false; }
  function reloadEngine() {
    hasLoadedFrame.current = false;
    ++revisionRef.current; artifact.current = undefined;
    pendingPlay.current = false; failedDocument.current = undefined;
    setPlaying(false); setStarting(false); setReady(false); setLoaded(false); setError(''); setNotice('Loading the game engine…');
    frame.current?.contentWindow?.location.reload();
  }
  const movingOwner = pickedSurface && artifact.current?.source === document ? authoredMovingSurfaceOwner(artifact.current, pickedSurface.zone, pickedSurface) : undefined;
  const paintTarget = pickedSurface && artifact.current?.source === document ? surfaceMaterialTarget(artifact.current, pickedSurface.zone, pickedSurface) : undefined;
  const polygonTarget = pickedSurface && artifact.current?.source === document ? polygonMaterialTarget(artifact.current, pickedSurface.zone, pickedSurface) : undefined;
  return <section className="engine-preview" aria-label={editorView ? "3D editing view" : "3D game engine preview"} data-revision={revision} data-rendered-pixels={nonblack} data-level-slot={actualSlot} data-exit-zone={actualExit} data-spawn-zone={spawnZone}>
    {wallPaint && wallPaint.source === document && <WallPaintDialog document={document} target={wallPaint.target} onCancel={() => setWallPaint(undefined)} onApply={command => { onEdit(command); setWallPaint(undefined); }} />}
    {polygonPaint && polygonPaint.source === document && <PolygonPaintDialog document={document} target={polygonPaint.target} onCancel={() => setPolygonPaint(undefined)} onApply={command => { onEdit(command); setPolygonPaint(undefined); }} />}
    <div className="engine-toolbar"><span className="eyebrow">{editorView ? "3D EDITOR" : "ENGINE VIEW"}</span><span className="engine-status" role="status">{playing ? 'Playtest running' : notice}</span>
      {!editorView && <><label className="engine-slot">Slot <select aria-label="Playtest level slot" title="Choose the project's game level slot. Saved with your project and undoable; this controls the game's level-specific setup." value={slot} disabled={playing || starting} onChange={event => onSlotChange(Number(event.target.value))}>{Array.from({ length: 16 }, (_, i) => <option key={i} value={i}>{i + 1}</option>)}</select></label>
      <button title={playing ? 'Stop playing and return to inspection. Your editable map is kept.' : 'Load the latest map into the game and start from player one. Click the game view to capture the mouse.'} disabled={!ready || starting} className={playing ? '' : 'primary-button'} onClick={() => { if (playing) stop(); else load(true); }}>{playing ? 'Stop' : starting ? 'Starting…' : 'Build & play'}</button></>}
      {editorView && <button title="Open Engine view for a gameplay test. Editing navigation does not change the saved player start." onClick={onEngineView}>Playtest</button>}
    </div>
    <div className="engine-stage"><iframe ref={frame} title="Alien Breed game engine" src={`${import.meta.env.BASE_URL}${import.meta.env.VITE_EDITOR_RUNTIME_PATH ?? 'runtime/'}frame.html`} allow="autoplay; fullscreen" onError={() => setError('The game preview could not be loaded. Your map is still available. Reload the preview to try again.')} />
      {!playing && loaded && <span className="engine-crosshair" aria-hidden="true">+</span>}
      {error && <div className="engine-error" role="alert"><Icon name="alert" /><p>{error}</p><button title="Restart the preview engine. Your editable document is kept." onClick={reloadEngine}>Reload engine</button></div>}
      {!playing && loaded && pickedSurface && <div className="engine-surface-info" role="note" aria-label="Picked surface" data-graphic={pickedSurface.offset} data-storey={pickedSurface.upper ? 'upper' : 'lower'}>
      <strong>{pickedSurface.kind} · {pickedSurface.upper ? 'Upper' : 'Lower'}</strong>
      {pickedSurface.texture !== undefined && <span>Material {pickedSurface.texture}</span>}
      {pickedSurface.height !== undefined && <span>Authored height {pickedSurface.height}</span>}
      <span>Room settings are in the inspector.</span>
      {paintTarget && <button type="button" className="small-button" style={{ pointerEvents: 'auto' }} title="Choose a material for this source wall only, including its generated pieces. Other faces keep their materials. Opens a cancellable draft."
        onClick={() => setWallPaint({ source: document, target: paintTarget })}>Paint this wall</button>}
      {movingOwner && <button type="button" className="small-button" style={{ pointerEvents: 'auto' }} title="Select this mechanism to edit its material and movement settings." onClick={() => onSelectRoom(movingOwner.room)}>Edit {movingOwner.kind}</button>}
      {polygonTarget && <button type="button" className="small-button" style={{ pointerEvents: 'auto' }} title="Choose a normal, bump or smooth material and its repeat for this source floor or ceiling surface. Opens a cancellable draft."
        onClick={() => setPolygonPaint({ source: document, target: polygonTarget })}>Paint this surface</button>}
      {!paintTarget && !polygonTarget && !movingOwner && artifact.current?.source === document && <span>This special surface has no painting control yet. Straight walls and normal, bump and smooth floor/ceiling materials are supported.</span>}
    </div>}
    </div>
    <div className="engine-controls">{playing ? <span>Click to capture mouse · WASD move · C crouch · Space use · Esc release mouse · Stop returns to editing</span> : <><button disabled={!loaded} aria-label="Turn preview left" title="Turn the inspection camera left. Left Arrow also turns when Engine view has focus." onClick={() => send('turn', { delta: -256 })}>←</button><button disabled={!loaded} aria-label="Turn preview right" title="Turn the inspection camera right. Right Arrow also turns when Engine view has focus." onClick={() => send('turn', { delta: 256 })}>→</button><button disabled={!loaded} aria-label="Look up in preview" title="Look up. Up Arrow also looks up when Engine view has focus." onClick={() => send('turn', { look: -12 })}>↑</button><button disabled={!loaded} aria-label="Look down in preview" title="Look down. Down Arrow also looks down when Engine view has focus." onClick={() => send('turn', { look: 12 })}>↓</button><span>{placement ? `Placing ${placement.name}: click a floor. Escape cancels.` : "WASD move / Right drag look / Q/E down/up / Shift faster / Click to edit / Controls for more keys"}</span></>}
      <details className="engine-key-help"><summary title="Show the keys for the current 3D mode. Enter or Space opens this list.">Controls</summary><div><strong>{playing ? 'Playtest controls' : '3D inspection controls'}</strong>{(playing ? playKeys : previewKeys).map(([key, action]) => <p key={key}><span>{action}</span><kbd>{key}</kbd></p>)}</div></details>
    </div>
  </section>;
}
