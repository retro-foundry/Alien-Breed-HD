import { useEffect, useRef, useState } from 'react';
import { type LevelDocument } from '../model/level';
import { type Selection } from './MapCanvas';
import { Icon } from './Icon';
import { previewKeys, playKeys } from './ControlsHelp';
import { encodeLevelSetup } from '../model/levelSetup';
import { validateAnimationReferences } from '../model/animationBuild';
import { preparePlayableLevel, type PlayableLevel } from '../model/playableBuild';
import { wallOverrideFiles } from '../model/wallOverrides';

interface EngineMessage { source: 'ab3d-engine'; type: string; revision?: number; zone?: number; zones?: number; message?: string; nonblack?: number; slot?: number; exitZone?: number }
interface Props { document: LevelDocument; selection: Selection; sourceRoomId?: string; layer: 'lower' | 'upper'; onSelectZone: (zone: number) => void; onSelectRoom: (room: string) => void; onSlotChange: (slot: number) => void }

export function EnginePreview({ document, selection, sourceRoomId, layer, onSelectZone, onSelectRoom, onSlotChange }: Props) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [ready, setReady] = useState(false), [loaded, setLoaded] = useState(false);
  const [playing, setPlaying] = useState(false), [starting, setStarting] = useState(false);
  const [error, setError] = useState(''), [notice, setNotice] = useState('Loading the game engine…');
  const slot = document.playtestSlot;
  const [revision, setRevision] = useState(0);
  const [nonblack, setNonblack] = useState(0);
  const [actualSlot, setActualSlot] = useState(-1);
  const [actualExit, setActualExit] = useState<number>();
  const [spawnZone, setSpawnZone] = useState<number>();
  const revisionRef = useRef(0), pendingPlay = useRef(false);
  const pickedViewTarget = useRef<{ source: LevelDocument; zone?: number; room?: string; layer: Props['layer'] } | undefined>(undefined);
  const reloadTimer = useRef<number | undefined>(undefined);
  const failedDocument = useRef<LevelDocument | undefined>(undefined);
  const artifact = useRef<PlayableLevel | undefined>(undefined), previousDocument = useRef(document), readyRef = useRef(ready);
  readyRef.current = ready;
  const current = useRef({ document, selection, sourceRoomId, layer, onSelectZone, onSelectRoom });
  current.current = { document, selection, sourceRoomId, layer, onSelectZone, onSelectRoom };
  const send = (type: string, detail: object = {}) => frame.current?.contentWindow?.postMessage({ source: 'ab3d-editor', type, ...detail }, location.origin);
  function stopInvalidPreview(message: string, notice: string) {
    failedDocument.current = current.current.document;
    send('play', { playing: false }); pendingPlay.current = false; setPlaying(false); setStarting(false); setLoaded(false);
    setNotice(notice); setError(message);
    setActualSlot(-1); setActualExit(undefined); setSpawnZone(undefined); setNonblack(0);
  }
  function positionCamera() {
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
      const points = cell.vertices.map(id => built.topology!.points[id]!);
      send('camera', { camera: { zone: cell.id,
        x: Math.round(points.reduce((sum, point) => sum + point.x, 0) / points.length),
        z: Math.round(points.reduce((sum, point) => sum + point.z, 0) / points.length),
        upper: layer === 'upper' && document.zones.find(zone => zone.id === cell.id)!.upper.length > 0 ? 1 : 0, angle: 0 } });
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
      if (message.revision !== undefined && message.revision >= 0 && message.revision !== revisionRef.current) return;
      if (message.type === 'ready') { setReady(true); setError(''); }
      if (message.type === 'loaded' && message.revision === revisionRef.current && artifact.current?.source === current.current.document) {
        setNonblack(message.nonblack ?? 0);
        setActualSlot(message.slot ?? -1);
        setActualExit(message.exitZone);
        setSpawnZone(message.zone);
        setLoaded(true); setNotice(`${message.zones} zones loaded · Game engine`);
        if (pendingPlay.current) { pendingPlay.current = false; setPlaying(true); setStarting(false); send('play', { playing: true }); }
        else positionCamera();
      }
      if (message.type === 'error') {
        failedDocument.current = current.current.document; pendingPlay.current = false;
        setError(message.message ?? 'The engine preview stopped.'); setStarting(false); setPlaying(false); setLoaded(false);
      }
      if (message.type === 'stopped') { setPlaying(false); setStarting(false); setNotice('Playtest ended. Build and play again to restart.'); }
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
    send('play', { playing: false }); setPlaying(false); setStarting(false); setLoaded(false);
    setActualSlot(-1); setActualExit(undefined); setSpawnZone(undefined); setNonblack(0);
  }, [document]);

  useEffect(() => {
    const picked = pickedViewTarget.current; pickedViewTarget.current = undefined;
    // Picking changes the inspector, not the inspection camera. A later map /
    // list selection still visits its room normally, including the same room.
    if (picked?.source === document && picked.layer === layer && (
      picked.room !== undefined && picked.room === sourceRoomId
      || picked.zone !== undefined && selection?.kind === 'zone' && selection.id === picked.zone)) return;
    if (loaded && !playing) positionCamera();
  }, [selection, sourceRoomId, layer, loaded]);
  useEffect(() => {
    if (ready) send('instructions', { text: (playing ? playKeys : previewKeys).map(([key, action]) => `${key}: ${action}`).join(' · ') });
  }, [ready, playing]);

  function stop() { send('play', { playing: false }); setPlaying(false); setStarting(false); pendingPlay.current = false; }
  function reloadEngine() {
    ++revisionRef.current; artifact.current = undefined;
    pendingPlay.current = false; failedDocument.current = undefined;
    setPlaying(false); setStarting(false); setReady(false); setLoaded(false); setError(''); setNotice('Loading the game engine…');
    frame.current?.contentWindow?.location.reload();
  }
  return <section className="engine-preview" aria-label="3D game engine preview" data-revision={revision} data-rendered-pixels={nonblack} data-level-slot={actualSlot} data-exit-zone={actualExit} data-spawn-zone={spawnZone}>
    <div className="engine-toolbar"><span className="eyebrow">3D VIEW</span><span className="engine-status" role="status">{playing ? 'Playtest running' : notice}</span>
      <label className="engine-slot">Slot <select aria-label="Playtest level slot" title="Choose the project's game level slot. Saved with your project and undoable; this controls the game's level-specific setup." value={slot} disabled={playing || starting} onChange={event => onSlotChange(Number(event.target.value))}>{Array.from({ length: 16 }, (_, i) => <option key={i} value={i}>{i + 1}</option>)}</select></label>
      <button title={playing ? 'Stop playing and return to inspection. Your editable map is kept.' : 'Load the latest map into the game and start from player one. Click the game view to capture the mouse.'} disabled={!ready || starting} className={playing ? '' : 'primary-button'} onClick={() => { if (playing) stop(); else load(true); }}>{playing ? 'Stop' : starting ? 'Starting…' : 'Build & play'}</button>
    </div>
    <div className="engine-stage"><iframe ref={frame} title="Alien Breed game engine" src={`${import.meta.env.BASE_URL}runtime/frame.html`} allow="autoplay; fullscreen" onError={() => setError('The game preview could not be loaded. Your map is still available. Reload the preview to try again.')} />
      {!playing && loaded && <span className="engine-crosshair" aria-hidden="true">+</span>}
      {error && <div className="engine-error" role="alert"><Icon name="alert" /><p>{error}</p><button title="Restart the preview engine. Your editable document is kept." onClick={reloadEngine}>Reload engine</button></div>}
    </div>
    <div className="engine-controls">{playing ? <span>Click to capture mouse · WASD move · C crouch · Space use · Esc release mouse · Stop returns to editing</span> : <><button disabled={!loaded} aria-label="Turn preview left" title="Turn the inspection camera left. Left Arrow also turns when the 3D view has focus." onClick={() => send('turn', { delta: -256 })}>←</button><button disabled={!loaded} aria-label="Turn preview right" title="Turn the inspection camera right. Right Arrow also turns when the 3D view has focus." onClick={() => send('turn', { delta: 256 })}>→</button><button disabled={!loaded} aria-label="Look up in preview" title="Look up. Up Arrow also looks up when the 3D view has focus." onClick={() => send('turn', { look: -12 })}>↑</button><button disabled={!loaded} aria-label="Look down in preview" title="Look down. Down Arrow also looks down when the 3D view has focus." onClick={() => send('turn', { look: 12 })}>↓</button><span>Select a zone to visit · Click a visible room to select it ? Arrow keys turn/look ? Enter selects at the crosshair</span></>}
      <details className="engine-key-help"><summary title="Show the keys for the current 3D mode. Enter or Space opens this list.">Controls</summary><div><strong>{playing ? 'Playtest controls' : '3D inspection controls'}</strong>{(playing ? playKeys : previewKeys).map(([key, action]) => <p key={key}><span>{action}</span><kbd>{key}</kbd></p>)}</div></details>
    </div>
  </section>;
}
