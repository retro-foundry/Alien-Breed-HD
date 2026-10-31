import { useEffect, useRef, useState } from 'react';
import { type LevelDocument } from '../model/level';
import { type Selection } from './MapCanvas';
import { Icon } from './Icon';
import { previewKeys, playKeys } from './ControlsHelp';

interface EngineMessage { source: 'ab3d-engine'; type: string; revision?: number; zone?: number; zones?: number; message?: string; nonblack?: number; slot?: number }
interface Props { document: LevelDocument; selection: Selection; layer: 'lower' | 'upper'; onSelectZone: (zone: number) => void }

export function EnginePreview({ document, selection, layer, onSelectZone }: Props) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [ready, setReady] = useState(false), [loaded, setLoaded] = useState(false);
  const [playing, setPlaying] = useState(false), [starting, setStarting] = useState(false);
  const [error, setError] = useState(''), [notice, setNotice] = useState('Loading the game engine…');
  const [slot, setSlot] = useState(() => Math.max(0, Math.min(15, Number(/^Level (\d+)$/.exec(document.name)?.[1] ?? 1) - 1)));
  const [revision, setRevision] = useState(0);
  const [nonblack, setNonblack] = useState(0);
  const [actualSlot, setActualSlot] = useState(-1);
  const revisionRef = useRef(0), pendingPlay = useRef(false);
  const current = useRef({ document, selection, layer, onSelectZone });
  current.current = { document, selection, layer, onSelectZone };
  const send = (type: string, detail: object = {}) => frame.current?.contentWindow?.postMessage({ source: 'ab3d-editor', type, ...detail }, location.origin);

  function positionCamera() {
    const { document, selection, layer } = current.current;
    const object = selection?.kind === 'object' ? document.objects.find(object => object.id === selection.id) : undefined;
    const start = selection?.kind === 'start' ? document.starts.find(start => start.player === selection.id) : undefined;
    const zoneId = selection?.kind === 'zone' ? selection.id : object?.zone ?? start?.zone;
    const zone = document.zones.find(zone => zone.id === zoneId);
    if (!zone) return;
    const lines = zone.boundaryIds.map(id => document.lines[id]).filter(line => !!line);
    if (lines.length === 0) return;
    const x = Math.round(object?.x ?? start?.x ?? lines.reduce((sum, line) => sum + line.x, 0) / lines.length);
    const z = Math.round(object?.z ?? start?.z ?? lines.reduce((sum, line) => sum + line.z, 0) / lines.length);
    send('camera', { camera: { zone: zone.id, x, z, upper: layer === 'upper' && zone.upper.length > 0 ? 1 : 0, angle: 0 } });
  }

  function load(play = false) {
    if (!ready) return;
    pendingPlay.current = play;
    setStarting(play); setLoaded(false); setError(''); setNotice('Loading current level…');
    const next = ++revisionRef.current; setRevision(next);
    send('load', { slot, revision: next, files: current.current.document.files });
  }

  useEffect(() => {
    const receive = (event: MessageEvent<EngineMessage>) => {
      if (event.origin !== location.origin || event.source !== frame.current?.contentWindow || event.data?.source !== 'ab3d-engine') return;
      const message = event.data;
      if (message.type === 'ready') { setReady(true); setError(''); }
      if (message.type === 'loaded' && message.revision === revisionRef.current) {
        setNonblack(message.nonblack ?? 0);
        setActualSlot(message.slot ?? -1);
        setLoaded(true); setNotice(`${message.zones} zones loaded · Game engine`);
        if (pendingPlay.current) { pendingPlay.current = false; setPlaying(true); setStarting(false); send('play', { playing: true }); }
        else positionCamera();
      }
      if (message.type === 'error') { setError(message.message ?? 'The engine preview stopped.'); setStarting(false); }
      if (message.type === 'stopped') { setPlaying(false); setStarting(false); setNotice('Playtest ended. Build and play again to restart.'); }
      if (message.type === 'pick' && message.zone !== undefined && message.zone >= 0) current.current.onSelectZone(message.zone);
    };
    window.addEventListener('message', receive);
    return () => window.removeEventListener('message', receive);
  }, []);

  useEffect(() => {
    if (!ready || playing || starting) return;
    const timer = window.setTimeout(() => load(), 180);
    return () => window.clearTimeout(timer);
  }, [ready, document, slot, playing]);

  useEffect(() => { if (loaded && !playing) positionCamera(); }, [selection, layer, loaded]);
  useEffect(() => {
    if (ready) send('instructions', { text: (playing ? playKeys : previewKeys).map(([key, action]) => `${key}: ${action}`).join(' · ') });
  }, [ready, playing]);

  function stop() { send('play', { playing: false }); setPlaying(false); setStarting(false); pendingPlay.current = false; }
  return <section className="engine-preview" aria-label="3D game engine preview" data-revision={revision} data-rendered-pixels={nonblack} data-level-slot={actualSlot}>
    <div className="engine-toolbar"><span className="eyebrow">3D VIEW</span><span className="engine-status" role="status">{playing ? 'Playtest running' : notice}</span>
      <label className="engine-slot">Slot <select aria-label="Playtest level slot" title="Choose which game level slot to use for this playtest. This controls the game's level-specific setup." value={slot} disabled={playing || starting} onChange={event => setSlot(Number(event.target.value))}>{Array.from({ length: 16 }, (_, i) => <option key={i} value={i}>{i + 1}</option>)}</select></label>
      <button title={playing ? 'Stop playing and return to inspection. Your editable map is kept.' : 'Load the latest map into the game and start from player one. Click the game view to capture the mouse.'} disabled={!ready || starting} className={playing ? '' : 'primary-button'} onClick={() => { if (playing) stop(); else load(true); }}>{playing ? 'Stop' : starting ? 'Starting…' : 'Build & play'}</button>
    </div>
    <div className="engine-stage"><iframe ref={frame} title="Alien Breed game engine" src={`${import.meta.env.BASE_URL}runtime/frame.html`} allow="autoplay; fullscreen" onError={() => setError('The game preview could not be loaded. Your map is still available. Reload the preview to try again.')} />
      {!playing && loaded && <span className="engine-crosshair" aria-hidden="true">+</span>}
      {error && <div className="engine-error" role="alert"><Icon name="alert" /><p>{error}</p><button title="Restart the preview engine. Your editable document is kept." onClick={() => { setReady(false); setLoaded(false); setError(''); setNotice('Loading the game engine…'); frame.current?.contentWindow?.location.reload(); }}>Reload engine</button></div>}
    </div>
    <div className="engine-controls">{playing ? <span>Click to capture mouse · WASD move · C crouch · Space use · Esc release mouse · Stop returns to editing</span> : <><button disabled={!loaded} aria-label="Turn preview left" title="Turn the inspection camera left. Left Arrow also turns when the 3D view has focus." onClick={() => send('turn', { delta: -256 })}>←</button><button disabled={!loaded} aria-label="Turn preview right" title="Turn the inspection camera right. Right Arrow also turns when the 3D view has focus." onClick={() => send('turn', { delta: 256 })}>→</button><button disabled={!loaded} aria-label="Look up in preview" title="Look up. Up Arrow also looks up when the 3D view has focus." onClick={() => send('turn', { look: -12 })}>↑</button><button disabled={!loaded} aria-label="Look down in preview" title="Look down. Down Arrow also looks down when the 3D view has focus." onClick={() => send('turn', { look: 12 })}>↓</button><span>Select a zone to visit · Click the view to select under the crosshair and focus arrow keys</span></>}
      <details className="engine-key-help"><summary title="Show the keys for the current 3D mode. Enter or Space opens this list.">Controls</summary><div><strong>{playing ? 'Playtest controls' : '3D inspection controls'}</strong>{(playing ? playKeys : previewKeys).map(([key, action]) => <p key={key}><span>{action}</span><kbd>{key}</kbd></p>)}</div></details>
    </div>
  </section>;
}
