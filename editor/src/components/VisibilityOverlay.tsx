import { useEffect, useMemo, useState } from 'react';
import type { LevelDocument } from '../model/level';
import { preparePlayableLevel, type PlayableLevel } from '../model/playableBuild';
import { visibilityInspection } from '../model/visibilityInspection';
import './VisibilityOverlay.css';

export type VisibilityInspection = ReturnType<typeof visibilityInspection>;
export function useVisibilityInspection(document: LevelDocument, room: string | undefined, zone: number | undefined) {
  const [enabled, setEnabled] = useState(false), [retry, setRetry] = useState(0), [cell, setCell] = useState<number>();
  const [target, setTarget] = useState<number>();
  const [result, setResult] = useState<{ source: LevelDocument; artifact?: PlayableLevel; error?: string }>();
  useEffect(() => {
    if (!enabled) return;
    let active = true;
    setResult(undefined);
    void preparePlayableLevel(document).then(artifact => { if (active) setResult({ source: document, artifact }); },
      error => { if (active) setResult({ source: document, error: error instanceof Error ? error.message : String(error) }); });
    return () => { active = false; };
  }, [enabled, document, retry]);
  const current = enabled && result?.source === document ? result : undefined;
  const artifact = current?.artifact;
  const inspected = useMemo(() => {
    try { return { inspection: artifact ? visibilityInspection(artifact, room, zone, cell, target) : undefined }; }
    catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
  }, [artifact, room, zone, cell, target]);
  return { enabled, toggle: () => setEnabled(value => !value), retry: () => setRetry(value => value + 1),
    setCell, setTarget, inspection: inspected.inspection, error: current?.error ?? inspected.error, loading: enabled && !current };
}

export function VisibilityOverlay({ state, onFrame }: { state: ReturnType<typeof useVisibilityInspection>; onFrame: () => void }) {
  const [controlsOpen, setControlsOpen] = useState(true);
  return <section className="visibility-overlay" aria-label="Visibility inspection">
    <button aria-pressed={state.enabled} title="Inspect the exact visibility tables used by preview and export. Cyan marks potential visibility, amber the selected runtime cell, green its portals. These are conservative candidates, not exact line of sight."
      onClick={state.toggle}>Visibility overlay</button>
    {state.enabled && <button aria-expanded={controlsOpen} aria-controls="visibility-controls" title="Hide or show the inspection controls while keeping the visibility colours, portals and clipping points drawn on the map."
      onClick={() => setControlsOpen(open => !open)}>{controlsOpen ? 'Hide visibility controls' : 'Show visibility controls'}</button>}
    {state.enabled && <div id="visibility-controls" className="visibility-content" hidden={!controlsOpen}>
      {state.loading ? <p role="status">Building current visibility…</p> : state.error ? <><p role="alert">{state.error}</p><button onClick={state.retry} title="Retry compiling visibility from the current source. No source edits are made.">Retry visibility</button></>
        : !state.inspection?.cells.length ? <p role="status">Select a room to inspect its visibility.</p> : <>
          <label>Runtime cell <select aria-label="Visibility runtime cell" title="Concave source rooms may compile to several convex cells. Inspect each cell's own visibility table." value={state.inspection.selected} onChange={event => state.setCell(Number(event.target.value))}>
            {state.inspection.cells.map(id => <option key={id} value={id}>Zone {id}</option>)}
          </select></label>
          <p role="status">{state.inspection.rows.length} visibility entries · {state.inspection.portals.length} portals</p>
          <p>Amber: selected cell · Cyan: potential visibility · Green: portals</p>
          <p>Conservative candidates, not exact line of sight. Both floors share this table.</p>
          <button onClick={onFrame} title="Fit the selected runtime cell and all its potentially visible cells into the map. Changes only the view, not source or history.">Frame visibility</button>
          {state.inspection.rows.some(row => row.bounds) && <><label>Clip target <select aria-label="Visibility clip target" value={state.inspection.target} title="Choose a target cell to show the actual left and right clip boundary points. Blue marks left points; pink marks right points. These points constrain runtime clipping, not a fixed viewing cone." onChange={event => state.setTarget(Number(event.target.value))}>
            {state.inspection.rows.filter(row => row.bounds).map((row, index) => <option key={index} value={row.zone}>Zone {row.zone}</option>)}
          </select></label><p>Blue: left clip points · Pink: right clip points</p></>}
          <details><summary title="Show the target zone, saved clipping flag, loader-assigned word offset and ordering mask for each visibility entry.">Runtime visibility table</summary>
            <table><thead><tr><th>Zone</th><th title="Saved reference; nonnegative values are assignment flags, not offsets.">Flag</th><th title="Actual word offset assigned by the native loader walking the clip stream.">Offset</th><th>Mask</th></tr></thead><tbody>{state.inspection.rows.map((row, index) => <tr key={index}><td>{row.zone}</td><td>{row.clip}</td><td>{row.bounds?.wordOffset ?? '—'}</td><td>{`0x${(row.mask >>> 0).toString(16)}`}</td></tr>)}</tbody></table>
          </details>
        </>}
    </div>}
  </section>;
}
