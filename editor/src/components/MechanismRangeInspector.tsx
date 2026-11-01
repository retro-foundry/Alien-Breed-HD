import { useEffect, useMemo, useState } from 'react';
import type { LevelDocument, Mechanism } from '../model/level';
import type { EditCommand } from '../model/commands';
import { setMechanismRange, mechanismNeedsEndpoint } from '../model/mechanismRanges';

export function MechanismRangeInspector({ document, mechanism, edit }: { document: LevelDocument; mechanism: Mechanism; edit: (command: EditCommand | (() => EditCommand)) => void }) {
  return <MovementRangeForm mechanism={mechanism} prefix={`${mechanism.kind} ${mechanism.id}`}
    createCommand={(top, bottom) => setMechanismRange(document, mechanism.kind, mechanism.id, top, bottom)} edit={edit} />;
}

export function MovementRangeForm({ mechanism, prefix, createCommand, edit }: {
  mechanism: Pick<Mechanism, 'kind' | 'id' | 'top' | 'bottom' | 'current' | 'direction' | 'conditions'>;
  prefix: string; createCommand: (top: number, bottom: number) => EditCommand;
  edit: (command: EditCommand | (() => EditCommand)) => void;
}) {
  const [top, setTop] = useState(String(mechanism.top)), [bottom, setBottom] = useState(String(mechanism.bottom));
  useEffect(() => { setTop(String(mechanism.top)); setBottom(String(mechanism.bottom)); }, [prefix, mechanism.kind, mechanism.id, mechanism.top, mechanism.bottom]);
  const result = useMemo(() => {
    if (!top.trim() || !bottom.trim()) return {};
    try { return { command: createCommand(Number(top), Number(bottom)) }; }
    catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
  }, [createCommand, top, bottom]);
  return <form aria-label={`${prefix} movement range`} onSubmit={event => { event.preventDefault(); if (result.command) edit(result.command); }}>
    <p className="field-note">Movement destinations use quarter units. Smaller values are higher. Initial position: {mechanism.current}.{mechanismNeedsEndpoint(mechanism) ? ' Keep one endpoint here so trigger rules can start the stationary mechanism.' : ''}</p>
    {(['top', 'bottom'] as const).map(field => <label className="number-field" key={field}><span>{field === 'top' ? mechanism.kind === 'door' ? 'Open height' : 'Top height' : mechanism.kind === 'door' ? 'Closed height' : 'Bottom height'}</span>
      <input type="number" required min={-8192} max={8191.75} step={0.25} aria-label={`${prefix} ${field} destination`} value={field === 'top' ? top : bottom}
        title="Set a movement destination in world units, in quarter-unit steps. The range must contain the fixed initial position; any additional activation constraint is explained above. Apply range commits both together; Escape restores this field."
        onChange={event => (field === 'top' ? setTop : setBottom)(event.target.value)} onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); (field === 'top' ? setTop : setBottom)(String(mechanism[field])); } }} />
    </label>)}
    {result.error && <p role="alert" className="field-note">{result.error}</p>}
    <button className="small-button" type="submit" disabled={!result.command || Number(top) === mechanism.top && Number(bottom) === mechanism.bottom}
      title="Apply both movement destinations as one undoable edit. The fixed initial position, speeds, trigger rules and animation attachments are retained.">Apply range</button>
    <button className="small-button" type="button" title="Discard both uncommitted destination fields." onClick={() => { setTop(String(mechanism.top)); setBottom(String(mechanism.bottom)); }}>Reset range draft</button>
  </form>;
}
