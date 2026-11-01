import { useEffect, useRef, useState } from 'react';
import type { LevelDocument } from '../model/level';
import { roomAppearanceChoices } from '../model/roomAppearance';
import { roomSettings } from '../model/roomHeights';
import type { SourceRoom } from '../model/roomSource';
import { createStairFlight, playerStepUpWorld, type StairProposal } from '../model/stairs';
import './StairBuilder.css';

export function StairBuilder({ document, template, selected, cursor, onCancel, onCreate }: {
  document: LevelDocument; template: number; selected?: SourceRoom; cursor: { x: number; z: number };
  onCancel: () => void; onCreate: (proposal: StairProposal) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null), initial = roomSettings(document, selected ?? { recordZone: template });
  const [recordZone, setRecordZone] = useState(template);
  const [draft, setDraft] = useState({ startX: String(Math.round(cursor.x)), startZ: String(Math.round(cursor.z)), endX: '', endZ: '',
    width: '', steps: '', startFloor: String(initial.floor), endFloor: '', clearance: String(initial.floor - initial.ceiling) });
  const choices = roomAppearanceChoices(document, { recordZone }), hasUpper = choices.template.upper.length > 0;
  const [lower, setLower] = useState(''), [upper, setUpper] = useState(''), [light, setLight] = useState('');
  const [proposal, setProposal] = useState<StairProposal>(), [error, setError] = useState('');
  useEffect(() => { const node = dialog.current!; node.showModal(); return () => node.close(); }, []);
  useEffect(() => {
    setProposal(undefined); setError('');
    if (Object.values(draft).some(value => !value.trim()) || !lower || !light || (hasUpper && !upper)) return;
    const timer = window.setTimeout(() => {
      try { setProposal(createStairFlight(document, { start: { x: Number(draft.startX), z: Number(draft.startZ) },
        end: { x: Number(draft.endX), z: Number(draft.endZ) }, width: Number(draft.width), steps: Number(draft.steps),
        startFloor: Number(draft.startFloor), endFloor: Number(draft.endFloor), clearance: Number(draft.clearance), recordZone,
        appearance: { lowerWall: Number(lower), upperWall: hasUpper ? Number(upper) : null, lightingPoint: Number(light) } })); }
      catch (error) { setError(error instanceof Error ? error.message : String(error)); }
    }, 180);
    return () => window.clearTimeout(timer);
  }, [document, draft, recordZone, lower, upper, light, hasUpper]);
  const fields = [
    ['startX', 'Stair start X', 1, -32768, 32767], ['startZ', 'Stair start Z', 1, -32768, 32767],
    ['endX', 'Stair end X', 1, -32768, 32767], ['endZ', 'Stair end Z', 1, -32768, 32767],
    ['width', 'Stair width', 1, 1, 65535], ['steps', 'Number of steps', 1, 1, 256],
    ['startFloor', 'Floor before first step', 0.25, -8192, 8191.75], ['endFloor', 'Final step floor', 0.25, -8192, 8191.75],
    ['clearance', 'Stair clearance', 0.25, 0.25, 8191.75],
  ] as const;
  const points = proposal?.rooms.flatMap(room => [...room.vertices]) ?? [];
  const minX = Math.min(...points.map(p => p.x)), maxX = Math.max(...points.map(p => p.x));
  const minZ = Math.min(...points.map(p => p.z)), maxZ = Math.max(...points.map(p => p.z));
  const margin = Math.max(maxX - minX, maxZ - minZ) / 12;
  return <dialog ref={dialog} className="modal stair-builder" aria-label="Create stairs" onCancel={event => { event.preventDefault(); onCancel(); }}>
    <h2>Build stairs</h2><p>Choose the centre of each end, width and floor heights. Smaller height values are higher. Each room is one step; the first rises from the floor before the flight. Corners snap to whole coordinates and floors to quarters.</p>
    <form onSubmit={event => { event.preventDefault(); if (proposal) onCreate(proposal); }}>
      <label className="select-field"><span>Material template</span><select aria-label="Stair material template" value={recordZone} title="Choose actual level materials. Geometry and height drafts stay as entered; choose the new template's styles again." onChange={event => { setRecordZone(Number(event.target.value)); setLower(''); setUpper(''); setLight(''); }}>
        {document.zones.map(zone => <option key={zone.id} value={zone.id}>Template {zone.id}</option>)}
      </select></label>
      <div className="stair-fields">{fields.map(([field, label, step, min, max]) => <label className="number-field" key={field}><span>{label}</span><input type="number" required aria-label={label} min={min} max={max} step={step} value={draft[field]}
        title={field === 'clearance' ? 'Floor-to-ceiling space follows every step. Verify standing/crouched clearance in the game preview.' : 'Enter the flight dimensions. The live layout validates the whole flight before Create stairs commits it. Escape cancels this dialog.'}
        onChange={event => setDraft(current => ({ ...current, [field]: event.target.value }))} /></label>)}</div>
      <div className="stair-fields">{[{ label: 'Stair lower wall style', records: choices.lower, value: lower, change: setLower },
        ...(hasUpper ? [{ label: 'Stair upper wall style', records: choices.upper, value: upper, change: setUpper }] : [])].map(field => <label className="select-field" key={field.label}><span>{field.label}</span><select required aria-label={field.label} value={field.value} title="Select an authored wall style. Texture, phase and sampling are retained." onChange={event => field.change(event.target.value)}>
          <option value="">Choose a wall style</option>{field.records.map((record, index) => <option key={record.offset} value={record.offset}>Texture {record.texture} · Style {index + 1}</option>)}
        </select></label>)}
        <label className="select-field"><span>Stair corner lighting</span><select required aria-label="Stair corner lighting" value={light} title="Choose an actual template lighting point for the generated corners." onChange={event => setLight(event.target.value)}><option value="">Choose lighting</option>{choices.points.map((id, index) => <option key={id} value={id}>Template point {index + 1}</option>)}</select></label>
      </div>
      {error && <p role="alert">{error}</p>}
      {proposal ? <div className="stair-layout"><svg role="img" aria-label="Proposed stair layout" viewBox={`${minX - margin} ${-maxZ - margin} ${maxX - minX + margin * 2} ${maxZ - minZ + margin * 2}`}>
        {proposal.rooms.map((room, index) => <g key={room.id}><polygon points={room.vertices.map(p => `${p.x},${-p.z}`).join(' ')} />
          {proposal.rooms.length <= 32 && <text x={room.vertices.reduce((sum, p) => sum + p.x, 0) / room.vertices.length} y={-room.vertices.reduce((sum, p) => sum + p.z, 0) / room.vertices.length} fontSize={margin * 0.55}>{index + 1}: {room.heights?.floor ?? choices.template.floor}</text>}
        </g>)}
      </svg><p role="status">{proposal.rooms.length} steps ready · Largest riser {Math.max(...proposal.rises.map(Math.abs))} · Native climb limit below {playerStepUpWorld}. Plan labels show floor heights.</p></div>
        : !error && <p className="field-note">Complete the fields and choose materials to preview the layout.</p>}
      <div className="stair-actions"><button type="button" onClick={onCancel} title="Discard this stair draft. No project geometry changes. Escape also cancels.">Cancel stairs</button>
        <button className="primary-button" type="submit" disabled={!proposal} title="Create this entire validated flight as one undoable edit. Existing rooms, starts and objects are kept.">Create stairs</button></div>
    </form>
  </dialog>;
}
