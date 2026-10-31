import { useEffect, useState } from 'react';
import type { SourceRoom } from '../model/roomSource';

/** The room picker is shared by gameplay placements; binary zone IDs and
 * source-only room identities must never be mixed in a numeric field. */
export function RoomPositionForm({ rooms, identity, room, x, z, label, step = 1, onApply }: {
  rooms: readonly SourceRoom[]; identity: string; room: string; x: number; z: number; label: string; step?: number | 'any';
  onApply: (room: string, x: number, z: number) => void;
}) {
  const [draft, setDraft] = useState({ x, z, room });
  useEffect(() => setDraft({ x, z, room }), [x, z, room, identity]);
  return <form className="position-form" onSubmit={event => { event.preventDefault(); onApply(draft.room, draft.x, draft.z); }}>
    <label className="number-field"><span>Room</span><select aria-label={label} title="Choose the editable room containing this position. Coordinates must be inside its outline. Apply position or press Enter to save." value={draft.room} onChange={event => setDraft({ ...draft, room: event.target.value })}>
      {!rooms.some(item => item.id === draft.room) && <option value={draft.room}>Choose an existing room</option>}
      {rooms.map(item => <option key={item.id} value={item.id}>{item.originZone === null ? `New room ${item.id.replace(/^room-/, '')}` : `Zone ${item.originZone}`}</option>)}
    </select></label>
    <div className="field-grid">{(['x', 'z'] as const).map(axis => <label className="number-field" key={axis}><span>{axis.toUpperCase()}</span><input required type="number" min={-32768} max={step === 1 ? 32767 : 32768 - 1 / 65536} step={step} aria-label={`Position ${axis.toUpperCase()}`} title={`${axis.toUpperCase()} coordinate inside the selected room. Apply position or press Enter to save.`} value={draft[axis]} onChange={event => setDraft({ ...draft, [axis]: Number(event.target.value) })} /></label>)}</div>
    <button type="submit" className="small-button" title="Validate the room and coordinates, then save the position. Undo restores its previous position." disabled={draft.x === x && draft.z === z && draft.room === room}>Apply position</button>
  </form>;
}
