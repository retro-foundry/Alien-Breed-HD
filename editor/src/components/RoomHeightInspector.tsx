import { useEffect, useState } from 'react';
import { editRoomSource, type EditCommand } from '../model/commands';
import type { LevelDocument } from '../model/level';
import { roomHeightFields, roomSettings, type RoomHeights } from '../model/roomHeights';
import { setRoomHeights, type SourceRoom } from '../model/roomSource';
import { RoomHeightSection } from './RoomHeightSection';

const labels = { floor: 'Lower floor height', ceiling: 'Lower ceiling height', upperFloor: 'Upper floor height', upperCeiling: 'Upper ceiling height' };
export function RoomHeightInspector({ document, room, edit }: { document: LevelDocument; room: SourceRoom; edit: (command: EditCommand | (() => EditCommand)) => void }) {
  const settings = roomSettings(document, room), upper = settings.upper.length > 0 && settings.upperFloor > settings.upperCeiling;
  const values = () => Object.fromEntries(roomHeightFields.map(field => [field, String(settings[field])])) as Record<keyof RoomHeights, string>;
  const [draft, setDraft] = useState(values), [copy, setCopy] = useState(''), [error, setError] = useState('');
  useEffect(() => { setDraft(values()); setCopy(''); setError(''); }, [room.id, room.heights, document.original.data,
    settings.floor, settings.ceiling, settings.upperFloor, settings.upperCeiling]);
  const fields = upper ? roomHeightFields : roomHeightFields.slice(0, 2);
  const apply = (heights: RoomHeights) => {
    try {
      const command = editRoomSource(setRoomHeights(document, document.roomSource, room.id, heights));
      setError(''); edit(command);
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
  };
  return <section className="property-section"><h3>Room heights</h3>
    <p className="field-note">Smaller height values are higher. Lower clearance: {settings.floor - settings.ceiling}. Apply all fields together; preview rebuilds automatically. Water keeps its own height.</p>
    <RoomHeightSection key={`${room.id}:${room.recordZone}`} settings={settings} baseline={document.original.data} onCommit={apply} />
    <form onSubmit={event => { event.preventDefault(); apply(Object.fromEntries(roomHeightFields.map(field => [field, Number(draft[field])])) as Record<keyof RoomHeights, number>); }}>
      <div className="field-grid">{fields.map(field => <label className="number-field" key={field}><span>{labels[field]}</span>
        <input type="number" required aria-label={labels[field]} min={-8192} max={8191.75} step={0.25} value={draft[field]}
          title="Native height in quarter units. Enter applies all heights; Escape restores this field. Smaller values are higher. Floor and ceiling must leave positive clearance."
          onChange={event => { setError(''); setDraft(current => ({ ...current, [field]: event.target.value })); }}
          onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setDraft(current => ({ ...current, [field]: String(settings[field]) })); } }} />
      </label>)}</div>
      <button className="small-button" type="submit" title="Apply the room's floor and ceiling heights as one undoable edit. Collision, surfaces and doorway openings rebuild together; missing doorway styles are reported before export.">Apply heights</button>
      {error && <p className="field-note" role="alert">{error}</p>}
    </form>
    <button className="small-button" disabled={!room.heights} title="Remove this room's height overrides and restore its original template heights as one undoable edit." onClick={() => {
      const original = document.zones.find(zone => zone.id === room.recordZone)!;
      setError(''); edit(() => editRoomSource(setRoomHeights(document, document.roomSource, room.id, Object.fromEntries(roomHeightFields.map(field => [field, original[field]])))));
    }}>Restore template heights</button>
    <label className="select-field"><span>Copy shown heights from</span><select aria-label="Copy heights from room" value={copy} title="Choose a room to load its visible floor/ceiling heights into this form. Apply heights commits them after review." onChange={event => setCopy(event.target.value)}>
      <option value="">Choose a room</option>{document.roomSource.rooms.filter(item => item.id !== room.id).map(item => <option key={item.id} value={item.id}>Room {item.id.replace(/^room-/, '')}</option>)}
    </select></label>
    <button className="small-button" disabled={!copy} title="Load the shown floor/ceiling fields from the selected room without changing the project. Apply heights commits them." onClick={() => {
      const source = document.roomSource.rooms.find(item => item.id === copy); if (!source) return;
      const chosen = roomSettings(document, source); setDraft(current => ({ ...current, ...Object.fromEntries(fields.map(field => [field, String(chosen[field])])) }));
    }}>Load heights</button>
  </section>;
}
