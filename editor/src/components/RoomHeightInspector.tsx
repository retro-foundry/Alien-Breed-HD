import { RoomStoreyInspector } from './RoomStoreyInspector';
import { NumberField } from './NumberField';
import { setRoomHeadroom } from '../model/roomHeadroom';
import { useEffect, useState } from 'react';
import { editRoomSource, type EditCommand } from '../model/commands';
import type { LevelDocument } from '../model/level';
import { roomHeightFields, roomSettings, type RoomHeights } from '../model/roomHeights';
import { setRoomHeights, type SourceRoom } from '../model/roomSource';
import { RoomHeightSection } from './RoomHeightSection';

const labels = { floor: 'Lower floor height', ceiling: 'Lower ceiling height', upperFloor: 'Upper floor height', upperCeiling: 'Upper ceiling height' };
export function RoomHeightInspector({ document, room, edit, onViewSide }: { document: LevelDocument; room: SourceRoom; onViewSide: () => void; edit: (command: EditCommand | (() => EditCommand)) => void }) {
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
  return <><RoomStoreyInspector key={`${room.id}:${!!room.upperStorey}:${upper}`} document={document} room={room} edit={edit} /><section className="property-section"><h3>Room heights</h3>
    <p className="field-note">A room is the usable space between its floor and ceiling. Increase its height to raise the ceiling; the floor stays put.</p>
    <p className="field-note">{upper
      ? 'This room has two storeys on the same map outline: a lower space and an upper space, each with its own floor and ceiling. Edit their heights below or drag the labelled lines in the side diagram.'
      : 'This room has one storey. To draw a new room with an upper storey, choose a template marked “2 storeys” in the toolbar, then Draw room. The Lower / Upper buttons choose which storey you edit; they do not add one.'}</p>
    <div className="room-headroom-controls">{(upper ? [false, true] : [false]).map(upperRoom => <NumberField key={String(upperRoom)} label={upper ? `${upperRoom ? 'Upper' : 'Lower'} room headroom` : 'Room headroom'} displayLabel={upper ? `${upperRoom ? 'Upper' : 'Lower'} room height` : 'Room height (headroom)'} min={.25} max={16383.75} step={.25} value={settings[upperRoom ? 'upperFloor' : 'floor'] - settings[upperRoom ? 'upperCeiling' : 'ceiling']}
      help="Distance from floor to ceiling. A larger size raises the ceiling; a smaller size lowers it. The floor, other storey and water remain fixed."
      onCommit={height => { try { const change = setRoomHeadroom(document, document.roomSource, room.id, upperRoom, height); setError(''); edit(editRoomSource(change)); } catch (error) { setError(error instanceof Error ? error.message : String(error)); } }} />)}</div>
    {error && <p className="field-note" role="alert">{error}</p>}
    {settings.floor === settings.ceiling && <p className="field-note">This room has no headroom at its starting position. For a door, use its mechanism controls to set the opening.</p>}
    <button className="small-button" onClick={onViewSide} title="Show this room from the front, framed to reveal its floor and ceiling height handles.">View room from side</button>
    <RoomHeightSection key={`${room.id}:${room.recordZone}`} settings={settings} baseline={document.original.data} onCommit={apply} />
    <details><summary>Ceiling openings and sky</summary>
      <p className="field-note">A hole between storeys connects the lower space to the upper space. An opening to the sky exposes the outdoors. These need different geometry; raising a ceiling does not cut a hole.</p>
      <p className="field-note">Use Create opening between storeys above to cut a playable hole, or Ceiling open to sky to remove the top ceiling. Show ceilings only helps you see inside the editor; turning it off does not create an opening in the game.</p>
    </details>
    <form onSubmit={event => { event.preventDefault(); apply(Object.fromEntries(roomHeightFields.map(field => [field, Number(draft[field])])) as Record<keyof RoomHeights, number>); }}>
      <h4>Absolute plane positions</h4><p className="field-note">These are positions, not room sizes. Smaller numbers are higher up. Moving a floor leaves its ceiling in place. Enter or Apply heights commits these fields together.</p>
      <div className="field-grid">{fields.map(field => <label className="number-field" key={field}><span>{labels[field]}</span>
        <input type="number" required aria-label={labels[field]} min={-8192} max={8191.75} step={0.25} value={draft[field]}
          title="Native height in quarter units. Enter applies all heights; Escape restores this field. Smaller values are higher. Floor and ceiling must leave positive clearance."
          onChange={event => { setError(''); setDraft(current => ({ ...current, [field]: event.target.value })); }}
          onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setDraft(current => ({ ...current, [field]: String(settings[field]) })); } }} />
      </label>)}</div>
      <button className="small-button" type="submit" title="Apply the room's floor and ceiling heights as one undoable edit. Collision, surfaces and doorway openings rebuild together; missing doorway styles are reported before export.">Apply heights</button>
    </form>
    <button className="small-button" disabled={!room.heights} title="Remove this room's height overrides and restore its original template heights as one undoable edit." onClick={() => {
      const original = document.zones.find(zone => zone.id === room.recordZone)!;
      const heights = { floor: original.floor, ceiling: original.ceiling,
        upperFloor: room.upperStorey ? original.ceiling : room.upperStorey === false ? settings.upperFloor : original.upperFloor,
        upperCeiling: room.upperStorey ? original.ceiling - (original.floor - original.ceiling) : room.upperStorey === false ? settings.upperCeiling : original.upperCeiling };
      try { setError(''); edit(() => editRoomSource(setRoomHeights(document, document.roomSource, room.id, heights))); }
      catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    }}>Restore template heights</button>
    <label className="select-field"><span>Copy shown heights from</span><select aria-label="Copy heights from room" value={copy} title="Choose a room to load its visible floor/ceiling heights into this form. Apply heights commits them after review." onChange={event => setCopy(event.target.value)}>
      <option value="">Choose a room</option>{document.roomSource.rooms.filter(item => item.id !== room.id).map(item => <option key={item.id} value={item.id}>Room {item.id.replace(/^room-/, '')}</option>)}
    </select></label>
    <button className="small-button" disabled={!copy} title="Load the shown floor/ceiling fields from the selected room without changing the project. Apply heights commits them." onClick={() => {
      const source = document.roomSource.rooms.find(item => item.id === copy); if (!source) return;
      const chosen = roomSettings(document, source); setDraft(current => ({ ...current, ...Object.fromEntries(fields.map(field => [field, String(chosen[field])])) }));
    }}>Load heights</button>
  </section></>;
}
