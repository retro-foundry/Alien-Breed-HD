import { useMemo, useState } from 'react';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import type { EditCommand } from '../model/commands';
import { roomPortalTargets, setRoomPortal, type RoomPortalTarget } from '../model/roomPortals';
import { roomAppearanceChoices, type RoomAppearance } from '../model/roomAppearance';
import { assertStaticRoom } from '../model/roomStoreys';

export function RoomPortalInspector({ document, room, edit, onInspect }: {
  document: LevelDocument; room: SourceRoom; edit: (command: EditCommand) => void; onInspect: (room: string) => void;
}) {
  const [error, setError] = useState(''), [closing, setClosing] = useState<RoomPortalTarget | null>(null);
  const [barrier, setBarrier] = useState(false);
  const [styles, setStyles] = useState<Record<string, { lowerWall: string; upperWall: string; lightingPoint: string }>>({});
  const inspection = useMemo(() => {
    try { return { targets: roomPortalTargets(document, room.id), error: '' }; }
    catch (reason) { return { targets: [], error: reason instanceof Error ? reason.message : String(reason) }; }
  }, [document, room.id]);
  const change = (target: RoomPortalTarget, open: boolean | 'barrier' | null, appearances: readonly RoomAppearance[] = []) => {
    try { edit(setRoomPortal(document, room.id, target.edges[0]!, target.neighbour, open, appearances)); setClosing(null); setError(''); }
    catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  const prepare = (target: RoomPortalTarget, barrier = false) => {
    const drafts = Object.fromEntries([room.id, target.neighbour].map(id => {
      const current = document.roomAppearances?.find(item => item.room === id);
      return [id, { lowerWall: current ? String(current.lowerWall) : '', upperWall: current?.upperWall == null ? '' : String(current.upperWall), lightingPoint: current ? String(current.lightingPoint) : '' }];
    }));
    setStyles(drafts); setClosing(target); setBarrier(barrier); setError('');
  };
  return <section className="property-section"><details><summary>Wall connections (solid / dotted)</summary>
    <p className="field-note">Open or close an existing shared wall on both sides. Dotted means connected; solid means closed. Floor heights and headroom still determine whether you can walk through.</p>
    {(error || inspection.error) && <p className="field-note" role="alert">{error || inspection.error}</p>}
    {!inspection.error && !inspection.targets.length && <p className="field-note">This room has no adjoining room along an edge. Draw a room beside it with an exactly shared edge first.</p>}
    {inspection.targets.map(target => {
      const other = document.roomSource.rooms.find(room => room.id === target.neighbour)!;
      const name = `Room ${other.id.replace(/^room-/, '')}`, edgeLabel = target.edges.map(edge => edge + 1).join(', ');
      let blocked = '';
      try { assertStaticRoom(document, room); assertStaticRoom(document, other); } catch (reason) { blocked = reason instanceof Error ? reason.message : String(reason); }
      return <div className="portal-connection-row" key={`${target.neighbour}:${edgeLabel}`}>
        <p>Edge {edgeLabel} to {name}: <strong>{target.barrier ? 'See-through barrier (blocks walking)' : target.open === null ? 'Mixed' : target.open ? 'Dotted connection' : 'Solid wall'}</strong></p>
        <button type="button" className="small-button" disabled={!!blocked || target.open === true && !target.barrier} onClick={() => change(target, true)}>Open edge {edgeLabel} to {name}</button>
        <button type="button" className="small-button" disabled={!!blocked || target.open === false} onClick={() => prepare(target)}>Close edge {edgeLabel} to {name}</button>
        <button type="button" className="small-button" disabled={!!blocked || target.barrier} onClick={() => prepare(target, true)}>See-through barrier on edge {edgeLabel} to {name}</button>
        <button type="button" className="small-button" onClick={() => onInspect(other.id)}>Show {name}</button>
        {target.authored && <button type="button" className="small-button" disabled={!!blocked} onClick={() => change(target, null)}>Use automatic edge {edgeLabel} to {name}</button>}
        {blocked && <p className="field-note">{blocked}</p>}
      </div>;
    })}
    {closing && <form onSubmit={event => {
      event.preventDefault();
      change(closing, barrier ? 'barrier' : false, [room.id, closing.neighbour].map(id => {
        const target = document.roomSource.rooms.find(room => room.id === id)!, choices = roomAppearanceChoices(document, target), draft = styles[id]!;
        return { room: id, lowerWall: Number(draft.lowerWall), upperWall: choices.template.upper.length ? Number(draft.upperWall) : null, lightingPoint: Number(draft.lightingPoint) };
      }));
    }}>
      <h4>{barrier ? 'See-through barrier' : 'Close shared wall'}</h4><p className="field-note">{barrier ? 'Choose the material and lighting on both sides. Transparent texels reveal the adjoining room, while the whole edge blocks walking. Apply creates both sides as one Undo. Choose Open edge afterwards to remove the barrier.' : 'Choose the wall appearance and lighting on each side. Apply closes collision and fills the wall together as one Undo. Cancel leaves the passage open.'}</p>
      {[room.id, closing.neighbour].map(id => {
        const target = document.roomSource.rooms.find(room => room.id === id)!, choices = roomAppearanceChoices(document, target), draft = styles[id]!;
        const name = `Room ${id.replace(/^room-/, '')}`;
        const field = (key: keyof typeof draft, value: string) => setStyles(current => ({ ...current, [id]: { ...current[id]!, [key]: value } }));
        return <fieldset className="portal-repair" key={id}><legend>{name}</legend>
          <label className="select-field"><span>Lower wall style</span><select required aria-label={`Wall style for ${name}`} value={draft.lowerWall} onChange={event => field('lowerWall', event.target.value)}><option value="">Choose a wall style</option>{choices.lower.map((record, index) => <option key={record.offset} value={record.offset}>Texture {record.texture}, style {index + 1}</option>)}</select></label>
          {!!choices.template.upper.length && <label className="select-field"><span>Upper wall style</span><select required aria-label={`Upper wall style for ${name}`} value={draft.upperWall} onChange={event => field('upperWall', event.target.value)}><option value="">Choose an upper style</option>{choices.upper.map((record, index) => <option key={record.offset} value={record.offset}>Texture {record.texture}, style {index + 1}</option>)}</select></label>}
          <label className="select-field"><span>Corner lighting</span><select required aria-label={`Corner lighting for ${name}`} value={draft.lightingPoint} onChange={event => field('lightingPoint', event.target.value)}><option value="">Choose lighting</option>{choices.points.map((point, index) => <option key={point} value={point}>Room point {index + 1}</option>)}</select></label>
        </fieldset>;
      })}
      <button type="button" className="small-button" onClick={() => setClosing(null)}>{barrier ? 'Cancel barrier' : 'Cancel wall closure'}</button><button type="submit" className="small-button">{barrier ? 'Apply see-through barrier' : 'Apply wall closure'}</button>
    </form>}
  </details></section>;
}
