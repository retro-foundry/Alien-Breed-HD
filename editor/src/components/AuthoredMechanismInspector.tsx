import { movingRoomOwnership } from '../model/movingRoomEligibility';
import { useEffect, useState } from 'react';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import { roomSettings } from '../model/roomHeights';
import { roomAppearanceChoices } from '../model/roomAppearance';
import { sourceMovingSettings, movingRoomFromDraft, movingWallChoices, setSourceMovingRoom } from '../model/sourceMovingRoom';
import { editRoomSource, type EditCommand } from '../model/commands';
import { buildPlayableLevel } from '../model/playableBuild';
import { MechanismRulesInspector } from './MechanismRulesInspector';
import { WallStyleBrowser } from './WallStyleBrowser';
import { CopyMechanismSettings, MoveAuthoredMechanism } from './MechanismTransfers';

export function AuthoredMechanismInspector({ document, room, edit, onShowRoom }: { document: LevelDocument; room: SourceRoom; edit: (command: EditCommand | (() => EditCommand)) => void; onShowRoom: (id: string) => void }) {
  const existing = room.moving && sourceMovingSettings(room.moving), settings = roomSettings(document, room);
  const initial = (kind: 'door' | 'lift' = existing?.kind ?? 'door') => {
    const saved = existing?.kind === kind ? existing : undefined;
    return { kind, top: String(saved?.top ?? (kind === 'door' ? settings.ceiling : settings.floor)),
      bottom: String(saved?.bottom ?? settings.floor), start: saved ? saved.current === saved.top ? 'top' : saved.current === saved.bottom ? 'bottom' : 'saved' : 'bottom',
      wall: String(room.moving?.wall ?? document.roomAppearances?.find(item => item.room === room.id)?.lowerWall ?? (room.originZone !== null ? roomAppearanceChoices(document, room).lower[0]?.offset : undefined) ?? ''),
      conditions: saved?.conditions ?? 0, openRule: saved?.openRule ?? 0, closeRule: saved?.closeRule ?? 0 };
  };
  const [active, setActive] = useState<'door' | 'lift' | undefined>(existing?.kind), [draft, setDraft] = useState(initial), [error, setError] = useState('');
  useEffect(() => { setActive(existing?.kind); setDraft(initial()); setError(''); }, [document, room.id]);
  const conflict = movingRoomOwnership(document, room);
  if (conflict && !room.moving) return <section className="property-section"><h3>Door or lift</h3><p className="field-note">{conflict.reason}</p>{conflict.room && conflict.room !== room.id && <button type="button" title="Select and frame the room with the existing mechanism." onClick={() => onShowRoom(conflict.room!)}>Go to existing {conflict.kind}</button>}</section>;
  const reset = () => { setDraft(initial()); setActive(existing?.kind); setError(''); };
  const begin = (kind: 'door' | 'lift') => { setDraft(initial(kind)); setActive(kind); setError(''); };
  const apply = () => {
    try {
      if (!draft.top.trim() || !draft.bottom.trim() || !draft.wall) throw new Error('Choose both destinations and a moving wall material.');
      const top = Number(draft.top), bottom = Number(draft.bottom);
      if (top === bottom) throw new Error('Choose different upper and lower destinations so the mechanism can move.');
      const value = movingRoomFromDraft({ ...draft, top, bottom, wall: Number(draft.wall),
        current: draft.start === 'top' ? top : draft.start === 'bottom' ? bottom : existing!.current });
      const change = setSourceMovingRoom(document, room.id, value);
      buildPlayableLevel({ ...document, roomSource: change.after }, -1);
      edit(editRoomSource(change)); setError('');
    } catch (error) { setError(error instanceof Error ? error.message : String(error)); }
  };
  return <section className="property-section" aria-label="Create door or lift"><h3>{existing ? `Authored ${existing.kind}` : 'Door or lift'}</h3>
    <p className="field-note">Use a connected room with no inward corners. Doors move its ceiling; lifts move its floor. The editor builds their wall and collision links automatically.</p>
    {!active ? <><button type="button" onClick={() => begin('door')} title="Configure an independent door in this room. Starts closed by default; nothing changes until Apply mechanism.">Make door</button>
      <button type="button" onClick={() => begin('lift')} title="Configure an independent lift. Choose its two destinations, then Apply mechanism.">Make lift</button></> :
      <form aria-label="Authored mechanism settings" onSubmit={event => { event.preventDefault(); apply(); }} onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); reset(); } }}>
        <label className="select-field"><span>Mechanism</span><select aria-label="New mechanism kind" value={draft.kind} title="Choose which room plane moves. Choose destinations again when changing kind." onChange={event => begin(event.target.value as 'door' | 'lift')}>
          <option value="door">Door</option><option value="lift">Lift</option></select></label>
        {(['top', 'bottom'] as const).map(field => <label className="number-field" key={field}><span>{field === 'top' ? draft.kind === 'door' ? 'Open height' : 'Upper destination' : draft.kind === 'door' ? 'Closed height' : 'Lower destination'}</span>
          <input required type="number" step={0.25} min={-8192} max={8191.75} aria-label={`New mechanism ${field}`} value={draft[field]} title="World height in quarter-unit steps. Smaller values are higher. Apply validates the complete range and adjoining rooms."
            onChange={event => setDraft({ ...draft, [field]: event.target.value })} /></label>)}
        <label className="select-field"><span>Starts</span><select aria-label="New mechanism initial position" value={draft.start} title="Choose its initial endpoint without changing the editable room's open clearance." onChange={event => setDraft({ ...draft, start: event.target.value })}>
          <option value="top">{draft.kind === 'door' ? 'Open' : 'At upper destination'}</option><option value="bottom">{draft.kind === 'door' ? 'Closed' : 'At lower destination'}</option>{existing?.kind === draft.kind && <option value="saved">Keep saved height ({existing.current})</option>}</select></label>
        <MechanismRulesInspector mechanism={draft} prefix="New mechanism" change={(conditions, openRule, closeRule) => setDraft({ ...draft, conditions, openRule, closeRule })} />
        <WallStyleBrowser records={movingWallChoices(document, room)} value={draft.wall} onChange={wall => setDraft({ ...draft, wall })} label="moving wall" overrides={document.wallOverrides} />
        <p className="field-note">Destinations use world units; smaller numbers are higher. Space uses doors and lifts in Build &amp; play. Choose trigger rules above. Apply is one Undo; Escape or Cancel discards the draft.</p>
        {error && <p role="alert">{error}</p>}
        <button type="button" onClick={reset} title="Discard uncommitted settings. Escape also cancels.">Cancel mechanism draft</button>
        <button type="submit" className="small-button" title="Validate geometry, create movement links and commit all settings as one undoable edit.">Apply mechanism</button>
      </form>}
    {existing && <button type="button" className="danger-button" title="Return this room to a static volume with its editable floor/ceiling heights. Undo restores the controller." onClick={() => edit(() => editRoomSource(setSourceMovingRoom(document, room.id, null)))}>Remove mechanism</button>}
    {existing && <><CopyMechanismSettings document={document} owner={{ type: 'authored', kind: existing.kind, room: room.id }} edit={edit} /><MoveAuthoredMechanism document={document} room={room} edit={edit} onShowRoom={onShowRoom} /></>}
  </section>;
}
