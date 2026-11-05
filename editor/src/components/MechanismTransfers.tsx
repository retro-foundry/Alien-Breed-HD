import { useEffect, useState } from 'react';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import { applyCommand, editRoomSource, type EditCommand } from '../model/commands';
import { buildPlayableLevel } from '../model/playableBuild';
import { captureMechanismSettings, mechanismOwnerKey, mechanismSettingsOwners, transferMechanismSettings, type MechanismOwner } from '../model/mechanismSettingsTransfer';
import { moveSourceMovingRoom } from '../model/sourceMovingRoom';

type Edit = (command: EditCommand | (() => EditCommand)) => void;
export function CopyMechanismSettings({ document, owner, edit }: { document: LevelDocument; owner: MechanismOwner; edit: Edit }) {
  const [source, setSource] = useState(''), [range, setRange] = useState(false), [rules, setRules] = useState(true), [requirements, setRequirements] = useState(false), [error, setError] = useState('');
  const key = mechanismOwnerKey(owner), options = mechanismSettingsOwners(document, owner.kind).filter(item => mechanismOwnerKey(item.owner) !== key);
  const selected = options.find(item => mechanismOwnerKey(item.owner) === source), settings = selected && captureMechanismSettings(document, selected.owner);
  useEffect(() => { setSource(''); setError(''); }, [document, key]);
  const reset = () => { setSource(''); setRange(false); setRules(true); setRequirements(false); setError(''); };
  const copy = () => {
    try {
      if (!settings) throw new Error('Choose a source mechanism.');
      const command = transferMechanismSettings(document, owner, settings, { range, rules, requirements });
      buildPlayableLevel(applyCommand(document, command), -1);
      edit(command); setError('');
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  return <div role="group" aria-label={`Copy ${owner.kind} settings`}><details onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); reset(); } }}>
    <summary>Copy settings from another {owner.kind}</summary>
    <p className="field-note">Choose a source and which settings to copy. The target keeps its starting height and wall/room attachments. Apply validates the whole level and commits one Undo.</p>
    <label className="select-field"><span>Copy settings from</span><select aria-label="Mechanism settings source" value={source} onChange={event => setSource(event.target.value)}>
      <option value="">Choose a {owner.kind}</option>{options.map(item => <option key={mechanismOwnerKey(item.owner)} value={mechanismOwnerKey(item.owner)}>{item.label}</option>)}
    </select></label>
    {!options.length && <p className="field-note">Create another {owner.kind} to copy its settings.</p>}
    <label className="check-field"><input type="checkbox" checked={rules} onChange={event => setRules(event.target.checked)} />Trigger rules</label>
    <label className="check-field"><input type="checkbox" checked={range} onChange={event => setRange(event.target.checked)} />Destinations</label>
    <label className="check-field"><input type="checkbox" checked={requirements} onChange={event => setRequirements(event.target.checked)} />Key and switch requirements</label>
    {settings && <p className="field-note">Source destinations: upper {settings.top}, lower {settings.bottom}. Copied destinations must include the target's starting height. Requirements use this level's key and switch numbers.</p>}
    {error && <p role="alert" className="field-note">{error}</p>}
    <button type="button" className="small-button" onClick={reset}>Reset transfer draft</button>
    <button type="button" className="small-button" disabled={!settings || !range && !rules && !requirements} onClick={copy}>Copy selected settings</button>
  </details></div>;
}

export function MoveAuthoredMechanism({ document, room, edit, onShowRoom }: { document: LevelDocument; room: SourceRoom; edit: Edit; onShowRoom: (id: string) => void }) {
  const [target, setTarget] = useState(''), [error, setError] = useState('');
  useEffect(() => { setTarget(''); setError(''); }, [document, room.id]);
  const move = () => {
    try {
      const command = editRoomSource(moveSourceMovingRoom(document, room.id, target));
      buildPlayableLevel(applyCommand(document, command), -1);
      edit(command); setError(''); onShowRoom(target);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  return <div role="group" aria-label={`Move authored ${room.moving!.kind}`}><details onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setTarget(''); setError(''); } }}>
    <summary>Move this {room.moving!.kind} to another room</summary>
    <p className="field-note">Choose a connected static room. The mechanism keeps its destinations, starting height, trigger rules, requirements and material. The old room returns to its editable static heights. Apply is one Undo.</p>
    <label className="select-field"><span>Destination room</span><select aria-label="Mechanism destination room" value={target} onChange={event => setTarget(event.target.value)}>
      <option value="">Choose a room</option>{document.roomSource.rooms.filter(item => item.id !== room.id).map(item => <option key={item.id} value={item.id}>Room {item.id.replace(/^room-/, '')}</option>)}
    </select></label>
    {error && <p role="alert" className="field-note">{error}</p>}
    <button type="button" className="small-button" onClick={() => { setTarget(''); setError(''); }}>Cancel move draft</button>
    <button type="button" className="small-button" disabled={!target} onClick={move}>Move mechanism</button>
  </details></div>;
}
