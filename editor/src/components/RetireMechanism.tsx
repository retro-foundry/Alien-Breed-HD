import { WallStyleBrowser } from './WallStyleBrowser';
import { useEffect, useState } from 'react';
import type { LevelDocument } from '../model/level';
import { editRoomSource, type EditCommand } from '../model/commands';
import { retireMechanism, reassignImportedMechanism, retiringMechanismSettings, type ImportedMechanismOwner } from '../model/retireMechanism';
import { mechanismOwnerKey } from '../model/mechanismSettingsTransfer';
import { movingWallChoices } from '../model/sourceMovingRoom';

export function RetireMechanism({ document, owner, edit, onShowRoom }: {
  document: LevelDocument; owner: ImportedMechanismOwner; edit: (command: EditCommand | (() => EditCommand)) => void; onShowRoom: (room: string) => void;
}) {
  const settings = retiringMechanismSettings(document, owner), key = mechanismOwnerKey(owner);
  const originalRoom = document.roomSource.rooms.find(room => room.id === settings.room)!;
  const choices = movingWallChoices(document, originalRoom);
  const initialHeight = owner.kind === 'door' ? settings.top : settings.current;
  const [operation, setOperation] = useState('remove'), [height, setHeight] = useState(String(initialHeight));
  const [target, setTarget] = useState(''), [wall, setWall] = useState(String(choices[0]?.offset ?? '')), [error, setError] = useState('');
  const reset = () => { setOperation('remove'); setHeight(String(initialHeight)); setTarget(''); setWall(String(choices[0]?.offset ?? '')); setError(''); };
  useEffect(reset, [document, key]);
  const apply = () => {
    try {
      if (!height.trim() || !Number.isFinite(Number(height))) throw new Error('Enter the height to leave in the old room.');
      const change = operation === 'move' ? reassignImportedMechanism(document, owner, target, Number(height), Number(wall)) : retireMechanism(document, owner, Number(height));
      edit(editRoomSource(change)); onShowRoom(operation === 'move' ? target : settings.room);
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  return <details className="mechanism-retirement" onKeyDown={event => { if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); reset(); } }}>
    <summary>Remove or move this {owner.kind}</summary>
    <p className="field-note">The old room becomes static at the height below. Its attached walls stop moving too. Other copies keep working. Apply checks the whole level and commits one Undo.</p>
    <label className="select-field"><span>Action</span><select aria-label="Controller removal action" value={operation} onChange={event => setOperation(event.target.value)}>
      <option value="remove">Remove movement</option><option value="move">Move to another room</option>
    </select></label>
    <label className="select-field"><span>Old room {owner.kind === 'door' ? 'ceiling' : 'floor'} height</span><input aria-label="Old room static height" type="number" step="0.25" min={settings.top} max={settings.bottom} value={height} onChange={event => setHeight(event.target.value)} /></label>
    <p className="field-note">Allowed heights: {settings.top} to {settings.bottom}. Smaller values are higher. A door starts with its open height selected; a lift starts with its saved height.</p>
    {operation === 'move' && <>
      <label className="select-field"><span>Destination room</span><select aria-label="Imported mechanism destination room" value={target} onChange={event => setTarget(event.target.value)}>
        <option value="">Choose a static room</option>{document.roomSource.rooms.filter(room => room.id !== settings.room).map(room => <option key={room.id} value={room.id}>Room {room.id.replace(/^room-/, '')}</option>)}
      </select></label>
      <WallStyleBrowser records={choices} value={wall} onChange={setWall} label="reassigned moving wall" overrides={document.wallOverrides} />
      <p className="field-note">The destination needs an adjoining room and enough room for the full movement range. It receives independent movement with the same destinations, starting height, trigger rules and requirements.</p>
    </>}
    {error && <p role="alert" className="field-note">{error}</p>}
    <button type="button" className="small-button" onClick={reset}>Cancel removal draft</button>
    <button type="button" className="small-button" disabled={operation === 'move' && (!target || !wall)} onClick={apply}>{operation === 'move' ? 'Move imported mechanism' : 'Remove movement'}</button>
  </details>;
}
