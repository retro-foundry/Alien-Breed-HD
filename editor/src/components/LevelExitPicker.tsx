import type { LevelDocument } from '../model/level';
import { setExitZone, setSourceExit, type EditCommand } from '../model/commands';

export function LevelExitPicker({ document, onEdit }: { document?: LevelDocument; onEdit: (command: EditCommand) => void }) {
  const selected = document?.roomSource.exitRoom;
  const room = document?.roomSource.rooms.find(room => room.id === selected);
  const value = selected !== undefined ? room?.originZone ?? `source:${selected}` : document?.exitZone ?? 'original';
  const missing = typeof value === 'number' && value >= 0 && !document?.roomSource.rooms.some(room => room.originZone === value);
  return <label className="reference-picker"><span>Level exit</span><select aria-label="Level exit"
    title="Finish when player one enters this room, including every section of a concave room. Original uses the game slot exit. No exit lets you playtest freely. Custom exits require the updated game port."
    disabled={!document} value={value} onChange={event => {
      if (!document) return;
      const value = event.target.value;
      onEdit(value.startsWith('source:') ? setSourceExit(document, value.slice(7)) : setExitZone(document, value === 'original' ? null : Number(value)));
    }}>
    <option value="original">Original slot exit</option><option value="-1">No exit (playtest freely)</option>
    {missing && <option value={value} disabled>Zone {value} (deleted; choose another exit)</option>}
    {document?.roomSource.rooms.map(room => <option key={room.id} value={room.originZone ?? `source:${room.id}`}>
      {room.originZone === null ? `Room ${room.id.replace(/^room-/, '')} (new)` : `Zone ${room.originZone}`}
    </option>)}
  </select></label>;
}
