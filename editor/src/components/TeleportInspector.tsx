import { setSourceTeleport, type EditCommand } from '../model/commands';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import { editableTeleport } from '../model/sourceTeleports';
import { RoomPositionForm } from './RoomPositionForm';
import { pointInRoomOutline } from '../model/geometry';

export function TeleportInspector({ document, room, edit, onShowDestination }: {
  document: LevelDocument; room: SourceRoom; edit: (command: EditCommand | (() => EditCommand)) => void;
  onShowDestination: (room: string) => void;
}) {
  const link = editableTeleport(document, room);
  const target = document.roomSource.rooms.find(room => room.id === link.destination);
  return <section className="property-section"><h3>Teleporter</h3>
    <p className="field-note">{link.destination === null ? 'Disabled. Choose a room and landing point, then Apply position to enable.'
      : !target ? 'Destination room is missing. Choose another room, or disable this teleporter.'
        : !pointInRoomOutline(target.vertices, link) ? 'The landing point is outside its room. Move it inside before building.'
        : 'Enabled. Entering this room teleports to the landing point below.'}</p>
    <RoomPositionForm rooms={document.roomSource.rooms} identity={`teleport-${room.id}`} room={link.destination ?? ''} x={link.x} z={link.z} label="Teleport destination room"
      onApply={(destination, x, z) => edit(() => setSourceTeleport(document, room.id, destination, x, z))} />
    <button className="small-button" disabled={!target} title="Frame the destination room in the map while keeping this teleporter selected." onClick={() => { if (target) onShowDestination(target.id); }}>Show destination</button>
    <button className="small-button" disabled={link.destination === null} title="Disable this teleporter as one undoable edit. Its landing coordinates are kept." onClick={() => edit(() => setSourceTeleport(document, room.id, null, link.x, link.z))}>Disable teleporter</button>
  </section>;
}
