import type { LevelDocument, Mechanism } from '../model/level';
import { mechanismDependencies } from '../model/mechanismDependencies';
import { switchSourceRoom } from '../model/nativeSwitches';

export function MechanismAttachments({ document, mechanism, onShowRoom }: {
  document: LevelDocument; mechanism: Mechanism; onShowRoom: (room: string) => void;
}) {
  const dependencies = mechanismDependencies(document, mechanism.kind, mechanism.id);
  const link = (zone: number) => {
    const room = document.roomSource.rooms.find(room => room.originZone === zone);
    return room ? <button className="small-button" title={`Frame Zone ${zone} on the map while keeping this mechanism selected.`}
      onClick={() => onShowRoom(room.id)}>Zone {zone}</button> : <span>Zone {zone} (deleted)</span>;
  };
  return <details className="mechanism-attachments">
    <summary title="Inspect the rooms containing the moving graphics and activation boundaries. A door or lift can affect neighbouring rooms.">
      Attachments: {dependencies.walls.length} moving {dependencies.walls.length === 1 ? 'wall' : 'walls'}
    </summary>
    <p className="field-note">The {mechanism.kind} moves a room plane and these walls together. Graphics and trigger boundaries can belong to different rooms.</p>
    <div className="info-row"><span>Controlled room</span>{link(dependencies.room)}</div>
    <div className="info-row"><span>Moving surface</span>{link(dependencies.surface.room)}</div>
    <table><caption>Moving wall attachments</caption><thead><tr><th scope="col">Wall</th><th scope="col">Graphics room</th><th scope="col">Trigger boundary rooms</th></tr></thead>
      <tbody>{dependencies.walls.map((wall, index) => <tr key={`${wall.line}-${wall.graphic}-${index}`}>
        <th scope="row">{index + 1}</th><td>{link(wall.room)}</td><td>{wall.collisionRooms.map(room => <span key={room}>{link(room)}{' '}</span>)}</td>
      </tr>)}</tbody>
    </table>
    {!!dependencies.keys.length && <p className="field-note">Requires {dependencies.keys.map(number => `Key ${number}`).join(', ')}.</p>}
    {!!dependencies.switches.length && <ul>{dependencies.switches.map(item => {
      const room = switchSourceRoom(document, item.number - 1);
      return <li key={item.number}>Switch {item.number}: {room ? <button className="small-button" onClick={() => onShowRoom(room)}>Room {room.replace(/^room-/, '')}</button> : 'not placed'}</li>;
    })}</ul>}
    <p className="field-note">Room links frame the room while keeping this mechanism selected. Key and switch requirements above must all be met; wall attachments determine where movement and touch/use activation happen.</p>
  </details>;
}
