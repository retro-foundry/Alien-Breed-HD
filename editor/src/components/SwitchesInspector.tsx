import { useEffect, useRef, useState } from 'react';
import type { RoomSetupRequest } from './RoomSetupRequest';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import type { EditCommand } from '../model/commands';
import { setSwitchAutomaticReset, setCopiedSwitchAutomaticReset } from '../model/switchSettings';
import { availableSwitchSlots, setSourceSwitch, switchRequirements, reassignNativeSwitch, moveSourceSwitch, reassignCopiedSwitch } from '../model/sourceSwitches';
import { activeNativeSwitches } from '../model/nativeSwitches';
import { applyCommand, editRoomSource } from '../model/commands';
import { buildPlayableLevel } from '../model/playableBuild';
import { createLinkedSwitch } from '../model/switchMechanismLinks';
import { mechanismOwnerKey } from '../model/mechanismSettingsTransfer';
import { SwitchMechanismConnections, SwitchTargetFields, switchTargets, type SwitchTargetDraft } from './SwitchMechanismConnections';

function SwitchPlacement({ document, slot, room, kind, edit }: {
  document: LevelDocument; slot: number; room?: SourceRoom; kind: 'native' | 'copy' | 'authored';
  edit: (command: EditCommand | (() => EditCommand)) => void;
}) {
  const [destination, setDestination] = useState(room?.id ?? ''), [edge, setEdge] = useState(''), [error, setError] = useState('');
  const target = document.roomSource.rooms.find(room => room.id === destination);
  const apply = (remove = false) => {
    try {
      const placement = remove ? null : { room: destination, edge: Number(edge) };
      const change = kind === 'native' ? reassignNativeSwitch(document, slot, placement) : kind === 'copy' ? reassignCopiedSwitch(document, slot, placement) : moveSourceSwitch(document, slot, placement!);
      const command = editRoomSource(change);
      buildPlayableLevel(applyCommand(document, command), -1);
      edit(command); setError(''); setEdge('');
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  const required = switchRequirements(document, slot);
  return <>
    <SwitchMechanismConnections document={document} slot={slot} edit={edit} />
    <details><summary>Move switch {slot + 1} to another wall or room</summary>
      <p className="field-note">Choose the destination room and its solid lower wall. The switch moves to the wall centre. Its number and door/lift requirements stay the same. Apply is one Undo.</p>
      <label className="select-field"><span>Destination room</span><select aria-label={`Switch ${slot + 1} destination room`} value={destination} onChange={event => { setDestination(event.target.value); setEdge(''); }}>
        <option value="">Choose a room</option>{document.roomSource.rooms.map(room => <option key={room.id} value={room.id}>Room {room.id.replace(/^room-/, '')}</option>)}
      </select></label>
      <label className="select-field"><span>Destination wall</span><select aria-label={`Switch ${slot + 1} destination wall`} value={edge} onChange={event => setEdge(event.target.value)}>
        <option value="">Choose a solid wall edge</option>{target?.vertices.flatMap((a, index) => {
          if (target.boundaryMaterials && target.boundaryMaterials.edges.indexOf(target.boundaryMaterials.edges[index]!) !== index) return [];
          const b = target.vertices[(index + 1) % target.vertices.length]!;
          return [<option key={index} value={index}>Edge {index + 1}: ({a.x}, {a.z}) to ({b.x}, {b.z})</option>];
        })}
      </select></label>
      <button className="small-button" disabled={!target || edge === ''} onClick={() => apply()}>Move switch {slot + 1}</button>
    </details>
    {kind !== 'authored' && <>
      <button className="small-button" disabled={required} onClick={() => apply(true)}>Remove switch {slot + 1}</button>
      {required && <p className="field-note">A door or lift requires this switch. Clear that requirement before removing it. Moving keeps the requirement connected.</p>}
    </>}
    {error && <p role="alert" className="field-note">{error}</p>}
  </>;
}

function AuthorSwitches({ document, room, edit, request }: { document: LevelDocument; room: SourceRoom; edit: (command: EditCommand | (() => EditCommand)) => void; request?: RoomSetupRequest }) {
  const free = availableSwitchSlots(document);
  const [slot, setSlot] = useState(() => String(free[0] ?? '')), [edge, setEdge] = useState(''), [error, setError] = useState('');
  const [connection, setConnection] = useState<SwitchTargetDraft>({ target: '', mode: 'activate' });
  const details = useRef<HTMLDetailsElement>(null);
  useEffect(() => { if (slot === '' || !free.includes(Number(slot))) setSlot(String(free[0] ?? '')); }, [free.join(',')]);
  useEffect(() => {
    if (request?.room !== room.id || request.kind !== 'switch' || !details.current) return;
    // Reveal after the inspector has cleared its filter and restored sections.
    const frame = window.requestAnimationFrame(() => {
      if (!details.current) return;
      details.current.open = true;
      details.current.scrollIntoView({ block: 'start' });
      details.current.querySelector<HTMLSelectElement>('[aria-label="New switch wall"]')?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [request?.sequence]);
  const change = (number: number, value: { edge: number; automatic: boolean } | null) => {
    try {
      const command = editRoomSource(setSourceSwitch(document, room.id, number, value));
      buildPlayableLevel(applyCommand(document, command), -1);
      edit(command); setError('');
      if (!room.switches?.some(item => item.slot === number)) { setSlot(''); setEdge(''); }
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  const add = () => {
    try {
      const target = connection.target ? switchTargets(document).find(item => mechanismOwnerKey(item.owner) === connection.target)?.owner : undefined;
      if (connection.target && !target) throw new Error('The selected door or lift no longer exists. Choose its connection again.');
      const command = createLinkedSwitch(document, room.id, Number(slot), Number(edge), target, connection.mode);
      buildPlayableLevel(applyCommand(document, command), -1);
      edit(command); setError(''); setEdge(''); setConnection({ target: '', mode: 'activate' });
    } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
  };
  let roomIssue = '';
  if (free.length) try { setSourceSwitch(document, room.id, free[0]!, { edge: 0, automatic: false }); }
  catch (reason) { roomIssue = reason instanceof Error ? reason.message : String(reason); }
  const wallEdge = (index: number) => room.boundaryMaterials?.edges.indexOf(room.boundaryMaterials.edges[index]!) ?? index;
  const walls = room.vertices.flatMap((a, index) => {
    if (wallEdge(index) !== index) return [];
    const b = room.vertices[(index + 1) % room.vertices.length]!;
    const edges = room.vertices.flatMap((_, other) => wallEdge(other) === index ? [other + 1] : []);
    return [<option key={index} value={index}>{edges.length > 1 ? `Edges ${edges.join(', ')}: one continuous wall` : `Edge ${index + 1}: (${a.x}, ${a.z}) to (${b.x}, ${b.z})`}</option>];
  });
  const left = Math.min(...room.vertices.map(point => point.x)), top = Math.min(...room.vertices.map(point => point.z));
  const scale = 160 / Math.max(1, Math.max(...room.vertices.map(point => point.x)) - left, Math.max(...room.vertices.map(point => point.z)) - top);
  const diagram = room.vertices.map(point => ({ x: 30 + (point.x - left) * scale, y: 30 + (point.z - top) * scale }));
  return <section className="property-section" aria-label="Wall switches"><details ref={details}><summary>Wall switches</summary>
    <p className="field-note">Use a solid lower-storey wall at least 64 units long, with 64 units of headroom. An open doorway cannot hold a panel. The panel sits at the wall centre, 32 to 64 units above the floor. Stand nearby and press Space in Playtest.</p>
    <p className="field-note">Choose the door or lift below to connect it directly. The level has eight switch numbers; the next unused number is selected for you.</p>
    {roomIssue && <p className="setup-summary" role="note">{roomIssue} Put the switch in a neighbouring static room, then connect it to the moving room.</p>}
    <h4 className="setup-step">1. Choose a wall</h4>
    <svg viewBox="0 0 220 220" width="100%" style={{ maxHeight: 220 }} aria-label="Numbered switch wall edges">
      <polygon points={diagram.map(point => `${point.x},${point.y}`).join(' ')} fill="var(--panel-bg, #202830)" stroke="currentColor" />
      {diagram.map((a, index) => {
        const b = diagram[(index + 1) % diagram.length]!, x = (a.x + b.x) / 2, y = (a.y + b.y) / 2;
        const assigned = room.switches?.find(item => room.boundaryMaterials?.edges[index] === item.boundary);
        const label = `Choose edge ${index + 1} for new switch`;
        return <g key={index} role="button" tabIndex={0} aria-label={label} style={{ cursor: 'pointer' }} onClick={() => setEdge(String(wallEdge(index)))}
          onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); setEdge(String(wallEdge(index))); } }}>
          <title>{label}</title><line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={edge === String(wallEdge(index)) ? '#f1bc61' : assigned ? '#b1db77' : 'currentColor'} strokeWidth={6} />
          <rect x={x - 14} y={y - 10} width={28} height={20} rx={3} fill="#202830" />
          <text x={x} y={y + 5} textAnchor="middle" fill="currentColor" fontSize={13}>{index + 1}</text>
          {assigned && <text x={x + 17} y={y - 12} fill="#b1db77" fontSize={11}>S{assigned.slot + 1}</text>}
        </g>;
      })}
    </svg>
    <p className="field-note">The diagram shows this room from above. Click a numbered edge to choose the new switch wall; S marks an existing switch.</p>
    {!free.length ? <p className="field-note">All eight switch numbers are used. Select an existing switch below to connect, move or remove it.</p> : <fieldset className="new-switch-setup" aria-label="Add a wall switch">
      <label className="select-field"><span>Switch number</span><select aria-label="New switch number" value={slot} onChange={event => setSlot(event.target.value)}><option value="">Choose an unused number</option>{free.map(number => <option key={number} value={number}>Switch {number + 1}{switchRequirements(document, number) ? ' (already required by a mechanism)' : ''}</option>)}</select></label>
      <label className="select-field"><span>Wall</span><select aria-label="New switch wall" value={edge} onChange={event => setEdge(event.target.value)}><option value="">Choose a solid wall edge</option>{walls}</select></label>
      <h4 className="setup-step">2. Connect a door or lift (optional)</h4>
      <SwitchTargetFields document={document} prefix="New switch" value={connection} change={setConnection} />
      <p className="field-note">Add places the panel{connection.target ? ' and connects its mechanism' : ''} as one edit. Undo restores both. Stand nearby and press Space in Playtest.</p>
      <div className="setup-buttons"><button type="button" className="small-button" onClick={() => {
        setEdge(''); setSlot(String(free[0] ?? '')); setConnection({ target: '', mode: 'activate' }); setError('');
        if (details.current) { details.current.open = false; details.current.querySelector('summary')?.focus(); }
      }}>Cancel switch setup</button>
      <button type="button" className="small-button" disabled={slot === '' || edge === '' || !!roomIssue} onClick={add}>Add wall switch</button></div>
    </fieldset>}
    {error && <p className="field-note" role="alert">{error}</p>}
    {room.switches?.map(item => {
      const edges = room.boundaryMaterials?.edges.map((boundary, index) => boundary === item.boundary ? index : -1).filter(index => index >= 0) ?? [];
      const used = switchRequirements(document, item.slot);
      return <div key={item.slot} role="group" aria-label={`Authored switch ${item.slot + 1}`}>
        <h4>Switch {item.slot + 1}</h4>
        <label className="select-field"><span>Wall</span><select aria-label={`Switch ${item.slot + 1} wall`} value={edges[0] ?? ''} onChange={event => change(item.slot, { edge: Number(event.target.value), automatic: item.automatic })}>
          {!edges.length && <option value="">Wall missing — choose a wall</option>}{walls}
        </select></label>
        <ResetControl enabled={item.automatic} prefix={`authored switch ${item.slot + 1}`} change={automatic => change(item.slot, { edge: edges[0]!, automatic })} />
        <SwitchPlacement document={document} slot={item.slot} room={room} kind="authored" edit={edit} />
        <button type="button" className="small-button" disabled={used} onClick={() => change(item.slot, null)}>Remove switch {item.slot + 1}</button>
        {used && <p className="field-note">A door or lift requires this switch. Clear that requirement before removing it.</p>}
      </div>;
    })}

  </details></section>;
}

function ResetControl({ enabled, prefix, change }: { enabled: boolean; prefix: string; change: (enabled: boolean) => void }) {
  return <label className="select-field"><span>After activation</span>
    <select aria-label={`${prefix} reset mode`} value={enabled ? 'automatic' : 'manual'}
      title="Choose whether the native timer switches this off automatically or it stays on until used again. Changes apply immediately; Undo restores the previous setting."
      onChange={event => change(event.target.value === 'automatic')}>
      <option value="manual">Use again to reset</option><option value="automatic">Automatically resets</option>
    </select>
  </label>;
}

export function SwitchesInspector({ document, room, zone, edit, frameRoom, request }: {
  document: LevelDocument; room?: SourceRoom; zone?: number;
  request?: RoomSetupRequest;
  edit: (command: EditCommand | (() => EditCommand)) => void; frameRoom: (room: string) => void;
}) {
  return <>
    {room && <AuthorSwitches key={room.id} document={document} room={room} edit={edit} request={request} />}
    {activeNativeSwitches(document).filter(item => item.zone === zone).map(item => <section className="property-section" key={`original-${item.id}`} aria-label={`Switch ${item.id + 1}`}>
      <h3>Switch {item.id + 1}</h3>
      <ResetControl enabled={item.type !== 0} prefix={`switch ${item.id + 1}`} change={enabled => edit(() => setSwitchAutomaticReset(document, item.id, enabled))} />
      <SwitchPlacement document={document} slot={item.id} room={room} kind="native" edit={edit} />
      <p className="field-note">Stand near the switch and press Space in Build &amp; play. To require it on a door or lift, select Switch {item.id + 1} under Requires all selected.</p>
    </section>)}
    {document.roomSource.switchCopies?.filter(item => item.room === room?.id).map(item => {
      const graphics = document.roomSource.rooms.find(room => room.id === item.graphicsRoom);
      return <section className="property-section" key={`copy-${item.slot}`} aria-label={`Copied switch ${item.slot + 1}`}>
        <h3>Copied switch {item.slot + 1}</h3>
        <ResetControl enabled={item.bytes[2] !== 0} prefix={`copied switch ${item.slot + 1}`} change={enabled => edit(() => setCopiedSwitchAutomaticReset(document, item.slot, enabled))} />
        <SwitchPlacement document={document} slot={item.slot} room={room} kind="copy" edit={edit} />
        <p className="field-note">Uses Switch {item.slot + 1} in this level's requirements. To require this copy on a door or lift, select that number under Requires all selected. Stand near it and press Space in Build &amp; play.</p>
        <div className="info-row"><span>Copied from</span><span>Switch {item.id + 1}</span></div>
        <div className="info-row"><span>Activation room</span><span>Room {item.room.replace(/^room-/, '')}</span></div>
        <button className="small-button" disabled={!graphics} title="Frame the room containing this switch's wall graphics while keeping the switch selected."
          onClick={() => { if (graphics) frameRoom(graphics.id); }}>Show switch graphics room</button>
        <p className="field-note">This copy has independent reset settings. Changing the reset mode keeps its native timer state and attachments. Undo restores each edit.</p>
      </section>;
    })}
  </>;
}
