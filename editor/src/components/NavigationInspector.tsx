import { useEffect, useState, type PointerEvent } from 'react';
import type { LevelDocument } from '../model/level';
import type { SourceRoom } from '../model/roomSource';
import type { EditCommand } from '../model/commands';
import { autoAssignNavigation, autoLinkNavigation, assignNavigationPoint, editableNavigation, navigationAssignment, placeNavigationPoint, removeNavigationPoint, setNavigationGraph, setNavigationLink } from '../model/navigation';
import { pointInRoomOutline } from '../model/geometry';
import { roomSettings } from '../model/roomHeights';
import { RoomPositionForm } from './RoomPositionForm';

export function NavigationInspector({ document, room, edit }: { document: LevelDocument; room: SourceRoom; edit: (command: EditCommand) => void }) {
  const navigation = editableNavigation(document), settings = roomSettings(document, room);
  const hasUpper = !!settings.upper.length && settings.upperFloor > settings.upperCeiling;
  const [selected, setSelected] = useState<number | null>(null), [upper, setUpper] = useState(false), [placing, setPlacing] = useState(false);
  const [error, setError] = useState(''), [target, setTarget] = useState('');
  const [importing, setImporting] = useState(false);
  const [drag, setDrag] = useState<{ id: number; x: number; z: number } | null>(null);
  useEffect(() => setDrag(null), [document]);
  const point = navigation.points.find(point => point.id === selected);
  const apply = (command: () => EditCommand) => { try { edit(command()); setError(''); return true; } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); return false; } };
  const left = Math.min(...room.vertices.map(p => p.x)), top = Math.min(...room.vertices.map(p => p.z));
  const width = Math.max(...room.vertices.map(p => p.x)) - left, height = Math.max(...room.vertices.map(p => p.z)) - top, margin = Math.max(width, height) / 15;
  const local = (event: PointerEvent<SVGSVGElement>) => {
    const svg = event.currentTarget, coordinate = svg.createSVGPoint(); coordinate.x = event.clientX; coordinate.y = event.clientY;
    const transformed = coordinate.matrixTransform(svg.getScreenCTM()!.inverse());
    return { x: Math.round(transformed.x), z: Math.round(transformed.y) };
  };
  const owner = point?.room ?? (point && document.roomSource.rooms.find(room => pointInRoomOutline(room.vertices, point))?.id) ?? room.id;
  const pointLabel = (id: number) => `C${navigation.points.findIndex(point => point.id === id) + 1}`;
  const linkFrom = navigation.points.findIndex(p => p.id === selected), linkTo = navigation.points.findIndex(p => p.id === Number(target));
  const linkValue = navigation.links && linkFrom >= 0 && linkTo >= 0 ? `${navigation.links[linkFrom * 100 + linkTo]}/${navigation.links[linkTo * 100 + linkFrom]}` : '0/0';
  return <section className="property-section"><details><summary>Control points and links</summary>
    <p className="field-note">Control points are the original editor's reference markers. Place them inside a room on its lower or upper storey, then assign rooms and connect points below. They do not change room outlines or dotted wall connections.</p>
    {error && <p className="field-note" role="alert">{error}</p>}
    <label className="select-field"><span>Editing storey</span><select aria-label="Control-point storey" value={upper ? 'upper' : 'lower'} onChange={event => { setUpper(event.target.value === 'upper'); setDrag(null); }}><option value="lower">Lower</option><option value="upper" disabled={!hasUpper}>Upper{!hasUpper ? ' (no upper storey)' : ''}</option></select></label>
    <button type="button" className="small-button" disabled={navigation.points.length >= 100} onClick={() => { setPlacing(!placing); setDrag(null); }}>{placing ? 'Cancel control-point placement' : 'Place control point in room'}</button>
    <p className="field-note">{placing ? 'Click inside the room diagram to add a point. Escape cancels.' : 'Click a point to select it; drag it to move. Release saves one Undo; Escape cancels the drag. Coordinates use whole map units.'}</p>
    <svg viewBox={`${left - margin} ${top - margin} ${width + margin * 2} ${height + margin * 2}`} width="240" height="200" role="img" aria-label="Control-point room diagram" tabIndex={0} style={{ width: '100%', background: '#141b22', touchAction: 'none' }}
      onKeyDown={event => { if (event.key === 'Escape') { setPlacing(false); setDrag(null); event.preventDefault(); event.stopPropagation(); } }}
      onPointerDown={event => {
        if (event.button !== 0) return;
        event.currentTarget.focus(); const location = local(event);
        if (placing) { const id = navigation.nextId; if (apply(() => placeNavigationPoint(document, null, room.id, upper, location.x, location.z))) { setSelected(id); setPlacing(false); } return; }
        const id = event.target instanceof Element ? event.target.getAttribute('data-control-id') : null;
        if (id !== null) { const point = navigation.points.find(p => p.id === Number(id))!; setSelected(point.id); setDrag({ id: point.id, x: point.x, z: point.z }); event.currentTarget.setPointerCapture(event.pointerId); }
      }}
      onPointerMove={event => { if (drag) setDrag({ id: drag.id, ...local(event) }); }}
      onPointerCancel={() => setDrag(null)}
      onPointerUp={event => { if (drag) { const moved = navigation.points.find(p => p.id === drag.id)!; if (drag.x !== moved.x || drag.z !== moved.z) apply(() => placeNavigationPoint(document, drag.id, room.id, upper, drag.x, drag.z)); setDrag(null); } if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}>
      <polygon points={room.vertices.map(p => `${p.x},${p.z}`).join(' ')} fill="#263e36" stroke="#b2de7a" strokeWidth={margin / 8} />
      {navigation.links && navigation.points.flatMap((a, i) => navigation.points.slice(i + 1).map((b, j) => navigation.links![i * 100 + i + 1 + j] && a.upper === upper && b.upper === upper ? <line key={`${a.id}:${b.id}`} x1={a.x} y1={a.z} x2={b.x} y2={b.z} stroke="#82caff" strokeWidth={margin / 12} /> : null))}
      {navigation.points.filter(p => p.upper === upper && pointInRoomOutline(room.vertices, p)).map(p => {
        const location = drag?.id === p.id ? drag : p;
        return <g key={p.id}><circle data-control-id={p.id} cx={location.x} cy={location.z} r={margin / 2.5} fill={p.id === selected ? '#f0c878' : '#82caff'} /><text x={location.x + margin / 2} y={location.z} fill="#fff" fontSize={margin} pointerEvents="none">{pointLabel(p.id)}</text></g>;
      })}
    </svg>
    <label className="select-field"><span>Selected control point ({navigation.points.length} / 100)</span><select aria-label="Selected control point" value={selected ?? ''} onChange={event => { const id = event.target.value === '' ? null : Number(event.target.value); setSelected(id); setPlacing(false); const point = navigation.points.find(p => p.id === id); if (point) setUpper(point.upper); }}><option value="">Choose a point</option>{navigation.points.map(p => <option key={p.id} value={p.id}>{pointLabel(p.id)}: {p.x}, {p.z} ({p.upper ? 'upper' : 'lower'})</option>)}</select></label>
    {point && <>
      <RoomPositionForm rooms={document.roomSource.rooms} identity={`control-${point.id}`} room={owner} x={point.x} z={point.z} label="Control-point room" onApply={(room, x, z) => apply(() => placeNavigationPoint(document, point.id, room, upper, x, z))} />
      <button type="button" className="small-button" onClick={() => apply(() => placeNavigationPoint(document, point.id, owner, upper, point.x, point.z))}>Apply control-point storey</button>
      <button type="button" className="danger-button" onClick={() => { if (apply(() => removeNavigationPoint(document, point.id))) setSelected(null); }}>Delete control point and links</button>
    </>}
    <h4>This room's assignments</h4>
    {(hasUpper ? [false, true] : [false]).map(upper => <label key={String(upper)} className="select-field"><span>{upper ? 'Upper' : 'Lower'} control point</span><select aria-label={`${upper ? 'Upper' : 'Lower'} room control point`} value={navigationAssignment(document, room.id, upper) ?? ''} onChange={event => apply(() => assignNavigationPoint(document, room.id, upper, event.target.value === '' ? null : Number(event.target.value)))}><option value="">None</option>{navigation.points.map(p => <option key={p.id} value={p.id}>{pointLabel(p.id)} ({p.upper ? 'upper' : 'lower'})</option>)}</select></label>)}
    <button type="button" className="small-button" onClick={() => apply(() => autoAssignNavigation(document, [room.id]))}>Assign nearest visible lower point</button>
    <button type="button" className="small-button" onClick={() => apply(() => autoAssignNavigation(document, document.roomSource.rooms.map(room => room.id)))}>Assign all lower rooms automatically</button>
    <h4>Control-point link graph</h4>
    {!navigation.links && <><p className="field-note">Runtime level files do not contain the editable link graph. Import the matching twolev.links file for this point list, or explicitly create a new empty graph.</p><button type="button" className="small-button" onClick={() => apply(() => setNavigationGraph(document, new Uint8Array(10000)))}>Create new empty link graph</button></>}
    <button type="button" className="small-button" onClick={() => setImporting(!importing)}>{importing ? 'Cancel link graph import' : 'Import twolev.links'}</button>
    {importing && <label className="select-field"><span>Choose twolev.links</span><input type="file" aria-label="Import control-point links" accept=".links" onChange={async event => {
      const file = event.target.files?.[0]; event.target.value = ''; if (!file) return;
      if (file.size !== 10000) { setError('twolev.links needs exactly 10000 bytes.'); return; }
      try { const bytes = new Uint8Array(await file.arrayBuffer()); if (apply(() => setNavigationGraph(document, bytes))) setImporting(false); } catch (reason) { setError(reason instanceof Error ? reason.message : String(reason)); }
    }} /></label>}
    {navigation.links && <>
      <button type="button" className="small-button" onClick={() => apply(() => autoLinkNavigation(document))}>Rebuild visual links automatically</button>
      <p className="field-note">Automatic visual links follow straight lines through adjoining room outlines. As in the original editor, they ignore wall type, doors and height differences, and preserve your physical links.</p>
      <button type="button" className="small-button" onClick={() => { const url = URL.createObjectURL(new Blob([new Uint8Array(navigation.links!)])); const link = window.document.createElement('a'); link.href = url; link.download = 'twolev.links'; link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 60000); }}>Download twolev.links</button>
      <p className="field-note">Select a point above, then choose its partner. Physical links can be one-way or two-way. Visual links record visibility only. These are the original AMOS graph categories. These source links are saved with your project; they do not add AI behavior to the game port.</p>
      <label className="select-field"><span>Link selected point to</span><select aria-label="Control-point link partner" value={target} onChange={event => setTarget(event.target.value)}><option value="">Choose another point</option>{navigation.points.filter(p => p.id !== selected).map(p => <option key={p.id} value={p.id}>{pointLabel(p.id)}</option>)}</select></label>
      <label className="select-field"><span>Link direction</span><select aria-label="Control-point link direction" value={linkValue} disabled={selected === null || target === '' || Number(target) === selected} onChange={event => { const [forward, reverse] = event.target.value.split('/').map(Number); apply(() => setNavigationLink(document, selected!, Number(target), forward!, reverse!)); }}><option value="0/0">No link</option><option value="1/1">Physical: both directions</option><option value="1/2">Physical: selected to partner</option><option value="2/1">Physical: partner to selected</option><option value="2/2">Visual link only</option></select></label>
    </>}
  </details></section>;
}
