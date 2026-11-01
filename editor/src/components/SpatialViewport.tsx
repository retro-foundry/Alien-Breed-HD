import { useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import type { LevelDocument } from '../model/level';
import { applyCommand, type EditCommand } from '../model/commands';
import { editorScene, pickSceneFace, spatialEdit, type SpatialEdit } from '../model/editorScene';
import { roomSettings } from '../model/roomHeights';
import type { SourceRoom } from '../model/roomSource';
import { add, sub, mul, cameraFor, cameraBasis, frameCamera, projectPoint, screenRay, rayAxis, rayPlane, clipCameraPolygon, zoomCamera, type EditingView, type Vector3, type EditorCamera } from '../model/editorCamera';
import type { Selection } from './MapCanvas';
import './SpatialViewport.css';

interface Props {
  document: LevelDocument; view: EditingView; layer: 'lower' | 'upper'; grid: number; showObjects: boolean;
  tool: 'select' | 'pan' | 'room' | 'box' | 'object'; placementLabel?: string;
  selection: Selection; selectedRoom?: SourceRoom; objectIds: readonly number[];
  onSelect: (selection: Selection) => void; onRoomSelect: (id: string, add?: boolean) => void;
  onObjectToggle: (id: number) => void; onEdit: (command: EditCommand) => void; onError: (error: unknown) => void;
  onPlace: (room: string | undefined, x: number, z: number) => void;
  onCursor: (x: number, z: number) => void; fitKey: number; frameSelection: number;
  focusRoom?: { id: string; request: number; frame?: boolean }; focusZone?: { id: number; request: number };
}
type Handle = { id: string; label: string; position: Vector3; axis?: Vector3; change: SpatialEdit; colour: string };
type Drag = { pointer: number; startX: number; startY: number; camera: EditorCamera } &
  ({ kind: 'pan' } | { kind: 'orbit' } | { kind: 'edit'; change: SpatialEdit; origin: Vector3; axis?: Vector3; normal: Vector3; start: Vector3 | number });
const vertical = { x: 0, y: 1, z: 0 };
const axes = [{ id: 'X', vector: { x: 1, y: 0, z: 0 }, colour: '#f28c83' }, { id: 'Y', vector: { x: 0, y: -1, z: 0 }, colour: '#b2de7a' }, { id: 'Z', vector: { x: 0, y: 0, z: 1 }, colour: '#82bfff' }];

export function SpatialViewport(props: Props) {
  const { document, view, layer, grid, tool, showObjects, selection, selectedRoom, onSelect, onRoomSelect, onEdit, onError } = props;
  const holder = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null), drag = useRef<Drag | null>(null), space = useRef(false);
  const [size, setSize] = useState({ width: 0, height: 0 }), [camera, setCamera] = useState(() => cameraFor(view));
  const [draft, setDraft] = useState<SpatialEdit | null>(null), [ceilings, setCeilings] = useState(false), [wireframe, setWireframe] = useState(false);
  const orbit = view === 'orthographic' || view === 'perspective';
  const preview = useMemo(() => {
    if (!draft) return { document, error: '' };
    try { return { document: applyCommand(document, spatialEdit(document, draft)), error: '' }; }
    catch (error) { return { document, error: error instanceof Error ? error.message : String(error) }; }
  }, [document, draft]);
  const scene = useMemo(() => editorScene(preview.document, layer === 'upper'), [preview.document, layer]);
  const faces = useMemo(() => scene.faces.filter(face => ceilings || face.kind !== 'ceiling'), [scene, ceilings]);
  const markers = useMemo(() => showObjects ? scene.markers : scene.markers.filter(m => m.kind === 'start'), [scene, showObjects]);
  const points = useMemo(() => [...scene.faces.flatMap(face => [...face.points]), ...markers.map(marker => marker.position)], [scene, markers]);
  const selectedMarker = markers.find(marker => marker.kind === selection?.kind && marker.id === selection.id);
  const sourceRoom = preview.document.roomSource.rooms.find(room => room.id === selectedRoom?.id);
  const project = (point: Vector3) => projectPoint(point, camera, size.width, size.height);
  const ray = (x: number, y: number, source = camera) => screenRay(x, y, source, size.width, size.height);
  const cancel = () => { if (drag.current?.kind === 'pan' || drag.current?.kind === 'orbit') setCamera(drag.current.camera); drag.current = null; setDraft(null); };
  useLayoutEffect(() => {
    const observer = new ResizeObserver(([entry]) => { if (entry) setSize({ width: entry.contentRect.width, height: entry.contentRect.height }); });
    observer.observe(holder.current!); return () => observer.disconnect();
  }, []);
  const initialized = useRef('');
  useLayoutEffect(() => {
    const key = `${view}:${layer}`;
    if (size.width < 100 || size.height < 100 || initialized.current === key) return;
    const projectionChanged = initialized.current.split(':')[0] !== view; initialized.current = key;
    const roomId = selectedRoom?.id ?? selectedMarker?.room;
    const roomPoints = roomId ? scene.faces.filter(f => f.room === roomId).flatMap(f => [...f.points]) : [];
    setCamera(current => frameCamera(projectionChanged ? cameraFor(view) : current, roomPoints.length ? [...roomPoints, ...(selectedMarker ? [selectedMarker.position] : [])] : points, size.width, size.height));
  }, [view, layer, size, points, selectedRoom, selectedMarker, scene]);
  const fit = useRef(props.fitKey), framed = useRef(props.frameSelection);
  useEffect(() => {
    if (fit.current !== props.fitKey) { fit.current = props.fitKey; setCamera(current => frameCamera(current, points, size.width, size.height)); }
    if (framed.current !== props.frameSelection) {
      framed.current = props.frameSelection;
      const selected = selectedMarker ? [selectedMarker.position] : sourceRoom ? scene.faces.filter(face => face.room === sourceRoom.id).flatMap(face => [...face.points]) : [];
      if (selected.length) setCamera(current => frameCamera(current, selected, size.width, size.height));
    }
  }, [props.fitKey, props.frameSelection, points, size, selectedMarker, sourceRoom, scene]);
  const focused = useRef<{ room?: Props['focusRoom']; zone?: Props['focusZone'] }>({});
  useEffect(() => {
    const roomChanged = focused.current.room !== props.focusRoom;
    if (!roomChanged && focused.current.zone === props.focusZone) return;
    focused.current = { room: props.focusRoom, zone: props.focusZone };
    const id = roomChanged && props.focusRoom ? props.focusRoom.id : document.roomSource.rooms.find(room => room.originZone === props.focusZone?.id)?.id;
    const target = [...scene.faces.filter(face => face.room === id).flatMap(face => [...face.points]), ...(selectedMarker && selectedMarker.room === id ? [selectedMarker.position] : [])];
    if (target.length) setCamera(current => {
      const framed = frameCamera(current, target, size.width, size.height);
      return roomChanged && props.focusRoom?.frame ? framed : { ...current, target: framed.target };
    });
  }, [props.focusRoom, props.focusZone, document.roomSource, scene, size, selectedMarker]);
  useEffect(() => { drag.current = null; setDraft(null); }, [document, tool, view, layer]);
  useEffect(() => { const blur = () => { space.current = false; drag.current = null; setDraft(null); }; window.addEventListener('blur', blur); return () => window.removeEventListener('blur', blur); }, []);

  const handles: Handle[] = [];
  if (tool === 'select' && selectedMarker) {
    const scale = Math.min(size.width, size.height) * .9 / (camera.perspective ? project(selectedMarker.position).depth : camera.distance);
    for (const axis of axes) {
      if (selectedMarker.kind === 'start' && axis.id === 'Y') continue;
      const position = add(selectedMarker.position, mul(axis.vector, 65 / scale)), a = project(selectedMarker.position), b = project(position);
      if (Math.hypot(b.x - a.x, b.y - a.y) < 20) continue;
      handles.push({ id: `axis-${axis.id}`, label: `Move ${selectedMarker.kind} ${axis.id}`, position, axis: axis.vector,
        change: { kind: 'marker', marker: selectedMarker, position: selectedMarker.position }, colour: axis.colour });
    }
  }
  if (tool === 'select' && sourceRoom) {
    const settings = roomSettings(preview.document, sourceRoom), floor = settings[layer === 'upper' ? 'upperFloor' : 'floor'], ceiling = settings[layer === 'upper' ? 'upperCeiling' : 'ceiling'];
    const centre = { x: sourceRoom.vertices.reduce((sum, p) => sum + p.x, 0) / sourceRoom.vertices.length, z: sourceRoom.vertices.reduce((sum, p) => sum + p.z, 0) / sourceRoom.vertices.length };
    const lower = project({ ...centre, y: floor }), upper = project({ ...centre, y: ceiling });
    if (Math.hypot(lower.x - upper.x, lower.y - upper.y) > 20) for (const field of ['floor', 'ceiling'] as const) {
      const position = { ...centre, y: field === 'floor' ? floor : ceiling };
      handles.push({ id: field, label: `Drag room ${field} height`, position, axis: vertical,
        change: { kind: 'height', room: sourceRoom.id, field: layer === 'upper' ? field === 'floor' ? 'upperFloor' : 'upperCeiling' : field, value: position.y }, colour: field === 'floor' ? '#b2de7a' : '#e6bd81' });
    }
    sourceRoom.vertices.forEach((position, index) => handles.push({ id: `corner-${index}`, label: `Move room corner ${index + 1}`, position: { ...position, y: floor }, axis: view === 'left' || view === 'right' ? axes[2]!.vector : view === 'front' || view === 'back' ? axes[0]!.vector : undefined, colour: '#b2de7a',
      change: { kind: 'corner', room: sourceRoom.id, index, position: { ...position, y: floor } } }));
  }
  useLayoutEffect(() => {
    const node = canvas.current, context = node?.getContext('2d'); if (!node || !context || !size.width || !size.height) return;
    const ratio = window.devicePixelRatio || 1;
    node.width = Math.round(size.width * ratio); node.height = Math.round(size.height * ratio); context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = '#161c24'; context.fillRect(0, 0, size.width, size.height);
    const projected = faces.map(face => ({ face, pixels: clipCameraPolygon(face.points, camera).map(project) })).filter(face => face.pixels.length >= 3);
    projected.sort((a, b) => b.pixels.reduce((s, p) => s + p.depth, 0) / b.pixels.length - a.pixels.reduce((s, p) => s + p.depth, 0) / a.pixels.length);
    for (const { face, pixels } of projected) {
      const selected = face.room === selectedRoom?.id;
      context.beginPath(); pixels.forEach((point, i) => i ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)); context.closePath();
      if (!wireframe) { context.fillStyle = face.kind === 'wall' ? selected ? '#b2de7a16' : '#718aa008' : selected ? '#344b37' : face.kind === 'ceiling' ? '#3a3b45' : '#263441'; context.fill(); }
      context.strokeStyle = selected ? '#b2de7a88' : face.kind === 'wall' ? '#667d8c60' : '#5c758766'; context.lineWidth = selected ? 1.5 : .7; if (face.kind === 'wall') context.stroke();
    }
    for (const marker of [...markers].sort((a, b) => project(b.position).depth - project(a.position).depth)) {
      const point = project(marker.position); if (!point.visible) continue;
      const selected = selection?.kind === marker.kind && selection.id === marker.id || marker.kind === 'object' && props.objectIds.includes(marker.id);
      context.fillStyle = selected ? '#c8ef9c' : marker.kind === 'start' ? '#ffcf81' : '#a3c7dc'; context.strokeStyle = '#16212b'; context.lineWidth = 2;
      context.beginPath(); context.arc(point.x, point.y, selected ? 7 : 4, 0, Math.PI * 2); context.fill(); context.stroke();
      if (selected || marker.kind === 'start') { context.font = '11px sans-serif'; context.fillText(marker.name, point.x + 11, point.y - 8); }
    }
    for (const handle of handles.filter(handle => handle.axis && handle.change.kind === 'marker')) {
      const start = project(selectedMarker!.position), end = project(handle.position);
      context.strokeStyle = handle.colour; context.lineWidth = 2; context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
    }
    const basis = cameraBasis(camera);
    for (const axis of axes) {
      const x = axis.vector.x * basis.right.x + axis.vector.y * basis.right.y + axis.vector.z * basis.right.z;
      const y = axis.vector.x * basis.up.x + axis.vector.y * basis.up.y + axis.vector.z * basis.up.z;
      context.strokeStyle = axis.colour; context.fillStyle = axis.colour; context.lineWidth = 2; context.beginPath(); context.moveTo(40, size.height - 65); context.lineTo(40 + x * 27, size.height - 65 - y * 27); context.stroke();
      context.fillText(axis.id, 37 + x * 38, size.height - 61 - y * 38);
    }
  });

  function local(event: { clientX: number; clientY: number }) { const rect = canvas.current!.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
  function startEdit(event: ReactPointerEvent, change: SpatialEdit, explicitAxis?: Vector3) {
    const point = local(event), origin = change.kind === 'height' ? handles.find(h => h.change.kind === 'height' && h.change.field === change.field)!.position : change.position;
    let axis = explicitAxis;
    if (!axis && change.kind === 'corner' && ['front', 'back', 'left', 'right'].includes(view)) axis = view === 'front' || view === 'back' ? axes[0]!.vector : axes[2]!.vector;
    const normal = change.kind === 'marker' && !orbit && ['front', 'back', 'left', 'right'].includes(view) ? cameraBasis(camera).back : vertical;
    const initialRay = ray(point.x, point.y), start = axis ? rayAxis(initialRay, origin, axis) : rayPlane(initialRay, origin, normal);
    if (start === null) return;
    drag.current = { kind: 'edit', pointer: event.pointerId, startX: point.x, startY: point.y, camera, change, origin, axis, normal, start };
    canvas.current!.focus(); canvas.current!.setPointerCapture(event.pointerId);
  }
  function markerAt(x: number, y: number) {
    return [...markers].filter(marker => { const p = project(marker.position); return p.visible && Math.hypot(p.x - x, p.y - y) < 10; }).sort((a, b) => project(a.position).depth - project(b.position).depth)[0];
  }
  function nudge(handle: Handle, direction: number, shift: boolean) {
    try {
      const step = handle.change.kind === 'height' ? shift ? 4 : .25 : handle.axis?.y ? shift ? 4 : .5 : shift ? Math.max(1, grid) * 4 : Math.max(1, grid);
      const change = handle.change.kind === 'height' ? { ...handle.change, value: handle.change.value + direction * step }
        : { ...handle.change, position: add(handle.change.position, mul(handle.axis ?? axes[0]!.vector, direction * step)) };
      onEdit(spatialEdit(document, change));
    } catch (error) { onError(error); }
  }
  return <div ref={holder} className="map-holder spatial-viewport">
    <canvas ref={canvas} aria-label="Editing viewport" aria-describedby="spatial-controls" data-view={view} data-camera={JSON.stringify(camera)} data-size={JSON.stringify(size)} tabIndex={0}
      style={{ width: size.width, height: size.height, cursor: tool === 'object' ? 'crosshair' : tool === 'pan' ? 'grab' : 'default' }}
      onContextMenu={event => event.preventDefault()}
      onKeyDown={event => { if (event.key === ' ') { space.current = true; event.preventDefault(); } if (event.key === 'Escape' && drag.current) { event.preventDefault(); event.stopPropagation(); cancel(); } }}
      onKeyUp={event => { if (event.key === ' ') space.current = false; }}
      onWheel={event => { const p = local(event); setCamera(current => zoomCamera(current, p.x, p.y, Math.exp(Math.max(-1, Math.min(1, event.deltaY * .001))), size.width, size.height)); }}
      onPointerDown={event => {
        event.currentTarget.focus(); const p = local(event);
        if (event.button !== 0 || space.current || tool === 'pan') { drag.current = { kind: orbit && event.button === 2 && !space.current ? 'orbit' : 'pan', pointer: event.pointerId, startX: p.x, startY: p.y, camera }; event.currentTarget.setPointerCapture(event.pointerId); return; }
        const hit = pickSceneFace(tool === 'object' && (orbit || view === 'bottom') ? faces.filter(face => face.kind === 'floor') : faces, ray(p.x, p.y));
        if (tool === 'object') { const step = event.altKey || grid === 0 ? 1 : grid; props.onPlace(hit?.face.room, Math.round((hit?.position.x ?? 0) / step) * step, Math.round((hit?.position.z ?? 0) / step) * step); return; }
        const marker = markerAt(p.x, p.y);
        if (marker) {
          if (marker.kind === 'object' && (event.shiftKey || event.ctrlKey || event.metaKey)) { props.onObjectToggle(marker.id); return; }
          onSelect({ kind: marker.kind, id: marker.id }); startEdit(event, { kind: 'marker', marker, position: marker.position }); return;
        }
        if (hit) { onRoomSelect(hit.face.room, event.shiftKey || event.ctrlKey || event.metaKey); return; }
        onSelect(null); drag.current = { kind: orbit ? 'orbit' : 'pan', pointer: event.pointerId, startX: p.x, startY: p.y, camera }; event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={event => {
        const p = local(event), current = drag.current;
        const hit = pickSceneFace(faces, ray(p.x, p.y)); if (hit) props.onCursor(hit.position.x, hit.position.z);
        if (!current) return;
        const dx = p.x - current.startX, dy = p.y - current.startY;
        if (current.kind === 'orbit') { setCamera({ ...current.camera, yaw: current.camera.yaw - dx * .008, pitch: Math.max(-1.5, Math.min(1.5, current.camera.pitch + dy * .008)) }); return; }
        if (current.kind === 'pan') { const basis = cameraBasis(current.camera), scale = current.camera.distance / (Math.min(size.width, size.height) * .9); setCamera({ ...current.camera, target: add(current.camera.target, add(mul(basis.right, -dx * scale), mul(basis.up, dy * scale))) }); return; }
        if (Math.hypot(dx, dy) < 3) return;
        const currentRay = ray(p.x, p.y, current.camera);
        let point: Vector3;
        if (current.axis) { const value = rayAxis(currentRay, current.origin, current.axis); if (value === null) return; point = add(current.origin, mul(current.axis, value - (current.start as number))); }
        else { const value = rayPlane(currentRay, current.origin, current.normal); if (!value) return; point = add(current.origin, sub(value, current.start as Vector3)); }
        const horizontalStep = event.altKey || grid === 0 ? 1 : grid;
        const yStep = current.change.kind === 'height' ? event.altKey ? .25 : 1 : .5;
        const snapped = { x: Math.abs(point.x - current.origin.x) < 1e-7 ? current.origin.x : Math.round(point.x / horizontalStep) * horizontalStep,
          y: Math.abs(point.y - current.origin.y) < 1e-7 ? current.origin.y : Math.round(point.y / yStep) * yStep,
          z: Math.abs(point.z - current.origin.z) < 1e-7 ? current.origin.z : Math.round(point.z / horizontalStep) * horizontalStep };
        setDraft(current.change.kind === 'height' ? { ...current.change, value: snapped.y } : { ...current.change, position: snapped });
      }}
      onPointerUp={event => {
        const current = drag.current; drag.current = null;
        if (current?.kind === 'edit' && draft) { try { onEdit(spatialEdit(document, draft)); } catch (error) { onError(error); } }
        setDraft(null); if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={cancel} onLostPointerCapture={() => { drag.current = null; setDraft(null); }} />
    {!faces.length && <div className="spatial-empty" role="status">{layer === 'upper' ? 'No upper-floor rooms are visible. Choose Lower to edit the lower floor.' : 'No rooms yet. Use Draw room in Top view to create one.'}</div>}
    <div className="map-label">EDITING / {view.toUpperCase()} / {layer.toUpperCase()}</div>
    <div className="spatial-options"><label title="Show ceiling surfaces in this editing view. Hiding them lets you look inside; it does not remove ceilings from the game."><input type="checkbox" checked={ceilings} onChange={event => setCeilings(event.target.checked)} />Show ceilings</label><label title="Show room outlines without filled surfaces."><input type="checkbox" checked={wireframe} onChange={event => setWireframe(event.target.checked)} />Wireframe</label></div>
    {handles.toSorted((a, b) => project(b.position).depth - project(a.position).depth).map(handle => { const p = project(handle.position); return p.visible && p.x > 5 && p.x < size.width - 5 && p.y > 5 && p.y < size.height - 5 ? <button key={handle.id} className={`spatial-handle ${handle.id.startsWith('corner') ? 'spatial-corner' : ''}`} aria-label={handle.label} title={`${handle.label}. Drag to preview; release applies one edit. Escape cancels. Arrow keys nudge; Shift increases the step. Y increases downwards.`}
      style={{ left: p.x, top: p.y, borderColor: handle.colour, color: handle.colour }} onPointerDown={event => { event.preventDefault(); startEdit(event, handle.change, handle.axis); }}
      onKeyDown={event => { if (['ArrowUp', 'ArrowRight', 'ArrowDown', 'ArrowLeft'].includes(event.key)) { event.preventDefault(); event.stopPropagation(); nudge(handle, (event.key === 'ArrowUp' || event.key === 'ArrowRight' ? 1 : -1) * (handle.change.kind === 'height' ? -1 : 1), event.shiftKey); } }}>
      {handle.id.startsWith('axis-') ? handle.id.slice(5) : handle.id === 'floor' ? 'F' : handle.id === 'ceiling' ? 'C' : ''}</button> : null; })}
    <div id="spatial-controls" className="map-help">{tool === 'object' ? `Placing ${props.placementLabel}: click a room; Escape cancels.` : `${orbit ? 'Right drag: orbit. ' : ''}Space/middle drag: pan. Wheel: zoom. Click: select. Drag objects/handles: edit. F: fit. Shift+F: selection.`}</div>
    {preview.error && <div className="spatial-validation" role="status">{preview.error} Release rejects the edit; Escape cancels.</div>}
  </div>;
}
