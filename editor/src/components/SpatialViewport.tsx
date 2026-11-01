import {spatialRoomsInSelectionBox} from '../model/roomSelection';
import { texturedEditorScene, type TexturedFace } from '../model/texturedEditorScene';
import { SpatialRenderer, loadSpatialMaterials, type IndexedMaterial } from '../model/spatialRenderer';
import { surfaceMaterialTarget, type SurfaceMaterialTarget } from '../model/wallMaterials';
import { polygonMaterialTarget, type PolygonMaterialTarget } from '../model/polygonMaterials';
import { WallPaintDialog } from './WallPaintDialog';
import { PolygonPaintDialog } from './PolygonPaintDialog';
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import type { LevelDocument } from '../model/level';
import { applyCommand, type EditCommand } from '../model/commands';
import { editorScene, pickSceneFace, spatialEdit, type SpatialEdit } from '../model/editorScene';
import { roomSettings } from '../model/roomHeights';
import type { SourceRoom } from '../model/roomSource';
import { add, sub, mul, cameraFor, cameraBasis, frameCamera, projectPoint, screenRay, rayAxis, rayPlane, dot, clipCameraPolygon, zoomCamera, lookFromEye, moveEditingCamera, type EditingView, type Vector3, type EditorCamera } from '../model/editorCamera';
import type { Selection } from './MapCanvas';
import './SpatialViewport.css';

interface Props {
  document: LevelDocument; view: EditingView; layer: 'lower' | 'upper'; grid: number; showObjects: boolean;
  tool: 'select' | 'pan' | 'room' | 'box' | 'object'; placementLabel?: string;
  selection: Selection; selectedRoom?: SourceRoom; roomGroup: readonly SourceRoom[]; objectIds: readonly number[];
  onSelect: (selection: Selection) => void; onRoomSelect: (id: string, add?: boolean) => void; onRoomBoxSelect:(ids:readonly string[],add:boolean)=>void;
  onObjectToggle: (id: number) => void; onEdit: (command: EditCommand) => void; onError: (error: unknown) => void;
  onPlace: (room: string | undefined, x: number, z: number, upper?: boolean) => void;
  onCursor: (x: number, z: number) => void; fitKey: number; frameSelection: number;
  focusRoom?: { id: string; request: number; frame?: boolean }; focusZone?: { id: number; request: number };
}
type Handle = { id: string; label: string; position: Vector3; axis?: Vector3; change: SpatialEdit; colour: string };
type Drag = { pointer: number; startX: number; startY: number; camera: EditorCamera } &
  ({ kind: 'pan' } | { kind: 'orbit' } | {kind:'select';add:boolean} | { kind: 'edit'; change: SpatialEdit; origin: Vector3; axis?: Vector3; normal: Vector3; start: Vector3 | number });
const vertical = { x: 0, y: 1, z: 0 };
const axes = [{ id: 'X', vector: { x: 1, y: 0, z: 0 }, colour: '#f28c83' }, { id: 'Y', vector: { x: 0, y: -1, z: 0 }, colour: '#b2de7a' }, { id: 'Z', vector: { x: 0, y: 0, z: 1 }, colour: '#82bfff' }];

export function SpatialViewport(props: Props) {
  const { document, view, layer, grid, tool, showObjects, selection, selectedRoom, onSelect, onRoomSelect, onEdit, onError } = props;
  const holder = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null), drag = useRef<Drag | null>(null), space = useRef(false);
  const [size, setSize] = useState({ width: 0, height: 0 }), [camera, setCamera] = useState(() => cameraFor(view));
  const [draft, setDraft] = useState<SpatialEdit | null>(null), [ceilings, setCeilings] = useState(true), [wireframe, setWireframe] = useState(false);
  const [walking,setWalking]=useState(false), movementKeys=useRef(new Set<string>());
  const [selectionBox,setSelectionBox]=useState<{start:{x:number;y:number};end:{x:number;y:number}}|null>(null);
  const orbit = view === 'orthographic' || view === 'perspective';
  const textureCanvas = useRef<HTMLCanvasElement>(null), renderer = useRef<SpatialRenderer | undefined>(undefined);
  const [cutaway, setCutaway] = useState(true), [materialError, setMaterialError] = useState(''), [rendererError,setRendererError]=useState(''), [retry, setRetry] = useState(0);
  const [materials, setMaterials] = useState<ReadonlyMap<string,IndexedMaterial>>();
  const [picked, setPicked] = useState<TexturedFace>(), [wallPaint, setWallPaint] = useState<SurfaceMaterialTarget>(), [polygonPaint, setPolygonPaint] = useState<PolygonMaterialTarget>();

  const preview = useMemo(() => {
    if (!draft) return { document, error: '' };
    try { return { document: applyCommand(document, spatialEdit(document, draft)), error: '' }; }
    catch (error) { return { document, error: error instanceof Error ? error.message : String(error) }; }
  }, [document, draft]);
  const mesh = useMemo(() => {
    if (!orbit) return { scene: undefined, error: '' };
    try { return { scene: texturedEditorScene(preview.document), error: '' }; }
    catch(error) { return { scene: undefined, error: error instanceof Error ? error.message : String(error) }; }
  }, [preview.document, orbit]);
  const materialKeys = mesh.scene ? [...new Set(mesh.scene.faces.map(f=>f.material))].sort().join(',') : '';
  const materialReady=!!materials && !!mesh.scene && mesh.scene.faces.every(face=>materials.has(face.material));
  useEffect(() => {
    if (!orbit || !mesh.scene) return;
    let active=true; setMaterialError(''); setMaterials(undefined);
    loadSpatialMaterials(mesh.scene.faces,preview.document).then(result=>{if(active)setMaterials(result);}).catch(error=>{if(active)setMaterialError(String(error.message??error));});
    return ()=>{active=false;};
  },[orbit,materialKeys,preview.document.wallOverrides,retry]);
  useLayoutEffect(() => {
    if(!orbit || !textureCanvas.current)return;
    setRendererError('');
    try { renderer.current=new SpatialRenderer(textureCanvas.current); }
    catch(error){setRendererError(error instanceof Error?error.message:String(error));}
    const lost=(event:Event)=>{event.preventDefault();setRendererError('The textured editing graphics context was lost. Retry the 3D renderer; your map is kept.');};
    const node=textureCanvas.current;node.addEventListener('webglcontextlost',lost);
    return ()=>{node.removeEventListener('webglcontextlost',lost);renderer.current?.dispose();renderer.current=undefined;};
  },[orbit,retry]);
  useLayoutEffect(()=>{
    if(!renderer.current || !mesh.scene || !materials || !materialReady || !size.width || !size.height)return;
    try { renderer.current.setScene(mesh.scene.faces,materials);renderer.current.draw(camera,size.width,size.height,ceilings,cutaway,wireframe); }
    catch(error){setRendererError(error instanceof Error?error.message:String(error));}
  },[mesh.scene,materials,materialReady,camera,size,ceilings,cutaway,wireframe,retry]);
  useEffect(()=>{setPicked(undefined);setWallPaint(undefined);setPolygonPaint(undefined);},[document]);
  const scene = useMemo(() => editorScene(preview.document, orbit ? 'both' : layer === 'upper'), [preview.document, layer, orbit]);
  const faces = useMemo(() => (mesh.scene?.faces ?? scene.faces).filter(face => ceilings || face.kind !== 'ceiling'), [scene, mesh.scene, ceilings]);
  const pickFaces = (x:number,y:number, floorOnly=false) => { const source=ray(x,y); return pickSceneFace(faces.filter(face=>(!floorOnly || face.kind==='floor') && (!orbit || !cutaway || !('normal' in face) || dot((face as TexturedFace).normal,source.direction)<0)),source); };
  const markers = useMemo(() => showObjects ? scene.markers : scene.markers.filter(m => m.kind === 'start'), [scene, showObjects]);
  const points = useMemo(() => [...scene.faces.flatMap(face => [...face.points]), ...markers.map(marker => marker.position)], [scene, markers]);
  const selectedMarker = markers.find(marker => marker.kind === selection?.kind && marker.id === selection.id);
  const sourceRoom = preview.document.roomSource.rooms.find(room => room.id === selectedRoom?.id);
  const project = (point: Vector3) => projectPoint(point, camera, size.width, size.height);
  const ray = (x: number, y: number, source = camera) => screenRay(x, y, source, size.width, size.height);
  const cancel = () => { if (drag.current?.kind === 'pan' || drag.current?.kind === 'orbit') setCamera(drag.current.camera); drag.current = null; setDraft(null);setSelectionBox(null); };
  useLayoutEffect(() => {
    const observer = new ResizeObserver(([entry]) => { if (entry) setSize({ width: entry.contentRect.width, height: entry.contentRect.height }); });
    observer.observe(holder.current!); return () => observer.disconnect();
  }, []);
  const initialized = useRef('');
  useLayoutEffect(() => {
    const key = orbit ? view : `${view}:${layer}`;
    if (size.width < 100 || size.height < 100 || initialized.current === key) return;
    const projectionChanged = initialized.current.split(':')[0] !== view; initialized.current = key;
    const roomId = selectedRoom?.id ?? selectedMarker?.room;
    const roomPoints = props.roomGroup.length ? scene.faces.filter(f=>props.roomGroup.some(room=>room.id===f.room)).flatMap(f=>[...f.points]) : roomId ? scene.faces.filter(f => f.room === roomId).flatMap(f => [...f.points]) : [];
    setCamera(current => frameCamera(projectionChanged ? cameraFor(view) : current, roomPoints.length ? [...roomPoints, ...(selectedMarker ? [selectedMarker.position] : [])] : points, size.width, size.height));
  }, [view, layer, size, points, selectedRoom, selectedMarker, scene]);
  const fit = useRef(props.fitKey), framed = useRef(props.frameSelection);
  useEffect(() => {
    if (fit.current !== props.fitKey) { fit.current = props.fitKey; setCamera(current => frameCamera(current, points, size.width, size.height)); }
    if (framed.current !== props.frameSelection) {
      framed.current = props.frameSelection;
      const selected = props.roomGroup.length ? scene.faces.filter(face=>props.roomGroup.some(room=>room.id===face.room)).flatMap(face=>[...face.points]) : selectedMarker ? [selectedMarker.position] : sourceRoom ? scene.faces.filter(face => face.room === sourceRoom.id).flatMap(face => [...face.points]) : [];
      if (selected.length) setCamera(current => frameCamera(current, selected, size.width, size.height));
    }
  }, [props.fitKey, props.frameSelection, props.roomGroup, points, size, selectedMarker, sourceRoom, scene]);
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
  useEffect(() => { drag.current = null; setDraft(null);setSelectionBox(null); }, [document, tool, view, layer]);
  useEffect(() => { const blur = () => { space.current = false; drag.current = null; setDraft(null);setSelectionBox(null); }; window.addEventListener('blur', blur); return () => window.removeEventListener('blur', blur); }, []);

  useEffect(()=>{if(view!=='perspective')setWalking(false);},[view]);
  useEffect(()=>{
    if(!walking)return;
    let previous:number|undefined,frame=0;
    const clear=()=>movementKeys.current.clear();
    const move=(time:number)=>{
      const seconds=previous===undefined?0:(time-previous)/1000;previous=time;
      if(movementKeys.current.size){const key=(code:string)=>movementKeys.current.has(code)?1:0;
        setCamera(current=>moveEditingCamera(current,key('KeyW')-key('KeyS'),key('KeyD')-key('KeyA'),key('KeyQ')-key('KeyE'),seconds,!!(key('ShiftLeft')||key('ShiftRight'))));}
      frame=requestAnimationFrame(move);
    };
    frame=requestAnimationFrame(move);window.addEventListener('blur',clear);
    return()=>{cancelAnimationFrame(frame);window.removeEventListener('blur',clear);clear();};
  },[walking]);
  function walkFromStart(){
    const authored=document.sourceStarts?.find(s=>s.player===1),start=authored??document.starts.find(s=>s.player===1);
    const room=authored?document.roomSource.rooms.find(r=>r.id===authored.room):document.roomSource.rooms.find(r=>r.originZone===document.starts.find(s=>s.player===1)?.zone);
    if(start && room){const settings=roomSettings(document,room),height=settings.floor-settings.ceiling;
      // preview.c / PLAYER_HEIGHT: 12*1024 in 24.8 world Y, or 48 units.
      const eye={x:start.x,z:start.z,y:settings.floor-(height>48?48:height/2)};
      setCamera({...cameraFor('perspective'),yaw:0,pitch:0,distance:128,target:{...eye,z:eye.z+128}});
    }
    setCeilings(true);setWalking(true);canvas.current?.focus();
  }
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
  const heightRooms = props.roomGroup.length > 1 ? props.roomGroup : sourceRoom ? [sourceRoom] : [];
  const heightRoom = preview.document.roomSource.rooms.find(room => room.id === heightRooms.at(-1)?.id);
  if (tool === 'select' && heightRoom) {
    const settings = roomSettings(preview.document, heightRoom), floor = settings[layer === 'upper' ? 'upperFloor' : 'floor'];
    const centre = { x: heightRoom.vertices.reduce((sum, p) => sum + p.x, 0) / heightRoom.vertices.length, z: heightRoom.vertices.reduce((sum, p) => sum + p.z, 0) / heightRoom.vertices.length };
    const storeys = orbit && settings.upper.length && settings.upperFloor > settings.upperCeiling ? [false, true] : [layer === 'upper'];
    for (const upper of storeys.filter(upper => !upper || heightRooms.every(room => { const s = roomSettings(preview.document, room); return s.upper.length > 0 && s.upperFloor > s.upperCeiling; }))) for (const field of ['floor', 'ceiling'] as const) {
      const setting = upper ? field === 'floor' ? 'upperFloor' : 'upperCeiling' : field;
      const separation = orbit && storeys.length === 2 ? (upper ? 1 : -1) * Math.min(camera.distance / 20, 32) : 0;
      const position = add({ ...centre, y: settings[setting] }, mul(cameraBasis(camera).right, separation));
      handles.push({ id: orbit ? `${upper ? 'upper' : 'lower'}-${field}` : field,
        label: orbit ? `Drag ${upper ? 'upper' : 'lower'} ${field} height` : `Drag room ${field} height`, position, axis: vertical,
        change: { kind: 'height', room: heightRoom.id, rooms: heightRooms.map(room => room.id), field: setting, value: position.y }, colour: field === 'floor' ? '#b2de7a' : '#e6bd81' });
    }
    sourceRoom?.vertices.forEach((position, index) => handles.push({ id: `corner-${index}`, label: `Move room corner ${index + 1}`, position: { ...position, y: floor }, axis: view === 'left' || view === 'right' ? axes[2]!.vector : view === 'front' || view === 'back' ? axes[0]!.vector : undefined, colour: '#b2de7a',
      change: { kind: 'corner', room: sourceRoom.id, index, position: { ...position, y: floor } } }));
  }
  useLayoutEffect(() => {
    const node = canvas.current, context = node?.getContext('2d'); if (!node || !context || !size.width || !size.height) return;
    const ratio = window.devicePixelRatio || 1;
    node.width = Math.round(size.width * ratio); node.height = Math.round(size.height * ratio); context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0,0,size.width,size.height);
    if(!orbit){context.fillStyle = '#161c24'; context.fillRect(0, 0, size.width, size.height);}
    const projected = faces.map(face => ({ face, pixels: clipCameraPolygon(face.points, camera).map(project) })).filter(face => face.pixels.length >= 3);
    projected.sort((a, b) => b.pixels.reduce((s, p) => s + p.depth, 0) / b.pixels.length - a.pixels.reduce((s, p) => s + p.depth, 0) / a.pixels.length);
    for (const { face, pixels } of projected) {
      const selected = face.room === selectedRoom?.id || props.roomGroup.some(room=>room.id===face.room);
      context.beginPath(); pixels.forEach((point, i) => i ? context.lineTo(point.x, point.y) : context.moveTo(point.x, point.y)); context.closePath();
      if (!orbit && !wireframe) { context.fillStyle = face.kind === 'wall' ? selected ? '#b2de7a16' : '#718aa008' : selected ? '#344b37' : face.kind === 'ceiling' ? '#3a3b45' : '#263441'; context.fill(); }
      context.strokeStyle = selected ? '#b2de7a88' : face.kind === 'wall' ? '#667d8c60' : '#5c758766'; context.lineWidth = selected ? 1.5 : .7; if ((!orbit && face.kind === 'wall') || wireframe || selected || orbit && (!mesh.scene || materialError || rendererError || !materials)) context.stroke();
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
  function startEdit(event: ReactPointerEvent, change: SpatialEdit, explicitAxis?: Vector3, anchor?: Vector3) {
    const point = local(event), origin = anchor ?? (change.kind === 'height' ? handles.find(h => h.change.kind === 'height' && h.change.field === change.field)!.position : change.position);
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
  const built=mesh.scene?.built;
  const pickIdentity=picked?{offset:picked.graphic.offset,type:picked.graphic.type,upper:picked.upper}:undefined;
  const wallTarget=built && picked && pickIdentity ? surfaceMaterialTarget(built,picked.zone,pickIdentity):undefined;
  const polygonTarget=built && picked && pickIdentity ? polygonMaterialTarget(built,picked.zone,pickIdentity):undefined;
  return <div ref={holder} className="map-holder spatial-viewport">
    {wallPaint && <WallPaintDialog document={document} target={wallPaint} onCancel={()=>setWallPaint(undefined)} onApply={command=>{onEdit(command);setWallPaint(undefined);}} />}
    {polygonPaint && <PolygonPaintDialog document={document} target={polygonPaint} onCancel={()=>setPolygonPaint(undefined)} onApply={command=>{onEdit(command);setPolygonPaint(undefined);}} />}
    {orbit && <canvas key={retry} ref={textureCanvas} className="spatial-textures" aria-label="Textured 3D geometry" data-textured-faces={materialReady?mesh.scene?.faces.length:0} data-material-count={materialReady?materials?.size:0} data-storeys="lower upper" style={{width:size.width,height:size.height,visibility:mesh.error||materialError||rendererError||!materialReady?'hidden':'visible'}} />}
    <canvas ref={canvas} aria-label="Editing viewport" aria-describedby="spatial-controls" data-view={view} data-camera={JSON.stringify(camera)} data-size={JSON.stringify(size)} tabIndex={0}
      style={{ width: size.width, height: size.height, cursor: tool === 'object' ? 'crosshair' : 'default' }}
      onContextMenu={event => event.preventDefault()}
      onBlur={()=>movementKeys.current.clear()}
      onKeyDown={event => { if(walking && !event.ctrlKey && !event.metaKey && !event.altKey && ['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ShiftLeft','ShiftRight'].includes(event.code)){event.preventDefault();movementKeys.current.add(event.code);return;} if(event.key==='Escape')movementKeys.current.clear(); if (event.key === ' ') { space.current = true; event.preventDefault(); } if (event.key === 'Escape' && drag.current) { event.preventDefault(); event.stopPropagation(); cancel(); } }}
      onKeyUp={event => { movementKeys.current.delete(event.code); if (event.key === ' ') space.current = false; }}
      onWheel={event => { const p = local(event); setCamera(current => zoomCamera(current, p.x, p.y, Math.exp(Math.max(-1, Math.min(1, event.deltaY * .001))), size.width, size.height)); }}
      onPointerDown={event => {
        event.currentTarget.focus(); const p = local(event);
        if (event.button !== 0) { setSelectionBox(null);setDraft(null); drag.current = { kind: orbit && event.button === 2 ? 'orbit' : 'pan', pointer: event.pointerId, startX: p.x, startY: p.y, camera }; event.currentTarget.setPointerCapture(event.pointerId); return; }
        const hit = pickFaces(p.x,p.y,tool === 'object' && (orbit || view === 'bottom'));
        if (tool === 'object') { const step = event.altKey || grid === 0 ? 1 : grid; props.onPlace(hit?.face.room, Math.round((hit?.position.x ?? 0) / step) * step, Math.round((hit?.position.z ?? 0) / step) * step, hit?.face.upper); return; }
        drag.current={kind:'select',pointer:event.pointerId,startX:p.x,startY:p.y,camera,add:event.shiftKey||event.ctrlKey||event.metaKey};
        event.currentTarget.setPointerCapture(event.pointerId);

      }}
      onPointerMove={event => {
        const p = local(event), current = drag.current;
        const hit = pickFaces(p.x,p.y); if (hit) props.onCursor(hit.position.x, hit.position.z);
        if (!current) return;
        const dx = p.x - current.startX, dy = p.y - current.startY;
        if(current.kind==='select'){if(Math.hypot(dx,dy)>4)setSelectionBox({start:{x:current.startX,y:current.startY},end:p});return;}
        if (current.kind === 'orbit') { setCamera(walking ? lookFromEye(current.camera,current.camera.yaw-dx*.008,current.camera.pitch+dy*.008) : { ...current.camera, yaw: current.camera.yaw - dx * .008, pitch: Math.max(-1.5, Math.min(1.5, current.camera.pitch + dy * .008)) }); return; }
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
        if(current?.kind==='select'){
          const p=local(event),start={x:current.startX,y:current.startY};
          if(Math.hypot(p.x-start.x,p.y-start.y)>4){
            const eye=cameraBasis(current.camera).eye,visible=faces.filter(face=>!cutaway||!orbit||!('normal' in face)||dot((face as TexturedFace).normal,sub(eye,face.points[0]!))>0);
            props.onRoomBoxSelect(spatialRoomsInSelectionBox(visible,current.camera,size.width,size.height,start,p),current.add);
          }else{
            const hit=pickFaces(start.x,start.y),marker=markerAt(start.x,start.y),zones=current.add&&(!!selectedRoom||props.roomGroup.length>0||selection?.kind!=='object');
            if(marker && !zones){if(marker.kind==='object'&&current.add)props.onObjectToggle(marker.id);else onSelect({kind:marker.kind,id:marker.id});}
            else if(hit){onRoomSelect(hit.face.room,current.add);if('graphic' in hit.face)setPicked(hit.face as TexturedFace);}
            else if(zones&&marker)onRoomSelect(marker.room,true);
            else if(!current.add)onSelect(null);
          }
          setSelectionBox(null);if(event.currentTarget.hasPointerCapture(event.pointerId))event.currentTarget.releasePointerCapture(event.pointerId);return;
        }
        if (current?.kind === 'edit' && draft) { try { onEdit(spatialEdit(document, draft)); } catch (error) { onError(error); } }
        setDraft(null); if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={cancel} onLostPointerCapture={() => { drag.current = null; setDraft(null);setSelectionBox(null); }} />
    {selectionBox && <div className="zone-selection-box" aria-label="Zone selection rectangle" style={{left:Math.min(selectionBox.start.x,selectionBox.end.x),top:Math.min(selectionBox.start.y,selectionBox.end.y),width:Math.abs(selectionBox.end.x-selectionBox.start.x),height:Math.abs(selectionBox.end.y-selectionBox.start.y)}} />}
    {!faces.length && <div className="spatial-empty" role="status">{layer === 'upper' ? 'No upper-floor rooms are visible. Choose Lower to edit the lower floor.' : 'No rooms yet. Use Draw room in Top view to create one.'}</div>}
    <div className="map-label">EDITING / {view.toUpperCase()} / {orbit ? 'LOWER + UPPER' : layer.toUpperCase()}</div>
    <div className="spatial-options">{view==='perspective' && <><button aria-pressed={!walking} title="Orbit and frame editable geometry; right drag rotates around the selection." onClick={()=>{setWalking(false);setCamera(frameCamera(cameraFor(view),sourceRoom?scene.faces.filter(f=>f.room===sourceRoom.id).flatMap(f=>[...f.points]):points,size.width,size.height));}}>Orbit</button><button aria-pressed={walking} title="Inspect from player one inside this same editable 3D view. WASD moves, right drag looks, Q/E moves down/up and Shift moves faster. Does not run the game." onClick={walkFromStart}>Walk</button></>}{orbit && <label title="Cut away outward-facing surfaces to see inside both storeys without changing the level."><input type="checkbox" checked={cutaway} onChange={event=>setCutaway(event.target.checked)} />Cutaway</label>}<label title="Show ceiling surfaces in this editing view. Hiding them lets you look inside; it does not remove ceilings from the game."><input type="checkbox" checked={ceilings} onChange={event => setCeilings(event.target.checked)} />Show ceilings</label><label title="Show room outlines without filled surfaces."><input type="checkbox" checked={wireframe} onChange={event => setWireframe(event.target.checked)} />Wireframe</label></div>
    {handles.toSorted((a, b) => project(b.position).depth - project(a.position).depth).map(handle => { const p = project(handle.position); return p.visible && p.x > 5 && p.x < size.width - 5 && p.y > 5 && p.y < size.height - 5 ? <button key={handle.id} className={`spatial-handle ${handle.id.startsWith('corner') ? 'spatial-corner' : ''}`} aria-label={handle.label} title={`${handle.label}.${handle.change.kind === 'height' && heightRooms.length > 1 ? ` Moves all ${heightRooms.length} selected zones by the same amount; preserves their height differences.` : ''} Drag to preview; release applies one edit. Escape cancels. Arrow keys nudge; Shift increases the step. Y increases downwards.`}
      style={{ left: p.x, top: p.y, borderColor: handle.colour, color: handle.colour }} onPointerDown={event => { event.preventDefault(); startEdit(event, handle.change, handle.axis); }}
      onKeyDown={event => { if (['ArrowUp', 'ArrowRight', 'ArrowDown', 'ArrowLeft'].includes(event.key)) { event.preventDefault(); event.stopPropagation(); nudge(handle, (event.key === 'ArrowUp' || event.key === 'ArrowRight' ? 1 : -1) * (handle.change.kind === 'height' ? -1 : 1), event.shiftKey); } }}>
      {handle.id.startsWith('axis-') ? handle.id.slice(5) : handle.change.kind === 'height' ? `${handle.change.field.startsWith('upper') ? 'U' : 'L'}${handle.change.field.toLowerCase().endsWith('floor') ? 'F' : 'C'}` : ''}</button> : null; })}
    <div id="spatial-controls" className="map-help">{tool === 'object' ? `Placing ${props.placementLabel}: click a room; Escape cancels.` : `${walking ? 'WASD: move. Right drag: look. Q/E: down/up. Shift: faster. ' : orbit ? 'Right drag: orbit. ' : ''}${orbit ? 'LF/LC/UF/UC: drag heights. ' : ''}Middle drag: pan. Wheel: zoom. Left click/drag: select zones. Shift: add zones. Handles: move/edit. F: fit. Shift+F: selection.`}</div>
    {orbit && (mesh.error || materialError || rendererError) && <div className="spatial-material-error" role="alert">{mesh.error || materialError || rendererError} {mesh.error ? 'Geometry outlines remain editable. Apply the room appearance to enable its texture preview.' : <button onClick={()=>setRetry(n=>n+1)}>Retry 3D materials</button>}</div>}
    {orbit && !mesh.error && !materialError && !rendererError && !materialReady && <div className="spatial-material-error" role="status">Loading original textures and lighting...</div>}
    {orbit && materialReady && <span className="spatial-render-status" role="status">Textured geometry / Lower + Upper{mesh.scene?.unsupported.length ? ` / ${mesh.scene.unsupported.join(', ')}: use Engine view for these animated graphics` : ''}</span>}
    {picked && orbit && <div className="spatial-picked"><strong>{picked.upper?'Upper':'Lower'} {picked.kind}</strong>{wallTarget && <button onClick={()=>setWallPaint(wallTarget)}>Paint this wall</button>}{polygonTarget && <button onClick={()=>setPolygonPaint(polygonTarget)}>Paint this surface</button>}<button aria-label="Dismiss picked surface" onClick={()=>setPicked(undefined)}>Close</button></div>}
    {preview.error && <div className="spatial-validation" role="status">{preview.error} Release rejects the edit; Escape cancels.</div>}
  </div>;
}
