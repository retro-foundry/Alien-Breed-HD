import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { zoneContainsPoint, type LevelDocument, type Zone } from '../model/level';
import { Icon } from './Icon';
import { pointInRoomOutline, normaliseRoomOutline } from '../model/geometry';
import { importRoomSource, type RoomDraft, type SourceRoom } from '../model/roomSource';
import { editableStarts } from '../model/sourceStarts';
import { editableObjects, objectIsActive } from '../model/sourceObjects';
import { editableTeleport } from '../model/sourceTeleports';
import { VisibilityOverlay, useVisibilityInspection } from './VisibilityOverlay';
import { objectsInRectangle } from '../model/objectGroups';
import { cornerHandleHalfSize, objectMarkerRadius, selectedObjectRadius, selectedObjectRingRadius, objectMarkerStroke, startLabelFont, startLabelOffset, startMarkerPoints, startMarkerHit } from './mapMarkers';

export type Selection = { kind: 'zone' | 'object' | 'start' | 'point'; id: number } | null;
export interface MapView { x: number; z: number; scale: number }
interface Props {
  document: LevelDocument; selection: Selection; onSelect: (selection: Selection) => void;
  objectIds: readonly number[]; onObjectToggle: (id: number) => void;
  onObjectBoxSelect: (ids: readonly number[], add: boolean) => void;
  onGroupMove: (ids: readonly number[], x: number, z: number) => void;
  frameSelection: number;
  placementLabel?: string; onObjectPlace: (room: string | undefined, x: number, z: number) => void;
  onMove: (selection: Exclude<Selection, null>, x: number, z: number) => void;
  tool: 'select' | 'pan' | 'room' | 'box' | 'object'; layer: 'lower' | 'upper'; grid: number; showObjects: boolean;
  roomDraft: RoomDraft | null; selectedRoom?: SourceRoom; roomGroup: readonly SourceRoom[]; onRoomSelect: (id: string, add?: boolean) => void;
  onRoomPoint: (x: number, z: number) => void; onRoomFinish: () => void;
  onRoomCornerMove: (index: number, x: number, z: number) => void;
  focusRoom?: { id: string; request: number; frame?: boolean };
  onCursor: (x: number, z: number) => void; focusZone?: { id: number; request: number }; fitKey: number;
}

function floorPolygons(zone: Zone, upper: boolean) {
  return (upper ? zone.upper : zone.lower).filter(graphic => [1, 8, 9, 10, 11].includes(graphic.type) && graphic.points.length >= 3);
}

export function MapCanvas({ placementLabel, onObjectPlace, document, selection, objectIds, frameSelection, onObjectToggle, onObjectBoxSelect, onGroupMove, onSelect, onMove, tool, layer, grid, showObjects, onCursor, focusZone, fitKey, roomDraft, selectedRoom, roomGroup, onRoomSelect, onRoomPoint, onRoomFinish, onRoomCornerMove, focusRoom }: Props) {
  const visibility = useVisibilityInspection(document, selectedRoom?.id, selection?.kind === 'zone' ? selection.id : undefined);
  const canvas = useRef<HTMLCanvasElement>(null), holder = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const lastFit = useRef<{ data: Uint8Array; fit: number } | null>(null);
  const [view, setView] = useState<MapView>({ x: 0, z: 0, scale: 0.15 });
  const space = useRef(false);
  const [placementPoint, setPlacementPoint] = useState<{ x: number; z: number } | null>(null);
  const drag = useRef<{ x: number; y: number; view: MapView; selection: Selection; moved: boolean } | null>(null);
  const [ghost, setGhost] = useState<{ selection: Exclude<Selection, null>; x: number; z: number } | null>(null);
  const groupDrag = useRef<readonly number[] | null>(null);
  const [groupGhost, setGroupGhost] = useState<{ ids: readonly number[]; x: number; z: number } | null>(null);
  const boxDrag = useRef<{ start: { x: number; z: number }; end: { x: number; z: number }; add: boolean } | null>(null);
  const [boxGhost, setBoxGhost] = useState<typeof boxDrag.current>(null);
  useEffect(() => { boxDrag.current = null; setBoxGhost(null); groupDrag.current = null; setGroupGhost(null); drag.current = null; setGhost(null); }, [document, tool, layer]);
  useEffect(() => {
    if (groupDrag.current && (groupDrag.current.length !== objectIds.length || groupDrag.current.some((id, index) => id !== objectIds[index]))) {
      groupDrag.current = null; setGroupGhost(null); drag.current = null;
    }
  }, [objectIds]);
  const cornerDrag = useRef<{ index: number } | null>(null);
  const [cornerGhost, setCornerGhost] = useState<{ index: number; x: number; z: number } | null>(null);
  useEffect(() => { cornerDrag.current = null; setCornerGhost(null); }, [selectedRoom?.id, tool, document.roomSource]);
  const sourceView = useMemo(() => {
    const baseline = new Map(importRoomSource(document).rooms.map(room => [room.id, room]));
    const changed = document.roomSource.rooms.filter(room => JSON.stringify(room) !== JSON.stringify(baseline.get(room.id)));
    const visibleOrigins = new Set(document.roomSource.rooms.map(room => room.originZone));
    const hidden = new Set(document.zones.filter(zone => !visibleOrigins.has(zone.id) || changed.some(room => room.originZone === zone.id)).map(zone => zone.id));
    return { changed, hidden };
  }, [document]);
  const screen = useCallback((x: number, z: number) => ({ x: size.width / 2 + (x - view.x) * view.scale, y: size.height / 2 - (z - view.z) * view.scale }), [view, size]);
  const world = useCallback((x: number, y: number) => ({ x: view.x + (x - size.width / 2) / view.scale, z: view.z - (y - size.height / 2) / view.scale }), [view, size]);
  const framePoints = useCallback((points: readonly { x: number; z: number }[]) => {
    if (!points.length || size.width <= 100 || size.height <= 100) return;
    const xs = points.map(point => point.x), zs = points.map(point => point.z);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
    setView({ x: (minX + maxX) / 2, z: (minZ + maxZ) / 2, scale: Math.min(8, (size.width - 100) / Math.max(200, maxX - minX), (size.height - 100) / Math.max(200, maxZ - minZ)) });
  }, [size]);


  useLayoutEffect(() => {
    const node = holder.current;
    if (!node) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (size.width < 120 || size.height < 120) return;
    if (lastFit.current?.data === document.encoded.data && lastFit.current.fit === fitKey) return;
    const points = [...document.points, ...document.roomSource.rooms.flatMap(room => [...room.vertices])];
    if (!points.length) return;
    const xs = points.map(point => point.x), zs = points.map(point => point.z);
    const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
    setView({ x: (minX + maxX) / 2, z: (minZ + maxZ) / 2, scale: Math.min((size.width - 100) / Math.max(200, maxX - minX), (size.height - 100) / Math.max(200, maxZ - minZ)) });
    lastFit.current = { data: document.encoded.data, fit: fitKey };
    // Fit when opening a map or explicitly requested; ordinary edits keep the camera.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [document.encoded.data, fitKey, size.width, size.height]);

  useEffect(() => {
    if (focusZone === undefined) return;
    const zone = document.zones.find(zone => zone.id === focusZone.id);
    if (!zone) return;
    const lines = zone.boundaryIds.map(id => document.lines[id]).filter(line => !!line);
    if (lines.length) setView(view => ({ ...view, x: lines.reduce((sum, line) => sum + line.x, 0) / lines.length, z: lines.reduce((sum, line) => sum + line.z, 0) / lines.length }));
  }, [focusZone, document.encoded.data]);
  useEffect(() => {
    const room = document.roomSource.rooms.find(room => room.id === focusRoom?.id);
    if (room?.vertices.length) setView(view => {
      if (focusRoom?.frame) {
        const xs = room.vertices.map(point => point.x), zs = room.vertices.map(point => point.z);
        const minX = Math.min(...xs), maxX = Math.max(...xs), minZ = Math.min(...zs), maxZ = Math.max(...zs);
        return { x: (minX + maxX) / 2, z: (minZ + maxZ) / 2,
          scale: Math.max(0.02, Math.min(8, (size.width - 100) / Math.max(200, maxX - minX), (size.height - 100) / Math.max(200, maxZ - minZ))) };
      }
      return { ...view, x: room.vertices.reduce((sum, point) => sum + point.x, 0) / room.vertices.length,
        z: room.vertices.reduce((sum, point) => sum + point.z, 0) / room.vertices.length };
    });
  }, [focusRoom]);

  const lastSelectionFrame = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!frameSelection || size.width < 120 || size.height < 120 || lastSelectionFrame.current === frameSelection) return;
    if (roomGroup.length > 1) framePoints(roomGroup.flatMap(room => [...room.vertices]));
    else if (selection?.kind === 'object') {
      const ids = objectIds.length ? objectIds : [selection.id];
      framePoints(editableObjects(document).filter(object => ids.includes(object.id) && objectIsActive(object)));
    } else if (selection?.kind === 'start') framePoints(editableStarts(document).filter(start => start.player === selection.id));
    else if (selectedRoom) framePoints(selectedRoom.vertices);
    else if (selection?.kind === 'zone') framePoints(document.zones.find(zone => zone.id === selection.id)?.boundaryIds.map(id => document.lines[id]!) ?? []);
    else if (selection?.kind === 'point') framePoints(document.points.filter(point => point.id === selection.id));
    lastSelectionFrame.current = frameSelection;
  }, [frameSelection, size, selection, objectIds, document, selectedRoom, roomGroup, framePoints]);

  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      const control = event.target instanceof Element && event.target.closest('button, input, textarea, select, summary, a, [contenteditable="true"]');
      if (event.code === 'Space' && !control) { space.current = true; event.preventDefault(); }
      if (event.key === 'Escape') { boxDrag.current = null; setBoxGhost(null); groupDrag.current = null; setGroupGhost(null); drag.current = null; setGhost(null); cornerDrag.current = null; setCornerGhost(null); }
    };
    const up = (event: KeyboardEvent) => { if (event.code === 'Space') space.current = false; };
    const blur = () => { boxDrag.current = null; setBoxGhost(null); space.current = false; drag.current = null; setGhost(null); groupDrag.current = null; setGroupGhost(null); cornerDrag.current = null; setCornerGhost(null); };
    window.addEventListener('keydown', down); window.addEventListener('keyup', up); window.addEventListener('blur', blur);
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up); window.removeEventListener('blur', blur); };
  }, []);

  useEffect(() => {
    const node = canvas.current;
    if (!node) return;
    const wheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = node.getBoundingClientRect(), px = event.clientX - rect.left - size.width / 2, py = event.clientY - rect.top - size.height / 2;
      setView(old => {
        const scale = Math.max(0.02, Math.min(8, old.scale * Math.exp(-event.deltaY * 0.0015)));
        return { scale, x: old.x + px / old.scale - px / scale, z: old.z - py / old.scale + py / scale };
      });
    };
    node.addEventListener('wheel', wheel, { passive: false });
    return () => node.removeEventListener('wheel', wheel);
  }, [size]);

  useEffect(() => {
    const node = canvas.current, context = node?.getContext('2d');
    if (!node || !context) return;
    const ratio = window.devicePixelRatio || 1;
    node.width = Math.round(size.width * ratio); node.height = Math.round(size.height * ratio);
    context.scale(ratio, ratio);
    context.fillStyle = '#161b21'; context.fillRect(0, 0, size.width, size.height);
    const topLeft = world(0, 0), bottomRight = world(size.width, size.height);
    if (grid > 0) {
      let step = grid;
      while (step * view.scale < 18) step *= 2;
      context.lineWidth = 1;
      for (let x = Math.floor(topLeft.x / step) * step; x <= bottomRight.x; x += step) {
        const pixel = screen(x, 0).x; context.strokeStyle = x === 0 ? '#35404b' : '#242d36';
        context.beginPath(); context.moveTo(Math.round(pixel) + 0.5, 0); context.lineTo(Math.round(pixel) + 0.5, size.height); context.stroke();
      }
      for (let z = Math.floor(bottomRight.z / step) * step; z <= topLeft.z; z += step) {
        const pixel = screen(0, z).y; context.strokeStyle = z === 0 ? '#35404b' : '#242d36';
        context.beginPath(); context.moveTo(0, Math.round(pixel) + 0.5); context.lineTo(size.width, Math.round(pixel) + 0.5); context.stroke();
      }
    }
    const polygon = (indices: number[]) => {
      context.beginPath();
      indices.forEach((id, i) => { const point = document.points[id]; if (point) { const pixel = screen(point.x, point.z); if (i === 0) context.moveTo(pixel.x, pixel.y); else context.lineTo(pixel.x, pixel.y); } });
      context.closePath();
    };
    for (const zone of document.zones) {
      if (sourceView.hidden.has(zone.id)) continue;
      const selected = selection?.kind === 'zone' && selection.id === zone.id;
      const height = layer === 'upper' ? zone.upperFloor : zone.floor;
      const shade = Math.max(17, Math.min(36, 24 + height / 45));
      for (const graphic of floorPolygons(zone, layer === 'upper')) {
        polygon(graphic.points); context.fillStyle = selected ? '#34483d' : `hsl(210 14% ${shade}%)`; context.fill();
      }
      for (const graphic of (layer === 'upper' ? zone.upper : zone.lower).filter(graphic => graphic.type === 7)) {
        polygon(graphic.points); context.fillStyle = '#2e789033'; context.fill();
      }
    }
    for (const zone of document.zones) {
      if (sourceView.hidden.has(zone.id)) continue;
      const selected = selection?.kind === 'zone' && selection.id === zone.id;
      for (const id of zone.boundaryIds) {
        const line = document.lines[id]; if (!line) continue;
        const a = screen(line.x, line.z), b = screen(line.x + line.dx, line.z + line.dz);
        context.strokeStyle = selected ? '#b2de7a' : line.connect >= 0 ? '#527080' : '#89939b';
        context.lineWidth = selected ? 2.25 : line.connect >= 0 ? 1 : 1.5;
        context.setLineDash(line.connect >= 0 && !selected ? [3, 4] : []);
        context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
      }
    }
    context.setLineDash([]);
    const drawVisibility = () => { if (visibility.enabled && visibility.inspection) {
      for (const polygon of visibility.inspection.polygons) {
        context.beginPath();
        polygon.vertices.forEach((point, index) => { const pixel = screen(point.x, point.z); if (index) context.lineTo(pixel.x, pixel.y); else context.moveTo(pixel.x, pixel.y); });
        context.closePath(); context.fillStyle = polygon.origin ? '#efba5555' : '#39bfd133'; context.fill();
        context.strokeStyle = polygon.origin ? '#efba55' : '#39bfd1'; context.lineWidth = polygon.origin ? 2.5 : 1; context.stroke();
      }
      for (const line of visibility.inspection.portals) {
        const a = screen(line.x, line.z), b = screen(line.x + line.dx, line.z + line.dz);
        context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.strokeStyle = '#84e29b'; context.lineWidth = 4; context.stroke();
      }
      for (const point of visibility.inspection.clipPoints) {
        const pixel = screen(point.x, point.z);
        context.beginPath(); context.arc(pixel.x, pixel.y, 5, 0, Math.PI * 2);
        context.fillStyle = point.side === 'left' ? '#7bbaff' : '#ff9bdf'; context.fill();
        context.font = '11px monospace'; context.fillText(`${point.side === 'left' ? 'L' : 'R'}${point.id}`, pixel.x + 7, pixel.y - 7);
      }
    }
    };
    const outline = (vertices: readonly { x: number; z: number }[], fill: boolean, colour: string) => {
      context.beginPath(); vertices.forEach((point, index) => { const pixel = screen(point.x, point.z); if (index) context.lineTo(pixel.x, pixel.y); else context.moveTo(pixel.x, pixel.y); });
      if (vertices.length >= 3) context.closePath();
      if (fill) { context.fillStyle = '#36483d'; context.fill(); }
      context.strokeStyle = colour; context.lineWidth = 2; context.stroke();
      for (const point of vertices) { const pixel = screen(point.x, point.z); context.fillStyle = colour; context.fillRect(pixel.x - cornerHandleHalfSize, pixel.y - cornerHandleHalfSize, cornerHandleHalfSize * 2, cornerHandleHalfSize * 2); }
    };
    const previewVertices = (room: SourceRoom) => room.id === selectedRoom?.id && cornerGhost ? room.vertices.map((point, i) => i === cornerGhost.index ? { x: cornerGhost.x, z: cornerGhost.z } : point) : room.vertices;
    let cornerValid = true;
    if (selectedRoom && cornerGhost) try { normaliseRoomOutline(previewVertices(selectedRoom)); } catch { cornerValid = false; }
    for (const room of sourceView.changed) outline(previewVertices(room), true, room.id === selectedRoom?.id ? cornerValid ? '#d0f6a5' : '#e98980' : '#84b99d');
    if (selectedRoom && !sourceView.changed.some(room => room.id === selectedRoom.id)) outline(previewVertices(selectedRoom), false, cornerValid ? '#d0f6a5' : '#e98980');
    if (roomGroup.length > 1) for (const room of roomGroup) outline(room.vertices, false, '#d0f6a5');
    if (selectedRoom) {
      const teleport = editableTeleport(document, selectedRoom);
      if (teleport.destination !== null) {
        const target = document.roomSource.rooms.find(room => room.id === teleport.destination);
        const valid = target && pointInRoomOutline(target.vertices, teleport);
        if (target && target.id !== selectedRoom.id) outline(target.vertices, false, valid ? '#bbabef' : '#e98980');
        const from = screen(selectedRoom.vertices.reduce((sum, point) => sum + point.x, 0) / selectedRoom.vertices.length,
          selectedRoom.vertices.reduce((sum, point) => sum + point.z, 0) / selectedRoom.vertices.length);
        const to = screen(teleport.x, teleport.z), angle = Math.atan2(to.y - from.y, to.x - from.x);
        context.strokeStyle = valid ? '#bbabef' : '#e98980'; context.fillStyle = context.strokeStyle;
        context.lineWidth = 2; context.setLineDash([6, 4]); context.beginPath(); context.moveTo(from.x, from.y); context.lineTo(to.x, to.y); context.stroke(); context.setLineDash([]);
        context.beginPath(); context.arc(to.x, to.y, 10, 0, Math.PI * 2); context.stroke();
        context.beginPath(); context.moveTo(to.x, to.y); context.lineTo(to.x - 10 * Math.cos(angle - 0.4), to.y - 10 * Math.sin(angle - 0.4));
        context.lineTo(to.x - 10 * Math.cos(angle + 0.4), to.y - 10 * Math.sin(angle + 0.4)); context.closePath(); context.fill();
        context.font = '11px ui-monospace, monospace'; context.fillText('Teleport destination', to.x + 14, to.y - 12);
      }
    }
    if (roomDraft?.vertices.length) {
      let valid = roomDraft.vertices.length < 3;
      try { normaliseRoomOutline(roomDraft.vertices); valid = true; } catch { /* Incomplete gestures remain uncommitted. */ }
      outline(roomDraft.vertices, roomDraft.vertices.length >= 3, valid ? '#f0c77e' : '#e98980');
    }
    if (view.scale > 0.3) for (const point of document.points) {
      const pixel = screen(point.x, point.z); context.fillStyle = '#adb6ba'; context.fillRect(pixel.x - 1.5, pixel.y - 1.5, 3, 3);
    }
    drawVisibility();
    if (showObjects) for (const object of editableObjects(document)) {
      if (!objectIsActive(object) || (layer === 'upper') !== object.upper) continue;
      const moving = ghost?.selection.kind === 'object' && ghost.selection.id === object.id;
      const grouped = groupGhost?.ids.includes(object.id) ? groupGhost : null;
      const pixel = screen(grouped ? object.x + grouped.x : moving ? ghost.x : object.x, grouped ? object.z + grouped.z : moving ? ghost.z : object.z);
      const selected = objectIds.includes(object.id) || selection?.kind === 'object' && selection.id === object.id;
      const enemy = [0, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19].includes(object.type);
      context.fillStyle = selected ? '#d1f5a5' : enemy ? '#de847c' : object.type === -1 ? '#85929d' : '#efbf6a';
      context.strokeStyle = '#151a20'; context.lineWidth = objectMarkerStroke; context.beginPath(); context.arc(pixel.x, pixel.y, selected ? selectedObjectRadius : objectMarkerRadius, 0, Math.PI * 2); context.fill(); context.stroke();
      if (selected) { context.strokeStyle = '#d1f5a5'; context.beginPath(); context.arc(pixel.x, pixel.y, selectedObjectRingRadius, 0, Math.PI * 2); context.stroke(); }
    }
    for (const start of editableStarts(document)) {
      const moving = ghost?.selection.kind === 'start' && ghost.selection.id === start.player;
      const pixel = screen(moving ? ghost.x : start.x, moving ? ghost.z : start.z);
      context.fillStyle = start.player === 1 ? '#b2de7a' : '#9ec7f0'; context.strokeStyle = '#161b21'; context.lineWidth = 2;
      context.beginPath(); startMarkerPoints(pixel.x, pixel.y).forEach((point, index) => { if (index) context.lineTo(point.x, point.z); else context.moveTo(point.x, point.z); }); context.closePath(); context.fill(); context.stroke();
      context.font = startLabelFont; context.fillText(`P${start.player}`, pixel.x + startLabelOffset.x, pixel.y + startLabelOffset.y);
    }
    if (selection?.kind === 'zone') {
      const zone = document.zones.find(zone => zone.id === selection.id);
      if (zone) {
        const lines = zone.boundaryIds.map(id => document.lines[id]).filter(line => !!line);
        if (lines.length) {
          const point = screen(lines.reduce((sum, line) => sum + line.x, 0) / lines.length, lines.reduce((sum, line) => sum + line.z, 0) / lines.length);
          context.font = '11px ui-monospace, monospace'; const text = `ZONE ${zone.id}`, width = context.measureText(text).width;
          context.fillStyle = '#b2de7a'; context.fillRect(point.x - width / 2 - 7, point.y - 12, width + 14, 21); context.fillStyle = '#172117'; context.fillText(text, point.x - width / 2, point.y + 2);
        }
      }
    }
    if (tool === 'object' && placementPoint) {
      const point = screen(placementPoint.x, placementPoint.z);
      context.strokeStyle = '#b2de7a'; context.lineWidth = 2;
      context.beginPath(); context.arc(point.x, point.y, 8, 0, Math.PI * 2);
      context.moveTo(point.x - 13, point.y); context.lineTo(point.x + 13, point.y);
      context.moveTo(point.x, point.y - 13); context.lineTo(point.x, point.y + 13); context.stroke();
    }
    if (boxGhost) {
      const a = screen(boxGhost.start.x, boxGhost.start.z), b = screen(boxGhost.end.x, boxGhost.end.z);
      context.fillStyle = '#82caff22'; context.strokeStyle = '#82caff'; context.lineWidth = 1.5; context.setLineDash([5, 3]);
      context.fillRect(a.x, a.y, b.x - a.x, b.y - a.y); context.strokeRect(a.x, a.y, b.x - a.x, b.y - a.y); context.setLineDash([]);
    }
  }, [tool, placementPoint, document, selection, objectIds, view, size, grid, showObjects, layer, ghost, groupGhost, boxGhost, screen, world, sourceView, roomDraft, selectedRoom, roomGroup, cornerGhost, visibility.enabled, visibility.inspection]);

  function hit(x: number, y: number): Selection {
    const context = canvas.current!.getContext('2d');
    if (!context) return null;
    context.save(); context.font = startLabelFont;
    const start = editableStarts(document).toReversed().find(start => startMarkerHit(x, y, screen(start.x, start.z), context.measureText(`P${start.player}`)));
    context.restore();
    if (start) return { kind: 'start', id: start.player };
    if (showObjects) for (const object of editableObjects(document).toReversed()) {
      if (!objectIsActive(object) || (layer === 'upper') !== object.upper) continue;
      const selected = objectIds.includes(object.id) || selection?.kind === 'object' && selection.id === object.id;
      const radius = (selected ? selectedObjectRingRadius : objectMarkerRadius) + objectMarkerStroke / 2;
      const pixel = screen(object.x, object.z); if (Math.hypot(x - pixel.x, y - pixel.y) <= radius) return { kind: 'object', id: object.id };
    }
    const point = world(x, y);
    const zones = document.zones.filter(zone => !sourceView.hidden.has(zone.id) && zoneContainsPoint(document, zone, point.x, point.z));
    const zone = zones.find(zone => floorPolygons(zone, layer === 'upper').length > 0) ?? zones[0];
    return zone ? { kind: 'zone', id: zone.id } : null;
  }

  return <div className="map-holder" ref={holder}>
    <canvas ref={canvas} aria-label="Level map" aria-describedby="map-controls" data-world-x={view.x} data-world-z={view.z} data-world-scale={view.scale} tabIndex={0} style={{ width: size.width, height: size.height, cursor: tool === 'pan' ? 'grab' : tool === 'room' || tool === 'box' || tool === 'object' ? 'crosshair' : 'default' }}
      onPointerLeave={() => setPlacementPoint(null)}
      onContextMenu={event => event.preventDefault()}
      onKeyDown={event => {
        if (event.key === 'Escape' && (boxDrag.current || groupDrag.current || drag.current || cornerDrag.current)) {
          event.preventDefault(); event.stopPropagation();
          boxDrag.current = null; setBoxGhost(null); groupDrag.current = null; setGroupGhost(null);
          drag.current = null; setGhost(null); cornerDrag.current = null; setCornerGhost(null);
        }
      }}
      onPointerDown={event => {
        const rect = event.currentTarget.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
        const panning = event.button !== 0 || space.current || tool === 'pan';
        if (!panning && tool === 'object') {
          const point = world(x, y), step = event.altKey || grid === 0 ? 1 : grid;
          const snapped = { x: Math.round(point.x / step) * step, z: Math.round(point.z / step) * step };
          const changed = sourceView.changed.toReversed().find(room => pointInRoomOutline(room.vertices, snapped));
          const zones = document.zones.filter(zone => !sourceView.hidden.has(zone.id) && zoneContainsPoint(document, zone, snapped.x, snapped.z));
          const zone = zones.find(zone => floorPolygons(zone, layer === 'upper').length > 0) ?? zones[0];
          const room = changed ?? document.roomSource.rooms.find(room => room.originZone === zone?.id);
          event.currentTarget.focus(); onObjectPlace(room?.id, snapped.x, snapped.z); return;
        }
        if (!panning && tool === 'box') {
          const point = world(x, y); boxDrag.current = { start: point, end: point, add: event.shiftKey }; setBoxGhost(boxDrag.current);
          event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId); return;
        }
        if (!panning && tool === 'room' && roomDraft) {
          event.currentTarget.focus(); const point = world(x, y), step = event.altKey || grid === 0 ? 1 : grid;
          const first = roomDraft.vertices[0], firstPixel = first ? screen(first.x, first.z) : undefined;
          if (roomDraft.vertices.length >= 3 && firstPixel && Math.hypot(x - firstPixel.x, y - firstPixel.y) < 9) onRoomFinish();
          else onRoomPoint(Math.round(point.x / step) * step, Math.round(point.z / step) * step);
          return;
        }
        if (!panning && tool === 'select' && selectedRoom && !event.shiftKey && !event.ctrlKey && !event.metaKey) {
          const index = selectedRoom.vertices.findIndex(point => { const pixel = screen(point.x, point.z); return Math.abs(x - pixel.x) <= cornerHandleHalfSize && Math.abs(y - pixel.y) <= cornerHandleHalfSize; });
          if (index >= 0) {
            const point = selectedRoom.vertices[index]!; cornerDrag.current = { index }; setCornerGhost({ index, x: point.x, z: point.z });
            event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId); return;
          }
        }
        if (!panning && sourceView.changed.length) {
          const point = world(x, y), room = sourceView.changed.toReversed().find(room => pointInRoomOutline(room.vertices, point));
          const selected = hit(x, y);
          if (room && (!selected || selected.kind === 'zone')) { onRoomSelect(room.id, event.shiftKey || event.ctrlKey || event.metaKey); event.currentTarget.focus(); return; }
        }
        const selected = panning ? null : hit(x, y);
        if (selected?.kind === 'zone' && (event.shiftKey || event.ctrlKey || event.metaKey)) {
          const room = document.roomSource.rooms.find(room => room.originZone === selected.id);
          if (room) { onRoomSelect(room.id, true); event.currentTarget.focus(); return; }
        }
        if (selected?.kind === 'object' && (event.shiftKey || event.ctrlKey || event.metaKey)) { onObjectToggle(selected.id); event.currentTarget.focus(); return; }
        if (selected?.kind === 'object' && objectIds.length > 1 && objectIds.includes(selected.id)) {
          groupDrag.current = [...objectIds]; drag.current = { x, y, view, selection: selected, moved: false };
          event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId); return;
        }
        if (!panning) onSelect(selected);
        drag.current = { x, y, view, selection: selected, moved: false };
        event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={event => {
        const rect = event.currentTarget.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
        const point = world(x, y); onCursor(point.x, point.z);
        if (tool === 'object') { const step = event.altKey || grid === 0 ? 1 : grid; setPlacementPoint({ x: Math.round(point.x / step) * step, z: Math.round(point.z / step) * step }); }
        if (boxDrag.current) { boxDrag.current = { ...boxDrag.current, end: point }; setBoxGhost(boxDrag.current); return; }
        if (cornerDrag.current) {
          const step = event.altKey || grid === 0 ? 1 : grid;
          setCornerGhost({ index: cornerDrag.current.index, x: Math.round(point.x / step) * step, z: Math.round(point.z / step) * step }); return;
        }
        const current = drag.current;
        if (!current) return;
        const dx = x - current.x, dy = y - current.y;
        if (Math.hypot(dx, dy) > 4) current.moved = true;
        if (current.moved && groupDrag.current) {
          const step = event.altKey || grid === 0 ? 1 : grid;
          setGroupGhost({ ids: groupDrag.current, x: Math.round(dx / current.view.scale / step) * step, z: Math.round(-dy / current.view.scale / step) * step }); return;
        }
        if (!current.selection) setView({ ...current.view, x: current.view.x - dx / current.view.scale, z: current.view.z + dy / current.view.scale });
        else if (current.moved && (current.selection.kind === 'object' || current.selection.kind === 'start')) {
          const step = event.altKey || grid === 0 ? 1 : grid;
          setGhost({ selection: current.selection, x: Math.round(point.x / step) * step, z: Math.round(point.z / step) * step });
        }
      }}
      onPointerUp={event => {
        if (boxDrag.current) {
          const rect = event.currentTarget.getBoundingClientRect(), end = world(event.clientX - rect.left, event.clientY - rect.top), box = boxDrag.current;
          boxDrag.current = null; setBoxGhost(null);
          onObjectBoxSelect(showObjects ? objectsInRectangle(document, box.start, end, layer === 'upper') : [], box.add);
          if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
          return;
        }
        if (groupGhost && groupDrag.current && drag.current?.moved && (groupGhost.x !== 0 || groupGhost.z !== 0)) onGroupMove(groupGhost.ids, groupGhost.x, groupGhost.z);
        groupDrag.current = null; setGroupGhost(null);
        if (cornerDrag.current && cornerGhost) onRoomCornerMove(cornerGhost.index, cornerGhost.x, cornerGhost.z);
        cornerDrag.current = null; setCornerGhost(null);
        if (ghost && drag.current?.moved) onMove(ghost.selection, ghost.x, ghost.z);
        setGhost(null); drag.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={() => { boxDrag.current = null; setBoxGhost(null); groupDrag.current = null; setGroupGhost(null); setGhost(null); drag.current = null; cornerDrag.current = null; setCornerGhost(null); }}
      onLostPointerCapture={() => { boxDrag.current = null; setBoxGhost(null); groupDrag.current = null; setGroupGhost(null); setGhost(null); drag.current = null; cornerDrag.current = null; setCornerGhost(null); }} />
    <div className="map-label"><span className="status-dot" /> ORTHOGRAPHIC <span className="muted">/</span> {layer.toUpperCase()} FLOOR</div>
    <div className="map-compass"><span>N</span><svg viewBox="0 0 32 32" aria-hidden="true"><path d="m16 3 6 23-6-5-6 5z" fill="currentColor" /></svg></div>
    <VisibilityOverlay state={visibility} onFrame={() => framePoints(visibility.inspection?.polygons.flatMap(polygon => polygon.vertices) ?? [])} />
    <div id="map-controls" className="map-help">{tool === 'object' ? <>Placing {placementLabel} - Click to place - Escape cancels - Alt bypasses snap</> : roomDraft ? <><strong>{roomDraft.vertices.length} corners</strong> · Click corners · Enter or first corner finishes · Backspace removes last · Esc cancels · Alt bypasses snap</> : tool === 'box' ? <>Drag a rectangle · Shift adds objects · Escape cancels · Returns to Select after completion</> : <><kbd>Space</kbd> drag to pan <span>·</span> wheel to zoom <span>·</span> <kbd>F</kbd> fit <span>·</span> <kbd>?</kbd> controls</>}</div>
    <div className="map-scale"><Icon name="grid" size={14} /> {Math.round(view.scale * 100)}%</div>
  </div>;
}
