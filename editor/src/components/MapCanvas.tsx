import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { zoneContainsPoint, type LevelDocument, type Zone } from '../model/level';
import { Icon } from './Icon';

export type Selection = { kind: 'zone' | 'object' | 'start' | 'point'; id: number } | null;
export interface MapView { x: number; z: number; scale: number }
interface Props {
  document: LevelDocument; selection: Selection; onSelect: (selection: Selection) => void;
  onMove: (selection: Exclude<Selection, null>, x: number, z: number) => void;
  tool: 'select' | 'pan'; layer: 'lower' | 'upper'; grid: number; showObjects: boolean;
  onCursor: (x: number, z: number) => void; focusZone?: { id: number; request: number }; fitKey: number;
}

function floorPolygons(zone: Zone, upper: boolean) {
  return (upper ? zone.upper : zone.lower).filter(graphic => [1, 8, 9, 10, 11].includes(graphic.type) && graphic.points.length >= 3);
}

export function MapCanvas({ document, selection, onSelect, onMove, tool, layer, grid, showObjects, onCursor, focusZone, fitKey }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null), holder = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const lastFit = useRef<{ data: Uint8Array; fit: number } | null>(null);
  const [view, setView] = useState<MapView>({ x: 0, z: 0, scale: 0.15 });
  const space = useRef(false);
  const drag = useRef<{ x: number; y: number; view: MapView; selection: Selection; moved: boolean } | null>(null);
  const [ghost, setGhost] = useState<{ selection: Exclude<Selection, null>; x: number; z: number } | null>(null);
  const screen = useCallback((x: number, z: number) => ({ x: size.width / 2 + (x - view.x) * view.scale, y: size.height / 2 - (z - view.z) * view.scale }), [view, size]);
  const world = useCallback((x: number, y: number) => ({ x: view.x + (x - size.width / 2) / view.scale, z: view.z - (y - size.height / 2) / view.scale }), [view, size]);

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
    const points = document.points;
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
    const down = (event: KeyboardEvent) => { if (event.code === 'Space' && !(event.target instanceof HTMLInputElement) && !(event.target instanceof HTMLTextAreaElement)) { space.current = true; event.preventDefault(); } };
    const up = (event: KeyboardEvent) => { if (event.code === 'Space') space.current = false; };
    const blur = () => { space.current = false; };
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
    if (view.scale > 0.3) for (const point of document.points) {
      const pixel = screen(point.x, point.z); context.fillStyle = '#adb6ba'; context.fillRect(pixel.x - 1.5, pixel.y - 1.5, 3, 3);
    }
    if (showObjects) for (const object of document.objects) {
      if (object.zone < 0 || (layer === 'upper') !== object.upper) continue;
      const moving = ghost?.selection.kind === 'object' && ghost.selection.id === object.id;
      const pixel = screen(moving ? ghost.x : object.x, moving ? ghost.z : object.z);
      const selected = selection?.kind === 'object' && selection.id === object.id;
      const enemy = [0, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19].includes(object.type);
      context.fillStyle = selected ? '#d1f5a5' : enemy ? '#de847c' : object.type === -1 ? '#85929d' : '#efbf6a';
      context.strokeStyle = '#151a20'; context.lineWidth = 1.5; context.beginPath(); context.arc(pixel.x, pixel.y, selected ? 5 : 3.5, 0, Math.PI * 2); context.fill(); context.stroke();
      if (selected) { context.strokeStyle = '#d1f5a5'; context.beginPath(); context.arc(pixel.x, pixel.y, 9, 0, Math.PI * 2); context.stroke(); }
    }
    for (const start of document.starts) {
      const moving = ghost?.selection.kind === 'start' && ghost.selection.id === start.player;
      const pixel = screen(moving ? ghost.x : start.x, moving ? ghost.z : start.z);
      context.fillStyle = start.player === 1 ? '#b2de7a' : '#9ec7f0'; context.strokeStyle = '#161b21'; context.lineWidth = 2;
      context.beginPath(); context.moveTo(pixel.x, pixel.y - 9); context.lineTo(pixel.x + 7, pixel.y + 6); context.lineTo(pixel.x - 7, pixel.y + 6); context.closePath(); context.fill(); context.stroke();
      context.font = '10px ui-monospace, monospace'; context.fillText(`P${start.player}`, pixel.x + 10, pixel.y + 4);
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
  }, [document, selection, view, size, grid, showObjects, layer, ghost, screen, world]);

  function hit(x: number, y: number): Selection {
    for (const start of document.starts) { const pixel = screen(start.x, start.z); if (Math.hypot(x - pixel.x, y - pixel.y) < 13) return { kind: 'start', id: start.player }; }
    if (showObjects) for (const object of document.objects.toReversed()) {
      if (object.zone < 0 || (layer === 'upper') !== object.upper) continue;
      const pixel = screen(object.x, object.z); if (Math.hypot(x - pixel.x, y - pixel.y) < 9) return { kind: 'object', id: object.id };
    }
    const point = world(x, y);
    const zones = document.zones.filter(zone => zoneContainsPoint(document, zone, point.x, point.z));
    const zone = zones.find(zone => floorPolygons(zone, layer === 'upper').length > 0) ?? zones[0];
    return zone ? { kind: 'zone', id: zone.id } : null;
  }

  return <div className="map-holder" ref={holder}>
    <canvas ref={canvas} aria-label="Level map" tabIndex={0} style={{ width: size.width, height: size.height, cursor: tool === 'pan' ? 'grab' : 'default' }}
      onContextMenu={event => event.preventDefault()}
      onPointerDown={event => {
        const rect = event.currentTarget.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
        const panning = event.button !== 0 || space.current || tool === 'pan';
        const selected = panning ? null : hit(x, y);
        if (!panning) onSelect(selected);
        drag.current = { x, y, view, selection: selected, moved: false };
        event.currentTarget.focus(); event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={event => {
        const rect = event.currentTarget.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
        const point = world(x, y); onCursor(point.x, point.z);
        const current = drag.current;
        if (!current) return;
        const dx = x - current.x, dy = y - current.y;
        if (Math.hypot(dx, dy) > 4) current.moved = true;
        if (!current.selection) setView({ ...current.view, x: current.view.x - dx / current.view.scale, z: current.view.z + dy / current.view.scale });
        else if (current.moved && (current.selection.kind === 'object' || current.selection.kind === 'start')) {
          const step = event.altKey || grid === 0 ? 1 : grid;
          setGhost({ selection: current.selection, x: Math.round(point.x / step) * step, z: Math.round(point.z / step) * step });
        }
      }}
      onPointerUp={event => {
        if (ghost && drag.current?.moved) onMove(ghost.selection, ghost.x, ghost.z);
        setGhost(null); drag.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      }}
      onPointerCancel={() => { setGhost(null); drag.current = null; }}
      onLostPointerCapture={() => { setGhost(null); drag.current = null; }} />
    <div className="map-label"><span className="status-dot" /> ORTHOGRAPHIC <span className="muted">/</span> {layer.toUpperCase()} FLOOR</div>
    <div className="map-compass"><span>N</span><svg viewBox="0 0 32 32" aria-hidden="true"><path d="m16 3 6 23-6-5-6 5z" fill="currentColor" /></svg></div>
    <div className="map-help"><kbd>Space</kbd> drag to pan <span>·</span> scroll to zoom <span>·</span> <kbd>F</kbd> frame map</div>
    <div className="map-scale"><Icon name="grid" size={14} /> {Math.round(view.scale * 100)}%</div>
  </div>;
}
