import { useEffect, useMemo, useRef, useState } from 'react';
import type { LevelDocument } from '../model/level';
import { pasteRooms, type RoomClipboard } from '../model/roomClipboard';
import './ObjectBuilder.css';

type Validation = { document: LevelDocument; clipboard: RoomClipboard; x: string; z: string; proposal?: ReturnType<typeof pasteRooms>; error?: string };

export function RoomPaste({ document, clipboard, cursor, grid, onCancel, onPaste }: {
  document: LevelDocument; clipboard: RoomClipboard; cursor: { x: number; z: number }; grid: number;
  onCancel: () => void; onPaste: (proposal: ReturnType<typeof pasteRooms>) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const rooms = useMemo(() => clipboard.ids.map(id => clipboard.source.roomSource.rooms.find(room => room.id === id)!), [clipboard]);
  const anchor = rooms[0]!.vertices[0]!, snap = (value: number) => Math.round(value / (grid || 1)) * (grid || 1);
  const [x, setX] = useState(String(snap(cursor.x))), [z, setZ] = useState(String(snap(cursor.z)));
  const [validation, setValidation] = useState<Validation>();
  useEffect(() => { const node = dialog.current!; node.showModal(); return () => node.close(); }, []);
  useEffect(() => {
    if (!x.trim() || !z.trim()) return;
    const timer = window.setTimeout(() => {
      const input = { document, clipboard, x, z };
      try { setValidation({ ...input, proposal: pasteRooms(document, clipboard, Number(x) - anchor.x, Number(z) - anchor.z) }); }
      catch (error) { setValidation({ ...input, error: error instanceof Error ? error.message : String(error) }); }
    }, 160);
    return () => window.clearTimeout(timer);
  }, [document, clipboard, x, z, anchor.x, anchor.z]);
  const result = validation?.document === document && validation.clipboard === clipboard && validation.x === x && validation.z === z ? validation : undefined;
  const points = rooms.flatMap(room => [...room.vertices]), minX = Math.min(...points.map(point => point.x)), maxX = Math.max(...points.map(point => point.x));
  const minZ = Math.min(...points.map(point => point.z)), maxZ = Math.max(...points.map(point => point.z));
  const margin = Math.max(32, Math.max(maxX - minX, maxZ - minZ) * 0.05), marker = Math.max(maxX - minX, maxZ - minZ) / 90;
  const requested = clipboard.requestedIds?.length ?? clipboard.ids.length;
  return <dialog ref={dialog} className="modal object-builder room-paste" aria-label="Place pasted rooms" onCancel={event => { event.preventDefault(); onCancel(); }}>
    <h2>Place {rooms.length} copied {rooms.length === 1 ? 'room' : 'rooms'}</h2>
    <p>{requested} selected{rooms.length > requested ? `, plus ${rooms.length - requested} attached rooms to preserve mechanisms and materials` : ''}. Shapes, heights, objects and internal links move together. Starts and external links stay in this project. Nothing changes until Paste rooms here.</p>
    <details><summary title="Review every captured room before committing the paste. Attached rooms preserve existing motion and wall relationships.">Rooms included ({rooms.length})</summary>
      <ul>{rooms.map(room => <li key={room.id}>{room.originZone === null ? `Room ${room.id.replace(/^room-/, '')}` : `Zone ${room.originZone}`}{clipboard.requestedIds && !clipboard.requestedIds.includes(room.id) ? ' — attached' : ''}</li>)}</ul>
    </details>
    <svg role="img" aria-label="Copied room group with placement anchor marked" viewBox={`${minX - margin} ${-maxZ - margin} ${maxX - minX + margin * 2} ${maxZ - minZ + margin * 2}`} style={{ width: '100%', height: 130, background: '#151b22', marginTop: 12 }}>
      {rooms.map(room => <polygon key={room.id} points={room.vertices.map(point => `${point.x},${-point.z}`).join(' ')} fill="#b2de7a22" stroke="#b2de7a" strokeWidth={Math.max(1, marker / 3)} />)}
      <circle cx={anchor.x} cy={-anchor.z} r={Math.max(3, marker)} fill="#f4ca83" />
    </svg>
    <form onSubmit={event => { event.preventDefault(); if (result?.proposal) onPaste(result.proposal); }}>
      <p className="field-note">The marked corner of the first room goes to these coordinates. All rooms keep their spacing. Use map cursor snaps that corner to the current grid.</p>
      <div className="field-grid">{(['x', 'z'] as const).map(axis => <label className="number-field" key={axis}><span>Room anchor {axis.toUpperCase()}</span>
        <input required type="number" min={-32768} max={32767} step={1} aria-label={`Room paste anchor ${axis.toUpperCase()}`} value={axis === 'x' ? x : z}
          title="Set the destination of the marked corner in whole world units. Every room moves by the same amount. The full group is checked before applying."
          onChange={event => (axis === 'x' ? setX : setZ)(event.target.value)} /></label>)}</div>
      <button type="button" title="Place the marked corner at the last map pointer position, snapped to the selected grid. You can also enter exact whole coordinates."
        onClick={() => { setX(String(snap(cursor.x))); setZ(String(snap(cursor.z))); }}>Use map cursor</button>
      {result?.error && <p role="alert">{result.error}</p>}
      {!result && x.trim() && z.trim() && <p role="status">Checking room geometry and dependencies…</p>}
      {result?.proposal && <p role="status">Ready to paste {rooms.length} rooms and {result.proposal.objects.length} objects. One Undo removes the whole paste.</p>}
      <div className="object-builder-actions"><button type="button" title="Discard this placement without editing the project. Escape also cancels." onClick={onCancel}>Cancel room paste</button>
        <button className="primary-button" type="submit" disabled={!result?.proposal} title="Commit the validated rooms, objects, links and mechanisms as one undoable edit.">Paste rooms here</button></div>
    </form>
  </dialog>;
}
