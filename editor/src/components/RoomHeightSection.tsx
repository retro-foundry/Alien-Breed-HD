import { useEffect, useRef, useState, type PointerEvent } from 'react';
import type { Zone } from '../model/level';
import { restoreRoomHeights, type RoomHeights } from '../model/roomHeights';
import './RoomHeightSection.css';

const sectionHeight = 400, plotTop = 24, plotHeight = sectionHeight - plotTop * 2;
const names = { floor: 'Lower floor', ceiling: 'Lower ceiling', upperFloor: 'Upper floor', upperCeiling: 'Upper ceiling' };
type Plane = keyof RoomHeights;
interface Gesture { pointer: number; field: Plane; startY: number; startHeight: number; min: number; max: number; next: RoomHeights }

/** A section through the actual air bands, not an alternative game renderer.
 * Only a completed gesture changes source; native height validation is shared
 * with the numerical inspector and compiler. */
export function RoomHeightSection({ settings, baseline, onCommit }: { settings: Zone; baseline: Uint8Array; onCommit: (values: RoomHeights) => void }) {
  const svg = useRef<SVGSVGElement>(null), gesture = useRef<Gesture | null>(null);
  const [draft, setDraft] = useState<RoomHeights | null>(null);
  useEffect(() => { gesture.current = null; setDraft(null); }, [baseline, settings.floor, settings.ceiling, settings.upperFloor, settings.upperCeiling]);
  const fields: Plane[] = settings.upper.length && settings.upperFloor > settings.upperCeiling
    ? ['upperCeiling', 'upperFloor', 'ceiling', 'floor'] : ['ceiling', 'floor'];
  const original = { floor: settings.floor, ceiling: settings.ceiling, upperFloor: settings.upperFloor, upperCeiling: settings.upperCeiling };
  const heights = draft ?? original;
  const lowerHeadroom = heights.floor - heights.ceiling, upperHeadroom = heights.upperFloor - heights.upperCeiling;
  const sharedPlane = fields.length === 4 && heights.upperFloor === heights.ceiling;
  const gap = fields.length === 4 ? heights.ceiling - heights.upperFloor : 0;
  const values = fields.map(field => settings[field]), span = Math.max(16, Math.max(...values) - Math.min(...values));
  const min = gesture.current?.min ?? Math.min(...values) - span * 0.2;
  const max = gesture.current?.max ?? Math.max(...values) + span * 0.2;
  const y = (height: number) => plotTop + (height - min) / (max - min) * plotHeight;
  let error = '';
  if (draft) try { restoreRoomHeights(draft, settings); } catch (reason) { error = reason instanceof Error ? reason.message : String(reason); }
  const cancel = () => { gesture.current = null; setDraft(null); };
  const pointY = (event: PointerEvent<SVGGElement>) => {
    const matrix = svg.current!.getScreenCTM();
    if (!matrix) throw new Error('The height diagram is not visible.');
    return new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse()).y;
  };
  const move = (event: PointerEvent<SVGGElement>) => {
    const active = gesture.current;
    if (!active || active.pointer !== event.pointerId) return;
    const value = active.startHeight + (pointY(event) - active.startY) * (active.max - active.min) / plotHeight;
    const bounded = Math.max(-8192, Math.min(8191.75, Math.round(value * 4) / 4));
    const next = { ...original, [active.field]: bounded }; active.next = next; setDraft(next);
  };
  return <div className="room-height-section">
    <p className="field-note" id="height-section-help">Side section: shaded bands are usable room space. Drag a floor or ceiling up/down. Release applies one edit; Escape cancels. Focus a line and use ↑/↓ for quarters, Shift for 4 units. Preview rebuilds after applying.</p>
    <svg ref={svg} viewBox={`0 0 240 ${sectionHeight}`} width="240" height={sectionHeight} role="group" aria-label="Room height section" aria-describedby="height-section-help">
      {gap > 0 && <rect className="height-between-storeys" x="12" width="216" y={y(heights.upperFloor)} height={Math.max(0, y(heights.ceiling) - y(heights.upperFloor))} />}
      <rect className="height-air" x="12" width="216" y={y(heights.ceiling)} height={Math.max(0, y(heights.floor) - y(heights.ceiling))} />
      {fields.length === 4 && <rect className="height-air upper" x="12" width="216" y={y(heights.upperCeiling)} height={Math.max(0, y(heights.upperFloor) - y(heights.upperCeiling))} />}
      {lowerHeadroom > 0 && y(heights.floor) - y(heights.ceiling) > 30 && <text className="height-space-label" x="120" y={(y(heights.floor) + y(heights.ceiling)) / 2} textAnchor="middle">{fields.length === 4 ? 'Lower room' : 'Room'}: {lowerHeadroom} headroom</text>}
      {fields.length === 4 && upperHeadroom > 0 && y(heights.upperFloor) - y(heights.upperCeiling) > 30 && <text className="height-space-label" x="120" y={(y(heights.upperFloor) + y(heights.upperCeiling)) / 2} textAnchor="middle">Upper room: {upperHeadroom} headroom</text>}
      {fields.map(field => <g key={field} className={`height-plane ${draft && gesture.current?.field === field && error ? 'invalid' : ''}`}
        role="slider" tabIndex={0} aria-label={`${names[field]} handle`} aria-orientation="vertical" aria-valuemin={-8192} aria-valuemax={8191.75}
        aria-valuenow={heights[field]} aria-valuetext={`${names[field]} ${heights[field]}; smaller values are higher`} aria-describedby="height-section-help"
        data-control-help="Drag vertically to change this plane; release applies one undoable edit. Escape cancels. Up/Down changes quarters; Shift changes 4 units. Smaller values are higher."
        onPointerDown={event => { if (event.button !== 0 || gesture.current) return; event.preventDefault(); event.currentTarget.focus();
          gesture.current = { pointer: event.pointerId, field, startY: pointY(event), startHeight: settings[field], min, max, next: original };
          setDraft(original); event.currentTarget.setPointerCapture(event.pointerId); }}
        onPointerMove={move} onPointerCancel={cancel} onLostPointerCapture={cancel}
        onPointerUp={event => { if (event.button !== 0 || gesture.current?.pointer !== event.pointerId) return;
          // Commit the last displayed proposal; release is not another drag.
          const next = gesture.current.next; gesture.current = null; setDraft(null);
          event.currentTarget.releasePointerCapture(event.pointerId); onCommit(next); }}
        onKeyDown={event => {
          if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); cancel(); }
          if (event.key === 'ArrowUp' || event.key === 'ArrowDown') { event.preventDefault(); event.stopPropagation();
            if (gesture.current) return;
            onCommit({ ...original, [field]: settings[field] + (event.key === 'ArrowUp' ? -1 : 1) * (event.shiftKey ? 4 : 0.25) }); }
        }}>
        <line className="height-hit" x1={sharedPlane && field === 'upperFloor' ? 120 : 12} x2={sharedPlane && field === 'ceiling' ? 112 : 228} y1={y(heights[field])} y2={y(heights[field])} />
        <line className="height-line" x1="12" x2="228" y1={y(heights[field])} y2={y(heights[field])} />
        <text x={sharedPlane && field === 'upperFloor' ? 208 : 16} textAnchor={sharedPlane && field === 'upperFloor' ? 'end' : 'start'} y={y(heights[field]) - 7}>{names[field]} {heights[field]}</text>
        <circle cx={fields.length === 4 && field === 'ceiling' ? 25 : 215} cy={y(heights[field])} r="5" />
      </g>)}
    </svg>
    {fields.length === 4 && <p className="field-note">{sharedPlane ? 'The upper floor and lower ceiling meet at the same height. Left handle: lower ceiling. Right handle: upper floor.' : gap > 0 ? `The two rooms have ${gap} units between the lower ceiling and upper floor.` : 'The storey planes overlap. Move them back into a valid order.'}</p>}
    <p className="field-note">Each storey has its own floor and ceiling heights. Hiding ceilings in an editing view only helps you look inside; it does not change the game.</p>
    {draft && <p className="field-note" role="status">{error || `Lower clearance ${heights.floor - heights.ceiling}. Release to apply; Escape to cancel.`}</p>}
  </div>;
}
