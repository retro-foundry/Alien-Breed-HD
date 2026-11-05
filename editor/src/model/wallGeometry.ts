import { BinaryReader } from './binary';
import { readWallSurface } from './surfaces';
import { mechanismsForRoom } from './mechanismDependencies';
import { normaliseRoomOutline, type Position } from './geometry';
import type { Graphic, LevelDocument } from './level';
import type { GraphicsRecordInput } from './graphicsBuild';
import type { RoomSource, SourceRoom } from './roomSource';
import type { RoomTopology } from './roomTopology';
import { encodeWallSurface, type WallSurface } from './surfaces';
import { wallForSpan } from './surfaceGeometry';
import { changedPortalEdges, heightPortalTarget, hasUpperLayer, newPortalTarget, portalWallSpans, unassignedBoundaryEdges } from './portalWalls';
import { roomSettings } from './roomHeights';
import { hasNativeWallMaterials } from './roomMaterials';
import { nativeRoomTranslation, outlineTranslation } from './roomTranslation';
import { sourceBoundaryIndex } from './boundaryMaterials';
import { importedBarrierModeOverride, importedBarrierModeAt } from './importedBarriers';

export interface CellWalls { lower: GraphicsRecordInput[]; upper: GraphicsRecordInput[] }
export interface NewRoomWallMaterials { lower: readonly WallSurface[]; upper?: readonly WallSurface[] }
export interface ClosingWallMaterials { walls: NewRoomWallMaterials; lightingPoint: number }
export interface WallGeometry { topology: RoomTopology; walls: Map<number, CellWalls> }
const distance = (a: Position, b: Position) => Math.hypot(b.x - a.x, b.z - a.z);
const on = (p: Position, a: Position, b: Position) => (p.x - a.x) * (b.z - a.z) === (p.z - a.z) * (b.x - a.x)
  && p.x >= Math.min(a.x, b.x) && p.x <= Math.max(a.x, b.x) && p.z >= Math.min(a.z, b.z) && p.z <= Math.max(a.z, b.z);
const key = (points: readonly Position[]) => JSON.stringify(normaliseRoomOutline(points.map(({ x, z }) => ({ x, z }))));
function wallChain(room: SourceRoom, lineId: number) {
  let total = 0;
  const first = room.wallSources.findIndex((id, index) => id === lineId && room.wallSources[(index + room.wallSources.length - 1) % room.wallSources.length] !== lineId);
  const pieces = room.vertices.flatMap((_, step) => {
    const index = (Math.max(first, 0) + step) % room.vertices.length, start = room.vertices[index]!;
    if (room.wallSources[index] !== lineId) return [];
    const end = room.vertices[(index + 1) % room.vertices.length]!, length = distance(start, end), offset = total;
    total += length; return [{ start, end, length, offset }];
  });
  return { pieces, total };
}

/** Bind source boundary materials to deformed/split cell edges. Preserve
 * strip density, phase, sampling and heights (WallRoutine3.ChipMem.s), and
 * retain source record identities for subsequent mechanism attachment mapping.
 * Partial boundary decorations add their own point identities to the returned
 * topology; callers must use that point table in all following compiler stages.
 * New-room perimeter materials are explicit selections, not guessed textures.
 */
export function buildCellWallGeometry(document: LevelDocument, source: RoomSource, topology: RoomTopology,
  newMaterials: ReadonlyMap<string, NewRoomWallMaterials> = new Map(), closingMaterials: ReadonlyMap<string, ClosingWallMaterials> = new Map()): WallGeometry {
  const points = topology.points.map(point => ({ ...point })), coordinates = new Map<string, number>();
  points.forEach((point, id) => { const name = `${point.x},${point.z}`; if (!coordinates.has(name)) coordinates.set(name, id); });
  const pointId = (point: Position): number => {
    const rounded = { x: Math.round(point.x), z: Math.round(point.z) };
    if (Math.abs(rounded.x - point.x) > 1e-7 || Math.abs(rounded.z - point.z) > 1e-7)
      throw new Error('A wall decoration falls between whole world coordinates after reshaping. Choose a grid-aligned room shape or reassign this decoration before building.');
    const name = `${rounded.x},${rounded.z}`, existing = coordinates.get(name);
    if (existing !== undefined) return existing;
    if (points.length >= 2048) throw new Error('Wall geometry exceeds the engine limit of 2,048 points.');
    const id = points.length; points.push(rounded); coordinates.set(name, id); return id;
  };
  const walls = new Map<number, CellWalls>();
  for (const cell of topology.cells) {
    if (walls.has(cell.id)) throw new Error('Wall geometry contains duplicate runtime room identities.');
    const room = source.rooms.find(room => room.id === cell.sourceId), template = document.zones.find(zone => zone.id === cell.recordZone);
    if (!room || !template) throw new Error(`Room ${cell.sourceId} has a missing wall source or settings template.`);
    const native = hasNativeWallMaterials(room.originZone, room.wallSources);
    const settings = roomSettings(document, cell);
    const unassigned = unassignedBoundaryEdges(topology, cell);
    const changed = [...changedPortalEdges(document, topology, cell), ...unassigned], closing = closingMaterials.get(room.id);
    if (unassigned.length && !closing?.walls.lower.length) throw new Error(`Room ${room.id.replace(/^room-/, '')} has new boundary walls needing a wall style. Select the room and apply its wall repair appearance before building.`);
    if (changed.length && !closing?.walls.lower.length) throw new Error(`Room ${room.id.replace(/^room-/, '')} has a changed doorway needing a wall style. Select the room and apply its wall repair appearance before building.`);
    if (changed.length && !room.upperStorey && hasUpperLayer(document, settings) && !closing?.walls.upper?.length)
      throw new Error(`Room ${room.id.replace(/^room-/, '')} needs an upper wall style for its changed doorway. Apply its upper wall repair appearance before building.`);
    const outline = cell.vertices.map(id => {
      const point = points[id]; if (!point) throw new Error(`Room ${cell.sourceId} has a missing wall corner.`); return point;
    });
    const originalOutline = template.boundaryIds.map(id => ({ x: document.lines[id]!.x, z: document.lines[id]!.z }));
    const unchanged = key(outline) === key(originalOutline);
    const translation = nativeRoomTranslation(document, room), whole = outlineTranslation(room.vertices, outline);
    const dx = translation?.x ?? 0, dz = translation?.z ?? 0;
    const translatedWholeRoom = !!translation && whole?.x === 0 && whole.z === 0;
    const retained = (record: Graphic) => record.points.every(id => points[id]?.x === document.points[id]!.x && points[id]?.z === document.points[id]!.z);
    // A moving neighbour's static frame surrounds its full swept aperture.
    // Closing it at its initial plane would permanently seal an animated door.
    const aperture=(target: typeof cell,original=false)=>{
      let result=original?document.zones.find(zone=>zone.id===target.recordZone)!:roomSettings(document,target);
      for(const mechanism of mechanismsForRoom(document,target.recordZone))result={...result,
        ...(mechanism.kind==='door'?{ceiling:Math.min(result.ceiling,mechanism.top)}:{floor:Math.max(result.floor,mechanism.bottom)})};
      return result;
    };
    const layer = (records: readonly Graphic[], selected: readonly WallSurface[] | undefined, upper: boolean): GraphicsRecordInput[] => {
      if (!native) {
        if (!selected) throw new Error(`Room ${room.id} needs explicit wall materials before building.`);
        return cell.vertices.flatMap((a, index) => {
          const neighbour = cell.neighbours[index]!;
          if (neighbour >= 0 && topology.cells.find(target => target.id === neighbour)?.sourceId === cell.sourceId) return [];
          const destination = neighbour < 0 ? undefined : topology.cells.find(target => target.id === neighbour);
          const b = cell.vertices[(index + 1) % cell.vertices.length]!;
          return selected.map(material => {
            const start = document.points[material.points[0]], end = document.points[material.points[1]];
            if (!start || !end) throw new Error(`Room ${room.id} has a missing wall material reference endpoint.`);
            const bound = wallForSpan(material, distance(start, end), [a, b], 0, distance(points[a]!, points[b]!));
            const spans = destination ? portalWallSpans(document, bound, roomSettings(document, destination)) : [bound];
            const boundaryIndex = sourceBoundaryIndex(room, points[a]!, points[b]!);
            if (boundaryIndex < 0) throw new Error(`Room ${room.id} lost its source boundary identity.`);
            return spans.map(span => ({ bytes: encodeWallSurface(span), boundaryIndex }));
          }).flat();
        });
      }
      const original = records.filter(record => record.wall).flatMap(record => {
        const original = record.wall!, floor = upper ? 'upperFloor' : 'floor', roof = upper ? 'upperCeiling' : 'ceiling';
        // The draw stream owns the storey. Match that storey's authored air
        // boundaries exactly; other decorative world heights stay authored.
        let material = { ...original, topFixed: original.topFixed === template[roof] * 256 ? settings[roof] * 256 : original.topFixed,
          bottomFixed: original.bottomFixed === template[floor] * 256 ? settings[floor] * 256 : original.bottomFixed };
        let heightChanged = material.topFixed !== original.topFixed || material.bottomFixed !== original.bottomFixed;
        const a = document.points[material.points[0]]!, b = document.points[material.points[1]]!;
        const lineId = template.boundaryIds.find(id => {
          const line = document.lines[id]!; return on(a, line, { x: line.x + line.dx, z: line.z + line.dz }) && on(b, line, { x: line.x + line.dx, z: line.z + line.dz });
        });
        const heightEdge=cell.vertices.findIndex((_,edge)=>cell.sourceLines[edge]===lineId && heightPortalTarget(document,topology,cell,edge));
        const heightTarget=heightEdge<0?undefined:heightPortalTarget(document,topology,cell,heightEdge);
        const controlled=document.mechanisms.some(item=>item.walls.some(wall=>wall.graphic===record.offset));
        if(heightTarget && !controlled){
          const before=document.zones.find(zone=>zone.id===heightTarget.recordZone)!,after=roomSettings(document,heightTarget);
          for(const field of ['floor','upperFloor'] as const)if(original.topFixed===before[field]*256 && original.topFixed!==template[roof]*256)material={...material,topFixed:after[field]*256};
          for(const field of ['ceiling','upperCeiling'] as const)if(original.bottomFixed===before[field]*256 && original.bottomFixed!==template[floor]*256)material={...material,bottomFixed:after[field]*256};
          if(material.bottomFixed<=material.topFixed)return [];
          heightChanged=material.topFixed!==original.topFixed||material.bottomFixed!==original.bottomFixed;
        }
        const opened = cell.vertices.some((_, edge) => cell.sourceLines[edge] === lineId && newPortalTarget(document, topology, cell, edge));
        const changedLine = changed.some(edge => cell.sourceLines[edge] === lineId);
        const modeChanged = importedBarrierModeOverride(document, room, record);
        if (unchanged && retained(record) && !opened && !changedLine && !heightChanged && !modeChanged) return [{ sourceOffset: record.offset, bytes: document.files.graphics.slice(record.offset, record.offset + record.size) }];
        if (lineId === undefined) {
          if (translatedWholeRoom) return [{ sourceOffset: record.offset, bytes: encodeWallSurface({ ...material,
            points: [pointId({ x: a.x + dx, z: a.z + dz }), pointId({ x: b.x + dx, z: b.z + dz })] }) }];
          throw new Error(`Room ${room.id} wall at byte ${record.offset} is not attached to a single boundary. Assign its geometry before compiling this room.`);
        }
        const line = document.lines[lineId]!, lengthSquared = line.dx ** 2 + line.dz ** 2;
        const fraction = (p: Position) => ((p.x - line.x) * line.dx + (p.z - line.z) * line.dz) / lengthSquared;
        const { pieces, total } = wallChain(room, lineId);
        const first = fraction(a) * total, last = fraction(b) * total, result: GraphicsRecordInput[] = [], uncut: WallSurface[] = [];
        for (let index = 0; index < cell.vertices.length; index++) {
          if (cell.sourceLines[index] !== lineId) continue;
          const start = points[cell.vertices[index]!]!, end = points[cell.vertices[(index + 1) % cell.vertices.length]!]!;
          const piece = pieces.find(piece => on(start, piece.start, piece.end) && on(end, piece.start, piece.end));
          if (!piece) throw new Error(`Room ${room.id} lost its wall chain while splitting geometry.`);
          const edgeStart = piece.offset + distance(piece.start, start), edgeEnd = piece.offset + distance(piece.start, end);
          const low = Math.max(Math.min(first, last), Math.min(edgeStart, edgeEnd)), high = Math.min(Math.max(first, last), Math.max(edgeStart, edgeEnd));
          if (high <= low) continue;
          const position = (arc: number) => ({ x: start.x + (end.x - start.x) * (arc - edgeStart) / (edgeEnd - edgeStart),
            z: start.z + (end.z - start.z) * (arc - edgeStart) / (edgeEnd - edgeStart) });
          const from = last > first ? low : high, to = last > first ? high : low;
          const bound = wallForSpan(modeChanged && importedBarrierModeAt(document, room, start, end) ? { ...material, type: 0 } : material,
            distance(a, b), [pointId(position(from)), pointId(position(to))], Math.abs(from - first), Math.abs(to - first));
          uncut.push(bound);
          if (changed.includes(index)) continue;
          const target = newPortalTarget(document, topology, cell, index);
          const spans = target && !(heightTarget && (controlled || material.type===13)) ? portalWallSpans(document, bound, heightTarget ? aperture(target) : roomSettings(document, target)) : [bound];
          result.push(...spans.map(span => ({ sourceOffset: record.offset, bytes: encodeWallSurface(span) })));
        }
        if ((!opened && !changedLine) || !uncut.length) return result;
        const starts = new Set(uncut.map(wall => wall.points[0])), ends = new Set(uncut.map(wall => wall.points[1]));
        const start = [...starts].filter(id => !ends.has(id)), end = [...ends].filter(id => !starts.has(id));
        if (start.length !== 1 || end.length !== 1) throw new Error(`Room ${room.id} has an ambiguous doorway material-lighting span.`);
        return result.map(item => ({ ...item, lightingSpan: [points[start[0]!]!, points[end[0]!]!] as const }));
      });
      const inheritedFills=cell.vertices.flatMap((a,edge)=>{
        const target=heightPortalTarget(document,topology,cell,edge);if(!target || upper && !hasUpperLayer(document,template))return [];
        const b=cell.vertices[(edge+1)%cell.vertices.length]!,from=points[a]!,to=points[b]!,length=distance(from,to);
        const roof=upper?'upperCeiling':'ceiling',floor=upper?'upperFloor':'floor';
        if(settings[floor]<=settings[roof])return [];
        const targetTemplate=document.zones.find(zone=>zone.id===target.recordZone)!;
        const ownRecords=[...records,...(upper?template.lower:template.upper)].filter(record=>record.wall && record.wall.texture!==11);
        const sourceRecords=ownRecords.length?ownRecords:[...(upper?targetTemplate.upper:targetTemplate.lower),...(upper?targetTemplate.lower:targetTemplate.upper)].filter(record=>record.wall&&record.wall.texture!==11);
        const translated=(id:number)=>({x:document.points[id]!.x+dx,z:document.points[id]!.z+dz});
        const along=(p:Position)=>((p.x-from.x)*(to.x-from.x)+(p.z-from.z)*(to.z-from.z))/length;
        const existing=original.map(item=>readWallSurface(new BinaryReader(item.bytes,'height doorway wall'),0));
        const divisions=[0,length,...existing.flatMap(wall=>wall.points.map(id=>points[id]!).filter(p=>on(p,from,to)).map(along))].sort((a,b)=>a-b).filter((n,i,list)=>i===0||Math.abs(n-list[i-1]!)>1e-7);
        const result:GraphicsRecordInput[]=[];
        for(let i=0;i<divisions.length-1;i++){
          const start=divisions[i]!,end=divisions[i+1]!;if(end-start<1e-7)continue;
          const position=(n:number)=>({x:from.x+(to.x-from.x)*n/length,z:from.z+(to.z-from.z)*n/length});
          const p=position(start),q=position(end),middle=position((start+end)/2);
          // Prefer this edge's actual band; otherwise continue a wall meeting
          // its corner. Never choose a random level material or copy a switch.
          const candidates=sourceRecords.filter(record=>{const x=translated(record.points[0]!),y=translated(record.points[1]!);return on(middle,x,y)||on(x,from,to)||on(y,from,to)||on(from,x,y)||on(to,x,y);});
          let reference=candidates.find(record=>on(middle,translated(record.points[0]!),translated(record.points[1]!)))
            ??candidates.find(record=>distance(translated(record.points[1]!),from)<1e-7)??candidates[0];
          if(!reference){
            const materialRoom=ownRecords.length?room:source.rooms.find(room=>room.id===target.sourceId)!;
            const boundary=Math.max(0,sourceBoundaryIndex(materialRoom,from,to));
            // An open portal can meet other open edges. Follow the same room's
            // perimeter to its next authored wall instead of guessing a sheet.
            for(let step=1;step<=materialRoom.vertices.length && !reference;step++){
              const index=(boundary-step+materialRoom.vertices.length)%materialRoom.vertices.length,x=materialRoom.vertices[index]!,y=materialRoom.vertices[(index+1)%materialRoom.vertices.length]!;
              reference=sourceRecords.find(record=>on(translated(record.points[0]!),x,y)&&on(translated(record.points[1]!),x,y));
            }
          }
          const shape={topFixed:settings[roof]*256,bottomFixed:settings[floor]*256};
          let missing=portalWallSpans(document,shape,aperture(target)).map(wall=>({top:wall.topFixed,bottom:wall.bottomFixed}));
          // Preserve intentional native omissions: add only newly exposed
          // solid area, not every theoretical wall absent in the original art.
          const previous=portalWallSpans(document,{topFixed:template[roof]*256,bottomFixed:template[floor]*256},aperture(target,true));
          for(const wall of previous)missing=missing.flatMap(span=>wall.bottomFixed<=span.top||wall.topFixed>=span.bottom?[span]:[{top:span.top,bottom:Math.min(span.bottom,wall.topFixed)},{top:Math.max(span.top,wall.bottomFixed),bottom:span.bottom}].filter(span=>span.bottom>span.top));
          for(const wall of existing.filter(wall=>on(middle,points[wall.points[0]]!,points[wall.points[1]]!)))missing=missing.flatMap(span=>wall.bottomFixed<=span.top||wall.topFixed>=span.bottom?[span]:[{top:span.top,bottom:Math.min(span.bottom,wall.topFixed)},{top:Math.max(span.top,wall.bottomFixed),bottom:span.bottom}].filter(span=>span.bottom>span.top));
          if(!missing.length)continue;
          if(!reference)throw new Error(`Room ${room.id} has no adjoining authored wall material for its changed height band (${upper?"upper":"lower"}, ${sourceRecords.length} styles).`);
          const referenceRoom=ownRecords.length?room:source.rooms.find(room=>room.id===target.sourceId)!;
          const referenceUpper=(ownRecords.length?template:targetTemplate).upper.some(record=>record.offset===reference.offset);
          const sourceWall=reference.wall!,paint=referenceRoom.wallMaterials?.find(item=>item.record===reference.offset&&item.upper===referenceUpper),material={...sourceWall,...(paint?{texture:paint.texture,verticalMask:paint.verticalMask,verticalShift:paint.verticalShift,horizontalMask:paint.horizontalMask}:{})};
          const x=translated(sourceWall.points[0]),y=translated(sourceWall.points[1]),referenceLength=distance(x,y),density=(material.stripEnd-material.stripStart)/referenceLength;
          const offset=on(middle,x,y)?((p.x-x.x)*(y.x-x.x)+(p.z-x.z)*(y.z-x.z))/referenceLength:distance(y,from)<1e-7?referenceLength+start:start;
          const base={...material,stripStart:material.stripStart+Math.trunc(offset*density),stripEnd:material.stripStart+Math.trunc(offset*density)+(material.stripEnd-material.stripStart)};
          const pointA=pointId(p),pointB=pointId(q),bound=wallForSpan(base,referenceLength,[pointA,pointB],0,end-start),boundaryIndex=sourceBoundaryIndex(room,points[pointA]!,points[pointB]!);
          if(boundaryIndex<0)throw new Error(`Room ${room.id} lost its height-band boundary identity.`);
          result.push(...missing.map(span=>({bytes:encodeWallSurface({...bound,topFixed:span.top,bottomFixed:span.bottom}),insertBefore:records.some(record=>record.offset===reference.offset)?reference.offset:records.find(record=>record.wall||record.polygon)?.offset??-1,boundaryIndex,lightingSpan:[from,to] as const})));
        }
        return result;
      });
      const fills = changed.flatMap(edge => (selected ?? []).flatMap(material => {
        const start = document.points[material.points[0]], end = document.points[material.points[1]];
        if (!start || !end) throw new Error(`Room ${room.id} has a missing closure material endpoint.`);
        const a = cell.vertices[edge]!, b = cell.vertices[(edge + 1) % cell.vertices.length]!;
        const line = cell.sourceLines[edge];
        const pieces = line == null ? room.vertices.flatMap((start, index) => room.wallSources[index] === null
          ? [{ start, end: room.vertices[(index + 1) % room.vertices.length]!, offset: 0 }] : []) : wallChain(room, line).pieces;
        const piece = pieces.find(piece => on(points[a]!, piece.start, piece.end) && on(points[b]!, piece.start, piece.end));
        if (!piece) throw new Error(`Room ${room.id} lost its doorway closure material chain.`);
        const offset = piece.offset + distance(piece.start, points[a]!);
        const bound = wallForSpan(material, distance(start, end), [a, b], offset, offset + distance(points[a]!, points[b]!));
        const target = newPortalTarget(document, topology, cell, edge);
        const spans = target ? portalWallSpans(document, bound, roomSettings(document, target)) : [bound];
        const boundaryIndex = sourceBoundaryIndex(room, points[a]!, points[b]!);
        if (boundaryIndex < 0) throw new Error(`Room ${room.id} lost its repaired boundary identity.`);
        return spans.map(span => ({ bytes: encodeWallSurface(span), lightingPoint: closing!.lightingPoint, boundaryIndex }));
      }));
      return [...original, ...inheritedFills, ...fills];
    };
    const selected = native ? closing?.walls : newMaterials.get(room.id);
    walls.set(cell.id, { lower: layer(template.lower, selected?.lower, false), upper: layer(template.upper, selected?.upper ?? (native ? undefined : []), true) });
  }
  return { topology: { ...topology, points }, walls };
}
