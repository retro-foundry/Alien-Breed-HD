import type { LevelDocument, Graphic } from './level';
import type { SceneFace } from './editorScene';
import { buildPlayableLevel, type PlayableLevel } from './playableBuild';
import { convexParts } from './geometry';
import { cross, dot, sub, type Vector3 } from './editorCamera';
import { BinaryReader } from './binary';
import { polygonMaterialBank, polygonPlane, type PolygonMaterialMode } from './surfaces';

export interface GeometryVertex { position: Vector3; u: number; v: number; brightness: number }
export interface TexturedFace extends SceneFace {
  zone: number; graphic: Graphic; vertices: readonly GeometryVertex[]; material: string;
  normal: Vector3; boundary: boolean;
}
export interface TexturedScene { built: PlayableLevel; faces: readonly TexturedFace[]; unsupported: readonly string[] }
export const floorMaterialKey = (mode: PolygonMaterialMode, texture: number) => `floor:${mode}:${mode === 'smooth' ? 0 : texture}`;
/** Use the shared compiler, including authored portals, partial polygons, paint,
 * heights and native wall strip/phase fields. Never stretch a room-volume box
 * over doors or replace independently painted faces with a template texture. */
export function texturedEditorScene(source: LevelDocument): TexturedScene {
  const built = buildPlayableLevel(source, -1), document = built.document, faces: TexturedFace[] = [], unsupported = new Set<string>();
  const reader = new BinaryReader(document.files.data, 'editor point lighting');
  const pointBrightness = (id: number, upper: boolean) => {
    const word = reader.i16(document.pointBrightnessOffset + id * 4 + (upper ? 2 : 0));
    return (word << 24) >> 24; // signed authored base; high byte carries animation flags
  };
  for (const zone of document.zones) {
    const room = built.topology?.cells.find(cell => cell.id === zone.id)?.sourceId ?? source.roomSource.rooms.find(room => room.originZone === zone.id)?.id;
    if (!room) throw new Error(`Room ${zone.id} has no editable source identity.`);
    const centrePoints = zone.boundaryIds.map(id => document.lines[id]!);
    const centre = { x: centrePoints.reduce((n,p)=>n+p.x,0)/Math.max(1,centrePoints.length), z: centrePoints.reduce((n,p)=>n+p.z,0)/Math.max(1,centrePoints.length) };
    for (const upper of [false,true]) for (const graphic of upper ? zone.upper : zone.lower) {
      const wall = graphic.wall, polygon = graphic.polygon;
      if (wall) {
        const a=document.points[wall.points[0]]!, b=document.points[wall.points[1]]!, top=wall.topFixed/256, bottom=wall.bottomFixed/256;
        if (top === bottom) continue;
        const brightnessA=pointBrightness(a.id,upper)+wall.brightness, brightnessB=pointBrightness(b.id,upper)+wall.brightness;
        const vertices: GeometryVertex[]=[
          {position:{x:a.x,z:a.z,y:top},u:wall.stripStart,v:0,brightness:brightnessA},
          {position:{x:b.x,z:b.z,y:top},u:wall.stripEnd,v:0,brightness:brightnessB},
          {position:{x:b.x,z:b.z,y:bottom},u:wall.stripEnd,v:bottom-top,brightness:brightnessB},
          {position:{x:a.x,z:a.z,y:bottom},u:wall.stripStart,v:bottom-top,brightness:brightnessA}];
        const normal=cross(sub(vertices[1]!.position,vertices[0]!.position),sub(vertices[2]!.position,vertices[0]!.position));
        if(dot(normal,{x:centre.x-a.x,y:0,z:centre.z-a.z})<0) vertices.reverse();
        faces.push({room,upper,zone:zone.id,graphic,kind:'wall',points:vertices.map(v=>v.position),vertices,normal:cross(sub(vertices[1]!.position,vertices[0]!.position),sub(vertices[2]!.position,vertices[0]!.position)),material:`wall:${wall.texture}`,boundary:true});
      } else if(polygon && polygonPlane(polygon.type)) {
        const kind=polygonPlane(polygon.type)!, y=graphic.type===1 ? upper?zone.upperFloor:zone.floor : graphic.type===2 ? upper?zone.upperCeiling:zone.ceiling : polygon.heightQuarter/4;
        const boundaryHeight=kind==='floor' ? upper?zone.upperFloor:zone.floor : upper?zone.upperCeiling:zone.ceiling;
        const material=floorMaterialKey(polygonMaterialBank(polygon.type),polygon.texture), scale=2**Math.max(-15,Math.min(15,polygon.scale));
        const coordinates=polygon.points.map(id=>document.points[id]!);
        // Native polygons are not necessarily whole room floors or convex.
        for(const part of convexParts(coordinates)) for(let i=1;i<part.length-1;i++) {
          const triangle=[part[0]!,part[i]!,part[i+1]!], normal={x:0,y:kind==='floor'?-1:1,z:0};
          const vertices=triangle.map(p=>({position:{x:p.x,z:p.z,y},u:p.x*scale,v:p.z*scale,brightness:(upper?zone.upperBrightness:zone.brightness)+polygon.brightness}));
          if(dot(cross(sub(vertices[1]!.position,vertices[0]!.position),sub(vertices[2]!.position,vertices[0]!.position)),normal)<0) vertices.reverse();
          faces.push({room,upper,zone:zone.id,graphic,kind,points:vertices.map(v=>v.position),vertices,normal,material,boundary:y===boundaryHeight});
        }
      } else if(graphic.type===7) unsupported.add('Animated water');
      else if(graphic.type===5) unsupported.add('Curved wall graphics');
    }
  }
  return {built,faces,unsupported:[...unsupported]};
}
