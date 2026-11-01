import type {LevelDocument} from './level';
import {pointInRoomOutline} from './geometry';
import {roomSettings} from './roomHeights';
import {clipCameraPolygon,projectPoint,type EditorCamera} from './editorCamera';
import type {SceneFace} from './editorScene';

export interface SelectionPoint {x:number;y:number}
/** Crossing selection, independent of drag direction. Includes a rectangle
 * wholly inside a zone and zones cut by its edges, not just visible corners. */
export function polygonInSelectionBox(polygon:readonly SelectionPoint[],start:SelectionPoint,end:SelectionPoint):boolean{
 if(!polygon.length)return false;
 const left=Math.min(start.x,end.x),right=Math.max(start.x,end.x),top=Math.min(start.y,end.y),bottom=Math.max(start.y,end.y);
 if(![left,right,top,bottom,...polygon.flatMap(p=>[p.x,p.y])].every(Number.isFinite))throw new Error('Selection coordinates must be finite.');
 const inside=(p:SelectionPoint)=>p.x>=left&&p.x<=right&&p.y>=top&&p.y<=bottom;
 if(polygon.some(inside))return true;
 const corners=[{x:left,y:top},{x:right,y:top},{x:right,y:bottom},{x:left,y:bottom}];
 if(corners.some(p=>pointInRoomOutline(polygon.map(p=>({x:p.x,z:p.y})),{x:p.x,z:p.y})))return true;
 const cross=(a:SelectionPoint,b:SelectionPoint,c:SelectionPoint)=>(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
 const on=(p:SelectionPoint,a:SelectionPoint,b:SelectionPoint)=>cross(a,b,p)===0&&p.x>=Math.min(a.x,b.x)&&p.x<=Math.max(a.x,b.x)&&p.y>=Math.min(a.y,b.y)&&p.y<=Math.max(a.y,b.y);
 const intersects=(a:SelectionPoint,b:SelectionPoint,c:SelectionPoint,d:SelectionPoint)=>{
  const x=cross(a,b,c),y=cross(a,b,d),z=cross(c,d,a),w=cross(c,d,b);
  return x*y<0&&z*w<0 || x===0&&on(c,a,b)||y===0&&on(d,a,b)||z===0&&on(a,c,d)||w===0&&on(b,c,d);
 };
 return polygon.some((a,i)=>corners.some((c,j)=>intersects(a,polygon[(i+1)%polygon.length]!,c,corners[(j+1)%4]!)));
}
export function mapRoomsInSelectionBox(document:LevelDocument,start:{x:number;z:number},end:{x:number;z:number},upper:boolean):string[]{
 return document.roomSource.rooms.filter(room=>{
  const settings=roomSettings(document,room);if(upper&&(!settings.upper.length||settings.upperFloor<=settings.upperCeiling))return false;
  return polygonInSelectionBox(room.vertices.map(p=>({x:p.x,y:p.z})),{x:start.x,y:start.z},{x:end.x,y:end.z});
 }).map(room=>room.id);
}
export function spatialRoomsInSelectionBox(faces:readonly SceneFace[],camera:EditorCamera,width:number,height:number,start:SelectionPoint,end:SelectionPoint):string[]{
 return [...new Set(faces.filter(face=>polygonInSelectionBox(clipCameraPolygon(face.points,camera).map(p=>projectPoint(p,camera,width,height)),start,end)).map(face=>face.room))];
}
