import {it,expect} from 'vitest';
import {polygonInSelectionBox,mapRoomsInSelectionBox,spatialRoomsInSelectionBox} from '../src/model/roomSelection';
import {roomHeightFixture} from './roomHeightFixture';
import {editorScene} from '../src/model/editorScene';
import {cameraFor,frameCamera,projectPoint} from '../src/model/editorCamera';

it('selection handles containment, crossing edges, touching, reversed drags and misses',()=>{
 const polygon=[{x:-10,y:4},{x:10,y:4},{x:10,y:6},{x:-10,y:6}];
 for(const [a,b] of [[{x:0,y:0},{x:2,y:10}],[{x:2,y:10},{x:0,y:0}],[{x:0,y:4.5},{x:1,y:5.5}],[{x:10,y:4},{x:12,y:8}]] as const)expect(polygonInSelectionBox(polygon,a,b)).toBe(true);
 expect(polygonInSelectionBox(polygon,{x:11,y:0},{x:12,y:2})).toBe(false);
 expect(polygonInSelectionBox([],{x:0,y:0},{x:1,y:1})).toBe(false);
});
it('map and 3D zone rectangles select current source rooms without editing source',()=>{
 const fixture=roomHeightFixture(),source=fixture.document.roomSource;
 expect(mapRoomsInSelectionBox(fixture.document,{x:20020,z:20020},{x:21240,z:20240},false)).toEqual([fixture.from.id,fixture.to.id]);
 expect(mapRoomsInSelectionBox(fixture.document,{x:20020,z:20020},{x:20040,z:20040},false)).toEqual([fixture.from.id]);
 for(const view of ['perspective','orthographic'] as const){
  const scene=editorScene(fixture.document,'both'),faces=scene.faces.filter(f=>f.room===fixture.from.id||f.room===fixture.to.id),camera=frameCamera(cameraFor(view),faces.flatMap(f=>[...f.points]),800,600),p=faces.flatMap(f=>f.points.map(p=>projectPoint(p,camera,800,600)));
  const a={x:Math.min(...p.map(p=>p.x))-5,y:Math.min(...p.map(p=>p.y))-5},b={x:Math.max(...p.map(p=>p.x))+5,y:Math.max(...p.map(p=>p.y))+5};
  expect(new Set(spatialRoomsInSelectionBox(faces,camera,800,600,a,b))).toEqual(new Set([fixture.from.id,fixture.to.id]));
 }
 expect(fixture.document.roomSource).toBe(source);
});
