import { expect,it } from 'vitest';
import { originalFiles } from './helpers';
import { openLevel } from '../src/model/level';
import { texturedEditorScene } from '../src/model/texturedEditorScene';
import { editorScene, spatialEdit } from '../src/model/editorScene';
import { roomHeightFixture,twoStoreyHeightTemplate } from './roomHeightFixture';
import { roomSettings } from '../src/model/roomHeights';
import { execute,undo,redo } from '../src/model/commands';

it('all sixteen levels supply original textured wall/polygon identities for both storeys',()=>{
 for(let i=1;i<=16;i++){
  const document=openLevel(`Level ${i}`,originalFiles(i)),scene=texturedEditorScene(document);
  expect(scene.faces.some(f=>f.kind==='wall')).toBe(true);expect(scene.faces.some(f=>f.kind==='floor')).toBe(true);
  for(const face of scene.faces){
   expect(document.roomSource.rooms.some(r=>r.id===face.room)).toBe(true);
   expect((face.upper?document.zones[face.zone]!.upper:document.zones[face.zone]!.lower).some(r=>r.offset===face.graphic.offset)).toBe(true);
   expect(face.vertices.every(v=>[v.position.x,v.position.y,v.position.z,v.u,v.v,v.brightness].every(Number.isFinite))).toBe(true);
   if(face.graphic.wall)expect(face.vertices.map(v=>v.u)).toContain(face.graphic.wall.stripStart);
  }
 }
});
it('both-storey editing retains each plane and independent height history',()=>{
 const fixture=roomHeightFixture(twoStoreyHeightTemplate()),base=roomSettings(fixture.document,fixture.from),scene=editorScene(fixture.document,'both');
 expect(scene.faces.filter(f=>f.room===fixture.from.id&&f.kind==='floor').some(f=>f.upper&&f.points.every(p=>p.y===base.upperFloor))).toBe(true);
 expect(scene.faces.filter(f=>f.room===fixture.from.id&&f.kind==='floor').some(f=>!f.upper&&f.points.every(p=>p.y===base.floor))).toBe(true);
 for(const field of ['floor','ceiling','upperFloor','upperCeiling'] as const){
  const value=base[field]+(field.toLowerCase().endsWith('floor')?-.25:.25);
  let history=execute({document:fixture.document,past:[],future:[]},spatialEdit(fixture.document,{kind:'height',room:fixture.from.id,field,value}));
  const changed=roomSettings(history.document,history.document.roomSource.rooms.find(r=>r.id===fixture.from.id)!);
  expect(changed[field]).toBe(value);for(const other of ['floor','ceiling','upperFloor','upperCeiling'] as const)if(other!==field)expect(changed[other]).toBe(base[other]);
  expect(texturedEditorScene(history.document).faces.some(f=>f.room===fixture.from.id&&f.upper===field.startsWith('upper')&&f.points.every(p=>p.y===value))).toBe(true);
  history=undo(history);expect(history.document.roomSource).toBe(fixture.document.roomSource);history=redo(history);expect(history.past).toHaveLength(1);
 }
});
