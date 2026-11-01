import {surfaceMaterialCurrent,surfaceMaterialTarget,setSurfaceMaterial,wallMaterialChoices} from '../src/model/wallMaterials';
import {createHash} from 'node:crypto';
import {it,expect} from 'vitest';
import {originalFiles} from './helpers';
import {openLevel} from '../src/model/level';
import {spatialEdit} from '../src/model/editorScene';
import {applyCommand,editRoomSource,execute,undo,redo} from '../src/model/commands';
import {texturedEditorScene} from '../src/model/texturedEditorScene';
import {roomSettings} from '../src/model/roomHeights';
import {setRoomHeights} from '../src/model/roomSource';
import {mechanismsForRoom} from '../src/model/mechanismDependencies';
import {encodeProject,decodeProject} from '../src/model/project';

it('ceiling height edits preserve textures around the reported room 5 doorway',()=>{
 const d=openLevel('L1',originalFiles(1));
 for(const id of [4,5,6]){const room=d.roomSource.rooms.find(r=>r.originZone===id)!,settings=roomSettings(d,room);if(settings.floor<=settings.ceiling)continue;
  const changed=applyCommand(d,spatialEdit(d,{kind:'height',room:room.id,field:'ceiling',value:settings.ceiling+16}));
  expect(texturedEditorScene(changed).faces.length).toBeGreaterThan(0);
  expect(changed.roomAppearances).toBeUndefined();
 }
});
it('the reported -40 ceiling / 240 floor compiles, saves and undoes without repair appearance',()=>{
 const document=openLevel('Height repair',originalFiles(1)),room=document.roomSource.rooms.find(r=>r.originZone===4)!,settings=roomSettings(document,room);
 let history=execute({document,past:[],future:[]},editRoomSource(setRoomHeights(document,document.roomSource,room.id,{floor:240,ceiling:-40,upperFloor:settings.upperFloor,upperCeiling:settings.upperCeiling})));
 const saved=decodeProject(encodeProject(history.document)),mesh=texturedEditorScene(saved);
 expect(roomSettings(saved,saved.roomSource.rooms.find(r=>r.id===room.id)!)).toMatchObject({ceiling:-40,floor:240});
 expect(mesh.faces.some(f=>f.room===room.id&&f.kind==='floor'&&f.points.every(p=>p.y===240))).toBe(true);
 expect(mesh.faces.some(f=>f.room===document.roomSource.rooms.find(r=>r.originZone===5)!.id&&f.kind==='wall')).toBe(true);
 expect(saved.roomAppearances).toBeUndefined();for(const key of ['data','graphics','clips'] as const)expect(createHash('sha256').update(history.document.files[key]).digest('hex')).toBe(createHash('sha256').update(document.files[key]).digest('hex'));
 history=undo(history);expect(history.document.roomSource).toBe(document.roomSource);history=redo(history);expect(texturedEditorScene(history.document).faces.length).toBeGreaterThan(0);
});
it('an imported portal room in every original level can change ceiling without losing its material mesh',()=>{
 for(let level=1;level<=16;level++){
  const document=openLevel(`L${level}`,originalFiles(level));
  const room=document.roomSource.rooms.find(r=>{const s=roomSettings(document,r);return s.floor-s.ceiling>8 && s.lower.some(g=>g.wall) && !mechanismsForRoom(document,r.recordZone).length && s.boundaryIds.some(id=>document.lines[id]!.connect>=0);})!;
  expect(room).toBeDefined();const settings=roomSettings(document,room);
  const changed=applyCommand(document,spatialEdit(document,{kind:'height',room:room.id,field:'ceiling',value:settings.ceiling+4}));
  try{expect(texturedEditorScene(changed).faces.length,`level ${level}`).toBeGreaterThan(0);}catch(error){throw new Error(`Level ${level}: ${String(error)}`);}
 }
});

it('height-generated bands expose their real material to the picker without a repair appearance',()=>{
 const original=openLevel('Paint inherited doorway',originalFiles(1)),room=original.roomSource.rooms.find(r=>r.originZone===4)!,settings=roomSettings(original,room);
 const document=applyCommand(original,editRoomSource(setRoomHeights(original,original.roomSource,room.id,{floor:240,ceiling:-40,upperFloor:settings.upperFloor,upperCeiling:settings.upperCeiling}))),scene=texturedEditorScene(document);
 const face=scene.faces.find(face=>face.graphic.wall && scene.built.surfaceSources?.some(s=>s.room===face.zone&&s.offset===face.graphic.offset&&s.upper===face.upper&&s.boundaryIndex!==undefined&&s.sourceOffset===undefined))!;
 expect(face).toBeDefined();const target=surfaceMaterialTarget(scene.built,face.zone,{offset:face.graphic.offset,type:face.graphic.type,upper:face.upper})!;
 expect(surfaceMaterialCurrent(document,target)).toMatchObject({texture:face.graphic.wall!.texture});
 const style=wallMaterialChoices(document).find(r=>r.wall!.texture===face.graphic.wall!.texture&&r.wall!.horizontalMask===face.graphic.wall!.horizontalMask&&r.wall!.verticalMask===face.graphic.wall!.verticalMask)!;
 expect(style).toBeDefined();const command=editRoomSource(setSurfaceMaterial(document,target,style.offset)),painted=applyCommand(document,command);
 expect(()=>texturedEditorScene(decodeProject(encodeProject(painted)))).not.toThrow();expect(painted.roomAppearances).toBeUndefined();
});
