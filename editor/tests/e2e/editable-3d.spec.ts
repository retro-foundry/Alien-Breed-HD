import { texturedEditorScene } from '../../src/model/texturedEditorScene';
import { pickSceneFace } from '../../src/model/editorScene';
import { screenRay,dot,type EditorCamera } from '../../src/model/editorCamera';
import { expect,test,type Page } from '@playwright/test';
import { encodeProject,decodeProject } from '../../src/model/project';
import { readFile } from 'node:fs/promises';
import { roomHeightFixture,twoStoreyHeightTemplate } from '../roomHeightFixture';
import { roomSettings } from '../../src/model/roomHeights';

async function pixels(page:Page){return page.locator('canvas[aria-label="Textured 3D geometry"]').evaluate(node=>{const canvas=node as HTMLCanvasElement,gl=canvas.getContext('webgl2')!,data=new Uint8Array(canvas.width*canvas.height*4);gl.readPixels(0,0,canvas.width,canvas.height,gl.RGBA,gl.UNSIGNED_BYTE,data);const colours=new Set<number>();let hash=0;for(let i=0;i<data.length;i+=4){colours.add(data[i]!*65536+data[i+1]!*256+data[i+2]!);hash=(Math.imul(hash,31)+data[i]!+data[i+1]!+data[i+2]!)>>>0;}return{colours:colours.size,hash};});}

test('default is a textured editable 3D viewport with Engine view separate',async({page},info)=>{
 await page.goto('/');await expect(page.getByRole('combobox',{name:'Editing view',exact:true})).toHaveValue('perspective');
 const canvas=page.locator('canvas[aria-label="Editing viewport"]');await expect(canvas).toBeVisible();await expect(page.locator('iframe')).toHaveCount(0);
 await expect(page.locator('canvas[aria-label="Textured 3D geometry"]')).not.toHaveAttribute('data-material-count','0');
 await expect(page.getByRole('alert')).toHaveCount(0);expect((await pixels(page)).colours).toBeGreaterThan(30);
 await page.screenshot({path:info.outputPath('editable-textured-perspective.png')});
 await page.getByRole('combobox',{name:'Editing view',exact:true}).selectOption('orthographic');await expect(canvas).toHaveAttribute('data-view','orthographic');expect((await pixels(page)).colours).toBeGreaterThan(30);
 await page.getByRole('button',{name:'Engine view',exact:true}).click();await expect(page.getByRole('region',{name:'3D game engine preview'})).toHaveAttribute('data-level-slot','0',{timeout:60000});
 await page.getByRole('button',{name:'Editor view',exact:true}).click();await expect(canvas).toHaveAttribute('data-view','orthographic');
});

for(const view of ['perspective','orthographic'])test(`${view} shows both storeys and drags all four height planes with Undo`,async({page},info)=>{
 const fixture=roomHeightFixture(twoStoreyHeightTemplate()),baseline=roomSettings(fixture.document,fixture.from);
 await page.goto('/?view=top');await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
 await page.locator('input[type=file]').setInputFiles({name:'storeys.ab3d',mimeType:'application/zip',buffer:Buffer.from(encodeProject(fixture.document))});
 await page.getByRole('button',{name:new RegExp(`^Room ${fixture.from.id.replace(/^room-/,'')}`)}).click();
 await page.getByRole('combobox',{name:'Editing view',exact:true}).selectOption(view);
 const canvas=page.locator('canvas[aria-label="Editing viewport"]');
 await expect(page.locator('canvas[aria-label="Textured 3D geometry"]')).not.toHaveAttribute('data-material-count','0');await expect(page.getByRole('alert')).toHaveCount(0);
 const camera=await canvas.getAttribute('data-camera');
 for(const storey of ['lower','upper'])for(const plane of ['floor','ceiling'])await expect(page.getByRole('button',{name:`Drag ${storey} ${plane} height`,exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Upper storey',exact:true}).click();await expect(canvas).toHaveAttribute('data-camera',camera!);await expect(page.getByRole('button',{name:'Drag lower floor height',exact:true})).toBeVisible();
 await page.getByRole('button',{name:'Lower storey',exact:true}).click();
 for(const [storey,plane,field] of [['lower','floor','floor'],['lower','ceiling','ceiling'],['upper','floor','upperFloor'],['upper','ceiling','upperCeiling']] as const){
  const handle=page.getByRole('button',{name:`Drag ${storey} ${plane} height`,exact:true}),rect=(await handle.boundingBox())!,before=(await pixels(page)).hash;
  await page.mouse.move(rect.x+rect.width/2,rect.y+rect.height/2);await page.mouse.down();await page.mouse.move(rect.x+rect.width/2,rect.y+rect.height/2+(plane==='floor'?-8:8),{steps:5});await page.mouse.up();
  await expect(page.getByRole('alert')).toHaveCount(0);
  const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save project',exact:true}).click();const path=info.outputPath(`${field}.ab3d`);await(await download).saveAs(path);
  const saved=decodeProject(new Uint8Array(await readFile(path))),settings=roomSettings(saved,saved.roomSource.rooms.find(r=>r.id===fixture.from.id)!);
  expect(settings[field]).not.toBe(baseline[field]);for(const other of ['floor','ceiling','upperFloor','upperCeiling'] as const)if(other!==field)expect(settings[other]).toBe(baseline[other]);
  await expect.poll(async()=>(await pixels(page)).hash).not.toBe(before);
  await page.getByRole('button',{name:'Undo',exact:true}).click();await expect.poll(async()=>(await pixels(page)).hash).toBe(before);
  await expect(canvas).toHaveAttribute('data-camera',camera!);
 }
 await page.screenshot({path:info.outputPath(`${view}-both-storeys.png`)});
});

test('Alt-left drag selects without changing room planes',async({page},info)=>{
 const fixture=roomHeightFixture(twoStoreyHeightTemplate()),mesh=texturedEditorScene(fixture.document),baseline=roomSettings(fixture.document,fixture.from);
 await page.goto('/?view=top');await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
 await page.locator('input[type=file]').setInputFiles({name:'surface-drag.ab3d',mimeType:'application/zip',buffer:Buffer.from(encodeProject(fixture.document))});
 await page.getByRole('combobox',{name:'Editing view',exact:true}).selectOption('perspective');await page.getByRole('button',{name:'Walk',exact:true}).click();
 const canvas=page.locator('canvas[aria-label="Editing viewport"]'),bounds=(await canvas.boundingBox())!;
 for(const kind of ['floor','ceiling'] as const){
  if(kind==='ceiling'){await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2);await page.mouse.down({button:'right'});await page.mouse.move(bounds.x+bounds.width/2,bounds.y+bounds.height/2-90,{steps:5});await page.mouse.up({button:'right'});}
  const camera=JSON.parse((await canvas.getAttribute('data-camera'))!) as EditorCamera;let point:{x:number;y:number}|undefined;
  for(let y=.2;y<.9&&!point;y+=.025)for(let x=.2;x<.9&&!point;x+=.025){const ray=screenRay(x*bounds.width,y*bounds.height,camera,bounds.width,bounds.height),hit=pickSceneFace(mesh.faces.filter(f=>dot(f.normal,ray.direction)<0),ray);if(hit?.face.kind===kind && 'boundary' in hit.face && hit.face.boundary)point={x:x*bounds.width,y:y*bounds.height};}
  expect(point).toBeDefined();await page.keyboard.down('Alt');await page.mouse.move(bounds.x+point!.x,bounds.y+point!.y);await page.mouse.down();await page.mouse.move(bounds.x+point!.x,bounds.y+point!.y+(kind==='floor'?-12:12),{steps:5});await page.mouse.up();await page.keyboard.up('Alt');
  const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save project',exact:true}).click();const path=info.outputPath(`drag-${kind}.ab3d`);await(await download).saveAs(path);
  const saved=decodeProject(new Uint8Array(await readFile(path))),settings=roomSettings(saved,saved.roomSource.rooms.find(r=>r.id===fixture.from.id)!);expect(settings[kind]).toBe(baseline[kind]);
  await expect(page.getByRole('button',{name:'Undo',exact:true})).toBeDisabled();
 }
});

test('texture load errors keep geometry editable and retry restores the real materials',async({page})=>{
 await page.route('**/reference/walls/catalog.json',route=>route.fulfill({status:503,body:'unavailable'}));await page.goto('/');
 await expect(page.getByRole('alert')).toContainText('wall material catalog');await expect(page.locator('canvas[aria-label="Editing viewport"]')).toBeVisible();await expect(page.locator('iframe')).toHaveCount(0);
 await page.unroute('**/reference/walls/catalog.json');await page.getByRole('button',{name:'Retry 3D materials',exact:true}).click();
 await expect(page.locator('canvas[aria-label="Textured 3D geometry"]')).not.toHaveAttribute('data-material-count','0');await expect(page.getByRole('alert')).toHaveCount(0);expect((await pixels(page)).colours).toBeGreaterThan(30);
});
