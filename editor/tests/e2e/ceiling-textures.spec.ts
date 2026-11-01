import {importedAnimationTables} from '../../src/model/animationBuild';
import {buildGraphicsStreams} from '../../src/model/graphicsBuild';
import {movingControllerCopyFixture} from '../movingControllerCopyFixture';
import {expect,test} from '@playwright/test';
import {readFile} from 'node:fs/promises';
import {decodeProject,encodeProject} from '../../src/model/project';
import {roomSettings} from '../../src/model/roomHeights';
import {texturedEditorScene} from '../../src/model/texturedEditorScene';
import {openLevel} from '../../src/model/level';
import {spatialEdit} from '../../src/model/editorScene';
import {applyCommand} from '../../src/model/commands';
import {mechanismsForRoom} from '../../src/model/mechanismDependencies';
import {engineMechanismTrace} from '../engineCapture';

test('original-room ceiling diagram stays 400px high and keeps textures after doorway height edits',async({page},info)=>{
 await page.goto('/');await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
 await page.getByRole('button',{name:/Zone 004/}).click();
 const floor=page.getByRole('spinbutton',{name:'Lower floor height',exact:true});await floor.fill('240');await page.getByRole('button',{name:'Apply heights',exact:true}).click();
 const svg=page.getByRole('group',{name:'Room height section',exact:true});await svg.scrollIntoViewIfNeeded();expect((await svg.boundingBox())!.height).toBe(400);
 const texture=page.locator('canvas[aria-label="Textured 3D geometry"]');await expect(texture).not.toHaveAttribute('data-material-count','0');
 const handle=page.getByRole('slider',{name:'Lower ceiling handle',exact:true});const before=Number(await handle.getAttribute('aria-valuenow'));
 const circle=(await handle.locator('circle').boundingBox())!;await page.mouse.move(circle.x+circle.width/2,circle.y+circle.height/2);await page.mouse.down();await page.mouse.move(circle.x+circle.width/2,circle.y+circle.height/2+108,{steps:8});await page.mouse.up();
 const height=Number(await handle.getAttribute('aria-valuenow'));expect(height).toBeGreaterThan(before);
 await expect(page.locator('.spatial-material-error')).toHaveCount(0);await expect(texture).not.toHaveAttribute('data-material-count','0');expect((await svg.boundingBox())!.height).toBe(400);
 const download=page.waitForEvent('download');await page.getByRole('button',{name:'Save project',exact:true}).click();const path=info.outputPath('ceiling-portal.ab3d');await(await download).saveAs(path);
 const saved=decodeProject(new Uint8Array(await readFile(path))),room=saved.roomSource.rooms.find(r=>r.originZone===4)!;expect(roomSettings(saved,room)).toMatchObject({floor:240,ceiling:height});expect(saved.roomAppearances).toBeUndefined();expect(texturedEditorScene(saved).faces.length).toBeGreaterThan(0);
 await page.getByRole('button',{name:'Undo',exact:true}).click();await expect(handle).toHaveAttribute('aria-valuenow',String(before));await expect(page.locator('.spatial-material-error')).toHaveCount(0);
 await page.getByRole('button',{name:'Redo',exact:true}).click();await expect(handle).toHaveAttribute('aria-valuenow',String(height));await expect(texture).not.toHaveAttribute('data-material-count','0');
 await page.screenshot({path:info.outputPath('fixed-height-diagram-textured.png')});
});

test('height-only edits beside an original door retain its native animation attachments',async({page})=>{
 test.setTimeout(60000);
 const baseline=movingControllerCopyFixture('door').baseline,tables=importedAnimationTables(baseline).tables,record=tables.doors![0]!,header=record.header.slice();
 header[16]=4; // Native unconditional opening mode, as used by the existing controller-copy motion fixtures.
 const document=openLevel('Door neighbour height',{...baseline.files,graphics:buildGraphicsStreams(baseline,undefined,undefined,{animations:{...tables,doors:[{...record,header}]}})}),door=document.mechanisms.find(m=>m.kind==='door')!;
 const target=document.zones[door.zone]!,neighbor=target.boundaryIds.map(id=>document.lines[id]!.connect).find(id=>id>=0&&!mechanismsForRoom(document,id).length&&document.zones[id]!.floor-document.zones[id]!.ceiling>64)!;
 expect(neighbor).toBeDefined();const room=document.roomSource.rooms.find(r=>r.originZone===neighbor)!;
 let source=applyCommand(document,spatialEdit(document,{kind:'height',room:room.id,field:'ceiling',value:roomSettings(document,room).ceiling+4}));
 await page.goto('/?view=top');await expect(page.locator('.project-picker strong')).toHaveText('Level 1');await page.locator('input[type=file]').setInputFiles({name:'door-height.ab3d',mimeType:'application/zip',buffer:Buffer.from(encodeProject(source))});
 await page.getByRole('button',{name:'Engine view',exact:true}).click();const preview=page.getByRole('region',{name:'3D game engine preview'});await expect(preview.getByRole('status')).toContainText('zones loaded',{timeout:60000});
 await preview.getByRole('button',{name:'Build & play',exact:true}).click();await expect(preview.getByRole('status')).toHaveText('Playtest running');
 const trace=await engineMechanismTrace(page,'door',door.id,false);expect(trace.at(-1)!.current).toBe(door.top);for(const state of trace)expect(state.unsynchronizedWalls).toBe(0);
 await preview.getByRole('button',{name:'Stop',exact:true}).click();await expect(preview.getByRole('alert')).toHaveCount(0);
});
