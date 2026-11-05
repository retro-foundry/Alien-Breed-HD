import { expect, test } from '@playwright/test';
import { nativeRoomCopyFixture } from '../nativeRoomCopyFixture';
import { buildPlayableLevel } from '../../src/model/playableBuild';
import { encodeProject, decodeProject } from '../../src/model/project';
import { captureEngine, enginePickDetails } from '../engineCapture';
import { readFile } from 'node:fs/promises';

for (const type of [1,2]) test(`native ${type===1?'floor':'ceiling'} normal materials paint, save and undo while unused bump choices stay disabled`,async({page},testInfo)=>{
  test.setTimeout(90000);
  const {document,copy}=nativeRoomCopyFixture(), built=buildPlayableLevel(document,-1), cell=built.topology!.cells.find(c=>c.sourceId===copy.id)!, zone=built.document.zones[cell.id]!;
  const camera={zone:cell.id,x:Math.round(copy.vertices.reduce((n,p)=>n+p.x,0)/copy.vertices.length),z:Math.round(copy.vertices.reduce((n,p)=>n+p.z,0)/copy.vertices.length),upper:0,angle:0,height:type===1?zone.floor-1:zone.ceiling+1};
  await page.goto('/?view=top'); await expect(page.locator('.project-picker strong')).toHaveText('Level 1');
  await page.locator('input[type=file]').setInputFiles({name:'banks.ab3d',mimeType:'application/zip',buffer:Buffer.from(encodeProject(document))});
  await page.getByRole('button',{name:'Engine view',exact:true}).click();
  const preview=page.getByRole('region',{name:'3D game engine preview'});
  await expect(preview.getByRole('status')).toContainText('zones loaded',{timeout:60000});
  let checksum=await captureEngine(page,camera);
  const points=Array.from({length:361},(_,i)=>({x:(i%19+1)/20,y:(Math.floor(i/19)+1)/20}));
  const canvas=page.frameLocator('iframe[title="Alien Breed game engine"]').locator('canvas');
  for(const [mode,opcode] of [['normal',type]] as const){
    const picks=await enginePickDetails(page,points), index=picks.findIndex(p=>p.zone===cell.id && p.surface && [type,type+7,type+9].includes(p.surface.type));
    expect(index).toBeGreaterThanOrEqual(0); const bounds=(await canvas.boundingBox())!;
    await canvas.click({position:{x:points[index]!.x*bounds.width,y:points[index]!.y*bounds.height}});
    await page.getByRole('button',{name:'Paint this surface',exact:true}).click();
    const dialog=page.getByRole('dialog',{name:'Paint floor or ceiling material'});
    const modes=dialog.getByRole('combobox',{name:'Surface material mode'});
    await expect(modes.locator('option[value=bump]')).toHaveJSProperty('disabled',true);
    await expect(modes.locator('option[value=smooth]')).toHaveJSProperty('disabled',true);
    await modes.selectOption(mode);
    await dialog.getByRole('combobox',{name:'Surface pattern',exact:true}).selectOption('257');
    await expect(dialog.getByRole('group',{name:'Floor pattern choices'}).getByRole('img')).toHaveCount(16);
    const revision=(await preview.getAttribute('data-revision'))!;
    await dialog.getByRole('button',{name:'Apply surface material',exact:true}).click();
    await expect(preview).not.toHaveAttribute('data-revision',revision); await expect(preview.getByRole('status')).toContainText('zones loaded');
    const next=await captureEngine(page,camera); expect(next).not.toBe(checksum);
    const after=await enginePickDetails(page,points); expect(after.some(p=>p.zone===cell.id && p.surface?.type===opcode)).toBe(true);
    const download=page.waitForEvent('download'); await page.getByRole('button',{name:'Save project',exact:true}).click();
    const path=testInfo.outputPath(`${mode}.ab3d`); await(await download).saveAs(path);
    const saved=decodeProject(new Uint8Array(await readFile(path)));
    expect(saved.roomSource.rooms.find(r=>r.id===copy.id)!.polygonMaterials!.some(p=>p.mode===mode)).toBe(true);
    const undoRevision=(await preview.getAttribute('data-revision'))!; await page.getByRole('button',{name:'Undo',exact:true}).click();
    await expect(preview).not.toHaveAttribute('data-revision',undoRevision); await expect(preview.getByRole('status')).toContainText('zones loaded'); expect(await captureEngine(page,camera)).toBe(checksum);
    const redoRevision=(await preview.getAttribute('data-revision'))!; await page.getByRole('button',{name:'Redo',exact:true}).click();
    await expect(preview).not.toHaveAttribute('data-revision',redoRevision); await expect(preview.getByRole('status')).toContainText('zones loaded'); expect(await captureEngine(page,camera)).toBe(next); checksum=next;
  }
});
