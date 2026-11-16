#!/usr/bin/env python3
"""Capture the real WebGL weapon/water/HUD composition in Chromium."""
import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emcc', default='emcc')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    work = root / 'build/weapon-water/web'
    work.mkdir(parents=True, exist_ok=True)
    compiler = ([sys.executable, args.emcc] if args.emcc.endswith('.py')
                else [shutil.which(args.emcc) or args.emcc])
    sources = re.findall(r'src/[\w/]+\.c', (root / 'cmake/game_sources.cmake').read_text())
    sources = [p for p in sources if p not in ('src/main.c', 'src/renderer.c', 'src/display.c')]
    command = compiler + ['tests/weapon_water_display.c', 'tests/feedback_weapon_water.c', *sources,
                          '-Isrc', '-Ithird_party', '-O2', '-DAB3D_WEAPON_WATER_SCENE_LIB',
                          '-DAB3D_NO_THREADS', '-DAB3D_WEAPON_WATER_EXTERNAL_STATE',
                          '-DAB3D_ENABLE_FLOOR_COL_FAST=1', '-sUSE_SDL=2',
                          '-sFULL_ES3=1', '-sMIN_WEBGL_VERSION=2', '-sMAX_WEBGL_VERSION=2',
                          '-sALLOW_MEMORY_GROWTH=1', '-sINITIAL_MEMORY=134217728',
                          '-sEXIT_RUNTIME=1', '-sENVIRONMENT=web',
                          '--preload-file', 'amiga/data@/amiga/data',
                          '--preload-file', 'fonts@/fonts', '-o', str(work / 'scene.js')]
    with (work / 'build.log').open('w') as output:
        subprocess.run(command, cwd=root, stdout=output, stderr=subprocess.STDOUT, check=True)
    (work / 'index.html').write_text('''<!doctype html><canvas id="canvas" width="320" height="240"></canvas>
<script>
window.results=[];window.failure=null;
var Module={canvas:document.getElementById('canvas'),arguments:['--gl'],
 print:text=>{console.log(text);window.results.push(text)},
 printErr:text=>console.error(text),onAbort:text=>window.failure=String(text)};
</script><script src="scene.js"></script>''')
    # Resolve Playwright from the editor's existing pinned development dependency.
    runner = work / 'run.cjs'
    runner.write_text(r'''
const http=require('node:http'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require(path.resolve('editor/node_modules/playwright'));
const folder=__dirname,errors=[];
const server=http.createServer((req,res)=>{
  const file=path.resolve(folder,'.'+decodeURIComponent(req.url.split('?')[0]));
  if(!file.startsWith(folder+path.sep)){res.writeHead(403).end();return;}
  const name=file.endsWith(path.sep)?path.join(file,'index.html'):file;
  const type={'.html':'text/html','.js':'text/javascript','.wasm':'application/wasm'}[path.extname(name)];
  if(type)res.setHeader('Content-Type',type);
  fs.createReadStream(name).on('error',()=>res.writeHead(404).end()).pipe(res);
});
(async()=>{
  let browser;
  try {
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    browser=await chromium.launch({headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader']});
    const page=await browser.newPage({viewport:{width:320,height:240}});
    page.on('pageerror',error=>errors.push(String(error)));
    page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
    await page.goto(`http://127.0.0.1:${server.address().port}/index.html`,{timeout:120000});
    await page.waitForFunction(()=>window.failure||window.results.some(v=>v.startsWith('PASS:')),{},{timeout:120000});
    const result=await page.evaluate(()=>({lines:window.results,failure:window.failure}));
    console.log(result.lines.join('\n'));
    if(result.failure||errors.length)throw Error(JSON.stringify({failure:result.failure,errors}));
    if(!result.lines.some(v=>v.includes('GPU (GL_R16UI + shader)')))throw Error('WebGL unpack was not active');
    await page.screenshot({path:path.join(folder,'underwater.png')});
  } finally {if(browser)await browser.close();server.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
''')
    with (work / 'browser.log').open('w') as output:
        subprocess.run([shutil.which('node') or 'node', str(runner)], cwd=root,
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    print('\n'.join(v for v in (work / 'browser.log').read_text().splitlines() if v.startswith('PASS:')))


if __name__ == '__main__':
    main()
