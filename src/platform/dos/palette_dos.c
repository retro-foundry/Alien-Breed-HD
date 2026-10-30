#include "palette_dos.h"
#include "io.h"
#include "renderer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
uint8_t dos_c12_to_index[4096];
uint16_t dos_index_to_c12[256];
unsigned dos_palette_epoch;
static SDL_Palette *palette;
static SDL_Window *palette_window;
static int current = -2;

static void fail(const char *operation)
{fprintf(stderr,"fatal: DOS indexed palette %s: %s\n",operation,SDL_GetError());exit(1);}

void dos_palette_attach(SDL_Surface *surface)
{if(!surface || !palette || !SDL_SetSurfacePalette(surface,palette))fail("attach");}
void dos_palette_install(SDL_Window *window)
{palette_window=window;dos_palette_attach(SDL_GetWindowSurface(window));}

void dos_palette_load(int level)
{
    if(level < -1 || level >= 16)fail("level outside source range");
    if(current==level)return;
    char logical[64],path[512];unsigned char header[516];
    snprintf(logical,sizeof(logical),"indexed/palette%02d.bin",level+1);
    io_make_data_path(path,sizeof(path),logical);
    FILE *f=fopen(path,"rb");
    if(!f || fread(header,1,sizeof(header),f)!=sizeof(header) || memcmp(header,"ABP8",4) ||
       fread(dos_c12_to_index,1,4096,f)!=4096 || fgetc(f)!=EOF) {
        fprintf(stderr,"fatal: invalid/missing DOS palette %s\n",path);exit(1);
    }
    fclose(f);
    SDL_Color colors[256];
    for(unsigned i=0;i<256;i++) {
        unsigned c=header[4+i*2]|(header[5+i*2]<<8);
        if(c>4095)fail("invalid source RGB word");
        dos_index_to_c12[i]=c;
        colors[i]=(SDL_Color){((c>>8)&15)*17,((c>>4)&15)*17,(c&15)*17,255};
    }
    for(unsigned i=0;i<4096;i++)if(dos_c12_to_index[i]==255)fail("reserved mask index in color map");
    if(!palette)palette=SDL_CreatePalette(256);
    if(!palette || !SDL_SetPaletteColors(palette,colors,0,256))fail("colors");
    current=level;dos_palette_epoch++;
    if(palette_window)dos_palette_install(palette_window);
    renderer_palette_changed();
    printf("[DOS PALETTE] level=%d epoch=%u; 255 source-derived colors, reserved cutout index\n",level+1,dos_palette_epoch);
}

/* UI sources remain RGB565. Build a native indexed cache on palette changes,
 * preserving cutouts explicitly; SDL performs no per-frame quantization. */
SDL_Surface *dos_palette_convert(SDL_Surface *source)
{
    SDL_Surface *out=SDL_CreateSurface(source->w,source->h,SDL_PIXELFORMAT_INDEX8);
    if(!out || !SDL_LockSurface(source))fail("UI cache");
    dos_palette_attach(out);
    Uint32 key=0;bool masked=SDL_SurfaceHasColorKey(source);
    if(masked && !SDL_GetSurfaceColorKey(source,&key))fail("UI mask");
    for(int y=0;y<source->h;y++) {
        const Uint16 *in=(const Uint16 *)((const Uint8 *)source->pixels+y*source->pitch);
        Uint8 *dst=(Uint8 *)out->pixels+y*out->pitch;
        for(int x=0;x<source->w;x++) {
            unsigned c=in[x];
            unsigned c12=(((c>>12)&15)<<8)|(((c>>7)&15)<<4)|((c>>1)&15);
            dst[x]=(masked && c==key)?255:dos_c12_to_index[c12];
        }
    }
    SDL_UnlockSurface(source);
    if((masked && !SDL_SetSurfaceColorKey(out,true,255)) ||
       !SDL_SetSurfaceBlendMode(out,SDL_BLENDMODE_NONE) || !SDL_SetSurfaceRLE(out,true))fail("UI native cache");
    return out;
}
