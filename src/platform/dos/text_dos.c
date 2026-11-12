/* AB3DI.s:PlayTheGame selects TextCop for the original 80-character blurb.
 * Use VGA text memory here; the 4 MB port keeps its gameplay surfaces intact. */
#include <SDL.h>
#include "text_dos.h"
#include <dpmi.h>
#include <sys/movedata.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

extern bool Ab3dDos_BeginTextScreen(SDL_Window *window);
extern bool Ab3dDos_EndTextScreen(SDL_Window *window);
static int active;

static void text_fail(const char *operation)
{
    fprintf(stderr,"fatal: DOS briefing %s: %s\n",operation,SDL_GetError());exit(1);
}

void dos_text_begin(SDL_Window *window)
{
    if(active)return;
    if(!Ab3dDos_BeginTextScreen(window))text_fail("mode selection");
    active=1;
}

void dos_text_end(SDL_Window *window)
{
    if(!active)return;
    if(!Ab3dDos_EndTextScreen(window))text_fail("graphics restoration");
    active=0;
}

int dos_text_active(void) { return active; }

void dos_text_present(const char *const *lines,unsigned count,int alpha)
{
    if(!active || count>DOS_TEXT_ROWS) {
        SDL_SetError("expected at most 25 rows on an active 80-column screen");text_fail("layout");
    }
    uint16_t cells[DOS_TEXT_ROWS][DOS_TEXT_COLUMNS];
    for(unsigned y=0;y<DOS_TEXT_ROWS;y++)for(unsigned x=0;x<DOS_TEXT_COLUMNS;x++)cells[y][x]=0x0720;
    unsigned top=(DOS_TEXT_ROWS-count)/2;
    for(unsigned row=0;row<count;row++) {
        const char *line=lines[row];if(!line)continue;
        while(*line==' ' || *line=='\t')line++;
        size_t length=strlen(line);
        while(length && (line[length-1]==' ' || line[length-1]=='\t'))length--;
        if(length>DOS_TEXT_COLUMNS) {
            SDL_SetError("row %u has %u characters; maximum is 80",row,(unsigned)length);text_fail("line width");
        }
        unsigned left=(DOS_TEXT_COLUMNS-(unsigned)length)/2;
        if(alpha)for(unsigned x=0;x<length;x++)cells[top+row][left+x]=(uint16_t)(0x0700|(uint8_t)line[x]);
    }
    dosmemput(cells,sizeof(cells),0xB8000);
}

void dos_text_read(uint16_t cells[DOS_TEXT_ROWS][DOS_TEXT_COLUMNS])
{
    if(!active){SDL_SetError("text readback requires VGA briefing mode");text_fail("readback");}
    dosmemget(0xB8000,DOS_TEXT_ROWS*DOS_TEXT_COLUMNS*sizeof(uint16_t),cells);
}

void dos_text_check_graphics_palette(SDL_Surface *surface)
{
    SDL_Palette *palette=SDL_GetSurfacePalette(surface);
    if(!palette)return;
    for(int i=0;i<palette->ncolors;i++) {
        __dpmi_regs regs={0};regs.x.ax=0x1015;regs.x.bx=(unsigned)i;
        __dpmi_int(0x10,&regs);
        SDL_Color color=palette->colors[i];
        if(regs.h.dh!=(color.r>>2) || regs.h.ch!=(color.g>>2) || regs.h.cl!=(color.b>>2)) {
            SDL_SetError("graphics palette entry %d was not restored",i);text_fail("DAC readback");
        }
    }
}

static void capture_fail(const char *filename)
{
    SDL_SetError("%s: %s",filename,strerror(errno));text_fail("capture write");
}

/* Diagnostic reconstruction from the actual text cells and VGA ROM glyphs.
 * Mode 3 uses nine-dot cells; printable ASCII leaves the ninth column blank. */
void dos_text_capture(const char *filename)
{
    uint16_t cells[DOS_TEXT_ROWS][DOS_TEXT_COLUMNS];uint8_t font[4096];
    __dpmi_regs regs={0};regs.x.ax=0x1130;regs.h.bh=6;
    __dpmi_int(0x10,&regs);
    if(regs.x.cx!=16){SDL_SetError("VGA briefing font is not 16 scanlines");text_fail("font readback");}
    dos_text_read(cells);
    dosmemget(((unsigned long)regs.x.es<<4)+regs.x.bp,sizeof(font),font);
    enum { WIDTH=80*9,HEIGHT=25*16,STRIDE=WIDTH*3 };
    uint8_t header[54]={0},pixels[STRIDE];char file_buffer[4096];
    /* Read the actual VGA attribute/DAC mapping used by text colour 7. */
    memset(&regs,0,sizeof(regs));regs.x.ax=0x1007;regs.h.bl=7;
    __dpmi_int(0x10,&regs);
    unsigned index=regs.h.bh;
    memset(&regs,0,sizeof(regs));regs.x.ax=0x1015;regs.x.bx=index;
    __dpmi_int(0x10,&regs);
    uint8_t red=regs.h.dh*255/63,green=regs.h.ch*255/63,blue=regs.h.cl*255/63;
    unsigned size=54+STRIDE*HEIGHT;
    header[0]='B';header[1]='M';
    for(unsigned i=0;i<4;i++)header[2+i]=(uint8_t)(size>>(8*i));
    header[10]=54;header[14]=40;header[26]=1;header[28]=24;
    for(unsigned i=0;i<4;i++){header[18+i]=(uint8_t)(WIDTH>>(8*i));header[22+i]=(uint8_t)(HEIGHT>>(8*i));}
    FILE *file=fopen(filename,"wb");if(!file){SDL_SetError("cannot create %s",filename);text_fail("capture");}
    if(setvbuf(file,file_buffer,_IOFBF,sizeof(file_buffer)) || fwrite(header,1,sizeof(header),file)!=sizeof(header))capture_fail(filename);
    for(int y=HEIGHT-1;y>=0;y--) {
        for(unsigned x=0;x<WIDTH;x++) {
            uint16_t cell=cells[y/16][x/9];
            unsigned ink=x%9<8 && (font[(cell&255)*16+y%16]&(128u>>(x%9)));
            pixels[x*3]=ink?blue:0;pixels[x*3+1]=ink?green:0;pixels[x*3+2]=ink?red:0;
        }
        if(fwrite(pixels,1,sizeof(pixels),file)!=sizeof(pixels))capture_fail(filename);
    }
    if(fclose(file))capture_fail(filename);
}
