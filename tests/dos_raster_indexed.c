/* Byte-pixel versions of the independent BitMapObj raster/coverage oracles. */
#include "raster_dos.h"
#include "sprite_format.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t random_state=17;
static uint32_t next_value(void)
{
    random_state=random_state*1664525u+1013904223u;
    return random_state;
}

#include "dos_sprite_raster_cases.h"

static void check_indexed_colours(void)
{
    enum { WIDTH=67, HEIGHT=8, PITCH=73, PIXELS=PITCH*(HEIGHT+2)+8 };
    static RendererPixel actual[PIXELS],expected[PIXELS];
    uint8_t wad[18]={0},ptr[16]={0};
    RendererPixel palette[32];int rows[HEIGHT];
    for(unsigned column=0;column<4;column++) {
        unsigned offset=2+column*4,mode=column%3;
        ptr[column*4]=(uint8_t)mode;ptr[column*4+3]=(uint8_t)offset;
        for(unsigned row=0;row<2;row++) {
            unsigned texel=(column+row)%3?31:0;
            sprite_word_write(wad+offset+row*2,(uint16_t)(texel<<(mode*5)));
        }
    }
    for(unsigned y=0;y<HEIGHT;y++)rows[y]=(int)((y/2)%2);
    unsigned checked=0;
    for(unsigned colour=0;colour<256;colour++)for(unsigned clip=0;clip<4;clip++) {
        for(unsigned i=0;i<32;i++)palette[i]=(RendererPixel)colour;
        for(unsigned i=0;i<PIXELS;i++)actual[i]=expected[i]=0x96;
        unsigned width=WIDTH-clip,drawn=0,u=clip*4096u+123u;
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=0;x<width;x++) {
            unsigned column=(u+x*4096u)>>16;if(column>3)column=3;
            const uint8_t *entry=ptr+column*4;
            unsigned address=entry[3]+(unsigned)rows[y]*2;
            unsigned word=(unsigned)wad[address]|((unsigned)wad[address+1]<<8);
            unsigned texel=(word>>(entry[0]*5))&31;
            if(texel){expected[PITCH+3+y*PITCH+x]=(RendererPixel)colour;drawn++;}
        }
        DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),palette,rows,0,u,4096u,4,0};
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+PITCH+3,width,HEIGHT,PITCH,&sprite);
        assert(got.visible==width*HEIGHT && got.drawn==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));checked++;
    }
    printf("PASS: %u indexed colour cases preserve all 256 pixel values, opaque zero/255, source-zero cutouts, repeated rows, byte-copy tails and guards\n",checked);
}

int main(void)
{
    _Static_assert(sizeof(RendererPixel)==1,"indexed byte-pixel oracle");
    freopen("INDEXED.ERR","w",stderr);setvbuf(stderr,NULL,_IONBF,0);
    setvbuf(stdout,NULL,_IOLBF,0);
    check_sprite_columns();check_sprite_rows();check_sprite_batches();
    check_sprite_batch_repeated_rows();check_sprite_narrow_coverage();check_sprite_history();
    check_sprite_band_masks();
    check_sprite_fresh_runs();
    check_sprite_copy_widths();
    check_sprite_dda_rows();
    check_indexed_colours();
    puts("DOS indexed raster regression passed");
    return 0;
}
