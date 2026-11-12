#include "sprite_alpha_dos.h"

static void check_sprite_alpha_hints(void)
{
    enum { HEIGHT=136,PIXELS=320*HEIGHT };
    FILE *file=fopen("ALPHA.BIN","rb");assert(file);
    char magic[4];unsigned header[2];
    assert(fread(magic,1,4,file)==4 && !memcmp(magic,"ALF1",4));
    assert(fread(header,4,2,file)==2 && header[1]==DOS_SPRITE_ALPHA_ROWS);
    RendererPixel *actual=malloc((PIXELS+16)*sizeof(*actual));
    RendererPixel *expected=malloc((PIXELS+16)*sizeof(*expected));
    assert(actual && expected);
    int rows[HEIGHT];
    for(unsigned i=0;i<128;i++)rows[i]=(int)(i/2);
    const int edges[8]={-1,63,64,65,127,255,INT_MAX,INT_MIN};
    memcpy(rows+128,edges,sizeof(edges));
    RendererPixel palette[32];
    for(unsigned i=0;i<32;i++)palette[i]=(RendererPixel)(i==31?0:i*1973u);
    unsigned checked=0,draws=0,empty_rows=0;
    for(unsigned bank=0;bank<header[0];bank++) {
        unsigned h[4];assert(fread(h,4,4,file)==4);
        unsigned type=h[0],wad_bytes=h[1],ptr_bytes=h[2],queries=h[3];
        uint8_t *wad=malloc(wad_bytes),*ptr=malloc(ptr_bytes);assert(wad && ptr);
        assert(fread(wad,1,wad_bytes,file)==wad_bytes && fread(ptr,1,ptr_bytes,file)==ptr_bytes);
        assert(!dos_sprite_alpha_rows(type,65536u,0,64));
        assert(!dos_sprite_alpha_rows(type,0,256u,64));
        assert(!dos_sprite_alpha_rows(type,0,0,256u));
        assert(!dos_sprite_alpha_rows(UINT_MAX,0,0,64));
        for(unsigned query=0;query<queries;query++) {
            unsigned q[3];assert(fread(q,4,3,file)==3);
            unsigned offset=q[0],down=q[1],columns=q[2];assert(columns<=320);
            unsigned available=ptr_bytes>offset?(ptr_bytes-offset)/4:0;
            const uint32_t *mask=dos_sprite_alpha_rows(type,offset,down,columns);
            uint32_t truth[2]={0,0};
            for(unsigned row=0;row<DOS_SPRITE_ALPHA_ROWS;row++) {
                for(unsigned col=0;col<columns && col<available;col++) {
                    const uint8_t *entry=ptr+offset+col*4;
                    unsigned mode=entry[0],start=((unsigned)entry[1]<<16)|((unsigned)entry[2]<<8)|entry[3];
                    if((!mode && !start) || start>=wad_bytes)continue;
                    unsigned words=(wad_bytes-start)/2;
                    if(words<=down || row>words-1-down)continue;
                    unsigned value=sprite_word_read(wad+start+2*(down+row));
                    if((value>>(!mode?0:mode==1?5:10))&31)truth[row>>5]|=1u<<(row&31);
                }
                if(!(truth[row>>5]&(1u<<(row&31))))empty_rows++;
            }
            if(truth[0]==UINT32_MAX && truth[1]==UINT32_MAX)assert(!mask);
            else assert(mask && mask[0]==truth[0] && mask[1]==truth[1]);
            for(unsigned batch=0;batch<2;batch++) {
                for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=(RendererPixel)0xa55a;
                DosSpriteBatch actual_mask,expected_mask;
                memset(&expected_mask,0,sizeof(expected_mask));
                memset(expected_mask.open_words,10,sizeof(expected_mask.open_words));
                unsigned visible=0,drawn=0;
                for(unsigned x=0;available && x<columns;x++) {
                    unsigned col=x<available?x:available-1;
                    const uint8_t *entry=ptr+offset+col*4;
                    unsigned mode=entry[0],start=((unsigned)entry[1]<<16)|((unsigned)entry[2]<<8)|entry[3];
                    if((!mode && !start) || start>=wad_bytes)continue;
                    unsigned words=(wad_bytes-start)/2;
                    if(words<=down)continue;
                    visible+=HEIGHT;
                    for(unsigned y=0;y<HEIGHT;y++) {
                        if((unsigned)rows[y]>words-1-down)continue;
                        unsigned value=sprite_word_read(wad+start+2*(down+(unsigned)rows[y]));
                        unsigned texel=(value>>(!mode?0:mode==1?5:10))&31;
                        if(!texel)continue;
                        expected[8+y*320+x]=palette[texel];drawn++;
                        expected_mask.covered[y][x/32]|=1u<<(x%32);
                    }
                }
                for(unsigned y=0;y<HEIGHT;y++)for(unsigned word=0;word<10;word++)
                    if(expected_mask.covered[y][word]==UINT32_MAX)expected_mask.open_words[y]--;
                dos_raster_sprite_batch_end();
                if(batch)dos_raster_sprite_batch_begin(actual+8,&actual_mask);
                DosSpriteRows sprite={wad,ptr,wad_bytes,ptr_bytes,palette,rows,offset,0,65536u,columns,down};
                sprite.nonempty_rows=mask;
                DosSpriteCoverage got=dos_raster_sprite_rows(actual+8,columns,HEIGHT,320,&sprite);
                dos_raster_sprite_batch_end();
                assert(got.visible==visible && got.drawn==drawn);
                assert(!memcmp(actual,expected,(PIXELS+16)*sizeof(*actual)));
                if(batch)assert(!memcmp(&actual_mask,&expected_mask,sizeof(actual_mask)));
                draws++;
            }
            checked++;
        }
        free(ptr);free(wad);
    }
    assert(fgetc(file)==EOF);fclose(file);free(expected);free(actual);
    printf("PASS: %u original sprite alpha recipes, %u guarded painter/batch images and %u empty rows retain independent pixels, masks/counts, black, repeated bands and row-cap boundaries\n",checked,draws,empty_rows);
}
