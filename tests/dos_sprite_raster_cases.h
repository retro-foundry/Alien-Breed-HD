#ifndef AB3D_DOS_SPRITE_RASTER_CASES_H
#define AB3D_DOS_SPRITE_RASTER_CASES_H

/* Shared independent BitMapObj pixel/coverage oracles for both DOS pixel sizes. */
static void check_sprite_columns(void)
{
    enum { CAPACITY=41*242+8 };
    static RendererPixel actual[CAPACITY],expected[CAPACITY],palette[32];
    static uint8_t packed[513],native[513],texels[256];
    static int rows[240];
    static const unsigned counts[]={0,1,2,3,7,31,127,239,240};
    static const unsigned strides[]={1,2,3,7,41};
    static const unsigned shifts[]={0,5,10};
    for(unsigned i=0;i<sizeof(packed);i++)packed[i]=(uint8_t)next_value();
    for(unsigned i=0;i<256;i++){native[1+i*2]=packed[2+i*2];native[2+i*2]=packed[1+i*2];}
    for(unsigned i=0;i<32;i++)palette[i]=(RendererPixel)(i==1?0:i*1973);
    assert(!dos_raster_sprite_column(NULL,0,0,NULL,NULL,NULL,0,0));
    assert(!dos_raster_sprite_texel_column(NULL,0,0,NULL,NULL,NULL,0));
    for(unsigned test=0;test<384;test++) {
        unsigned count=counts[test%9],stride=strides[test%5],shift=shifts[test%3];
        unsigned limit=test%256,drawn=0;
        const uint8_t *source=packed+1; /* Include unaligned packed words. */
        for(unsigned i=0;i<256;i++) {
            unsigned word=((unsigned)source[i*2]<<8)|source[i*2+1];
            texels[i]=(uint8_t)((word>>shift)&31);
        }
        for(unsigned y=0;y<240;y++) {
            rows[y]=(int)(next_value()%340);
            if(y%31==0)rows[y]=-1;
            if(y%47==0)rows[y]=INT_MAX;
        }
        for(unsigned i=0;i<CAPACITY;i++)actual[i]=expected[i]=(RendererPixel)next_value();
        for(unsigned y=0;y<count;y++) {
            if((unsigned)rows[y]>limit)continue;
            const uint8_t *p=source+rows[y]*2;
            unsigned word=((unsigned)p[0]<<8)|p[1],texel=(word>>shift)&31;
            if(texel){expected[3+y*stride]=palette[texel];drawn++;}
        }
        assert(dos_raster_sprite_column(actual+3,count,stride,native+1,palette,rows,limit,shift)==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
        /* Both source representations must write exactly the same pixels.
         * Reset only the column so all surrounding guards remain checked. */
        for(unsigned y=0;y<count;y++)actual[3+y*stride]=((RendererPixel)0x6b5a);
        memcpy(expected,actual,sizeof(actual));
        for(unsigned y=0;y<count;y++)if((unsigned)rows[y]<=limit && texels[rows[y]])
            expected[3+y*stride]=palette[texels[rows[y]]];
        assert(dos_raster_sprite_texel_column(actual+3,count,stride,texels,palette,rows,limit)==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
    }
    puts("PASS: 768 packed/decoded sprite columns preserve clipping, modes, cutouts, opaque black, bounds and guards");
}

static void check_sprite_rows(void)
{
    static RendererPixel actual[324*242],expected[324*242];
    static uint8_t wad[4099],native[4099],ptr[131];
    static RendererPixel palette[32];
    static int rows[240];
    static const unsigned widths[]={0,1,2,3,7,31,127,319,320};
    uint32_t seed=0x13579bdf;unsigned checked=0;
    for(unsigned i=0;i<sizeof(wad);i++) {
        seed=seed*1664525u+1013904223u;wad[i]=(uint8_t)(seed>>24);
    }
    for(unsigned i=0;i+1<sizeof(wad);i+=2){native[i]=wad[i+1];native[i+1]=wad[i];}
    /* The isolated final column deliberately starts at an odd address. Its
     * native words do not overlap any other sampled column in this fixture. */
    native[4097]=wad[4098];native[4098]=wad[4097];
    for(unsigned i=0;i<32;i++)palette[i]=(RendererPixel)(i==1?0:i*1973u);
    for(unsigned test=0;test<540;test++) {
        unsigned width=widths[test%9],height=test==287?240:1+(test%61);
        for(unsigned i=1;i<32;i++) {
            /* Adjacent opaque samples may share a colour, including black.
             * Alpha still comes from the source texel rather than that colour. */
            palette[i]=(RendererPixel)(test%4==0?0:test%4==1?1:
                                       test%4==2?(i/8)*1973:i*1973);
        }
        unsigned pitch=width+4;
        for(unsigned i=0;i<32;i++) {
            unsigned offset=2+i*64;
            if(i==0 && test%3==0)offset=0;
            if(i==31)offset=4097;
            ptr[i*4]=(uint8_t)((i+test)%4);
            ptr[i*4+1]=(uint8_t)(offset>>16);ptr[i*4+2]=(uint8_t)(offset>>8);ptr[i*4+3]=(uint8_t)offset;
        }
        for(unsigned i=0;i<height;i++)rows[i]=(int)(i/(1+test%7))-(test%11==0?1:0);
        for(unsigned i=0;i<sizeof(actual)/sizeof(*actual);i++)actual[i]=expected[i]=((RendererPixel)0xa55a);
        DosSpriteRows sprite={native,ptr,sizeof(wad)-(test%5),sizeof(ptr)-(test%7),palette,rows,
                              test%4,(test%3)*65536u+333u,test%4==0?0:65536u/(1+test%8),
                              32,test%13==0?65535u:test%3};
        if(test>=288) {
            /* Compare the row renderer with the independent per-pixel DDA
             * across fractional boundaries, clamped columns and 32-bit wraps. */
            static const uint32_t steps[]={0,1,2,3,127,4095,4096,4097,8191,8192,8193,
                16383,16384,16385,32767,32768,32769,65535,65536,65537,UINT32_MAX};
            static const uint32_t origins[]={0,1,65534,65535,65536,65537,31*65536u,
                                             32*65536u-1u,32*65536u,UINT32_MAX-65535u,
                                             UINT32_MAX-1u,UINT32_MAX};
            unsigned n=test-288;
            sprite.du=steps[n%21];sprite.u=origins[n/21];
            if(n%7==0)sprite.columns=1;
        }
        DosSpriteCoverage coverage={0,0};
        unsigned available=(sprite.ptr_bytes-sprite.ptr_offset)/4;
        for(unsigned x=0;x<width;x++) {
            unsigned column=(sprite.u+x*sprite.du)>>16;
            if(column>=sprite.columns)column=sprite.columns-1;
            if(column>=available)column=available-1;
            const uint8_t *entry=ptr+sprite.ptr_offset+column*4;
            unsigned offset=((unsigned)entry[1]<<16)|((unsigned)entry[2]<<8)|entry[3];
            unsigned mode=entry[0];
            if((!mode && !offset) || (uint64_t)offset+(uint64_t)sprite.down_strip*2+2>sprite.wad_bytes)continue;
            coverage.visible+=height;
            for(unsigned y=0;y<height;y++) {
                if(rows[y]<0)continue;
                uint64_t address=(uint64_t)offset+((uint64_t)sprite.down_strip+(unsigned)rows[y])*2;
                if(address+2>sprite.wad_bytes)continue;
                /* Malformed PTR views can point at odd byte offsets. Their
                 * bounds and samples follow the native transport ABI, while
                 * valid authored columns retain the original word values. */
                unsigned word=(unsigned)native[address]|((unsigned)native[address+1]<<8);
                unsigned texel=(word>>(!mode?0:mode==1?5:10))&31;
                if(texel){expected[(y+1)*pitch+x+2]=palette[texel];coverage.drawn++;}
            }
        }
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+pitch+2,width,height,pitch,&sprite);
        if(memcmp(actual,expected,sizeof(actual))) {
            for(unsigned i=0;i<sizeof(actual)/sizeof(*actual);i++)if(actual[i]!=expected[i]) {
                fprintf(stderr,"Sprite row case %u, pixel %u, width=%u height=%u: actual=%04x expected=%04x\n",
                        test,i,width,height,(unsigned)actual[i],(unsigned)expected[i]);break;
            }
        }
        assert(got.visible==coverage.visible && got.drawn==coverage.drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
        checked++;
    }
    /* Exercise both run states at their first/last pixel and the full
     * 160-span capacity. Texel zero alone is transparent, even with a
     * black palette; repeated and rejected rows must reset run state. */
    static uint8_t run_ptr[320*4],run_wad[320*2+2];
    static const unsigned run_widths[]={1,2,3,319,320};
    rows[0]=rows[2]=rows[3]=0;rows[1]=-1;
    for(unsigned pattern=0;pattern<4;pattern++)for(unsigned w=0;w<5;w++) {
        unsigned width=run_widths[w],pitch=width+4,drawn=0;
        palette[31]=(RendererPixel)(w&1?0:1);
        for(unsigned i=0;i<sizeof(actual)/sizeof(*actual);i++)actual[i]=expected[i]=((RendererPixel)0xa55a);
        for(unsigned x=0;x<320;x++) {
            unsigned offset=2+x*2;
            run_ptr[x*4]=run_ptr[x*4+1]=0;
            run_ptr[x*4+2]=(uint8_t)(offset>>8);run_ptr[x*4+3]=(uint8_t)offset;
            int opaque=pattern==1 || (pattern>=2 && (x&1)==pattern-2);
            run_wad[offset]=(uint8_t)(opaque?31:0);run_wad[offset+1]=0;
            if(x<width && opaque)for(unsigned y=0;y<4;y++)if(rows[y]>=0) {
                expected[(y+1)*pitch+x+2]=palette[31];drawn++;
            }
        }
        DosSpriteRows sprite={run_wad,run_ptr,sizeof(run_wad),sizeof(run_ptr),palette,rows,
                              0,0,65536u,320,0};
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+pitch+2,width,4,pitch,&sprite);
        assert(got.visible==width*4 && got.drawn==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));checked++;
    }
    printf("PASS: %u sprite row rasters preserve packed modes, clipping origins, repeated rows, black/transparent texels, run boundaries/capacity, bounds and guards\n",checked);
}

static void check_sprite_batches(void)
{
    enum { PIXELS=320*240, LAYERS=12 };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16],palettes[LAYERS][32];
    static uint8_t touched[PIXELS];
    static uint8_t wad[4098],native[4098],ptr[128],decoded[64];
    static int rows[LAYERS][240];
    struct Rect { unsigned x,y,w,h,u,du,down; } rect[LAYERS];
    DosSpriteBatch scratch;
    DosSpriteHistory history;
    uint32_t random=0x87451a9bu;
    for(unsigned test=0;test<32;test++) {
        for(unsigned i=0;i<sizeof(wad);i++) {
            random=random*1664525u+1013904223u;wad[i]=(uint8_t)(random>>24);
        }
        /* A fully opaque source also checks black ownership and complete
         * covered-row rejection, including reset between separate batches. */
        if(test%4==0)for(unsigned i=0;i<sizeof(wad);i+=2){wad[i]=4;wad[i+1]=33;}
        for(unsigned i=0;i<sizeof(wad);i+=2){native[i]=wad[i+1];native[i+1]=wad[i];}
        for(unsigned x=0;x<32;x++) {
            unsigned offset=2+x*128;
            ptr[x*4]=(uint8_t)((x+test)%3);ptr[x*4+1]=0;
            ptr[x*4+2]=(uint8_t)(offset>>8);ptr[x*4+3]=(uint8_t)offset;
        }
        for(unsigned layer=0;layer<LAYERS;layer++) {
            struct Rect *r=&rect[layer];
            random=random*1664525u+1013904223u;
            r->x=layer==LAYERS-1 && test%4==0?0:(random>>16)%320;
            r->w=1+(random>>8)%(320-r->x);
            r->y=(random>>24)%100;r->h=1+random%(240-r->y);
            if(layer==LAYERS-1 && test%4==0){r->w=320;r->y=0;r->h=240;}
            r->u=(layer%3)*65536u+333u;r->du=(32u*65536u)/r->w;
            r->down=layer%3;
            for(unsigned y=0;y<r->h;y++)rows[layer][y]=(int)(y*61/r->h)-(test%3==0?1:0);
            for(unsigned i=0;i<32;i++)palettes[layer][i]=(RendererPixel)(i==1?0:i==2?1:((i+layer*32)*1973u)&0x7fff);
        }
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=((RendererPixel)0xa55a);
        memset(touched,0,sizeof(touched));
        /* Independent original-order pixel sampler: later opaque pixels win. */
        for(unsigned layer=0;layer<LAYERS;layer++) {
            const struct Rect *r=&rect[layer];
            for(unsigned y=0;y<r->h;y++)for(unsigned x=0;x<r->w;x++) {
                int row=rows[layer][y];if(row<0)continue;
                unsigned column=(r->u+x*r->du)>>16;if(column>=32)column=31;
                const uint8_t *entry=ptr+column*4;
                unsigned offset=((unsigned)entry[1]<<16)|((unsigned)entry[2]<<8)|entry[3];
                unsigned address=offset+((unsigned)row+r->down)*2;
                if(address+2>sizeof(wad))continue;
                unsigned word=((unsigned)wad[address]<<8)|wad[address+1];
                unsigned texel=(word>>(entry[0]*5))&31;
                if(texel){expected[8+(r->y+y)*320+r->x+x]=palettes[layer][texel];touched[(r->y+y)*320+r->x+x]=1;}
            }
        }
        dos_raster_sprite_batch_begin(actual+8,&scratch);
        if(test&1)dos_raster_sprite_batch_history_begin(&history);
        unsigned written=0;
        for(unsigned step=0;step<LAYERS;step++) {
            unsigned layer=LAYERS-1-step;const struct Rect *r=&rect[layer];
            RendererPixel *out=actual+8+r->y*320+r->x;
            if(layer%3!=0 || test%4==0) {
                DosSpriteRows sprite={native,ptr,sizeof(wad),sizeof(ptr),palettes[layer],rows[layer],0,r->u,r->du,32,r->down};
                written+=dos_raster_sprite_rows(out,r->w,r->h,320,&sprite).drawn;
            } else for(unsigned x=0;x<r->w;x++) {
                unsigned column=(r->u+x*r->du)>>16;if(column>=32)column=31;
                const uint8_t *entry=ptr+column*4;
                unsigned offset=((unsigned)entry[1]<<16)|((unsigned)entry[2]<<8)|entry[3];
                const uint8_t *source=wad+offset+r->down*2;
                unsigned limit=(sizeof(wad)-offset)/2-1-r->down,shift=entry[0]*5;
                if(x&1)written+=dos_raster_sprite_column(out+x,r->h,320,native+offset+r->down*2,palettes[layer],rows[layer],limit,shift);
                else {
                    for(unsigned y=0;y<64 && y<=limit;y++)decoded[y]=(uint8_t)((((unsigned)source[y*2]<<8)|source[y*2+1])>>shift)&31;
                    written+=dos_raster_sprite_texel_column(out+x,r->h,320,decoded,palettes[layer],rows[layer],limit<63?limit:63);
                }
            }
        }
        dos_raster_sprite_batch_end();
        assert(!memcmp(actual,expected,sizeof(actual)));
        unsigned changed=0;for(unsigned i=0;i<PIXELS;i++)changed+=touched[i];
        assert(written==changed);
        /* Ending a pass restores ordinary writes at previously covered pixels. */
        int zero=0;uint8_t one=1;RendererPixel palette[2]={0,((RendererPixel)0x1234)};
        assert(dos_raster_sprite_texel_column(actual+8,1,320,&one,palette,&zero,0)==1);
        assert(actual[8]==((RendererPixel)0x1234));
    }
    puts("PASS: 32 reverse billboard batches match independent painter order, row/column cutouts, black, mask edges, complete coverage, reset and guards");
}

static void check_sprite_batch_repeated_rows(void)
{
    enum { HEIGHT=8, PIXELS=320*HEIGHT };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16];
    uint8_t wad[4]={0,0,31,0},ptr[128]={0},back_ptr[128]={0},texels[2]={0,1};
    RendererPixel front[2]={0,((RendererPixel)0x1357)},middle[32]={0},back[32]={0};
    int rows[HEIGHT],constant[HEIGHT]={0};
    static const unsigned clips[3][2]={{0,320},{1,318},{33,254}};
    DosSpriteBatch scratch;
    for(unsigned x=0;x<32;x++)back_ptr[x*4+3]=2;
    for(unsigned stencil=0;stencil<3;stencil++)
    for(unsigned clip=0;clip<3;clip++)for(unsigned pattern=0;pattern<6;pattern++) {
        unsigned left=clips[clip][0],width=clips[clip][1],written=0;
        for(unsigned x=0;x<32;x++) {
            int opaque=stencil==0 || (stencil==1?x>=8 && x<24:
                                      (x>=2 && x<6) || (x>=24 && x<30));
            ptr[x*4+3]=opaque?2:0;
        }
        middle[31]=(RendererPixel)(pattern&1?0:1);back[31]=((RendererPixel)0x2468);
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=((RendererPixel)0xa55a);
        dos_raster_sprite_batch_begin(actual+8,&scratch);
        for(unsigned x=0;x<320;x++) {
            for(unsigned y=0;y<HEIGHT;y++) {
                int opaque;
                switch(pattern) {
                    case 0:opaque=x&1;break; /* Maximum 160 fresh copy ranges. */
                    case 1:opaque=(x+y)&1;break;
                    case 2:opaque=(x&1)^(x==31 && y>=4);break;
                    case 3:opaque=x!=(31+y);break;
                    case 4:opaque=(x<32 || x>=288) && (y&1);break;
                    default:opaque=y&1;break;
                }
                rows[y]=opaque;
                unsigned column=x>=left?(x-left)/8:0;
                if(column>31)column=31;
                int middle_opaque=x>=left && x<left+width && ptr[column*4+3]!=0;
                expected[8+y*320+x]=opaque?front[1]:middle_opaque?middle[31]:back[31];
            }
            written+=dos_raster_sprite_texel_column(actual+8+x,HEIGHT,320,
                                                      texels,front,rows,1);
        }
        DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),middle,constant,
                              0,0,8192u,32,0};
        written+=dos_raster_sprite_rows(actual+8+left,width,HEIGHT,320,&sprite).drawn;
        sprite.palette=back;
        sprite.ptr=back_ptr;
        written+=dos_raster_sprite_rows(actual+8,320,HEIGHT,320,&sprite).drawn;
        for(unsigned y=0;y<HEIGHT;y++)assert(!scratch.open_words[y]);
        dos_raster_sprite_batch_end();
        assert(written==PIXELS && !memcmp(actual,expected,sizeof(actual)));
    }
    puts("PASS: 54 repeated billboard bands preserve 160 copy ranges, changing masks, transparent margins and gaps, boundary words, outside clips, closed-word counts, black and guards");
}

static void check_sprite_band_masks(void)
{
    enum { HEIGHT=8, PIXELS=320*HEIGHT };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16];
    static const unsigned positions[]={0,1,2,15,30,31,32,33,63,64,255,319};
    static const unsigned widths[]={1,33,65,320};
    uint8_t wad[642]={0},ptr[1280]={0},texels[320];
    RendererPixel palette[32];int rows[HEIGHT]={0};
    struct GuardedMask { uint32_t before[4];DosSpriteBatch mask;uint32_t after[4]; } actual_mask,expected_mask;
    unsigned checked=0;
    for(unsigned test=0;test<192;test++)for(unsigned direct=0;direct<2;direct++) {
        unsigned left=positions[test%12],width=widths[(test/12)%4];
        if(width>320-left)width=320-left;
        unsigned stencil=test/48,pattern=(test/3)%8,drawn=0;
        int hidden=1;
        if(!test){left=0;width=320;stencil=1;pattern=0;}
        for(unsigned i=0;i<32;i++)palette[i]=(RendererPixel)(i==31?0:i*1973u+test);
        for(unsigned column=0;column<320;column++) {
            int opaque=stencil==0 || (stencil==1?(column&1):
                          stencil==2?column%17<11:(column>=3 && column<29) || column>=63);
            unsigned offset=2+column*2,mode=column%3;
            texels[column]=(uint8_t)(opaque?1+(column+test)%31:0);
            ptr[column*4]=(uint8_t)mode;
            ptr[column*4+2]=(uint8_t)(offset>>8);ptr[column*4+3]=(uint8_t)offset;
            sprite_word_write(wad+offset,(uint16_t)(texels[column]<<(mode*5)));
        }
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=(RendererPixel)0xa55a;
        memset(&actual_mask,0xa5,sizeof(actual_mask));
        dos_raster_sprite_batch_begin(actual+8,&actual_mask.mask);
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=0;x<320;x++) {
            unsigned column=x>=left?x-left:320;
            int inside=x>=left && x<left+width;
            int covered=pattern==0?0:pattern==1?1:pattern==2?((x+y)&1):
                pattern==3?(inside && texels[column]):
                pattern==4?(x<160 && y>=4):pattern==5?(x!=31+y):
                pattern==6?((x&31)<16 || (y&1)):((y<4)?(x&1):((x+y)&1));
            if(covered) {
                actual_mask.mask.covered[y][x/32]|=1u<<(x%32);
                actual[8+y*320+x]=expected[8+y*320+x]=(RendererPixel)0x1357;
            } else if(inside)hidden=0;
        }
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=actual_mask.mask.covered[y][word]!=UINT32_MAX;
            actual_mask.mask.open_words[y]=(uint8_t)open;
        }
        expected_mask=actual_mask;
        /* Independent original-order pixel writes and ownership oracle. */
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned dx=0;dx<width;dx++) {
            unsigned x=left+dx;uint32_t bit=1u<<(x%32);
            if(texels[dx] && !(expected_mask.mask.covered[y][x/32]&bit)) {
                expected[8+y*320+x]=palette[texels[dx]];
                expected_mask.mask.covered[y][x/32]|=bit;drawn++;
            }
        }
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=expected_mask.mask.covered[y][word]!=UINT32_MAX;
            expected_mask.mask.open_words[y]=(uint8_t)open;
        }
        DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),palette,direct?NULL:rows,
                              0,0,65536u,320,0,0,0,0};
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+8+left,width,HEIGHT,320,&sprite);
        dos_raster_sprite_batch_end();
        assert(got.visible==(hidden?0:width*HEIGHT) && got.drawn==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
        assert(!memcmp(&actual_mask,&expected_mask,sizeof(actual_mask)));checked++;
    }
    printf("PASS: %u opaque band-mask cases preserve disjoint/same-word spans, 160 ranges, word joins, empty fresh rows, changing/outside coverage, direct/table sampling, black, complete masks/counts and guards\n",checked);
}

/* Independent pixels exercise every contiguous 32-bit gap, including runs
 * ending at bit 32, then unrelated multi-run masks. Both single-row copying
 * and repeated bands must preserve ownership outside their rectangle. */
static void check_sprite_fresh_runs(void)
{
    enum { HEIGHT=4, PIXELS=320*HEIGHT };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16];
    uint8_t wad[130]={0},ptr[257]={0},texels[64];
    RendererPixel palette[32];int rows[HEIGHT]={0};
    struct GuardedMask { uint32_t before[4];DosSpriteBatch mask;uint32_t after[4]; } got_mask,want_mask;
    uint32_t random=0x41792b35u;unsigned checked=0;
    for(unsigned test=0;test<1024;test++)for(unsigned height=1;height<=HEIGHT;height+=3) {
        unsigned left=test%7==0?0:test%7==1?31:test%7==2?32:
                      test%7==3?127:test%7==4?255:test%7==5?1:256;
        unsigned width=64,drawn=0;int hidden=1;
        for(unsigned i=0;i<32;i++)palette[i]=(RendererPixel)(i==31?0:i*1973u+test);
        for(unsigned x=0;x<64;x++) {
            unsigned offset=2+x*2,mode=x%3;
            texels[x]=(uint8_t)(test>=528 && (test&1) && x%5==0?0:1+(x+test)%31);
            /* Deliberately unaligned PTR entries still contain four bytes. */
            ptr[1+x*4]=(uint8_t)mode;
            ptr[4+x*4]=(uint8_t)offset;
            sprite_word_write(wad+offset,(uint16_t)(texels[x]<<(mode*5)));
        }
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=(RendererPixel)0xa55a;
        memset(&got_mask,0xa5,sizeof(got_mask));
        dos_raster_sprite_batch_begin(actual+8,&got_mask.mask);
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned word=0;word<10;word++) {
            uint32_t fresh;
            if(test<528) {
                unsigned n=test,start=0;
                while(n>=32-start){n-=32-start;start++;}
                unsigned end=start+n+1;
                /* Construct the oracle by testing each bit, independently
                 * of the production run extraction and its carry. */
                fresh=0;
                for(unsigned bit=0;bit<32;bit++)if(bit>=start && bit<end)fresh|=1u<<bit;
            } else {
                random=random*1664525u+1013904223u;fresh=random;
            }
            if(y>=2)fresh=~got_mask.mask.covered[y-2][word];
            got_mask.mask.covered[y][word]=~fresh;
        }
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=got_mask.mask.covered[y][word]!=UINT32_MAX;
            got_mask.mask.open_words[y]=(uint8_t)open;
        }
        want_mask=got_mask;
        for(unsigned y=0;y<height;y++)for(unsigned dx=0;dx<width;dx++) {
            unsigned x=left+dx;uint32_t bit=1u<<(x%32);
            if(!(want_mask.mask.covered[y][x/32]&bit)) {
                hidden=0;
                if(texels[dx]) {
                    expected[8+y*320+x]=palette[texels[dx]];
                    want_mask.mask.covered[y][x/32]|=bit;drawn++;
                }
            }
        }
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=want_mask.mask.covered[y][word]!=UINT32_MAX;
            want_mask.mask.open_words[y]=(uint8_t)open;
        }
        DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),palette,rows,
                              1,0,65536u,64,0};
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+8+left,width,height,320,&sprite);
        dos_raster_sprite_batch_end();
        assert(got.visible==(hidden?0:width*height) && got.drawn==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
        assert(!memcmp(&got_mask,&want_mask,sizeof(got_mask)));checked++;
    }
    printf("PASS: %u fresh-run pixel cases preserve every contiguous word gap, arbitrary multi-run coverage, unaligned PTRs, single/repeated rows, cutouts, black, ownership, counts and guards\n",checked);
}

/* A single opaque source texel produces every possible row-copy width.
 * Cover both source/destination dword alignments, black, ordinary painter
 * copies and the repeated-band range list, using independent pixel writes. */
static void check_sprite_copy_widths(void)
{
    enum { HEIGHT=3, PIXELS=320*HEIGHT };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16];
    const uint8_t wad[4]={0,0,31,0},ptr[4]={0,0,0,2};
    RendererPixel palette[32]={0};int rows[HEIGHT]={0};
    struct GuardedMask { uint32_t before[4];DosSpriteBatch mask;uint32_t after[4]; } got_mask,want_mask;
    unsigned checked=0;
    for(unsigned width=1;width<=320;width++)for(unsigned alignment=0;alignment<2;alignment++)
    for(unsigned batch=0;batch<2;batch++) {
        unsigned left=alignment?320-width:0;
        palette[31]=(RendererPixel)(width%4==0?0:width%4==1?0xffff:width%4==2?1:width*1973u);
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=(RendererPixel)0xa55a;
        memset(&got_mask,0xa5,sizeof(got_mask));
        if(batch)dos_raster_sprite_batch_begin(actual+8,&got_mask.mask);
        want_mask=got_mask;
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=left;x<left+width;x++) {
            expected[8+y*320+x]=palette[31];
            if(batch)want_mask.mask.covered[y][x/32]|=1u<<(x%32);
        }
        if(batch)for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=want_mask.mask.covered[y][word]!=UINT32_MAX;
            want_mask.mask.open_words[y]=(uint8_t)open;
        }
        DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),palette,rows,0,0,0,1,0};
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+8+left,width,HEIGHT,320,&sprite);
        dos_raster_sprite_batch_end();
        assert(got.visible==width*HEIGHT && got.drawn==width*HEIGHT);
        assert(!memcmp(actual,expected,sizeof(actual)));
        if(batch)assert(!memcmp(&got_mask,&want_mask,sizeof(got_mask)));
        checked++;
    }
    printf("PASS: %u sprite-copy width cases preserve widths 1..320, dword alignments, black, painter/repeated-band copies, masks/counts and guards\n",checked);
}

/* Repeated source rows can reach the full 240-row compositor height. Seed
 * changing coverage independently, then compare every pixel and mask word. */
static void check_sprite_tall_bands(void)
{
    enum { HEIGHT=240,PIXELS=320*HEIGHT };
    RendererPixel *actual=malloc((PIXELS+16)*sizeof(*actual));
    assert(actual);
    const unsigned widths[]={1,2,33,160,319,320},heights[]={2,3,31,240};
    const uint8_t wad[6]={0,0,31,0,0,0};uint8_t ptr[1280]={0};
    RendererPixel palette[32]={0};
    struct GuardedMask { uint32_t before[4];DosSpriteBatch mask;uint32_t after[4]; } got_mask,want_mask;
    unsigned checked=0;
    for(unsigned w=0;w<6;w++)for(unsigned h=0;h<4;h++)
    for(unsigned shape=0;shape<3;shape++)for(unsigned covered=0;covered<2;covered++) {
        unsigned width=widths[w],height=heights[h],left=w&1?320-width:0,drawn=0,any_open=0;
        palette[31]=(RendererPixel)(covered?0:0xf731);
        for(unsigned x=0;x<320;x++) {
            int opaque=shape==0 || (shape==1?x&1:x%64<7 || x%64>=49);
            ptr[x*4+3]=(uint8_t)(opaque?2:4);
        }
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=(RendererPixel)0xa55a;
        memset(&got_mask,0xa5,sizeof(got_mask));
        dos_raster_sprite_batch_begin(actual+8,&got_mask.mask);
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=0;x<320;x++)
            if(covered && ((x%11==0)^((y/7)&1)))got_mask.mask.covered[y][x/32]|=1u<<(x%32);
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=got_mask.mask.covered[y][word]!=UINT32_MAX;
            got_mask.mask.open_words[y]=(uint8_t)open;
        }
        want_mask=got_mask;
        for(unsigned y=0;y<height;y++)for(unsigned x=0;x<width;x++) {
            unsigned at=left+x;
            any_open|=!(want_mask.mask.covered[y][at/32]&(1u<<(at%32)));
            if(ptr[x*4+3]!=2 || (want_mask.mask.covered[y][at/32]&(1u<<(at%32))))continue;
            drawn++;
            want_mask.mask.covered[y][at/32]|=1u<<(at%32);
        }
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=want_mask.mask.covered[y][word]!=UINT32_MAX;
            want_mask.mask.open_words[y]=(uint8_t)open;
        }
        DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),palette,NULL,0,0,65536u,width,0,0,0,0};
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+8+left,width,height,320,&sprite);
        dos_raster_sprite_batch_end();
        assert(got.visible==(any_open?width*height:0) && got.drawn==drawn);
        for(unsigned i=0;i<8;i++)assert(actual[i]==(RendererPixel)0xa55a && actual[8+PIXELS+i]==(RendererPixel)0xa55a);
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=0;x<320;x++) {
            int claimed=covered && ((x%11==0)^((y/7)&1));
            int opaque=y<height && x>=left && x<left+width && ptr[(x-left)*4+3]==2;
            RendererPixel pixel=(RendererPixel)(opaque && !claimed?palette[31]:0xa55a);
            assert(actual[8+y*320+x]==pixel);
        }
        assert(!memcmp(&got_mask,&want_mask,sizeof(got_mask)));checked++;
    }
    free(actual);
    printf("PASS: %u tall sprite bands preserve 2/3/31/240 rows, odd/even copy widths, 160 spans, changing/unused coverage, masks/counts, black and guards\n",checked);
}

static void check_sprite_narrow_coverage(void)
{
    enum { HEIGHT=8, PIXELS=320*HEIGHT };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16];
    static const unsigned widths[]={1,2,31,32,33,63,64,65};
    uint8_t wad[4]={0,0,31,0},ptr[4]={0,0,0,2};
    RendererPixel palette[32]={0};int rows[HEIGHT]={0};
    DosSpriteBatch scratch,expected_mask;unsigned checked=0;
    for(unsigned bit=0;bit<32;bit++)for(unsigned w=0;w<8;w++)
    for(unsigned pattern=0;pattern<4;pattern++) {
        unsigned left=224+bit,width=widths[w],drawn=0;
        palette[31]=(RendererPixel)(w&1?0:((RendererPixel)0x2468));
        for(unsigned i=0;i<PIXELS+16;i++)actual[i]=expected[i]=((RendererPixel)0xa55a);
        dos_raster_sprite_batch_begin(actual+8,&scratch);
        /* Independent pixel coverage: fully closed rows alternate with
         * outside holes, while the rectangle is covered except for selected
         * first/last pixels in its first, last or interior rows. */
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=0;x<320;x++) {
            int inside=x>=left && x<left+width;
            int covered=inside || !(y&1) || (x&1);
            if((pattern==1 && y==0 && x==left) ||
               (pattern==2 && y==HEIGHT-1 && x==left+width-1) ||
               (pattern==3 && ((y==1 && x==left) ||
                               (y==HEIGHT-2 && x==left+width-1))))covered=0;
            if(covered)scratch.covered[y][x/32]|=1u<<(x%32);
            actual[8+y*320+x]=expected[8+y*320+x]=covered?((RendererPixel)0x1357):((RendererPixel)0xa55a);
        }
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=scratch.covered[y][word]!=UINT32_MAX;
            scratch.open_words[y]=(uint8_t)open;
        }
        expected_mask=scratch;
        for(unsigned y=0;y<HEIGHT;y++)for(unsigned x=left;x<left+width;x++) {
            uint32_t mask=1u<<(x%32);
            if(!(scratch.covered[y][x/32]&mask)) {
                expected[8+y*320+x]=palette[31];drawn++;
                expected_mask.covered[y][x/32]|=mask;
            }
        }
        for(unsigned y=0;y<HEIGHT;y++) {
            unsigned open=0;
            for(unsigned word=0;word<10;word++)open+=expected_mask.covered[y][word]!=UINT32_MAX;
            expected_mask.open_words[y]=(uint8_t)open;
        }
        DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),palette,rows,
                              0,0,65536u,1,0};
        DosSpriteCoverage got=dos_raster_sprite_rows(actual+8+left,width,HEIGHT,320,&sprite);
        dos_raster_sprite_batch_end();
        assert(got.visible==(pattern?width*HEIGHT:0) && got.drawn==drawn);
        assert(!memcmp(actual,expected,sizeof(actual)));
        assert(!memcmp(&scratch,&expected_mask,sizeof(scratch)));checked++;
    }
    printf("PASS: %u narrow billboard coverage cases preserve every bit alignment, one/two/three-word clips, first/last/interior gaps, closed rows, black, counts and guards\n",checked);
}

static void check_sprite_history(void)
{
    enum { PIXELS=320*240, STEPS=12 };
    static RendererPixel actual[PIXELS+16],expected[PIXELS+16],palette[2][32];
    static uint8_t wad[2][16*32*2+2],ptr[2][16*4];
    int rows[240];
    DosSpriteBatch scratch,expected_mask;
    DosSpriteHistory history;
    DosSpriteCoverage reference[STEPS];
    for(unsigned source=0;source<2;source++)for(unsigned x=0;x<16;x++) {
        unsigned offset=2+x*64,ptr_offset=2+((x+source)%16)*64;
        ptr[source][x*4]=(uint8_t)(x%3);
        ptr[source][x*4+2]=(uint8_t)(ptr_offset>>8);
        ptr[source][x*4+3]=(uint8_t)ptr_offset;
        for(unsigned y=0;y<32;y++) {
            unsigned texel=(x+y+source)%3?0:1+(x+y)%31;
            sprite_word_write(wad[source]+offset+y*2,(uint16_t)(texel|(texel<<5)|(texel<<10)));
        }
    }
    /* Compare the optional cache with the same uncached mask kernel, including
     * every sampling input, palette-only changes, reused row storage and reset.
     * Transparent holes keep repeated rectangles out of the whole-box reject. */
    for(unsigned variant=0;variant<25;variant++)for(unsigned cached=0;cached<2;cached++) {
        RendererPixel *surface=cached?actual:expected;
        for(unsigned i=0;i<PIXELS+16;i++)surface[i]=((RendererPixel)0xa55a);
        dos_raster_sprite_batch_begin(surface+8,&scratch);
        if(cached)dos_raster_sprite_batch_history_begin(&history);
        for(unsigned step=0;step<STEPS;step++) {
            for(unsigned y=0;y<240;y++)rows[y]=(int)(y/4)%24;
            for(unsigned i=0;i<32;i++) {
                palette[0][i]=(RendererPixel)(i==1?0:i*1973u);
                palette[1][i]=(RendererPixel)(i*917u);
            }
            unsigned width=310,height=239,pitch=320,offset=8;
            if(variant>=23)height=variant==23?1:240;
            unsigned changed_row=(variant==18 || variant==22 || variant==23)?0:
                                  (variant==19 || variant==21)?height/2:height-1;
            if(variant>=18) {
                /* Begin with a rejected row, then expose its valid low-byte
                 * sample without changing the reused row-array address.
                 * Differences in any byte must invalidate the footprint. */
                uint32_t high=variant==21?65536u:variant==22?0x80000000u:256u;
                rows[changed_row]=(int)((uint32_t)rows[changed_row]|high);
            }
            DosSpriteRows sprite={wad[0],ptr[0],sizeof(wad[0]),sizeof(ptr[0]),
                                  palette[0],rows,0,0,65536u/20,16,0};
            if(step%4==1 || step%4==2)switch(variant) {
                case 0:sprite.palette=palette[1];break;
                case 1:for(unsigned y=0;y<240;y++)rows[y]++;break;
                case 2:sprite.wad=wad[1];break;
                case 3:sprite.ptr=ptr[1];break;
                case 4:sprite.wad_bytes-=128;break;
                case 5:sprite.ptr_bytes=16;break;
                case 6:sprite.ptr_offset=4;break;
                case 7:sprite.u=65536;break;
                case 8:sprite.du=65536u/10;break;
                case 9:sprite.columns=8;break;
                case 10:sprite.down_strip=2;break;
                case 11:width--;break;
                case 12:height++;break;
                case 13:offset++;break;
                case 14:offset+=320;break;
                case 15:width=32;height=1;break;
                case 16:for(unsigned i=1;i<32;i++)palette[0][i]=0;break;
                case 17:for(unsigned y=0;y<240;y++)rows[y]=(y&1)?-1:INT_MAX;break;
                case 18:case 19:case 20:case 21:case 22:case 23:case 24:
                    rows[changed_row]=(int)((changed_row/4)%24);break;
            }
            /* More than four distinct footprints exercise eviction, then
             * revisiting the original verifies that missing entries are safe. */
            if(step>=4 && step<10)offset+=step;
            DosSpriteCoverage got=dos_raster_sprite_rows(surface+offset,width,height,pitch,&sprite);
            if(!cached)reference[step]=got;
            else assert(got.visible==reference[step].visible && got.drawn==reference[step].drawn);
        }
        dos_raster_sprite_batch_end();
        if(!cached)expected_mask=scratch;
        else {
            assert(!memcmp(actual,expected,sizeof(actual)));
            assert(!memcmp(&scratch,&expected_mask,sizeof(scratch)));
        }
    }
    puts("PASS: 300 cached billboard draws retain uncached pixels, coverage, sampling keys, palette changes, row storage reuse, high-byte changes, first/last rows, one/full-height clips, eviction, reset and guards");
}

static void check_sprite_dda_rows(void)
{
    enum { PIXELS=320*240, DRAWS=4 };
    static RendererPixel pixels[2][PIXELS+16];
    static const unsigned heights[]={1,2,3,7,31,63,127,240};
    static const unsigned widths[]={1,2,31,32,33,64,127,320};
    static const uint32_t steps[]={0,1,255,4095,4096,8191,32768,65535,65536,99991,7*65536};
    static const uint32_t edge_steps[]={1,2,255,256,4095,4096,4097,8191,8192,8193,
        16383,16384,16385,21845,21846,32767,32768,32769,65534,65535};
    static const uint32_t edge_fractions[]={0,1,2,32767,32768,65534,65535};
    uint8_t wad[514],ptr[40];
    RendererPixel palette[32];int rows[240];
    DosSpriteBatch mask[2];DosSpriteHistory history;
    DosSpriteCoverage result[2][DRAWS];
    unsigned checked=0;
    for(unsigned i=0;i<sizeof(wad);i++)wad[i]=(uint8_t)next_value();
    for(unsigned c=0;c<10;c++) {
        unsigned offset=c==3?0:2+c*48;
        ptr[c*4]=(uint8_t)(c==3?0:c%3);
        ptr[c*4+1]=0;ptr[c*4+2]=(uint8_t)(offset>>8);ptr[c*4+3]=(uint8_t)offset;
    }
    for(unsigned test=0;test<256;test++)for(unsigned mode=0;mode<3;mode++) {
        unsigned x=test%33,y=test%4,width=widths[test%8],height=heights[(test/8)%8];
        if(width>320-x)width=320-x;
        if(height>240-y)height=240-y;
        uint32_t start=((test%25)<<16)+(test*1493u&65535u),step=steps[test%11];
        if(test>=128) {
            step=edge_steps[(test-128)%20];
            start=((test%25)<<16)+edge_fractions[(test-128)%7];
        }
        unsigned last=test%65;
        if(test%16==0)start=0x7ffe0000u,step=1; /* Clamping near the signed limit. */
        assert((uint64_t)start+(height-1)*(uint64_t)step+3<=INT_MAX);
        for(unsigned side=0;side<2;side++) {
            for(unsigned i=0;i<PIXELS+16;i++)pixels[side][i]=(RendererPixel)0xa55a;
            memset(&mask[side],0,sizeof(mask[side]));
            if(mode) {
                dos_raster_sprite_batch_begin(pixels[side]+8,&mask[side]);
                if(mode==2)dos_raster_sprite_batch_history_begin(&history);
                /* Partial existing coverage, including complete words and holes. */
                for(unsigned r=y;r<y+height;r++)for(unsigned w=0;w<10;w++) {
                    uint32_t bits=(r+w+test)%3==0?UINT32_MAX:(r&1?0x55555555u:0x80000001u);
                    mask[side].covered[r][w]=bits;
                    if(bits==UINT32_MAX)mask[side].open_words[r]--;
                }
            }
            for(unsigned draw=0;draw<DRAWS;draw++) {
                uint32_t v=start+(draw>=2?3u:0u);
                unsigned clamp=last+(draw==3);
                /* Independent per-pixel fixed-point reference, without band grouping. */
                for(unsigned r=0;r<height;r++) {
                    unsigned sample=(unsigned)(((uint64_t)v+(uint64_t)r*step)/65536u);
                    rows[r]=(int)(sample>clamp?clamp:sample);
                }
                for(unsigned c=0;c<32;c++)palette[c]=(RendererPixel)(c==1?0:c==2?255:(c+draw*17u)*1973u);
                DosSpriteRows sprite={wad,ptr,sizeof(wad),sizeof(ptr),palette,rows,0,
                    (test%7)*65536u+173u,(test%3?58981u:4096u),10,test%3,v,step,clamp};
                if(side && draw!=1)sprite.rows=NULL; /* Mix table/DDA history entries. */
                result[side][draw]=dos_raster_sprite_rows(pixels[side]+8+y*320+x,width,height,320,&sprite);
                if(side) {
                    assert(result[side][draw].visible==result[0][draw].visible);
                    assert(result[side][draw].drawn==result[0][draw].drawn);
                }
            }
            if(mode)dos_raster_sprite_batch_end();
        }
        assert(!memcmp(pixels[0],pixels[1],sizeof(pixels[0])));
        assert(!memcmp(&mask[0],&mask[1],sizeof(mask[0])));
        checked+=DRAWS;
    }
    printf("PASS: %u direct sprite row walks match independent 64-bit row tables, clamping, fractional steps, zero/large steps, clips, packed modes, black/255, mixed history, coverage and guards\n",checked);
}

#endif
