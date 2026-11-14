/* Independent byte-pixel oracles for AB3DI floor/water and WallRoutine3. */
static RendererPixel reference_water_pixel(RendererPixel p0,RendererPixel p1,
                                           const uint8_t *brighten,unsigned row,unsigned alpha,int single)
{
    unsigned s0=renderer_pixel_gb_index(p0),s1=renderer_pixel_gb_index(p1);
    const uint8_t *b0=brighten+(row*256+s0)*2,*b1=brighten+(row*256+s1)*2;
    unsigned c0=(b0[0]<<8)|b0[1],c1=(b1[0]<<8)|b1[1],result=0;
    c0=renderer_pixel_from_c12((uint16_t)c0);
    if(single)return (RendererPixel)c0;
    c0=renderer_pixel_to_c12((uint16_t)c0);
    c1=renderer_pixel_to_c12(renderer_pixel_from_c12((uint16_t)c1));
    for(unsigned shift=0;shift<=8;shift+=4) {
        unsigned a=((c0>>shift)&15)*17,b=((c1>>shift)&15)*17;
        unsigned byte=(a*(255-alpha)+b*alpha)/255;
        result|=((byte+8)/17)<<shift;
    }
    return renderer_pixel_from_c12((uint16_t)result);
}

static void check_water(void)
{
    enum { PITCH=324, ROWS=5 };
    static uint8_t brighten[22*512],phase[4096];
    static RendererPixel actual[PITCH*ROWS],expected[PITCH*ROWS];
    for(unsigned row=0;row<22;row++)for(unsigned sample=0;sample<256;sample++) {
        unsigned color=row?((row*197+sample*37)&4095):
            ((((sample*3)&15)<<8)|(((sample*7)&15)<<4)|(sample&15));
        brighten[(row*256+sample)*2]=(uint8_t)(color>>8);
        brighten[(row*256+sample)*2+1]=(uint8_t)color;
    }
    dos_raster_prepare_water(brighten);
    for(unsigned row=0;row<22;row++)for(unsigned sample=0;sample<256;sample++) {
        unsigned at=(row*256+sample)*2;
        assert(dos_raster_water_color(row,sample)==renderer_pixel_from_c12((uint16_t)(((unsigned)brighten[at]<<8)|brighten[at+1])));
    }
    for(unsigned pair=0;pair<256;pair++) {
        actual[pair]=(RendererPixel)pair;
        actual[PITCH+pair]=(RendererPixel)(pair*131+17);
    }
    memset(phase,0,sizeof(phase));
    for(unsigned alpha=0;alpha<256;alpha++) {
        dos_raster_water(actual+PITCH*2,256,actual,actual+PITCH,phase,0,0,alpha,0,0,0,0);
        for(unsigned pair=0;pair<256;pair++)
            assert(actual[PITCH*2+pair]==reference_water_pixel(actual[pair],actual[PITCH+pair],brighten,0,alpha,0));
    }
    /* Every byte index must retain its representative palette color
     * before the original low-byte Amiga green/blue selection. */
    for(unsigned block=0;block<1;block++) {
        for(unsigned x=0;x<256;x++)actual[x]=(RendererPixel)((block<<8)|x);
        dos_raster_water(actual+PITCH*2,256,actual,NULL,phase,1,0,231,0,0,0,0);
        for(unsigned x=0;x<256;x++)
            assert(actual[PITCH*2+x]==reference_water_pixel(actual[x],actual[x],brighten,1,0,1));
    }
    for(unsigned i=0;i<sizeof(phase);i++)phase[i]=(uint8_t)(next_value()%10);
    static const unsigned widths[]={0,1,2,3,7,17,159,319,320};
    for(unsigned test=0;test<768;test++) {
        for(unsigned i=0;i<PITCH*ROWS;i++)actual[i]=expected[i]=(RendererPixel)next_value();
        unsigned count=widths[test%9],shade=next_value()%13;
        unsigned offset=test%9==0?65535u:test%9==1?65536u:test%64,alpha=test%256;
        int single=test>=384;
        unsigned out=PITCH*2+2,src0=(test%5)*PITCH+2,src1=((test+1)%5)*PITCH+2;
        uint32_t u=next_value(),v=next_value(),du=next_value(),dv=next_value(),ru=u,rv=v;
        dos_raster_water(actual+out,count,actual+src0,single?NULL:actual+src1,phase,shade,offset,alpha,u,v,du,dv);
        for(unsigned x=0;x<count;x++,ru+=du,rv+=dv) {
            uint8_t ub=(uint8_t)(ru>>16),vb=(uint8_t)(rv>>16);
            uint16_t uv=(uint16_t)(((uint16_t)vb<<8)|ub);
            uv=(uint16_t)(uv+offset);uv&=0x3f3f;
            unsigned level=phase[((uv>>8)<<6)|(uv&63)];
            expected[out+x]=reference_water_pixel(expected[src0+x],single?expected[src0+x]:expected[src1+x],brighten,shade+level,single?0:alpha,single);
        }
        assert(!memcmp(actual,expected,sizeof(actual)));
    }
    /* AB3DI texturedwater forms a 16-bit UV word before adding wateroff.
     * Target both U-byte carries and V/word wraps with independent sampling,
     * including the boundaries of the optional horizontal span walk. */
    static const uint32_t steps[]={0,1,65535,65536,65537,131071,131072,131073,
                                  262143,262144,262145,524288,1048576,0xffffff,0x1000000,UINT32_MAX};
    static const unsigned offsets[]={0,1,63,64,127,128,192,255,256,65535,65536,0x12345};
    static const unsigned vrows[]={0,1,62,63,64,255,256,65535};
    static const unsigned lengths[]={0,1,2,3,7,31,32,63,64,65,159,319,320,321,322};
    uint32_t seed=0xa93c715du;
    for(unsigned test=0;test<2048;test++) {
        for(unsigned i=0;i<PITCH*ROWS;i++) {
            seed=seed*1664525u+1013904223u;
            actual[i]=expected[i]=(RendererPixel)(seed>>8);
        }
        unsigned offset=offsets[test%12],boundary=(256-(offset&255))<<16;
        uint32_t starts[]={boundary-1,boundary,boundary+1,0,0xfffffe,0xffffff,0x1000000,UINT32_MAX};
        uint32_t u=(seed&0xff000000u)|(starts[(test/12)%8]&0xffffffu);
        uint32_t v=(vrows[(test/192)%8]<<16)|0x1234u;
        uint32_t du=steps[(test/3)%16],dv=test%8==0?1:test%8==1?65536u:0;
        unsigned count=lengths[(test/9)%15],shade=(test/17)%13,alpha=(test*29)&255;
        unsigned out=PITCH*2+2,src0=(test%5)*PITCH+2,src1=((test+1)%5)*PITCH+2;
        if(test%7==0)src0=out-1;
        if(test%7==1)src1=out+1;
        int single=(test/13)%3==0;
        dos_raster_water(actual+out,count,actual+src0,single?NULL:actual+src1,
                         phase,shade,offset,alpha,u,v,du,dv);
        for(unsigned x=0;x<count;x++,u+=du,v+=dv) {
            uint16_t uv=(uint16_t)((((v>>16)&255)<<8)|((u>>16)&255));
            uv=(uint16_t)(uv+offset);
            unsigned level=phase[(((uv>>8)&63)*64)+(uv&63)];
            expected[out+x]=reference_water_pixel(expected[src0+x],single?expected[src0+x]:expected[src1+x],
                                                  brighten,shade+level,single?0:alpha,single);
        }
        assert(!memcmp(actual,expected,sizeof(actual)));
    }
    puts("PASS: 2048 water carry spans preserve U-byte/V-row/word wraps, step limits, zero steps, clipping, source overlap, alpha symmetry, single-source rounding and guards");
    printf("Water: 5632 shared colours, 256 indexed inputs, 65536 alpha/colour pairs and 768 single/blended wrapping/alias/clip spans passed\n");
}

static void check_floors(void)
{
    uint8_t *texture = malloc(65536);
    assert(texture);
    for (unsigned i = 0; i < 65536; i++) texture[i] = next_value() >> 24;
    RendererPixel palettes[15][256], actual[322], expected[322];
    for (unsigned l = 0; l < 15; l++) {
        for (unsigned t = 0; t < 256; t++) palettes[l][t] = renderer_pixel_from_c12(l * 256 + t);
    }
    for (unsigned c = 0; c < 4096; c++) {
        unsigned count = c % 321;
        uint32_t u = next_value(), v = next_value(), du = next_value(), dv = next_value();
        if (c % 3 == 0) du = 0;
        if (c % 3 == 1) dv = 0;
        /* Dense sub-texel steps exercise exact positive carries and negative
         * borrows, zero steps, both axes and all unrolled tails. */
        static const uint32_t small_steps[]={0,1,8192,16384,32767,32768,32769,65535,
            65536,0u-1,0u-8192,0u-16384,0u-32767,0u-32768,0u-32769,0u-65535,
            123,4095,4096,4097,8191,8193,0u-123,0u-4095,
            0u-4096,0u-4097,0u-8191,0u-8193,17,0u-17,7777,0u-7777};
        if(c%6==0)dv=small_steps[(c/6)%32];
        if(c%6==1)du=small_steps[(c/6)%32];
        int32_t left = c % 15, right = (c / 15) % 15;
        int32_t level = left * 65536;
        int32_t dlevel = count > 1 ? (right - left) * 65536 / (int32_t)(count - 1) : 0;
        memset(actual, 0x91, sizeof(actual)); memcpy(expected, actual, sizeof(actual));
        uint32_t a = u, b = v;
        int32_t shade = level;
        for (unsigned x = 0; x < count; x++) {
            /* AB3DI d5.w = (V&63)*256 + (U&63), then sample d5.w*4. */
            unsigned offset = (((b >> 16) & 63) * 256 + ((a >> 16) & 63)) * 4;
            expected[x + 1] = palettes[shade >> 16][texture[offset]];
            a += du; b += dv; shade += dlevel;
        }
        dos_raster_floor_gouraud(actual + 1, count, texture, palettes, u, v, du, dv, level, dlevel);
        assert(!memcmp(actual, expected, sizeof(actual)));
        for (unsigned x = 0; x < count; x++) {
            unsigned offset = (((v >> 16) & 63) * 256 + ((u >> 16) & 63)) * 4;
            expected[x + 1] = palettes[left][texture[offset]];
            u += du; v += dv;
        }
        /* Reconstruct the original accumulators after the reference walk. */
        u -= du * count; v -= dv * count;
        dos_raster_floor(actual + 1, count, texture, palettes[left], u, v, du, dv);
        assert(!memcmp(actual, expected, sizeof(actual)));
    }
    free(texture);
    puts("PASS: 8192 floor spans retain UV wrapping, shade gradients and bounds");
}


static void check_indexed_floor_caches(void)
{
    static uint8_t texture[65536],other[65536],raw[3][15*512];
    static RendererPixel expected_pal[3][15][256];
    const RendererPixel (*prepared[3])[256];
    for(unsigned i=0;i<65536;i++) {
        texture[i]=(uint8_t)((i%256%7==0 || i/256%8==0)?0:i&3);
        other[i]=(uint8_t)(next_value()>>24);
    }
    for(unsigned bank=0;bank<3;bank++)for(unsigned shade=0;shade<15;shade++)for(unsigned c=0;c<256;c++) {
        unsigned color=(bank==1 && shade==14)?0x761:(c*37+shade*173+bank*977)&4095;
        unsigned at=(shade*256+c)*2;
        raw[bank][at]=(uint8_t)(color>>8);raw[bank][at+1]=(uint8_t)color;
    }
    unsigned checked=0;
    for(unsigned epoch=0;epoch<2;epoch++) {
        if(epoch)for(unsigned c=0;c<4096;c++)dos_c12_to_index[c]^=0x57;
        for(unsigned bank=0;bank<3;bank++) {
            if(!epoch)dos_raster_prepare_floor(bank,bank==1?other:texture,raw[bank]);
            for(unsigned shade=0;shade<15;shade++)for(unsigned c=0;c<256;c++) {
                unsigned at=(shade*256+c)*2;
                expected_pal[bank][shade][c]=renderer_pixel_from_c12((raw[bank][at]<<8)|raw[bank][at+1]);
            }
        }
        if(epoch)dos_raster_indexed_floor_palette_changed();
        for(unsigned bank=0;bank<3;bank++) {
            prepared[bank]=dos_raster_indexed_floor_palettes(raw[bank]);
            assert(prepared[bank] && !memcmp(prepared[bank],expected_pal[bank],sizeof(expected_pal[bank])));
        }
        static const uint32_t steps[]={0,1,16384,32768,65535,65536,98304,0u-1,0u-32768,0u-98304};
        for(unsigned n=0;n<2304;n++) {
            unsigned bank=n%3,channel=(n/3)%16,count=(n*19)%321,shade=n%7?14:13;
            uint32_t u=next_value(),v=next_value(),du=steps[(n/48)%10],dv=0;
            if(n%3==1){dv=du;du=0;}if(n%3==2)dv=0u-du;
            if(n%4==0){u=0;v=0;}
            const uint8_t *tex=(bank==1?other:texture)+(channel/4)*256+channel%4;
            RendererPixel actual[322],expected[322];
            memset(actual,0x91,sizeof(actual));memcpy(expected,actual,sizeof(actual));
            uint32_t ru=u,rv=v;
            for(unsigned x=0;x<count;x++,ru+=du,rv+=dv) {
                unsigned at=(((rv>>16)&63)*256+((ru>>16)&63))*4;
                expected[x+1]=expected_pal[bank][shade][tex[at]];
            }
            dos_raster_floor_shaded(actual+1,count,tex,prepared[bank],shade,u,v,du,dv);
            assert(!memcmp(actual,expected,sizeof(actual)));checked++;
        }
    }
    for(unsigned bank=0;bank<3;bank++) {
        dos_raster_prepare_floor(bank,NULL,NULL);
        assert(!dos_raster_indexed_floor_palettes(raw[bank]));
        memset(raw[bank],0,sizeof(raw[bank]));texture[0]^=1;
        dos_raster_prepare_floor(bank,texture,raw[bank]);
        assert(dos_raster_indexed_floor_palettes(raw[bank])[14][0]==dos_c12_to_index[0]);
        dos_raster_prepare_floor(bank,NULL,NULL);
    }
    assert(!dos_raster_indexed_floor_palettes(other));
    printf("PASS: %u indexed classified floor spans, all shade-cache entries, palette changes and same-address reloads\n",checked);
}

static void check_indexed_walls(void)
{
    static uint8_t texels[33][512];
    RendererPixel palette[4][32];DosWallColumn columns[33];
    static RendererPixel actual[320*240+8],expected[320*240+8];
    for(unsigned col=0;col<33;col++)for(unsigned row=0;row<512;row++)texels[col][row]=(col*13+row*7)&31;
    for(unsigned p=0;p<4;p++)for(unsigned c=0;c<32;c++)palette[p][c]=(RendererPixel)(p*71+c*29);
    for(unsigned test=0;test<192;test++) {
        unsigned width=test%34,height=test%151,pitch=40,mask=(1u<<(test%6+4))-1;
        if(test%7==0)mask&=0x55;
        uint32_t y=next_value(),dy=test%3?next_value():8192;
        const RendererPixel *shared=test%2?palette[test%4]:NULL;
        for(unsigned x=0;x<33;x++) {columns[x].texels=texels[x];columns[x].palette=shared?NULL:palette[x%4];}
        memset(actual,0x91,sizeof(actual));memcpy(expected,actual,sizeof(actual));
        for(unsigned row=0;row<height;row++)for(unsigned x=0;x<width;x++)
            expected[3+row*pitch+x]=(shared?shared:palette[x%4])[texels[x][((y+row*dy)>>16)&mask]];
        dos_raster_wall_rows(actual+3,width,height,pitch,columns,shared,y,dy,mask);
        assert(!memcmp(actual,expected,sizeof(actual)));
        for(unsigned copies=1;copies<=3;copies++) {
            memset(actual,0x91,sizeof(actual));memcpy(expected,actual,sizeof(actual));
            unsigned count=test*23%240,stride=test%2?320:40;
            for(unsigned row=0;row<count;row++)for(unsigned x=0;x<copies;x++)
                expected[3+row*stride+x]=palette[0][texels[0][((y+row*dy)>>16)&mask]];
            if(copies==1)dos_raster_wall(actual+3,count,stride,texels[0],palette[0],y,dy,mask);
            else dos_raster_wall_edges(actual+3,count,stride,texels[0],palette[0],y,dy,mask,copies);
            assert(!memcmp(actual,expected,sizeof(actual)));
        }
    }
    puts("PASS: 768 indexed wall row, shared/per-column palette, column and edge rasters preserve samples and guards");
}
