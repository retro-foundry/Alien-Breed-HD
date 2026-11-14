/* Shared native wall-row program and traversal, instantiated by raster_dos.c. */
__attribute__((noinline,optimize("O2")))
static void WALL_ROW_FUNCTION(RendererPixel *out,unsigned width,unsigned height,unsigned pitch,
                                 const DosWallColumn *columns,const RendererPixel *palette,
                                 uint32_t y,uint32_t dy,unsigned mask)
{
#ifndef WALL_PAIR_RENDER
    static const uint8_t pixel[15]={
        0x0f,0xb6,0x83,0,0,0,0, /* movzbl source(%ebx),%eax */
        0x66,0x8b,0x04,0x46,   /* movw (%esi,%eax,2),%ax */
        0x66,0x89,0x47,0       /* movw %ax,offset(%edi) */
    };
#endif
    _Static_assert(sizeof(uintptr_t)==4,"DOS row operands are 32-bit");
    for(unsigned first=0;first<width;first+=WALL_CODE_COLUMNS) {
        unsigned count=width-first;
        if(count>WALL_CODE_COLUMNS)count=WALL_CODE_COLUMNS;
#ifdef WALL_PAIR_RENDER
        /* DL/DH retain the two original five-bit indices. One native DWORD
         * lookup/store replaces two palette word lookups/stores. 486 permits
         * either word alignment of the destination, including odd row pitch. */
        static const uint8_t prefix[15]={
            0x8a,0x93,0,0,0,0, /* movb source0(%ebx),%dl */
            0x8a,0xb3,0,0,0,0, /* movb source1(%ebx),%dh */
            0x8b,0x04,0x96     /* movl (%esi,%edx,4),%eax */
        };
#else
        static const uint8_t prefix[22]={
            0x0f,0xb6,0x83,0,0,0,0, 0x0f,0xb6,0x93,0,0,0,0,
            0x66,0x8b,0x04,0x46, 0x66,0x8b,0x14,0x56
        };
#endif
        unsigned bytes=0,x=0;
        for(;x+1<count;x+=2) {
            uint8_t *instruction=wall_row_code+bytes;
            uintptr_t source0=(uintptr_t)columns[first+x].texels;
            uintptr_t source1=(uintptr_t)columns[first+x+1].texels;
            uint32_t offset=x*sizeof(*out);
            memcpy(instruction,prefix,sizeof(prefix));
#ifdef WALL_PAIR_RENDER
            memcpy(instruction+2,&source0,4);memcpy(instruction+8,&source1,4);
            instruction[15]=0x89;
            if(offset<=127){instruction[16]=0x47;instruction[17]=(uint8_t)offset;bytes+=18;}
            else{instruction[16]=0x87;memcpy(instruction+17,&offset,4);bytes+=21;}
#else
            memcpy(instruction+3,&source0,4);memcpy(instruction+10,&source1,4);
            instruction[22]=0x66;instruction[23]=0x89;
            if(offset<=127) {
                instruction[24]=0x47;instruction[25]=(uint8_t)offset;
                instruction[26]=0x66;instruction[27]=0x89;instruction[28]=0x57;instruction[29]=(uint8_t)(offset+2);
                bytes+=30;
            } else {
                instruction[24]=0x87;memcpy(instruction+25,&offset,4);offset+=2;
                instruction[29]=0x66;instruction[30]=0x89;instruction[31]=0x97;memcpy(instruction+32,&offset,4);
                bytes+=36;
            }
#endif
        }
        if(x<count) {
            uint8_t *instruction=wall_row_code+bytes;
            uintptr_t source=(uintptr_t)columns[first+x].texels;
            uint32_t offset=x*sizeof(*out);
#ifdef WALL_PAIR_RENDER
            /* Zero-extend the final index so the low word of pair[index]
             * is exactly palette[index], regardless of the preceding pair. */
            static const uint8_t tail[11]={0x0f,0xb6,0x93,0,0,0,0,0x66,0x8b,0x04,0x96};
            memcpy(instruction,tail,11);memcpy(instruction+3,&source,4);
#else
            memcpy(instruction,pixel,11);memcpy(instruction+3,&source,4);
#endif
            instruction[11]=0x66;instruction[12]=0x89;
            if(offset<=127){instruction[13]=0x47;instruction[14]=(uint8_t)offset;bytes+=15;}
            else{instruction[13]=0x87;memcpy(instruction+14,&offset,4);bytes+=18;}
        }
        wall_row_code[bytes]=0xc3;
        RendererPixel *dest=out+first;
        uint32_t coordinate=y;unsigned previous=~0u;
        for(unsigned row=0;row<height;row++,coordinate+=dy,dest+=pitch) {
            unsigned sample=(coordinate>>16)&mask;
            if(sample==previous)memcpy(dest,dest-pitch,count*sizeof(*dest));
            else __asm__ volatile(
#ifdef WALL_PAIR_RENDER
                "xorl %%edx,%%edx\n\t"
#endif
                "call %c[code]"
#ifdef WALL_PAIR_RENDER
                : : [code] "i"(wall_row_code),"D"(dest),"b"(sample),"S"(wall_pair_lookup->pixels)
#else
                : : [code] "i"(wall_row_code),"D"(dest),"b"(sample),"S"(palette)
#endif
                : "eax","edx","cc","memory");
            previous=sample;
        }
    }
}
