#include "raster_dos.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static void fill_words(int16_t *out, unsigned count, uint16_t value)
{
    unsigned pairs = count / 2;
    uint32_t repeated = value | ((uint32_t)value << 16);
    __asm__ volatile("rep stosl" : "+D"(out), "+c"(pairs) : "a"(repeated) : "memory", "cc");
    if (count & 1) *out = (int16_t)value;
}

#ifdef AB3D_NATIVE_RGB565
/* Two 1024-bit maps classify every interleaved horizontal/vertical tile row
 * at the authored darkest shade. No pre-shaded texture bank is retained. */
typedef struct {
    const uint8_t *texture;
    const RendererPixel *palette;
    int whole_solid;
    uint8_t solid[2][128];
} DosFloorSolid;
static DosFloorSolid floor_solid[3];

void dos_raster_prepare_floor(unsigned slot,const uint8_t *texture,const uint8_t *palettes)
{
    if(slot>=3)return;
    DosFloorSolid *e=&floor_solid[slot];
    memset(e,0,sizeof(*e));
    if(!texture || !palettes)return;
    e->texture=texture;e->palette=(const RendererPixel *)(const void *)palettes+14*256;
    e->whole_solid=1;
    for(unsigned i=1;i<256;i++)if(e->palette[i]!=e->palette[0]){e->whole_solid=0;break;}
    if(e->whole_solid){memset(e->solid,255,sizeof(e->solid));return;}
    for(unsigned axis=0;axis<2;axis++)for(unsigned key=0;key<1024;key++) {
        unsigned start=axis?(key>>2)+(key&3)*256:(key>>2)*256+(key&3);
        unsigned stride=axis?1024:4,i;
        RendererPixel color=e->palette[texture[start]];
        for(i=1;i<64;i++)if(e->palette[texture[start+i*stride]]!=color)break;
        if(i==64)e->solid[axis][key>>3]|=(uint8_t)(1u<<(key&7));
    }
}

__attribute__((always_inline)) static inline
int dos_raster_solid_floor(RendererPixel *out,unsigned count,const uint8_t *texture,
                            const RendererPixel *palette,uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
    for(unsigned i=0;i<3;i++) {
        const DosFloorSolid *e=&floor_solid[i];
        if(palette!=e->palette)continue;
        if(e->whole_solid){fill_words((int16_t *)out,count,palette[0]);return 1;}
        if(du && dv)continue;
        unsigned axis=dv!=0;
        const uint8_t *source=texture+(axis?((u>>14)&0xfc):((v>>6)&0xfc00));
        uintptr_t offset=(uintptr_t)source-(uintptr_t)e->texture;
        if(offset>=65536 || (axis?(offset>>8)>3:(offset&255)>3))continue;
        unsigned key=axis?((offset&255)<<2)+(offset>>8):((offset>>8)<<2)+(offset&3);
        if(e->solid[axis][key>>3]&(1u<<(key&7))) {
            fill_words((int16_t *)out,count,palette[source[0]]);return 1;
        }
    }
    return 0;
}
#endif

void dos_raster_floor_edges_init(int16_t *left, int16_t *right,
                                  unsigned count, int16_t empty_left)
{
    fill_words(left, count, empty_left);
    fill_words(right, count, (uint16_t)-1);
    /* A visible endpoint always replaces its empty bound and writes its
     * brightness at the same time. Invalid rows never consume brightness. */
}

__attribute__((noinline, optimize("O2")))
void dos_raster_floor_edge(int16_t *restrict left, int16_t *restrict right,
                            int16_t *restrict left_bright, int16_t *restrict right_bright,
                            unsigned count, uint32_t x, uint32_t dx, uint32_t bright, uint32_t dbright)
{
    for (unsigned i = 0; i < count; i++) {
        int l = (int32_t)x >> 16;
        int r = l + ((int32_t)x > 0 && (x & 65535) != 0);
        if (l < left[i]) {
            left[i] = (int16_t)l;
            if (left_bright) left_bright[i] = (int16_t)(bright >> 16);
        }
        if (r > right[i]) {
            right[i] = (int16_t)(r > INT16_MAX ? INT16_MAX : r);
            if (right_bright) right_bright[i] = (int16_t)(bright >> 16);
        }
        x += dx;
        bright += dbright;
    }
}

/* WallRoutine3.ChipMem.s: three five-bit columns share each big-endian
 * word. Keep word order within each plane: levels can interpret one texture
 * at different VALSHIFT values. A four-byte prefix records the plane stride. */
uint8_t *dos_load_wall(uint8_t *data, size_t bytes, size_t *source_bytes)
{
    if(bytes<DOS_WALL_HEADER_BYTES+8 || memcmp(data,"ABW1",4)) {
        fprintf(stderr,"Invalid preconverted DOS wall; rebuild the complete DOS package\n");
        exit(1);
    }
    uint32_t native=*(const uint32_t *)(const void *)(data+4);
#ifdef AB3D_NATIVE_RGB565
    const uint32_t expected_native=1;
#else
    const uint32_t expected_native=0;
#endif
    uint8_t *palette=data+8;
    uint32_t words=*(const uint32_t *)(const void *)(palette+DOS_WALL_PALETTE_BYTES);
    size_t payload=bytes-8-DOS_WALL_HEADER_BYTES;
    if(native!=expected_native || !words || payload%3 || payload/3!=words) {
        fprintf(stderr,"Invalid DOS wall format/plane size; rebuild the matching DOS package\n");exit(1);
    }
    for(size_t i=0;i<payload;i++)if(palette[DOS_WALL_HEADER_BYTES+i]>31) {
        fprintf(stderr,"Invalid DOS wall texel at %lu\n",(unsigned long)i);exit(1);
    }
    *source_bytes=DOS_WALL_PALETTE_BYTES+(size_t)words*2;
    return palette;
}

__attribute__((noinline, optimize("O2")))
void dos_raster_wall_rows(RendererPixel *restrict out, unsigned width, unsigned height, unsigned pitch,
                           const DosWallColumn *restrict columns, const RendererPixel *shared_palette,
                           uint32_t y, uint32_t dy, unsigned mask)
{
#ifdef AB3D_NATIVE_RGB565
    /* A caller with constant brightness already knows the palette. Its column
     * palette fields are unused; varying spans retain the exact classification. */
    if (!shared_palette) {
        shared_palette = width ? columns[0].palette : NULL;
        for (unsigned x = 1; x < width; x++) {
            if (columns[x].palette != shared_palette) { shared_palette = NULL; break; }
        }
    }
#else
    (void)shared_palette;
#endif
    unsigned previous_sample = ~0u;
    for (unsigned row = 0; row < height; row++) {
        unsigned sample = (y >> 16) & mask;
        if (sample == previous_sample)
            memcpy(out, out - pitch, width * sizeof(*out));
        else {
#ifdef AB3D_NATIVE_RGB565
            if (width) {
                RendererPixel *dest = out;
                const DosWallColumn *source = columns;
                unsigned count = width;
                if (shared_palette) {
#define WALL_ROW_PIXEL(COLUMN, PIXEL) \
                    "movl " #COLUMN "(%%esi), %%eax\n\t" \
                    "movzbl (%%eax,%%ebx), %%eax\n\t" \
                    "movw (%[pal],%%eax,2), %%ax\n\t" \
                    "movw %%ax, " #PIXEL "(%%edi)\n\t"
                    unsigned groups = count / 8;
                    if (groups) __asm__ volatile(
                        "1:\n\t"
                        WALL_ROW_PIXEL(0, 0)
                        WALL_ROW_PIXEL(8, 2)
                        WALL_ROW_PIXEL(16, 4)
                        WALL_ROW_PIXEL(24, 6)
                        WALL_ROW_PIXEL(32, 8)
                        WALL_ROW_PIXEL(40, 10)
                        WALL_ROW_PIXEL(48, 12)
                        WALL_ROW_PIXEL(56, 14)
                        "addl $64, %%esi\n\t"
                        "addl $16, %%edi\n\t"
                        "decl %%ecx\n\t"
                        "jnz 1b"
                        : "+&D"(dest), "+&S"(source), "+&c"(groups)
                        : "b"(sample), [pal] "r"(shared_palette) : "eax", "cc", "memory");
                    count &= 7;
                    if (count) __asm__ volatile(
                        "1:\n\t"
                        WALL_ROW_PIXEL(0, 0)
                        "addl $8, %%esi\n\t"
                        "addl $2, %%edi\n\t"
                        "decl %%ecx\n\t"
                        "jnz 1b"
                        : "+&D"(dest), "+&S"(source), "+&c"(count)
                        : "b"(sample), [pal] "r"(shared_palette) : "eax", "cc", "memory");
#undef WALL_ROW_PIXEL
                } else {
                    __asm__ volatile(
                    "1: movl (%%esi), %%eax\n\t"
                    "movl 4(%%esi), %%edx\n\t"
                    "movzbl (%%eax,%%ebx), %%eax\n\t"
                    "movw (%%edx,%%eax,2), %%ax\n\t"
                    "movw %%ax, (%%edi)\n\t"
                    "addl $8, %%esi\n\t"
                    "addl $2, %%edi\n\t"
                    "decl %%ecx\n\t"
                    "jnz 1b"
                    : "+&D"(dest), "+&S"(source), "+&c"(count)
                    : "b"(sample) : "eax", "edx", "cc", "memory");
                }
            }
#else
            for (unsigned x = 0; x < width; x++)
                out[x] = columns[x].palette[columns[x].texels[sample]];
#endif
        }
        previous_sample = sample;
        out += pitch;
        y += dy;
    }
}

__attribute__((noinline, optimize("O2")))
void dos_raster_wall(RendererPixel *restrict out, unsigned count, unsigned stride,
                      const uint8_t *restrict column, const RendererPixel *restrict palette,
                      uint32_t y, uint32_t dy, unsigned mask)
{
#ifdef AB3D_NATIVE_RGB565
    if (!count) return;
    /* Native 320-pixel modes can address eight rows directly. Keep every
     * original fixed-point sample, while stepping the destination once. */
    if (stride == 320 && count >= 8) {
#define WALL_COLUMN_PIXEL(OFFSET, SAMPLE_SHIFT, MASK_OP) \
        "movl %%eax, %%edx\n\t" \
        "shrl $" #SAMPLE_SHIFT ", %%edx\n\t" \
        MASK_OP \
        "movzbl (%%esi,%%edx), %%edx\n\t" \
        "movw (%[pal],%%edx,2), %%dx\n\t" \
        "movw %%dx, " #OFFSET "(%%edi)\n\t" \
        "addl %%ebx, %%eax\n\t"
#define WALL_COLUMN_GROUPS(COORD_SHIFT, SAMPLE_SHIFT, MASK_OP) do { \
        unsigned groups = count / 8, tail = count & 7; \
        uint32_t coord = y << COORD_SHIFT, delta = dy << COORD_SHIFT; \
        __asm__ volatile( \
            "1:\n\t" \
            WALL_COLUMN_PIXEL(0, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(640, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(1280, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(1920, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(2560, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(3200, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(3840, SAMPLE_SHIFT, MASK_OP) \
            WALL_COLUMN_PIXEL(4480, SAMPLE_SHIFT, MASK_OP) \
            "addl $5120, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(coord), "+&D"(out), "+&c"(groups) \
            : "S"(column), "b"(delta), [pal] "r"(palette), [mask] "m"(mask) \
            : "edx", "cc", "memory"); \
        if (tail) __asm__ volatile( \
            "1:\n\t" \
            WALL_COLUMN_PIXEL(0, SAMPLE_SHIFT, MASK_OP) \
            "addl $640, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(coord), "+&D"(out), "+&c"(tail) \
            : "S"(column), "b"(delta), [pal] "r"(palette), [mask] "m"(mask) \
            : "edx", "cc", "memory"); \
        return; \
        } while (0)
        /* Shift the authored wrapping bits to the top of the accumulator.
         * Unsigned addition commutes with that shift, so a single right
         * shift gives the exact sample without loading a mask per pixel. */
        switch (mask) {
            case 15:  WALL_COLUMN_GROUPS(12, 28, ""); break;
            case 31:  WALL_COLUMN_GROUPS(11, 27, ""); break;
            case 63:  WALL_COLUMN_GROUPS(10, 26, ""); break;
            case 127: WALL_COLUMN_GROUPS(9, 25, ""); break;
            case 255: WALL_COLUMN_GROUPS(8, 24, ""); break;
            default:  WALL_COLUMN_GROUPS(0, 16, "andl %[mask], %%edx\n\t"); break;
        }
#undef WALL_COLUMN_GROUPS
#undef WALL_COLUMN_PIXEL
    }
    unsigned stride_bytes = stride * sizeof(*out);
    __asm__ volatile(
        "1: movl %%eax, %%edx\n\t"
        "shrl $16, %%edx\n\t"
        "andl %[mask], %%edx\n\t"
        "movzbl (%%esi,%%edx), %%edx\n\t"
        "movw (%[pal],%%edx,2), %%dx\n\t"
        "movw %%dx, (%%edi)\n\t"
        "addl %%ebx, %%eax\n\t"
        "addl %[stride], %%edi\n\t"
        "decl %%ecx\n\t"
        "jnz 1b"
        : "+&a"(y), "+&D"(out), "+&c"(count)
        : "S"(column), "b"(dy), [pal] "r"(palette), [mask] "m"(mask), [stride] "m"(stride_bytes)
        : "edx", "cc", "memory");
#else
    for (unsigned i = 0; i < count; i++) {
        *out = palette[column[(y >> 16) & mask]];
        out += stride;
        y += dy;
    }
#endif
}

/* Side extensions use the completed core's exact sample and shade. Store all
 * two/three adjacent words while that color is still in a register. */
__attribute__((noinline, optimize("O2")))
void dos_raster_wall_edges(RendererPixel *restrict out, unsigned count, unsigned stride,
                            const uint8_t *restrict column, const RendererPixel *restrict palette,
                            uint32_t y, uint32_t dy, unsigned mask, unsigned copies)
{
#ifdef AB3D_NATIVE_RGB565
    if (stride == 320) {
#define EDGE_STORE_TWO(OFFSET) "movw %%dx, " #OFFSET "+2(%%edi)\n\t"
#define EDGE_STORE_THREE(OFFSET) EDGE_STORE_TWO(OFFSET) "movw %%dx, " #OFFSET "+4(%%edi)\n\t"
#define EDGE_PIXEL(OFFSET, EXTRA) \
        "movl %%eax, %%edx\n\t" \
        "shrl $16, %%edx\n\t" \
        "andl %[mask], %%edx\n\t" \
        "movzbl (%%esi,%%edx), %%edx\n\t" \
        "movw (%[pal],%%edx,2), %%dx\n\t" \
        "movw %%dx, " #OFFSET "(%%edi)\n\t" \
        EXTRA(OFFSET) \
        "addl %%ebx, %%eax\n\t"
#define EDGE_COLUMNS(EXTRA) do { \
        unsigned groups = count / 8; \
        if (groups) __asm__ volatile( \
            "1:\n\t" \
            EDGE_PIXEL(0, EXTRA) \
            EDGE_PIXEL(640, EXTRA) \
            EDGE_PIXEL(1280, EXTRA) \
            EDGE_PIXEL(1920, EXTRA) \
            EDGE_PIXEL(2560, EXTRA) \
            EDGE_PIXEL(3200, EXTRA) \
            EDGE_PIXEL(3840, EXTRA) \
            EDGE_PIXEL(4480, EXTRA) \
            "addl $5120, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(y), "+&D"(out), "+&c"(groups) \
            : "S"(column), "b"(dy), [pal] "r"(palette), [mask] "m"(mask) \
            : "edx", "cc", "memory"); \
        count &= 7; \
        if (count) __asm__ volatile( \
            "1:\n\t" \
            EDGE_PIXEL(0, EXTRA) \
            "addl $640, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(y), "+&D"(out), "+&c"(count) \
            : "S"(column), "b"(dy), [pal] "r"(palette), [mask] "m"(mask) \
            : "edx", "cc", "memory"); \
        } while (0)
        if (copies == 3) EDGE_COLUMNS(EDGE_STORE_THREE);
        else EDGE_COLUMNS(EDGE_STORE_TWO);
#undef EDGE_COLUMNS
#undef EDGE_PIXEL
#undef EDGE_STORE_THREE
#undef EDGE_STORE_TWO
        return;
    }
#endif
    for (unsigned row = 0; row < count; row++, out += stride, y += dy) {
        RendererPixel color = palette[column[(y >> 16) & mask]];
        out[0] = out[1] = color;
        if (copies == 3) out[2] = color;
    }
}

#ifdef AB3D_NATIVE_RGB565
__attribute__((noinline, optimize("O2")))
void dos_raster_floor_dark(RendererPixel *out,unsigned count,const uint8_t *texture,
                            const RendererPixel *palette,uint32_t u,uint32_t v,uint32_t du,uint32_t dv)
{
    if(count>=16 && (du || dv) && dos_raster_solid_floor(out,count,texture,palette,u,v,du,dv))return;
    dos_raster_floor(out,count,texture,palette,u,v,du,dv);
}
#endif

/* Separate small loops keep the water/projection state out of 486 registers.
 * Unsigned additions retain the renderer's exact 32-bit UV wrapping. */
__attribute__((noinline, optimize("O2")))
void dos_raster_floor(RendererPixel *restrict out, unsigned count, const uint8_t *restrict texture,
                       const RendererPixel *restrict palette, uint32_t u, uint32_t v,
                       uint32_t du, uint32_t dv)
{
    if (!count) return;
#ifdef AB3D_NATIVE_RGB565
    /* The 486 has seven general registers. Keep UV, source, destination and
     * palette in them; only the two steps/end pointer need stack operands. */
    if (!dv || !du) {
        uint32_t coordinate, step;
        if (!dv) {
            texture += (v >> 6) & 0xfc00;
            coordinate = u; step = du;
        } else {
            texture += (u >> 14) & 0xfc;
            coordinate = v; step = dv;
        }
        uint32_t magnitude=(int32_t)step<0?0u-step:step;
        if(magnitude<=32768) {
            /* AB3DI sideloop: at most half a texel per pixel. Walk the exact
             * 16-bit fraction and fetch a new color only on carry/borrow. */
#define FLOOR_CACHED_PIXEL(SHIFT, MASK, OP, OFFSET, LABEL) \
            "movw %%dx, " #OFFSET "(%%edi)\n\t" \
            #OP "w %%bx, %%ax\n\t" \
            "jnc " #LABEL "f\n\t" \
            #OP "l $65536, %%eax\n\t" \
            "movl %%eax, %%edx\n\t" \
            "shrl $" #SHIFT ", %%edx\n\t" \
            "andl $" #MASK ", %%edx\n\t" \
            "movzbl (%%esi,%%edx), %%edx\n\t" \
            "movw (%[pal],%%edx,2), %%dx\n\t" \
            #LABEL ":\n\t"
#define FLOOR_CACHED_AXIS(SHIFT, MASK, OP) do { \
            unsigned color=palette[texture[(coordinate>>SHIFT)&MASK]]; \
            if(!magnitude){fill_words((int16_t *)out,count,(uint16_t)color);return;} \
            unsigned groups=count/8; \
            if(groups)__asm__ volatile( \
                "1:\n\t" \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,0,2) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,2,3) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,4,4) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,6,5) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,8,6) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,10,7) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,12,8) \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,14,9) \
                "addl $16, %%edi\n\t" \
                "decl %%ecx\n\t" \
                "jnz 1b" \
                : "+&a"(coordinate), "+&D"(out), "+&c"(groups), "+&d"(color) \
                : "S"(texture), "b"(magnitude), [pal] "r"(palette) : "cc", "memory"); \
            count&=7; \
            if(count)__asm__ volatile( \
                "1:\n\t" \
                FLOOR_CACHED_PIXEL(SHIFT,MASK,OP,0,2) \
                "addl $2, %%edi\n\t" \
                "decl %%ecx\n\t" \
                "jnz 1b" \
                : "+&a"(coordinate), "+&D"(out), "+&c"(count), "+&d"(color) \
                : "S"(texture), "b"(magnitude), [pal] "r"(palette) : "cc", "memory"); \
            } while(0)
            if(!dv) {
                if((int32_t)step<0)FLOOR_CACHED_AXIS(14,0xfc,sub);
                else FLOOR_CACHED_AXIS(14,0xfc,add);
            } else {
                if((int32_t)step<0)FLOOR_CACHED_AXIS(6,0xfc00,sub);
                else FLOOR_CACHED_AXIS(6,0xfc00,add);
            }
#undef FLOOR_CACHED_AXIS
#undef FLOOR_CACHED_PIXEL
            return;
        }
#define FLOOR_AXIS_PIXEL(SHIFT, MASK, OFFSET) \
            "movl %%eax, %%edx\n\t" \
            "shrl $" #SHIFT ", %%edx\n\t" \
            "andl $" #MASK ", %%edx\n\t" \
            "movzbl (%%esi,%%edx), %%edx\n\t" \
            "movw (%[pal],%%edx,2), %%dx\n\t" \
            "movw %%dx, " #OFFSET "(%%edi)\n\t" \
            "addl %%ebx, %%eax\n\t"
#define FLOOR_AXIS(SHIFT, MASK) do { \
        unsigned groups = count / 8; \
        if (groups) __asm__ volatile( \
            "1:\n\t" \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 0) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 2) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 4) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 6) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 8) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 10) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 12) \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 14) \
            "addl $16, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(coordinate), "+&D"(out), "+&c"(groups) \
            : "S"(texture), "b"(step), [pal] "r"(palette) \
            : "edx", "cc", "memory"); \
        count &= 7; \
        if (count) __asm__ volatile( \
            "1:\n\t" \
            FLOOR_AXIS_PIXEL(SHIFT, MASK, 0) \
            "addl $2, %%edi\n\t" \
            "decl %%ecx\n\t" \
            "jnz 1b" \
            : "+&a"(coordinate), "+&D"(out), "+&c"(count) \
            : "S"(texture), "b"(step), [pal] "r"(palette) \
            : "edx", "cc", "memory"); \
        } while (0)
        if (!dv) FLOOR_AXIS(14, 0xfc);
        else FLOOR_AXIS(6, 0xfc00);
#undef FLOOR_AXIS
#undef FLOOR_AXIS_PIXEL
        return;
    }
#define FLOOR_UV_PIXEL(OFFSET) \
        "movl %%eax, %%ecx\n\t" \
        "shrl $14, %%ecx\n\t" \
        "andl $0xfc, %%ecx\n\t" \
        "movl %%ebx, %%edx\n\t" \
        "shrl $6, %%edx\n\t" \
        "andl $0xfc00, %%edx\n\t" \
        "orl %%edx, %%ecx\n\t" \
        "movzbl (%%esi,%%ecx), %%ecx\n\t" \
        "movw (%[pal],%%ecx,2), %%cx\n\t" \
        "movw %%cx, " #OFFSET "(%%edi)\n\t" \
        "addl %[du], %%eax\n\t" \
        "addl %[dv], %%ebx\n\t"
    if (count >= 8) {
        RendererPixel *groups_end = out + (count & ~7u);
        __asm__ volatile(
            "1:\n\t"
            FLOOR_UV_PIXEL(0)
            FLOOR_UV_PIXEL(2)
            FLOOR_UV_PIXEL(4)
            FLOOR_UV_PIXEL(6)
            FLOOR_UV_PIXEL(8)
            FLOOR_UV_PIXEL(10)
            FLOOR_UV_PIXEL(12)
            FLOOR_UV_PIXEL(14)
            "addl $16, %%edi\n\t"
            "cmpl %[end], %%edi\n\t"
            "jne 1b"
            : "+&a"(u), "+&b"(v), "+&D"(out)
            : "S"(texture), [pal] "r"(palette), [du] "m"(du), [dv] "m"(dv), [end] "m"(groups_end)
            : "ecx", "edx", "cc", "memory");
        count &= 7;
    }
    if (count) {
        RendererPixel *native_end = out + count;
        __asm__ volatile(
            "1:\n\t"
            FLOOR_UV_PIXEL(0)
            "addl $2, %%edi\n\t"
            "cmpl %[end], %%edi\n\t"
            "jne 1b"
            : "+&a"(u), "+&b"(v), "+&D"(out)
            : "S"(texture), [pal] "r"(palette), [du] "m"(du), [dv] "m"(dv), [end] "m"(native_end)
            : "ecx", "edx", "cc", "memory");
    }
#undef FLOOR_UV_PIXEL
    return;
#endif
    RendererPixel *end = out + count;
    if (!dv) {
        texture += (v >> 6) & 0xfc00;
        while (out != end) {
            *out++ = palette[texture[(u >> 14) & 0xfc]];
            u += du;
        }
        return;
    }
    if (!du) {
        texture += (u >> 14) & 0xfc;
        while (out != end) {
            *out++ = palette[texture[(v >> 6) & 0xfc00]];
            v += dv;
        }
        return;
    }
    while (out != end) {
        *out++ = palette[texture[((u >> 14) & 0xfc) | ((v >> 6) & 0xfc00)]];
        u += du;
        v += dv;
    }
}

__attribute__((noinline, optimize("O2")))
void dos_raster_floor_gouraud(RendererPixel *restrict out, unsigned count, const uint8_t *restrict texture,
                               const RendererPixel (*restrict palettes)[256],
                               uint32_t u, uint32_t v, uint32_t du, uint32_t dv,
                               int32_t level, int32_t dlevel)
{
    if (!dlevel) {
        dos_raster_floor_shaded(out, count, texture, palettes, level >> 16, u, v, du, dv);
        return;
    }
    unsigned gradient = dlevel < 0 ? 0u - (uint32_t)dlevel : (uint32_t)dlevel;
    /* Long constant-shade runs keep Gouraud selection out of the pixel loop.
     * Run boundaries are exact for either sign of the 16.16 gradient. */
    if (gradient < 8192) {
        while (count) {
            unsigned fraction = (uint32_t)level & 65535;
            unsigned run = dlevel < 0 ? fraction / gradient + 1
                : (65536 - fraction + gradient - 1) / gradient;
            if (run > count) run = count;
            dos_raster_floor_shaded(out, run, texture, palettes, level >> 16, u, v, du, dv);
            out += run;
            u += du * run;
            v += dv * run;
            level += dlevel * (int32_t)run;
            count -= run;
        }
        return;
    }
    RendererPixel *end = out + count;
    if (!dv) {
        texture += (v >> 6) & 0xfc00;
        while (out != end) {
            *out++ = palettes[level >> 16][texture[(u >> 14) & 0xfc]];
            u += du;
            level += dlevel;
        }
        return;
    }
    if (!du) {
        texture += (u >> 14) & 0xfc;
        while (out != end) {
            *out++ = palettes[level >> 16][texture[(v >> 6) & 0xfc00]];
            v += dv;
            level += dlevel;
        }
        return;
    }
    while (out != end) {
        /* Both gradient endpoints are clamped to 0..14 by the caller. */
        *out++ = palettes[level >> 16][texture[((u >> 14) & 0xfc) | ((v >> 6) & 0xfc00)]];
        u += du;
        v += dv;
        level += dlevel;
    }
}
